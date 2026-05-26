"""End-to-end test: simulate a spot preemption mid-sweep, verify resume.

Run:
    python -m tests.test_preemption

Exits 0 on success, 1 on any assertion failure. Takes ~15 seconds.

Scenario:
1. Spawn a fake metadata HTTP server controlled by a flag file.
2. Run the sweep runner with fakes (fake_server.py, fake_bench.py).
3. While item 2 is running, touch the flag file -> metadata returns TRUE.
4. Runner should detect preemption, mark item 3 'interrupted', exit 75.
5. Remove the flag; re-spawn runner.
6. Runner skips done items 1 & 2, completes items 3 & 4, exits 0.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SWEEP_PATH = Path(__file__).resolve().parent / "sweep_preempt.yaml"


# ── Fake metadata server ───────────────────────────────────────────────


def _make_metadata_server(flag_path: Path) -> tuple[HTTPServer, int]:
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            val = b"TRUE" if flag_path.exists() else b"FALSE"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(val)

        def log_message(self, *a, **kw):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


# ── Test helpers ───────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _read_status(item_dir: Path) -> dict | None:
    p = item_dir / "status.json"
    return json.loads(p.read_text()) if p.exists() else None


def _wait_for(predicate, timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out after {timeout_s}s waiting for: {what}")


def _spawn_runner(
    state_dir: Path,
    server_port: int,
    preempt_url: str,
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable, "-m", "fault_tolerance.runner",
            "--sweep", str(SWEEP_PATH),
            "--state-dir", str(state_dir),
            "--server-cmd", f"{sys.executable} {REPO_ROOT}/tests/fake_server.py",
            "--bench-cmd", f"{sys.executable} {REPO_ROOT}/tests/fake_bench.py",
            "--preempt-url", preempt_url,
            "--preempt-poll-s", "0.3",
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )


def _drain(proc: subprocess.Popen, label: str) -> None:
    """Pump subprocess stdout to our stdout with a prefix."""
    assert proc.stdout is not None
    for line in proc.stdout:
        print(f"  [{label}] {line}", end="", flush=True)


# ── Assertions ─────────────────────────────────────────────────────────


def assert_eq(actual, expected, what: str) -> None:
    if actual != expected:
        print(f"\n[FAIL] {what}: got {actual!r}, expected {expected!r}", flush=True)
        sys.exit(1)
    print(f"[ok]   {what} = {actual!r}", flush=True)


def assert_true(cond: bool, what: str) -> None:
    if not cond:
        print(f"\n[FAIL] {what}", flush=True)
        sys.exit(1)
    print(f"[ok]   {what}", flush=True)


# ── Main test ──────────────────────────────────────────────────────────


def main() -> int:
    print("[test] starting", flush=True)
    with tempfile.TemporaryDirectory(prefix="ft-test-") as td:
        td_path = Path(td)
        state_dir = td_path / "runs"
        flag_path = td_path / "preempt-flag"

        srv, meta_port = _make_metadata_server(flag_path)
        try:
            preempt_url = f"http://127.0.0.1:{meta_port}/preempted"

            # ── Run 1: should be interrupted ──────────────────────────
            print(f"[test] run 1: preempt-url={preempt_url}", flush=True)
            proc = _spawn_runner(state_dir, _free_port(), preempt_url)
            log_thread = threading.Thread(
                target=_drain, args=(proc, "run1"), daemon=True
            )
            log_thread.start()

            items_dir = state_dir / "preempt-test" / "items"

            # Wait until item2 starts running, then flip preemption.
            _wait_for(
                lambda: (
                    (s := _read_status(items_dir / "item2")) is not None
                    and s.get("state") == "running"
                ),
                timeout_s=30,
                what="item2 to enter 'running'",
            )
            print("[test] item2 running -> setting preempt flag", flush=True)
            flag_path.touch()

            try:
                exit1 = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                print("\n[FAIL] runner did not exit within 30s", flush=True)
                return 1

            log_thread.join(timeout=2)

            print("\n[test] === run 1 assertions ===", flush=True)
            assert_eq(exit1, 75, "run 1 exit code")
            s1 = _read_status(items_dir / "item1")
            s2 = _read_status(items_dir / "item2")
            s3 = _read_status(items_dir / "item3")
            s4 = _read_status(items_dir / "item4")
            assert_true(s1 is not None and s1["state"] == "done", "item1 = done")
            assert_true(s2 is not None and s2["state"] == "done", "item2 = done")
            assert_true(s3 is not None and s3["state"] == "interrupted",
                        "item3 = interrupted")
            assert_true(s4 is None, "item4 never reached")
            assert_true((items_dir / "item1" / "server.log").exists(),
                        "item1 has server.log (started server)")
            assert_true((items_dir / "item2" / "server.reused").exists(),
                        "item2 has server.reused (reused server)")

            # ── Run 2: resume ─────────────────────────────────────────
            flag_path.unlink()
            print("\n[test] run 2: flag cleared, resuming", flush=True)
            proc = _spawn_runner(state_dir, _free_port(), preempt_url)
            log_thread = threading.Thread(
                target=_drain, args=(proc, "run2"), daemon=True
            )
            log_thread.start()

            try:
                exit2 = proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                print("\n[FAIL] resume runner did not exit within 30s", flush=True)
                return 1

            log_thread.join(timeout=2)

            print("\n[test] === run 2 assertions ===", flush=True)
            assert_eq(exit2, 0, "run 2 exit code")
            for iid in ("item1", "item2", "item3", "item4"):
                s = _read_status(items_dir / iid)
                assert_true(
                    s is not None and s["state"] == "done", f"{iid} = done"
                )
            assert_true((items_dir / "item3" / "server.log").exists(),
                        "item3 has server.log (restarted server on resume)")
            assert_true((items_dir / "item4" / "server.reused").exists(),
                        "item4 has server.reused")
            assert_true((state_dir / "preempt-test" / "summary.csv").exists(),
                        "summary.csv exists")

            # Items 1 & 2 should not have been re-run (attempts unchanged at 1).
            s1 = _read_status(items_dir / "item1")
            s2 = _read_status(items_dir / "item2")
            assert_eq(s1["attempts"], 1, "item1 attempts (not re-run)")
            assert_eq(s2["attempts"], 1, "item2 attempts (not re-run)")

            print("\n[test] PASS", flush=True)
            return 0
        finally:
            srv.shutdown()


if __name__ == "__main__":
    sys.exit(main())
