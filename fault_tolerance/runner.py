"""Sweep runner — runs on the VM, resumable, reuses server across same-config items.

Usage:
    python -m fault_tolerance.runner --sweep sweeps/milestone2.yaml --state-dir ~/runs

Resumes automatically: items with status "done" are skipped on every run.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from fault_tolerance.preempt import PreemptWatcher
from fault_tolerance.server_proc import ServerProcess, ServerStartError
from fault_tolerance.sweep import Item, Sweep, load_sweep


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_status(item_dir: Path) -> dict:
    p = item_dir / "status.json"
    if not p.exists():
        return {"state": "queued", "attempts": 0}
    return json.loads(p.read_text())


def _write_status(item_dir: Path, **fields) -> dict:
    item_dir.mkdir(parents=True, exist_ok=True)
    status = _read_status(item_dir)
    status.update(fields)
    status["updated_at"] = _utcnow()
    (item_dir / "status.json").write_text(json.dumps(status, indent=2))
    return status


def _run_bench(item: Item, sweep: Sweep, item_dir: Path) -> tuple[int, str | None]:
    argv = item.bench_cli(sweep.model, sweep.port)
    bench_log = item_dir / "bench.stdout"
    item_dir.mkdir(parents=True, exist_ok=True)
    with bench_log.open("ab", buffering=0) as fh:
        fh.write(f"$ {' '.join(argv)}\n".encode())
        try:
            proc = subprocess.run(
                argv,
                stdout=fh,
                stderr=subprocess.STDOUT,
                timeout=sweep.bench_timeout_s,
            )
        except subprocess.TimeoutExpired:
            return -1, f"bench timeout after {sweep.bench_timeout_s}s"
    return proc.returncode, None


def run_sweep(sweep_path: Path, state_root: Path) -> int:
    sweep = load_sweep(sweep_path)
    state_dir = state_root / sweep.sweep_id
    items_dir = state_dir / "items"
    state_dir.mkdir(parents=True, exist_ok=True)
    items_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(sweep_path, state_dir / "sweep.yaml")

    print(f"[runner] sweep={sweep.sweep_id} items={len(sweep.items)} "
          f"state_dir={state_dir}", flush=True)

    preempt = PreemptWatcher().start()
    server: ServerProcess | None = None
    current_fp: str | None = None

    def kill_server():
        nonlocal server, current_fp
        if server is not None:
            print("[runner] stopping server", flush=True)
            server.stop()
            server = None
            current_fp = None

    exit_code = 0
    try:
        for item in sweep.items:
            item_dir = items_dir / item.id
            status = _read_status(item_dir)
            if status.get("state") == "done":
                continue
            if status.get("attempts", 0) >= sweep.max_attempts_per_item:
                print(f"[runner] {item.id}: skipping (attempts exhausted)", flush=True)
                continue
            if preempt.preempted:
                print("[runner] preemption detected, stopping sweep", flush=True)
                _write_status(item_dir, state="interrupted")
                exit_code = 75  # EX_TEMPFAIL — driver should retry
                break

            fp = item.server_fingerprint()
            if fp != current_fp:
                kill_server()
                argv = item.server_cli(sweep.model, sweep.port)
                print(f"[runner] {item.id}: starting server fp={fp}", flush=True)
                print(f"[runner]   $ {' '.join(argv)}", flush=True)
                server = ServerProcess(
                    argv=argv,
                    log_path=item_dir / "server.log",
                    port=sweep.port,
                )
                try:
                    server.start()
                    server.wait_for_ready(timeout_s=sweep.server_warmup_timeout_s)
                except ServerStartError as e:
                    print(f"[runner] {item.id}: server failed: {e}", flush=True)
                    _write_status(
                        item_dir,
                        state="failed",
                        attempts=status.get("attempts", 0) + 1,
                        error=str(e),
                    )
                    kill_server()
                    continue
                current_fp = fp
            else:
                item_dir.mkdir(parents=True, exist_ok=True)
                (item_dir / "server.reused").write_text(f"reused server fp={fp}\n")

            attempt = status.get("attempts", 0) + 1
            _write_status(
                item_dir,
                state="running",
                attempts=attempt,
                started_at=_utcnow(),
                server_fingerprint=fp,
            )
            print(f"[runner] {item.id}: running bench (attempt {attempt})", flush=True)
            t0 = time.monotonic()
            rc, err = _run_bench(item, sweep, item_dir)
            elapsed = time.monotonic() - t0

            if rc == 0:
                _write_status(
                    item_dir,
                    state="done",
                    ended_at=_utcnow(),
                    duration_s=round(elapsed, 2),
                    exit_code=rc,
                )
                print(f"[runner] {item.id}: done in {elapsed:.1f}s", flush=True)
            else:
                _write_status(
                    item_dir,
                    state="failed",
                    ended_at=_utcnow(),
                    duration_s=round(elapsed, 2),
                    exit_code=rc,
                    error=err,
                )
                print(f"[runner] {item.id}: FAILED rc={rc} err={err}", flush=True)
                # Bench crash may have left server in a bad state — restart for next item.
                kill_server()
    finally:
        kill_server()
        preempt.stop()
        _write_summary(sweep, state_dir)

    return exit_code


def _write_summary(sweep: Sweep, state_dir: Path) -> None:
    rows = []
    for item in sweep.items:
        st = _read_status(state_dir / "items" / item.id)
        rows.append({
            "item_id": item.id,
            "state": st.get("state", "queued"),
            "attempts": st.get("attempts", 0),
            "duration_s": st.get("duration_s"),
            "server": json.dumps(item.server, sort_keys=True),
            "bench": json.dumps(item.bench, sort_keys=True),
        })
    with (state_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    p = argparse.ArgumentParser(description="Run a benchmark sweep with fault tolerance")
    p.add_argument("--sweep", required=True, type=Path)
    p.add_argument("--state-dir", required=True, type=Path,
                   help="Directory holding per-sweep state (resumable).")
    args = p.parse_args()
    return run_sweep(args.sweep, args.state_dir)


if __name__ == "__main__":
    sys.exit(main())
