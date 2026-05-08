"""Manage a miniengine server subprocess: start, wait-for-ready, stop."""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class ServerStartError(RuntimeError):
    pass


class ServerProcess:
    def __init__(self, argv: list[str], log_path: Path, port: int):
        self.argv = argv
        self.log_path = log_path
        self.port = port
        self._proc: subprocess.Popen | None = None
        self._log_fh = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = self.log_path.open("ab", buffering=0)
        # New process group so we can SIGTERM the whole tree (uvicorn workers, etc.).
        self._proc = subprocess.Popen(
            self.argv,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    def wait_for_ready(self, timeout_s: float, poll_interval_s: float = 2.0) -> None:
        url = f"http://localhost:{self.port}/health"
        deadline = time.monotonic() + timeout_s
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._proc is None or self._proc.poll() is not None:
                rc = self._proc.returncode if self._proc else None
                raise ServerStartError(
                    f"server exited before becoming ready (rc={rc}); see {self.log_path}"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                last_err = e
            time.sleep(poll_interval_s)
        raise ServerStartError(
            f"server failed health check within {timeout_s}s "
            f"(last error: {last_err}); see {self.log_path}"
        )

    def stop(self, grace_s: float = 15.0) -> None:
        if self._proc is None:
            return
        if self._proc.poll() is None:
            try:
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self._proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self._proc.wait(timeout=5)
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        self._proc = None
