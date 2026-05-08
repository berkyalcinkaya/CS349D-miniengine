"""Poll the GCE metadata server for a spot-preemption notice."""

from __future__ import annotations

import threading
import time
import urllib.error
import urllib.request


METADATA_URL = (
    "http://metadata.google.internal/computeMetadata/v1/instance/preempted"
)


class PreemptWatcher:
    def __init__(self, poll_interval_s: float = 5.0, url: str = METADATA_URL):
        self._poll = poll_interval_s
        self._url = url
        self._stop = threading.Event()
        self._preempted = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def preempted(self) -> bool:
        return self._preempted.is_set()

    def start(self) -> "PreemptWatcher":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _loop(self) -> None:
        req = urllib.request.Request(self._url, headers={"Metadata-Flavor": "Google"})
        while not self._stop.wait(self._poll):
            try:
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.read().decode().strip().upper() == "TRUE":
                        self._preempted.set()
                        return
            except (urllib.error.URLError, ConnectionError, TimeoutError):
                # Metadata server may be unreachable (e.g., off-GCE). Treat as
                # not-preempted; the driver-side liveness check is the backstop.
                pass
