"""Fake miniengine server for tests. Exposes /health after a short warmup."""

from __future__ import annotations

import argparse
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    ready_at: float = 0.0

    def do_GET(self) -> None:
        if self.path != "/health":
            self.send_response(404)
            self.end_headers()
            return
        if time.time() < Handler.ready_at:
            self.send_response(503)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, *args, **kwargs) -> None:
        pass


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--warmup-s", type=float, default=0.5)
    # Accept and ignore everything else (the runner passes the real server's flags).
    args, unknown = p.parse_known_args()
    Handler.ready_at = time.time() + args.warmup_s
    print(f"[fake-server] listening on :{args.port}  warmup={args.warmup_s}s  "
          f"ignored={unknown}", flush=True)
    HTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
