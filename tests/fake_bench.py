"""Fake benchmark for tests. Sleeps then exits 0 (or 1 if --fail)."""

from __future__ import annotations

import argparse
import sys
import time


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--duration-s", type=float, default=2.0)
    p.add_argument("--fail", action="store_true")
    args, unknown = p.parse_known_args()
    print(f"[fake-bench] sleeping {args.duration_s}s  ignored={unknown}", flush=True)
    time.sleep(args.duration_s)
    if args.fail:
        print("[fake-bench] exiting with failure", flush=True)
        return 1
    print("[fake-bench] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
