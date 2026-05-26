"""Local driver — ensures the GCE VM is up, rsyncs code, ssh's and invokes the runner.

Restarts the VM and re-invokes the runner whenever the ssh session drops or
the runner exits with the temp-fail code (75) signaling preemption.

Usage:
    python -m fault_tolerance.driver run \
        --instance inference-engine-vm --zone us-central1-a \
        --sweep sweeps/milestone2.yaml \
        --remote-repo ~/CS349D-miniengine \
        --remote-state-dir ~/runs

    python -m fault_tolerance.driver pull \
        --instance inference-engine-vm --zone us-central1-a \
        --remote-state-dir ~/runs --local-state-dir ./runs
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


# Exit code the remote runner uses to signal preemption-induced stop.
RUNNER_TEMP_FAIL = 75


def _gcloud(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["gcloud", *args], capture_output=True, text=True)


def instance_status(instance: str, zone: str) -> str:
    r = _gcloud(
        "compute", "instances", "describe", instance,
        "--zone", zone, "--format=value(status)",
    )
    if r.returncode != 0:
        return f"UNKNOWN({r.stderr.strip()})"
    return r.stdout.strip()


def ensure_running(instance: str, zone: str, max_wait_s: int = 1800) -> None:
    """Block until the instance is RUNNING. Retries `instances start` with backoff
    since spot capacity comes and goes."""
    deadline = time.monotonic() + max_wait_s
    backoff = 30
    while time.monotonic() < deadline:
        status = instance_status(instance, zone)
        print(f"[driver] instance status: {status}", flush=True)
        if status == "RUNNING":
            return
        if status in ("TERMINATED", "STOPPED", "STOPPING"):
            r = _gcloud(
                "compute", "instances", "start", instance, "--zone", zone,
            )
            if r.returncode == 0:
                continue
            print(f"[driver] start failed: {r.stderr.strip()}", flush=True)
            print(f"[driver] backing off {backoff}s", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 600)
            continue
        # PROVISIONING, STAGING, REPAIRING — just wait.
        time.sleep(15)
    raise TimeoutError(f"instance {instance} did not reach RUNNING within {max_wait_s}s")


def rsync_to(instance: str, zone: str, local: Path, remote: str) -> None:
    """rsync local repo dir to the VM. Excludes typical junk."""
    cmd = [
        "gcloud", "compute", "rsync",
        "--zone", zone,
        "--recurse",
        "--exclude-from=-",  # supplied via stdin below
        str(local) + "/",
        f"{instance}:{remote}/",
    ]
    excludes = "\n".join([
        ".git/", ".venv/", "__pycache__/", "*.pyc",
        "runs/", ".claude/",
    ]) + "\n"
    # gcloud compute rsync isn't universally available; fall back to ssh+rsync.
    fallback = [
        "rsync", "-az", "--delete",
        "--exclude=.git", "--exclude=.venv", "--exclude=__pycache__",
        "--exclude=*.pyc", "--exclude=runs/", "--exclude=.claude/",
        "-e", f"gcloud compute ssh --zone={zone} --tunnel-through-iap --",
        str(local) + "/",
        f"{instance}:{remote}/",
    ]
    print(f"[driver] rsync -> {instance}:{remote}", flush=True)
    r = subprocess.run(fallback)
    if r.returncode != 0:
        # Last-ditch: tar | ssh
        print("[driver] rsync failed; falling back to tar pipe", flush=True)
        tar = subprocess.Popen(
            ["tar", "-cz",
             "--exclude=.git", "--exclude=.venv", "--exclude=__pycache__",
             "--exclude=*.pyc", "--exclude=runs", "--exclude=.claude",
             "-C", str(local), "."],
            stdout=subprocess.PIPE,
        )
        ssh = subprocess.Popen(
            ["gcloud", "compute", "ssh", instance, "--zone", zone,
             "--", f"mkdir -p {shlex.quote(remote)} && tar -xz -C {shlex.quote(remote)}"],
            stdin=tar.stdout,
        )
        tar.stdout.close()
        ssh.wait()
        tar.wait()
        if ssh.returncode != 0:
            raise RuntimeError("code transfer failed")


def ssh_run(instance: str, zone: str, remote_cmd: str) -> int:
    """Run a shell command on the VM. Returns the runner's exit code (or
    255-class for ssh failures)."""
    cmd = [
        "gcloud", "compute", "ssh", instance, "--zone", zone, "--",
        remote_cmd,
    ]
    print(f"[driver] ssh: {remote_cmd}", flush=True)
    r = subprocess.run(cmd)
    return r.returncode


def cmd_run(args: argparse.Namespace) -> int:
    sweep_local = Path(args.sweep).resolve()
    if not sweep_local.exists():
        print(f"[driver] sweep file not found: {sweep_local}", flush=True)
        return 2
    repo_local = Path(args.local_repo).resolve()

    sweep_remote = f"{args.remote_repo}/{sweep_local.relative_to(repo_local)}"

    while True:
        ensure_running(args.instance, args.zone, max_wait_s=args.start_timeout_s)
        rsync_to(args.instance, args.zone, repo_local, args.remote_repo)
        remote_cmd = (
            f"cd {shlex.quote(args.remote_repo)} && "
            f"python3 -m fault_tolerance.runner "
            f"--sweep {shlex.quote(sweep_remote)} "
            f"--state-dir {shlex.quote(args.remote_state_dir)}"
        )
        rc = ssh_run(args.instance, args.zone, remote_cmd)
        if rc == 0:
            print("[driver] sweep completed cleanly", flush=True)
            return 0
        if rc == RUNNER_TEMP_FAIL:
            print("[driver] runner reported preemption, will resume", flush=True)
            time.sleep(30)
            continue
        # Any other non-zero: likely ssh dropped (255) or VM died mid-run.
        # Distinguish only by VM status; in either case, loop and resume.
        status = instance_status(args.instance, args.zone)
        print(f"[driver] runner exited rc={rc}, vm status={status} — resuming", flush=True)
        if status == "RUNNING":
            # The runner itself errored, not the VM. Avoid tight loop.
            print("[driver] backing off 60s before retry", flush=True)
            time.sleep(60)


def cmd_pull(args: argparse.Namespace) -> int:
    cmd = [
        "rsync", "-az",
        "-e", f"gcloud compute ssh --zone={args.zone} --tunnel-through-iap --",
        f"{args.instance}:{args.remote_state_dir}/",
        str(Path(args.local_state_dir).resolve()) + "/",
    ]
    print(f"[driver] pulling state -> {args.local_state_dir}", flush=True)
    r = subprocess.run(cmd)
    return r.returncode


def main() -> int:
    p = argparse.ArgumentParser(description="Local driver for fault-tolerant sweeps")
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a sweep on the remote VM (resumable)")
    run.add_argument("--instance", required=True)
    run.add_argument("--zone", required=True)
    run.add_argument("--sweep", required=True, help="Path to sweep yaml (under local repo)")
    run.add_argument("--local-repo", default=".", help="Local repo root to rsync")
    run.add_argument("--remote-repo", default="~/CS349D-miniengine",
                     help="Path on VM where the repo is rsynced")
    run.add_argument("--remote-state-dir", default="~/runs",
                     help="Directory on VM holding sweep state")
    run.add_argument("--start-timeout-s", type=int, default=1800)
    run.set_defaults(func=cmd_run)

    pull = sub.add_parser("pull", help="rsync remote state dir to local")
    pull.add_argument("--instance", required=True)
    pull.add_argument("--zone", required=True)
    pull.add_argument("--remote-state-dir", default="~/runs")
    pull.add_argument("--local-state-dir", default="./runs")
    pull.set_defaults(func=cmd_pull)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
