#!/bin/bash
# Pulls server.log and benchmark.log from inference-engine-vm, then runs `claude -p`
# to analyze them for errors and detect spot-instance termination (stale logs
# or failed scp). Launched every 5 min by launchd.

set -u

export PATH="/Users/berk/opt/google-cloud-sdk/bin:/Users/berk/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

REPO="/Users/berk/code/CS349D-miniengine"
LOGS_DIR="$REPO/logs"
REPORT="$LOGS_DIR/claude_report.md"
RUN_LOG="$LOGS_DIR/launchd_run.log"
VM="inference-engine-vm"
ZONE="us-central1-a"
REMOTE="/workspace/CS349D-miniengine"

mkdir -p "$LOGS_DIR"

TIMESTAMP="$(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "=== Run $TIMESTAMP ===" >> "$RUN_LOG"

# Don't abort on scp failure — a failure itself is a signal (VM unreachable / terminated).
SERVER_SCP_OUT="$(gcloud compute scp "${VM}:${REMOTE}/server.log" "$LOGS_DIR/" --zone="$ZONE" 2>&1)"
SERVER_SCP_RC=$?
BENCH_SCP_OUT="$(gcloud compute scp "${VM}:${REMOTE}/benchmark.log" "$LOGS_DIR/" --zone="$ZONE" 2>&1)"
BENCH_SCP_RC=$?

echo "scp server.log rc=$SERVER_SCP_RC" >> "$RUN_LOG"
echo "scp benchmark.log  rc=$BENCH_SCP_RC"  >> "$RUN_LOG"

PROMPT="Analyze VM logs for errors and signs of spot-instance termination.

Run context ($TIMESTAMP):
- gcloud scp server.log exit code: $SERVER_SCP_RC
- gcloud scp benchmark.log  exit code: $BENCH_SCP_RC
- scp output (server.log): $SERVER_SCP_OUT
- scp output (benchmark.log):  $BENCH_SCP_OUT

Tasks:
1. Read $LOGS_DIR/server.log and $LOGS_DIR/benchmark.log (use Read; for large files, read the tail).
2. Check file mtimes via Bash ('stat -f %m' on macOS, but these are local files, so any stat works). If either log has no new activity in >15 minutes, or if scp returned non-zero, flag potential spot-instance termination / VM unreachable.
3. Summarize any errors, exceptions, tracebacks, or warnings in the last ~500 lines of each file. Be concise — bullets with line numbers.
4. Estimate time to completion for the current concurrency run: infer total request count, completed count, and throughput (requests/sec) from the benchmark.log, then project ETA. If the run has finished or can't be determined, say so.
5. Write the full report to $REPORT as markdown with sections: Status, Server Log, Benchmark Log, Termination Check, ETA. Overwrite the file each run. Start with the run timestamp.

Be concise. No preamble in your response."

claude -p "$PROMPT" --dangerously-skip-permissions >> "$RUN_LOG" 2>&1
echo "claude rc=$?" >> "$RUN_LOG"
echo "" >> "$RUN_LOG"
