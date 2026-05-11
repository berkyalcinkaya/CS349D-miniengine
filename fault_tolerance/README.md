# Fault-tolerant benchmark sweeps

Automate parameter sweeps across server + benchmark configs on a GCE spot
instance, with resume-from-preemption.

## Layout

```
fault_tolerance/
  sweep.py        # YAML loader, item expansion, server-config fingerprint
  server_proc.py  # miniengine subprocess: start, /health probe, clean stop
  preempt.py      # GCE metadata poller for spot preemption
  runner.py       # on-VM: iterate items, reuse server when config matches
  driver.py       # local: gcloud start, rsync, ssh, retry loop
sweeps/
  milestone2.yaml # example
```

## Usage

Install the extra:
```bash
pip install -e '.[fault-tolerance]'
```

Run from your laptop:
```bash
python -m fault_tolerance.driver run \
    --instance inference-engine-vm --zone us-central1-a \
    --sweep sweeps/milestone2.yaml
```

The driver ensures the VM is RUNNING (retrying `gcloud instances start` with
backoff for spot capacity), rsyncs the repo, and invokes the runner over ssh.
If ssh drops or the runner reports preemption (exit 75), the driver loops.

Pull results back at any time:
```bash
python -m fault_tolerance.driver pull \
    --instance inference-engine-vm --zone us-central1-a
```

Run the runner directly on the VM (e.g., for local debugging):
```bash
python -m fault_tolerance.runner \
    --sweep sweeps/milestone2.yaml --state-dir ~/runs
```

## Sweep file

```yaml
sweep_id: my-sweep            # name of the run; state lives under <state-dir>/<sweep_id>/
model: Qwen/Qwen3-8B
port: 8000

server_warmup_timeout_s: 900  # generous: torch.compile + cuda-graph capture is slow
bench_timeout_s: 7200
max_attempts_per_item: 2

defaults:
  server: { mem_fraction_static: 0.85, page_size: 32 }

items:
  - id: acc-paged
    server: { mode: paged }
    bench: { script: bench_accuracy, dataset: mmlu, num_samples: 200 }

  - id: thru-paged-compile
    server: { mode: paged, torch_compile: true }
    bench:
      script: bench_serving
      input_len: 1024
      output_len: 512
      concurrencies: [1, 2, 4, 8, 16, 32]   # expanded into 6 sibling items
```

CLI translation: `snake_case → --kebab-case`, booleans become bare flags,
lists become comma-separated values. Unrecognized keys are passed through
verbatim, so the harness doesn't need to know the server's flag set.

## Server reuse

Items run in order. Each item has a server fingerprint (sha256 of its server
config). When a new item's fingerprint matches the current server's, the
runner skips kill + restart and just runs the benchmark. This collapses
N concurrency-level items with the same `--torch-compile` settings into one
warmup, paid once.

A bench failure restarts the server before the next item, since the
server may have been left in a bad state.

## State layout (resumable)

```
<state-dir>/<sweep_id>/
  sweep.yaml               # copy of input
  summary.csv              # one row per item: state, attempts, duration
  items/<item_id>/
    status.json            # {state, attempts, started_at, ended_at, exit_code, ...}
    server.log             # server stdout/stderr (only items that started one)
    server.reused          # marker for items reusing previous item's server
    bench.stdout           # benchmark output
```

`status.state ∈ {queued, running, done, failed, interrupted}`. On every
invocation the runner skips items already `done` and retries others up to
`max_attempts_per_item`. State files are the source of truth — no DB.

## Preemption signals

1. **In-VM:** `preempt.py` polls
   `http://metadata.google.internal/computeMetadata/v1/instance/preempted`.
   When it flips, the runner finishes the current bench attempt, marks the
   item `interrupted`, and exits 75.
2. **Driver-side:** if ssh drops or the VM goes `TERMINATED`, the driver
   restarts the VM and re-invokes the runner. Status files make resume
   automatic.

## Testing

End-to-end preemption test (no GCP needed, ~15s):

```bash
python -m tests.test_preemption
```

Spawns the runner with fake server/bench/metadata-server, flips the
preemption flag mid-sweep, and asserts that:

- the runner exits 75,
- the in-flight item is marked `interrupted`,
- a second invocation resumes and completes the remaining items.

The runner takes `--server-cmd`, `--bench-cmd`, `--preempt-url`, and
`--preempt-poll-s` flags to make this swap-in possible — they're also useful
in production if you want to point at a different inference server.

## Known gaps

- `bench_serving` / `bench_accuracy` don't emit a structured result file;
  the runner captures raw stdout. A `--output-json` flag on the bench
  scripts would let `summary.csv` carry actual TTFT/TPOT/throughput numbers.
- `driver.py` currently shells out to `rsync` over `gcloud compute ssh`. On
  setups requiring `--tunnel-through-iap` or a custom ssh config, edit
  `rsync_to()`.
