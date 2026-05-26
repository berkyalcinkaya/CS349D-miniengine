# Semi-fault-tolerant benchmarking workflow (Milestone 2)

<!--
  How to use this doc with a coding LLM:
  - Read Quick links first, then Context, then Goals vs Current process.
  - Before writing code, fill or answer the sections under Implementation handoff.
  - Prefer citing repo paths from Quick links when proposing changes.
-->

## Document purpose

- **Intent:** Shape automation and scripts for GPU spot-instance benchmarking: detect termination, relaunch, and rerun interrupted workflows.
- **Audience:** Implementer (human or LLM) turning this brainstorm into code and runbooks.
- **Status:** Ideas / planning (not a finalized spec).

## Quick links

| Resource | Path |
|----------|------|
| Milestone 2 report expectations | `milestones/milestone2.md` |
| Serving benchmark | `benchmark/bench_serving.py` |
| Accuracy benchmark | `benchmark/bench_accuracy.py` |

## Context

We run benchmark workloads on a GPU spot instance, meaning we can loose the instance whenever. Help me design an automation job and accompanying scripts that detects spot instance termination and relaunchs the benchmark job needed.

## Goals (problem statement)

### Two primary challenges

1. benchmarking requires parameter sweeping on both the server and benchmark CLI arguments (see milestones/milestone2.md report section and benchmarks/ for an understanding here). This requires a lot of human labor to run each individually.
2. spot instances are frequently terminated in the middle of runs. We want to detect termination, restart, and rerun interrupted benchmarking worflows.

### What is a benchmark worflow

See this example from milestones/milestone2.md
"
1. **Accuracy.** `bench_accuracy` on MMLU and/or GSM8K showing your
   paged engine (with and without `--torch-compile`) matches the
   milestone-1 baseline within noise.

2. **Throughput.** `bench_serving` for at least:
   - milestone-1 `batched`,
   - milestone-2 `paged` (≥ 2× throughput target),
   - `paged + torch.compile` (≥ 10% over `paged`),
   - `paged + torch.compile + cuda-graph` if you did the extra
     credit (≥ 20% over `paged + torch.compile`).

   Each screenshot should show TTFT p50/p99, TPOT p50/p99, and
   generation throughput.

3. **Page-size comparison.** `bench_serving` at two `--page-size`
   values (e.g., 16 vs 128) with a short comment.
" where these arguments specify CLI args for the server.

Additionally, note that benchmark_serving

### The commands that we currently run to perform a benchmark worflow

1. locally, manually try `gcloud compute instances start inference-engine-vm --zone=us-central1-a` until the instance is successfully started.
2. locally, ssh into remote with `gcloud compute ssh inference-engine-vm --zone=us-central1-a`
3. on remote, cd into current directory start server with correct CLI arguments for benchmark:

```bash
python3 -m miniengine --model Qwen/Qwen3-8B \
    --mode paged --mem-fraction-static 0.85 \
    --page-size 32 --torch-compile
```

4. in another terminal, run the benchmark either @benchmark/bench_accuracy.py or
@benchmarks/bench_serving.py. Often times we want to sweep over parameters (see milestone2.md report section), which involves running and aggregating results manually on the remote.

5. instance terminates. In this case we must restart the workflow:

## Requirements and design ideas

1. Manually waiting for each individual parameter sweep requires human oversight. We want to determine a format for defining benchmark sweeps locally that can be exited in the remote in an automated fashion. This involves defining the individual commands that make up a sweep and executing them in a fault tolerant manner
2. Concurrency sweeps in bench_serving are problematic but neccessary. In our workflow ch concurrency level should be specified independently on sweeps but this means

## Design ideas

A primitive would be a way to run an individual command in a manner that detects and automates restarts. Then a benchmark workflow amalgamates these commands.

---

## Implementation handoff (for coding agents)

Use this section to force structured output before or alongside implementation.

### Clarifications to produce

- List **ambiguous or truncated** bullets in this doc and state what you need from the author vs what you will assume.
- List **external dependencies** (GCP APIs, metadata server behavior, SSH, quotas) you will rely on.

### Constraints to state explicitly

- **In scope:** (derive from Goals and Requirements above; keep tight.)
- **Out of scope:** (what this change will explicitly not do.)
- **Safety / ops:** (e.g., no destructive actions on shared VMs, secrets handling.)

### Artifacts to deliver

Check off as you produce them (leave unchecked until done):

- [ ] Sweep definition format (schema or file format) and example file
- [ ] Runner/orchestrator behavior on clean start vs mid-sweep resume
- [ ] Spot / preemption detection approach and restart policy
- [ ] Where results and logs land; how aggregation matches current manual flow