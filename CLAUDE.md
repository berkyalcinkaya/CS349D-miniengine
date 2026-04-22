# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

MiniEngine is an educational LLM serving engine for **CS349D**. It serves an OpenAI-compatible HTTP API and is deliberately structured so the **scheduler is the primary optimization target** across a series of milestone assignments. See `milestones/milestone1.md` for the active assignment (batching / continuous batching).

## Common commands

```bash
# Install (editable, with benchmark deps)
pip install -e .[bench]

# Launch the server (downloads model on first run)
python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507 --dtype bfloat16

# Serving benchmark — throughput/latency under concurrency sweep
python -m benchmark.bench_serving --input-len 512 --output-len 256 --concurrencies 1,4,8,16

# Accuracy benchmark — MMLU or GSM8K (verifies optimizations don't regress quality)
python -m benchmark.bench_accuracy --dataset mmlu --num-samples 200
python -m benchmark.bench_accuracy --dataset gsm8k --num-samples 100
```

The project has no test suite, linter, or formatter configured. Validation is done via the two benchmarks against a running server.

GPU development happens on a remote VM (see `setup-vm/README.md`). `scripts/fetch_and_analyze_logs.sh` is a launchd-invoked helper that scp's `server.log`/`benchmark.log` back from `inference-engine-vm:/workspace/CS349D-miniengine` into `logs/` and runs Claude on them — it's tooling, not part of the engine.

## Architecture

The request flow is: **client → FastAPI server → Scheduler (background thread) → Engine → model** — and back along a per-request `token_queue` for streaming.

- `miniengine/core.py` — the data types that thread through the whole system: `Request` (carries `input_ids`, `output_ids`, `kv_cache`, `token_queue`, `status`), `SamplingParams`, `TokenOutput`, `RequestStatus`. The `kv_cache` lives on the `Request` itself, not in a central pool.
- `miniengine/model.py` — bare-bone Qwen3 transformer in pure PyTorch (RMSNorm, GQA + QK-Norm + RoPE attention, SwiGLU MLP). Module and weight names match the HuggingFace checkpoint so `load_weights()` can `load_state_dict()` directly from safetensors. **No HF model classes** — future work like tensor parallelism shards the `nn.Linear` layers directly.
- `miniengine/engine.py` — thin wrapper around `CausalLM` providing `prefill(request)` and `decode_step(request)`. Both are **single-request** today; batched variants are the first optimization (see milestone 1). Also owns the tokenizer and the stop-token set.
- `miniengine/scheduler.py` — the optimization target. Runs in a daemon thread calling `step()` in a loop. **Current `step()` is maximally naive**: pulls one request off `waiting`, prefills, then decodes to completion before touching anything else. `add_request()` is the thread-safe entry point from the server.
- `miniengine/server.py` — FastAPI app, OpenAI-compatible `/v1/chat/completions` (streaming + non-streaming), `/v1/models`, `/health`. Thin — only HTTP ↔ `Request` translation. Module-level globals `engine`, `scheduler`, `model_id` are wired up by `__main__.py` before `uvicorn.run`.
- `miniengine/sampler.py` — `sample_token(logits, params, output_ids)` for greedy / temperature / top-k / top-p / repetition-penalty. Operates on `(1, vocab)` logits; batched sampling will need a new entry point.
- `miniengine/__main__.py` — CLI entry point; constructs `Engine` + `Scheduler`, assigns them onto `server` module globals, starts the scheduler thread, then launches uvicorn.

### Key invariants to preserve when changing the scheduler/engine

- **Stream semantics**: each generated token must be pushed to `req.token_queue` as a `TokenOutput`, and a final `TokenOutput(finished=True)` must be pushed exactly once per request, else the server's `_stream_response` / `_collect_full_response` will hang.
- **KV cache shape**: `request.kv_cache` is a `list[(k, v)]` per layer, each tensor shaped `(batch, num_kv_heads, cache_len, head_dim)`. `decode_step` derives the next position id from `kv_cache[0][0].shape[2]`.
- **Finish conditions** live in `Scheduler._check_finished`: `req.is_finished` (hit `max_new_tokens`) or `engine.is_stop_token(token_id)`. The sentinel `TokenOutput` must follow.
- **RoPE correctness when batching**: each request's `position_ids` are its own cache length, not the padded length (see milestone 1 notes).
- **Prefill stays unbatched** per milestone 1 guidance — the throughput win is in the decode phase.

### Model assumptions

Built around Qwen3-style models (GQA, QK-Norm, tied embeddings for Qwen3-4B). `ModelConfig.from_pretrained` pulls fields off HF's `AutoConfig`; the loader tolerates sharded checkpoints and tied `lm_head`. `trust_remote_code=True` is used for the tokenizer.
