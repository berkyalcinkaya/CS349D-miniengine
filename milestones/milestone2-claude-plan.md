# Milestone 2 Implementation Plan

## Context recap (what we're working with)

- `core.Request.kv_cache: Any` — currently a `list[(K,V)]` per layer, each shape `(1, num_kv_heads, seq_len, head_dim)`. Grows via `torch.cat` in `model.Attention.forward` (`miniengine/model.py:247-248`).
- `engine.batched_decode` (`miniengine/engine.py:181-266`) pads every request's KV to `max_cache_len`, stacks, runs SDPA, then re-slices each request's real KV out — wasteful both in compute (padding) and in memory churn (every step rebuilds tensors).
- `engine.prefill` is per-request (no batched/packed prefill).
- `__main__.py` only accepts `--mode {baseline,batched}`.
- The `kv_memory_pool.py` skeleton fixes the public API but lets us pick storage layout, free-list shape, and page-table representation.

The 2× throughput target relative to `batched` is achievable mostly because milestone-1 batched is leaving large amounts of compute on the table to padding. Replacing it with **packed prefill + paged decode** removes the padding and removes the `torch.cat` per step.

---

## Design decisions

**Pool storage layout (flash-attn compatible):** per layer, two tensors of shape
`(num_pages, page_size, num_kv_heads, head_dim)`. This is exactly what `flash_attn_with_kvcache(..., block_table=..., cache_seqlens=...)` expects, so no transpose is needed in the hot path.

**Free list:** `collections.deque[int]` of free page indices. Allocation pops from the front; free pushes to the back. O(1) on both ends, simple, easy to reason about.

**Page table:** per-request `list[int]` of page indices, stored on the `Request`. We'll add two paging fields to `core.Request`:
- `page_table: list[int]`
- `num_kv_tokens: int` (logical KV length so far; lets us derive last-page offset cheaply)

We'll set `kv_cache` to `None` in paged mode and use the new fields exclusively. (We won't delete `kv_cache` — `batched` mode still uses it.)

**Pool capacity from `--mem-fraction-static`:** at engine init, after loading model weights, read `torch.cuda.mem_get_info()`, compute the budget as
```
budget = total_gpu_mem * mem_fraction_static - bytes_used_by_weights - safety_margin
num_pages = budget // (2 * num_layers * page_size * num_kv_heads * head_dim * dtype_bytes)
```
and instantiate the pool with `from_budget(...)`.

**Phased correctness → performance:** I'll build a *naïve paged path first* (gather pages → SDPA) so we have a known-good reference for accuracy. Then swap the attention call to flash-attn (`flash_attn_varlen_func` for prefill, `flash_attn_with_kvcache` for decode) for the throughput win. This isolates "did I get paging right" from "did I get the flash-attn API right."

---

## Step-by-step plan

### 1. `kv_memory_pool.py` — implement the pool (low risk, isolated)
- `__init__`: allocate two `(num_pages, page_size, num_kv_heads, head_dim)` tensors per layer (`list[tuple[K,V]]`), seed `_free` deque with `range(num_pages)`.
- `allocate(n)`: pop `n` indices, raise `RuntimeError` if `n > num_free`.
- `free(indices)`: extend the deque.
- `pages_needed(seq_len)`: `ceil_div(seq_len, page_size)`.
- `kv_caches`: return the per-layer tensor list (stable references, never reassigned).
- `from_budget`: classmethod that derives `num_pages` from a byte budget.

### 2. `core.py` — extend `Request`
Add:
```python
page_table: list[int] = field(default_factory=list)
num_kv_tokens: int = 0
```
Leaving `kv_cache` in place so milestone-1 modes keep working.

### 3. `model.py` — add a paged attention path
- Keep the existing `Attention.forward` as-is for `baseline`/`batched`.
- Add `Attention.forward_paged(hidden, cos, sin, k_cache, v_cache, slot_mapping, block_table, cache_seqlens, varlen_meta)` that:
  - Projects Q/K/V, applies `q_norm`/`k_norm`, RoPE.
  - **Scatters new K/V** into the pool: `k_cache.view(-1, num_kv_heads, head_dim)[slot_mapping] = k_new`.
  - **Computes attention.** Initial implementation: gather pages → SDPA (correctness reference). Final implementation: flash-attn (`flash_attn_varlen_func` if `varlen_meta` is set for prefill, else `flash_attn_with_kvcache` for decode).
- Add a parallel `TransformerBlock.forward_paged` and `TransformerModel.forward_paged` and `CausalLM.forward_paged` that thread the paging args through.

### 4. `engine.py` — packed prefill + paged decode
- `Engine.__init__`: when `mode == "paged"`, build the `KVMemoryPool` from `mem_fraction_static` and `page_size`. Store on `self.pool`.
- `prefill_packed(reqs: list[Request]) -> list[int]`:
  - For each request: allocate `pages_needed(seq_len)` pages, set `req.page_table`, set `req.num_kv_tokens = seq_len`.
  - Build packed `input_ids`, packed `position_ids` (per-request restart at 0), `cu_seqlens` (`int32` cumulative offsets), `slot_mapping` (one slot per packed token, derived from each request's page table).
  - Single forward through `model.forward_paged(..., is_prefill=True)`.
  - Sample one token per request from each request's last position.
- `decode_paged(reqs: list[Request]) -> list[int]`:
  - For each request: if `num_kv_tokens % page_size == 0`, allocate one more page; bump `num_kv_tokens += 1`.
  - Build `input_ids` (B, 1), `position_ids` (B, 1) = current `num_kv_tokens - 1`.
  - Build `block_table` (B, max_pages_per_req, padded with 0 where empty), `cache_seqlens` (B,).
  - Build `slot_mapping` for the new token only.
  - Single forward via `model.forward_paged(..., is_prefill=False)`. Returns logits → sample.
- Free pages: when a request finishes (in scheduler), call `pool.free(req.page_table)`.

### 5. `scheduler.py` — `_step_paged`
- Same iteration-level structure as `_step_batched`, but:
  - Phase 1: collect waiting requests; if `pool.num_free < pages_needed_for_prompt(req)`, leave it waiting (admission control).
  - Run `engine.prefill_packed(admitted)` once for the whole batch.
  - Phase 2: call `engine.decode_paged(self.running)`.
- On `_finish_request`, `pool.free(req.page_table)` and clear the page table.

### 6. `__main__.py` — CLI flags and wiring
Add to `parse_args`:
- Extend `--mode` choices to include `"paged"`.
- `--mem-fraction-static` (float, default 0.85).
- `--page-size` (int, default 32).
- `--torch-compile` (store_true).
Pass them through to `Engine` (which builds the pool) and `Scheduler`.

### 7. `torch.compile` (Part C)
- Target `MLP.forward` per-layer (stable shapes once shape-padding/bucketing is in place).
- Wrap each layer's `mlp` with `torch.compile(mlp, mode="default")` at engine init when `--torch-compile` is set.
- Avoid compiling the attention path (paged metadata varies in shape per step → recompiles).
- If MLP-only doesn't hit 10 %, expand to compile a "post-attn norm + MLP" residual block, still with stable shapes.

### 8. Verification
- **Correctness first:** with the SDPA gather-based paged path, run `bench_accuracy --dataset mmlu --num-samples 200` against `paged` and confirm parity with `batched`.
- **Then swap to flash-attn**, re-run accuracy.
- **Throughput:** `bench_serving --concurrencies 1,2,4,8,16,32` for `batched`, `paged`, `paged+torch-compile`.
- **Page-size sweep:** rerun at `--page-size 16` and `--page-size 128`.

---

## Risks / things that will probably bite

1. **Flash-attn API surface.** `flash_attn_with_kvcache` expects `block_table` as `(batch, max_blocks_per_seq) int32`, K-cache as `(num_blocks, page_size, num_kv_heads, head_dim)`. The version installed must support paged block tables — needs `flash-attn>=2.5`. Worth confirming on the L4 VM before committing to the API.
2. **Slot mapping correctness in prefill.** A packed prefill with `cu_seqlens` requires `slot_mapping` to be in the same packed order. Off-by-one or page-boundary bugs here are the #1 source of silent accuracy regressions. The phased SDPA-first plan exists specifically to catch these without the flash-attn variable in play.
3. **`torch.compile` recompile thrashing.** If decode batch size varies per step, dynamo will recompile. Either (a) compile only the MLP (input shape = `(total_tokens, hidden)`, varies but is 1D so dynamic compile works) with `dynamic=True`, or (b) bucket the decode batch size. (a) is simpler; try it first.
4. **Mem-fraction calculation.** `torch.cuda.mem_get_info()` returns *free* memory after weights load — don't double-subtract. The activation overhead during prefill+decode also needs a margin (10 % is reasonable).
5. **Existing `batched` mode still depends on `Request.kv_cache`.** Keep that field; only add paging fields. Don't break milestone 1.

---

## Suggested commit sequence
1. Pool implementation + unit-feel sanity (allocate/free round-trips).
2. `Request` paging fields + `--mode paged` skeleton wired through, but no model changes yet (just routes to a stub that errors).
3. `forward_paged` with **SDPA-based** gather-attn — correct but slow. Verify on MMLU.
4. Swap attention to flash-attn (`flash_attn_with_kvcache` for decode, `flash_attn_varlen_func` for prefill). Re-verify MMLU. Hit 2× throughput target.
5. `torch.compile` on MLP. Hit 10 % delta.
6. Page-size sweep + report numbers.
