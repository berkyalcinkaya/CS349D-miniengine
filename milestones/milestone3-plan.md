# Milestone 3 Implementation Plan — Chunked Prefill + Radix Prefix Cache

This plan implements the two optimizations in `milestone3.md` on top of the
milestone-2 paged engine. **Part A first, then Part B on top of it.** A key
insight drives the design: a radix-cache hit turns prefill into a *partial*
prefill (compute only the uncached suffix, attend to the full prefix), which is
exactly the same machinery chunked prefill needs (process a token sub-range
that attends to everything before it). So both features share one generalized
"segmented prefill" path.

---

## Background: how milestone-2 paged prefill works today

`Scheduler._step_paged`:
1. **Admit** waiting reqs while `pool.num_free >= pages_needed(prompt)`.
2. **Prefill** — `engine.paged_batched_prefill(reqs)` allocates a full page
   table per request, packs *all* prompts into one varlen forward
   (`cu_seqlens`, `slot_mapping`/`block_table` for flash_attn, or a planned
   `fi_ctx` for flashinfer), gathers the last-position logit per request
   (`logits_indices`), samples the first token. Sets `state.cache_seq_len = n`.
3. **Decode** — `paged_batched_decode` grows page tables (via `pool.allocate`)
   and runs one token/request.
4. **Finish** — `free_paged_state` returns the request's pages to the pool.

The OOM Part A targets is **activation memory**: that single packed prefill
forward materializes attention scores + MLP intermediates proportional to the
total q-token count (`sum(prompt_len)`), independent of the (static)
pre-allocated KV pool. With high concurrency × long prompts that tensor
outsizes the `1 - mem_fraction_static` activation headroom → CUDA OOM.

---

## Part A — Chunked Prefill

### Idea
Replace the "one forward over all prompt tokens" with a **sequence of forwards,
each capped at `prefill_chunk_size` q-tokens**. Activation memory then scales
with the chunk size, not the prompt length. KV for earlier chunks is already in
the pool; a later chunk's queries attend (causally) to all prior tokens of the
same request via the page table — both flash_attn (`causal=True` with
`seqlen_q < seqlen_k` → bottom-right alignment) and flashinfer
(`BatchPrefillWithPagedKVCacheWrapper`, same convention) support this directly.

### Generalized segmented prefill (subsumes single-shot + cache hits)
Define a prefill *segment* for one request in one forward:
- `q_start` — absolute token offset where this forward's queries begin,
- `q_len`   — number of query tokens this forward,
- `kv_len = q_start + q_len` — tokens attended (everything up to the chunk end).

A forward processes a list of segments (possibly from several requests), packed
varlen. `q_start = prefix_len` (0 normally; >0 after a radix hit — Part B).
Chunking just walks each request's `[prefill_start, prompt_len)` range in
`chunk_size`-token slices, greedily packing slices from multiple requests up to
the per-forward budget. `chunk_size == 0` ⇒ budget = ∞ ⇒ exactly one segment
per request spanning the whole prompt ⇒ **bit-identical to the milestone-2
single-shot path** (verified by reduction in the metadata builders below).

When a segment's `kv_len == prompt_len`, that request *completes* prefill in
this forward → its last query position is added to `logits_indices`, its first
token sampled, `state.cache_seq_len = prompt_len`, and it joins the decode set.

### Files / changes
- **`engine.py`**
  - `Engine.__init__`: add `prefill_chunk_size: int = 0`, store on `self`.
  - Rewrite `paged_batched_prefill(requests)` to:
    1. for each req: radix `match_prefix` (Part B) → `prefix_len`, borrowed
       pages; allocate only the suffix pages; build the page table
       (`borrowed + new`); set `prefill_start = prefix_len`, `cache_hit_tokens`.
       (With cache disabled, `prefix_len = 0`, allocate full table — same as
       today.)
    2. Build the chunk schedule (greedy pack to `prefill_chunk_size`).
    3. For each chunk-forward: build metadata via generalized builders, run,
       gather completing-request logits, sample.
  - Replace `_fa_kwargs_prefill` / `_fi_kwargs_prefill` with generalized
    `_fa_kwargs_prefill_segmented(segs, states)` /
    `_fi_kwargs_prefill_segmented(segs, states)` that take per-segment
    `(req, q_start, q_len, kv_len)`. Reduction check: one full segment per
    request reproduces today's tensors exactly.
- **`__main__.py`**: add `--prefill-chunk-size N` (default `0`), thread into
  `Engine(...)`.
- **`scheduler.py`**: admission gate uses `pool.available_pages`
  (`num_free + num_evictable`) instead of `num_free` so cache-held pages can be
  reclaimed (needed once Part B holds pages); otherwise unchanged — the engine
  loops chunks internally, so the scheduler still calls
  `paged_batched_prefill(to_prefill)` once per step.

### Part A targets / report
- **OOM avoidance**: e.g. `--input-len 8192 --concurrencies 16` (or higher)
  OOMs at `--prefill-chunk-size 0`, succeeds at `--prefill-chunk-size 512/1024`.
- **No regression**: `--prefill-chunk-size 512` with `--input-len 4096`
  (~8 chunks/req) vs `0`, at conc 1/4/16 — throughput, TTFT p50/p99, TPOT
  within noise; `bench_accuracy` MMLU within noise.
- **Chunk-size reasoning** (for the report): too small ⇒ kernel-launch overhead
  per chunk dominates; too large ⇒ defeats OOM avoidance. Sweet spot makes the
  per-chunk activation tensor (`chunk_size × intermediate_size`, attention
  `chunk_size × kv_len`) fit comfortably in the `1 - mem_fraction_static`
  headroom while keeping chunk count (hence launch overhead) low. Characterize
  the `chunk_size × input_len × concurrency` vs free-activation-memory relation.

---

## Part B — Radix Prefix Cache

### Data structure (`radix_cache.py`)
Radix tree, **page-granular** (every edge key length and page count is a
multiple of `page_size`). Each `RadixNode`: `parent`, `children` (keyed by the
edge's first token id), `key` (edge tokens), `pages` (KV page indices,
`len == len(key)//page_size`), `ref_count` (# locked leaves in subtree),
`last_access` (LRU). The cache owns its pages — they are **not** in the pool
free list until evicted.

Methods to implement:
- `match_prefix(tokens) -> MatchResult`: walk root→down comparing page-aligned
  spans; accumulate `matched_pages`/`matched_tokens` until a partial page,
  divergence, or token exhaustion; `last_node` = deepest matched node. Update
  `total_lookups/total_query_tokens/total_hit_tokens`.
- `inc_lock_ref(node)` / `dec_lock_ref(node)`: walk node→root adjusting
  `ref_count`; refresh `last_access`.
- `insert_and_return(tokens, pages) -> (leaf, redundant_pages)`: walk down
  matching page-aligned spans, **splitting** a node at a page boundary on
  partial match, then attach the remaining `(tokens, pages)` as a new child.
  For spans that match an existing edge: if the incoming page index **equals**
  the cached one (borrowed-prefix re-insert) it is kept silently; if it
  **differs** (two requests computed the same prefix independently) the incoming
  page is *redundant* and returned for the caller to free. Bump
  `total_inserted_pages`.
- `evict(n) -> int`: LRU over unlocked leaves (min-heap on `last_access`); free
  pages to the pool, delete node, re-leaf parents; stop at `n` or exhaustion.
  Bump `total_evicted_pages`. Never touch a node with `ref_count > 0`.
- `num_cached_pages`, `num_evictable_pages`, `reset`.

### Pool / engine / scheduler wiring
- **`kv_memory_pool.py`**:
  - add `self.radix_cache = None`; `allocate(n)`: if `free < n` and a cache is
    attached, call `radix_cache.evict(n - free)` before the OOM check.
  - add `num_evictable` property (delegates to cache) and `available_pages`
    (`num_free + num_evictable`).
- **`engine.py`**:
  - `__init__`: add `disable_radix_cache: bool = False`; in paged mode create
    `self.radix_cache = RadixCache(self.pool)` unless disabled, and set
    `self.pool.radix_cache = self.radix_cache`. Expose `self.radix_cache = None`
    attribute always (server `/cache_stats` reads it).
  - prefill: per request `match_prefix` (cap to `prompt_len - 1` tokens floored
    to a page so at least the last token is recomputed for its logits); borrow
    matched pages, `inc_lock_ref(last_node)`, store `last_node` + `prefix_len`
    in `_PagedState`; set `req.cache_hit_tokens = prefix_len`.
  - `free_paged_state` (cache on): build `tokens = input_ids + output_ids[:-1]`
    (KV exists for all but the last sampled token), floor to full pages,
    `insert_and_return(aligned_tokens, page_table[:n_full_pages])`,
    `dec_lock_ref(stored last_node)`, free `redundant_pages` + the partial tail
    page back to the pool. Pages that entered the tree stay owned by the cache.
- **`_PagedState`**: add `prefix_len: int = 0` and `prefix_node: RadixNode|None`.
- **`__main__.py`**: add `--disable-radix-cache` (cache on by default), thread
  into `Engine(...)`.

### Concurrency note
All pool/cache mutation happens in the single scheduler thread (prefill →
decode → finish). The async HTTP layer only enqueues requests and reads
`metrics`, so the tree needs no extra locking.

### Part B targets / report
- `shared` workload, `--shared-prefix-len 2000`, cache on vs off → ≥2×
  throughput and ≥2× TTFT improvement; sweep prefix-len {200,500,2000,4000}.
- `multiturn`, long enough conversation → ≥50% throughput and TTFT improvement
  aggregate; per-turn hit rate 0% at turn 0, climbing after.
- `bench_serving` default (low sharing): cache on vs off within noise at
  conc 1/4/16.

---

## Bonus — Retraction (optional, implement if time permits)
When `pool.allocate` during decode can't be satisfied even after eviction, pick
a victim from the running set (youngest / largest remaining work), free its KV
pages, and requeue it to `waiting` for re-prefill. Edge cases: don't retract a
request mid-chunked-prefill; pinned (locked) cache pages are already protected
by `ref_count`. Deferred unless required scope finishes with margin.

---

## Verification strategy
- **Local (no GPU here):** a pure-Python unit test (`tests/test_radix_cache.py`)
  with a fake pool exercising `match_prefix`, `insert_and_return` (incl. node
  split + redundant-page dedup), `evict` (LRU + lock protection), and
  lock-ref counting. Plus `python -m py_compile` on all touched modules.
- **On the L4 GPU (for the report):** run the benchmark matrix above; capture
  terminal screenshots of the OOM/success pair, MMLU parity, the cache
  speedups, the per-turn curve, and the no-regression serving runs.

## Implementation order
1. `kv_memory_pool.py` — eviction hook + `num_evictable`/`available_pages`.
2. `radix_cache.py` — fill in all method bodies.
3. `tests/test_radix_cache.py` — unit test the tree (runnable locally).
4. `engine.py` — generalized segmented prefill + cache wiring + `free_paged_state`.
5. `scheduler.py` — admission gate via `available_pages`.
6. `__main__.py` — `--prefill-chunk-size`, `--disable-radix-cache`.
7. `py_compile` everything; run the unit test.
