"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Two decode paths:
  - decode_step(req)        : one request, used by baseline scheduler
  - batched_decode(reqs)    : many requests, one forward pass with padded
                              KV + attention mask, used by batched mode

Prefill stays per-request — variable prompt lengths make batched prefill
complex, and decode is where the throughput gain lives.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.kv_memory_pool import KVMemoryPool
from miniengine.model import CausalLM, ModelConfig, PagedMeta, load_weights
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


class Engine:
    """Model wrapper supporting baseline (per-request) and batched decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        mode: str = "batched",
        mem_fraction_static: float = 0.85,
        page_size: int = 32,
        torch_compile: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.mode = mode
        self.mem_fraction_static = mem_fraction_static
        self.page_size = page_size
        self.torch_compile = torch_compile

        # Populated below when mode == "paged".
        self.pool: KVMemoryPool | None = None

        # ── Tokenizer (still from HF — it's just a tokenizer) ──────────
        logger.info("Loading tokenizer from %s …", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )

        # ── Model (bare-bone PyTorch, loaded from safetensors) ──────────
        logger.info("Loading model config from %s …", model_path)
        config = ModelConfig.from_pretrained(model_path)
        logger.info(
            "Config: layers=%d, hidden=%d, heads=%d, kv_heads=%d, head_dim=%d, "
            "intermediate=%d, vocab=%d, tie_embed=%s",
            config.num_hidden_layers,
            config.hidden_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            config.intermediate_size,
            config.vocab_size,
            config.tie_word_embeddings,
        )

        # Build on meta device — load_weights replaces parameters with
        # GPU tensors directly, so we never allocate a CPU fp32 copy.
        with torch.device("meta"):
            self.model = CausalLM(config)
        load_weights(self.model, model_path, dtype=dtype, device=device)
        self.model.eval()

        # ── Stop tokens ─────────────────────────────────────────────────
        self.stop_token_ids: set[int] = set()
        if self.tokenizer.eos_token_id is not None:
            self.stop_token_ids.add(self.tokenizer.eos_token_id)
        for tok_name in ("eos_token", "pad_token"):
            tid = getattr(self.tokenizer, f"{tok_name}_id", None)
            if tid is not None:
                self.stop_token_ids.add(tid)
        for token_str in ("<|im_end|>", "<|endoftext|>", "<|end|>"):
            tid = self.tokenizer.convert_tokens_to_ids(token_str)
            if tid is not None and tid != self.tokenizer.unk_token_id:
                self.stop_token_ids.add(tid)

        logger.info(
            "Engine ready  —  vocab=%d, stop_ids=%s, params=%dM",
            len(self.tokenizer),
            self.stop_token_ids,
            sum(p.numel() for p in self.model.parameters()) // 1_000_000,
        )

        # ── Paged KV pool (milestone 2) ────────────────────────────────
        if mode == "paged":
            self._init_kv_pool(config)

        # ── torch.compile (Part C) ─────────────────────────────────────
        if torch_compile:
            self._apply_torch_compile()

    # ── torch.compile ───────────────────────────────────────────────────

    def _apply_torch_compile(self) -> None:
        """Compile per-layer MLPs.

        We target only the MLP sub-region. The attention path takes
        per-step paged metadata whose tensor shapes vary (B, total_tokens,
        cache_seqlens), which would trigger dynamo recompiles or
        graph-break fallbacks. The MLP is pure linear + SiLU + multiply on
        a (B, S, hidden) input — same op sequence in every forward.

        `dynamic=True` lets a single compiled graph handle the varying
        leading dim (total_tokens during prefill, B during decode) without
        recompiling per shape.
        """
        if not self.device.startswith("cuda"):
            logger.warning("--torch-compile requires CUDA; skipping")
            return

        layers = self.model.model.layers
        for layer in layers:
            layer.mlp = torch.compile(layer.mlp, mode="default", dynamic=True)
        logger.info(
            "torch.compile applied to %d MLP modules (mode=default, dynamic=True)",
            len(layers),
        )

    # ── Paged KV pool init ──────────────────────────────────────────────

    def _init_kv_pool(self, config: ModelConfig) -> None:
        """Allocate the paged KV pool sized from --mem-fraction-static.

        Budget = total_gpu_mem * mem_fraction_static
                 - bytes_used_by_weights
                 - 5% safety margin (activations / fragmentation).
        """
        if not self.device.startswith("cuda"):
            raise RuntimeError("--mode paged requires --device cuda")

        device_idx = (
            torch.device(self.device).index
            if torch.device(self.device).index is not None
            else torch.cuda.current_device()
        )
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)

        weights_bytes = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        )
        static_budget = int(total_bytes * self.mem_fraction_static)
        safety_margin = int(0.05 * total_bytes)
        pool_budget = static_budget - weights_bytes - safety_margin
        if pool_budget <= 0:
            raise RuntimeError(
                f"KV pool budget non-positive: total={total_bytes / 1e9:.1f} GB, "
                f"weights={weights_bytes / 1e9:.1f} GB, "
                f"mem_fraction_static={self.mem_fraction_static}. "
                "Lower --mem-fraction-static or use a smaller model."
            )

        self.pool = KVMemoryPool.from_budget(
            num_layers=config.num_hidden_layers,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            page_size=self.page_size,
            dtype=self.dtype,
            device=self.device,
            bytes_budget=pool_budget,
        )
        logger.info(
            "KV pool ready  —  %d pages × %d tokens (%.2f GB), "
            "weights=%.2f GB, total_gpu=%.2f GB",
            self.pool.num_pages,
            self.page_size,
            (pool_budget) / 1e9,
            weights_bytes / 1e9,
            total_bytes / 1e9,
        )

    # ── Tokenization ────────────────────────────────────────────────────

    def tokenize_messages(self, messages: list[dict[str, str]]) -> list[int]:
        """Apply the model's chat template and tokenize into ids."""
        kwargs: dict[str, Any] = dict(
            tokenize=False,
            add_generation_prompt=True,
        )
        # Qwen3 models support enable_thinking; silently ignore if unsupported
        try:
            text = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **kwargs
            )
        except TypeError:
            text = self.tokenizer.apply_chat_template(messages, **kwargs)
        return self.tokenizer.encode(text, add_special_tokens=False)

    def decode_token(self, token_id: int) -> str:
        """Decode a single token id back to a string."""
        return self.tokenizer.decode([token_id], skip_special_tokens=True)

    # ── Forward passes ──────────────────────────────────────────────────

    @torch.inference_mode()
    def prefill(self, request: Request) -> int:
        """
        Run the prefill phase for one request.

        Processes the full prompt in a single forward pass, stores the
        resulting KV cache on the request, and samples the first output
        token.

        Returns:
            The first generated token id.
        """
        input_ids = torch.tensor(
            [request.input_ids], dtype=torch.long, device=self.device
        )
        seq_len = input_ids.shape[1]
        position_ids = torch.arange(seq_len, device=self.device).unsqueeze(0)

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=None)
        request.kv_cache = kv_caches

        # Sample from the last position
        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def decode_step(self, request: Request) -> int:
        """
        Run one decode step for a request that has already been prefilled.

        Feeds the last generated token through the model together with the
        cached KV values, updates the cache, and samples the next token.

        Returns:
            The next generated token id.
        """
        input_ids = torch.tensor(
            [[request.output_ids[-1]]], dtype=torch.long, device=self.device
        )
        # Position = current KV cache length (= num tokens already processed)
        cache_len = request.kv_cache[0][0].shape[2]  # layer 0, key tensor, seq dim
        position_ids = torch.tensor([[cache_len]], device=self.device)

        logits, kv_caches = self.model(
            input_ids, position_ids, kv_caches=request.kv_cache
        )
        request.kv_cache = kv_caches

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids

    # ── Batched decode ──────────────────────────────────────────────────

    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]) -> list[int]:
        """
        Decode one token for each request in a single forward pass.

        Pads per-request KV caches to the longest in the batch, builds a
        float attention mask that ignores padding, runs the model once,
        then extracts each request's actual KV (real prefix + new token)
        and samples its next token.
        """
        if not requests:
            return []

        batch_size = len(requests)
        num_layers = len(requests[0].kv_cache)

        # Stack last generated token from each request → (batch, 1)
        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long,
            device=self.device,
        )

        # Each request's current KV length and the per-request RoPE position
        cache_lens = [req.kv_cache[0][0].shape[2] for req in requests]
        max_cache_len = max(cache_lens)
        position_ids = torch.tensor(
            [[cl] for cl in cache_lens],
            dtype=torch.long,
            device=self.device,
        )

        # Pad and stack KV caches per layer to (batch, kv_heads, max_cache_len, head_dim)
        padded_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_list, v_list = [], []
            for req in requests:
                k, v = req.kv_cache[layer_idx]
                pad_len = max_cache_len - k.shape[2]
                if pad_len > 0:
                    k = F.pad(k, (0, 0, 0, pad_len))
                    v = F.pad(v, (0, 0, 0, pad_len))
                k_list.append(k)
                v_list.append(v)
            padded_kv_caches.append(
                (torch.cat(k_list, dim=0), torch.cat(v_list, dim=0))
            )

        # Mask shape (batch, 1, 1, max_cache_len + 1): the attention forward
        # appends the new token to the cache, so kv_len = max_cache_len + 1.
        # Mask only the padding window [cl, max_cache_len) per request.
        attention_mask = torch.zeros(
            batch_size,
            1,
            1,
            max_cache_len + 1,
            device=self.device,
            dtype=self.dtype,
        )
        for i, cl in enumerate(cache_lens):
            attention_mask[i, 0, 0, cl:max_cache_len] = float("-inf")

        logits, new_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=padded_kv_caches,
            attention_mask=attention_mask,
        )

        # Extract each request's real KV (actual prefix + new token at -1).
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            cl = cache_lens[i]
            per_req_kv = []
            for layer_idx in range(num_layers):
                k_full = new_kv_caches[layer_idx][0][i : i + 1]
                v_full = new_kv_caches[layer_idx][1][i : i + 1]
                k_new = torch.cat([k_full[:, :, :cl, :], k_full[:, :, -1:, :]], dim=2)
                v_new = torch.cat([v_full[:, :, :cl, :], v_full[:, :, -1:, :]], dim=2)
                per_req_kv.append((k_new, v_new))
            req.kv_cache = per_req_kv
            token_ids.append(
                sample_token(
                    logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids

    # ── Paged path (milestone 2) ────────────────────────────────────────

    @torch.inference_mode()
    def prefill_packed(self, requests: list[Request]) -> list[int]:
        """Packed batched prefill over the paged KV pool.

        Allocates pages for each request, packs all prompts into a
        single forward pass, scatters the resulting K/V into the pool,
        and returns the first sampled token per request.
        """
        if not requests:
            return []
        assert self.pool is not None
        page_size = self.pool.page_size

        # Allocate pages and stamp each request's page table / KV length.
        for req in requests:
            seq_len = req.num_input_tokens
            num_pages = self.pool.pages_needed(seq_len)
            req.page_table = self.pool.allocate(num_pages)
            req.num_kv_tokens = seq_len

        seq_lens = [r.num_input_tokens for r in requests]
        total_tokens = sum(seq_lens)

        # Packed input_ids / position_ids (positions restart per request).
        flat_input_ids: list[int] = []
        flat_positions: list[int] = []
        for req, s in zip(requests, seq_lens):
            flat_input_ids.extend(req.input_ids)
            flat_positions.extend(range(s))
        input_ids = torch.tensor(
            flat_input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)  # (1, total_tokens)
        position_ids = torch.tensor(
            flat_positions, dtype=torch.long, device=self.device
        ).unsqueeze(0)  # (1, total_tokens)

        # cu_seqlens (Python list — used for slicing q per request inside attn)
        cu_seqlens: list[int] = [0]
        for s in seq_lens:
            cu_seqlens.append(cu_seqlens[-1] + s)

        # slot_mapping: one entry per packed token, in packed order.
        slot_mapping = self._build_slot_mapping(requests, seq_lens, page_size)

        # block_table padded to max blocks across the batch.
        block_table = self._build_block_table(requests)

        # GPU tensors required by flash-attn (varlen / with_kvcache).
        cu_seqlens_tensor = torch.tensor(
            cu_seqlens, dtype=torch.int32, device=self.device
        )
        cache_seqlens_tensor = torch.tensor(
            seq_lens, dtype=torch.int32, device=self.device
        )

        paged_meta = PagedMeta(
            is_prefill=True,
            page_size=page_size,
            slot_mapping=slot_mapping,
            block_table=block_table,
            cu_seqlens=cu_seqlens,
            cache_seqlens=list(seq_lens),
            cu_seqlens_tensor=cu_seqlens_tensor,
            cache_seqlens_tensor=cache_seqlens_tensor,
            max_seqlen=max(seq_lens),
        )

        logits = self.model.forward_paged(
            input_ids, position_ids, self.pool.kv_caches, paged_meta
        )  # (1, total_tokens, vocab)

        # Sample first token from each request's last position.
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            last_idx = cu_seqlens[i + 1] - 1
            token_ids.append(
                sample_token(
                    logits[:, last_idx, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids

    @torch.inference_mode()
    def decode_paged(self, requests: list[Request]) -> list[int]:
        """One paged decode step for every running request.

        Allocates a fresh page for any request that's about to overflow
        its current last page, scatters the new K/V into the pool, runs
        attention through the per-request page table, and samples one
        new token per request.
        """
        if not requests:
            return []
        assert self.pool is not None
        page_size = self.pool.page_size

        # Grow page tables and bump KV lengths.
        for req in requests:
            new_pos = req.num_kv_tokens  # 0-indexed slot of the new token
            if new_pos // page_size >= len(req.page_table):
                req.page_table.extend(self.pool.allocate(1))
            req.num_kv_tokens += 1

        bsz = len(requests)
        # input_ids: (B, 1) — last generated token per request.
        input_ids = torch.tensor(
            [[r.output_ids[-1]] for r in requests],
            dtype=torch.long,
            device=self.device,
        )
        # position_ids: (B, 1) — index of the new KV slot.
        position_ids = torch.tensor(
            [[r.num_kv_tokens - 1] for r in requests],
            dtype=torch.long,
            device=self.device,
        )

        slot_mapping = self._build_decode_slot_mapping(requests, page_size)
        block_table = self._build_block_table(requests)

        cache_seqlens_list = [r.num_kv_tokens for r in requests]
        cu_seqlens_list = list(range(bsz + 1))
        cu_seqlens_tensor = torch.tensor(
            cu_seqlens_list, dtype=torch.int32, device=self.device
        )
        cache_seqlens_tensor = torch.tensor(
            cache_seqlens_list, dtype=torch.int32, device=self.device
        )

        paged_meta = PagedMeta(
            is_prefill=False,
            page_size=page_size,
            slot_mapping=slot_mapping,
            block_table=block_table,
            cu_seqlens=cu_seqlens_list,
            cache_seqlens=cache_seqlens_list,
            cu_seqlens_tensor=cu_seqlens_tensor,
            cache_seqlens_tensor=cache_seqlens_tensor,
            max_seqlen=max(cache_seqlens_list) if cache_seqlens_list else 0,
        )

        logits = self.model.forward_paged(
            input_ids, position_ids, self.pool.kv_caches, paged_meta
        )  # (B, 1, vocab)

        token_ids: list[int] = []
        for i, req in enumerate(requests):
            token_ids.append(
                sample_token(
                    logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
                )
            )
        return token_ids

    # ── Paged-mode metadata builders ────────────────────────────────────

    def _build_slot_mapping(
        self, requests: list[Request], seq_lens: list[int], page_size: int
    ) -> torch.Tensor:
        """Slot indices (in the flat pool view) for each packed prefill token."""
        slots: list[int] = []
        for req, s in zip(requests, seq_lens):
            for tok_pos in range(s):
                page_idx = req.page_table[tok_pos // page_size]
                slots.append(page_idx * page_size + (tok_pos % page_size))
        return torch.tensor(slots, dtype=torch.long, device=self.device)

    def _build_decode_slot_mapping(
        self, requests: list[Request], page_size: int
    ) -> torch.Tensor:
        """Slot indices for the single new token each request appends this step."""
        slots: list[int] = []
        for req in requests:
            new_pos = req.num_kv_tokens - 1
            page_idx = req.page_table[new_pos // page_size]
            slots.append(page_idx * page_size + (new_pos % page_size))
        return torch.tensor(slots, dtype=torch.long, device=self.device)

    def _build_block_table(self, requests: list[Request]) -> torch.Tensor:
        """(B, max_blocks) int32 page-id table padded with 0 past each row's length.

        int32 is the layout flash-attn expects.  The SDPA gather path
        casts to long internally for fancy indexing.
        """
        max_blocks = max(len(r.page_table) for r in requests)
        bt = torch.zeros(
            (len(requests), max_blocks), dtype=torch.int32, device=self.device
        )
        for i, r in enumerate(requests):
            bt[i, : len(r.page_table)] = torch.tensor(
                r.page_table, dtype=torch.int32, device=self.device
            )
        return bt
