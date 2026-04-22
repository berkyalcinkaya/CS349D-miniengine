"""
Model engine — wraps the bare-bone CausalLM for serving.

The engine is a "black box" that the scheduler calls into.  It handles:
  1. Model loading and GPU placement (via model.py + safetensors)
  2. Tokenization / detokenization (chat-template aware via AutoTokenizer)
  3. Prefill (prompt → first token + KV cache)
  4. Decode  (previous token + KV cache → next token + updated KV cache)
  5. Token sampling (delegated to sampler.py)

Design note:
  The current API is single-request (prefill / decode_step).  A natural
  first optimisation is to add batched versions that pad sequences and run
  multiple requests through a single forward pass.

  For tensor parallelism, the bare-bone nn.Linear layers in model.py can
  be sharded directly: Q/K/V/gate/up column-wise, O/down row-wise, with
  an all-reduce after the row-parallel matmul.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import AutoTokenizer

from miniengine.core import Request
from miniengine.model import CausalLM, ModelConfig, load_weights
from miniengine.sampler import sample_token

logger = logging.getLogger(__name__)


class Engine:
    """Bare-bone model wrapper for single-request prefill and decode."""

    def __init__(
        self,
        model_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.device = device
        self.dtype = dtype

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

        logits, kv_caches = self.model(input_ids, position_ids, kv_caches=request.kv_cache)
        request.kv_cache = kv_caches

        return sample_token(
            logits[:, -1, :], request.sampling_params, request.output_ids
        )

    @torch.inference_mode()
    def batched_decode(self, requests: list[Request]) -> list[int]:
        """
        Run one decode step for a batch of already-prefilled requests in a
        single model forward pass.

        Per-request KV caches have different sequence lengths, so we pad all
        caches to ``max_cache_len`` along the sequence dim and build an
        attention mask that hides padding positions.  After the forward, we
        scatter the newly-generated token from the trailing padding slot back
        into each request's real cache_len position, then hand each request a
        view into the padded tensor — no per-request torch.cat per layer.

        Args:
            requests: list of RUNNING requests (must have ``kv_cache`` set).

        Returns:
            A list of sampled token ids, one per input request.
        """
        if not requests:
            return []

        device = self.device
        batch_size = len(requests)

        # Per-layer KV caches live as (k, v) tuples on each request.  Every
        # request has the same number of layers / heads / head_dim, only the
        # seq dimension (cache length) differs between requests.
        num_layers = len(requests[0].kv_cache)
        sample_k = requests[0].kv_cache[0][0]
        _, num_kv_heads, _, head_dim = sample_k.shape
        kv_dtype = sample_k.dtype

        # Cache lengths per request — used for padding, RoPE position ids,
        # mask construction, and later KV-cache slicing.
        cache_lens = [req.kv_cache[0][0].shape[2] for req in requests]
        max_cache_len = max(cache_lens)

        # Inputs: each request contributes its most recent output token and a
        # position id equal to its current cache length (the slot the new
        # token will occupy, independent of padding).
        input_ids = torch.tensor(
            [[req.output_ids[-1]] for req in requests],
            dtype=torch.long,
            device=device,
        )
        position_ids = torch.tensor(
            [[cl] for cl in cache_lens], dtype=torch.long, device=device
        )

        # Pad per-layer KV caches to (batch, num_kv_heads, max_cache_len,
        # head_dim).  Requests shorter than max_cache_len have trailing
        # zeros that the attention mask will hide.
        batched_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_idx in range(num_layers):
            k_padded = torch.zeros(
                (batch_size, num_kv_heads, max_cache_len, head_dim),
                dtype=kv_dtype,
                device=device,
            )
            v_padded = torch.zeros_like(k_padded)
            for i, req in enumerate(requests):
                k_i, v_i = req.kv_cache[layer_idx]
                seq_len_i = cache_lens[i]
                k_padded[i, :, :seq_len_i, :] = k_i[0]
                v_padded[i, :, :seq_len_i, :] = v_i[0]
            batched_kv_caches.append((k_padded, v_padded))

        # Additive attention mask of shape (batch, 1, 1, max_cache_len + 1).
        # The "+ 1" accounts for the new token the attention layer will
        # concatenate onto the cached K/V before computing attention.
        # For request i, positions [cache_lens[i], max_cache_len) are pad
        # (set to -inf).  Position `max_cache_len` is the new token and is
        # always valid.  Positions [0, cache_lens[i]) are real cached tokens.
        total_kv_len = max_cache_len + 1
        attention_mask = torch.zeros(
            (batch_size, 1, 1, total_kv_len), dtype=kv_dtype, device=device
        )
        for i, seq_len_i in enumerate(cache_lens):
            if seq_len_i < max_cache_len:
                attention_mask[i, 0, 0, seq_len_i:max_cache_len] = float("-inf")

        # Single batched forward pass.
        logits, new_kv_caches = self.model(
            input_ids,
            position_ids,
            kv_caches=batched_kv_caches,
            attention_mask=attention_mask,
        )
        # The pre-forward padded input is no longer needed — drop it so the
        # allocator can reuse the storage instead of holding old + new KV
        # buffers concurrently.
        del batched_kv_caches

        # Vectorized KV cache rebuild.
        #
        # The forward left the new token at the trailing slot (index
        # max_cache_len) of every layer's batched K/V.  Request i's real
        # cache ends at position cache_lens[i] — the slots in between are
        # the zero padding we initialized.  We scatter the new token into
        # position cache_lens[i] per row, then hand each request a view
        # [:, :, :cache_lens[i] + 1, :] into the padded tensor.
        #
        # Cost: one indexed-assign per layer (2 × num_layers kernel launches)
        # vs. the old approach of batch × num_layers × 2 torch.cats — at
        # batch=16 that's 72 launches vs 1152.  Each request's kv_cache is
        # now a view into the padded buffer; the padded storage stays alive
        # via those views until the next step replaces them.
        cache_lens_t = torch.tensor(cache_lens, dtype=torch.long, device=device)
        batch_idx_t = torch.arange(batch_size, device=device)
        for k_layer, v_layer in new_kv_caches:
            # Clone before assigning — advanced-index assignment with an
            # aliased source can tear when the destination position equals
            # the source position (request i with cache_lens[i] == max_cache_len).
            k_new = k_layer[:, :, -1, :].clone()  # (batch, kv_heads, head_dim)
            v_new = v_layer[:, :, -1, :].clone()
            k_layer[batch_idx_t, :, cache_lens_t, :] = k_new
            v_layer[batch_idx_t, :, cache_lens_t, :] = v_new

        for i, req in enumerate(requests):
            new_len = cache_lens[i] + 1
            req.kv_cache = [
                (k_layer[i : i + 1, :, :new_len, :], v_layer[i : i + 1, :, :new_len, :])
                for k_layer, v_layer in new_kv_caches
            ]

        # Per-request sampling — sampling params and output history differ
        # between requests, so we cannot easily fuse this step.
        token_ids: list[int] = []
        for i, req in enumerate(requests):
            tid = sample_token(
                logits[i : i + 1, -1, :], req.sampling_params, req.output_ids
            )
            token_ids.append(tid)
        return token_ids

    def is_stop_token(self, token_id: int) -> bool:
        return token_id in self.stop_token_ids
