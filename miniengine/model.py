"""
Bare-bone Qwen3 transformer in pure PyTorch.

No HuggingFace model classes — just nn.Module, nn.Linear, and manual
attention with KV cache.  Weight names match the HuggingFace checkpoint
so we can load safetensors directly via load_state_dict().

Architecture (Qwen3-4B as reference):
    Embedding(151936, 2560)
    36 x TransformerBlock:
        RMSNorm → Attention(GQA + QK-Norm + RoPE) → RMSNorm → SwiGLU MLP
    RMSNorm
    LM Head (tied with embedding)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ── Flash-attn (optional; falls back to SDPA gather if unavailable) ─────

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # type: ignore

    _HAS_FLASH_ATTN = True
except Exception:  # noqa: BLE001 — broad: import may fail in many ways
    flash_attn_varlen_func = None  # type: ignore[assignment]
    flash_attn_with_kvcache = None  # type: ignore[assignment]
    _HAS_FLASH_ATTN = False


# ── Paged attention metadata ────────────────────────────────────────────


@dataclass
class PagedMeta:
    """Per-step metadata describing how to read/write the paged KV pool.

    The engine builds one of these per forward; every layer reads from it.
    Both the SDPA-gather and flash-attn paths consume the same struct —
    Python-list fields drive the SDPA per-request loop, tensor fields
    drive the flash-attn calls.

    Attributes:
        is_prefill:    True for packed-prefill forwards, False for decode.
        page_size:     Tokens per page in the pool (mirror of pool.page_size).
        slot_mapping:  (total_tokens,) int64 — global slot in the flat
                       (num_pages * page_size) view where each input token's
                       K/V should be written. Order matches input order.
        block_table:   (B, max_blocks) int32 — pool page indices per request,
                       padded with 0 past the request's num_blocks.
                       (int32 because flash-attn requires it; SDPA path
                       casts to long for fancy indexing.)
        cu_seqlens:    (B+1,) Python list — cumulative token offsets in the
                       packed Q for prefill.  For decode, this is range(B+1).
        cache_seqlens: (B,) Python list — full KV length per request *after*
                       the new tokens have been scattered in this step.
        cu_seqlens_tensor:    (B+1,) int32 on device — flash-attn varlen.
        cache_seqlens_tensor: (B,)   int32 on device — flash-attn with_kvcache.
        max_seqlen:    Longest per-request seq length this step (Python int);
                       required by flash_attn_varlen_func.
    """

    is_prefill: bool
    page_size: int
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    cu_seqlens: list[int]
    cache_seqlens: list[int]
    cu_seqlens_tensor: torch.Tensor
    cache_seqlens_tensor: torch.Tensor
    max_seqlen: int


# ── Config ──────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    """Model architecture config, loaded from HuggingFace config.json."""

    vocab_size: int = 151936
    hidden_size: int = 2560
    intermediate_size: int = 9728
    num_hidden_layers: int = 36
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128  # explicit, NOT hidden_size // num_heads
    rms_norm_eps: float = 1e-6
    rope_theta: float = 5_000_000.0
    max_position_embeddings: int = 262144
    tie_word_embeddings: bool = True

    @classmethod
    def from_pretrained(cls, model_path: str) -> ModelConfig:
        from transformers import AutoConfig

        hf = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return cls(
            vocab_size=hf.vocab_size,
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_hidden_layers=hf.num_hidden_layers,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
            rms_norm_eps=hf.rms_norm_eps,
            rope_theta=getattr(hf, "rope_theta", 10000.0),
            max_position_embeddings=getattr(hf, "max_position_embeddings", 4096),
            tie_word_embeddings=getattr(hf, "tie_word_embeddings", False),
        )


# ── Building blocks ─────────────────────────────────────────────────────


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Variance in fp32 — bf16 mean-of-squares loses too much precision.
        input_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Precomputes and caches cos/sin tables, indexed by position_ids at
    forward time.  The cache grows on-demand so we never allocate for the
    full 256K context upfront.
    """

    def __init__(self, head_dim: int, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None
        self._cached_len: int = 0

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            position_ids: (batch, seq_len) integer positions.

        Returns:
            cos, sin each of shape (batch, 1, seq_len, head_dim) —
            broadcastable over the head dimension.
        """
        max_pos = int(position_ids.max().item()) + 1

        if self._cos is None or max_pos > self._cached_len:
            length = max(max_pos, self._cached_len * 2, 256)
            t = torch.arange(
                length, device=self.inv_freq.device, dtype=self.inv_freq.dtype
            )
            freqs = torch.outer(t, self.inv_freq)  # (length, head_dim/2)
            emb = torch.cat([freqs, freqs], dim=-1)  # (length, head_dim)
            self._cos = emb.cos()
            self._sin = emb.sin()
            self._cached_len = length

        # Index into cache: (batch, seq_len, head_dim) → add head dim
        cos = self._cos[position_ids].unsqueeze(2)  # (batch, seq_len, 1, head_dim)
        sin = self._sin[position_ids].unsqueeze(2)
        return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """
    Apply RoPE to x.

    x:   (batch, num_heads, seq_len, head_dim)
    cos: (batch, seq_len, 1, head_dim)  — broadcast over heads
    sin: same shape as cos
    """
    # Cast cos/sin to x.dtype — fp32 cos/sin would silently promote q/k.
    cos = cos.transpose(1, 2).to(x.dtype)
    sin = sin.transpose(1, 2).to(x.dtype)
    return x * cos + _rotate_half(x) * sin


# ── Attention ───────────────────────────────────────────────────────────


class Attention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA), QK-Norm,
    and Rotary Position Embeddings.

    Q projects to  num_attention_heads  × head_dim
    K projects to  num_key_value_heads  × head_dim
    V projects to  num_key_value_heads  × head_dim
    O projects back to hidden_size
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            config.hidden_size, self.num_kv_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, config.hidden_size, bias=False
        )

        # Qwen3: RMSNorm on Q and K after projection (per-head)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            hidden:         (batch, seq_len, hidden_size)
            cos, sin:       from RotaryEmbedding, broadcastable
            kv_cache:       optional (cached_k, cached_v), each
                            (batch, num_kv_heads, cache_len, head_dim)
            attention_mask: optional float mask (batch, 1, q_len, kv_len)
                            for batched decode with padded KV; 0 = attend,
                            -inf = ignore.

        Returns:
            output:       (batch, seq_len, hidden_size)
            new_kv_cache: (k, v) with updated cache
        """
        bsz, seq_len, _ = hidden.shape

        # Project Q, K, V and reshape to (batch, heads, seq_len, head_dim)
        q = (
            self.q_proj(hidden)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Append to KV cache
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv = (k, v)

        # GQA: expand KV heads to match Q heads
        if self.num_kv_groups > 1:
            k = k[:, :, None, :, :].expand(-1, -1, self.num_kv_groups, -1, -1)
            k = k.reshape(bsz, self.num_heads, -1, self.head_dim)
            v = v[:, :, None, :, :].expand(-1, -1, self.num_kv_groups, -1, -1)
            v = v.reshape(bsz, self.num_heads, -1, self.head_dim)

        # Batched decode passes an explicit float mask; otherwise fall
        # back to the is_causal kernel path.
        if attention_mask is not None:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        else:
            is_causal = kv_cache is None and seq_len > 1
            out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

        # Merge heads → project back
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(out), new_kv

    # ── Paged attention (milestone 2) ───────────────────────────────────

    def forward_paged(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        paged_meta: PagedMeta,
    ) -> torch.Tensor:
        """Paged-KV attention.

        Projects Q/K/V, applies QK-Norm + RoPE, scatters new K/V into the
        pool via `paged_meta.slot_mapping`, then dispatches to one of two
        attention backends:

          - flash-attn (`flash_attn_varlen_func` for prefill,
            `flash_attn_with_kvcache` for decode) when available + CUDA +
            half precision.
          - SDPA gather (per-request loop) as a fallback for CPU / fp32 /
            no-flash environments.

        Args:
            hidden:    (1, total_tokens, hidden) for prefill,
                       (B, 1, hidden) for decode.
            cos, sin:  RoPE tables matching `hidden`'s layout.
            k_cache,
            v_cache:   Pool tensors, shape (num_pages, page_size,
                       num_kv_heads, head_dim). Mutated in place.
            paged_meta: see PagedMeta.

        Returns:
            (bsz, seq_len, hidden_size) attention output.
        """
        bsz, seq_len, _ = hidden.shape

        # Project Q, K, V → (bsz, num_heads, seq_len, head_dim)
        q = (
            self.q_proj(hidden)
            .view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden)
            .view(bsz, seq_len, self.num_kv_heads, self.head_dim)
            .transpose(1, 2)
        )

        # QK-Norm + RoPE
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Scatter freshly computed K/V into the pool. flash-attn's decode
        # path could write the new K/V itself, but we always scatter here
        # so prefill and decode share one code path and so the pool
        # always reflects the post-write state.
        # k, v: (bsz, num_kv_heads, seq_len, head_dim)
        # → (bsz * seq_len, num_kv_heads, head_dim) token-major.
        k_flat_in = k.transpose(1, 2).reshape(
            bsz * seq_len, self.num_kv_heads, self.head_dim
        )
        v_flat_in = v.transpose(1, 2).reshape(
            bsz * seq_len, self.num_kv_heads, self.head_dim
        )
        k_cache_flat = k_cache.view(-1, self.num_kv_heads, self.head_dim)
        v_cache_flat = v_cache.view(-1, self.num_kv_heads, self.head_dim)
        k_cache_flat[paged_meta.slot_mapping] = k_flat_in
        v_cache_flat[paged_meta.slot_mapping] = v_flat_in

        # Dispatch.
        use_flash = (
            _HAS_FLASH_ATTN
            and hidden.is_cuda
            and hidden.dtype in (torch.float16, torch.bfloat16)
        )
        if use_flash:
            if paged_meta.is_prefill:
                out = self._attn_flash_prefill(q, k, v, paged_meta)
            else:
                out = self._attn_flash_decode(q, k_cache, v_cache, paged_meta)
        else:
            out = self._attn_sdpa_gather(q, k_cache, v_cache, paged_meta)

        # `out` shape: (bsz, seq_len, hidden_size) — ready for o_proj.
        return self.o_proj(out)

    # ── Flash-attn paths ────────────────────────────────────────────────

    def _attn_flash_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        paged_meta: PagedMeta,
    ) -> torch.Tensor:
        """Packed varlen attention on the freshly computed Q/K/V.

        Reads K/V directly from the just-projected tensors rather than
        gathering through the pool — saves a redundant gather since the
        prefill writes and reads the same K/V.
        """
        # q: (1, num_heads, total, head_dim) → (total, num_heads, head_dim)
        q_flash = q.transpose(1, 2).squeeze(0).contiguous()
        k_flash = k.transpose(1, 2).squeeze(0).contiguous()
        v_flash = v.transpose(1, 2).squeeze(0).contiguous()

        out = flash_attn_varlen_func(  # type: ignore[misc]
            q_flash,
            k_flash,
            v_flash,
            cu_seqlens_q=paged_meta.cu_seqlens_tensor,
            cu_seqlens_k=paged_meta.cu_seqlens_tensor,
            max_seqlen_q=paged_meta.max_seqlen,
            max_seqlen_k=paged_meta.max_seqlen,
            causal=True,
        )
        # out: (total, num_heads, head_dim) → (1, total, hidden_size)
        return out.reshape(1, -1, self.num_heads * self.head_dim)

    def _attn_flash_decode(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        paged_meta: PagedMeta,
    ) -> torch.Tensor:
        """Paged-KV attention for decode (seqlen_q = 1).

        The new K/V have already been scattered into the pool, so we
        pass `k=None, v=None` and rely on `block_table` + `cache_seqlens`
        to read the full per-request KV.
        """
        # q: (B, num_heads, 1, head_dim) → (B, 1, num_heads, head_dim)
        q_flash = q.transpose(1, 2).contiguous()
        bsz = q_flash.shape[0]

        out = flash_attn_with_kvcache(  # type: ignore[misc]
            q_flash,
            k_cache,
            v_cache,
            k=None,
            v=None,
            cache_seqlens=paged_meta.cache_seqlens_tensor,
            block_table=paged_meta.block_table,
            causal=False,
        )
        # out: (B, 1, num_heads, head_dim) → (B, 1, hidden_size)
        return out.reshape(bsz, 1, self.num_heads * self.head_dim)

    # ── SDPA gather path (fallback / reference) ─────────────────────────

    def _attn_sdpa_gather(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        paged_meta: PagedMeta,
    ) -> torch.Tensor:
        """Per-request gather + SDPA. Correct but slow."""
        bsz, _, seq_len, _ = q.shape
        page_size = paged_meta.page_size
        outputs: list[torch.Tensor] = []

        if paged_meta.is_prefill:
            cu = paged_meta.cu_seqlens
            for i, cache_len in enumerate(paged_meta.cache_seqlens):
                qs, qe = cu[i], cu[i + 1]
                q_i = q[:, :, qs:qe, :]
                k_full, v_full = self._gather_paged_kv(
                    k_cache, v_cache, paged_meta.block_table[i], cache_len, page_size
                )
                out_i = F.scaled_dot_product_attention(
                    q_i, k_full, v_full, is_causal=True
                )
                outputs.append(out_i)
            out = torch.cat(outputs, dim=2)
        else:
            for i, cache_len in enumerate(paged_meta.cache_seqlens):
                q_i = q[i : i + 1]
                k_full, v_full = self._gather_paged_kv(
                    k_cache, v_cache, paged_meta.block_table[i], cache_len, page_size
                )
                out_i = F.scaled_dot_product_attention(
                    q_i, k_full, v_full, is_causal=False
                )
                outputs.append(out_i)
            out = torch.cat(outputs, dim=0)

        # (bsz, num_heads, seq_len, head_dim) → (bsz, seq_len, hidden_size)
        return out.transpose(1, 2).contiguous().view(bsz, seq_len, -1)

    def _gather_paged_kv(
        self,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_ids_row: torch.Tensor,
        cache_len: int,
        page_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather a request's full K/V from the pool and shape for SDPA.

        Returns K, V each shaped (1, num_heads, cache_len, head_dim) with
        GQA expansion already applied so SDPA can be called directly.
        """
        num_blocks = (cache_len + page_size - 1) // page_size
        # block_table is int32 (flash-attn convention); fancy indexing
        # into k_cache needs int64.
        block_ids = block_ids_row[:num_blocks].long()
        k_pages = k_cache[block_ids]  # (num_blocks, page_size, num_kv_heads, head_dim)
        v_pages = v_cache[block_ids]
        k_full = k_pages.reshape(
            num_blocks * page_size, self.num_kv_heads, self.head_dim
        )[:cache_len]
        v_full = v_pages.reshape(
            num_blocks * page_size, self.num_kv_heads, self.head_dim
        )[:cache_len]
        # → (1, num_kv_heads, cache_len, head_dim)
        k_full = k_full.transpose(0, 1).unsqueeze(0)
        v_full = v_full.transpose(0, 1).unsqueeze(0)
        if self.num_kv_groups > 1:
            k_full = (
                k_full[:, :, None, :, :]
                .expand(-1, -1, self.num_kv_groups, -1, -1)
                .reshape(1, self.num_heads, -1, self.head_dim)
            )
            v_full = (
                v_full[:, :, None, :, :]
                .expand(-1, -1, self.num_kv_groups, -1, -1)
                .reshape(1, self.num_heads, -1, self.head_dim)
            )
        return k_full, v_full


# ── MLP ─────────────────────────────────────────────────────────────────


class MLP(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ── Transformer block ──────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    """Pre-norm transformer layer: LN → Attn → residual → LN → MLP → residual."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden, new_kv = self.self_attn(hidden, cos, sin, kv_cache, attention_mask)
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        hidden = residual + hidden

        return hidden, new_kv

    def forward_paged(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        paged_meta: PagedMeta,
    ) -> torch.Tensor:
        """Paged variant — KV is mutated in-place inside the pool."""
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn.forward_paged(
            hidden, cos, sin, k_cache, v_cache, paged_meta
        )
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        hidden = residual + hidden
        return hidden


# ── Full model ──────────────────────────────────────────────────────────


class TransformerModel(nn.Module):
    """The core transformer: embedding → N layers → final norm."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config.head_dim, theta=config.rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids:      (batch, seq_len)
            position_ids:   (batch, seq_len)
            kv_caches:      list of per-layer (key, value) caches, or None
            attention_mask: optional float mask for batched-decode SDPA

        Returns:
            hidden:         (batch, seq_len, hidden_size)
            new_kv_caches:  list of per-layer (key, value) with appended tokens
        """
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids)

        new_kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            kv = kv_caches[i] if kv_caches is not None else None
            hidden, new_kv = layer(hidden, cos, sin, kv, attention_mask)
            new_kv_caches.append(new_kv)

        hidden = self.norm(hidden)
        return hidden, new_kv_caches

    def forward_paged(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        paged_meta: PagedMeta,
    ) -> torch.Tensor:
        """Paged variant — kv_caches are pool tensors, mutated in place."""
        hidden = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(position_ids)

        for i, layer in enumerate(self.layers):
            k_cache, v_cache = kv_caches[i]
            hidden = layer.forward_paged(
                hidden, cos, sin, k_cache, v_cache, paged_meta
            )

        hidden = self.norm(hidden)
        return hidden


class CausalLM(nn.Module):
    """
    Complete causal language model: transformer + LM head.

    The LM head may be tied with the embedding (Qwen3-4B) or separate.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = TransformerModel(config)
        if not config.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Returns:
            logits:        (batch, seq_len, vocab_size)
            new_kv_caches: per-layer KV caches
        """
        hidden, new_kv_caches = self.model(
            input_ids, position_ids, kv_caches, attention_mask
        )
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden)
        return logits, new_kv_caches

    def forward_paged(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
        paged_meta: PagedMeta,
    ) -> torch.Tensor:
        """Paged variant — returns logits only (KV mutated in pool)."""
        hidden = self.model.forward_paged(
            input_ids, position_ids, kv_caches, paged_meta
        )
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden, self.model.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden)
        return logits


# ── Weight loading ──────────────────────────────────────────────────────


def load_weights(
    model: CausalLM,
    model_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> None:
    """
    Load weights from HuggingFace safetensors into the model.

    Handles both single-file and sharded checkpoints.  Weight names in the
    checkpoint match our module hierarchy exactly (by design), so we can
    use load_state_dict() directly.
    """
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    logger.info("Downloading / locating model files for %s …", model_path)
    local_path = Path(
        snapshot_download(
            model_path,
            allow_patterns=["*.safetensors", "*.json"],
        )
    )

    # Gather all safetensor shard files
    st_files = sorted(local_path.glob("model*.safetensors"))
    if not st_files:
        # Fallback: some repos use a single "model.safetensors"
        st_files = sorted(local_path.glob("*.safetensors"))
    if not st_files:
        raise FileNotFoundError(f"No safetensors files in {local_path}")

    logger.info("Loading %d safetensors shard(s) …", len(st_files))

    # Load shards straight to the target device — avoids a full CPU copy.
    state_dict: dict[str, torch.Tensor] = {}
    for f in st_files:
        for key, tensor in load_file(str(f), device=device).items():
            state_dict[key] = tensor.to(dtype=dtype)

    # Drop checkpoint keys the model doesn't expect.
    model_keys = set(model.state_dict().keys())
    extra = set(state_dict.keys()) - model_keys
    for key in extra:
        del state_dict[key]
    if extra:
        logger.info("Skipped %d unexpected checkpoint keys", len(extra))

    if "lm_head.weight" in model_keys and "lm_head.weight" not in state_dict:
        logger.info("Tying lm_head.weight to embed_tokens.weight")
        state_dict["lm_head.weight"] = state_dict["model.embed_tokens.weight"]

    # assign=True: replace meta tensors in-place rather than copy_ into them.
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    del state_dict
    if missing:
        logger.warning("Missing keys after load: %s", missing)
    if unexpected:
        logger.warning("Unexpected keys after load: %s", unexpected)

    # RoPE inv_freq is a non-persistent buffer (not in checkpoint), so it's
    # still on the meta device after assign=True — materialize it now.
    for module in model.modules():
        if isinstance(module, RotaryEmbedding):
            module.inv_freq = 1.0 / (
                module.theta
                ** (
                    torch.arange(
                        0, module.head_dim, 2, device=device, dtype=torch.float32
                    )
                    / module.head_dim
                )
            )
    logger.info(
        "Weights loaded — %d parameters on %s (%s)",
        sum(p.numel() for p in model.parameters()),
        device,
        dtype,
    )
