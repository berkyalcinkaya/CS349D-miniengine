"""Pre-allocated paged KV cache memory pool — Milestone 2, Part A.

The pool owns a fixed amount of GPU memory, divided into equal-size
**pages**. Each page holds the KV state for `page_size` tokens for one
layer. Requests acquire pages as their KV grows and return them when
they finish; the cache itself never reallocates.

Storage layout
--------------
Per layer we hold two tensors with shape

    (num_pages, page_size, num_kv_heads, head_dim)

— one for K, one for V. This is the layout flash-attn's paged-KV path
expects (`flash_attn_with_kvcache(..., block_table=...)`), so the hot
path can index pages directly without any transpose.

Free list
---------
A `collections.deque[int]` of free page indices. `allocate` pops from
the left, `free` extends on the right — both O(1).

Page tables
-----------
Each request stores its own `list[int]` of page indices on the
`Request` object. The pool itself is pageᐧ↔ᐧrequest agnostic: it only
hands out and reclaims indices.
"""

from __future__ import annotations

from collections import deque

import torch


class KVMemoryPool:
    """Pre-allocated paged KV cache pool.

    Args:
        num_pages:    Total pages in the pool (capacity).
        page_size:    Tokens per page. Tunable knob — exposed as
                      `--page-size` on the CLI. Smaller = less
                      fragmentation, bigger page tables; larger = the
                      opposite.
        num_layers:   Number of transformer layers.
        num_kv_heads: KV heads per layer (GQA).
        head_dim:     Per-head dimension.
        dtype:        KV dtype (typically bfloat16).
        device:       e.g. "cuda".
    """

    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        if num_pages <= 0:
            raise ValueError(f"num_pages must be positive, got {num_pages}")
        if page_size <= 0:
            raise ValueError(f"page_size must be positive, got {page_size}")

        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        shape = (num_pages, page_size, num_kv_heads, head_dim)
        self._kv_caches: list[tuple[torch.Tensor, torch.Tensor]] = [
            (
                torch.empty(shape, dtype=dtype, device=device),
                torch.empty(shape, dtype=dtype, device=device),
            )
            for _ in range(num_layers)
        ]

        self._free: deque[int] = deque(range(num_pages))

    def allocate(self, num_pages: int) -> list[int]:
        """Reserve `num_pages` pages and return their indices.

        Raises if the pool cannot satisfy the request.
        """
        if num_pages < 0:
            raise ValueError(f"num_pages must be non-negative, got {num_pages}")
        if num_pages > len(self._free):
            raise RuntimeError(
                f"KV pool exhausted: requested {num_pages} pages, "
                f"only {len(self._free)} free"
            )
        return [self._free.popleft() for _ in range(num_pages)]

    def free(self, page_indices: list[int]) -> None:
        """Return the listed pages to the free pool."""
        self._free.extend(page_indices)

    def pages_needed(self, seq_len: int) -> int:
        """How many pages are required to store `seq_len` tokens."""
        if seq_len <= 0:
            return 0
        return (seq_len + self.page_size - 1) // self.page_size

    @property
    def num_free(self) -> int:
        """Pages currently available for allocation."""
        return len(self._free)

    @property
    def kv_caches(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Per-layer (K, V) cache tensors.

        The attention path holds references to these and indexes into
        them via per-request page tables. Stable identity: the pool
        never reallocates or replaces these tensors after construction.
        """
        return self._kv_caches

    @classmethod
    def from_budget(
        cls,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        dtype: torch.dtype,
        device: str,
        bytes_budget: int,
    ) -> KVMemoryPool:
        """Convenience: derive `num_pages` from a memory budget.

        Each page costs `2 * num_layers * page_size * num_kv_heads *
        head_dim * dtype_bytes` bytes (factor of 2 for K and V).
        """
        if bytes_budget <= 0:
            raise ValueError(f"bytes_budget must be positive, got {bytes_budget}")

        dtype_bytes = torch.empty((), dtype=dtype).element_size()
        bytes_per_page = (
            2 * num_layers * page_size * num_kv_heads * head_dim * dtype_bytes
        )
        num_pages = bytes_budget // bytes_per_page
        if num_pages <= 0:
            raise RuntimeError(
                f"bytes_budget={bytes_budget} too small for one page "
                f"(bytes_per_page={bytes_per_page})"
            )

        return cls(
            num_pages=int(num_pages),
            page_size=page_size,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype,
            device=device,
        )
