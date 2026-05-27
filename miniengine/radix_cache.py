"""Radix-tree prefix cache — Milestone 3, Part B.

Stores already-computed KV pages keyed by token prefix so a new request whose
prompt starts with a cached prefix can reuse those pages instead of
recomputing them.

The data structure is a radix tree whose nodes own KV pages from the
``KVMemoryPool``.  Pages held by the cache are *not* in the pool's free list;
they return there only when the cache evicts them (LRU) or when an in-flight
insert chooses to free a redundant duplicate.

Everything here runs in the single scheduler thread (prefill → decode →
finish), so the tree needs no internal locking — the async HTTP layer only
enqueues requests and reads ``metrics``.

Performance counters in ``CacheMetrics`` are read by the ``/cache_stats``
endpoint and by the scheduler's per-prefill-batch INFO log line.
"""

from __future__ import annotations

import heapq
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miniengine.kv_memory_pool import KVMemoryPool

logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """Aggregate cache statistics — surfaced via ``/cache_stats``."""

    total_lookups: int = 0
    total_query_tokens: int = 0
    total_hit_tokens: int = 0
    total_inserted_pages: int = 0
    total_evicted_pages: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_query_tokens == 0:
            return 0.0
        return self.total_hit_tokens / self.total_query_tokens


class RadixNode:
    """A radix-tree node.

    * ``parent`` / ``children`` form the tree; ``children`` is keyed by the
      first token id of each child's edge.
    * ``key`` carries the tokens on the edge from the parent (always a
      multiple of ``page_size`` long).
    * ``pages`` are the KV pages for those tokens
      (``len(pages) == len(key) // page_size``).
    * ``ref_count`` = number of locked leaves in this subtree, so eviction
      can skip a whole locked subtree with one field check.  Manipulated by
      ``inc_lock_ref`` / ``dec_lock_ref``.
    * ``last_access`` drives LRU.
    """

    __slots__ = ("parent", "children", "key", "pages", "ref_count", "last_access")

    def __init__(self) -> None:
        self.parent: RadixNode | None = None
        self.children: dict[int, RadixNode] = {}
        self.key: list[int] = []
        self.pages: list[int] = []
        self.ref_count: int = 0
        self.last_access: float = time.monotonic()


@dataclass
class MatchResult:
    """Result of a prefix lookup.

    ``matched_tokens`` is page-aligned (multiple of ``page_size``);
    ``matched_pages`` carries the KV pages for those tokens.
    ``last_node`` is the deepest node the walk reached — callers lock it
    (``inc_lock_ref``) for the lifetime of the borrowing request.
    """

    matched_pages: list[int] = field(default_factory=list)
    matched_tokens: int = 0
    last_node: "RadixNode | None" = None


class RadixCache:
    """Token-prefix → KV-pages cache backed by a radix tree.

      * page-aligned matching — never returns a partial-page result,
      * LRU eviction of unlocked subtrees,
      * eviction-on-allocate (``KVMemoryPool.allocate`` calls ``evict``),
      * ``inc_lock_ref`` / ``dec_lock_ref`` protect in-flight requests
        (same names as sglang's radix cache).
    """

    def __init__(self, pool: "KVMemoryPool") -> None:
        self.pool = pool
        self.page_size = pool.page_size
        self.root = RadixNode()
        self.metrics = CacheMetrics()
        self._cached_pages = 0  # running total of pages held by the tree

    @property
    def num_cached_pages(self) -> int:
        """Total pages currently held by the tree."""
        return self._cached_pages

    def num_evictable_pages(self) -> int:
        """Pages that an LRU sweep could free right now.

        A node with ``ref_count == 0`` has no locked descendants, so its
        edge pages are evictable.  Summing edge pages over all such nodes
        counts each page exactly once (a node's pages live only on its own
        edge).
        """
        total = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.ref_count == 0:
                total += len(node.pages)
            stack.extend(node.children.values())
        return total

    # ── Lookup ─────────────────────────────────────────────────────────

    def match_prefix(self, tokens: list[int]) -> MatchResult:
        """Find the longest page-aligned prefix of ``tokens`` in the tree."""
        ps = self.page_size
        self.metrics.total_lookups += 1
        self.metrics.total_query_tokens += len(tokens)

        matched_pages: list[int] = []
        node = self.root
        idx = 0  # token index into ``tokens`` consumed so far
        n = len(tokens)
        now = time.monotonic()

        while idx < n:
            if n - idx < ps:
                break  # not a full page left → page-granular matching stops
            # Children are keyed by their first *page* (not first token), so a
            # found child is guaranteed to share that whole page.
            child = node.children.get(tuple(tokens[idx : idx + ps]))
            if child is None:
                break
            # Compare the edge against the query, page by page.
            shared = _shared_pages(child.key, tokens, idx, ps)
            if shared == 0:
                break
            matched_pages.extend(child.pages[:shared])
            idx += shared * ps
            child.last_access = now  # a lookup hit counts as a use (LRU)
            if shared < len(child.pages):
                # Diverged inside this edge — stop at the page boundary.
                node = child
                break
            node = child  # consumed the whole edge; descend further

        matched_tokens = len(matched_pages) * ps
        self.metrics.total_hit_tokens += matched_tokens
        # ``node`` is root only when nothing matched → report no node.
        last = node if node is not self.root else None
        return MatchResult(
            matched_pages=matched_pages,
            matched_tokens=matched_tokens,
            last_node=last,
        )

    # ── Lock ref counting (sglang-style) ───────────────────────────────

    def inc_lock_ref(self, node: "RadixNode | None") -> None:
        """Lock ``node`` (and the path to root) against eviction."""
        now = time.monotonic()
        while node is not None:
            node.ref_count += 1
            node.last_access = now
            node = node.parent

    def dec_lock_ref(self, node: "RadixNode | None") -> None:
        """Release a lock.  Refresh ``last_access`` while walking."""
        now = time.monotonic()
        while node is not None:
            assert node.ref_count > 0, "dec_lock_ref under-flow"
            node.ref_count -= 1
            node.last_access = now
            node = node.parent

    # ── Insertion ──────────────────────────────────────────────────────

    def insert_and_return(
        self, tokens: list[int], pages: list[int]
    ) -> tuple["RadixNode", list[int]]:
        """Insert ``(tokens, pages)`` (page-aligned) into the tree.

        Returns ``(leaf_node, redundant_pages)``.  ``redundant_pages`` are
        pages the caller handed in that duplicate pages already cached at
        the same prefix (a *different* page index covering identical
        tokens) — the caller returns them to the pool.  Pages whose index
        already matches the cached one (a borrowed-prefix re-insert) are
        kept silently.
        """
        ps = self.page_size
        assert len(tokens) == len(pages) * ps, "insert keys must be page-aligned"

        node = self.root
        redundant: list[int] = []
        idx = 0          # token index
        pidx = 0         # page index
        now = time.monotonic()
        n = len(tokens)

        while idx < n:
            node.last_access = now
            # Insert keys are page-aligned, so a full page is always available.
            ckey = tuple(tokens[idx : idx + ps])
            child = node.children.get(ckey)
            if child is None:
                # No matching edge — attach the rest as a fresh child.
                new = RadixNode()
                new.parent = node
                new.key = tokens[idx:]
                new.pages = pages[pidx:]
                node.children[ckey] = new
                self._cached_pages += len(new.pages)
                self.metrics.total_inserted_pages += len(new.pages)
                new.last_access = now
                return new, redundant

            shared = _shared_pages(child.key, tokens, idx, ps)
            if shared < len(child.pages):
                # Partial match → split the child at the page boundary so the
                # shared head becomes a node we can descend into / branch off.
                child = self._split_node(child, shared)
            # The incoming pages for this shared span duplicate cached ones.
            for k in range(shared):
                if pages[pidx + k] != child.pages[k]:
                    redundant.append(pages[pidx + k])
            idx += shared * ps
            pidx += shared
            node = child

        node.last_access = now
        return node, redundant

    def _split_node(self, node: "RadixNode", n_pages: int) -> "RadixNode":
        """Split ``node``'s edge after ``n_pages`` pages.

        A fresh *head* node takes the first ``n_pages`` (parent side) and is
        spliced in under ``node``'s old parent; ``node`` keeps its identity as
        the *tail* (child side) with the remaining pages and all its children.
        Keeping ``node``'s identity matters: a previously locked deep node
        stays the same object, so a later ``dec_lock_ref`` still walks through
        it.  Both head and tail carry the original ``ref_count`` (the head is
        an ancestor of the tail, so it counts the same locked leaves).
        """
        ps = self.page_size
        n_tokens = n_pages * ps
        parent = node.parent
        assert parent is not None

        head = RadixNode()
        head.parent = parent
        head.key = node.key[:n_tokens]
        head.pages = node.pages[:n_pages]
        head.ref_count = node.ref_count
        head.last_access = node.last_access
        # Children keyed by first page.  ``head`` shares ``node``'s first page,
        # so this overwrites ``node``'s old slot under ``parent``.
        head.children = {tuple(node.key[n_tokens : n_tokens + ps]): node}
        parent.children[tuple(head.key[:ps])] = head

        node.parent = head
        node.key = node.key[n_tokens:]
        node.pages = node.pages[n_pages:]
        return head

    # ── Eviction ───────────────────────────────────────────────────────

    def evict(self, n_pages_needed: int) -> int:
        """LRU-evict at least ``n_pages_needed`` pages (best effort).

        Walks leaves oldest-first; a locked node (``ref_count > 0``) is never
        touched, and a node with locked descendants is never a leaf.  Frees
        pages back to the pool, deletes the node, and re-leafs its parent.
        Returns the number of pages actually freed.
        """
        if n_pages_needed <= 0:
            return 0

        # Min-heap of evictable leaves keyed by (last_access, id).
        heap: list[tuple[float, int, RadixNode]] = []
        for leaf in self._evictable_leaves():
            heapq.heappush(heap, (leaf.last_access, id(leaf), leaf))

        freed = 0
        while heap and freed < n_pages_needed:
            _, _, leaf = heapq.heappop(heap)
            if leaf.children or leaf.ref_count > 0 or leaf is self.root:
                continue  # stale heap entry (re-leafed/locked since pushed)
            parent = leaf.parent
            assert parent is not None
            self.pool.free(leaf.pages)
            freed += len(leaf.pages)
            self._cached_pages -= len(leaf.pages)
            self.metrics.total_evicted_pages += len(leaf.pages)
            del parent.children[tuple(leaf.key[: self.page_size])]
            leaf.parent = None
            # Parent may now be an evictable leaf — reconsider it.
            if parent is not self.root and not parent.children and parent.ref_count == 0:
                heapq.heappush(heap, (parent.last_access, id(parent), parent))

        return freed

    def _evictable_leaves(self) -> list["RadixNode"]:
        leaves: list[RadixNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node is not self.root and not node.children and node.ref_count == 0:
                leaves.append(node)
            stack.extend(node.children.values())
        return leaves

    # ── Maintenance ────────────────────────────────────────────────────

    def reset(self) -> None:
        """Drop the whole tree, return every page to the pool."""
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.pages:
                self.pool.free(node.pages)
            stack.extend(node.children.values())
        self._cached_pages = 0
        self.root = RadixNode()


def _shared_pages(key: list[int], tokens: list[int], offset: int, ps: int) -> int:
    """Number of leading whole pages of ``key`` that equal ``tokens[offset:]``.

    Both are compared in ``ps``-token units; a page counts only if every one
    of its ``ps`` tokens matches (page-granular sharing).
    """
    max_pages = min(len(key) // ps, (len(tokens) - offset) // ps)
    shared = 0
    for p in range(max_pages):
        a = p * ps
        if key[a : a + ps] == tokens[offset + a : offset + a + ps]:
            shared += 1
        else:
            break
    return shared
