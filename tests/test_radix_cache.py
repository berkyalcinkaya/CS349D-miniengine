"""Unit tests for the radix prefix cache (Milestone 3, Part B).

Pure-Python — uses a fake pool so it runs without torch / CUDA.  Exercises
page-granular matching, node splitting, redundant-page dedup, LRU eviction,
lock protection, and the pool's eviction-on-allocate hook.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniengine.radix_cache import RadixCache  # noqa: E402


class FakePool:
    """Free-list-only stand-in for KVMemoryPool (no GPU tensors)."""

    def __init__(self, num_pages: int, page_size: int) -> None:
        self.page_size = page_size
        self._free = list(range(1, num_pages))
        self.radix_cache = None

    def allocate(self, n: int) -> list[int]:
        if len(self._free) < n and self.radix_cache is not None:
            self.radix_cache.evict(n - len(self._free))
        if len(self._free) < n:
            raise RuntimeError(f"pool exhausted: want {n} have {len(self._free)}")
        out = self._free[:n]
        self._free = self._free[n:]
        return out

    def free(self, pages: list[int]) -> None:
        self._free.extend(pages)

    @property
    def num_free(self) -> int:
        return len(self._free)


def make_cache(num_pages=100, page_size=2):
    pool = FakePool(num_pages, page_size)
    cache = RadixCache(pool)
    pool.radix_cache = cache
    return pool, cache


def seq(*ids):
    return list(ids)


def test_empty_lookup_misses():
    _, cache = make_cache()
    r = cache.match_prefix(seq(1, 2, 3, 4))
    assert r.matched_tokens == 0
    assert r.matched_pages == []
    assert r.last_node is None


def test_insert_then_full_match():
    pool, cache = make_cache(page_size=2)
    pages = pool.allocate(2)  # 2 pages = 4 tokens
    leaf, redundant = cache.insert_and_return(seq(1, 2, 3, 4), pages)
    assert redundant == []
    assert cache.num_cached_pages == 2
    r = cache.match_prefix(seq(1, 2, 3, 4, 9, 9))
    assert r.matched_tokens == 4
    assert r.matched_pages == pages
    assert r.last_node is leaf


def test_page_granular_partial_page_not_matched():
    pool, cache = make_cache(page_size=2)
    pages = pool.allocate(2)
    cache.insert_and_return(seq(1, 2, 3, 4), pages)
    # Shares only the first page (tokens 1,2); token 3 matches but 4 differs,
    # so the second page is a partial match and must NOT be returned.
    r = cache.match_prefix(seq(1, 2, 3, 7))
    assert r.matched_tokens == 2
    assert r.matched_pages == pages[:1]


def test_split_on_divergent_branch():
    pool, cache = make_cache(page_size=2)
    a = pool.allocate(3)  # tokens 1,2,3,4,5,6
    cache.insert_and_return(seq(1, 2, 3, 4, 5, 6), a)
    b = pool.allocate(2)  # tokens 1,2,7,8  → shares first page, diverges
    leaf, redundant = cache.insert_and_return(seq(1, 2, 7, 8), b)
    # First page (tokens 1,2) is shared; its incoming page differs → redundant.
    assert redundant == [b[0]]
    # Both branches now reachable.
    r1 = cache.match_prefix(seq(1, 2, 3, 4, 5, 6))
    assert r1.matched_tokens == 6 and r1.matched_pages == a
    r2 = cache.match_prefix(seq(1, 2, 7, 8))
    assert r2.matched_tokens == 4
    assert r2.matched_pages == [a[0], b[1]]


def test_borrowed_prefix_reinsert_not_redundant():
    pool, cache = make_cache(page_size=2)
    a = pool.allocate(2)
    cache.insert_and_return(seq(1, 2, 3, 4), a)
    # A borrowing request reuses the same prefix pages, then extends.
    new = pool.allocate(1)
    leaf, redundant = cache.insert_and_return(seq(1, 2, 3, 4, 5, 6), a + new)
    assert redundant == []  # a[0], a[1] equal the cached pages → kept silently
    assert cache.num_cached_pages == 3
    r = cache.match_prefix(seq(1, 2, 3, 4, 5, 6))
    assert r.matched_tokens == 6 and r.matched_pages == a + new


def test_lru_eviction_frees_oldest_and_returns_pages():
    pool, cache = make_cache(page_size=2)
    p1 = pool.allocate(1)
    cache.insert_and_return(seq(1, 2), p1)
    p2 = pool.allocate(1)
    cache.insert_and_return(seq(3, 4), p2)
    # Touch the first so the second is older.
    cache.match_prefix(seq(1, 2))
    free_before = pool.num_free
    freed = cache.evict(1)
    assert freed == 1
    assert pool.num_free == free_before + 1
    # The evicted (older) branch is gone; the touched one survives.
    assert cache.match_prefix(seq(3, 4)).matched_tokens == 0
    assert cache.match_prefix(seq(1, 2)).matched_tokens == 2


def test_locked_node_not_evicted():
    pool, cache = make_cache(page_size=2)
    p1 = pool.allocate(1)
    leaf, _ = cache.insert_and_return(seq(1, 2), p1)
    cache.inc_lock_ref(leaf)
    freed = cache.evict(10)
    assert freed == 0
    assert cache.match_prefix(seq(1, 2)).matched_tokens == 2
    cache.dec_lock_ref(leaf)
    assert cache.evict(10) == 1


def test_eviction_on_allocate():
    pool, cache = make_cache(num_pages=5, page_size=2)  # 4 usable pages
    # Fill the cache with all four usable pages across two branches.
    cache.insert_and_return(seq(1, 2, 3, 4), pool.allocate(2))
    cache.insert_and_return(seq(5, 6, 7, 8), pool.allocate(2))
    assert pool.num_free == 0
    # Allocation must trigger eviction rather than raise.
    got = pool.allocate(2)
    assert len(got) == 2
    assert cache.metrics.total_evicted_pages == 2


def test_num_evictable_respects_locks():
    pool, cache = make_cache(page_size=2)
    leaf_a, _ = cache.insert_and_return(seq(1, 2), pool.allocate(1))
    cache.insert_and_return(seq(3, 4), pool.allocate(1))
    assert cache.num_evictable_pages() == 2
    cache.inc_lock_ref(leaf_a)
    assert cache.num_evictable_pages() == 1
    cache.dec_lock_ref(leaf_a)
    assert cache.num_evictable_pages() == 2


def test_locked_node_survives_split_and_unlocks_cleanly():
    # Lock a deep node, then insert a sequence that shares only its first
    # page (forcing a split of the locked node's edge).  After dec_lock_ref
    # every node must return to ref_count 0 so all pages become evictable.
    pool, cache = make_cache(page_size=2)
    deep, _ = cache.insert_and_return(seq(1, 2, 3, 4, 5, 6), pool.allocate(3))
    cache.inc_lock_ref(deep)  # locks the [1,2,3,4,5,6] leaf and its ancestors
    # Insert (1,2,9,9): shares page [1,2], diverges → splits the locked edge.
    cache.insert_and_return(seq(1, 2, 9, 9), pool.allocate(2))
    # The originally-locked object is still reachable via match and locked.
    assert cache.num_evictable_pages() < cache.num_cached_pages
    cache.dec_lock_ref(deep)
    # Now nothing is locked → the entire tree is evictable.
    n = cache.num_cached_pages
    assert cache.num_evictable_pages() == n
    assert cache.evict(n) == n
    assert cache.num_cached_pages == 0


def test_reset_returns_all_pages():
    pool, cache = make_cache(page_size=2)
    cache.insert_and_return(seq(1, 2, 3, 4), pool.allocate(2))
    cache.insert_and_return(seq(5, 6), pool.allocate(1))
    free_before = pool.num_free
    cache.reset()
    assert pool.num_free == free_before + 3
    assert cache.num_cached_pages == 0
    assert cache.match_prefix(seq(1, 2)).matched_tokens == 0


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\nAll {len(fns)} radix-cache tests passed.")


if __name__ == "__main__":
    _run_all()
