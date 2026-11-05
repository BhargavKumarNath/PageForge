"""The Rust allocator through its Python bindings (no GPU needed)."""

from __future__ import annotations

import random

import pytest

_pageforge = pytest.importorskip("_pageforge")
PageForge = _pageforge.PageForge


def test_fresh_pool_is_all_free():
    pf = PageForge(32, 16)
    assert (pf.free_pages(), pf.used_pages(), pf.total_pages(), pf.page_size) == (32, 0, 32, 16)
    pf.check_invariants()


def test_alloc_returns_distinct_ids_and_records_them():
    pf = PageForge(32, 16)
    a = pf.alloc_for_seq(0, 5)
    b = pf.alloc_for_seq(1, 3)
    assert len(set(a) | set(b)) == 8
    assert pf.pages_for(0) == a and pf.pages_for(1) == b
    assert pf.num_sequences() == 2


def test_free_returns_pages_and_they_are_reused():
    pf = PageForge(4, 16)
    a = pf.alloc_for_seq(0, 4)
    assert pf.free_seq(0) == a
    assert pf.free_pages() == 4
    assert sorted(pf.alloc_for_seq(1, 4)) == sorted(a)
    pf.check_invariants()


def test_out_of_pages_raises_memoryerror_and_allocates_nothing():
    pf = PageForge(4, 16)
    pf.alloc_for_seq(0, 3)
    with pytest.raises(MemoryError):
        pf.alloc_for_seq(1, 2)
    assert pf.free_pages() == 1
    assert pf.pages_for(1) == []
    pf.check_invariants()


def test_free_unknown_sequence_raises_keyerror():
    pf = PageForge(4, 16)
    with pytest.raises(KeyError):
        pf.free_seq(99)


def test_double_free_of_sequence_raises():
    pf = PageForge(4, 16)
    pf.alloc_for_seq(0, 2)
    pf.free_seq(0)
    with pytest.raises(KeyError):
        pf.free_seq(0)


@pytest.mark.parametrize("n_pages,page_size", [(0, 16), (8, 0)])
def test_invalid_configuration_is_rejected(n_pages, page_size):
    with pytest.raises(ValueError):
        PageForge(n_pages, page_size)


@pytest.mark.parametrize("tokens,pages", [(0, 0), (1, 1), (15, 1), (16, 1), (17, 2), (59, 4), (64, 4), (65, 5)])
def test_pages_needed(tokens, pages):
    assert PageForge(8, 16).pages_needed(tokens) == pages


def test_ensure_capacity_grows_one_page_per_boundary():
    pf = PageForge(16, 16)
    grown = [len(pf.ensure_capacity(0, n)) for n in range(1, 66)]
    # A new page exactly when the token count crosses a 16-token boundary.
    assert [n + 1 for n, g in enumerate(grown) if g] == [1, 17, 33, 49, 65]
    assert len(pf.pages_for(0)) == 5


def test_large_seq_ids_are_supported():
    pf = PageForge(4, 16)
    pf.alloc_for_seq(2**63, 1)
    assert pf.free_seq(2**63) == [0]


def test_randomised_workload_matches_model():
    rng = random.Random(1234)
    pf = PageForge(128, 16)
    model: dict[int, int] = {}
    for _ in range(5000):
        seq = rng.randrange(20)
        if rng.random() < 0.3:
            if seq in model:
                assert len(pf.free_seq(seq)) == model.pop(seq)
            else:
                with pytest.raises(KeyError):
                    pf.free_seq(seq)
        else:
            n = rng.randrange(8)
            if n <= pf.free_pages():
                pf.alloc_for_seq(seq, n)
                model[seq] = model.get(seq, 0) + n
            else:
                with pytest.raises(MemoryError):
                    pf.alloc_for_seq(seq, n)
        assert pf.used_pages() == sum(model.values())
    pf.check_invariants()


def test_stress_cycles_leave_no_leaks():
    pf = PageForge(512, 16)
    for _ in range(200):
        for s in range(16):
            pf.alloc_for_seq(s, 8)
        for s in range(16):
            pf.free_seq(s)
    assert pf.free_pages() == 512 and pf.num_sequences() == 0
    pf.check_invariants()
