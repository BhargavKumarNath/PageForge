"""Gather/scatter kernels: bit-exact round trips, page placement, isolation."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

P, LAYERS, HEADS, D = 16, 12, 12, 64
C = LAYERS * HEADS


@pytest.fixture
def pool():
    torch.manual_seed(0)
    return torch.randn(32, P, C, D, dtype=torch.float16, device="cuda")


def _kernels():
    from pageforge import kernels

    return kernels


def test_round_trip_is_bit_exact(pool):
    k = _kernels()
    x = torch.randn(59, C, D, dtype=torch.float16, device="cuda")
    pages = [7, 3, 30, 1]  # non-contiguous and out of order
    k.scatter_kv(x, pages, pool)
    assert torch.equal(k.gather_kv(pool, pages, 59), x)


def test_tokens_land_in_the_right_page_slot(pool):
    k = _kernels()
    x = torch.randn(40, C, D, dtype=torch.float16, device="cuda")
    pages = [5, 9, 2]
    k.scatter_kv(x, pages, pool)
    for t in range(40):
        assert torch.equal(pool[pages[t // P], t % P], x[t])


def test_scatter_leaves_other_pages_untouched(pool):
    k = _kernels()
    before = pool.clone()
    k.scatter_kv(torch.randn(20, C, D, dtype=torch.float16, device="cuda"), [4, 11], pool)
    untouched = [i for i in range(32) if i not in (4, 11)]
    assert torch.equal(pool[untouched], before[untouched])
    # Page 11 holds tokens 16..19 only; its remaining slots must be untouched.
    assert torch.equal(pool[11, 4:], before[11, 4:])


def test_scatter_at_offset_appends_after_existing_tokens(pool):
    k = _kernels()
    pages = [3, 8]
    first = torch.randn(14, C, D, dtype=torch.float16, device="cuda")
    more = torch.randn(5, C, D, dtype=torch.float16, device="cuda")  # crosses the page boundary
    k.scatter_kv(first, pages, pool)
    k.scatter_kv_at_offset(more, pages, pool, tok_offset=14)
    assert torch.equal(k.gather_kv(pool, pages, 19), torch.cat([first, more]))


@pytest.mark.parametrize("layer", [0, 5, LAYERS - 1])
def test_scatter_layer_writes_only_its_heads(pool, layer):
    k = _kernels()
    pages = [1, 2]
    before = k.gather_kv(pool, pages, 20)
    new = torch.randn(3, HEADS, D, dtype=torch.float16, device="cuda")
    k.scatter_kv_layer(new, pages, pool, tok_offset=17, head_offset=layer * HEADS)
    after = k.gather_kv(pool, pages, 20)
    cols = slice(layer * HEADS, (layer + 1) * HEADS)
    assert torch.equal(after[17:20, cols], new)
    expected = before.clone()
    expected[17:20, cols] = new
    assert torch.equal(after, expected)


def test_scalar_path_for_unaligned_widths():
    """head_dim=6 is not a multiple of 8, forcing the 2-byte (non-vector) kernel."""
    k = _kernels()
    pool = torch.zeros(8, 4, 6, 6, dtype=torch.float16, device="cuda")
    x = torch.randn(10, 6, 6, dtype=torch.float16, device="cuda")
    k.scatter_kv(x, [6, 0, 3], pool)
    assert torch.equal(k.gather_kv(pool, [6, 0, 3], 10), x)
    layer = torch.randn(2, 3, 6, dtype=torch.float16, device="cuda")
    k.scatter_kv_layer(layer, [6, 0, 3], pool, tok_offset=8, head_offset=3)
    assert torch.equal(k.gather_kv(pool, [6, 0, 3], 10)[8:, 3:], layer)


def test_bfloat16_round_trip():
    k = _kernels()
    pool = torch.zeros(4, P, C, D, dtype=torch.bfloat16, device="cuda")
    x = torch.randn(30, C, D, dtype=torch.bfloat16, device="cuda")
    k.scatter_kv(x, [2, 0], pool)
    assert torch.equal(k.gather_kv(pool, [2, 0], 30), x)


def test_page_ids_may_be_a_cuda_tensor(pool):
    k = _kernels()
    x = torch.randn(17, C, D, dtype=torch.float16, device="cuda")
    ids = torch.tensor([9, 4], dtype=torch.int32, device="cuda")
    k.scatter_kv(x, ids, pool)
    assert torch.equal(k.gather_kv(pool, ids, 17), x)


def test_zero_tokens_is_a_no_op(pool):
    k = _kernels()
    before = pool.clone()
    k.scatter_kv(torch.empty(0, C, D, dtype=torch.float16, device="cuda"), [], pool)
    assert k.gather_kv(pool, [], 0).shape == (0, C, D)
    assert torch.equal(pool, before)


def test_too_few_pages_is_rejected(pool):
    k = _kernels()
    with pytest.raises(ValueError, match="cannot hold"):
        k.scatter_kv(torch.randn(17, C, D, dtype=torch.float16, device="cuda"), [0], pool)


def test_dtype_and_shape_mismatches_are_rejected(pool):
    k = _kernels()
    with pytest.raises(ValueError):
        k.scatter_kv(torch.randn(4, C, D, dtype=torch.float32, device="cuda"), [0], pool)
    with pytest.raises(ValueError):
        k.scatter_kv(torch.randn(4, C - 1, D, dtype=torch.float16, device="cuda"), [0], pool)
    with pytest.raises(ValueError):
        k.scatter_kv_layer(torch.randn(4, HEADS, D, dtype=torch.float16, device="cuda"), [0], pool, 0, C - 1)
    with pytest.raises(ValueError):
        k.gather_kv(pool.float(), [0], 4)


def test_kernels_follow_the_current_torch_stream(pool):
    k = _kernels()
    x = torch.randn(48, C, D, dtype=torch.float16, device="cuda")
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        k.scatter_kv(x, [0, 1, 2], pool)
        out = k.gather_kv(pool, [0, 1, 2], 48)
    side.synchronize()
    assert torch.equal(out, x)


def test_dlpack_bridge_is_zero_copy():
    from pageforge.bridge import cupy_to_torch, shares_memory, torch_to_cupy

    t = torch.zeros(16, device="cuda")
    c = torch_to_cupy(t)
    assert shares_memory(t, c)
    c[3] = 7
    assert t[3].item() == 7
    back = cupy_to_torch(c)
    assert back.data_ptr() == t.data_ptr()
    with pytest.raises(ValueError):
        torch_to_cupy(torch.zeros(2))
