"""Fused all-layer scatter: one launch must write exactly what the per-layer scatters write."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.gpu

from tests.test_cache import PROMPTS, _greedy, _pool  # noqa: E402


def _pools(n_pages, page_size, n_layers, n_heads, head_dim, dtype):
    shape = (n_pages, page_size, n_layers * n_heads, head_dim)
    a = torch.zeros(shape, dtype=dtype, device="cuda")
    return a, torch.zeros_like(a), torch.zeros_like(a), torch.zeros_like(a)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout", ["contiguous", "hf_transposed"])
@pytest.mark.parametrize("n_new,tok_offset", [(1, 0), (1, 37), (5, 14), (33, 0)])
@pytest.mark.parametrize("head_dim", [64, 20])
def test_matches_per_layer_scatter_bit_for_bit(dtype, layout, n_new, tok_offset, head_dim):
    from pageforge.kernels import scatter_kv_layer, scatter_kv_layers_fused

    n_layers, n_heads, page_size = 5, 4, 16
    n_pages = 8
    pages = torch.tensor([6, 2, 5, 0, 7, 1, 4, 3], dtype=torch.int32, device="cuda")
    ref_k, ref_v, got_k, got_v = _pools(n_pages, page_size, n_layers, n_heads, head_dim, dtype)
    keys, values = [], []
    for _ in range(n_layers):
        if layout == "contiguous":
            k = torch.randn(1, n_heads, n_new, head_dim, device="cuda").to(dtype)
            v = torch.randn(1, n_heads, n_new, head_dim, device="cuda").to(dtype)
        else:  # (1, T, H, D) viewed as (1, H, T, D): what a transformers attention layer produces
            k = torch.randn(1, n_new, n_heads, head_dim, device="cuda").to(dtype).transpose(1, 2)
            v = torch.randn(1, n_new, n_heads, head_dim, device="cuda").to(dtype).transpose(1, 2)
        keys.append(k)
        values.append(v)
    for layer, (k, v) in enumerate(zip(keys, values)):
        scatter_kv_layer(k[0].transpose(0, 1).contiguous(), pages, ref_k, tok_offset, layer * n_heads)
        scatter_kv_layer(v[0].transpose(0, 1).contiguous(), pages, ref_v, tok_offset, layer * n_heads)
    assert scatter_kv_layers_fused(keys, values, pages, got_k, got_v, tok_offset)
    assert torch.equal(ref_k, got_k) and torch.equal(ref_v, got_v)


def test_declines_mismatched_inputs_without_writing():
    from pageforge.kernels import scatter_kv_layers_fused

    pool_k, pool_v, _, _ = _pools(4, 16, 2, 2, 64, torch.float16)
    pages = torch.arange(4, dtype=torch.int32, device="cuda")
    k = [torch.ones(1, 2, 1, 64, dtype=torch.float16, device="cuda") for _ in range(3)]  # 3 layers, pool has 2
    assert scatter_kv_layers_fused(k, k, pages, pool_k, pool_v, 0) is False
    assert not pool_k.any() and not pool_v.any()


@pytest.mark.model
@pytest.mark.parametrize("prompt", PROMPTS)
def test_gpt2_logits_identical_with_fused_scatter(gpt2, prompt):
    from transformers import DynamicCache

    from pageforge.cache import PagedKVCache

    model, tok = gpt2
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    ref = _greedy(model, ids, DynamicCache(), 50)
    cache = PagedKVCache(pool=_pool(model), fused_scatter=True)
    got = _greedy(model, ids, cache, 50)
    assert (ref - got).abs().max().item() == 0.0
    # The pool must hold exactly what the unfused cache stores.
    plain = PagedKVCache(pool=_pool(model))
    _greedy(model, ids, plain, 50)
    n = cache.get_seq_length()
    from pageforge.kernels import gather_kv

    for a, b in ((cache, plain),):
        assert torch.equal(gather_kv(a.pool.pool_k, a.page_ids, n), gather_kv(b.pool.pool_k, b.page_ids, n))
        assert torch.equal(gather_kv(a.pool.pool_v, a.page_ids, n), gather_kv(b.pool.pool_v, b.page_ids, n))
