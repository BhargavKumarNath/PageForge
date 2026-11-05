"""PagedKVCache inside a real transformers model (GPT-2 small, fp16)."""

from __future__ import annotations

import pytest
import torch

pytestmark = [pytest.mark.gpu, pytest.mark.model]

PROMPTS = [
    "The quick brown fox jumps over the lazy",
    "In 1905, Albert Einstein published",
    "Paged attention stores the key-value cache in fixed-size blocks so that",
    "def fibonacci(n):",
]


def _greedy(model, ids, cache, steps):
    """Prefill + (steps-1) greedy decode steps; returns (steps, vocab) fp32 logits."""
    logits = []
    x = ids
    with torch.no_grad():
        for _ in range(steps):
            out = model(x, past_key_values=cache, use_cache=True)
            logits.append(out.logits[0, -1].float())
            x = out.logits[:, -1:].argmax(-1)
    return torch.stack(logits)


def _pool(model, n_pages=256, page_size=16):
    from pageforge.pool import PagedPool

    return PagedPool.for_model(model.config, n_pages, page_size)


@pytest.mark.parametrize("prompt", PROMPTS)
def test_logits_match_dynamic_cache_exactly_over_50_steps(gpt2, prompt):
    from transformers import DynamicCache

    from pageforge.cache import PagedKVCache

    model, tok = gpt2
    ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    ref = _greedy(model, ids, DynamicCache(), 50)
    cache = PagedKVCache(pool=_pool(model))
    got = _greedy(model, ids, cache, 50)
    assert (ref - got).abs().max().item() == 0.0
    assert cache.get_seq_length() == ids.shape[1] + 49


@pytest.mark.parametrize("page_size", [1, 7, 16, 64])
def test_parity_holds_for_any_page_size(gpt2, page_size):
    from transformers import DynamicCache

    from pageforge.cache import PagedKVCache

    model, tok = gpt2
    ids = tok(PROMPTS[2], return_tensors="pt").input_ids.cuda()
    ref = _greedy(model, ids, DynamicCache(), 20)
    got = _greedy(model, ids, PagedKVCache(pool=_pool(model, 512, page_size)), 20)
    assert torch.equal(ref, got)


def test_generate_api_produces_identical_tokens(gpt2):
    from pageforge.cache import PagedKVCache

    model, tok = gpt2
    ids = tok(PROMPTS[1], return_tensors="pt").input_ids.cuda()
    mask = torch.ones_like(ids)
    kw = dict(max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id, attention_mask=mask)
    ref = model.generate(ids, **kw)
    got = model.generate(ids, past_key_values=PagedKVCache(pool=_pool(model)), **kw)
    assert torch.equal(ref, got)


def test_page_count_tracks_sequence_length(gpt2):
    from pageforge.cache import PagedKVCache

    model, tok = gpt2
    pool = _pool(model)
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids.cuda()  # 8 tokens
    cache = PagedKVCache(pool=pool)
    _greedy(model, ids, cache, 51)  # 8 + 50 = 58 tokens
    assert cache.get_seq_length() == 58
    assert len(cache.page_ids) == 4  # ceil(58 / 16)
    assert pool.used_pages() == 4


def test_interleaved_sequences_share_a_pool_without_interference(gpt2):
    from transformers import DynamicCache

    from pageforge.cache import PagedKVCache

    model, tok = gpt2
    pool = _pool(model, n_pages=64)
    ids = [tok(p, return_tensors="pt").input_ids.cuda() for p in PROMPTS]
    refs = [_greedy(model, i, DynamicCache(), 30) for i in ids]

    caches = [PagedKVCache(pool=pool) for _ in ids]
    xs, logits = list(ids), [[] for _ in ids]
    with torch.no_grad():
        for _ in range(30):  # round-robin one step per sequence, so their pages interleave
            for j, cache in enumerate(caches):
                out = model(xs[j], past_key_values=cache, use_cache=True)
                logits[j].append(out.logits[0, -1].float())
                xs[j] = out.logits[:, -1:].argmax(-1)
    for ref, got in zip(refs, logits):
        assert torch.equal(ref, torch.stack(got))
    owned = [set(c.page_ids) for c in caches]
    assert sum(len(o) for o in owned) == len(set().union(*owned))  # no page shared
    pool.allocator.check_invariants()


def test_free_returns_pages_for_reuse(gpt2):
    from pageforge.cache import PagedKVCache

    model, tok = gpt2
    pool = _pool(model, n_pages=8)
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids.cuda()
    a = PagedKVCache(pool=pool)
    _greedy(model, ids, a, 40)
    held = set(a.page_ids)
    assert set(a.free()) == held
    assert pool.free_pages() == 8
    assert a.free() == []  # idempotent
    b = PagedKVCache(pool=pool)
    _greedy(model, ids, b, 40)
    assert set(b.page_ids) <= held | set(range(8))
    with pytest.raises(RuntimeError, match="freed"):
        _greedy(model, ids, a, 1)


def test_pool_exhaustion_raises_memoryerror(gpt2):
    from pageforge.cache import PagedKVCache

    model, tok = gpt2
    pool = _pool(model, n_pages=2, page_size=16)  # room for 32 tokens
    ids = tok(PROMPTS[0], return_tensors="pt").input_ids.cuda()
    with pytest.raises(MemoryError):
        _greedy(model, ids, PagedKVCache(pool=pool), 40)


def test_batch_size_above_one_is_rejected(gpt2):
    from pageforge.cache import PagedKVCache

    model, tok = gpt2
    ids = tok([PROMPTS[0]] * 2, return_tensors="pt").input_ids.cuda()
    with pytest.raises(ValueError, match="one sequence"):
        with torch.no_grad():
            model(ids, past_key_values=PagedKVCache(pool=_pool(model)), use_cache=True)


def test_private_pool_from_config(gpt2):
    from pageforge.cache import PagedKVCache

    model, tok = gpt2
    cache = PagedKVCache(n_pages=16, config=model.config)
    ids = tok(PROMPTS[3], return_tensors="pt").input_ids.cuda()
    _greedy(model, ids, cache, 5)
    assert cache.pool.n_pages == 16 and cache.get_seq_length() == ids.shape[1] + 4
    with pytest.raises(ValueError):
        PagedKVCache()
