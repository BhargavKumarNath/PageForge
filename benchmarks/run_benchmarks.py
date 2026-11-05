"""Measure PageForge on the local GPU and write benchmarks/results/results.json.

Every number in the README, dashboard and case study comes from this file's
output. Nothing here is computed from a formula unless the JSON key says so
(``*_analytic``); everything else is measured on the device.

    .venv/bin/python benchmarks/run_benchmarks.py

Sections:
  parity      max |logit difference| vs DynamicCache over 50 greedy steps
  latency     per-decode-step P50/P99, DynamicCache vs PagedKVCache, interleaved trials
  capacity    sequences that fit in one fixed KV budget: paged (run until the
              pool is exhausted) vs static pre-allocation to max_len (measured bytes)
  memory      KV bytes held by 32 sequences at decode step 50, measured
  kernels     gather / scatter bandwidth vs a device-to-device copy on the same GPU
  allocator   Rust allocator throughput through the Python bindings
"""

from __future__ import annotations

import gc
import json
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import _pageforge
import torch
from transformers import AutoTokenizer, DynamicCache, GPT2LMHeadModel, StaticCache

from pageforge.cache import PagedKVCache
from pageforge.kernels import gather_kv, scatter_kv_layer
from pageforge.pool import PagedPool

OUT = Path(__file__).parent / "results" / "results.json"
DASHBOARD_COPY = Path(__file__).resolve().parents[1] / "dashboard" / "lib" / "measured" / "results.json"
PROMPT = "The quick brown fox jumps over the lazy dog"  # 9 GPT-2 tokens
PAGE_SIZE = 16
MAX_LEN = 512  # static pre-allocation length used as the baseline


def _versions() -> dict:
    import cupy
    import transformers

    props = torch.cuda.get_device_properties(0)
    try:
        driver = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        driver = "unknown"
    return {
        "gpu": props.name,
        "vram_gb": round(props.total_memory / 1e9, 2),
        "sm": f"sm_{props.major}{props.minor}",
        "driver": driver,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cupy": cupy.__version__,
        "cupy_cuda_runtime": cupy.cuda.runtime.runtimeGetVersion(),
        "transformers": transformers.__version__,
        "pageforge_rs": _pageforge.__version__,
        "python": platform.python_version(),
    }


def _greedy_logits(model, ids, cache, steps):
    out_logits, x = [], ids
    with torch.no_grad():
        for _ in range(steps):
            out = model(x, past_key_values=cache, use_cache=True)
            out_logits.append(out.logits[0, -1].float())
            x = out.logits[:, -1:].argmax(-1)
    return torch.stack(out_logits)


def bench_parity(model, tok) -> dict:
    prompts = [
        PROMPT,
        "In 1905, Albert Einstein published",
        "Paged attention stores the key-value cache in fixed-size blocks so that",
        "def fibonacci(n):",
    ]
    pool = PagedPool.for_model(model.config, 256, PAGE_SIZE)
    worst = 0.0
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        ref = _greedy_logits(model, ids, DynamicCache(), 50)
        cache = PagedKVCache(pool=pool)
        got = _greedy_logits(model, ids, cache, 50)
        cache.free()
        worst = max(worst, (ref - got).abs().max().item())
    return {"prompts": len(prompts), "steps": 50, "max_abs_logit_diff": worst}


def _decode_times(model, ids, cache, warmup, steps) -> list[float]:
    times = []
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
        x = out.logits[:, -1:].argmax(-1)
        for i in range(warmup + steps):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = model(x, past_key_values=cache, use_cache=True)
            x = out.logits[:, -1:].argmax(-1)
            end.record()
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(start.elapsed_time(end))
    return times


def _pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def bench_latency(model, tok, trials=6, warmup=10, steps=100) -> dict:
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    pool = PagedPool.for_model(model.config, 512, PAGE_SIZE)
    hf, paged = [], []
    for t in range(trials):  # alternate order so neither side always runs first
        order = ["hf", "paged"] if t % 2 == 0 else ["paged", "hf"]
        for name in order:
            if name == "hf":
                hf += _decode_times(model, ids, DynamicCache(), warmup, steps)
            else:
                cache = PagedKVCache(pool=pool)
                paged += _decode_times(model, ids, cache, warmup, steps)
                cache.free()
    res = {
        "setup": f"GPT-2 small fp16, batch 1, prompt 9 tokens, {trials} trials x {steps} decode steps "
        f"({warmup} warmup each), CUDA events, trial order alternated",
        "samples_per_cache": len(hf),
        "dynamic_cache_ms": {"p50": _pct(hf, 0.5), "p99": _pct(hf, 0.99), "mean": statistics.fmean(hf)},
        "paged_ms": {"p50": _pct(paged, 0.5), "p99": _pct(paged, 0.99), "mean": statistics.fmean(paged)},
    }
    res["p50_overhead_pct"] = (res["paged_ms"]["p50"] / res["dynamic_cache_ms"]["p50"] - 1) * 100
    res["p99_overhead_pct"] = (res["paged_ms"]["p99"] / res["dynamic_cache_ms"]["p99"] - 1) * 100
    return res


def _static_cache_bytes(model, n_seqs: int) -> int:
    """Measured bytes torch allocates for n_seqs StaticCache(max_len) sequences."""
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    caches = []
    ids = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    with torch.no_grad():
        for _ in range(n_seqs):
            c = StaticCache(config=model.config, max_cache_len=MAX_LEN)
            model(ids, past_key_values=c, use_cache=True)  # forces the lazy allocation
            caches.append(c)
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() - before
    del caches
    gc.collect()
    return used


def _dynamic_cache_bytes(model, tok, n_seqs: int, decode_steps: int) -> int:
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    caches = []
    with torch.no_grad():
        for _ in range(n_seqs):
            c = DynamicCache()
            out = model(ids, past_key_values=c, use_cache=True)
            x = out.logits[:, -1:].argmax(-1)
            for _ in range(decode_steps):
                out = model(x, past_key_values=c, use_cache=True)
                x = out.logits[:, -1:].argmax(-1)
            del out
            caches.append(c)
    torch.cuda.synchronize()
    used = torch.cuda.memory_allocated() - before
    del caches
    gc.collect()
    return used


def bench_memory_and_capacity(model, tok, n_pages=512, n_seqs=32, decode_steps=50) -> dict:
    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    prompt_len = ids.shape[1]
    final_len = prompt_len + decode_steps

    # Paged: 32 sequences decoding round-robin in one pool.
    pool = PagedPool.for_model(model.config, n_pages, PAGE_SIZE)
    caches = [PagedKVCache(pool=pool) for _ in range(n_seqs)]
    xs = [ids] * n_seqs
    with torch.no_grad():
        for _ in range(decode_steps + 1):
            for j, c in enumerate(caches):
                out = model(xs[j], past_key_values=c, use_cache=True)
                xs[j] = out.logits[:, -1:].argmax(-1)
    assert all(c.get_seq_length() == final_len for c in caches)
    paged_used = pool.used_bytes()
    for c in caches:
        c.free()

    static_one = _static_cache_bytes(model, 1)
    static_n = _static_cache_bytes(model, n_seqs)
    dynamic_n = _dynamic_cache_bytes(model, tok, n_seqs, decode_steps)

    # Capacity: keep admitting sequences into the same pool until it is full.
    budget = pool.reserved_bytes
    admitted: list[PagedKVCache] = []
    xs = []
    try:
        with torch.no_grad():
            while True:
                c = PagedKVCache(pool=pool)
                out = model(ids, past_key_values=c, use_cache=True)
                for _ in range(decode_steps):
                    out = model(out.logits[:, -1:].argmax(-1), past_key_values=c, use_cache=True)
                admitted.append(c)
    except MemoryError:
        pass
    paged_capacity = len(admitted)
    for c in admitted:
        c.free()
    del pool
    gc.collect()

    return {
        "memory": {
            "setup": f"{n_seqs} GPT-2 sequences, prompt {prompt_len} tokens + {decode_steps} decode steps "
            f"= {final_len} tokens each, page_size {PAGE_SIZE}",
            "paged_pages_in_use_mb": paged_used / 1e6,
            "static_prealloc_max512_mb": static_n / 1e6,
            "dynamic_cache_mb": dynamic_n / 1e6,
            "static_vs_paged_ratio": static_n / paged_used,
            "note": "paged = bytes of pages owned by live sequences; the pool itself is reserved up front. "
            "DynamicCache also grows with the sequence, so the saving is against static pre-allocation.",
        },
        "capacity": {
            "setup": f"one KV budget = a {n_pages}-page pool ({budget / 1e6:.1f} MB K+V); "
            f"every sequence runs to {final_len} tokens",
            "kv_budget_mb": budget / 1e6,
            "paged_sequences": paged_capacity,
            "static_bytes_per_sequence_mb": static_one / 1e6,
            "static_sequences": int(budget // static_one),
            "ratio": paged_capacity / max(1, int(budget // static_one)),
            "method": "paged: sequences admitted until the allocator raised MemoryError; "
            "static: budget divided by the measured bytes of one StaticCache(max_cache_len=512)",
        },
    }


def _time_ms(fn, iters=50) -> float:
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench_kernels(seq_len=2048) -> dict:
    pool = PagedPool(n_pages=seq_len // PAGE_SIZE + 8, page_size=PAGE_SIZE)
    pages = torch.randperm(pool.n_pages, device="cuda")[: seq_len // PAGE_SIZE].to(torch.int32)
    row_bytes = pool.geometry.n_cols * pool.geometry.head_dim * 2
    moved = 2 * seq_len * row_bytes  # read + write

    gather_ms = _time_ms(lambda: gather_kv(pool.pool_k, pages, seq_len))
    layer = torch.randn(seq_len, pool.geometry.n_heads, pool.geometry.head_dim, dtype=torch.float16, device="cuda")
    layer_moved = 2 * layer.numel() * 2
    scatter_ms = _time_ms(lambda: scatter_kv_layer(layer, pages, pool.pool_k, 0, 5 * pool.geometry.n_heads))
    src = torch.empty(seq_len * row_bytes // 2, dtype=torch.float16, device="cuda")
    dst = torch.empty_like(src)
    copy_ms = _time_ms(lambda: dst.copy_(src))
    return {
        "setup": f"{seq_len} tokens, fp16, random page order; gather = all 12 layers x 12 heads x 64, "
        "scatter = one layer",
        "gather_gb_s": moved / gather_ms / 1e6,
        "scatter_layer_gb_s": layer_moved / scatter_ms / 1e6,
        "torch_d2d_copy_gb_s": moved / copy_ms / 1e6,
        "note": "GB/s counts bytes read + written; the torch device-to-device copy of the same bytes "
        "is the measured ceiling on this GPU",
    }


def bench_allocator(cycles=500, seqs=16, pages_per_seq=8) -> dict:
    pf = _pageforge.PageForge(512, PAGE_SIZE)
    best = float("inf")
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(cycles):
            for s in range(seqs):
                pf.alloc_for_seq(s, pages_per_seq)
            for s in range(seqs):
                pf.free_seq(s)
        best = min(best, time.perf_counter() - t0)
    pf.check_invariants()
    pages_moved = cycles * seqs * pages_per_seq * 2
    return {
        "setup": f"{cycles} cycles x {seqs} sequences x {pages_per_seq} pages, alloc then free, best of 5, "
        "called from Python",
        "calls_per_s": cycles * seqs * 2 / best,
        "pages_per_s": pages_moved / best,
        "leaked_pages": 512 - pf.free_pages(),
    }


def main() -> int:
    if not torch.cuda.is_available():
        print("needs a CUDA GPU", file=sys.stderr)
        return 1
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().half().eval()

    results: dict = {"measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env": _versions()}
    for name, fn in [
        ("parity", lambda: bench_parity(model, tok)),
        ("latency", lambda: bench_latency(model, tok)),
        ("memory_capacity", lambda: bench_memory_and_capacity(model, tok)),
        ("kernels", bench_kernels),
        ("allocator", bench_allocator),
    ]:
        print(f"running {name}...", flush=True)
        r = fn()
        if name == "memory_capacity":
            results.update(r)
        else:
            results[name] = r
        torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    # The dashboard deploys from dashboard/, so it gets its own copy.
    DASHBOARD_COPY.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_COPY.write_text(json.dumps(results, indent=2) + "\n")
    OUT.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
