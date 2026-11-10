"""Phase 4 measurements: page-size and pool-size sweeps, fused scatter, a second (larger, GQA) model.

    .venv/bin/python benchmarks/phase4.py            # everything
    .venv/bin/python benchmarks/phase4.py sweeps     # allocator + kernel sweeps only (no model needed)

Writes benchmarks/results/phase4.json. Nothing in it is typed in: every value comes from this run.
Protocol: benchmarks/PHASE4_PROTOCOL.md (written before the first run).
"""

from __future__ import annotations

import gc
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache

sys.path.insert(0, str(Path(__file__).parent))
import run_benchmarks as rb  # noqa: E402

from pageforge.cache import PagedKVCache  # noqa: E402
from pageforge.kernels import gather_kv, scatter_kv_layer  # noqa: E402
from pageforge.pool import KVGeometry, PagedPool  # noqa: E402

OUT = Path(__file__).parent / "results" / "phase4.json"
PAGE_SIZES = [8, 16, 32, 64]
POOL_PAGES = [128, 256, 512, 1024, 2048]
GPT2_BUDGET_TOKENS = 512 * 16  # the existing 302 MB budget, expressed in tokens
LEN_LO, LEN_HI, N_ORDERINGS = 16, 512, 20


def _bytes_per_token(geom: KVGeometry) -> int:
    return 2 * geom.n_cols * geom.head_dim * 2


def capacity_by_allocator(geom: KVGeometry, page_size: int, n_pages: int, seed: int) -> dict:
    """Admit sequences of random length (uniform LEN_LO..LEN_HI) into a real pool until MemoryError."""
    pool = PagedPool(n_pages, page_size, geom)
    rng = np.random.default_rng(seed)
    admitted, tokens = 0, 0
    while True:
        n = int(rng.integers(LEN_LO, LEN_HI + 1))
        sid = pool.new_seq_id()
        try:
            pool.ensure_capacity(sid, n)
        except MemoryError:
            break
        admitted += 1
        tokens += n
    used = pool.used_pages()
    out = {"admitted": admitted, "tokens": tokens, "pages_used": used, "pages_total": n_pages,
           "internal_waste_pct": 100 * (used * page_size - tokens) / (used * page_size),
           "pool_utilisation_pct": 100 * tokens / (n_pages * page_size)}
    del pool
    return out


def sweeps(geom: KVGeometry) -> dict:
    res: dict = {"seq_len_range": [LEN_LO, LEN_HI], "orderings": N_ORDERINGS, "page_size": {}, "pool_size": {}}
    per_tok = _bytes_per_token(geom)
    budget = GPT2_BUDGET_TOKENS * per_tok
    static_seqs = int(budget // (per_tok * rb.MAX_LEN))
    res["budget_mb"] = budget / 1e6
    res["static_512_sequences_in_budget"] = static_seqs
    for ps in PAGE_SIZES:
        runs = [capacity_by_allocator(geom, ps, GPT2_BUDGET_TOKENS // ps, seed) for seed in range(N_ORDERINGS)]
        cap = [r["admitted"] for r in runs]
        # kernel bandwidth at this page size, 2048 tokens, random page order
        pool = PagedPool(GPT2_BUDGET_TOKENS // ps, ps, geom)
        n = 2048
        pages = torch.randperm(pool.n_pages, device="cuda")[: n // ps].to(torch.int32)
        dense = torch.randn(n, geom.n_cols, geom.head_dim, device="cuda").half()
        from pageforge.kernels import scatter_kv

        pool.pool_k.zero_()
        scatter_kv(dense, pages, pool.pool_k)
        t_g = rb._time_ms(lambda: gather_kv(pool.pool_k, pages, n))
        gbs = 2 * dense.numel() * 2 / (t_g * 1e6)
        res["page_size"][str(ps)] = {
            "pages_in_budget": GPT2_BUDGET_TOKENS // ps,
            "capacity_mean": statistics.fmean(cap), "capacity_sd": statistics.pstdev(cap),
            "capacity_min": min(cap), "capacity_max": max(cap),
            "internal_waste_pct_mean": statistics.fmean(r["internal_waste_pct"] for r in runs),
            "pool_utilisation_pct_mean": statistics.fmean(r["pool_utilisation_pct"] for r in runs),
            "capacity_vs_static_x": statistics.fmean(cap) / static_seqs,
            "gather_gb_s": gbs,
        }
        del pool, dense
        gc.collect(); torch.cuda.empty_cache()
    for npg in POOL_PAGES:
        runs = [capacity_by_allocator(geom, 16, npg, seed) for seed in range(N_ORDERINGS)]
        cap = [r["admitted"] for r in runs]
        static = int(npg * 16 * per_tok // (per_tok * rb.MAX_LEN))
        res["pool_size"][str(npg)] = {"pool_mb": npg * 16 * per_tok / 1e6, "capacity_mean": statistics.fmean(cap),
                                      "capacity_sd": statistics.pstdev(cap), "static_512_sequences": static,
                                      "capacity_vs_static_x": statistics.fmean(cap) / max(1, static),
                                      "pool_utilisation_pct_mean": statistics.fmean(r["pool_utilisation_pct"] for r in runs)}
    return res


def latency_compare(model, ids, makers: dict, trials=12, warmup=10, steps=100) -> dict:
    """makers: name -> zero-arg callable returning (cache, cleanup). Trial order rotates (Latin square)."""
    names = list(makers)
    samples = {n: [] for n in names}
    trial_p50 = {n: [] for n in names}
    for t in range(trials):
        order = names[t % len(names):] + names[: t % len(names)]
        for n in order:
            cache, cleanup = makers[n]()
            xs = rb._decode_times(model, ids, cache, warmup, steps)
            cleanup()
            samples[n] += xs
            trial_p50[n].append(rb._pct(xs, 0.5))
    out = {}
    for n in names:
        out[n] = {"p50": rb._pct(samples[n], 0.5), "p99": rb._pct(samples[n], 0.99), "mean": statistics.fmean(samples[n]),
                  "trial_p50_min": min(trial_p50[n]), "trial_p50_max": max(trial_p50[n]), "trial_p50": trial_p50[n]}
    return out


def overhead_summary(lat: dict, base="dynamic", others=()) -> dict:
    out = {}
    for o in others:
        d = [(p - b) for p, b in zip(lat[o]["trial_p50"], lat[base]["trial_p50"])]
        out[o] = {"p50_overhead_pct": (lat[o]["p50"] / lat[base]["p50"] - 1) * 100,
                  "p99_overhead_pct": (lat[o]["p99"] / lat[base]["p99"] - 1) * 100,
                  "gap_ms_trial_mean": statistics.fmean(d), "gap_ms_trial_min": min(d), "gap_ms_trial_max": max(d)}
    return out


def model_makers(model, pool_pages: int, page_size: int):
    pool = PagedPool.for_model(model.config, pool_pages, page_size)

    def mk(fused):
        def f():
            c = PagedKVCache(pool=pool, fused_scatter=fused)
            return c, c.free
        return f

    return {"dynamic": lambda: (DynamicCache(), lambda: None), "paged": mk(False), "paged_fused": mk(True)}


def parity_and_memory(model, tok, prompts, page_size=16, n_pages=512, n_seqs=16, decode_steps=50) -> dict:
    pool = PagedPool.for_model(model.config, n_pages, page_size)
    worst = {"unfused": 0.0, "fused": 0.0}
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.cuda()
        ref = rb._greedy_logits(model, ids, DynamicCache(), 50)
        for k, fused in (("unfused", False), ("fused", True)):
            c = PagedKVCache(pool=pool, fused_scatter=fused)
            got = rb._greedy_logits(model, ids, c, 50)
            c.free()
            worst[k] = max(worst[k], (ref - got).abs().max().item())
    ids = tok(prompts[0], return_tensors="pt").input_ids.cuda()
    final_len = ids.shape[1] + decode_steps
    caches = [PagedKVCache(pool=pool, fused_scatter=True) for _ in range(n_seqs)]
    xs = [ids] * n_seqs
    with torch.no_grad():
        for _ in range(decode_steps + 1):
            for j, c in enumerate(caches):
                out = model(xs[j], past_key_values=c, use_cache=True)
                xs[j] = out.logits[:, -1:].argmax(-1)
    paged_used = pool.used_bytes()
    for c in caches:
        c.free()
    static_n = rb._static_cache_bytes(model, n_seqs)
    dyn_n = rb._dynamic_cache_bytes(model, tok, n_seqs, decode_steps)
    g = pool.geometry
    budget = pool.reserved_bytes
    static_one = rb._static_cache_bytes(model, 1)
    return {"geometry": {"layers": g.n_layers, "kv_heads": g.n_heads, "head_dim": g.head_dim,
                         "bytes_per_token": _bytes_per_token(g)},
            "parity_max_abs_logit_diff": worst, "parity_prompts": len(prompts), "parity_steps": 50,
            "memory": {"n_seqs": n_seqs, "final_len": final_len, "paged_mb": paged_used / 1e6,
                       "static_512_mb": static_n / 1e6, "dynamic_mb": dyn_n / 1e6, "static_vs_paged_x": static_n / paged_used},
            "capacity_budget_mb": budget / 1e6, "static_sequences_in_budget": int(budget // static_one),
            "static_bytes_per_seq_mb": static_one / 1e6}


def main() -> int:
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res: dict = {"measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "env": rb._versions()}
    if OUT.exists():
        res.update({k: v for k, v in json.loads(OUT.read_text()).items() if k not in ("measured_at", "env")})
    print("sweeps...", flush=True)
    res["sweeps"] = sweeps(KVGeometry(12, 12, 64))
    if which != "sweeps":
        tok = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2", dtype=torch.float16).cuda().eval()
        ids = tok(rb.PROMPT, return_tensors="pt").input_ids.cuda()
        print("gpt2 latency (dynamic / paged / paged_fused)...", flush=True)
        lat = latency_compare(model, ids, model_makers(model, 512, 16))
        res["gpt2_latency"] = {"setup": "GPT-2 small fp16, batch 1, 9-token prompt, 12 trials x 100 decode steps (10 warm-up), "
                               "CUDA events, trial order rotated", **lat,
                               "overhead": overhead_summary(lat, "dynamic", ["paged", "paged_fused"])}
        o = res["gpt2_latency"]["overhead"]
        gap_u, gap_f = o["paged"]["gap_ms_trial_mean"], o["paged_fused"]["gap_ms_trial_mean"]
        res["gpt2_latency"]["fused_gap_shrink_pct"] = 100 * (gap_u - gap_f) / gap_u
        print("gpt2 latency by page size...", flush=True)
        res["gpt2_latency_by_page_size"] = {}
        for ps in PAGE_SIZES:
            r = latency_compare(model, ids, {k: v for k, v in model_makers(model, GPT2_BUDGET_TOKENS // ps, ps).items() if k != "paged_fused"},
                                trials=6)
            res["gpt2_latency_by_page_size"][str(ps)] = {"paged_p50": r["paged"]["p50"], "dynamic_p50": r["dynamic"]["p50"],
                                                         "paged_p99": r["paged"]["p99"], "dynamic_p99": r["dynamic"]["p99"],
                                                         "p50_overhead_pct": (r["paged"]["p50"] / r["dynamic"]["p50"] - 1) * 100}
        del model; gc.collect(); torch.cuda.empty_cache()

        print("tinyllama...", flush=True)
        name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        tok2 = AutoTokenizer.from_pretrained(name)
        m2 = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).cuda().eval()
        prompts = ["The quick brown fox jumps over the lazy", "In 1905, Albert Einstein published",
                   "Paged attention stores the key-value cache in fixed-size blocks so that", "def fibonacci(n):"]
        pm = parity_and_memory(m2, tok2, prompts)
        ids2 = tok2(rb.PROMPT, return_tensors="pt").input_ids.cuda()
        lat2 = latency_compare(m2, ids2, model_makers(m2, 512, 16))
        res["tinyllama"] = {"model": name, "params_b": sum(p.numel() for p in m2.parameters()) / 1e9, **pm,
                            "prompt_tokens": ids2.shape[1],
                            "latency": {"setup": "fp16, batch 1, 12 trials x 100 decode steps (10 warm-up), CUDA events", **lat2,
                                        "overhead": overhead_summary(lat2, "dynamic", ["paged", "paged_fused"])}}
        del m2; gc.collect(); torch.cuda.empty_cache()
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "env"}, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
