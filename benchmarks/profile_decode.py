"""Where does the paged decode step spend its extra time?

Writes benchmarks/results/profile.json (and the dashboard copy):
  gpu_ms_per_step      GPU kernel time per decode step (torch profiler, incl. CuPy kernels)
  kernels_per_step     GPU kernel launches per decode step
  cache_update_host_ms CPU time inside PagedKVCache._layer_update per step (paged only)

Run on an otherwise idle GPU:  .venv/bin/python benchmarks/profile_decode.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile
from transformers import AutoTokenizer, DynamicCache, GPT2LMHeadModel

from pageforge.cache import PagedKVCache
from pageforge.pool import PagedPool

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "benchmarks" / "results" / "profile.json"
DASHBOARD_COPY = ROOT / "dashboard" / "lib" / "measured" / "profile.json"
STEPS = 50


def _profile(model, ids, cache) -> dict:
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
        x = out.logits[:, -1:].argmax(-1)
        for _ in range(10):  # warm-up
            out = model(x, past_key_values=cache, use_cache=True)
            x = out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(STEPS):
                out = model(x, past_key_values=cache, use_cache=True)
                x = out.logits[:, -1:].argmax(-1)
            torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.device_type.name == "CUDA"]
    return {
        "gpu_ms_per_step": sum(e.self_device_time_total for e in events) / STEPS / 1000,
        "kernels_per_step": sum(e.count for e in events) / STEPS,
    }


def _cache_update_host_ms(model, ids, pool) -> float:
    original = PagedKVCache._layer_update
    spent = [0.0]

    def timed(self, *args):
        t0 = time.perf_counter()
        result = original(self, *args)
        spent[0] += time.perf_counter() - t0
        return result

    PagedKVCache._layer_update = timed
    try:
        cache = PagedKVCache(pool=pool)
        with torch.no_grad():
            out = model(ids, past_key_values=cache, use_cache=True)
            x = out.logits[:, -1:].argmax(-1)
            spent[0] = 0.0
            for _ in range(STEPS):
                out = model(x, past_key_values=cache, use_cache=True)
                x = out.logits[:, -1:].argmax(-1)
        cache.free()
    finally:
        PagedKVCache._layer_update = original
    return spent[0] / STEPS * 1000


def main() -> None:
    tok = AutoTokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().half().eval()
    ids = tok("The quick brown fox jumps over the lazy dog", return_tensors="pt").input_ids.cuda()
    pool = PagedPool.for_model(model.config, 512, 16)
    paged_cache = PagedKVCache(pool=pool)
    result = {
        "setup": f"GPT-2 small fp16, batch 1, {STEPS} profiled decode steps after 10 warm-up",
        "dynamic_cache": _profile(model, ids, DynamicCache()),
        "paged": _profile(model, ids, paged_cache),
    }
    paged_cache.free()
    result["paged"]["cache_update_host_ms"] = _cache_update_host_ms(model, ids, pool)
    for path in (OUT, DASHBOARD_COPY):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
