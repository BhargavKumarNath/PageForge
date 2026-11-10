"""vLLM GPT-2 fp16 decode latency, batch 1, same prompt length as phase4.py. Run in an isolated env (see PHASE4_PROTOCOL.md).
Per-token latency = (time for 100 new tokens - time for 1 new token) / 99, which removes prefill. 12 trials, 1 warm-up."""
import json, statistics, sys, time
from vllm import LLM, SamplingParams

llm = LLM("openai-community/gpt2", dtype="float16", gpu_memory_utilization=0.5, max_model_len=512, enforce_eager=False)
prompt = "Paged attention stores the"
def t(n):
    sp = SamplingParams(max_tokens=n, min_tokens=n, temperature=0.0, ignore_eos=True)
    s = time.perf_counter(); llm.generate([prompt], sp, use_tqdm=False); return time.perf_counter() - s
t(20); t(20)
per = []
for _ in range(12):
    a, b = t(1), t(101)
    per.append((b - a) / 100 * 1000)
out = {"model": "gpt2", "dtype": "float16", "per_token_ms_p50": statistics.median(per), "per_token_ms_min": min(per),
       "per_token_ms_max": max(per), "trials": per, "vllm_version": __import__("vllm").__version__}
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(out["per_token_ms_p50"], out["per_token_ms_min"], out["per_token_ms_max"])
