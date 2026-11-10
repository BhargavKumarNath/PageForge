# PageForge Phase 4 protocol

Written 2026-11-10 before `benchmarks/phase4.py` was run. Deviations are appended at the bottom.

## Questions and fixed methods
1. **Page size (8, 16, 32, 64) and pool size (128 to 2048 pages).** Capacity is counted with the real Rust allocator: sequences with uniform random length in [16, 512] tokens are admitted into a fixed budget (the existing 302 MB, i.e. 8,192 tokens) until `MemoryError`; 20 seeded orderings, mean and sd. Reported with internal waste and pool utilisation, and compared with static 512-token pre-allocation of the same budget. Decode latency for each page size on GPT-2 (6 trials). Gather bandwidth at each page size (2,048 tokens, random page order).
2. **Fused scatter.** The per-layer path issues 4 launches per layer (2 transpose copies, 2 scatters). The fused path defers them and issues one launch for all layers' K and V after the last layer. Checked bit-for-bit against the per-layer path (tests/test_fused_scatter.py) before timing. Latency of DynamicCache, PagedKVCache and PagedKVCache(fused) on GPT-2 with 12 trials x 100 decode steps, trial order rotated. The "shrink" is (unfused gap - fused gap) / unfused gap, where each gap is the mean over trials of paged P50 minus DynamicCache P50. The plan quoted +33%; whatever is measured is reported.
3. **Second model.** TinyLlama-1.1B-Chat (22 layers, 4 KV heads, head_dim 64, grouped-query attention, fp16). Logit parity against DynamicCache over 4 prompts x 50 greedy steps for the unfused and fused paths; memory of 16 sequences at decode step 50 against static 512-token pre-allocation and DynamicCache; latency as in 2.
4. **vLLM.** Attempted in an isolated environment, GPT-2 fp16 on the same GPU, only if it installs and runs within 8 GB. If it does not, that is reported as not done.

## Rules
- One run of `phase4.py`; if a run crashes before writing, it is repeated unchanged. No sweep, trial count or sequence length is changed after seeing a result.
- Latency runs happen with no other heavy job on the machine, because the paged path is CPU-launch-bound.

## Deviations
(none yet)
