# PageForge

**Paged KV-cache for LLM inference: a Rust page allocator, CUDA gather/scatter kernels, and a drop-in `transformers` cache.**

**[Live dashboard](https://page-forge-five.vercel.app/)** · [CI](.github/workflows/ci.yml) · numbers below come from [`benchmarks/results/`](benchmarks/results/)

A from-scratch implementation of the core idea behind [vLLM's PagedAttention](https://arxiv.org/abs/2309.06180). Instead of reserving a KV-cache for `max_seq_len` tokens per sequence up front, sequences take fixed-size *pages* from a shared pool as they grow and hand them back when they finish.

Measured on an RTX 4070 Laptop GPU with GPT-2 small (fp16):

| | Result |
|---|---|
| Correctness | Logits **bit-identical** to Hugging Face `DynamicCache` (max abs diff **0.0**, 4 prompts × 50 greedy steps) |
| Capacity | **128 vs 15** sequences fit in one 302 MB KV budget, against static 512-token pre-allocation (**8.5×**) |
| Memory | 32 sequences at decode step 50 hold **75.5 MB** of pages vs **604.2 MB** pre-allocated (**8.0×**) |
| Cost | Decode step **+88% at P50** (3.79 → 7.14 ms), almost all of it host-side launch overhead (see [Latency](#latency)) |
| Tests | **61 passing**: 54 Python + 7 Rust |

---

## Project history

This repo started as **SymboLR**, a genetic-programming engine in Rust that used MAP-Elites search to evolve symbolic learning-rate formulas of the form `lr = f(t, g, Δl)`. The search found formulas, but they gamed the proxy: the synthetic training tasks were easy enough that any learning rate in `[0.01, 10.0]` worked, so the system picked high-LR formulas that converged fast on proxies and fell apart on real training runs. I could not close that proxy-to-real gap without solving the original problem, so I pivoted to a problem with ground truth I can measure on hardware. Same stack (Rust, CUDA, PyTorch, PyO3), different problem: PageForge.

The first PageForge source tree was lost in a machine migration before it was pushed; only the dashboard survived. Everything in `pageforge-rs/`, `pageforge/` and `tests/` was rebuilt and every number re-measured. Numbers from the first version are not quoted anywhere.

---

## Problem

A KV-cache stores, for every generated token, a key and a value vector per layer per head. For GPT-2 small in fp16:

```
2 (K+V) × 12 layers × 12 heads × 64 dims × 2 bytes = 36,864 bytes per token
```

A server that pre-allocates `max_seq_len = 512` tokens per sequence (what `transformers.StaticCache` does, and what fixed-slot servers do) commits **18.9 MB per sequence** (measured) at the first token, whether the sequence ends after 10 tokens or 500. Most of that memory is never written.

Hugging Face's default `DynamicCache` does not pre-allocate: it grows with `torch.cat` every step. Its byte count is close to PageForge's (69.6 MB vs 75.5 MB for the 32-sequence case above, measured). What paging adds over it is a **fixed, shared pool** with an allocator that can admit, bound and recycle sequences, which is what a server needs to schedule many requests in a known amount of memory.

## Design

```
Sequence starts:   ensure_capacity(seq=0, 9 tokens)    → [page 3]
After 16 tokens:   ensure_capacity(seq=0, 17 tokens)   → [page 3, page 7]
Sequence ends:     free_seq(seq=0)                     → pages 3, 7 back on the free list
Next sequence:     ensure_capacity(seq=1, 9 tokens)    → [page 3]   (reuse, FIFO)
```

```
transformers model
    │  past_key_values=PagedKVCache(pool)        pageforge/cache.py
    ▼
PagedKVCache (DynamicCache subclass, one per sequence)
    ├── layer 0 update: allocator grows the block table; gather all layers' history (2 kernel calls)
    └── each layer:     scatter its new tokens into the pages (2 kernel calls), return cat(past, new)
    │
PagedPool                                         pageforge/pool.py
    ├── _pageforge.PageForge  (Rust, PyO3)        pageforge-rs/src/
    │     free list: VecDeque<u32>, O(1) per page, all-or-nothing allocation
    │     block table: HashMap<seq_id, Vec<page_id>>
    └── pool_k, pool_v: (n_pages, page_size, n_layers × n_heads, head_dim)  fp16 / bf16
            │
            ▼
    paged_copy kernel (CuPy RawKernel, NVRTC)     pageforge/kernels/kv_cache.py
```

**Decisions and why**

- **All layers in one pool row.** A token's K for every layer and head sits in one contiguous row, so a decode step gathers the whole history with 2 kernel calls instead of 2 × n_layers.
- **One kernel for every copy.** Gather, scatter, scatter-at-offset and per-layer scatter are all "copy rows between a dense tensor and paged rows, optionally into a column slice". One kernel handles both directions, using 16-byte vector loads when widths and offsets are multiples of 8 fp16 values and 2-byte loads otherwise. It copies raw 16-bit words, so fp16 and bf16 work alike.
- **CuPy RawKernel instead of a PyTorch C++ extension.** NVRTC compiles at runtime, so there is no host-compiler toolchain to match. Tensors cross over with DLPack (zero-copy), and kernels launch on torch's current stream, so no extra synchronisation is needed.
- **Allocator in Rust with no Python in its core.** `cargo test` checks the invariants (every page owned by exactly one of the free list or one sequence) with no interpreter. The PyO3 wrapper is a feature flag. Out of pages raises `MemoryError` and allocates nothing, so a scheduler can try to admit a request and back off cleanly.
- **Exact parity as the correctness bar.** `update()` returns the same `cat(past, new)` tensors `DynamicCache` does, so attention sees identical inputs and the logits match to the bit. Tests assert `== 0.0`, not a tolerance.

Limits: one sequence per cache object (batch size 1), no `crop`, beam search or sliding-window layers, and attention still runs on a gathered contiguous copy rather than reading pages directly.

---

## Performance

All numbers: RTX 4070 Laptop GPU (sm_89, 8 GB), driver 595.99, PyTorch 2.14 (CUDA 13.0), CuPy 14.2, transformers 5.17, GPT-2 small fp16, page size 16. Reproduce with `python benchmarks/run_benchmarks.py`; raw output in [`benchmarks/results/results.json`](benchmarks/results/results.json).

### Capacity and memory

| Measurement | Static pre-alloc (512) | PageForge | |
|---|---|---|---|
| Sequences that fit in a 302 MB KV budget, each run to 59 tokens | 15 | 128 | **8.5×** |
| KV memory held by 32 sequences at decode step 50 | 604.2 MB | 75.5 MB | **8.0×** |
| Same 32 sequences in HF `DynamicCache` | | | 69.6 MB |

PageForge capacity is counted by admitting real GPT-2 sequences into a 512-page pool until the allocator raised `MemoryError`. The static baseline divides the same budget by the measured bytes of one `StaticCache(max_cache_len=512)`. The advantage depends on sequence length: it is largest for short sequences and shrinks as they approach the reservation.

### Latency

| Cache | P50 | P99 |
|---|---|---|
| HF `DynamicCache` | 3.79 ms | 5.87 ms |
| PageForge `PagedKVCache` | 7.14 ms | 10.31 ms |
| Overhead | **+88%** | **+75%** |

Batch 1, 9-token prompt, 6 trials × 100 decode steps (10 warm-up each), CUDA events, trial order alternated.

**Where the overhead goes** (`python benchmarks/profile_decode.py`, [`results/profile.json`](benchmarks/results/profile.json)): GPU busy time per decode step is the same for both caches, 1.75 ms paged vs 1.79 ms `DynamicCache`, even though the paged step launches 27 more kernels (2 gathers, 24 per-layer scatters, one page-table upload). The difference is on the CPU: the paged update path spends **3.3 ms of host time per step** in Python and CuPy launch code, which matches the 3.35 ms P50 gap. At batch 1, GPT-2 decode is launch-bound for both caches, so the fix is fewer, cheaper launches, not faster kernels.

### Kernels and allocator

| | Measured |
|---|---|
| `gather_kv`, 2,048 tokens, all 144 layer-heads, random page order | **225 GB/s** (a plain device-to-device copy of the same bytes: 225 GB/s) |
| `scatter_kv_layer`, 2,048 tokens, one layer | 84 GB/s (1.5 KB strided writes per token) |
| Rust allocator through the Python bindings | 34.9M pages/s (500 cycles × 16 sequences × 8 pages, 0 leaks) |

GB/s counts bytes read plus written.

---

## Tests

**61 tests, all passing on the GPU** ([`benchmarks/results/tests.json`](benchmarks/results/tests.json), written by `python benchmarks/record_tests.py`):

| Suite | Tests | What it checks |
|---|---|---|
| `pageforge-rs` (`cargo test`) | 7 | Allocation round trip, FIFO reuse, all-or-nothing OOM, double free and invalid ids rejected without side effects, 20,000-step randomised run against a model with invariants checked every step |
| `tests/test_allocator.py` | 20 | The same allocator through PyO3: error types, page arithmetic, randomised workload, leak-free stress |
| `tests/test_kernels.py` | 15 | Bit-exact round trips, tokens land in the right page slot, untouched pages stay untouched, per-layer column isolation, 2-byte fallback path, bf16, CUDA side streams, DLPack zero-copy, input validation |
| `tests/test_cache.py` | 15 | GPT-2 logits identical to `DynamicCache` over 50 steps (4 prompts, page sizes 1/7/16/64), `model.generate()` token parity, interleaved sequences sharing a pool, free and reuse, pool exhaustion, batch > 1 rejected |
| `tests/test_cli.py` | 4 | CLI commands that need no GPU |

CI (GitHub Actions, no GPU) runs the Rust suite, clippy and rustfmt, builds the extension with maturin, runs the 24 CPU tests and reports the 30 GPU tests as skipped, then builds the dashboard. The GPU tests run locally.

---

## Setup

| Tool | Version |
|---|---|
| Python | 3.10+ |
| Rust | 1.80+ |
| GPU | Any CUDA GPU with a CuPy build for its driver; tested on RTX 4070 Laptop (sm_89) |

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]" --no-deps
pip install torch "transformers>=5.0" "cupy-cuda12x>=14.0" typer rich pydantic toml
cd pageforge-rs && maturin develop --release && cd ..   # builds _pageforge

pytest                                   # 54 Python tests (GPU tests skip without CUDA)
(cd pageforge-rs && cargo test)          # 7 Rust tests
python benchmarks/run_benchmarks.py      # writes benchmarks/results/results.json
```

Dashboard: `cd dashboard && npm install && npm run dev`. It reads the JSON files the benchmark scripts copy into `dashboard/lib/measured/`.

---

## Usage

### Single sequence

```python
from transformers import GPT2LMHeadModel, AutoTokenizer
from pageforge.cache import PagedKVCache
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().half().eval()
tok   = AutoTokenizer.from_pretrained("gpt2")
ids   = tok("The quick brown fox", return_tensors="pt").input_ids.cuda()

cache = PagedKVCache(n_pages=512, page_size=16, config=model.config)  # private pool

with torch.no_grad():
    out = model(ids, past_key_values=cache, use_cache=True)
    next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    for _ in range(49):
        out = model(next_tok, past_key_values=cache, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)

print(tok.decode(next_tok[0]))
cache.free()  # pages go back to the pool
```

### Shared pool: multiple sequences

```python
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from pageforge.cache import PagedKVCache
from pageforge.pool import PagedPool

model = GPT2LMHeadModel.from_pretrained("gpt2").cuda().half().eval()
tok   = AutoTokenizer.from_pretrained("gpt2")
pool  = PagedPool.for_model(model.config, n_pages=64, page_size=16)  # one pool, many sequences

def step(cache, ids):
    with torch.no_grad():
        return model(ids, past_key_values=cache, use_cache=True).logits[:, -1:].argmax(-1)

a, b = PagedKVCache(pool=pool), PagedKVCache(pool=pool)
next_a = step(a, tok("Paging works because", return_tensors="pt").input_ids.cuda())
next_b = step(b, tok("The allocator returns", return_tensors="pt").input_ids.cuda())
print(pool.used_pages(), "pages in use")   # one page each

a.free()                                   # a's page goes back to the free list
print(pool.free_pages(), "of", pool.n_pages, "pages free")
# When the pool is full, the forward pass raises MemoryError and allocates
# nothing, so a scheduler can wrap admission in try/except.
```

---

## CLI

```bash
# Check system readiness (GPU, CUDA, Rust extension, deps)
pageforge info

# Generate text using paged KV-cache
pageforge run "The quick brown" --model gpt2 --steps 50 --pages 512 --page-size 16

# Show/edit persistent config (~/.pageforge/config.toml)
pageforge config show
pageforge config set pool.n_pages 1024

# Pool capacity analysis and stress test
pageforge pool status              # capacity at current seq_len
pageforge pool status --seqs 128   # project for 128 concurrent sequences
pageforge pool stress --seqs 8 --cycles 100

# Benchmarks
pageforge bench vram       # VRAM vs decode step curves
pageforge bench latency    # P50/P99 vs HF DynamicCache
pageforge bench multi      # multi-sequence lifecycle
```

### Demo commands (no GPU or Rust required)

Three commands demonstrate the full system on any machine with only Python and PyTorch installed.

#### `pageforge demo`

Full-screen animated memory visualization. A 32x16 grid of 512 pages updates in real time as 8 color-coded sequences arrive, accumulate pages across decode steps, then free and reuse them. Live stats panel shows VRAM paged vs naive and pool utilization. Runs for ~10 seconds by default.

```bash
pageforge demo
pageforge demo --seqs 8 --steps 150 --delay 0.15   # slower, longer
```

The animation is a pure-Python simulation of the allocator: freed pages (gray) are immediately available to the next sequence.

#### `pageforge pool simulate`

Step-through table showing the physical page ids assigned to each sequence at key decode steps, across two consecutive batches, ending with a leak check.

```bash
pageforge pool simulate
pageforge pool simulate --seqs 4 --steps 60 --batches 2
```

Example output (abbreviated):

```
Batch 1  (seqs 0 - 3)
 Step | Seq 0       | Seq 1       | Seq 2       | Seq 3       | Used | Free | VRAM
    0 | [0]         | [1]         | [2]         | [3]         |    4 |  508 | 2.4 MB
   10 | [0, 4]      | [1, 5]      | [2, 6]      | [3, 7]      |    8 |  504 | 4.7 MB
   59 | [0, 4, 8..] | [1, 5, 9..] | [2, 6, 10.] | [3, 7, 11.] |   20 |  492 | 11.8 MB
Batch 1 freed: 20 pages returned -- pool: 512/512 free

Pool fully recovered: 512/512 pages free | Leaks: 0 | Peak VRAM: 11.8 MB vs naive 75.5 MB
```

#### `pageforge run --cpu`

Real GPT-2 text generation on CPU (no GPU needed) with a live panel showing how many pages PageForge would allocate at each decode step against a static 512-token reservation. The panel is page arithmetic, not a GPU measurement.

```bash
pageforge run "Memory management in operating systems involves allocating and" --cpu --steps 25
pageforge run "The research team announced that their new system achieved" --cpu --steps 30
pageforge run "It was the best of times, it was the worst of" --cpu --steps 25
```

A repetition penalty (default 1.3) is applied automatically to prevent greedy-decoding loops. Adjust with `--rep-penalty`:

```bash
pageforge run "The transformer architecture" --cpu --steps 30 --rep-penalty 1.5
```

Example VRAM output at step 25 (prompt length 9, page_size=16):

```
Cache              Tokens   Pages   VRAM       Usage
Naive pre-alloc      512      32   18.9 MB   ████████████████████████████
PageForge paged       34       3    1.8 MB   ██░░░░░░░░░░░░░░░░░░░░░░░░░░
Savings                             17.1 MB   (10.7x)
```

The naive allocator reserves 512 tokens of VRAM at sequence start regardless of actual length. PageForge allocates only the pages that contain real tokens, releasing the rest to the pool for other sequences.

---

## Roadmap

| Item | Why |
|---|---|
| Cut host-side dispatch | The profile shows the overhead is 3.3 ms of CPU launch time per step: cache CuPy views of the pool, fuse the 24 per-layer scatters into one launch, avoid the per-step page-table upload |
| CUDA Graph capture | Replay the decode step with near-zero launch cost |
| Attention that reads pages directly | Remove the per-step gather and `cat` |
| Batched sequences | One forward pass over many sequences with a block table per row |
| Prefix sharing, copy-on-write | Share prompt pages across sequences; fork block tables for beam search |

---

## Reference

Kwon, W. et al. (2023). **Efficient Memory Management for Large Language Model Serving with PagedAttention.** SOSP '23. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
