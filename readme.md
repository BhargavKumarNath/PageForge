# PageForge

![Rust](https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![arXiv](https://img.shields.io/badge/arXiv-2309.06180-b31b1b?style=for-the-badge)
![LLM](https://img.shields.io/badge/LLM-inference-lightgrey?style=for-the-badge)
![Throughput](https://img.shields.io/badge/throughput-scalable-blue?style=for-the-badge)
![Memory](https://img.shields.io/badge/VRAM-optimized-green?style=for-the-badge)
![KV Cache](https://img.shields.io/badge/KV--Cache-paged%20attention-orange?style=for-the-badge)
![Transformers](https://img.shields.io/badge/Transformers-FFCC00?style=for-the-badge&logo=huggingface&logoColor=black)

**[→ View Live Dashboard](https://page-forge-five.vercel.app/)**

**Paged KV-cache memory manager for LLM inference — Rust + CUDA + Python.**

PageForge reimplements the core memory innovation from [vLLM's PagedAttention](https://arxiv.org/abs/2309.06180) from scratch: instead of pre-allocating a fixed KV-cache tensor for `max_seq_len` tokens at sequence start, it allocates fixed-size *pages* on demand. Only tokens that exist consume VRAM. The result is **8-32× more concurrent sequences** on the same GPU, with near-zero fragmentation and O(1) page alloc/free.

```
GPT-2 124M · fp16 · RTX 4070 Laptop · decode step 50

  Naive pre-alloc (max 512)   603.98 MB for 32 seqs   53 seqs/GB
  Naive pre-alloc (max 1024)                           27 seqs/GB
  PageForge paged              75.50 MB for 32 seqs   424 seqs/GB   ← 8×
```

---

## Problem

Standard HuggingFace KV-cache allocates a full contiguous tensor per sequence at the first forward pass:

```
shape = (batch, n_layers, 2, n_heads, max_seq_len, d_head)
```

For GPT-2 with `max_seq_len=512`: each sequence reserves **18.9 MB** of K+V VRAM regardless of generation length. A request that generates 60 tokens holds a 512-token slot. At 32 concurrent sequences, 90%+ of allocated VRAM is idle before decode step 100. This is the primary GPU memory bottleneck in LLM serving.

```
KV bytes per token (GPT-2 124M):
  2 (K+V) × 12 layers × 12 heads × 64 d_head × 2 bytes (fp16) = 36,864 B/token

At seq_len=512:  18.87 MB/sequence
At seq_len=1024: 37.75 MB/sequence
At seq_len=59 (decode step 50):  2.36 MB/sequence with paged allocation
```

## Solution

PageForge maintains a fixed GPU memory pool divided into uniform pages (`page_size=16` tokens each). The Rust allocator tracks a free-list and a per-sequence block table. Pages are assigned on demand as new tokens are generated and returned instantly on `free()`.

```
Sequence starts:   alloc_for_seq(seq_id=0, n_pages=1)  → [page_3]
After 16 tokens:   alloc_for_seq(seq_id=0, n_pages=1)  → [page_3, page_7]
Sequence ends:     free_seq(seq_id=0)                  → pages [3, 7] back to free-list
Next sequence:     alloc_for_seq(seq_id=1, n_pages=1)  → [page_3]   ← reuse
```

Pages are non-contiguous in physical memory; CuPy CUDA kernels handle gather (scattered pages → contiguous buffer for attention) and scatter (new tokens → pages) at memory-bandwidth speeds (157-227 GB/s on sm_89).

---

## Architecture

```
Python scheduler
    │
    ▼
PagedPool (pageforge/pool.py)
    ├── PageForge [Rust via PyO3] ─── free-list VecDeque, BlockTable HashMap
    │       alloc_for_seq(seq_id, n)  →  [page_id, ...]   O(1)
    │       free_seq(seq_id)          →  [page_id, ...]   O(1)
    │
    └── pool_k, pool_v [CuPy GPU tensors]
            shape: (N_pages, page_size, n_layers×n_heads, d_head)  fp16
            ├── gather_kv(pool, page_ids, seq_len)  →  (seq_len, heads, d_head)
            └── scatter_kv_layer(kv, page_ids, pool, tok_offset, layer_offset)

Per-sequence PagedKVCache (pageforge/cache.py)
    ├── Subclasses HuggingFace DynamicCache — drop-in for transformers
    ├── _start_step():   2 gather kernel calls for all layers K and V
    └── _layer_update(): 1 scatter call + torch.cat(past, new)
```

### CUDA Kernels (`pageforge/kernels/kv_cache.py`)

All kernels use CuPy `RawKernel` (NVRTC) — no MSVC dependency, no PyTorch C++ extension build.

| Kernel | Operation | Use |
|--------|-----------|-----|
| `gather_kv` | sparse pages → contiguous `(seq_len, heads, d_head)` | prefill read |
| `scatter_kv` | full sequence → pages | prefill write |
| `scatter_kv_at_offset` | new tokens → pages at logical offset | decode write |
| `scatter_kv_layer` | one layer → combined pool (head-offset aware) | decode write, all-layer pool |

The combined-layer pool layout (`n_layers×n_heads` in one tensor) reduces gather calls per decode step from `2×n_layers` (24 for GPT-2) down to 2 — one for all K, one for all V.

### Rust Core (`pageforge-rs/`)

```rust
// O(1) allocator via VecDeque free-list
pub struct PageAllocator { free_list: VecDeque<u32>, ... }
impl PageAllocator {
    pub fn alloc(&mut self, n: usize) -> PyResult<Vec<u32>>  // O(1)
    pub fn free(&mut self, ids: Vec<u32>)                    // O(n)
    pub fn free_pages(&self) -> usize
}

// Per-sequence page tracking
pub struct BlockTable { table: HashMap<u64, Vec<u32>> }
impl BlockTable {
    pub fn append(&mut self, seq_id: u64, page_ids: Vec<u32>)
    pub fn evict(&mut self, seq_id: u64) -> Option<Vec<u32>>
    pub fn get(&self, seq_id: u64) -> Option<&Vec<u32>>
}

// Unified facade exposed to Python
#[pyclass] pub struct PageForge { allocator: PageAllocator, block_table: BlockTable }
```

### DLPack Bridge

`pageforge/bridge.py`: Zero-copy tensor exchange between PyTorch (CUDA) and CuPy via the [DLPack](https://dmlpack.org) protocol. No host-side memcopy, no intermediate buffers.

---

## Performance

### VRAM vs. Naive Pre-allocation

32 concurrent sequences, `page_size=16`, `n_pages=512`:

| Decode step | Seq len | Naive (max=512) | PageForge | Savings |
|-------------|---------|-----------------|-----------|---------|
| 0           | 9       | 603.98 MB       | 18.87 MB  | 32×     |
| 10          | 19      | 603.98 MB       | 37.75 MB  | 16×     |
| 25          | 34      | 603.98 MB       | 75.50 MB  | 8×      |
| 50          | 59      | 603.98 MB       | 75.50 MB  | 8×      |
| 100         | 109     | 603.98 MB       | 150.99 MB | 4×      |

### Concurrent Sequences per 1 GB KV-Cache VRAM (step 50)

| Approach | Seqs / GB |
|----------|-----------|
| Naive (max=512) | 53 |
| Naive (max=1024) | 27 |
| **PageForge paged** | **424** |

In a 512-page pool (~302 MB reserved K+V): supports 256 concurrent sequences at step 10, 128 at step 50, 73 at step 100.

### Decode Latency (single sequence, 50 iterations, 5 warmup)

| Cache | P50 | P99 |
|-------|-----|-----|
| HF DynamicCache | 7.5 ms | 10.3 ms |
| PageForge paged | 10.0 ms | 11.9 ms |
| **Overhead** | **+33%** | **+16% P99** |

P50 overhead comes from 24 Python scatter kernel dispatches per decode step (1 scatter per layer × 12 layers × K+V). Root cause: non-contiguous page layout requires 24 `scatter_kv_layer` calls + 24 `torch.cat` ops. A fused scatter-attention kernel would eliminate these dispatches — target ≤8 ms P50.

### Kernel Bandwidth

`gather_kv` / `scatter_kv_layer`: **157-227 GB/s** on RTX 4070 Laptop (theoretical peak: ~250 GB/s). Purely memory-bound — no arithmetic bottleneck.

### Allocator Throughput

**1.5M pages/sec** (stress-tested: 500 alloc + free cycles, 16 sequences/cycle, 0 leaks, 0 OOM errors).

---

## Tests

**56 Python tests + 6 Rust unit tests — all passing.**

| File | Tests | Coverage |
|------|-------|----------|
| `test_week1_milestone.py` | 19 | PageAllocator, BlockTable, PageForge (Rust + PyO3 bindings), error paths |
| `test_week2_kernels.py` | 10 | gather/scatter round-trip (bit-exact), non-contiguous pages, isolation, GB/s |
| `test_week3_integration.py` | 7 | GPT-2 logit parity across 50 decode steps (max diff = 0.0, bit-exact) |
| `test_week4_benchmark.py` | 20 | VRAM accounting, efficiency formulas, latency (non-zero, finite, bounded) |
| `pageforge-rs` (`cargo test`) | 6 | Allocator invariants, BlockTable consistency, OOM, double-free |

**Key integration result:** `max(abs(hf_logits - paged_logits))` over 50 decode steps across all test prompts is exactly `0.0`. PagedKVCache is bit-for-bit identical to HF DynamicCache for GPT-2.

---

## Setup

### Requirements

| Tool | Version |
|------|---------|
| Python | 3.12+ |
| Rust | 1.80+ (`rustup install stable`) |
| CUDA | 12.x (nvcc on PATH) |
| GPU | Any CUDA-capable; tested on RTX 4070 Laptop (sm_89) |

**Windows note:** `torch.utils.cpp_extension` is broken on MSVC 14.41 + PyTorch 2.11. All GPU code uses CuPy RawKernel (NVRTC), which compiles independently — this is by design.

### Install

```bash
# 1. Build the Rust extension
cd pageforge-rs
maturin develop --release
cd ..

# 2. Install Python package
pip install -e .

# 3. Verify
python -c "from _pageforge import PageForge; pf = PageForge(512, 16); print(pf)"

# 4. Run tests
pytest tests/ -v

# 5. Run benchmarks
python -m benchmarks.benchmark_week4
```

### Dependencies

```
# Python (pyproject.toml)
torch>=2.0
transformers>=5.0
cupy-cuda12x>=14.0
typer>=0.12
rich>=13.0
pydantic>=2.0
toml>=0.10

# Rust (Cargo.toml)
pyo3 = "0.22"  # features: ["extension-module"]
```

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

cache = PagedKVCache(n_pages=512, page_size=16)

with torch.no_grad():
    out = model(ids, past_key_values=cache, use_cache=True)
    next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    for _ in range(49):
        out = model(next_tok, past_key_values=cache, use_cache=True)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)

print(tok.decode(next_tok[0]))
cache.free()  # O(1) — pages returned to pool
```

### Shared pool — multiple sequences

```python
from pageforge.pool import PagedPool
from pageforge.cache import PagedKVCache

pool = PagedPool(n_pages=2048, page_size=16)

cache_a = PagedKVCache(pool=pool, seq_id=0)
cache_b = PagedKVCache(pool=pool, seq_id=1)

# ... run inference for A ...
cache_a.free()  # pages returned to pool immediately

# C reuses A's physical pages — no fragmentation
cache_c = PagedKVCache(pool=pool, seq_id=2)
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
pageforge bench vram       # VRAM vs decode step curves (saves PNG)
pageforge bench latency    # P50/P99 vs HF DynamicCache
pageforge bench multi      # multi-sequence lifecycle
```

---

## Verified Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA RTX 4070 Laptop, Ada Lovelace, sm_89, 8 GB VRAM |
| CUDA runtime | 12.8 |
| nvcc | 12.6 |
| Rust | 1.90.0 stable |
| PyTorch | 2.11 |
| CuPy | 14.1.1 |
| transformers | 5.9.0 |
| Python | 3.12.7 |

---

## Roadmap

| Item | Notes |
|------|-------|
| Fused scatter-attention kernel | Eliminate 24 Python dispatch hops per decode step; est. -50% P50 overhead |
| Prefix KV sharing | Shared physical pages for common prompt prefixes across sequences |
| Beam search | Fork BlockTable entries on beam expansion; copy-on-write pages |
| CUDA Graph capture | Capture decode loop; near-zero kernel launch overhead |
| CPU eviction | Page-out cold sequences to host DRAM; re-page on reschedule |
| Batched attention | Single `forward()` over N sequences simultaneously |

---

## Reference

Kwon, W. et al. (2023). **Efficient Memory Management for Large Language Model Serving with PagedAttention.** SOSP '23. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)
