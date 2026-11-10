"""Paged KV-cache gather/scatter kernels (CuPy RawKernel, compiled by NVRTC).

Pool layout, shared by K and V (one tensor each):

    pool: (n_pages, page_size, n_cols, head_dim)   n_cols = n_layers * n_heads

so one *token row* (all layers, all heads of one token) is contiguous. Logical
token ``t`` of a sequence lives at ``pool[page_ids[t // page_size], t % page_size]``.

Every operation here is the same thing: copy rows between a dense
``(n_tokens, width)`` tensor and the paged pool, where ``width`` may be a
slice of the pool row starting at ``col_offset`` (one layer's heads). A
single kernel does it, in either direction, using 16-byte vector loads when
the row width and offset allow and 2-byte loads otherwise.

All functions take and return torch tensors; the CuPy views are zero-copy
(DLPack) and the kernels run on torch's current stream, so no extra
synchronisation is needed.
"""

from __future__ import annotations

import functools

import cupy as cp
import numpy as np
import torch

_SOURCE = r"""
extern "C" __global__
void paged_copy_{T}(
    const {T}* __restrict__ src,
    {T}* __restrict__ dst,
    const int* __restrict__ page_ids,
    const int n_tokens,      // rows to copy
    const int tok_offset,    // logical position of row 0 within the sequence
    const int page_size,     // tokens per page
    const int width,         // elements of type {T} per copied row
    const int pool_row,      // elements of type {T} per full pool row
    const int col_offset,    // where the copied slice starts in a pool row
    const int to_pool)       // 1: dense -> pool (scatter), 0: pool -> dense (gather)
{
    const long long total = (long long)n_tokens * width;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < total;
         i += (long long)gridDim.x * blockDim.x) {
        const int t = (int)(i / width);
        const int c = (int)(i - (long long)t * width);
        const int logical = tok_offset + t;
        const long long page = page_ids[logical / page_size];
        const long long pool_idx =
            (page * page_size + (logical % page_size)) * (long long)pool_row + col_offset + c;
        if (to_pool) dst[pool_idx] = src[i];
        else         dst[i] = src[pool_idx];
    }
}
"""

_THREADS = 256


@functools.cache
def _kernel(vector: bool) -> cp.RawKernel:
    # uint4 = 16 bytes = 8 fp16 values per load; ushort = one fp16/bf16 value.
    t = "uint4" if vector else "unsigned short"
    name = f"paged_copy_{t.replace(' ', '_')}"
    src = _SOURCE.replace("{T}", t).replace(f"paged_copy_{t}", name)
    return cp.RawKernel(src, name)


def _as_cupy(x: torch.Tensor) -> cp.ndarray:
    # The kernel moves raw 16-bit words, so hand CuPy an int16 view of any
    # 16-bit float. CuPy cannot import bfloat16 over DLPack; int16 always works.
    x = x.detach()
    if x.is_floating_point() and x.element_size() == 2:
        x = x.view(torch.int16)
    return cp.from_dlpack(x)


def _check_pool(pool: torch.Tensor) -> None:
    if pool.dim() != 4 or not pool.is_cuda or not pool.is_contiguous():
        raise ValueError("pool must be a contiguous CUDA tensor (n_pages, page_size, n_cols, head_dim)")
    if pool.element_size() != 2:
        raise ValueError(f"pool dtype must be 16-bit (fp16/bf16), got {pool.dtype}")


def _page_table(page_ids: torch.Tensor | list[int], device: torch.device) -> torch.Tensor:
    if isinstance(page_ids, torch.Tensor):
        return page_ids.to(device=device, dtype=torch.int32).contiguous()
    return torch.tensor(page_ids, dtype=torch.int32, device=device)


def _launch(
    dense: torch.Tensor,
    pool: torch.Tensor,
    page_ids: torch.Tensor | list[int],
    tok_offset: int,
    col_offset: int,
    to_pool: bool,
) -> None:
    """Copy ``dense`` (n_tokens, width) to/from the pool. Widths in 2-byte elements."""
    _check_pool(pool)
    if dense.dtype != pool.dtype or dense.device != pool.device or not dense.is_contiguous():
        raise ValueError("dense tensor must be contiguous, on the pool's device, with the pool's dtype")
    n_pages, page_size, n_cols, head_dim = pool.shape
    pool_row = n_cols * head_dim
    n_tokens = dense.shape[0]
    width = dense[0].numel() if n_tokens else 0
    if col_offset < 0 or col_offset + width > pool_row:
        raise ValueError(f"slice [{col_offset}, {col_offset + width}) exceeds pool row of {pool_row}")
    table = _page_table(page_ids, pool.device)
    if n_tokens == 0:
        return
    if (tok_offset + n_tokens + page_size - 1) // page_size > table.numel():
        raise ValueError(
            f"{table.numel()} pages cannot hold tokens [{tok_offset}, {tok_offset + n_tokens}) at page_size={page_size}"
        )

    # 16-byte vectors when every row and offset is a multiple of 8 fp16 values.
    vector = width % 8 == 0 and pool_row % 8 == 0 and col_offset % 8 == 0
    scale = 8 if vector else 1
    total = n_tokens * width // scale
    blocks = min((total + _THREADS - 1) // _THREADS, 65535)

    src, dst = (dense, pool) if to_pool else (pool, dense)
    stream = cp.cuda.Stream.from_external(torch.cuda.current_stream(pool.device))
    with cp.cuda.Device(pool.device.index), stream:
        _kernel(vector)(
            (blocks,),
            (_THREADS,),
            (
                _as_cupy(src),
                _as_cupy(dst),
                _as_cupy(table),
                cp.int32(n_tokens),
                cp.int32(tok_offset),
                cp.int32(page_size),
                cp.int32(width // scale),
                cp.int32(pool_row // scale),
                cp.int32(col_offset // scale),
                cp.int32(1 if to_pool else 0),
            ),
        )


def gather_kv(pool: torch.Tensor, page_ids: torch.Tensor | list[int], seq_len: int) -> torch.Tensor:
    """Pages -> contiguous ``(seq_len, n_cols, head_dim)`` (all layers and heads)."""
    _check_pool(pool)
    out = torch.empty((seq_len, *pool.shape[2:]), dtype=pool.dtype, device=pool.device)
    _launch(out, pool, page_ids, tok_offset=0, col_offset=0, to_pool=False)
    return out


def scatter_kv(kv: torch.Tensor, page_ids: torch.Tensor | list[int], pool: torch.Tensor) -> None:
    """Full-width ``(n_tokens, n_cols, head_dim)`` -> pages, starting at token 0."""
    scatter_kv_at_offset(kv, page_ids, pool, tok_offset=0)


def scatter_kv_at_offset(
    kv: torch.Tensor, page_ids: torch.Tensor | list[int], pool: torch.Tensor, tok_offset: int
) -> None:
    """Full-width ``(n_tokens, n_cols, head_dim)`` -> pages at logical ``tok_offset``."""
    _check_pool(pool)
    if tuple(kv.shape[1:]) != tuple(pool.shape[2:]):
        raise ValueError(f"kv rows {tuple(kv.shape[1:])} do not match pool rows {tuple(pool.shape[2:])}")
    _launch(kv, pool, page_ids, tok_offset=tok_offset, col_offset=0, to_pool=True)


def scatter_kv_layer(
    kv: torch.Tensor,
    page_ids: torch.Tensor | list[int],
    pool: torch.Tensor,
    tok_offset: int,
    head_offset: int,
) -> None:
    """One layer's ``(n_tokens, n_heads, head_dim)`` -> the pool's heads
    ``[head_offset, head_offset + n_heads)`` at logical ``tok_offset``."""
    _check_pool(pool)
    if kv.dim() != 3 or kv.shape[2] != pool.shape[3]:
        raise ValueError(f"kv must be (n_tokens, n_heads, {pool.shape[3]}), got {tuple(kv.shape)}")
    _launch(kv, pool, page_ids, tok_offset=tok_offset, col_offset=head_offset * pool.shape[3], to_pool=True)


# ---------------------------------------------------------------------------
# Fused all-layer scatter
# ---------------------------------------------------------------------------
# One decode step normally scatters each layer's new K and V separately: 2 launches
# plus 2 transpose-copies per layer. The fused kernel takes every layer's
# (1, H, T, D) key/value tensor by pointer (passed by value in a struct, so no
# host-to-device upload), reads them with their own strides and writes all of
# them into the pool in a single launch for K and V together.

MAX_FUSED_LAYERS = 64

_FUSED_SOURCE = r"""
struct Ptrs { unsigned long long k[64]; unsigned long long v[64]; };
extern "C" __global__
void paged_scatter_layers_{T}(
    Ptrs src,
    {T}* __restrict__ pool_k,
    {T}* __restrict__ pool_v,
    const int* __restrict__ page_ids,
    const int n_layers, const int n_new, const int n_heads, const int head_vecs,
    const int tok_offset, const int page_size, const int pool_row,
    const long long stride_h, const long long stride_t)   // source strides in {T} units
{
    const int which = blockIdx.y;                       // 0: K, 1: V
    const long long total = (long long)n_layers * n_new * n_heads * head_vecs;
    {T}* pool = which ? pool_v : pool_k;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        long long r = i;
        const int d = (int)(r % head_vecs); r /= head_vecs;
        const int h = (int)(r % n_heads);   r /= n_heads;
        const int t = (int)(r % n_new);     r /= n_new;
        const int layer = (int)r;
        const {T}* base = (const {T}*)(which ? src.v[layer] : src.k[layer]);
        const int logical = tok_offset + t;
        const long long page = page_ids[logical / page_size];
        const long long dst = (page * page_size + (logical % page_size)) * (long long)pool_row
                              + ((long long)layer * n_heads + h) * head_vecs + d;
        pool[dst] = base[h * stride_h + t * stride_t + d];
    }
}
"""


@functools.cache
def _fused_kernel(vector: bool) -> cp.RawKernel:
    t = "uint4" if vector else "unsigned short"
    name = f"paged_scatter_layers_{t.replace(' ', '_')}"
    src = _FUSED_SOURCE.replace("{T}", t).replace(f"paged_scatter_layers_{t}", name)
    return cp.RawKernel(src, name)


_PTRS_DTYPE = np.dtype([("k", np.uint64, (MAX_FUSED_LAYERS,)), ("v", np.uint64, (MAX_FUSED_LAYERS,))])


def scatter_kv_layers_fused(
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
    page_ids: torch.Tensor | list[int],
    pool_k: torch.Tensor,
    pool_v: torch.Tensor,
    tok_offset: int,
) -> bool:
    """Write every layer's new tokens into the pool with one launch.

    ``keys[l]`` / ``values[l]`` are layer ``l``'s ``(1, n_heads, T, head_dim)`` tensors, in layer order, all with
    the same strides and a contiguous last dimension. Returns False (and does nothing) if those conditions do
    not hold, so the caller can fall back to the per-layer path.
    """
    _check_pool(pool_k)
    n_layers = len(keys)
    if n_layers == 0 or n_layers > MAX_FUSED_LAYERS or len(values) != n_layers:
        return False
    k0 = keys[0]
    if k0.dim() != 4 or k0.shape[0] != 1:
        return False
    _, n_heads, n_new, head_dim = k0.shape
    n_pages, page_size, n_cols, pool_head_dim = pool_k.shape
    if head_dim != pool_head_dim or n_cols != n_layers * n_heads:
        return False
    for t in (*keys, *values):
        if t.shape != k0.shape or t.stride() != k0.stride() or t.stride(3) != 1 or t.dtype != pool_k.dtype or not t.is_cuda:
            return False
    stride_h, stride_t = k0.stride(1), k0.stride(2)
    vector = (
        head_dim % 8 == 0 and stride_h % 8 == 0 and stride_t % 8 == 0
        and all(t.data_ptr() % 16 == 0 for t in (*keys, *values))
    )
    scale = 8 if vector else 1
    table = _page_table(page_ids, pool_k.device)
    if (tok_offset + n_new + page_size - 1) // page_size > table.numel():
        raise ValueError("page table too small for the tokens being scattered")
    ptrs = np.zeros((), dtype=_PTRS_DTYPE)
    ptrs["k"][:n_layers] = [t.data_ptr() for t in keys]
    ptrs["v"][:n_layers] = [t.data_ptr() for t in values]
    head_vecs = head_dim // scale
    total = n_layers * n_new * n_heads * head_vecs
    blocks = min((total + _THREADS - 1) // _THREADS, 65535)
    stream = cp.cuda.Stream.from_external(torch.cuda.current_stream(pool_k.device))
    with cp.cuda.Device(pool_k.device.index), stream:
        _fused_kernel(vector)(
            (blocks, 2),
            (_THREADS,),
            (
                ptrs,
                _as_cupy(pool_k),
                _as_cupy(pool_v),
                _as_cupy(table),
                cp.int32(n_layers),
                cp.int32(n_new),
                cp.int32(n_heads),
                cp.int32(head_vecs),
                cp.int32(tok_offset),
                cp.int32(page_size),
                cp.int32(n_cols * head_dim // scale),
                cp.int64(stride_h // scale),
                cp.int64(stride_t // scale),
            ),
        )
    return True
