"""PagedKVCache: a transformers ``DynamicCache`` backed by pool pages.

Drop-in for ``past_key_values``. One instance holds one sequence (batch size
1); many instances share one ``PagedPool``.

Per forward pass (prefill or one decode step) with ``T`` new tokens:

* layer 0's ``update`` grows the sequence's block table to ``len + T`` tokens
  (Rust allocator) and, if there is history, gathers every layer's past K and
  V out of the pages: two kernel calls in total, one for K and one for V;
* each layer's ``update`` scatters its ``T`` new tokens into the pages (two
  kernel calls) and returns ``cat(past, new)`` exactly as ``DynamicCache``
  would, so attention sees identical tensors.

Between steps the only KV storage the sequence holds is its pages. The
gathered history is a per-step scratch tensor, released at the next step.
"""

from __future__ import annotations

import torch
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer

from pageforge.kernels import gather_kv, scatter_kv_layer, scatter_kv_layers_fused
from pageforge.pool import PagedPool


class PagedLayer(DynamicLayer):
    """One attention layer's view of a PagedKVCache."""

    is_croppable = False

    def __init__(self, owner: PagedKVCache, layer_idx: int) -> None:
        super().__init__()
        self._owner = owner
        self._layer_idx = layer_idx
        self._length = 0

    def lazy_initialization(self, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, *args, **kwargs):
        # key_states, value_states: (1, n_heads, T, head_dim)
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        keys, values = self._owner._layer_update(self._layer_idx, self._length, key_states, value_states)
        self._length += key_states.shape[-2]
        return keys, values

    def get_seq_length(self) -> int:
        return self._length

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        return self._length + query_length, 0

    def crop(self, tokens_to_remove: int) -> None:  # pragma: no cover - explicit unsupported path
        raise NotImplementedError("PagedKVCache does not support crop()")

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:  # pragma: no cover
        raise NotImplementedError("PagedKVCache does not support beam search")

    def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError("use PagedKVCache.free() and a new cache instead")


class PagedKVCache(DynamicCache):
    """KV-cache for one sequence whose storage is pages in a shared pool.

    Either pass an existing ``pool`` (shared across sequences) or ``n_pages``
    plus a model ``config`` to create a private one.
    """

    def __init__(
        self,
        pool: PagedPool | None = None,
        seq_id: int | None = None,
        *,
        n_pages: int | None = None,
        page_size: int = 16,
        config=None,
        fused_scatter: bool = False,
    ) -> None:
        if pool is None:
            if n_pages is None or config is None:
                raise ValueError("pass a PagedPool, or n_pages and the model config")
            pool = PagedPool.for_model(config, n_pages, page_size)
        self.pool = pool
        self.seq_id = pool.new_seq_id() if seq_id is None else seq_id
        self._freed = False
        self._past_k: torch.Tensor | None = None  # (len, n_cols, head_dim), valid for the current step
        self._past_v: torch.Tensor | None = None
        self._step_pages: torch.Tensor | None = None
        # fused_scatter=True defers the per-layer scatters and issues one launch after the last layer.
        self.fused_scatter = fused_scatter
        self._pending: list[tuple[torch.Tensor, torch.Tensor]] = []
        layers = [PagedLayer(self, i) for i in range(pool.geometry.n_layers)]
        Cache.__init__(self, layers=layers)

    # -- core ---------------------------------------------------------------

    def _layer_update(
        self, layer_idx: int, past_len: int, key_states: torch.Tensor, value_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._freed:
            raise RuntimeError(f"sequence {self.seq_id} was freed; create a new cache")
        g = self.pool.geometry
        batch, n_heads, n_new, head_dim = key_states.shape
        if batch != 1:
            raise ValueError(f"PagedKVCache holds one sequence; got batch size {batch}")
        if (n_heads, head_dim) != (g.n_heads, g.head_dim) or key_states.dtype != self.pool.dtype:
            raise ValueError(
                f"key_states {tuple(key_states.shape)} {key_states.dtype} do not match the pool "
                f"({g.n_heads} heads x {g.head_dim}, {self.pool.dtype})"
            )

        if layer_idx == 0:
            # Start of a forward pass: reserve pages for the new tokens, then
            # pull every layer's history out of the pages in two kernel calls.
            self.pool.ensure_capacity(self.seq_id, past_len + n_new)
            self._step_pages = torch.tensor(self.pool.page_ids(self.seq_id), dtype=torch.int32, device=self.pool.device)
            if past_len > 0:
                self._past_k = gather_kv(self.pool.pool_k, self._step_pages, past_len)
                self._past_v = gather_kv(self.pool.pool_v, self._step_pages, past_len)
            else:
                self._past_k = self._past_v = None
        assert self._step_pages is not None

        head_offset = layer_idx * g.n_heads
        if self.fused_scatter:
            self._pending.append((key_states, value_states))
            if layer_idx == g.n_layers - 1:
                self._flush_pending(past_len)
        else:
            # (1, H, T, D) -> (T, H, D), the pool's row layout for one layer.
            new_k = key_states[0].transpose(0, 1).contiguous()
            new_v = value_states[0].transpose(0, 1).contiguous()
            scatter_kv_layer(new_k, self._step_pages, self.pool.pool_k, past_len, head_offset)
            scatter_kv_layer(new_v, self._step_pages, self.pool.pool_v, past_len, head_offset)

        if past_len == 0:
            return key_states, value_states

        assert self._past_k is not None and self._past_v is not None
        cols = slice(head_offset, head_offset + g.n_heads)
        # (len, H, D) -> (1, H, len, D)
        past_k = self._past_k[:, cols].permute(1, 0, 2).unsqueeze(0)
        past_v = self._past_v[:, cols].permute(1, 0, 2).unsqueeze(0)
        keys = torch.cat([past_k, key_states], dim=-2)
        values = torch.cat([past_v, value_states], dim=-2)

        if layer_idx == g.n_layers - 1:
            self._past_k = self._past_v = None  # release the per-step scratch
        return keys, values

    def _flush_pending(self, past_len: int) -> None:
        """Scatter all layers' new tokens at once; fall back to per-layer scatters if the fused kernel declines."""
        g = self.pool.geometry
        pending, self._pending = self._pending, []
        keys = [k for k, _ in pending]
        values = [v for _, v in pending]
        assert self._step_pages is not None
        if len(pending) == g.n_layers and scatter_kv_layers_fused(
            keys, values, self._step_pages, self.pool.pool_k, self.pool.pool_v, past_len
        ):
            return
        for layer_idx, (k, v) in enumerate(pending):
            scatter_kv_layer(k[0].transpose(0, 1).contiguous(), self._step_pages, self.pool.pool_k, past_len, layer_idx * g.n_heads)
            scatter_kv_layer(v[0].transpose(0, 1).contiguous(), self._step_pages, self.pool.pool_v, past_len, layer_idx * g.n_heads)

    # -- lifecycle ------------------------------------------------------------

    def free(self) -> list[int]:
        """Return this sequence's pages to the pool. Idempotent."""
        if self._freed:
            return []
        self._freed = True
        self._past_k = self._past_v = self._step_pages = None
        self._pending = []
        try:
            return self.pool.free_seq(self.seq_id)
        except KeyError:  # never allocated (no forward pass ran)
            return []

    @property
    def page_ids(self) -> list[int]:
        return self.pool.page_ids(self.seq_id)

    @property
    def batch_size(self) -> int:
        return 1
