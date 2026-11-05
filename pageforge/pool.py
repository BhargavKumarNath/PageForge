"""A shared pool of KV-cache pages on the GPU, managed by the Rust allocator.

One pool serves many sequences. Each sequence owns a list of page ids (its
block table, kept in Rust); its tokens live in those pages, which need not be
contiguous or in order.

Tensors, one for K and one for V:

    pool_k, pool_v: (n_pages, page_size, n_layers * n_heads, head_dim)

All layers of a token sit in one row, so a decode step gathers every layer's
history with one kernel call per tensor instead of one per layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

try:
    import _pageforge
except ImportError as exc:  # pragma: no cover - exercised only without the extension
    raise ImportError(
        "Rust extension _pageforge is not built. Run: cd pageforge-rs && maturin develop --release"
    ) from exc


@dataclass(frozen=True)
class KVGeometry:
    """Attention shape of the model the pool serves."""

    n_layers: int
    n_heads: int
    head_dim: int

    @classmethod
    def from_config(cls, config) -> KVGeometry:
        """Read geometry from a transformers config (GPT-2 or Llama-style names)."""
        text = config.get_text_config(decoder=True) if hasattr(config, "get_text_config") else config
        n_layers = getattr(text, "num_hidden_layers", None) or text.n_layer
        n_query_heads = getattr(text, "num_attention_heads", None) or text.n_head
        # Grouped-query attention caches only the KV heads.
        n_kv_heads = getattr(text, "num_key_value_heads", None) or n_query_heads
        hidden = getattr(text, "hidden_size", None) or text.n_embd
        head_dim = getattr(text, "head_dim", None) or hidden // n_query_heads
        return cls(int(n_layers), int(n_kv_heads), int(head_dim))

    @property
    def n_cols(self) -> int:
        return self.n_layers * self.n_heads


GPT2_SMALL = KVGeometry(n_layers=12, n_heads=12, head_dim=64)


class PagedPool:
    """Fixed-size page pool plus allocator. Memory is reserved once, up front."""

    def __init__(
        self,
        n_pages: int,
        page_size: int = 16,
        geometry: KVGeometry = GPT2_SMALL,
        dtype: torch.dtype = torch.float16,
        device: torch.device | str = "cuda",
    ) -> None:
        self.geometry = geometry
        self.page_size = page_size
        self.dtype = dtype
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.allocator = _pageforge.PageForge(n_pages, page_size)
        shape = (n_pages, page_size, geometry.n_cols, geometry.head_dim)
        self.pool_k = torch.empty(shape, dtype=dtype, device=self.device)
        self.pool_v = torch.empty(shape, dtype=dtype, device=self.device)
        self._next_seq_id = 0

    @classmethod
    def for_model(cls, config, n_pages: int, page_size: int = 16, **kwargs) -> PagedPool:
        return cls(n_pages, page_size, KVGeometry.from_config(config), **kwargs)

    # -- allocation -------------------------------------------------------

    def new_seq_id(self) -> int:
        seq_id = self._next_seq_id
        self._next_seq_id += 1
        return seq_id

    def ensure_capacity(self, seq_id: int, n_tokens: int) -> list[int]:
        """Grow ``seq_id`` to hold ``n_tokens``. Raises MemoryError if the pool is full."""
        return self.allocator.ensure_capacity(seq_id, n_tokens)

    def free_seq(self, seq_id: int) -> list[int]:
        return self.allocator.free_seq(seq_id)

    def page_ids(self, seq_id: int) -> list[int]:
        return self.allocator.pages_for(seq_id)

    # -- accounting -------------------------------------------------------

    @property
    def n_pages(self) -> int:
        return self.allocator.total_pages()

    def free_pages(self) -> int:
        return self.allocator.free_pages()

    def used_pages(self) -> int:
        return self.allocator.used_pages()

    @property
    def bytes_per_page(self) -> int:
        """K + V bytes for one page across all layers."""
        g = self.geometry
        return 2 * self.page_size * g.n_cols * g.head_dim * torch.finfo(self.dtype).bits // 8

    @property
    def reserved_bytes(self) -> int:
        return self.pool_k.numel() * self.pool_k.element_size() * 2

    def used_bytes(self) -> int:
        """Bytes of the pool held by live sequences."""
        return self.used_pages() * self.bytes_per_page

    def __repr__(self) -> str:
        return (
            f"PagedPool(n_pages={self.n_pages}, page_size={self.page_size}, {self.geometry}, "
            f"free={self.free_pages()}, reserved={self.reserved_bytes / 1e6:.1f} MB)"
        )
