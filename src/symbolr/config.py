"""
src/symbolr/config.py — Single source of truth for SymboLR configuration.

All numeric defaults are explicitly aligned with the Rust core's compiled defaults.
If you change a value here, verify it matches the corresponding Rust constant:
  - crossover_rate, mutation_rate → rust_core/src/evolution.rs
  - size_bins, com_bins, smoothness_bins → rust_core/src/archive.rs
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


# Runtime detection
def _detect_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _detect_rust() -> bool:
    try:
        import symbolr_rust  # noqa: F401
        return True
    except ImportError:
        return False


# Config dataclass
@dataclass
class SymboLRConfig:
    """
    Immutable configuration snapshot for one SymboLR run.

    Instantiate directly or load from YAML via SymboLRConfig.from_yaml(path).
    """

    # Runtime detection (populated on instantiation)
    torch_available: bool = field(default_factory=_detect_torch)
    cuda_available:  bool = field(default_factory=_detect_cuda)
    rust_available:  bool = field(default_factory=_detect_rust)

    # Evolution — must match rust_core/src/evolution.rs
    max_generations: int   = 50
    pop_size:        int   = 50
    seed:            int   = 42
    crossover_rate:  float = 0.20  # evolution.rs default
    mutation_rate:   float = 0.70  # evolution.rs default

    # MAP-Elites archive — must match rust_core/src/archive.rs
    size_bins:       int = 30   # archive.rs default
    com_bins:        int = 20   # archive.rs default
    smoothness_bins: int = 10   # archive.rs default

    # Evaluator
    evaluator:   str = "synthetic"  # "synthetic" | "cuda_batch" | "gradient_aware"
    time_steps:  int = 100

    # Synthetic evaluator parameters
    synth_n_dims:        int   = 5
    synth_n_evaluations: int   = 3
    synth_noise_scale:   float = 0.02
    synth_initial_loss:  float = 2.3

    # Gradient-aware evaluator (Phase 3)
    grad_eval_n_steps:          int   = 200   # training steps per formula
    grad_eval_batch_size:       int   = 128   # mini-batch size for proxy task
    grad_eval_base_lr:          float = 0.1   # LR during warmup window
    grad_eval_warmup_fraction:  float = 0.10  # fraction of steps for warmup
    grad_eval_halving_fraction: float = 0.50  # fraction of remaining steps for Phase 1

    # Logging
    log_dir: str = "research_journal/experiments"

    @property
    def device(self) -> Any:
        if not self.torch_available:
            return "cpu"
        import torch
        return torch.device("cuda" if self.cuda_available else "cpu")

    @property
    def is_gpu(self) -> bool:
        return self.cuda_available

    # Serialization
    def to_dict(self) -> dict:
        """Return a plain dict of config values (excludes runtime detection fields)."""
        return {
            "max_generations": self.max_generations,
            "pop_size":        self.pop_size,
            "seed":            self.seed,
            "crossover_rate":  self.crossover_rate,
            "mutation_rate":   self.mutation_rate,
            "size_bins":       self.size_bins,
            "com_bins":        self.com_bins,
            "smoothness_bins": self.smoothness_bins,
            "evaluator":       self.evaluator,
            "time_steps":      self.time_steps,
            "grad_eval_n_steps":          self.grad_eval_n_steps,
            "grad_eval_batch_size":       self.grad_eval_batch_size,
            "grad_eval_base_lr":          self.grad_eval_base_lr,
            "grad_eval_warmup_fraction":  self.grad_eval_warmup_fraction,
            "grad_eval_halving_fraction": self.grad_eval_halving_fraction,
        }

    @classmethod
    def from_yaml(cls, path: str) -> "SymboLRConfig":
        """Load config from a YAML file, overriding dataclass defaults."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        cfg = cls()
        cfg.update(**data)
        return cfg

    def update(self, **kwargs) -> None:
        """Apply keyword overrides to mutable fields."""
        for k, v in kwargs.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, v)


# Singleton
_config: Optional[SymboLRConfig] = None


def get_config() -> SymboLRConfig:
    """Return the global config singleton (created with defaults on first call)."""
    global _config
    if _config is None:
        _config = SymboLRConfig()
    return _config


def reset_config() -> None:
    """Reset the singleton. Use in tests to get a fresh config."""
    global _config
    _config = None
