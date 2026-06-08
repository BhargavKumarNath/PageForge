"""
src/symbolr/evaluators/gradient_aware.py — Phase 3.

Evaluates formulas by training real models on a synthetic proxy task.
Collects gradient norms (g) and loss slopes (dl) as live signals, normalizes
them into the ≈[-2,2] and [-1,1] ranges expected by VarG and VarDL, then
feeds them to gradient-aware formulas at each training step.

Batching backends:
  • _VmapBatchedTrainer  — torch.func.vmap; N models in one GPU forward+backward.
  • _SequentialTrainer   — Python for-loop; CPU or PyTorch < 2.0 fallback.

Evaluation protocol (three phases):
  1. Warmup   (warmup_fraction × n_steps)    : fixed base_lr; fit normalization.
  2. Phase 1  (halving_fraction × remaining) : all N formulas; prune bottom half.
  3. Phase 2  (remaining)                    : survivors continue; val loss = fitness.
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from src.symbolr.core.evaluator import BaseEvaluator
from src.symbolr.artifacts.prefix_parser import evaluate_formula

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Proxy task constants

PROXY_INPUT_DIM  = 784   # MNIST 28×28 flattened
PROXY_HIDDEN_DIM = 256   # MLP hidden layer (wider for MNIST complexity)
PROXY_N_CLASSES  = 10    # Number of classification targets
PROXY_N_SAMPLES  = 2000  # Total samples drawn from MNIST training split

# MNIST root: navigate from src/symbolr/evaluators/ up to project root then data/
_MNIST_ROOT = Path(__file__).resolve().parents[3] / "data"


# Input normalization

@dataclass
class _NormStats:
    """
    Per-run normalization statistics fitted from the warmup window.

    Raw gradient norms are log-transformed before z-scoring so that the
    log-normal distribution of gradient norms becomes approximately Gaussian.
    Raw loss slopes are z-scored then squashed through tanh to [-1, 1].

    Both operations map onto the coordinate ranges expected by the Rust AST:
      VarG  ≈ [-2, 2]   (log-space z-score, clamped at ±3)
      VarDL ∈ [-1, 1]   (z-score + tanh)
    """
    log_g_mean: float = 0.0
    log_g_std:  float = 1.0
    dl_mean:    float = 0.0
    dl_std:     float = 1.0
    fitted:     bool  = False

    @classmethod
    def fit(cls, log_g_samples: list[float], dl_samples: list[float]) -> "_NormStats":
        s = cls()
        if len(log_g_samples) >= 2:
            s.log_g_mean = float(np.mean(log_g_samples))
            s.log_g_std  = max(float(np.std(log_g_samples)), 1e-6)
        if len(dl_samples) >= 2:
            s.dl_mean = float(np.mean(dl_samples))
            s.dl_std  = max(float(np.std(dl_samples)), 1e-6)
        s.fitted = True
        return s

    def normalize_g(self, g_raw: float) -> float:
        """Log-space z-score → clamp to [-3, 3]."""
        z = (math.log(max(g_raw, 1e-8)) - self.log_g_mean) / self.log_g_std
        return float(max(-3.0, min(3.0, z)))

    def normalize_dl(self, dl_raw: float) -> float:
        """Z-score then tanh → [-1, 1]."""
        z = (dl_raw - self.dl_mean) / (self.dl_std + 1e-6)
        return float(math.tanh(z))


# Proxy task: dataset + model

def _build_proxy_dataset(seed: int, device) -> tuple:
    """
    2000-sample MNIST subset used as the proxy classification task.

    Replacing the previous Gaussian-cluster proxy with real MNIST images
    ensures three properties the synthetic proxy lacked:

    1. Gradient statistics during proxy evaluation match those seen during
       full MNIST training, so NormStats calibration transfers correctly.
    2. The fitness landscape penalises out-of-range learning rates: LR >> 1
       diverges; LR << 0.01 converges too slowly within the step budget.
    3. Adaptive schedules that decay or respond to gradient health earn a
       lower validation loss than a constant high-LR formula, because the
       task is hard enough that LR shape actually matters.

    Returns: (X_train, y_train, X_val, y_val) as float32 / int64 tensors.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError:
        raise RuntimeError(
            "GradientAwareEvaluator requires torchvision for the MNIST proxy. "
            "Install with: pip install torchvision"
        )

    rng = np.random.RandomState(seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST channel mean / std
    ])

    mnist = datasets.MNIST(
        root=str(_MNIST_ROOT), train=True, download=True, transform=transform
    )

    # Reproducible random subset of PROXY_N_SAMPLES from 60 K training images
    indices = rng.choice(len(mnist), size=PROXY_N_SAMPLES, replace=False).tolist()
    subset  = torch.utils.data.Subset(mnist, indices)
    loader  = torch.utils.data.DataLoader(
        subset, batch_size=PROXY_N_SAMPLES, shuffle=False, num_workers=0
    )
    X_raw, y_raw = next(iter(loader))
    X = X_raw.view(PROXY_N_SAMPLES, -1).to(device)        # (2000, 784)
    y = y_raw.to(device)                                    # (2000,)

    n_val = PROXY_N_SAMPLES // 5                            # 400 val, 1600 train
    return X[n_val:], y[n_val:], X[:n_val], y[:n_val]


class _ProxyMLP(nn.Module):
    """Lightweight two-layer MLP for the proxy classification task.

    Deliberately no BatchNorm so it works correctly inside torch.func.vmap.
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(PROXY_INPUT_DIM, PROXY_HIDDEN_DIM)
        self.fc2 = nn.Linear(PROXY_HIDDEN_DIM, PROXY_N_CLASSES)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


def _formula_seed(prefix_str: str) -> int:
    """Deterministic per-formula init seed from MD5 content hash."""
    return int(hashlib.md5(prefix_str.encode()).hexdigest()[:8], 16) % (2 ** 31)


def _vmap_available() -> bool:
    try:
        from torch.func import vmap, grad_and_value, functional_call  # noqa: F401
        return True
    except ImportError:
        return False


# Batched trainer: vmap backend
class _VmapBatchedTrainer:
    """
    Trains N models in parallel using torch.func.vmap (PyTorch >= 2.0).

    Parameters are stacked on a leading N-dimension so one call to
    vmap(grad_and_value(loss_fn)) computes gradients for all N models
    in a single GPU kernel launch.
    """

    def __init__(self, model_template: "nn.Module", N: int, device):
        from torch.func import vmap, grad_and_value, functional_call

        self._model = model_template
        self._N     = N
        self._device = device

        def _loss_fn(params: dict, x, y) -> "torch.Tensor":
            logits = functional_call(self._model, params, x)
            return F.cross_entropy(logits, y)

        self._batched_grad_loss = vmap(
            grad_and_value(_loss_fn),
            in_dims=(0, None, None),
        )

    def init_params(self, formula_strs: list[str]) -> dict:
        """Initialize N parameter sets, seeded per formula. Returns stacked dict."""
        param_list = []
        for fstr in formula_strs:
            torch.manual_seed(_formula_seed(fstr))
            m = _ProxyMLP().to(self._device)
            param_list.append({k: v.detach().clone() for k, v in m.state_dict().items()})
        return {k: torch.stack([p[k] for p in param_list]) for k in param_list[0]}

    def step(
        self,
        params: dict,
        x: "torch.Tensor",
        y: "torch.Tensor",
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """
        One gradient step for all N models (batched).

        Returns:
            grads      — stacked gradient dict (same structure as params)
            losses     — (N,) numpy array of cross-entropy losses
            grad_norms — (N,) numpy array of L2 gradient norms (pre-clip)
        """
        grads_stacked, losses_t = self._batched_grad_loss(params, x, y)

        # Per-model L2 gradient norm
        sq_sum_list = [
            g.reshape(self._N, -1).pow(2).sum(dim=-1)
            for g in grads_stacked.values()
        ]
        grad_norms_t = torch.sqrt(torch.stack(sq_sum_list).sum(dim=0) + 1e-12)

        return (
            grads_stacked,
            losses_t.detach().cpu().numpy(),
            grad_norms_t.detach().cpu().numpy(),
        )

    def apply_lrs(self, params: dict, grads: dict, lrs: np.ndarray) -> dict:
        """Apply per-model learning rates. Returns updated stacked params (detached)."""
        lrs_t = torch.tensor(lrs, dtype=torch.float32, device=self._device)
        updated = {}
        for k in params:
            p  = params[k]           # (N, *shape)
            g  = grads[k]            # (N, *shape)
            n_trail = p.dim() - 1
            lr_bc = lrs_t.reshape(-1, *([1] * n_trail))
            updated[k] = (p - lr_bc * g).detach()
        return updated

    def validate(self, params: dict, X_val: "torch.Tensor", y_val: "torch.Tensor") -> np.ndarray:
        """Return (N,) val-loss array for current parameter states."""
        from torch.func import vmap, functional_call

        def _val_loss(p, x, y):
            logits = functional_call(self._model, p, x)
            return F.cross_entropy(logits, y)

        batched_val = vmap(_val_loss, in_dims=(0, None, None))
        with torch.no_grad():
            losses = batched_val(params, X_val, y_val)
        return losses.cpu().numpy()


# Batched trainer: sequential fallback
class _SequentialTrainer:
    """
    Trains N models in a Python for-loop. Used when torch.func is unavailable.
    Correct on all PyTorch versions; slower than the vmap backend on GPU.
    """

    def __init__(self, N: int, device):
        self._N      = N
        self._device = device

    def init_params(self, formula_strs: list[str]) -> list:
        models = []
        for fstr in formula_strs:
            torch.manual_seed(_formula_seed(fstr))
            m = _ProxyMLP().to(self._device)
            m.train()
            models.append(m)
        return models

    def step(self, models: list, x, y) -> tuple[list, np.ndarray, np.ndarray]:
        losses   = np.empty(self._N, dtype=np.float64)
        g_norms  = np.empty(self._N, dtype=np.float64)
        grads    = []

        for i, m in enumerate(models):
            m.train()
            m.zero_grad()
            logits = m(x)
            loss   = F.cross_entropy(logits, y)
            loss.backward()

            # Raw gradient norm before any clipping
            total_sq = sum(
                p.grad.data.pow(2).sum().item()
                for p in m.parameters()
                if p.grad is not None
            )
            g_norms[i] = math.sqrt(total_sq + 1e-12)
            losses[i]  = loss.item()

            # Clip for numerical stability, then clone
            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=10.0)
            grads.append({
                n: p.grad.data.clone()
                for n, p in m.named_parameters()
                if p.grad is not None
            })

        return grads, losses, g_norms

    def apply_lrs(self, models: list, grads: list, lrs: np.ndarray) -> list:
        for i, m in enumerate(models):
            with torch.no_grad():
                for n, p in m.named_parameters():
                    if n in grads[i]:
                        p.data -= float(lrs[i]) * grads[i][n]
        return models

    def validate(self, models: list, X_val, y_val) -> np.ndarray:
        losses = np.empty(self._N, dtype=np.float64)
        for i, m in enumerate(models):
            m.eval()
            with torch.no_grad():
                logits = m(X_val)
                losses[i] = F.cross_entropy(logits, y_val).item()
            m.train()
        return losses


# Main evaluator
class GradientAwareEvaluator(BaseEvaluator):
    """
    Evaluates formulas by training models with them on a proxy classification task.

    Unlike SyntheticEvaluator (which evaluates at g=0, dl=0), this evaluator
    collects real gradient norms and loss slopes during training and passes them
    to gradient-aware formulas as live signals every step. A formula such as
    ``lr = 0.1 * exp(-g)`` will actually reduce the LR when gradients spike,
    and that adaptive behavior is measured and reflected in the fitness score.

    Args:
        n_steps:          Total training steps per formula evaluation.
        batch_size:       Mini-batch size for the proxy task.
        base_lr:          Fixed LR used during warmup (default: 0.1).
        warmup_fraction:  Fraction of n_steps for the warmup phase (default: 10%).
        halving_fraction: Fraction of remaining steps for Phase 1 (default: 50%).
        seed:             Controls proxy dataset construction and batch ordering.
        device:           PyTorch device string; auto-detected when None.
    """

    def __init__(
        self,
        n_steps: int = 200,
        batch_size: int = 128,
        base_lr: float = 0.1,
        warmup_fraction: float = 0.10,
        halving_fraction: float = 0.50,
        seed: int = 42,
        device: Optional[str] = None,
        instability_factor: float = 2.0,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise RuntimeError(
                "GradientAwareEvaluator requires PyTorch. "
                "Install it with: pip install torch"
            )

        self.n_steps            = n_steps
        self.batch_size         = batch_size
        self.base_lr            = base_lr
        self.warmup_fraction    = warmup_fraction
        self.halving_fraction   = halving_fraction
        self.seed               = seed
        self.instability_factor = instability_factor

        self._device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._X_train, self._y_train, self._X_val, self._y_val = (
            _build_proxy_dataset(seed, self._device)
        )
        self._model_template = _ProxyMLP().to(self._device)
        self._use_vmap = _vmap_available()

        logger.info(
            "GradientAwareEvaluator ready — device=%s  vmap=%s  "
            "n_steps=%d  warmup=%.0f%%  halving=%.0f%%",
            self._device, self._use_vmap, n_steps,
            warmup_fraction * 100, halving_fraction * 100,
        )

    @property
    def is_deterministic(self) -> bool:
        return True

    @property
    def name(self) -> str:
        return f"GradientAwareEvaluator(device={self._device}, n_steps={self.n_steps})"

    # Public API
    def evaluate(self, formulas: list[str]) -> list[float]:
        """
        Evaluate a batch of formulas by training N independent models.

        Returns:
            Fitness scores (lower = better). Returns inf for non-viable formulas.
        """
        N = len(formulas)
        if N == 0:
            return []

        warmup_steps = max(1, int(self.n_steps * self.warmup_fraction))
        remaining    = self.n_steps - warmup_steps
        phase1_steps = max(1, int(remaining * self.halving_fraction))
        phase2_steps = remaining - phase1_steps

        t0 = time.time()

        trainer = (
            _VmapBatchedTrainer(self._model_template, N, self._device)
            if self._use_vmap
            else _SequentialTrainer(N, self._device)
        )
        state = trainer.init_params(formulas)

        rng_data    = np.random.RandomState(self.seed)
        prev_losses = np.full(N, float("inf"))

        # Phase 0: warmup
        log_g_samples: list[float] = []
        dl_samples:    list[float] = []

        for _ in range(warmup_steps):
            x, y = self._next_batch(rng_data)
            grads, losses, g_norms = trainer.step(state, x, y)
            state = trainer.apply_lrs(state, grads, np.full(N, self.base_lr))

            for v in g_norms:
                if math.isfinite(float(v)) and float(v) > 0:
                    log_g_samples.append(math.log(float(v)))

            for i in range(N):
                if math.isfinite(losses[i]) and math.isfinite(prev_losses[i]):
                    dl_samples.append(float(losses[i] - prev_losses[i]))

            prev_losses = np.where(np.isfinite(losses), losses, prev_losses)

        norm_stats = _NormStats.fit(log_g_samples, dl_samples)
        logger.debug(
            "Warmup done: log_g=%.3f±%.3f  dl=%.4f±%.4f",
            norm_stats.log_g_mean, norm_stats.log_g_std,
            norm_stats.dl_mean, norm_stats.dl_std,
        )

        # Instability tracking: penalise formulas whose post-warmup training
        # loss exceeds instability_factor × the theoretical random-init CE.
        # Anchoring to initial_ce (not warmup_end_loss) makes the threshold
        # independent of warmup length — short warmups leave loss near
        # random-init CE, so a warmup-based threshold would be too tight.
        # A formula whose training loss rises above 2× random-init CE is
        # actively making the model worse and should receive fitness = inf.
        _initial_ce = math.log(PROXY_N_CLASSES)          # ≈ 2.303 for 10 classes
        _stab_threshold = np.full(N, _initial_ce * self.instability_factor)
        _diverged = ~np.isfinite(prev_losses)            # already diverged in warmup

        # Phase 1
        global_step = warmup_steps

        for step in range(phase1_steps):
            x, y = self._next_batch(rng_data)
            grads, losses, g_norms = trainer.step(state, x, y)
            t_norm = _t_norm(global_step + step, self.n_steps)
            lrs    = self._compute_lrs(formulas, t_norm, g_norms, losses, prev_losses, norm_stats)
            state  = trainer.apply_lrs(state, grads, lrs)
            prev_losses = np.where(np.isfinite(losses), losses, prev_losses)
            _diverged |= ~np.isfinite(losses) | (losses > _stab_threshold)

        global_step += phase1_steps
        phase1_val = trainer.validate(state, self._X_val, self._y_val)

        # Survivors: top ceil(N/2) by Phase 1 val loss
        n_survivors  = max(1, math.ceil(N / 2))
        survivor_set = set(int(i) for i in np.argsort(phase1_val)[:n_survivors])

        # Phase 2
        if phase2_steps > 0:
            for step in range(phase2_steps):
                x, y = self._next_batch(rng_data)
                grads, losses, g_norms = trainer.step(state, x, y)
                t_norm = _t_norm(global_step + step, self.n_steps)
                lrs    = self._compute_lrs(formulas, t_norm, g_norms, losses, prev_losses, norm_stats)
                # Freeze eliminated formulas
                for i in range(N):
                    if i not in survivor_set:
                        lrs[i] = 0.0
                state = trainer.apply_lrs(state, grads, lrs)
                prev_losses = np.where(np.isfinite(losses), losses, prev_losses)
                _diverged |= ~np.isfinite(losses) | (losses > _stab_threshold)

            final_val = trainer.validate(state, self._X_val, self._y_val)
        else:
            final_val = phase1_val

        # Apply instability penalty: override fitness for any formula whose
        # training loss spiked above the stability threshold post-warmup.
        if _diverged.any():
            final_val = final_val.copy()
            final_val[_diverged] = float("inf")
            logger.info(
                "Instability penalty applied to %d/%d formulas "
                "(training loss exceeded %.2f× warmup baseline)",
                int(_diverged.sum()), N, self.instability_factor,
            )

        elapsed    = time.time() - t0
        throughput = N / elapsed
        logger.info(
            "Evaluated %d formulas in %.2fs (%.1f /sec) on %s",
            N, elapsed, throughput, self._device,
        )

        return [
            float(v) if math.isfinite(float(v)) else float("inf")
            for v in final_val
        ]

    # Helpers

    def _next_batch(self, rng: "np.random.RandomState") -> tuple:
        idx   = rng.randint(0, len(self._X_train), size=self.batch_size)
        idx_t = torch.from_numpy(idx).to(self._device)
        return self._X_train[idx_t], self._y_train[idx_t]

    def _compute_lrs(
        self,
        formulas: list[str],
        t: float,
        g_norms: np.ndarray,
        losses: np.ndarray,
        prev_losses: np.ndarray,
        norm_stats: _NormStats,
    ) -> np.ndarray:
        """Compute per-formula learning rates at a single training step."""
        lrs = np.empty(len(formulas), dtype=np.float64)
        for i, fstr in enumerate(formulas):
            g_raw  = float(g_norms[i]) if math.isfinite(float(g_norms[i])) else 1.0
            dl_raw = (
                float(losses[i] - prev_losses[i])
                if math.isfinite(losses[i]) and math.isfinite(prev_losses[i])
                else 0.0
            )
            try:
                lr = evaluate_formula(
                    fstr,
                    t=t,
                    g=norm_stats.normalize_g(g_raw),
                    dl=norm_stats.normalize_dl(dl_raw),
                )
                lrs[i] = float(lr) if math.isfinite(lr) and lr > 0 else 1e-7
            except Exception:
                lrs[i] = 1e-7
        return lrs


def _t_norm(step: int, total_steps: int) -> float:
    """Normalize the current step to [0, 1]."""
    return float(step) / max(total_steps - 1, 1)
