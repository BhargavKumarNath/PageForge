"""
Fast synthetic evaluator for CI, smoke tests, and rapid local iteration.

Simulates training on a noisy quadratic loss landscape using seeded RNGs,
making evaluation fully deterministic without requiring PyTorch or a GPU.
"""
from __future__ import annotations
import hashlib
import numpy as np
import symbolr_rust
from src.symbolr.core.evaluator import BaseEvaluator
from src.symbolr.config import get_config


class SyntheticEvaluator(BaseEvaluator):
    """
    Evaluates formulas by simulating gradient descent on a quadratic landscape.

    Deterministic: identical formulas always produce identical fitness scores
    because the RNG seed is derived from the formula's content hash.
    """

    def __init__(self, time_steps: int = 100):
        self.t_array = np.linspace(0.0, 1.0, time_steps, dtype=np.float64)

    @property
    def is_deterministic(self) -> bool:
        return True

    def evaluate(self, formulas: list[str]) -> list[float]:
        try:
            schedules_list = symbolr_rust.evaluate_batch(formulas, self.t_array)
        except Exception as e:
            print(f"[SyntheticEvaluator] Rust evaluate_batch failed: {e}")
            return [float("inf")] * len(formulas)

        return [self._simulate(np.array(s, dtype=np.float64)) for s in schedules_list]

    def _simulate(self, lr_schedule: np.ndarray) -> float:
        cfg = get_config()

        if not np.all(np.isfinite(lr_schedule)) or np.all(lr_schedule < 1e-7) or np.any(lr_schedule > 10.0):
            return float("inf")

        n_steps = len(lr_schedule)
        if n_steps == 0:
            return float("inf")

        raw_mean = np.mean(np.abs(lr_schedule))
        if raw_mean > 1e-7:
            normalized = np.clip(lr_schedule * (0.01 / raw_mean), 1e-5, 0.1)
        else:
            normalized = np.full(n_steps, 0.01)

        schedule_hash = int(hashlib.md5(lr_schedule.tobytes()).hexdigest()[:8], 16)
        n_evals = cfg.synth_n_evaluations
        n_dims = cfg.synth_n_dims
        noise_scale = cfg.synth_noise_scale

        curvatures = np.pad(
            np.array([0.5, 1.0, 2.0, 4.0, 8.0])[:n_dims],
            (0, max(0, n_dims - 5)),
            constant_values=1.0,
        )

        ensemble_losses = []
        for eval_idx in range(n_evals):
            rng = np.random.RandomState((schedule_hash + eval_idx) % (2**32 - 1))
            w = rng.randn(n_dims) * 2.0
            w_star = np.zeros(n_dims)
            best_loss = float("inf")
            failed = False

            for step in range(n_steps):
                lr = float(normalized[step])
                diff = w - w_star
                loss = 0.5 * np.sum(curvatures * diff ** 2)

                if loss > 1000.0 or not np.isfinite(loss):
                    failed = True
                    break

                best_loss = min(best_loss, max(0.0, loss + rng.randn() * noise_scale * (1.0 + loss)))
                grad = curvatures * diff + rng.randn(n_dims) * 0.1 * np.sqrt(1.0 + abs(loss))
                w = np.clip(w - lr * grad, -50.0, 50.0)

            if failed:
                ensemble_losses.append(float("inf"))
                continue

            diff = w - w_star
            val_loss = 0.5 * np.sum(curvatures * diff ** 2)
            ensemble_losses.append(0.6 * val_loss + 0.4 * best_loss)

        finite = [l for l in ensemble_losses if np.isfinite(l)]
        if not finite:
            return float("inf")

        avg = float(np.mean(finite))
        if np.std(lr_schedule) < 1e-6:
            avg += 0.5
        if lr_schedule[-1] > lr_schedule[0] * 1.5:
            avg += 0.3

        return float(np.clip(avg, 0.0, 50.0))
