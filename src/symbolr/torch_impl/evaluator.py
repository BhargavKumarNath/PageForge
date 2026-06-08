import math
import torch
import numpy as np
from src.symbolr.core.evaluator import BaseEvaluator

class CUDABatchEvaluator(BaseEvaluator):
    """
    Evaluates prefix formulas against a surrogate dataset using GPU tensor ops.

    Implements successive halving: formulas are first evaluated on 50% of
    time steps; only the top half proceed to full evaluation. This halves
    the average evaluation cost without significantly affecting selection quality.
    """

    def __init__(self, data_labels: np.ndarray, device: str = 'cuda:0'):
        if not torch.cuda.is_available() and device.startswith('cuda'):
            device = 'cpu'
        self.device = torch.device(device)
        self.labels = torch.tensor(data_labels, dtype=torch.float32, device=self.device)
        self.time_steps = len(self.labels)
        self.t = torch.linspace(0.0, 1.0, self.time_steps, dtype=torch.float32, device=self.device)
        self._const_cache: dict = {}

        if self.device.type == 'cuda':
            print(f"[CUDABatchEvaluator] CUDA device ready. VRAM allocated for {self.time_steps} steps.")

    def _get_const(self, val: float) -> torch.Tensor:
        if val not in self._const_cache:
            self._const_cache[val] = torch.full_like(self.t, val)
        return self._const_cache[val]

    def _parse_and_evaluate(self, prefix_str: str, t: torch.Tensor = None) -> torch.Tensor:
        """
        Parse and evaluate a prefix formula string as a PyTorch tensor operation.

        Args:
            prefix_str: Formula in prefix notation.
            t: Time tensor to use. Defaults to self.t. When a non-default t is
               passed (e.g. for successive halving), the const cache is bypassed
               to avoid shape mismatches.
        """
        if t is None:
            t = self.t
        use_cache = t is self.t

        tokens = prefix_str.strip().split()
        stack = []

        for token in reversed(tokens):
            if token in ('x', 't'):
                stack.append(t)
            elif token == '+':
                stack.append(stack.pop() + stack.pop())
            elif token == '-':
                stack.append(stack.pop() - stack.pop())
            elif token == '*':
                stack.append(stack.pop() * stack.pop())
            elif token == '/':
                left, right = stack.pop(), stack.pop()
                safe_right = torch.where(
                    torch.abs(right) < 1e-6,
                    torch.sign(right) * 1e-6 + 1e-6 * (right == 0).float(),
                    right,
                )
                stack.append(left / safe_right)
            elif token == 'sin':
                stack.append(torch.sin(stack.pop()))
            elif token == 'cos':
                stack.append(torch.cos(stack.pop()))
            elif token == 'exp':
                stack.append(torch.exp(torch.clamp(stack.pop(), max=20.0)))
            elif token == 'log':
                val = stack.pop()
                stack.append(torch.log(torch.where(torch.abs(val) < 1e-6, torch.full_like(val, 1e-6), torch.abs(val))))
            else:
                try:
                    fval = float(token)
                    stack.append(self._get_const(fval) if use_cache else torch.full_like(t, fval))
                except ValueError:
                    stack.append(self._get_const(0.0) if use_cache else torch.full_like(t, 0.0))

        return stack[0] if stack else torch.zeros_like(t)

    def evaluate(self, formulas: list[str]) -> list[float]:
        """Evaluate a batch of formulas via successive halving on GPU."""
        if not formulas:
            return []

        losses = [float('inf')] * len(formulas)

        batch_size = 64
        if self.device.type == 'cuda':
            total_vram = torch.cuda.get_device_properties(self.device).total_memory
            batch_size = min(256, max(16, int(total_vram / max(1, self.time_steps * 400))))

        half_idx = self.time_steps // 2
        t_half = self.t[:half_idx]
        labels_half = self.labels[:half_idx]

        with torch.no_grad():
            for i in range(0, len(formulas), batch_size):
                batch_indices = list(range(i, min(i + batch_size, len(formulas))))
                batch_formulas = [formulas[idx] for idx in batch_indices]

                # Phase 1: evaluate on first half of time steps (no self.t mutation)
                half_losses = []
                for formula in batch_formulas:
                    try:
                        y_pred = self._parse_and_evaluate(formula, t=t_half)
                        mse = torch.mean((y_pred - labels_half) ** 2).item()
                        if not math.isfinite(mse):
                            mse = float('inf')
                    except Exception:
                        mse = float('inf')
                    half_losses.append(mse)

                valid = [l for l in half_losses if l != float('inf')]
                if not valid:
                    continue
                cutoff = float(np.median(valid))

                # Phase 2: survivors evaluated on full time steps
                for local_idx, global_idx in enumerate(batch_indices):
                    if half_losses[local_idx] <= cutoff:
                        try:
                            y_pred = self._parse_and_evaluate(formulas[global_idx])
                            mse = torch.mean((y_pred - self.labels) ** 2).item()
                            parsimony = 0.01 * len(formulas[global_idx].split())
                            final = mse + parsimony
                            if not math.isfinite(final):
                                final = float('inf')
                        except Exception:
                            final = float('inf')
                        losses[global_idx] = final

        return losses
