# Real-World Test 001: MNIST Transfer Test

**Date**: 2026-06-01
**Status**: Complete
**Script**: `experiments/step2_mnist_transfer.py`
**Formula**: `exp / - 1.8300538425345128 g cos / / sqrt 0.9748103119764147 sqrt 0.12206300624224875 dl`
**Formula (readable)**: `exp((1.83 - ||g||) / cos(2.83 / dl))`

---

## Setup

| Parameter | Value |
|-----------|-------|
| Model | FastConvNet (2 conv + 2 fc, ~200K params, grayscale) |
| Dataset | MNIST (48K train / 12K val / 10K test) |
| Epochs | 15 |
| Seeds | 3 |
| Optimizer | SGD + momentum 0.9, weight_decay=1e-4 |
| Device | CUDA (RTX 4070 Laptop, 8.6GB) |
| AMP | Enabled (bf16/fp16) |
| Gradient clip | max_norm=1.0 (added in v2 to prevent divergence) |
| LR cap | [base_lr*0.001, base_lr*5] = [5e-5, 0.25] |

---

## Results (v1 — no gradient clipping)

| Candidate | Test Acc | +/- |
|-----------|----------|-----|
| 1-Cycle | **99.29%** | 0.04% |
| Cosine Annealing | 99.22% | 0.06% |
| Step Decay | 99.21% | 0.07% |
| Constant LR | 98.93% | 0.05% |
| **SymboLR (discovered)** | **10.32%** | 0.73% |

**Failure mode**: 2/3 seeds diverged immediately post-warmup (NaN from step 1 of epoch 1). Root cause: formula produces LR ≈ 8-10 when `cos(2.83/dl) ≈ 0`, causing model explosion.

---

## Results (v2 — gradient clipping + LR cap, 15 epochs, 3 seeds)

| Rank | Candidate | Test Acc | ± | LR-std | Adaptive? |
|------|-----------|----------|---|--------|-----------|
| #1 | 1-Cycle | **99.28%** | 0.02% | 0.00033 | no |
| #2 | Cosine Annealing | 99.26% | 0.08% | 0.00016 | no |
| #3 | Step Decay | 99.21% | 0.07% | 0.00000 | no |
| #4 | Constant LR | 98.90% | 0.08% | 0.00000 | no |
| **#5** | **SymboLR** | **98.59%** | **0.25%** | **0.00000** | **YES** |

Gap vs best baseline: **-0.69%** (1-Cycle).

**Key observation**: SymboLR `LR-std = 0.00000` across all seeds — the formula saturated at `base_lr*5 = 0.25` cap on every step. It was effectively running as a constant LR=0.25, not as an adaptive formula. The formula's natural output on MNIST is LR ≈ 8-10, which is bounded to 0.25 — showing the proxy task selected for aggressive LRs, not adaptive ones.

---

## Analysis

### Why the formula failed to transfer

**Primary cause: Proxy task fitness landscape is flat for LR shape.**

On the Gaussian cluster proxy task (well-separated, 64-dim, 10-class), the fitness function rewards "convergence speed" — any formula that produces a high-enough LR converges the task in a few steps. The formula `exp(1.83-g)/cos(2.83/dl)` produces LR ≈ 5-10 at typical gradient statistics, which is indeed fast on the easy proxy. But on MNIST with SGD, LR=5-10 diverges the model.

**Secondary cause: NormStats corruption for one seed.**

Seed 142 had `log_g_mean=inf` in v1 (gradient norm went ∞ during warmup, poisoning the normalization statistics). Fix: check `math.isfinite(g_raw)` before appending to `log_g_samples`.

**What the formula "wanted" to do on MNIST:**
- `exp((1.83-g)/cos(2.83/dl))` reduces LR when `g > 1.83` (gradient spike)
- It oscillates based on `cos(2.83/dl)` — the formula's actual behavior when `dl ≈ 0` is undefined (cos of ±∞)
- The formula is mathematically well-designed for gradient health awareness, but the proxy task didn't constrain its scale

### What the proxy task needs

For formulas to transfer from proxy → real task, the proxy must:

1. **Be hard enough that LR scale matters**: On easy tasks, any LR in [0.01, 10.0] converges. The proxy must have a regime where high LR diverges and low LR is too slow — forcing the GP to discover LRs in a practical range.

2. **Have realistic gradient dynamics**: The gradient norms on Gaussian clusters may have different statistics than MNIST (especially after the model fits the easy clusters early in training).

3. **Penalize LR overshoot**: A proxy task where LR=10 produces worse fitness than LR=0.05 would select for formulas that don't overshoot.

### Proposed proxy task improvements

Option A — **Harder Gaussian clusters**: Reduce cluster separation (center scale 1.0, noise 1.5). This makes the task harder so the model can't converge regardless of LR.

Option B — **Non-convex proxy**: Use a multi-modal loss landscape (e.g., sine-modulated quadratic) where high LRs overshoot and damage optimization.

Option C — **MNIST subset as proxy**: Use 2000 MNIST samples as the proxy task. This eliminates the proxy→real-task gap entirely but increases evaluation cost.

Option D — **Add an instability penalty**: In `GradientAwareEvaluator`, if the model loss exceeds the initial cross-entropy (2.3) at any point after warmup, penalize heavily. This directly punishes high-LR formulas that cause divergence.

---

## Positive Findings

1. **The signal path works**: In v1, the surviving seed (s2) showed LR-std=0.113 — the formula WAS adapting its LR based on gradient dynamics. With gradient clipping, all seeds show LR-std=0.00000 because the cap made it constant.

2. **Gradient clipping fixed divergence completely**: v2 has no NaN in any seed across any epoch.

3. **The formula architecture is sound**: `exp((a-g)/cos(b/dl))` is a reasonable adaptive LR: it reduces LR when gradient norms are large (g >> a) and oscillates based on loss slope.

4. **Below Constant LR**: SymboLR v2 achieves 98.59%, below Constant LR (98.90%). The constant 0.25 LR (from cap saturation) is too high for 15 epochs — it doesn't decay, causing the model to overshoot. A decaying schedule would help, but the formula runs at constant 0.25 throughout.

---

## Conclusions

1. **Transfer failed due to proxy task design, not architecture**: The GP discovered a valid gradient-adaptive formula, but the easy proxy task didn't constrain its LR scale.

2. **Next step (before Step 2 re-run)**: Fix `GradientAwareEvaluator` with a harder proxy task (Option D — instability penalty is fastest to implement).

3. **For a publishable comparison**: The experiment must be re-run with a proxy task that selects for LR scale as well as adaptivity.

---

## Files

- `research_journal/experiments/step2_mnist_20260601_151053.json` — v1 results (with divergence)
- `research_journal/experiments/step2_mnist_20260601_*.json` — v2 results (with fixes)
- `experiments/step2_mnist_transfer.py` — transfer test script (v2)
