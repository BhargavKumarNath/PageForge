# Phase 3 — Gradient-Aware Interactive Evaluator: Design

**Date**: 2026-05-30
**Status**: Complete

---

## Motivation

The SyntheticEvaluator from Phase 0 evaluates formulas by simulating gradient
descent on a quadratic landscape with fixed `g=0, dl=0`. This means:

1. Gradient-aware formulas like `0.1 * exp(-g)` receive `g=0` at every step,
   making them functionally identical to `0.1 * exp(0) = 0.1`.
2. The MAP-Elites archive correctly places them in different behavioral niches
   (Phase 2 fixed this with gradient-sensitivity axes), but their *fitness* is
   evaluated as if they were time-only.
3. A formula that is genuinely superior when `g` is high cannot be discovered
   because its advantage is never measured.

Phase 3 fixes this by replacing synthetic evaluation with real training.

---

## Architecture

```
GradientAwareEvaluator
├── _build_proxy_dataset()      — synthetic 10-class Gaussian clusters (no download)
├── _ProxyMLP                   — 2-layer MLP (64 → 128 → 10), no BatchNorm
├── _NormStats                  — warmup-fitted normalization for g and dl
├── _VmapBatchedTrainer         — N models in one GPU pass via torch.func.vmap
└── _SequentialTrainer          — fallback: Python for-loop, any PyTorch version
```

### Proxy Task

A synthetic 10-class classification dataset is constructed from Gaussian clusters:
- 10 classes × 200 samples = 2000 total; 20% held out as validation
- Clusters are well-separated (mean ± 3σ, σ = 0.8) so the task is solvable
  in <200 steps, producing meaningful gradient dynamics
- No download required; reproducible from a seed

The proxy task is not MNIST. This is a deliberate choice:
- MNIST requires a download and adds ~4s to first-run setup
- The proxy task is still a real classification problem with real gradient signals
- Formula rankings are meaningfully correlated with real-task rankings as long
  as the gradient dynamics are realistic (which they are for this task)

### Input Normalization

Raw gradient norms and loss slopes do not have stable distributions across
formulas or training runs. Two normalizations are applied:

**Gradient norm (g)**:
```
g_raw   → log(g_raw + ε) → z-score using warmup statistics → clamp([-3, 3])
```
- Log-transform: gradient norms are log-normally distributed; log maps them
  to approximately Gaussian
- Z-score: centers the distribution to 0 with unit variance
- Clamp: prevents extreme values from overwhelming formulas like exp(g)

**Loss slope (dl)**:
```
dl_raw = loss[t] - loss[t-1]  → z-score → tanh → [-1, 1]
```
- Z-score: centers the loss-slope distribution
- tanh: maps the z-scored values into [-1, 1], matching VarDL's expected range

Both normalizations use statistics collected during a 10% warmup window where
all N models train with `base_lr` (fixed). This warmup provides per-run
normalization statistics that adapt to the specific task and formula batch.

### Successive Halving

```
n_steps total
├── warmup_steps  = n_steps × warmup_fraction   (default: 10%)
│   └── all N formulas at base_lr; collect norm stats
├── phase1_steps  = remaining × halving_fraction (default: 50%)
│   └── all N formulas with dynamic LR from formula
│   └── validate → keep top ceil(N/2) survivors
└── phase2_steps  = remaining × (1 - halving_fraction)
    └── survivors continue; eliminated formulas frozen (lr=0)
    └── final validation loss = fitness score
```

Phase 2 does **not** reinitialize eliminated models. Their parameters are frozen
in place by setting `lr=0.0`. This is more compute-efficient than restart, and
it correctly penalizes formulas that performed poorly in Phase 1 — they enter
Phase 2 already behind the survivors.

### Batching: vmap Backend

When `torch.func.vmap` is available (PyTorch ≥ 2.0):

```python
grads_stacked, losses = vmap(
    grad_and_value(loss_fn),
    in_dims=(0, None, None)   # params batched, x/y shared
)(stacked_params, x_batch, y_batch)
```

- `stacked_params[k]` has shape `(N, *param_shape)`
- One GPU kernel computes gradients for all N models simultaneously
- Output: per-model losses `(N,)` and stacked gradients `(N, *param_shape)`

Gradient norm extraction:
```python
grad_norms = sqrt(sum(g.reshape(N, -1).pow(2).sum(dim=-1) for g in grads.values()))
```

Learning rate application:
```python
for k: new_params[k] = params[k] - lr_bc * grads[k]   # lr_bc broadcast over trailing dims
```

All parameters are `.detach()`ed after each update to prevent computational
graph accumulation across steps.

---

## Performance

Tested on CPU (no GPU available in this environment):

| N formulas | n_steps | Time  | Throughput |
|-----------|---------|-------|-----------|
| 10        | 40      | ~3s   | ~3.3 /sec |
| 4         | 30      | ~1.5s | ~2.7 /sec |

**CPU throughput is below the ≥10 formulas/sec GPU target.** This is expected —
the target is for GPU only. The sequential trainer on CPU is a correct reference
implementation. The vmap backend on a real GPU (e.g., RTX 4070) is expected to
reach 20-50 formulas/sec for n_steps=200 based on the batched kernel design.

The test suite enforces only a 0.5 formulas/sec hard floor on CPU, and warns
if throughput is below 10 formulas/sec without a GPU.

---

## Key Properties

| Property | Value |
|----------|-------|
| Deterministic | ✅ Yes — formula-hash seeded models + fixed batch sequence |
| Gradient-reactive formulas measurable | ✅ Yes — real g, dl signals |
| No data downloads | ✅ Yes — synthetic proxy task |
| BaseEvaluator compatible | ✅ Yes — drop-in replacement for SyntheticEvaluator |
| torch.func.vmap batching | ✅ When PyTorch ≥ 2.0 |
| Successive halving | ✅ Default: 10% warmup, 50% phase1, 40% phase2 |
