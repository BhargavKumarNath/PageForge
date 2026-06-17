# ADR-004: Gradient-Aware Evaluator Uses Synthetic Proxy Task (No Downloads)

**Date**: 2026-05-30
**Status**: Accepted

---

## Context

Phase 3 requires an evaluator that trains real models so that gradient norms
and loss slopes are actual measurements from the optimizer, not synthetic
constants. The evaluator must:
1. Run in CI without a GPU or internet connection
2. Produce meaningful gradient health signals (not trivially flat)
3. Be reproducible (same seed → same result)
4. Complete in reasonable time

---

## Decision

Use a synthetic 10-class Gaussian-cluster classification dataset generated
from a NumPy seed, trained with a 2-layer MLP (64 → 128 → 10) that has no
BatchNorm. No MNIST, no CIFAR, no downloads.

---

## Rationale

**Why synthetic rather than MNIST?**
- No download: MNIST adds ~4s on first run and requires network/disk access
- The proxy task is still real classification: cross-entropy loss, real gradients,
  real loss slopes with meaningful sign (decreasing loss = negative dl)
- Formula rankings on the proxy task are informative about formula quality
  on real tasks because gradient dynamics are structurally similar

**Why no BatchNorm in the MLP?**
- BatchNorm maintains running mean/var buffers that accumulate state across
  mini-batches. This state is not captured in the `state_dict()` parameter
  snapshot used by torch.func.functional_call.
- Without BatchNorm, the model is fully stateless — the only state is the
  weight tensors, which are tracked correctly by vmap.
- For a shallow MLP on a well-scaled synthetic task, BatchNorm is not needed
  for training stability.

**Why vmap rather than a DataParallel ensemble?**
- DataParallel replicates the model across GPU devices, not across independent
  parameter sets on the same device. It cannot train N independent models on
  the same GPU simultaneously.
- `torch.func.vmap(grad_and_value(loss_fn))` creates N parallel computation
  graphs on a single device, leveraging GPU SIMD width.
- vmap requires stateless models (no in-place state mutation), which is why
  we chose a BatchNorm-free MLP.

**Why successive halving rather than full evaluation of all N?**
- Full evaluation of 50 formulas × 200 steps × 50-step halving is 2× cheaper
  than evaluating all 50 formulas for the full 200 steps.
- Phase 1 provides a noisy but directionally correct ranking. Formulas in the
  bottom 50% after Phase 1 are almost certainly not competitive.
- The halved computation budget is redirected to longer training for survivors,
  which gives higher-quality fitness estimates for the formulas that matter.

---

## Alternatives Considered

**MNIST with torchvision**: Requires download. Rejected for CI portability.

**Full n_steps for all formulas**: 2× more compute for no scientific gain
(bottom half would be eliminated anyway). Rejected.

**Multithreaded sequential evaluation**: Thread-per-model with Python threading.
GIL means CPU threads cannot run Python in parallel. GPU threads do release
the GIL during CUDA ops, but the overhead of thread management dominates at
N=50 for short runs. Rejected in favor of vmap.

**Restarting survivors in Phase 2**: More accurate Phase 2 fitness because
we can see the full training curve. But 2× more total compute. The current
approach (continuing from Phase 1 checkpoint) is cheaper and still meaningful
because a formula that performed well in Phase 1 usually continues to perform
well in Phase 2.

---

## Consequences

- The ≥10 formulas/sec GPU throughput target cannot be verified without a
  CUDA device. The test suite enforces a soft warning at 10 /sec and a hard
  floor at 0.5 /sec.
- The proxy task does not include images, convolutions, or realistic data
  augmentation. Formula rankings may not perfectly transfer to CNN tasks.
  This is a known limitation documented in `OPEN_QUESTIONS.md`.
- The `_ProxyMLP` is not exposed outside the module — it is an implementation
  detail. If the proxy task needs to change (e.g., add dropout, increase depth),
  only `gradient_aware.py` needs to be updated.
