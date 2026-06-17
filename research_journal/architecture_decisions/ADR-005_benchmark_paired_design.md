# ADR-005: Benchmark Uses Paired Landscape Seeds for Statistical Validity

**Date**: 2026-05-31
**Status**: Accepted

---

## Context

The benchmark harness must compare a discovered formula against baseline
schedules with a valid statistical design. The naive approach — run the formula
once, run each baseline once, compare the single values — is not statistically
valid: a single-point comparison has no error bars and cannot distinguish a
real improvement from a lucky run.

## Decision

Use a paired comparison design: for each trial `k`, both the formula and each
baseline are evaluated with the same landscape seed `base_seed + k`. The
`_simulate_seeded(lr_schedule, landscape_seed)` function accepts an explicit
seed parameter (instead of deriving it from the schedule hash), enabling this
paired design.

Statistical outputs:
1. **Bootstrap CI** on the mean paired difference (primary, always available)
2. **Wilcoxon signed-rank p-value** (requires scipy, reported with power caveat)
3. **Win rate** (fraction of seeds where formula beats baseline, interpretive)

## Rationale

**Why paired (not unpaired)?**
- Pairing controls for landscape variability — the most prominent source of
  noise in the synthetic quadratic evaluation
- Paired Wilcoxon test has higher power than unpaired (Mann-Whitney U) when
  the within-pair correlation is positive, which it is here (same landscape
  → correlated performances)
- Paired differences are the natural input for bootstrap CIs on the improvement

**Why `_simulate_seeded` instead of modifying `SyntheticEvaluator._simulate`?**
- `SyntheticEvaluator._simulate` seeds from the formula hash, which is the
  right behavior for the evolution loop (ensures determinism without explicit
  seed management)
- For paired comparison, we need the seed to come from the trial index, not
  the formula content. Adding an optional `seed` parameter to `_simulate`
  would change its interface and potentially break callers.
- A standalone `_simulate_seeded` in `benchmark.py` is a clean, non-breaking
  solution. It duplicates ~50 lines of the simulation code, but the duplication
  is intentional: the benchmark kernel is separate from the evaluation kernel.

**Why percentile bootstrap rather than BCa bootstrap?**
- Percentile bootstrap is simpler to implement correctly and is adequate for
  n=5-10 seeds with approximately Gaussian differences
- BCa (bias-corrected accelerated) bootstrap is more accurate for small n
  and skewed distributions, but requires jackknife estimates and is overkill
  for a proxy fitness benchmark
- If the project needs publication-quality statistics, BCa should be added

## Alternatives Considered

**Single-run comparison with a fixed seed**: Fast, deterministic, but no
uncertainty quantification. Cannot distinguish real improvement from luck.
Rejected.

**Unpaired comparison (each formula/baseline evaluated with different seeds)**:
Higher variance than paired design. Wilcoxon power is lower. Rejected.

**Using GradientAwareEvaluator for benchmarking**: More realistic proxy task,
but each run takes ~3-30 seconds (CPU). With n_seeds=5, that's 5 × (1 formula
+ 7 baselines) × 3s = 120s minimum. Too slow for interactive benchmarking.
The synthetic quadratic in `BenchmarkSuite` runs in <1 second for n_seeds=5.
GradientAwareEvaluator benchmarking can be added as a separate `--evaluator
gradient_aware` option in a future phase.

## Consequences

- Benchmark results are deterministic from `base_seed` (reproducible)
- With n_seeds=5, Wilcoxon test cannot reach α=0.05 (min p ≈ 0.063)
- With n_seeds=7, minimum p ≈ 0.031 (just below α=0.05)
- The CLI warns users about the n_seeds/power tradeoff
- `save_json()` exports trial-level fitnesses (not just means), so users can
  compute their own statistics downstream
