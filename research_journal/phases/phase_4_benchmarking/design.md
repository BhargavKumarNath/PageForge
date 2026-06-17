# Phase 4 — Canonical Fitness Pipeline & Benchmarking: Design

**Date**: 2026-05-31
**Status**: Complete

---

## Motivation

Prior to Phase 4, the only way to evaluate a discovered formula was to read its
fitness score from the evolution loop — a single number from a single run with
a single evaluator. There was no way to:
1. Compare the formula against known-good baseline schedules under identical conditions
2. Assess whether the improvement was statistically consistent or a fluke
3. Export a clean summary of results for inclusion in a paper or report

Phase 4 adds the `BenchmarkSuite` — a canonical harness that produces all three.

---

## Architecture

```
BenchmarkSuite
├── _simulate_seeded(lr_schedule, landscape_seed)  — shared fitness kernel
├── _bootstrap_ci(diffs, n_bootstrap)              — percentile bootstrap CI
├── _wilcoxon_p(diffs)                             — paired Wilcoxon p-value
├── BenchmarkSuite.compare(formula, baseline_names)
│   ├── _eval_formula(formula)    — K seeds via symbolr_rust.evaluate_batch
│   └── _eval_baseline(baseline)  — K seeds, same LR array, same landscape seeds
└── Returns BenchmarkResult
    ├── TrialResult          — per-candidate fitnesses + mean/std
    ├── ComparisonResult     — delta_mean, win_rate, CI, p-value
    └── rank                 — position among (formula + all baselines)
```

---

## Paired Comparison Design

**Critical design property**: formula and baseline are evaluated with the
same landscape seeds per trial.

Trial `k` uses landscape seed `base_seed + k` for **both** the formula and
each baseline. This means variation between trials is due to the landscape,
not an asymmetry between how formulas and baselines are evaluated.

Example for n_seeds=5, base_seed=42:

| Trial | Landscape seed | Formula fitness | Cosine Annealing fitness | Delta |
|-------|---------------|-----------------|--------------------------|-------|
| 0     | 42            | 0.41            | 0.39                     | +0.02 |
| 1     | 43            | 0.38            | 0.36                     | +0.02 |
| 2     | 44            | 0.45            | 0.43                     | +0.02 |
| 3     | 45            | 0.36            | 0.35                     | +0.01 |
| 4     | 46            | 0.42            | 0.40                     | +0.02 |

The deltas are the inputs to the Wilcoxon test and bootstrap CI.
The paired design removes landscape variance from the comparison.

---

## Fitness Kernel: `_simulate_seeded`

This function is identical to `SyntheticEvaluator._simulate()` except the
landscape seed is an explicit parameter rather than derived from the schedule hash.

The kernel:
1. Normalizes the LR schedule to have mean absolute LR of 0.01
2. Initializes a quadratic landscape with curvatures [0.5, 1.0, 2.0, 4.0, 8.0]
3. Runs gradient descent with the given LR schedule
4. Returns `0.6 × final_loss + 0.4 × best_loss`

The curvature spread [0.5, 8.0] = 16× creates an ill-conditioned landscape
where schedules that maintain low LR at early steps (before the landscape
is explored) or that decay gracefully tend to perform better.

---

## Statistical Methods

### Bootstrap CI (primary output)
- 1000 resamples of the paired differences with replacement
- Reports 2.5th and 97.5th percentiles of bootstrap means
- Interpretation: CI entirely below 0 → formula consistently better

### Wilcoxon Signed-Rank Test (secondary, requires scipy)
- Applied to paired differences per trial seed
- Non-parametric; no normality assumption
- **Power warning**: with n_seeds=5, minimum achievable p-value ≈ 0.063
  (2×(1/32)); use n_seeds ≥ 7 to achieve α=0.05 significance
- Returns None gracefully if scipy not installed

### Win Rate (interpretive metric)
- Fraction of seeds where formula beats baseline
- Range [0, 1]; > 0.5 means formula is usually better
- Robust to outliers; does not depend on scipy

### Rank
- Formula's position when all candidates (formula + all baselines) are sorted
  by mean fitness, ascending (1 = best)
- Rank 1 = formula outperforms all 7 baselines on average
- Rank 8 = formula is the worst performer

---

## CLI Command: `symbolr benchmark`

```
symbolr benchmark --formula "cos * 3.14159 t" --seeds 7 --time-steps 100
```

Output (Rich table):
```
Candidate             Mean Fitness   Std       Δ vs Formula  Win Rate   95% CI               p-value
★ Formula             0.41234        ±0.02     —             —          —                    —
Cosine Annealing      0.39012        ±0.018    ✗ +0.02222    43%        [-0.001, +0.045]     0.219
Step Decay            0.44501        ±0.031    ✓ -0.03267    71%        [-0.062, -0.003]     0.063
...
```

Ranking: #3 of 8 | Beats: 3/7 baselines

---

## Scientific Integrity

- No hardcoded comparison values anywhere in the harness
- All numbers computed fresh from the seed at runtime
- `save_json()` exports the full trial-level data (not just summaries)
- Statistical limitations documented inline (n_seeds warning, scipy fallback)
- The benchmark does NOT claim "SymboLR beats Adam" — it reports measured
  fitness differences on a synthetic proxy task with explicit uncertainty

---

## Test Coverage (31 tests)

| Category              | Tests |
|-----------------------|-------|
| `_simulate_seeded`    | 5     |
| `_bootstrap_ci`       | 4     |
| `_wilcoxon_p`         | 3     |
| `BenchmarkSuite`      | 10    |
| `BenchmarkResult`     | 3     |
| End-to-end            | 1     |
| Package integration   | 5     |
