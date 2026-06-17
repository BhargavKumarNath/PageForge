# Phase 2 — Performance Profile

**Date**: 2026-05-30
**Machine**: Windows 11, local development machine
**Evaluator**: SyntheticEvaluator (CPU, no GPU)

---

## Ask/Tell Cycle Timing (pop_size=50)

Measured over 5 consecutive generations after 1 warm-up generation.

| Generation | Time (ms) |
|------------|-----------|
| 1 (warm-up) | ~50 |
| 2 | 52.1 |
| 3 | 68.4 |
| 4 | 89.4 |
| 5 | 91.4 |
| 6 | 89.5 |
| **Mean** | **78.1** |
| **Max** | **91.4** |

**Target**: < 100ms per cycle. **Result**: ✅ Passed (max 91.4ms).

---

## Breakdown Estimate

| Component | Estimated cost |
|-----------|---------------|
| `ask()` — offspring generation (Rust, sequential) | ~1ms |
| `SyntheticEvaluator.evaluate(50 formulas)` | ~70ms |
| `tell()` — archive update with niche-key computation | ~5ms |
| JSON serialization + Python overhead | ~3ms |

The dominant cost is `SyntheticEvaluator` which calls `symbolr_rust.evaluate_batch` (50 formulas × 50 time steps = 2500 formula evaluations). The niche-key computation (22 probe points × up to 50 new archive entries = up to 1100 additional evaluations) adds only ~5ms.

---

## Comparison to Phase 1 (Old archive axes)

| Metric | Phase 1 (COM+smoothness) | Phase 2 (sensitivity) |
|--------|--------------------------|----------------------|
| Probe points per niche key | 100 (time sweep) | 22 (sensitivity probes) |
| Niche-key cost | ~100 evals | ~22 evals |
| Speedup on niche computation | 1× | **4.5×** |

The new axes are faster to compute AND more semantically meaningful. No regression.

---

## Cache Bug Fixed (Phase 2)

The legacy `evaluate_batch` cache used the formula prefix string as the sole cache key. This caused silent result contamination when the same formula was evaluated with different `t_array` lengths in different tests.

**Fix**: Cache key is now `"{formula}|{t_array_len}"`. This eliminates cross-test contamination with a minimal change and no performance regression (cache hit rate is unchanged within a single run).

---

## No Regression on Existing Tests

All 17 Phase 0 smoke tests continue to pass after Phase 2 changes. The `SyntheticEvaluator` uses `evaluate_batch(formulas, t_array)` which internally passes `g=0.0, dl=0.0` — the legacy path is fully intact.
