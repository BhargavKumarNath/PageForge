# Phase 6 — Research Validation & Ablation Studies: Design

**Date**: 2026-05-31
**Status**: Complete

---

## Motivation

Phases 2–4 built the machinery for gradient-health-aware formula discovery.
Phase 6 validates the core scientific claim: **does conditioning the terminal
set on gradient health signals actually produce better LR schedules?**

Without this validation, the system is technically complete but scientifically
untested. The ablation study is the minimum experiment needed to make a
defensible claim.

---

## What Was Built

### Engine wire-up (critical gap closed)

The `evolve` CLI command now supports three evaluator modes:

```
symbolr evolve --evaluator gradient_aware   # Phase 3 evaluator (real signals)
symbolr evolve --evaluator synthetic        # Phase 0 evaluator (g=0, dl=0)
symbolr evolve --evaluator cuda_batch       # legacy GPU batch evaluator
```

Prior to Phase 6, only `synthetic` and `cuda_batch` were wired. The
`GradientAwareEvaluator` existed (Phase 3) but was disconnected — gradient-
aware formulas could never gain a fitness advantage because they were always
evaluated at `g=0, dl=0`.

The evolve table now shows a `∇-Sens` column when using `gradient_aware`,
tracking the mean gradient-sensitivity bin index of archive elites over time.

`GenerationResult` now carries `gradient_sensitivity_mean` (parsed from the
`tell()` JSON that Rust already emitted in Phase 2) and includes it in `to_dict()`.

### TokenFilteredEvaluator

```python
# Controls effective terminal set from Python — no Rust changes needed
TokenFilteredEvaluator(base, forbidden={"g", "dl"})  # t-only
TokenFilteredEvaluator(base, forbidden={"dl"})        # t + g
base                                                   # full t + g + dl
```

When the Rust engine generates a formula containing a forbidden token, the
wrapper returns `float("inf")` as fitness. The formula enters the archive
(it occupies a gradient-sensitivity niche) but with inf fitness it can never
displace a time-only formula. The effect is equivalent to running evolution
with a restricted terminal set, achieved entirely from Python.

### AblationRunner

```python
runner = AblationRunner(
    base_evaluator  = GradientAwareEvaluator(n_steps=200),
    max_generations = 50,
    pop_size        = 50,
    seed            = 42,
    run_benchmark   = True,   # compare each best formula via BenchmarkSuite
)
result = runner.run_all()
result.save_json("research_journal/experiments/ablation_001.json")
```

Runs three configurations in sequence with identical seeds:
1. `t_only`  — time-only terminal set
2. `t_g`     — time + gradient norm
3. `t_g_dl`  — full set (time + gradient + loss slope)

For each, the best formula (tracked as the per-generation minimum fitness
across all generations) is benchmarked via `BenchmarkSuite`.

### Experiment script

`experiments/ablation_terminal_set.py` — standalone Python script, no
imports from test infrastructure. Accepts CLI arguments:

```
python experiments/ablation_terminal_set.py --generations 50 \
    --evaluator gradient_aware \
    --output research_journal/experiments/ablation_001.json
```

Prints a summary table and interprets the result:

```
ABLATION SUMMARY  (total: 142.3s)
========================================================================
Config                        Fitness  Archive  ∇-Sens    Rank  Beaten
------------------------------------------------------------------------
Full  lr = f(t, g, Δl)        0.38120       48   0.312      #2      5
Time + gradient  lr = f(t, g) 0.39401       51   0.148      #3      4
Time-only  lr = f(t)          0.42310       43   0.000      #5      2
========================================================================

Result: 'Full  lr = f(t, g, Δl)' produced the highest-ranking formula (rank #2).
  ✓ Gradient conditioning improved formula quality.
```

---

## Key Design Decision: Token-Level Ablation from Python

The alternative would be making Rust terminal-sampling probabilities
configurable via PyO3. This would be cleaner scientifically (the engine would
literally never generate forbidden tokens) but requires:
- New Rust struct fields
- New PyO3 constructor parameters
- Recompilation of the Rust extension

The Python token-filter approach achieves the same result (forbidden tokens
never win) with zero Rust changes. The Rust engine still generates them and
bins them into gradient-sensitivity niches, but with `fitness=inf` they never
enter the Pareto frontier. From the perspective of formula quality, this is
equivalent to not generating them at all.

See ADR-006 for the full decision record.

---

## Bug Found and Fixed

**Issue**: `bridge.hall_of_fame()` and `bridge.archive_stats()` return empty
results after the `stream()` generator is exhausted. The Rust engine finalizes
its internal state when `max_generations` is reached.

**Fix**: Collect best formula and final archive stats directly from the
per-generation stream results, tracking the minimum-fitness formula across all
generations in a running variable. Do not call post-stream archive methods.

This was a silent data loss — the `AblationRun` was silently populated with
`best_fitness=inf` and `archive_size=0` from the wrong source.

---

## Test Coverage (24 tests)

| Category                    | Tests |
|-----------------------------|-------|
| TokenFilteredEvaluator      | 9     |
| GenerationResult sensitivity | 3     |
| AblationRunner              | 4     |
| AblationResult              | 3     |
| AblationConfig constants    | 5     |
