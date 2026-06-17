# Phase 2 — Archive Redesign: Gradient-Aware Behavioral Axes

**Date**: 2026-05-30
**Status**: Complete

---

## Old Axes vs New Axes

| Dimension | Old | New |
|-----------|-----|-----|
| Axis 0 | Formula size (node count) | Formula size (node count) — kept |
| Axis 1 | Centre-of-mass of LR curve | Gradient sensitivity (CoV as g varies) |
| Axis 2 | Smoothness (total variation) | Loss sensitivity (CoV as dl varies) |

Grid size: 30 × 20 × 10 = **6000 niches** (unchanged)

## Why the Old Axes Were Replaced

The old axes (COM and smoothness) measured properties of the formula's output curve when evaluated on a time grid with `g=0, dl=0`. With multi-variable formulas, these properties are meaningless because:

1. COM is a function of all inputs, not just time
2. A formula `exp(-g)` has perfect smoothness over time but highly variable output — smoothness on a time grid would classify it as "constant"

The new axes directly answer the research question: **does this formula react to gradient health?**

## Sensitivity Computation

```
For gradient_sensitivity_idx:
    Probe g ∈ {-2.0, -1.6, ..., +2.0}  (11 points)
    Hold t=0.5, dl=0.0 fixed
    CoV = std(outputs) / (mean(|outputs|) + 1e-6)
    Clamp to [0, 1]
    
For loss_sensitivity_idx:
    Probe dl ∈ {-1.0, -0.8, ..., +1.0}  (11 points)
    Hold t=0.5, g=0.0 fixed
    CoV = std(outputs) / (mean(|outputs|) + 1e-6)
    Clamp to [0, 1]
```

**Cost**: 22 formula evaluations per archive insertion (vs 100 time-step evaluations before). This is 4.5× cheaper per niche-key computation.

## Formula Classification by Archive Position

| Archive region | Meaning |
|----------------|---------|
| (size_idx, 0, 0) | Time-only strategy: formula ignores gradient/loss signals |
| (size_idx, high, 0) | Gradient-reactive: strongly responds to ‖∇‖, ignores Δloss |
| (size_idx, 0, high) | Loss-reactive: responds to loss slope, ignores gradient norm |
| (size_idx, high, high) | Fully adaptive: responds to both health signals |

## Validity Check

A formula is rejected from the archive if it cannot produce a positive output for ANY of the 45 probe points across the (t, g, dl) space:
```
T_VALS ∈ {0.1, 0.3, 0.5, 0.7, 0.9}
G_VALS ∈ {-2.0, 0.0, 2.0}
DL_VALS ∈ {-1.0, 0.0, 1.0}
```

This correctly rejects `Const(0.0)` and `Const(-5.0)` (always non-positive) while accepting `VarDL` alone (positive at dl=1.0), `cos(π*t)` (positive at t=0.1), and all meaningful formula shapes.

## New Telemetry Field

`gradient_sensitivity_mean` is now emitted in every `tell()` response and `archive_stats()` call. It tracks the mean gradient-sensitivity bin index of all archive elites, normalized to [0, 1].

**Expected trajectory**: starts near 0.0 (initial population is random, mostly time-only), increases as search discovers gradient-reactive formulas, stabilizes when the gradient-aware niches are filled.

## ADR Reference

See `ADR-003_archive_behavioral_axes.md` for the full decision record including alternatives considered.
