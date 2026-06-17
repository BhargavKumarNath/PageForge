# ADR-003: Archive Behavioral Axes Changed to Gradient/Loss Sensitivity

**Date**: 2026-05-30
**Status**: Accepted

---

## Context

The MAP-Elites archive previously used three behavioral axes:
1. Formula size (node count)
2. Centre-of-mass of the LR output curve over `t ∈ [0, 1]`
3. Smoothness (normalized total variation) of the LR curve

These axes made sense for time-only formulas `lr = f(t)`. With Phase 2's multi-variable formulas `lr = f(t, g, dl)`, they break down:

- COM and smoothness are computed at `g=0, dl=0`, making them blind to gradient-reactive behavior
- A formula like `exp(-g)` that produces flat output at `g=0` would be classified as "low smoothness, constant COM" — indistinguishable from `Const(1.0)`
- The archive would fail to maintain diversity along the gradient-awareness dimension

## Decision

Replace axes 1 and 2 with sensitivity measures:

- **Axis 1**: Gradient sensitivity — CoV of formula output as `g` varies over `[-2, 2]` at `t=0.5, dl=0`
- **Axis 2**: Loss sensitivity — CoV of formula output as `dl` varies over `[-1, 1]` at `t=0.5, g=0`

Formula size (axis 0) is retained unchanged.

## Rationale

1. **Scientific alignment**: The research question is "which formulas are gradient-aware?" The behavioral axes should measure gradient-awareness directly.

2. **Diversity maintenance**: Time-only formulas land in (size, 0, 0) niches. Gradient-reactive formulas land in (size, high, 0). Loss-reactive formulas land in (size, 0, high). This creates genuine behavioral diversity along the research-relevant dimension.

3. **Computational cost**: Sensitivity computation requires 22 probe points vs 100 time-step evaluations. 4.5× cheaper.

4. **Interpretability**: The behavioral map is now directly readable: high `grad_idx` = gradient-reactive, high `loss_idx` = loss-slope-reactive. Previously, COM and smoothness required domain knowledge to interpret.

## Alternatives Considered

**Keep old axes, add new axes (4D archive)**: Would increase archive size to 30×20×10×10×5 = 300,000 niches. Too sparse for the search budget. Rejected.

**Use time-series properties on the multi-variable output**: Requires a canonical (t, g, dl) trajectory which doesn't exist until real training data is available. Rejected for Phase 2.

**No behavioral axes, just fitness-based MAP**: Degenerates to a regular EA without diversity maintenance. Rejected — diversity is the key property of MAP-Elites.

## Consequences

- Existing checkpoints (`.json` files from Phase 0) are **incompatible** with the Phase 2 archive structure. The `ArchiveConfig` fields changed from `com_bins`/`smoothness_bins` to `gradient_sensitivity_bins`/`loss_sensitivity_bins`.
- The `gradient_sensitivity_mean` telemetry field is now available in every `tell()` response, enabling dashboard visualization of how gradient-aware the archive is becoming.
- Phase 5 dashboard will need to update the behavioral map visualization to use the new axes.
