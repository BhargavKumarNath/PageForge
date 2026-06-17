# Phase 2 — AST Extension: Multi-Variable Formulas

**Date**: 2026-05-30
**Status**: Complete

---

## What Changed

The `Expr` enum in `rust_core/src/ast.rs` previously had a single terminal variable `Var` (always mapped to `t`). Phase 2 replaces it with three distinct terminals:

| Variant | Token | Semantics | Input Range |
|---------|-------|-----------|-------------|
| `VarT`  | `"t"` | Normalized training time | [0, 1] |
| `VarG`  | `"g"` | Log-normalized gradient norm `log(‖∇‖ + ε)` | ≈ [-2, 2] |
| `VarDL` | `"dl"` | Tanh-normalized loss slope `tanh(Δloss / σ)` | [-1, 1] |

## Key Design Decisions

**Backward compatibility via `eval_schedule_time_only`**: The `SyntheticEvaluator` and `evaluate_batch` legacy path continue to work by passing `g=0.0, dl=0.0`. This means:
- Time-only formulas (containing only `VarT` and `Const`) are completely unaffected
- Gradient-aware formulas evaluated with `g=0, dl=0` behave like constant × time formulas

**Terminal sampling probabilities** (`random_terminal` in `operators.rs`):
```
VarT  : 25%   (primary signal — all meaningful schedules depend on time)
VarG  : 15%   (gradient health signal)
VarDL : 10%   (loss slope signal)
Const : 50%   (scaling constants)
```

**Rationale for 50% VarT dominance**: Search already generates too many large constant-heavy formulas. Adding `VarG` and `VarDL` at 25%/15% ensures they appear frequently enough to be explored, while VarT remains dominant since all real schedules must decay or vary with time.

**Hash tags**: New variants use tags 13 and 14, preserving the original 0-11 tag space.

## Prefix String Examples

```
cos * 3.14159 t            → time-only cosine schedule
* 0.01 exp * -1 g          → 0.01 · exp(-||g||) — decreases when gradient spikes
+ t dl                     → t + Δl — time + loss momentum
* exp neg_g cos * pi t     → exp(-||g||) · cos(πt) — combined time+gradient
```

## What Was NOT Changed

- `operators.rs` tree traversal (`all_paths`, `get_subtree`, `replace_subtree`): wildcard `_` cases already handle all terminal variants
- `evolution.rs`: `generate_offspring` and `update_archive` work through the archive API — no changes needed
- `MAX_NODES = 15`, `MAX_DEPTH = 7`: caps unchanged
