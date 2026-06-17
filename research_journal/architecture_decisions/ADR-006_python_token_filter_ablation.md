# ADR-006: Ablation Uses Python Token Filter Instead of Rust Terminal-Set Config

**Date**: 2026-05-31
**Status**: Accepted

---

## Context

The Phase 6 ablation study requires running evolution with three terminal sets:
- t-only: formulas may only use `t` and constants
- t+g: formulas may use `t`, `g`, and constants
- t+g+dl: full set (current default)

The Rust engine's `random_terminal()` function (in `operators.rs`) has hardcoded
sampling probabilities: `VarT=25%, VarG=15%, VarDL=10%, Const=50%`. To truly
restrict the terminal set, one would need to make these configurable.

## Decision

Implement the ablation via `TokenFilteredEvaluator`: a Python wrapper that
returns `float("inf")` for any formula containing a forbidden token. No Rust
changes required.

## Rationale

**Equivalence argument**: If all formulas with forbidden tokens receive
`fitness=inf`, they can never displace time-only formulas in the MAP-Elites
archive. The archive will organically converge to formulas without forbidden
tokens. The search is effectively running over the restricted terminal set.

**No Rust recompile**: The approach requires zero changes to `rust_core/`.
Making terminal sampling configurable in Rust would require:
  - New `EvolutionConfig` struct fields
  - Updated PyO3 constructor signature
  - Rerunning `maturin develop`
  - Updating all callers

**Gradient-sensitivity niches**: Formulas with forbidden tokens still enter
gradient-sensitivity niches (archive structure), but with `fitness=inf` they
never become archive elites. This is a minor inefficiency (some archive slots
hold inf-fitness formulas) but does not affect result correctness.

**Speed**: The Python filter adds <1ms overhead per batch (a set intersection
per formula). The Rust change would be zero overhead but substantially more work.

## Alternatives Considered

**Configurable Rust terminal sampling**: Correct and clean, but requires Rust
changes, recompilation, and API changes. Deferred to a future phase where the
architecture may change anyway.

**Post-hoc filtering**: Run full evolution, then discard gradient-aware formulas
from the archive. This is NOT equivalent — the evolution trajectory differs
when gradient-aware formulas compete for niches. The filter must be applied
during evolution, not after.

## Consequences

- Rust engine still generates g/dl tokens; they just never win
- Archive structure may show high gradient-sensitivity bins occupied by inf-fitness
  formulas in t-only mode — this is cosmetically misleading but functionally correct
- A future ablation with configurable Rust sampling would be more efficient
  but scientifically equivalent
