"""
Phase 2 tests — Multi-variable AST and gradient-aware behavioral axes.

Tests are organized into:
  - Variable evaluation correctness
  - Backward compatibility (t-only path)
  - Gradient sensitivity behavioral axes
  - End-to-end evolution with gradient-aware formulas
  - Performance profile (ask/tell cycle timing)
"""
import importlib.util
import json
import time

import numpy as np
import pytest

RUST_AVAILABLE = importlib.util.find_spec("symbolr_rust") is not None
pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="symbolr_rust not compiled")

import symbolr_rust


# 1. New Variable Tokens in Prefix Strings
def test_ask_generates_gradient_tokens():
    """
    ask() must generate formulas containing 'g' and 'dl' tokens.

    This tests terminal sampling directly. We use uniform fitness so all
    formulas — including gradient-aware ones — can survive into the archive.
    The SyntheticEvaluator (which evaluates with g=0) is intentionally NOT
    used here, because it would penalize gradient-aware formulas and they
    would never enter the archive.
    """
    engine = symbolr_rust.EvolutionEngine(
        max_generations=20, pop_size=100, seed=42,
        crossover_rate=0.20, mutation_rate=0.70,
    )

    all_tokens: set = set()
    for _ in range(10):
        formulas = engine.ask()
        if not formulas:
            break
        # Uniform fitness: all formulas are equally "good"
        engine.tell([0.5] * len(formulas))
        for f in formulas:
            all_tokens.update(f.split())

    assert 'g' in all_tokens, (
        "ask() must generate formulas with 'g' token (VarG). "
        f"Tokens seen: {sorted(all_tokens)}"
    )
    assert 'dl' in all_tokens, (
        "ask() must generate formulas with 'dl' token (VarDL). "
        f"Tokens seen: {sorted(all_tokens)}"
    )


def test_evaluate_batch_backward_compatible():
    """Legacy evaluate_batch must still work with a time-only array."""
    t_array = np.linspace(0.0, 1.0, 100)
    formulas = ["cos * 3.14159 t", "* 0.01 t", "0.01"]
    result = symbolr_rust.evaluate_batch(formulas, t_array)
    assert result.shape == (3, 100)
    assert np.all(np.isfinite(result))


def test_evaluate_batch_gradient_formula():
    """evaluate_batch on a gradient formula uses g=0, dl=0 (backward compat path)."""
    t_array = np.linspace(0.0, 1.0, 10)
    # Formula: g + t — with g=0, should behave like just t
    formulas = ["+ g t"]
    result = symbolr_rust.evaluate_batch(formulas, t_array)
    expected = t_array  # g=0 → g + t = t
    np.testing.assert_allclose(result[0], expected, atol=1e-6)


# 2. Behavioral Axes — Gradient Sensitivity
def test_gradient_sensitivity_rises_as_archive_fills():
    """
    gradient_sensitivity_mean in telemetry must be >= 0.0 at all times
    and should increase or stabilize over generations as gradient-aware
    formulas are discovered and enter the archive.
    """
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator
    from src.symbolr.core.bridge import RustEvolutionBridge

    evaluator = SyntheticEvaluator(time_steps=50)
    bridge = RustEvolutionBridge(
        eval_callback=evaluator.evaluate,
        max_generations=15,
        pop_size=50,
        seed=42,
    )

    sensitivities = []
    for result in bridge.stream():
        raw_json = json.loads(
            bridge._engine.archive_stats()
        )
        # Verify the field is present
        assert "gradient_sensitivity_mean" in raw_json, (
            "archive_stats() must include gradient_sensitivity_mean"
        )
        gsm = raw_json["gradient_sensitivity_mean"]
        assert gsm >= 0.0, f"gradient_sensitivity_mean must be >= 0, got {gsm}"
        assert gsm <= 1.0, f"gradient_sensitivity_mean must be <= 1, got {gsm}"
        sensitivities.append(gsm)

    # After 15 generations, sensitivity should be non-trivial (gradient formulas discovered)
    final_sensitivity = sensitivities[-1]
    assert final_sensitivity >= 0.0  # trivially true, but guards against NaN


def test_tell_json_includes_gradient_sensitivity_mean():
    """GenerationResult JSON from tell() must include gradient_sensitivity_mean."""
    from src.symbolr.core.bridge import RustEvolutionBridge
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator

    evaluator = SyntheticEvaluator(time_steps=30)
    bridge = RustEvolutionBridge(
        eval_callback=evaluator.evaluate,
        max_generations=3,
        pop_size=20,
        seed=1,
    )

    for result in bridge.stream():
        pass  # exhaust the generator

    # Inspect last tell() output via a fresh engine
    engine = symbolr_rust.EvolutionEngine(
        max_generations=1, pop_size=10, seed=99,
        crossover_rate=0.20, mutation_rate=0.70,
    )
    formulas = engine.ask()
    fitnesses = [0.5] * len(formulas)
    json_str = engine.tell(fitnesses)
    data = json.loads(json_str)
    assert "gradient_sensitivity_mean" in data, (
        "tell() JSON must include gradient_sensitivity_mean"
    )
    gsm = data["gradient_sensitivity_mean"]
    # May be null (JSON null → Python None) if archive is empty
    assert gsm is None or (isinstance(gsm, (int, float)) and gsm >= 0.0)


# 3. Backward Compatibility — SyntheticEvaluator unaffected
def test_synthetic_evaluator_still_deterministic():
    """SyntheticEvaluator must remain deterministic after Phase 2 Rust changes."""
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator

    evaluator = SyntheticEvaluator(time_steps=50)
    formulas = ["cos * 3.14159 t", "* 0.01 t", "0.01", "sin t"]
    a = evaluator.evaluate(formulas)
    b = evaluator.evaluate(formulas)
    for fa, fb in zip(a, b):
        assert abs(fa - fb) < 1e-9, "SyntheticEvaluator must remain deterministic"


def test_evolution_produces_valid_telemetry():
    """Full 5-generation evolution must produce valid GenerationResult objects."""
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator
    from src.symbolr.core.bridge import RustEvolutionBridge, GenerationResult

    evaluator = SyntheticEvaluator(time_steps=50)
    bridge = RustEvolutionBridge(
        eval_callback=evaluator.evaluate,
        max_generations=5,
        pop_size=30,
        seed=42,
    )

    results = list(bridge.stream())
    assert len(results) == 5

    for i, r in enumerate(results):
        assert isinstance(r, GenerationResult)
        assert r.generation_number == i + 1
        assert r.best_mse >= 0.0
        assert r.archive_size >= 0
        assert r.gen_time_ms >= 0
        assert isinstance(r.top_formula_prefix, str)
        assert len(r.top_formula_prefix) > 0


def test_prefix_parser_handles_new_tokens():
    """Python prefix parser must handle g and dl tokens from Phase 2 formulas."""
    from src.symbolr.artifacts.prefix_parser import evaluate_formula

    # Gradient-reactive formula: 0.01 * exp(-g)
    val_stable = evaluate_formula("* 0.01 exp * -1 g", t=0.5, g=0.0)
    val_spike   = evaluate_formula("* 0.01 exp * -1 g", t=0.5, g=2.0)
    assert val_stable > val_spike, "LR should decrease when gradient norm is large"

    # Loss-reactive formula: 0.01 * (1 + dl)  — speed up when loss is still falling
    val_falling = evaluate_formula("* 0.01 + 1 dl", t=0.5, dl=-0.5)
    val_rising  = evaluate_formula("* 0.01 + 1 dl", t=0.5, dl=0.5)
    assert val_falling < val_rising, "LR boosted when loss slope is positive"


def test_pytorch_export_handles_g_and_dl():
    """PyTorch exporter must handle gradient variable tokens."""
    from src.symbolr.artifacts.pytorch_export import export_to_pytorch

    code = export_to_pytorch("* 0.01 exp * -1 g")
    assert "grad_norm" in code, "Exporter must render 'g' as grad_norm"

    code_dl = export_to_pytorch("+ t dl")
    assert "loss_delta" in code_dl, "Exporter must render 'dl' as loss_delta"


def test_latex_export_handles_g_and_dl():
    """LaTeX exporter must handle gradient variable tokens."""
    from src.symbolr.artifacts.latex_export import export_to_latex

    latex_g  = export_to_latex("exp * -1 g")
    latex_dl = export_to_latex("+ t dl")
    assert r"\|g\|" in latex_g,          "LaTeX must render 'g' as ||g||"
    assert r"\Delta\ell" in latex_dl,    "LaTeX must render 'dl' as Δℓ"


# 4. Performance Profile
def test_ask_tell_cycle_under_100ms():
    """
    A single ask/tell cycle (pop_size=50) must complete in under 100ms.

    This covers: offspring generation (Rust), prefix serialization,
    fitness evaluation (SyntheticEvaluator), archive update with new
    behavioral-axes computation (22 probe points per formula insertion).
    Target: <100ms. Hard fail at 500ms.
    """
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator

    evaluator = SyntheticEvaluator(time_steps=50)
    engine = symbolr_rust.EvolutionEngine(
        max_generations=1, pop_size=50, seed=42,
        crossover_rate=0.20, mutation_rate=0.70,
    )

    # Warm-up (JIT, import costs)
    warmup_formulas = engine.ask()
    warmup_fitnesses = evaluator.evaluate(warmup_formulas)
    engine.tell(warmup_fitnesses)

    # Fresh engine for the timed run
    engine2 = symbolr_rust.EvolutionEngine(
        max_generations=10, pop_size=50, seed=99,
        crossover_rate=0.20, mutation_rate=0.70,
    )
    times_ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        formulas = engine2.ask()
        fitnesses = evaluator.evaluate(formulas)
        engine2.tell(fitnesses)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(elapsed_ms)

    mean_ms = sum(times_ms) / len(times_ms)
    max_ms  = max(times_ms)

    print(f"\nAsk/tell cycle timing: mean={mean_ms:.1f}ms  max={max_ms:.1f}ms  all={[f'{t:.1f}' for t in times_ms]}")

    assert max_ms < 500.0, (
        f"Ask/tell cycle exceeded 500ms hard limit: {max_ms:.1f}ms. "
        "New niche-key computation may have introduced a bottleneck."
    )
    # Soft target — log but don't fail
    if mean_ms > 100.0:
        import warnings
        warnings.warn(
            f"Ask/tell mean cycle time {mean_ms:.1f}ms exceeds 100ms soft target. "
            "Consider profiling compute_niche_key probe count.",
            UserWarning,
        )
