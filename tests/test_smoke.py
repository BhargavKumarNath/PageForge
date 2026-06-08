"""
Phase 0 smoke test — verifies clean architecture and basic correctness.

These tests must pass on any machine, with or without a GPU.
Tests that require the compiled Rust extension are skipped gracefully
when symbolr_rust is not available.
"""
import importlib.util
import json
import math
import numpy as np
import pytest

RUST_AVAILABLE = importlib.util.find_spec("symbolr_rust") is not None


# Import sanity
def test_core_imports():
    """All restructured modules must be importable without error."""
    from src.symbolr.core.evaluator import BaseEvaluator
    from src.symbolr.core.bridge import RustEvolutionBridge, GenerationResult
    from src.symbolr.config import SymboLRConfig, get_config, reset_config
    from src.symbolr.artifacts.prefix_parser import evaluate_formula, parse_prefix
    from src.symbolr.artifacts.pytorch_export import export_to_pytorch
    from src.symbolr.artifacts.latex_export import export_to_latex
    from src.symbolr.baselines.schedules import BASELINE_SCHEDULES, evaluate_all_baselines
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator


# Config alignment
def test_config_matches_rust_defaults():
    """Python config defaults must match rust_core compiled constants."""
    from src.symbolr.config import SymboLRConfig
    cfg = SymboLRConfig()
    assert cfg.crossover_rate == pytest.approx(0.20), "Must match rust_core/src/evolution.rs"
    assert cfg.mutation_rate  == pytest.approx(0.70), "Must match rust_core/src/evolution.rs"
    assert cfg.size_bins      == 30,                  "Must match rust_core/src/archive.rs"
    assert cfg.com_bins       == 20,                  "Must match rust_core/src/archive.rs"
    assert cfg.smoothness_bins == 10,                 "Must match rust_core/src/archive.rs"


def test_config_from_yaml(tmp_path):
    """Config must load correctly from a YAML file."""
    from src.symbolr.config import SymboLRConfig
    yaml_content = "max_generations: 99\nseed: 1337\n"
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text(yaml_content)
    cfg = SymboLRConfig.from_yaml(str(cfg_file))
    assert cfg.max_generations == 99
    assert cfg.seed == 1337
    assert cfg.crossover_rate == pytest.approx(0.20)  # default unchanged


# Prefix parser
def test_prefix_parser_constant():
    from src.symbolr.artifacts.prefix_parser import evaluate_formula
    assert evaluate_formula("0.01", t=0.5) == pytest.approx(0.01)


def test_prefix_parser_cosine():
    from src.symbolr.artifacts.prefix_parser import evaluate_formula
    val = evaluate_formula("cos * 3.14159265 t", t=0.0)
    assert val == pytest.approx(1.0, abs=1e-4)
    val = evaluate_formula("cos * 3.14159265 t", t=1.0)
    assert val == pytest.approx(1e-7, abs=1e-6)  # clamped — cos(π) = -1 → below LR_MIN


def test_prefix_parser_division_safe():
    from src.symbolr.artifacts.prefix_parser import evaluate_formula
    val = evaluate_formula("/ t 0.0", t=0.5)
    assert math.isfinite(val)


def test_prefix_parser_new_variables():
    """Parser must handle gradient (g) and loss-slope (dl) variable tokens."""
    from src.symbolr.artifacts.prefix_parser import evaluate_formula
    val = evaluate_formula("* t g", t=0.5, g=2.0)
    assert val == pytest.approx(min(0.5 * 2.0, 10.0))


def test_prefix_parser_clamps_output():
    from src.symbolr.artifacts.prefix_parser import evaluate_formula
    assert evaluate_formula("1000.0") == pytest.approx(10.0)    # clamped to LR_MAX
    assert evaluate_formula("-999.0") == pytest.approx(1e-7)     # clamped to LR_MIN


# Exporters
def test_pytorch_export_produces_code():
    from src.symbolr.artifacts.pytorch_export import export_to_pytorch
    code = export_to_pytorch("cos * 3.14159 t")
    assert "LambdaLR" in code
    assert "lr_lambda" in code
    assert "torch.cos" in code


def test_latex_export_produces_latex():
    from src.symbolr.artifacts.latex_export import export_to_latex
    latex = export_to_latex("cos * 3.14159 t")
    assert "$$" in latex
    assert r"\cos" in latex


def test_exporters_handle_empty_input():
    from src.symbolr.artifacts.pytorch_export import export_to_pytorch
    from src.symbolr.artifacts.latex_export import export_to_latex
    assert "LambdaLR" in export_to_pytorch("")
    assert "$$" in export_to_latex("")


# Baseline schedules
def test_all_baselines_produce_valid_output():
    from src.symbolr.baselines.schedules import BASELINE_SCHEDULES
    t = np.linspace(0, 1, 100)
    for name, fn in BASELINE_SCHEDULES.items():
        lr = fn(t)
        assert len(lr) == 100, f"{name}: wrong length"
        assert np.all(np.isfinite(lr)), f"{name}: non-finite values"
        assert np.all(lr > 0), f"{name}: non-positive LR"


def test_baseline_count():
    from src.symbolr.baselines.schedules import BASELINE_SCHEDULES
    assert len(BASELINE_SCHEDULES) == 7


# GenerationResult parsing
def test_generation_result_parses_json():
    from src.symbolr.core.bridge import GenerationResult
    payload = {
        "generation_number": 3,
        "best_mse": 0.123,
        "average_mse": 0.456,
        "top_formula_latex": r"\cos(\pi t)",
        "top_formula_prefix": "cos * 3.14 t",
        "archive_size": 42,
        "new_entries": 7,
        "gen_time_ms": 250,
    }
    r = GenerationResult.from_json(json.dumps(payload))
    assert r.generation_number == 3
    assert r.best_mse == pytest.approx(0.123)
    assert r.archive_size == 42
    assert r.top_formula_prefix == "cos * 3.14 t"


def test_generation_result_handles_null_mse():
    from src.symbolr.core.bridge import GenerationResult
    payload = {
        "generation_number": 1,
        "best_mse": None,
        "average_mse": None,
        "top_formula_latex": "",
        "top_formula_prefix": "0.01",
        "archive_size": 0,
        "new_entries": 0,
        "gen_time_ms": 0,
    }
    r = GenerationResult.from_json(json.dumps(payload))
    assert r.best_mse == float("inf")


# Rust-dependent tests
@pytest.mark.skipif(not RUST_AVAILABLE, reason="symbolr_rust not compiled — run `maturin develop`")
def test_synthetic_evaluator_is_deterministic():
    """SyntheticEvaluator must return identical results for identical formulas."""
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator
    evaluator = SyntheticEvaluator(time_steps=50)
    formulas = ["cos * 3.14159 t", "* 0.01 t", "0.01"]
    a = evaluator.evaluate(formulas)
    b = evaluator.evaluate(formulas)
    for fa, fb in zip(a, b):
        assert abs(fa - fb) < 1e-9, "SyntheticEvaluator is not deterministic"


@pytest.mark.skipif(not RUST_AVAILABLE, reason="symbolr_rust not compiled — run `maturin develop`")
def test_evolution_3_generations():
    """Full end-to-end: 3 generations must complete and return typed results."""
    from src.symbolr.evaluators.synthetic import SyntheticEvaluator
    from src.symbolr.core.bridge import RustEvolutionBridge, GenerationResult

    evaluator = SyntheticEvaluator(time_steps=50)
    bridge = RustEvolutionBridge(
        eval_callback=evaluator.evaluate,
        max_generations=3,
        pop_size=20,
        seed=42,
    )

    results = list(bridge.stream())
    assert len(results) == 3

    for r in results:
        assert isinstance(r, GenerationResult)
        assert r.generation_number > 0
        assert r.best_mse >= 0.0
        assert isinstance(r.top_formula_prefix, str) and len(r.top_formula_prefix) > 0
        assert r.archive_size >= 0
        assert r.gen_time_ms >= 0
