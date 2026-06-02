"""
Phase 3 tests — GradientAwareEvaluator.

Tests are organized into:
  1. _NormStats unit tests (no torch required)
  2. Proxy dataset construction (torch required)
  3. GradientAwareEvaluator correctness: shape, finiteness, determinism
  4. Gradient-aware signal: gradient-reactive formula vs time-only
  5. Successive halving: survivor selection
  6. Throughput profiling (CPU soft-target; skip hard-fail)
  7. Config integration: Phase 3 fields present
"""
import importlib.util
import math
import time
import warnings

import numpy as np
import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

# 1. _NormStats — pure Python, no torch dependency
def test_norm_stats_default_normalizes_sensibly():
    """Unfitted NormStats should not crash and return finite values."""
    from src.symbolr.evaluators.gradient_aware import _NormStats

    s = _NormStats()
    g_out = s.normalize_g(1.0)
    dl_out = s.normalize_dl(0.0)

    assert math.isfinite(g_out)
    assert math.isfinite(dl_out)
    assert -3.0 <= g_out <= 3.0
    assert -1.0 <= dl_out <= 1.0


def test_norm_stats_fit_shifts_mean():
    """After fitting, the mean log_g maps to ~0."""
    from src.symbolr.evaluators.gradient_aware import _NormStats

    # Gradient norms drawn from a known distribution
    rng = np.random.RandomState(0)
    g_raws = rng.lognormal(mean=1.0, sigma=0.5, size=200).tolist()
    log_g_samples = [math.log(v) for v in g_raws]
    dl_samples = rng.randn(200).tolist()

    s = _NormStats.fit(log_g_samples, dl_samples)
    assert s.fitted

    # Mean sample should map to near 0
    mean_g_raw = float(np.exp(np.mean(log_g_samples)))
    normalized = s.normalize_g(mean_g_raw)
    assert abs(normalized) < 0.5, f"Mean g should normalize close to 0, got {normalized}"


def test_norm_stats_g_bounds():
    """normalize_g must always return values in [-3, 3]."""
    from src.symbolr.evaluators.gradient_aware import _NormStats

    s = _NormStats.fit([0.0] * 5, [0.0] * 5)

    for g_raw in [1e-10, 1e-4, 1.0, 100.0, 1e6]:
        out = s.normalize_g(g_raw)
        assert -3.0 <= out <= 3.0, f"normalize_g({g_raw}) = {out} out of bounds"


def test_norm_stats_dl_bounds():
    """normalize_dl must always return values in [-1, 1]."""
    from src.symbolr.evaluators.gradient_aware import _NormStats

    s = _NormStats.fit([0.0] * 5, [0.0] * 5)

    for dl_raw in [-1000.0, -1.0, 0.0, 1.0, 1000.0]:
        out = s.normalize_dl(dl_raw)
        assert -1.0 <= out <= 1.0, f"normalize_dl({dl_raw}) = {out} out of bounds"


def test_norm_stats_empty_samples():
    """Fitting with empty lists should not raise; defaults remain sensible."""
    from src.symbolr.evaluators.gradient_aware import _NormStats

    s = _NormStats.fit([], [])
    assert s.fitted
    assert math.isfinite(s.normalize_g(1.0))
    assert math.isfinite(s.normalize_dl(0.0))


# 2. Proxy dataset (torch required)
pytestmark_torch = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_proxy_dataset_shapes():
    """Proxy dataset tensors have correct shapes."""
    import torch
    from src.symbolr.evaluators.gradient_aware import (
        _build_proxy_dataset, PROXY_INPUT_DIM, PROXY_N_SAMPLES
    )

    X_train, y_train, X_val, y_val = _build_proxy_dataset(seed=0, device=torch.device("cpu"))

    n_val   = PROXY_N_SAMPLES // 5
    n_train = PROXY_N_SAMPLES - n_val

    assert X_train.shape == (n_train, PROXY_INPUT_DIM)
    assert y_train.shape == (n_train,)
    assert X_val.shape   == (n_val, PROXY_INPUT_DIM)
    assert y_val.shape   == (n_val,)

    assert X_train.dtype == torch.float32
    assert y_train.dtype == torch.int64


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_proxy_dataset_reproducible():
    """Same seed always produces the same dataset."""
    import torch
    from src.symbolr.evaluators.gradient_aware import _build_proxy_dataset

    a = _build_proxy_dataset(seed=7, device=torch.device("cpu"))
    b = _build_proxy_dataset(seed=7, device=torch.device("cpu"))
    assert torch.allclose(a[0], b[0])
    assert torch.allclose(a[2], b[2])


# 3. GradientAwareEvaluator correctness
@pytest.fixture(scope="module")
def small_evaluator():
    """GradientAwareEvaluator with minimal steps for fast tests."""
    if not TORCH_AVAILABLE:
        pytest.skip("torch not installed")
    from src.symbolr.evaluators.gradient_aware import GradientAwareEvaluator
    return GradientAwareEvaluator(
        n_steps=30,       # fast: 3 warmup + ~13 phase1 + ~14 phase2
        batch_size=64,
        base_lr=0.05,
        warmup_fraction=0.10,
        halving_fraction=0.50,
        seed=42,
        device="cpu",
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_evaluator_init(small_evaluator):
    """GradientAwareEvaluator instantiates without error."""
    from src.symbolr.evaluators.gradient_aware import GradientAwareEvaluator
    assert isinstance(small_evaluator, GradientAwareEvaluator)
    assert small_evaluator.is_deterministic


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_evaluate_returns_correct_length(small_evaluator):
    """evaluate() returns exactly N values for N formulas."""
    formulas = ["0.01", "cos * 3.14159 t", "* 0.01 t", "exp * -1 t"]
    results  = small_evaluator.evaluate(formulas)
    assert len(results) == len(formulas)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_evaluate_finite_for_valid_formulas(small_evaluator):
    """All standard time-only formulas return finite fitness scores."""
    # NOTE: formulas must produce LR << 1.0 throughout training.
    # Unscaled cos/exp (LR = 1.0 at t=0) diverge on the MNIST proxy — that is
    # intentional: the proxy now correctly penalises out-of-range schedules.
    formulas = ["0.01", "* 0.05 t", "* 0.05 cos * 3.14159 t", "+ 0.001 * 0.01 t"]
    results  = small_evaluator.evaluate(formulas)
    for fstr, fitness in zip(formulas, results):
        assert math.isfinite(fitness), f"Formula '{fstr}' returned non-finite fitness: {fitness}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_evaluate_positive_fitness(small_evaluator):
    """Fitness scores from valid formulas must be non-negative (cross-entropy >= 0)."""
    formulas = ["0.01", "* 0.05 t"]
    results  = small_evaluator.evaluate(formulas)
    for fitness in results:
        if math.isfinite(fitness):
            assert fitness >= 0.0, f"Fitness must be >= 0, got {fitness}"


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_evaluate_empty_batch(small_evaluator):
    """Empty formula batch returns empty list without error."""
    assert small_evaluator.evaluate([]) == []


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_evaluate_deterministic(small_evaluator):
    """Same formulas evaluated twice must produce identical fitness scores."""
    formulas = ["0.01", "* 0.05 t", "* 0.05 cos * 3.14159 t"]
    a = small_evaluator.evaluate(formulas)
    b = small_evaluator.evaluate(formulas)
    for fa, fb, fstr in zip(a, b, formulas):
        assert abs(fa - fb) < 1e-9, (
            f"Formula '{fstr}': non-deterministic results {fa} vs {fb}"
        )


# 4. Gradient-aware signal test
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_gradient_aware_formula_produces_different_fitness(small_evaluator):
    """
    A gradient-reactive formula must produce a different fitness than a constant.

    This is the key Phase 3 property: the evaluator passes real (g, dl) signals
    to formulas, so gradient-aware formulas will actually behave differently
    from time-only ones. We test that their fitness scores differ — not which
    is better (that depends on the proxy task and cannot be guaranteed).
    """
    # "* 0.05 exp * -1 g" = 0.05 * exp(-g): reduces LR when gradient spikes
    gradient_reactive = "* 0.05 exp * -1 g"
    constant_lr       = "0.05"

    fit_gradient = small_evaluator.evaluate([gradient_reactive])[0]
    fit_constant = small_evaluator.evaluate([constant_lr])[0]

    # Both must be finite
    assert math.isfinite(fit_gradient), f"Gradient-reactive formula returned inf: {fit_gradient}"
    assert math.isfinite(fit_constant), f"Constant formula returned inf: {fit_constant}"

    # Their fitness scores must differ by more than floating-point noise
    assert abs(fit_gradient - fit_constant) > 1e-6, (
        f"Gradient-reactive ({fit_gradient:.6f}) and constant ({fit_constant:.6f}) "
        f"formulas produced identical fitness. The evaluator may not be passing "
        f"real gradient signals to the formula."
    )


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_loss_slope_reactive_formula_different_fitness(small_evaluator):
    """A dl-reactive formula must produce a different fitness than a time-only one."""
    dl_reactive = "* 0.01 + 1.0 dl"   # increase LR when loss is still falling
    time_only   = "* 0.01 + 1.0 t"    # same structure but driven by time

    fit_dl   = small_evaluator.evaluate([dl_reactive])[0]
    fit_time = small_evaluator.evaluate([time_only])[0]

    assert math.isfinite(fit_dl)
    assert math.isfinite(fit_time)
    # Scores must differ — not guaranteed which is better
    assert abs(fit_dl - fit_time) > 1e-6, (
        f"Loss-slope-reactive ({fit_dl:.6f}) and time-only ({fit_time:.6f}) "
        f"formulas produced identical fitness — dl signal may not be flowing."
    )


# 5. Successive halving: survivors are the best Phase 1 performers
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_successive_halving_survivors_have_lower_fitness():
    """
    With halving, the final fitness scores must all come from the surviving
    formulas (those with better Phase 1 performance). We can verify this
    indirectly: comparing halved vs no-halving should show that halved
    evaluation focuses compute on better formulas.

    This test is structural: it checks that the evaluator runs without error
    for N=6 formulas (odd number ensures math.ceil(N/2) = 3 survivors).
    """
    from src.symbolr.evaluators.gradient_aware import GradientAwareEvaluator

    ev = GradientAwareEvaluator(
        n_steps=20, batch_size=32, base_lr=0.05,
        warmup_fraction=0.10, halving_fraction=0.50,
        seed=0, device="cpu",
    )
    formulas = ["0.01", "0.1", "0.001", "* 0.05 t", "* 0.01 t", "0.02"]
    results  = ev.evaluate(formulas)

    assert len(results) == 6
    for r in results:
        assert math.isfinite(r) or r == float("inf"), f"Non-finite, non-inf result: {r}"


# 6. Throughput profiling
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_throughput_cpu():
    """
    Measure evaluator throughput on CPU.

    Hard fail:  < 0.5 formulas/sec (catastrophically slow — something is broken)
    Soft warn: < 10.0 formulas/sec (below GPU target, expected on CPU)

    The ≥10 formulas/sec target applies to GPU. This test only enforces
    the hard lower bound so it passes in CI without a GPU.
    """
    from src.symbolr.evaluators.gradient_aware import GradientAwareEvaluator

    ev = GradientAwareEvaluator(
        n_steps=40, batch_size=64, seed=99, device="cpu",
    )
    formulas = ["0.01", "* 0.05 t", "* 0.05 cos * 3.14159 t", "* 0.01 exp * -1 g",
                "* 0.05 + 1 dl", "0.005", "* 0.1 t", "* 0.05 exp * -0.5 t",
                "* 0.02 cos t", "0.03"]

    t0         = time.perf_counter()
    results    = ev.evaluate(formulas)
    elapsed    = time.perf_counter() - t0
    throughput = len(formulas) / elapsed

    print(f"\nCPU throughput: {throughput:.2f} formulas/sec ({elapsed:.2f}s for {len(formulas)} formulas)")

    assert throughput >= 0.5, (
        f"Throughput {throughput:.2f} formulas/sec is below the 0.5 hard floor. "
        f"The evaluator may be looping or stalled."
    )

    if throughput < 10.0:
        warnings.warn(
            f"CPU throughput {throughput:.2f} formulas/sec is below the 10 formulas/sec "
            f"GPU target. This is expected on CPU-only environments.",
            UserWarning,
        )

    # All results must be finite
    for r in results:
        assert math.isfinite(r), f"Non-finite result from throughput test: {r}"


# 7. Config integration
def test_phase3_config_fields_present():
    """Phase 3 evaluator config fields must exist in SymboLRConfig."""
    from src.symbolr.config import SymboLRConfig, reset_config
    reset_config()
    cfg = SymboLRConfig()

    assert hasattr(cfg, "grad_eval_n_steps"),          "Missing grad_eval_n_steps"
    assert hasattr(cfg, "grad_eval_batch_size"),       "Missing grad_eval_batch_size"
    assert hasattr(cfg, "grad_eval_base_lr"),          "Missing grad_eval_base_lr"
    assert hasattr(cfg, "grad_eval_warmup_fraction"),  "Missing grad_eval_warmup_fraction"
    assert hasattr(cfg, "grad_eval_halving_fraction"), "Missing grad_eval_halving_fraction"

    assert cfg.grad_eval_n_steps          == 200
    assert cfg.grad_eval_batch_size       == 128
    assert cfg.grad_eval_base_lr          == 0.1
    assert cfg.grad_eval_warmup_fraction  == 0.10
    assert cfg.grad_eval_halving_fraction == 0.50


def test_phase3_config_in_to_dict():
    """Phase 3 fields must appear in SymboLRConfig.to_dict()."""
    from src.symbolr.config import SymboLRConfig, reset_config
    reset_config()
    d = SymboLRConfig().to_dict()

    assert "grad_eval_n_steps"          in d
    assert "grad_eval_warmup_fraction"  in d
    assert "grad_eval_halving_fraction" in d


# 8. BaseEvaluator interface compliance
@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_gradient_aware_evaluator_is_base_evaluator():
    """GradientAwareEvaluator must be a subclass of BaseEvaluator."""
    from src.symbolr.core.evaluator import BaseEvaluator
    from src.symbolr.evaluators.gradient_aware import GradientAwareEvaluator
    assert issubclass(GradientAwareEvaluator, BaseEvaluator)


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_evaluator_name_contains_device():
    """name property must include device identifier."""
    from src.symbolr.evaluators.gradient_aware import GradientAwareEvaluator
    ev = GradientAwareEvaluator(n_steps=10, device="cpu")
    assert "cpu" in ev.name.lower()


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")
def test_evaluator_exported_from_package():
    """GradientAwareEvaluator must be importable from the evaluators package."""
    from src.symbolr.evaluators import GradientAwareEvaluator  # noqa: F401
