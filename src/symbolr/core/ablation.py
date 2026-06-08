"""
src/symbolr/core/ablation.py — Phase 6 ablation study framework.

Runs the SymboLR evolution engine under three terminal-set configurations
and compares the best discovered formula from each configuration against
baseline schedules using BenchmarkSuite.

The three configurations are:
  t_only  — lr = f(t):       forbid g and dl
  t_g     — lr = f(t, g):    forbid dl only
  t_g_dl  — lr = f(t, g, Δl): full terminal set (no filter)

Scientific design:
  All three runs use identical: max_generations, pop_size, seed, evaluator.
  The only difference is which tokens can survive in the archive (via
  TokenFilteredEvaluator's fitness=inf penalty for forbidden tokens).
  The best formula from each run is then benchmarked under the same
  BenchmarkSuite to measure whether gradient conditioning improves fitness.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.symbolr.core.evaluator import BaseEvaluator
from src.symbolr.core.bridge import RustEvolutionBridge, GenerationResult
from src.symbolr.evaluators.filtered import TokenFilteredEvaluator
from src.symbolr.baselines.benchmark import BenchmarkSuite

logger = logging.getLogger(__name__)


# Canonical terminal-set configurations
@dataclass(frozen=True)
class AblationConfig:
    """One terminal-set configuration for the ablation study."""
    name:             str        # machine identifier: "t_only", "t_g", "t_g_dl"
    label:            str        # human label: "Time-only (t)"
    forbidden_tokens: frozenset  # tokens to forbid; empty = no filter


ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig(
        name="t_only",
        label="Time-only  lr = f(t)",
        forbidden_tokens=frozenset({"g", "dl"}),
    ),
    AblationConfig(
        name="t_g",
        label="Time + gradient  lr = f(t, g)",
        forbidden_tokens=frozenset({"dl"}),
    ),
    AblationConfig(
        name="t_g_dl",
        label="Full  lr = f(t, g, dl)",
        forbidden_tokens=frozenset(),
    ),
]


# Per-run result
@dataclass
class AblationRun:
    """Results of a single ablation configuration run."""
    config_name:               str
    config_label:              str
    best_formula_prefix:       str
    best_formula_latex:        str
    best_fitness:              float
    final_archive_size:        int
    final_gradient_sensitivity: float   # mean gradient sensitivity of final archive
    elapsed_sec:               float
    generation_log:            list[dict] = field(default_factory=list)
    benchmark_rank:            Optional[int]   = None  # rank among (formula + 7 baselines)
    benchmark_n_beaten:        Optional[int]   = None  # baselines beaten
    benchmark_result_dict:     Optional[dict]  = None  # full BenchmarkResult.to_dict()

    def to_dict(self) -> dict:
        return {
            "config_name":                self.config_name,
            "config_label":               self.config_label,
            "best_formula_prefix":        self.best_formula_prefix,
            "best_formula_latex":         self.best_formula_latex,
            "best_fitness":               self.best_fitness,
            "final_archive_size":         self.final_archive_size,
            "final_gradient_sensitivity": self.final_gradient_sensitivity,
            "elapsed_sec":                self.elapsed_sec,
            "generation_log":             self.generation_log,
            "benchmark_rank":             self.benchmark_rank,
            "benchmark_n_beaten":         self.benchmark_n_beaten,
        }


# Full ablation result
@dataclass
class AblationResult:
    """Results of all three terminal-set configurations."""
    runs:            dict[str, AblationRun]   # keyed by config_name
    total_elapsed_sec: float

    @property
    def configs_run(self) -> list[str]:
        return list(self.runs.keys())

    def summary_rows(self) -> list[dict]:
        """Return one summary dict per config, sorted by best_fitness ascending."""
        rows = [
            {
                "config":       r.config_label,
                "best_fitness": r.best_fitness,
                "archive_size": r.final_archive_size,
                "grad_sens":    r.final_gradient_sensitivity,
                "bench_rank":   r.benchmark_rank,
                "bench_beaten": r.benchmark_n_beaten,
                "formula":      r.best_formula_prefix,
                "elapsed_sec":  r.elapsed_sec,
            }
            for r in self.runs.values()
        ]
        return sorted(rows, key=lambda x: (x["best_fitness"] or float("inf")))

    def to_dict(self) -> dict:
        return {
            "runs":              {k: v.to_dict() for k, v in self.runs.items()},
            "total_elapsed_sec": self.total_elapsed_sec,
            "summary":           self.summary_rows(),
        }

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info("Ablation results saved to %s", path)


# AblationRunner
class AblationRunner:
    """
    Runs the three-way terminal-set ablation study.

    Each configuration uses an identical evolution setup (seed, pop_size,
    max_generations). The only difference is which tokens are forbidden
    in the fitness callback via TokenFilteredEvaluator.

    After evolution, the best formula from each run is benchmarked using
    BenchmarkSuite for a rigorous statistical comparison.

    Args:
        base_evaluator:  Evaluator used as the base (wrapped by TokenFilteredEvaluator).
                         Use GradientAwareEvaluator for the full signal path.
                         Use SyntheticEvaluator for fast CI / quick iteration.
        max_generations: Generations per configuration (default: 20).
        pop_size:        Population size (default: 50).
        seed:            Evolution seed — same for all three configurations.
        benchmark_seeds: Seeds for BenchmarkSuite comparison (default: 5).
        benchmark_steps: Time steps for BenchmarkSuite (default: 100).
    """

    def __init__(
        self,
        base_evaluator: BaseEvaluator,
        max_generations: int = 20,
        pop_size:        int = 50,
        seed:            int = 42,
        benchmark_seeds: int = 5,
        benchmark_steps: int = 100,
        run_benchmark:   bool = True,
    ) -> None:
        self.base_evaluator  = base_evaluator
        self.max_generations = max_generations
        self.pop_size        = pop_size
        self.seed            = seed
        self.benchmark_seeds = benchmark_seeds
        self.benchmark_steps = benchmark_steps
        self.run_benchmark   = run_benchmark

    # Public API
    def run_single(self, config: AblationConfig) -> AblationRun:
        """
        Run one ablation configuration end-to-end.

        Returns an AblationRun with per-generation telemetry and (optionally)
        a BenchmarkSuite result for the best discovered formula.
        """
        logger.info("Starting ablation: %s", config.label)
        t0 = time.time()

        # Wrap the base evaluator with the token filter
        if config.forbidden_tokens:
            evaluator = TokenFilteredEvaluator(
                self.base_evaluator,
                forbidden_tokens=set(config.forbidden_tokens),
            )
        else:
            evaluator = self.base_evaluator

        bridge = RustEvolutionBridge(
            eval_callback   = evaluator.evaluate,
            max_generations = self.max_generations,
            pop_size        = self.pop_size,
            seed            = self.seed,
        )

        generation_log: list[dict] = []
        _best_fitness = float("inf")
        _best_prefix  = ""
        _best_latex   = ""

        for result in bridge.stream():
            generation_log.append({
                "gen":               result.generation_number,
                "best_fitness":      result.best_mse,
                "avg_fitness":       result.average_mse,
                "archive_size":      result.archive_size,
                "grad_sensitivity":  result.gradient_sensitivity_mean,
                "top_formula":       result.top_formula_prefix,
                "top_formula_latex": result.top_formula_latex,
                "ms":                result.gen_time_ms,
            })
            if result.best_mse < _best_fitness:
                _best_fitness = result.best_mse
                _best_prefix  = result.top_formula_prefix
                _best_latex   = result.top_formula_latex

        elapsed = time.time() - t0

        best_prefix  = _best_prefix
        best_latex   = _best_latex
        best_fitness = _best_fitness

        if generation_log:
            last         = generation_log[-1]
            final_gsm    = float(last["grad_sensitivity"])
            archive_size = int(last["archive_size"])
        else:
            final_gsm    = 0.0
            archive_size = 0

        run = AblationRun(
            config_name                = config.name,
            config_label               = config.label,
            best_formula_prefix        = best_prefix,
            best_formula_latex         = best_latex,
            best_fitness               = best_fitness,
            final_archive_size         = archive_size,
            final_gradient_sensitivity = final_gsm,
            elapsed_sec                = elapsed,
            generation_log             = generation_log,
        )

        # Optional BenchmarkSuite comparison
        if self.run_benchmark and best_prefix:
            logger.info("Benchmarking best formula from %s: %s", config.name, best_prefix)
            try:
                suite  = BenchmarkSuite(
                    time_steps  = self.benchmark_steps,
                    n_seeds     = self.benchmark_seeds,
                    base_seed   = self.seed,
                )
                bench_result = suite.compare(best_prefix)
                run.benchmark_rank       = bench_result.rank
                run.benchmark_n_beaten   = bench_result.n_baselines_beaten
                run.benchmark_result_dict = bench_result.to_dict()
                logger.info(
                    "%s  rank=%d/%d  beats=%d baselines",
                    config.name, bench_result.rank, bench_result.n_candidates,
                    bench_result.n_baselines_beaten,
                )
            except Exception as exc:
                logger.warning("Benchmark failed for %s: %s", config.name, exc)

        return run

    def run_all(
        self,
        configs: Optional[list[AblationConfig]] = None,
    ) -> AblationResult:
        """
        Run all three ablation configurations and return combined results.

        Args:
            configs: Subset of configurations to run. Defaults to ABLATION_CONFIGS.
        """
        if configs is None:
            configs = ABLATION_CONFIGS

        t0   = time.time()
        runs: dict[str, AblationRun] = {}

        for cfg in configs:
            logger.info("=" * 60)
            logger.info("Ablation config: %s", cfg.label)
            logger.info("=" * 60)
            run = self.run_single(cfg)
            runs[cfg.name] = run
            logger.info(
                "Finished %s in %.1fs — best_fitness=%.5f  archive=%d  grad_sens=%.3f",
                cfg.name, run.elapsed_sec, run.best_fitness,
                run.final_archive_size, run.final_gradient_sensitivity,
            )

        return AblationResult(
            runs              = runs,
            total_elapsed_sec = time.time() - t0,
        )
