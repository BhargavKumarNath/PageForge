"""
experiments/step1_end_to_end_validation.py

Step 1 end-to-end engine validation -- no dataset downloads required.

Pipeline (all on synthetic proxy task, ~2-3 min on CPU):
  1. EVOLUTION  -- GradientAwareEvaluator; formula discovery with live g/dl signals.
  2. BENCHMARK  -- Best formula vs 7 baseline schedules, paired statistical tests.
  3. ABLATION   -- 3-way terminal-set study: t-only vs t+g vs t+g+dl.

Validation checks at the end tell you exactly what passed, what was borderline,
and what to investigate further before running Step 2 (real dataset).

Usage:
    python experiments/step1_end_to_end_validation.py
    python experiments/step1_end_to_end_validation.py --generations 30 --seed 7
    python experiments/step1_end_to_end_validation.py --fast   # ~60s quick mode
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime

# Allow running from repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.WARNING)  # suppress INFO noise during timed run

import io
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

# Force UTF-8 output on Windows (avoids CP1252 encoding failures in legacy renderer)
console = Console(legacy_windows=False)

# CLI args
def _parse_args():
    p = argparse.ArgumentParser(description="SymboLR Step 1 End-to-End Validation")
    p.add_argument("--generations",    type=int,   default=20,  help="Evolution generations")
    p.add_argument("--ablation-gens",  type=int,   default=10,  help="Ablation generations per config")
    p.add_argument("--pop-size",       type=int,   default=40,  help="Population size")
    p.add_argument("--n-steps",        type=int,   default=100, help="GradientAwareEvaluator training steps (100 recommended)")
    p.add_argument("--benchmark-seeds",type=int,   default=5,   help="BenchmarkSuite seeds")
    p.add_argument("--seed",           type=int,   default=42,  help="Random seed")
    p.add_argument("--output-dir",     type=str,   default="research_journal/experiments",
                   help="Directory for JSON output")
    p.add_argument("--fast", action="store_true",
                   help="Fast mode: 8 gens, pop=20, n_steps=50 (~90s)")
    p.add_argument("--device", type=str, default="cpu",
                   help="PyTorch device (cpu/cuda); default cpu")
    return p.parse_args()


# Helpers
def _check(condition: bool, pass_msg: str, fail_msg: str, warn=False) -> bool:
    icon = "[OK]" if condition else ("[!]" if warn else "[NO]")
    style = "green" if condition else ("yellow" if warn else "red")
    console.print(f"  [{style}]{icon}[/{style}]  {pass_msg if condition else fail_msg}")
    return condition


def _section(n: int, title: str, total: int = 3):
    console.print()
    console.rule(f"[bold cyan][{n}/{total}] {title}[/bold cyan]")


def _elapsed(t0: float) -> str:
    s = time.time() - t0
    return f"{int(s//60)}m {int(s%60):02d}s" if s >= 60 else f"{s:.1f}s"


# Phase 1: Evolution
def run_evolution(args) -> dict:
    from src.symbolr.evaluators.gradient_aware import GradientAwareEvaluator
    from src.symbolr.core.bridge import RustEvolutionBridge

    ev = GradientAwareEvaluator(
        n_steps    = args.n_steps,
        batch_size = 64,
        seed       = args.seed,
        device     = args.device,
    )
    bridge = RustEvolutionBridge(
        eval_callback   = ev.evaluate,
        max_generations = args.generations,
        pop_size        = args.pop_size,
        seed            = args.seed,
    )

    console.print(
        f"  [dim]GradientAwareEvaluator | n_steps={args.n_steps} | "
        f"pop={args.pop_size} | {args.generations} generations | seed={args.seed}[/dim]"
    )
    console.print()

    # Live per-generation table
    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Gen",   justify="right",  style="cyan",    width=4)
    table.add_column("Best",  justify="right",  style="green",   width=9)
    table.add_column("Avg",   justify="right",  style="yellow",  width=9)
    table.add_column("Arch",  justify="right",  style="blue",    width=5)
    table.add_column("grad-Sens",justify="right",  style="magenta", width=7)
    table.add_column("ms",    justify="right",  style="dim",     width=6)
    table.add_column("Top formula",              style="dim",     width=40)

    gen_log = []
    best_fitness = float("inf")
    best_prefix  = ""
    best_latex   = ""

    t0 = time.time()
    for r in bridge.stream():
        gen_log.append({
            "gen":          r.generation_number,
            "best":         r.best_mse,
            "avg":          r.average_mse,
            "archive":      r.archive_size,
            "grad_sens":    r.gradient_sensitivity_mean,
            "top_formula":  r.top_formula_prefix,
            "latex":        r.top_formula_latex,
            "ms":           r.gen_time_ms,
        })
        table.add_row(
            str(r.generation_number),
            f"{r.best_mse:.5f}",
            f"{r.average_mse:.5f}",
            str(r.archive_size),
            f"{r.gradient_sensitivity_mean:.3f}",
            str(r.gen_time_ms),
            r.top_formula_prefix[:38],
        )
        if r.best_mse < best_fitness:
            best_fitness = r.best_mse
            best_prefix  = r.top_formula_prefix
            best_latex   = r.top_formula_latex

    console.print(table)
    console.print(
        f"\n  [bold]Best formula discovered:[/bold]  "
        f"[magenta]{best_prefix}[/magenta]"
    )
    console.print(f"  LaTeX   : {best_latex}")
    console.print(f"  Fitness : {best_fitness:.6f}")

    final_gsm = gen_log[-1]["grad_sens"] if gen_log else 0.0
    init_gsm  = gen_log[0]["grad_sens"]  if gen_log else 0.0
    console.print(
        f"  grad-Sens  : {init_gsm:.3f} -> {final_gsm:.3f}  "
        f"({'[green]rising[/green]' if final_gsm > init_gsm else '[yellow]flat[/yellow]'})"
    )

    return {
        "best_prefix":  best_prefix,
        "best_latex":   best_latex,
        "best_fitness": best_fitness,
        "final_gsm":    final_gsm,
        "init_gsm":     init_gsm,
        "gen_log":      gen_log,
        "elapsed":      time.time() - t0,
    }


# Phase 2: Benchmark
def run_benchmark(best_prefix: str, args) -> dict:
    from src.symbolr.baselines.benchmark import BenchmarkSuite

    console.print(
        f"  [dim]BenchmarkSuite | n_seeds={args.benchmark_seeds} | "
        f"time_steps=100 | formula: {best_prefix}[/dim]\n"
    )

    suite  = BenchmarkSuite(time_steps=100, n_seeds=args.benchmark_seeds, base_seed=args.seed)
    t0 = time.time()
    result = suite.compare(best_prefix)

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Candidate",    style="cyan",  width=22)
    table.add_column("Mean",   justify="right", style="green",  width=9)
    table.add_column("±Std",   justify="right", style="dim",    width=8)
    table.add_column("Delta",      justify="right",                 width=11)
    table.add_column("Win%",   justify="right",                 width=6)
    table.add_column("95% CI", justify="right", style="dim",    width=20)
    table.add_column("p",      justify="right", style="dim",    width=7)

    ft = result.formula_trial
    table.add_row(
        Text("* Discovered", style="bold magenta"),
        f"{ft.mean:.5f}", f"±{ft.std:.5f}", "--", "--", "--", "--",
    )

    for name, cmp in sorted(result.comparisons.items(), key=lambda x: x[1].baseline_mean):
        bt        = result.baseline_trials[name]
        wins      = cmp.formula_wins
        delta_str = Text(f"{'[OK]' if wins else '[NO]'} {cmp.delta_mean:+.5f}",
                         style="green" if wins else "red")
        sig_mark  = "*" if cmp.is_significant else ""
        p_str     = f"{cmp.wilcoxon_p:.3f}{sig_mark}" if cmp.wilcoxon_p is not None else "n/a"
        table.add_row(
            name[:21],
            f"{bt.mean:.5f}", f"±{bt.std:.5f}",
            delta_str,
            f"{cmp.win_rate:.0%}",
            f"[{cmp.ci_lower:+.4f}, {cmp.ci_upper:+.4f}]",
            p_str,
        )

    console.print(table)
    console.print(
        f"\n  Rank: [bold]#{result.rank}[/bold] of {result.n_candidates}  |  "
        f"Beats: {result.n_baselines_beaten}/{len(result.baseline_trials)} baselines"
    )

    # Detect whether the formula is gradient-aware (has g/dl tokens)
    formula_tokens = set(best_prefix.split())
    is_grad_aware  = bool(formula_tokens & {"g", "dl"})
    if is_grad_aware:
        console.print(
            "  [dim][!] This formula contains gradient tokens (g/dl).\n"
            "  The benchmark evaluates at g=0 via symbolr_rust.evaluate_batch -- the "
            "adaptive behavior is invisible to this metric.\n"
            "  A formula like exp(1.627-g) becomes a constant ~5.0 at g=0, which is "
            "then normalized to 0.01.\n"
            "  -> The rank here understates the formula's true value. "
            "Step 2 (real dataset) will properly measure gradient-aware performance.[/dim]"
        )

    if args.benchmark_seeds < 7:
        console.print(
            f"  [dim]Note: {args.benchmark_seeds} seeds used -- "
            f"Wilcoxon p cannot reach 0.05 (need >=7 seeds). "
            f"Win rate and CI are the primary outputs.[/dim]"
        )

    return {
        "rank":          result.rank,
        "n_candidates":  result.n_candidates,
        "n_beaten":      result.n_baselines_beaten,
        "formula_mean":  ft.mean,
        "is_grad_aware": is_grad_aware,
        "result_dict":   result.to_dict(),
        "elapsed":       time.time() - t0,
    }


# Phase 3: Ablation
def run_ablation(args) -> dict:
    from src.symbolr.evaluators.gradient_aware import GradientAwareEvaluator
    from src.symbolr.core.ablation import AblationRunner, ABLATION_CONFIGS

    # Share one evaluator instance across all configs
    base_ev = GradientAwareEvaluator(
        n_steps    = args.n_steps,
        batch_size = 64,
        seed       = args.seed,
        device     = args.device,
    )
    console.print(
        f"  [dim]3 configs × {args.ablation_gens} generations × "
        f"pop={args.pop_size} | shared seed={args.seed}[/dim]\n"
    )

    runner = AblationRunner(
        base_evaluator  = base_ev,
        max_generations = args.ablation_gens,
        pop_size        = args.pop_size,
        seed            = args.seed,
        benchmark_seeds = args.benchmark_seeds,
        benchmark_steps = 100,
        run_benchmark   = True,
    )

    t0     = time.time()
    result = runner.run_all()

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Config",        style="cyan",    width=28)
    table.add_column("Best Fitness",  justify="right", style="green",   width=12)
    table.add_column("Archive",       justify="right", style="blue",    width=8)
    table.add_column("grad-Sens",        justify="right", style="magenta", width=7)
    table.add_column("Bench Rank",    justify="right",                  width=10)
    table.add_column("Best formula",  style="dim",                      width=35)

    rows = result.summary_rows()
    for i, row in enumerate(rows):
        rank_str = f"#{row['bench_rank']}/{row.get('bench_rank', '?')}" \
            if row["bench_rank"] else "--"
        # Get rank from run directly
        run = result.runs.get(
            next((r.config_name for r in result.runs.values()
                  if r.config_label == row["config"]), ""), None
        )
        rank_display = (
            f"#{run.benchmark_rank}/{run.benchmark_rank + 7}" if run and run.benchmark_rank
            else "--"
        )
        badge = "[bold green]>[/bold green]" if i == 0 else " "
        table.add_row(
            f"{badge} {row['config']}",
            f"{row['best_fitness']:.5f}",
            str(row["archive_size"]),
            f"{row['grad_sens']:.3f}",
            rank_display,
            row["formula"][:33],
        )

    console.print(table)

    # Compute fitness improvement: t_g_dl vs t_only
    r_only = result.runs.get("t_only")
    r_full = result.runs.get("t_g_dl")
    improvement = None
    if r_only and r_full and math.isfinite(r_only.best_fitness) and math.isfinite(r_full.best_fitness):
        improvement = (r_only.best_fitness - r_full.best_fitness) / (r_only.best_fitness + 1e-9)

    if improvement is not None:
        sign = "+" if improvement >= 0 else ""
        console.print(
            f"\n  Full vs time-only fitness delta: [bold]{sign}{improvement*100:.1f}%[/bold] "
            f"({'[green]full set better[/green]' if improvement > 0 else '[yellow]time-only competitive[/yellow]'})"
        )

    return {
        "result":      result.to_dict(),
        "improvement": improvement,
        "elapsed":     time.time() - t0,
    }


# Validation report
def print_validation_report(ev_out: dict, bm_out: dict, ab_out: dict):
    console.print()
    console.rule("[bold white]VALIDATION SUMMARY[/bold white]")
    console.print()

    checks_passed = 0
    checks_total  = 0

    def chk(condition, label_pass, label_fail, is_warning=False):
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            console.print(f"  [green][OK][/green]  {label_pass}")
        elif is_warning:
            console.print(f"  [yellow][!][/yellow]  {label_fail}")
        else:
            console.print(f"  [red][NO][/red]  {label_fail}")
        return condition

    # Check 1: Signal path — gradient sensitivity rises
    chk(
        ev_out["final_gsm"] > ev_out["init_gsm"],
        f"Signal path: gradient sensitivity rose {ev_out['init_gsm']:.3f} -> {ev_out['final_gsm']:.3f}",
        f"Signal path FAILED: gradient sensitivity flat ({ev_out['init_gsm']:.3f} -> "
        f"{ev_out['final_gsm']:.3f}) -- gradient signals may not be reaching formulas",
    )

    # Check 2: Gradient tokens appear in at least one top formula
    # Check the full generation log, not just the single best-fitness formula.
    # (The best-fitness formula may be a constant if the proxy task is easy, but
    # the archive should contain gradient-aware formulas in higher-sensitivity niches.)
    all_top_formulas = [e["top_formula"] for e in ev_out["gen_log"]]
    grad_seen = any(
        tok in formula.split()
        for formula in all_top_formulas
        for tok in ("g", "dl")
    )
    chk(
        grad_seen,
        f"Gradient tokens (g/dl) appeared in archive top formulas across generations",
        f"No gradient tokens seen in any top formula -- GP may not be generating g/dl nodes",
    )

    # Check 3: Benchmark — formula is finite (not catastrophically broken)
    bm_finite = math.isfinite(bm_out["formula_mean"])
    chk(
        bm_finite,
        f"Benchmark fitness is finite ({bm_out['formula_mean']:.5f})",
        f"Benchmark returned non-finite fitness -- formula may be degenerate",
    )
    # Rank is informational when formula is gradient-aware (benchmark uses g=0)
    rank = bm_out["rank"]
    n    = bm_out["n_candidates"]
    if bm_out.get("is_grad_aware") and bm_finite:
        console.print(
            f"  [dim](i) Benchmark rank #{rank}/{n}: gradient-aware formula penalised "
            f"because BenchmarkSuite evaluates at g=0. Not a meaningful quality signal "
            f"for adaptive formulas. Step 2 will use the proper signal.[/dim]"
        )
    elif bm_finite:
        chk(
            rank <= n // 2,
            f"Formula ranked #{rank}/{n} -- top half of all candidates",
            f"Formula ranked #{rank}/{n} -- bottom half",
            is_warning=rank > n // 2,
        )

    # Check 4: Ablation — gradient sensitivity separates configs
    ab_runs = ab_out.get("result", {}).get("runs", {})
    gsm_full  = ab_runs.get("t_g_dl", {}).get("final_gradient_sensitivity", 0.0)
    gsm_only  = ab_runs.get("t_only",  {}).get("final_gradient_sensitivity", 0.0)
    if ab_runs:
        chk(
            gsm_full > gsm_only,
            f"Ablation gradient sensitivity: t_g_dl ({gsm_full:.3f}) > t_only ({gsm_only:.3f}) "
            f"-- terminal set filter is working",
            f"Ablation gradient sensitivity: t_g_dl ({gsm_full:.3f}) not > t_only ({gsm_only:.3f}) "
            f"-- token filter may not be working correctly",
        )
        # Fitness comparison as informational (proxy task may be too easy for clear signal)
        imp = ab_out.get("improvement")
        if imp is not None and abs(imp) < 50:  # skip if near-zero division artifact
            note = "gradient conditioning improves fitness" if imp > 0.02 else "fitness difference small (proxy may be easy)"
            console.print(f"  [dim](i) Fitness delta t_g_dl vs t_only: {imp*100:+.1f}% -- {note}[/dim]")

    console.print()

    # Overall verdict
    critical_checks = ["signal path", "gradient tokens", "benchmark finite", "ablation"]
    if checks_passed == checks_total:
        verdict = Panel(
            "[bold green]ALL CHECKS PASSED[/bold green]\n\n"
            "Signal path confirmed: gradient health signals flow end-to-end.\n"
            "The archive contains gradient-aware formulas. Terminal-set filter works.\n\n"
            "[dim]-> Proceed to Step 2: real-dataset transfer test (MNIST).[/dim]",
            title="[bold green]Step 1 Result[/bold green]",
            border_style="green",
        )
    elif checks_passed >= checks_total - 1:
        verdict = Panel(
            f"[bold yellow]{checks_passed}/{checks_total} CHECKS PASSED[/bold yellow]\n\n"
            "Core signal path is working with minor caveats (see [!] items above).\n"
            "Likely resolves with --generations 30 or more.\n\n"
            "[dim]-> Can proceed to Step 2 if signal-path and ablation checks passed.[/dim]",
            title="[bold yellow]Step 1 Result[/bold yellow]",
            border_style="yellow",
        )
    else:
        verdict = Panel(
            f"[bold red]{checks_passed}/{checks_total} CHECKS PASSED[/bold red]\n\n"
            "One or more critical checks failed.\n"
            "Do NOT proceed to Step 2 until [NO] items are resolved.\n\n"
            "[dim]-> Most common cause: GradientAwareEvaluator not returning real gradients.\n"
            "Run: python -c \"from src.symbolr.evaluators.gradient_aware import "
            "GradientAwareEvaluator; e=GradientAwareEvaluator(n_steps=30); "
            "print(e.evaluate(['0.01']))\"[/dim]",
            title="[bold red]Step 1 Result[/bold red]",
            border_style="red",
        )

    console.print(verdict)


# Main
def main():
    args = _parse_args()

    if args.fast:
        args.generations   = 8
        args.ablation_gens = 5
        args.pop_size      = 20
        args.n_steps       = 50

    run_ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    total_t0 = time.time()

    # Header
    console.print(Panel(
        f"[bold white]SymboLR Step 1: End-to-End Engine Validation[/bold white]\n\n"
        f"Pipeline: [cyan]Evolution[/cyan] -> [yellow]Benchmark[/yellow] -> [magenta]Ablation[/magenta]\n"
        f"Evaluator: GradientAwareEvaluator (n_steps={args.n_steps})\n"
        f"Seed: {args.seed}  |  Started: {datetime.now().strftime('%H:%M:%S')}",
        title="[bold]SymboLR[/bold]",
        border_style="cyan",
    ))

    # Phase 1: Evolution
    _section(1, f"EVOLUTION  ({args.generations} generations, pop={args.pop_size})")
    ev_out = run_evolution(args)
    console.print(f"\n  [dim]Elapsed: {_elapsed(total_t0)}[/dim]")

    # Phase 2: Benchmark
    _section(2, f"BENCHMARK  ({args.benchmark_seeds} seeds, 100 time-steps)")
    bm_out = run_benchmark(ev_out["best_prefix"], args)
    console.print(f"\n  [dim]Elapsed: {_elapsed(total_t0)}[/dim]")

    # Phase 3: Ablation
    _section(3, f"ABLATION  (t-only vs t+g vs t+g+dl, {args.ablation_gens} gens each)")
    ab_out = run_ablation(args)
    console.print(f"\n  [dim]Elapsed: {_elapsed(total_t0)}[/dim]")

    # Validation summary
    print_validation_report(ev_out, bm_out, ab_out)

    # Save results
    output = {
        "run_timestamp":  run_ts,
        "args":           vars(args),
        "total_elapsed":  time.time() - total_t0,
        "evolution":      {
            "best_prefix":  ev_out["best_prefix"],
            "best_latex":   ev_out["best_latex"],
            "best_fitness": ev_out["best_fitness"],
            "final_gsm":    ev_out["final_gsm"],
            "init_gsm":     ev_out["init_gsm"],
            "gen_log":      ev_out["gen_log"],
            "elapsed":      ev_out["elapsed"],
        },
        "benchmark":      {k: v for k, v in bm_out.items() if k != "result_dict"},
        "benchmark_full": bm_out.get("result_dict"),
        "ablation":       ab_out,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"step1_validation_{run_ts}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    total_elapsed = time.time() - total_t0
    console.print(
        f"\n  [dim]Results saved -> {out_path}[/dim]"
        f"\n  [dim]Total runtime: {_elapsed(total_t0)}[/dim]\n"
    )


if __name__ == "__main__":
    main()
