"""
experiments/step2_mnist_transfer.py

Step 2: Real-dataset transfer test -- MNIST.

Tests whether the formula discovered in Step 1 (on synthetic Gaussian proxy)
transfers to a real image classification task. The formula was never trained
on images -- this is a genuine zero-shot transfer test.

Candidates evaluated:
  - SymboLR discovered formula  (from Step 1 JSON, or --formula)
  - Cosine Annealing            (strongest baseline in Phase 4 benchmark)
  - 1-Cycle Policy              (strong adaptive baseline)
  - Step Decay                  (classic milestone schedule)
  - Constant LR                 (simplest baseline)

Training setup:
  Model    : FastConvNet (2 conv + 2 fc, ~200K params, grayscale input)
  Task     : MNIST 60K train / 10K test, 10 classes
  Epochs   : 15 per candidate per seed
  Seeds    : 3  (for error bars; same init per seed across all candidates)
  Optimizer: SGD + momentum 0.9  (same for all candidates)
  Device   : auto (GPU preferred)
  AMP      : enabled on GPU (bf16/fp16 via torch.amp)

Formula warmup:
  The discovered formula uses live gradient norms (g) and loss slopes (dl).
  These signals need per-run normalization (different scale than the proxy task).
  The first warmup_fraction of training steps run at base_lr to fit NormStats.

Expected runtime:
  GPU (RTX 4070): ~3-5 min total
  CPU           : ~30-45 min total

Usage:
  python experiments/step2_mnist_transfer.py
  python experiments/step2_mnist_transfer.py --formula "cos * 3.14159 t"
  python experiments/step2_mnist_transfer.py --epochs 20 --seeds 5
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(legacy_windows=False)
logging.basicConfig(level=logging.WARNING)

# ── Project imports ───────────────────────────────────────────────────────────

from src.symbolr.artifacts.prefix_parser import evaluate_formula
from src.symbolr.evaluators.gradient_aware import _NormStats
from src.symbolr.baselines.schedules import (
    cosine_annealing, one_cycle, step_decay, constant_lr
)
from src.symbolr.torch_impl.models import FastConvNet

# ── CLI args ──────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="SymboLR Step 2: MNIST Transfer Test")
    p.add_argument("--formula",          type=str,   default="",
                   help="Formula prefix string. Default: auto-loaded from latest Step 1 JSON.")
    p.add_argument("--epochs",           type=int,   default=15,
                   help="Training epochs per candidate per seed.")
    p.add_argument("--seeds",            type=int,   default=3,
                   help="Number of seeds (controls error bars).")
    p.add_argument("--batch-size",       type=int,   default=128)
    p.add_argument("--base-lr",          type=float, default=0.05,
                   help="Peak LR for baselines and warmup phase of formula.")
    p.add_argument("--warmup-fraction",  type=float, default=0.05,
                   help="Fraction of total steps for formula warmup (NormStats fitting).")
    p.add_argument("--data-dir",         type=str,   default="data",
                   help="Directory for MNIST download.")
    p.add_argument("--output-dir",       type=str,   default="research_journal/experiments")
    p.add_argument("--no-amp",           action="store_true",
                   help="Disable AMP (automatic mixed precision) even on GPU.")
    return p.parse_args()


# ── Auto-load formula from Step 1 output ─────────────────────────────────────

def _load_step1_formula(output_dir: str = "research_journal/experiments") -> str:
    files = sorted(glob.glob(os.path.join(output_dir, "step1_validation_*.json")))
    if not files:
        return ""
    with open(files[-1]) as f:
        data = json.load(f)
    prefix = data.get("evolution", {}).get("best_prefix", "")
    if prefix:
        console.print(
            f"  [dim]Auto-loaded formula from {os.path.basename(files[-1])}[/dim]"
        )
    return prefix


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_mnist(data_dir: str, batch_size: int, device: torch.device):
    from torchvision import datasets, transforms

    os.makedirs(data_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    with console.status("[dim]Downloading / loading MNIST...[/dim]"):
        train_full = datasets.MNIST(data_dir, train=True,  download=True, transform=transform)
        test_ds    = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # 80/20 train/val split from training set
    n_val   = int(0.2 * len(train_full))
    n_train = len(train_full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        train_full, [n_train, n_val],
        generator=torch.Generator().manual_seed(0),
    )

    pin = device.type == "cuda"
    kw = dict(batch_size=batch_size, num_workers=0, pin_memory=pin)
    train_loader = torch.utils.data.DataLoader(train_ds, shuffle=True,  **kw)
    val_loader   = torch.utils.data.DataLoader(val_ds,   shuffle=False, **kw)
    test_loader  = torch.utils.data.DataLoader(test_ds,  shuffle=False, **kw)

    console.print(
        f"  [dim]MNIST: {n_train} train / {n_val} val / {len(test_ds)} test  "
        f"| {len(train_loader)} batches/epoch[/dim]"
    )
    return train_loader, val_loader, test_loader


# ── Model factory ─────────────────────────────────────────────────────────────

def _make_model(seed: int, device: torch.device) -> nn.Module:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = FastConvNet(in_channels=1, num_classes=10).to(device)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def _evaluate(model: nn.Module, loader, device: torch.device, amp: bool) -> tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            logits = model(x)
            total_loss += criterion(logits, y).item() * len(y)
        correct += logits.argmax(1).eq(y).sum().item()
        total   += len(y)
    return total_loss / max(total, 1), correct / max(total, 1)


# ── Training loop ─────────────────────────────────────────────────────────────

@dataclass
class EpochRecord:
    epoch:      int
    train_loss: float
    val_loss:   float
    val_acc:    float
    lr_mean:    float   # mean LR used in this epoch
    lr_std:     float   # std of LR across steps (shows adaptivity)


@dataclass
class RunResult:
    """Results of one training run (one candidate, one seed)."""
    name:       str
    seed:       int
    formula:    str
    epochs:     list[EpochRecord]
    test_loss:  float
    test_acc:   float
    elapsed_s:  float
    norm_stats: Optional[dict] = None  # fitted NormStats (formula only)


def _train(
    name:             str,
    model:            nn.Module,
    train_loader,
    val_loader,
    test_loader,
    n_epochs:         int,
    device:           torch.device,
    amp_enabled:      bool,
    lr_fn,            # callable(step, total_steps, g_raw, prev_loss, cur_loss, ns) -> float
    seed:             int,
    formula_str:      str = "",
    warmup_steps:     int = 0,
    base_lr:          float = 0.05,
) -> RunResult:
    """
    General training loop used for both the formula and all baselines.

    For the formula candidate:
      - First warmup_steps run at base_lr collecting NormStats
      - Remaining steps use lr_fn with fitted NormStats

    For baselines:
      - lr_fn returns the precomputed schedule value; warmup_steps=0
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler(enabled=amp_enabled)

    total_steps = n_epochs * len(train_loader)

    # NormStats fitting (formula only)
    log_g_samples: list[float] = []
    dl_samples:    list[float] = []
    ns: Optional[_NormStats]   = None

    step       = 0
    prev_loss  = None
    epoch_recs: list[EpochRecord] = []
    t0         = time.time()

    for epoch in range(n_epochs):
        model.train()
        step_losses: list[float] = []
        step_lrs:    list[float] = []

        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(x)
                loss   = criterion(logits, y)

            scaler.scale(loss).backward()

            # For formula candidates: unscale to get true gradient scale for
            # NormStats collection and gradient norm computation.
            # Baselines skip this (warmup_steps==0) — saves ~6x per step.
            if warmup_steps > 0:
                scaler.unscale_(optimizer)
                # Gradient clipping — prevents parameter explosion from LR spikes.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                g_raw = math.sqrt(sum(
                    p.grad.data.norm(2).item() ** 2
                    for p in model.parameters()
                    if p.grad is not None
                ) + 1e-12)
            else:
                g_raw = 1.0  # unused for baselines

            cur_loss = loss.item()

            # Determine LR for this step
            if warmup_steps > 0 and step < warmup_steps:
                lr = base_lr
                # Collect NormStats samples during warmup
                if prev_loss is not None:
                    dl_samples.append(cur_loss - prev_loss)
                if math.isfinite(g_raw) and g_raw > 0:
                    log_g_samples.append(math.log(g_raw))
                # Fit NormStats at the last warmup step
                if step == warmup_steps - 1:
                    ns = _NormStats.fit(log_g_samples, dl_samples)
            else:
                lr = lr_fn(step, total_steps, g_raw, prev_loss, cur_loss, ns)

            # For formula: clamp to [base_lr*0.001, base_lr*5].
            # Prevents LR spikes from destroying the model while preserving adaptivity.
            if warmup_steps > 0:
                lr = float(max(base_lr * 0.001, min(lr, base_lr * 5.0)))
            elif not math.isfinite(lr) or lr <= 0:
                lr = base_lr * 0.1

            for pg in optimizer.param_groups:
                pg["lr"] = float(lr)

            scaler.step(optimizer)
            scaler.update()

            step_losses.append(cur_loss)
            step_lrs.append(lr)
            prev_loss = cur_loss
            step += 1

        # Epoch-end evaluation
        val_loss, val_acc = _evaluate(model, val_loader, device, amp_enabled)
        train_loss = float(np.mean(step_losses))
        lr_mean    = float(np.mean(step_lrs))
        lr_std     = float(np.std(step_lrs))

        epoch_recs.append(EpochRecord(epoch + 1, train_loss, val_loss, val_acc, lr_mean, lr_std))

    test_loss, test_acc = _evaluate(model, test_loader, device, amp_enabled)
    elapsed = time.time() - t0

    return RunResult(
        name      = name,
        seed      = seed,
        formula   = formula_str,
        epochs    = epoch_recs,
        test_loss = test_loss,
        test_acc  = test_acc,
        elapsed_s = elapsed,
        norm_stats = (
            {"log_g_mean": ns.log_g_mean, "log_g_std": ns.log_g_std,
             "dl_mean": ns.dl_mean, "dl_std": ns.dl_std}
            if ns else None
        ),
    )


# ── Candidate definitions ─────────────────────────────────────────────────────

def _make_formula_lr_fn(prefix: str, base_lr: float):
    """Return an LR function that evaluates the prefix formula with live signals."""
    def lr_fn(step, total_steps, g_raw, prev_loss, cur_loss, ns: Optional[_NormStats]):
        t_norm = step / max(total_steps - 1, 1)
        if ns is not None:
            g_norm  = ns.normalize_g(g_raw)
            dl_raw  = (cur_loss - prev_loss) if prev_loss is not None else 0.0
            dl_norm = ns.normalize_dl(dl_raw)
        else:
            g_norm  = math.log(g_raw + 1e-8)
            dl_norm = 0.0
        return evaluate_formula(prefix, t=t_norm, g=g_norm, dl=dl_norm)
    return lr_fn


def _make_baseline_lr_fn(schedule_fn, total_steps: int, base_lr: float, **kwargs):
    """Pre-compute a baseline LR array and return a lookup function."""
    t_array  = np.linspace(0.0, 1.0, total_steps, dtype=np.float64)
    lr_array = np.clip(schedule_fn(t_array, **kwargs), 1e-7, 10.0)

    def lr_fn(step, _total, _g, _prev, _cur, _ns):
        return float(lr_array[min(step, len(lr_array) - 1)])

    return lr_fn


# ── Aggregated result ─────────────────────────────────────────────────────────

@dataclass
class CandidateResult:
    """Aggregated statistics across all seeds for one candidate."""
    name:            str
    formula:         str
    test_acc_mean:   float
    test_acc_std:    float
    test_acc_max:    float
    val_acc_final_mean: float
    train_loss_final_mean: float
    lr_mean_final:   float  # mean LR in last epoch
    lr_std_final:    float  # std of LR in last epoch (adaptivity proxy)
    elapsed_mean_s:  float
    per_seed:        list[RunResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name":              self.name,
            "formula":           self.formula,
            "test_acc_mean":     self.test_acc_mean,
            "test_acc_std":      self.test_acc_std,
            "test_acc_max":      self.test_acc_max,
            "val_acc_final_mean": self.val_acc_final_mean,
            "train_loss_final_mean": self.train_loss_final_mean,
            "lr_mean_final":     self.lr_mean_final,
            "lr_std_final":      self.lr_std_final,
            "elapsed_mean_s":    self.elapsed_mean_s,
            "per_seed": [
                {
                    "seed":      r.seed,
                    "test_acc":  r.test_acc,
                    "test_loss": r.test_loss,
                    "elapsed_s": r.elapsed_s,
                    "norm_stats": r.norm_stats,
                    "epoch_history": [
                        {"epoch": e.epoch, "train_loss": e.train_loss,
                         "val_loss": e.val_loss, "val_acc": e.val_acc,
                         "lr_mean": e.lr_mean, "lr_std": e.lr_std}
                        for e in r.epochs
                    ],
                }
                for r in self.per_seed
            ],
        }


def _aggregate(name: str, formula: str, runs: list[RunResult]) -> CandidateResult:
    accs    = [r.test_acc for r in runs]
    v_accs  = [r.epochs[-1].val_acc       for r in runs if r.epochs]
    t_loss  = [r.epochs[-1].train_loss    for r in runs if r.epochs]
    lr_m    = [r.epochs[-1].lr_mean       for r in runs if r.epochs]
    lr_s    = [r.epochs[-1].lr_std        for r in runs if r.epochs]

    return CandidateResult(
        name              = name,
        formula           = formula,
        test_acc_mean     = float(np.mean(accs)),
        test_acc_std      = float(np.std(accs)),
        test_acc_max      = float(np.max(accs)),
        val_acc_final_mean = float(np.mean(v_accs)) if v_accs else 0.0,
        train_loss_final_mean = float(np.mean(t_loss)) if t_loss else 0.0,
        lr_mean_final     = float(np.mean(lr_m)) if lr_m else 0.0,
        lr_std_final      = float(np.mean(lr_s)) if lr_s else 0.0,
        elapsed_mean_s    = float(np.mean([r.elapsed_s for r in runs])),
        per_seed          = runs,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args   = _parse_args()
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Auto-load formula
    formula = args.formula.strip()
    if not formula:
        formula = _load_step1_formula(args.output_dir)
    if not formula:
        formula = "cos * 3.14159 t"
        console.print(
            "[yellow]No formula provided and no Step 1 output found. "
            f"Using fallback: {formula}[/yellow]"
        )

    formula_tokens = set(formula.split())
    is_grad_aware  = bool(formula_tokens & {"g", "dl"})

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp       = (device.type == "cuda") and not args.no_amp

    # ── Header ────────────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold white]SymboLR Step 2: MNIST Transfer Test[/bold white]\n\n"
        f"Formula: [magenta]{formula[:80]}[/magenta]\n"
        f"Gradient-aware: {'[green]YES[/green]' if is_grad_aware else '[yellow]NO (time-only)[/yellow]'}\n"
        f"Device: {device}  AMP: {amp}  "
        f"Epochs: {args.epochs}  Seeds: {args.seeds}\n"
        f"Started: {datetime.now().strftime('%H:%M:%S')}",
        title="[bold]SymboLR[/bold]",
        border_style="cyan",
    ))

    # ── Load MNIST ────────────────────────────────────────────────────────────
    console.rule("[cyan]Loading MNIST[/cyan]")
    train_loader, val_loader, test_loader = _load_mnist(args.data_dir, args.batch_size, device)

    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = int(total_steps * args.warmup_fraction)

    console.print(
        f"  [dim]Total steps: {total_steps}  |  "
        f"Formula warmup: {warmup_steps} steps "
        f"({args.warmup_fraction*100:.0f}%)[/dim]"
    )

    # ── Define candidates ─────────────────────────────────────────────────────
    candidates = [
        {
            "name":          "SymboLR (discovered)",
            "formula":       formula,
            "lr_fn_factory": lambda: _make_formula_lr_fn(formula, args.base_lr),
            "warmup_steps":  warmup_steps,
            "is_formula":    True,
        },
        {
            "name":    "Cosine Annealing",
            "formula": "cosine_annealing",
            "lr_fn_factory": lambda: _make_baseline_lr_fn(
                cosine_annealing, total_steps, args.base_lr,
                lr_max=args.base_lr, lr_min=args.base_lr * 0.01,
            ),
            "warmup_steps": 0,
            "is_formula":   False,
        },
        {
            "name":    "1-Cycle",
            "formula": "one_cycle",
            "lr_fn_factory": lambda: _make_baseline_lr_fn(
                one_cycle, total_steps, args.base_lr,
                lr_max=args.base_lr, lr_min=args.base_lr * 0.01, warmup_frac=0.3,
            ),
            "warmup_steps": 0,
            "is_formula":   False,
        },
        {
            "name":    "Step Decay",
            "formula": "step_decay",
            "lr_fn_factory": lambda: _make_baseline_lr_fn(
                step_decay, total_steps, args.base_lr,
                lr_init=args.base_lr, gamma=0.3, n_steps=3,
            ),
            "warmup_steps": 0,
            "is_formula":   False,
        },
        {
            "name":    "Constant LR",
            "formula": "constant_lr",
            "lr_fn_factory": lambda: _make_baseline_lr_fn(
                constant_lr, total_steps, args.base_lr,
                lr=args.base_lr * 0.2,  # same total "budget" as decaying schedules
            ),
            "warmup_steps": 0,
            "is_formula":   False,
        },
    ]

    # ── Training loop ─────────────────────────────────────────────────────────
    all_results: list[CandidateResult] = []
    total_t0 = time.time()

    for cand in candidates:
        console.rule(f"[bold cyan]{cand['name']}[/bold cyan]")
        seed_runs: list[RunResult] = []

        # Progress table (per-epoch, one row per epoch of latest seed)
        progress_table = Table(box=None, padding=(0, 1), show_header=True)
        progress_table.add_column("Ep",   justify="right", style="cyan",   width=3)
        progress_table.add_column("T-Loss",justify="right",style="yellow", width=8)
        progress_table.add_column("V-Loss",justify="right",style="green",  width=8)
        progress_table.add_column("V-Acc", justify="right",style="green",  width=7)
        progress_table.add_column("LR-mean",justify="right",style="magenta",width=9)
        progress_table.add_column("LR-std", justify="right",style="dim",   width=8)

        for seed_idx in range(args.seeds):
            seed = 100 * seed_idx + 42  # deterministic seeds: 42, 142, 242

            model  = _make_model(seed, device)
            lr_fn  = cand["lr_fn_factory"]()

            result = _train(
                name          = cand["name"],
                model         = model,
                train_loader  = train_loader,
                val_loader    = val_loader,
                test_loader   = test_loader,
                n_epochs      = args.epochs,
                device        = device,
                amp_enabled   = amp,
                lr_fn         = lr_fn,
                seed          = seed,
                formula_str   = cand["formula"],
                warmup_steps  = cand["warmup_steps"],
                base_lr       = args.base_lr,
            )
            seed_runs.append(result)

            # Print last epoch of this seed to the progress table
            last = result.epochs[-1]
            progress_table.add_row(
                f"s{seed_idx+1}/e{last.epoch}",
                f"{last.train_loss:.4f}",
                f"{last.val_loss:.4f}",
                f"{last.val_acc*100:.2f}%",
                f"{last.lr_mean:.5f}",
                f"{last.lr_std:.5f}",
            )

        console.print(progress_table)

        agg = _aggregate(cand["name"], cand["formula"], seed_runs)
        all_results.append(agg)

        console.print(
            f"  Test: [bold green]{agg.test_acc_mean*100:.2f}%[/bold green] "
            f"+/-{agg.test_acc_std*100:.2f}%  "
            f"(best seed: {agg.test_acc_max*100:.2f}%)  "
            f"[dim]{agg.elapsed_mean_s:.1f}s/seed[/dim]"
        )

    # ── Final comparison table ────────────────────────────────────────────────
    console.print()
    console.rule("[bold white]RESULTS[/bold white]")
    console.print()

    sorted_results = sorted(all_results, key=lambda r: r.test_acc_mean, reverse=True)

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("Rank", justify="right", style="dim",     width=4)
    table.add_column("Candidate",             style="cyan",    width=24)
    table.add_column("Test Acc", justify="right", style="green",  width=10)
    table.add_column("+/-",      justify="right", style="dim",    width=7)
    table.add_column("Best",     justify="right", style="dim",    width=8)
    table.add_column("LR-std",   justify="right", style="magenta",width=8)
    table.add_column("Adaptive?",               style="dim",    width=10)

    for rank, r in enumerate(sorted_results, 1):
        is_formula = r.name.startswith("SymboLR")
        adaptive_str = (
            "[green]YES[/green]" if (is_formula and is_grad_aware)
            else "[dim]no[/dim]"
        )
        rank_style = "bold green" if rank == 1 else "dim"
        table.add_row(
            Text(f"#{rank}", style=rank_style),
            Text(r.name, style="bold magenta" if is_formula else "cyan"),
            f"{r.test_acc_mean*100:.2f}%",
            f"+/-{r.test_acc_std*100:.2f}%",
            f"{r.test_acc_max*100:.2f}%",
            f"{r.lr_std_final:.5f}",
            adaptive_str,
        )

    console.print(table)

    # Find SymboLR rank and delta vs top baseline
    symbolr = next((r for r in sorted_results if r.name.startswith("SymboLR")), None)
    best_baseline = next((r for r in sorted_results if not r.name.startswith("SymboLR")), None)

    if symbolr and best_baseline:
        delta_pct = (symbolr.test_acc_mean - best_baseline.test_acc_mean) * 100
        symbolr_rank = sorted_results.index(symbolr) + 1
        console.print()
        if delta_pct > 0:
            console.print(
                f"  [green]+{delta_pct:.2f}% vs best baseline ({best_baseline.name})[/green]"
                f"  SymboLR ranks #{symbolr_rank}/{len(sorted_results)}"
            )
        else:
            console.print(
                f"  [yellow]{delta_pct:.2f}% vs best baseline ({best_baseline.name})[/yellow]"
                f"  SymboLR ranks #{symbolr_rank}/{len(sorted_results)}"
            )

        # Adaptivity note
        if is_grad_aware and symbolr.lr_std_final > best_baseline.lr_std_final:
            console.print(
                f"  [dim]LR-std {symbolr.lr_std_final:.5f} > "
                f"{best_baseline.lr_std_final:.5f} ({best_baseline.name}): "
                f"formula adapts to gradient dynamics[/dim]"
            )

    # ── Interpretation ────────────────────────────────────────────────────────
    console.print()
    console.rule("[bold white]INTERPRETATION[/bold white]")
    if symbolr:
        sym_rank = sorted_results.index(symbolr) + 1
        n        = len(sorted_results)
        if sym_rank == 1:
            console.print(Panel(
                "[bold green]SymboLR formula ranks #1 on MNIST[/bold green]\n\n"
                "The formula discovered on synthetic Gaussian clusters transfers\n"
                "to real image classification.\n\n"
                f"Formula: {formula}\n\n"
                "[dim]This validates the gradient-health-aware discovery approach:[/dim]\n"
                "[dim]the proxy task produced a formula that generalizes.[/dim]",
                border_style="green",
            ))
        elif sym_rank <= n // 2:
            console.print(Panel(
                f"[bold yellow]SymboLR formula ranks #{sym_rank}/{n} on MNIST[/bold yellow]\n\n"
                "Formula is competitive but not the top performer.\n"
                "The proxy-task formula partially transfers to MNIST.\n\n"
                "[dim]Next steps:[/dim]\n"
                "[dim]  - Run more evolution generations for a better formula[/dim]\n"
                "[dim]  - Try CIFAR-10 for a harder transfer test[/dim]",
                border_style="yellow",
            ))
        else:
            console.print(Panel(
                f"[bold red]SymboLR formula ranks #{sym_rank}/{n} on MNIST[/bold red]\n\n"
                "The proxy-task formula does not transfer well to MNIST.\n\n"
                "[dim]Likely cause: proxy task (Gaussian clusters) is too easy -- any\n"
                "non-zero LR converges, so the formula was not under pressure to\n"
                "learn gradient-adaptive behavior that generalizes.\n"
                "Fix: harder proxy task OR more training steps in the proxy.[/dim]",
                border_style="red",
            ))

    # ── Save ──────────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_t0
    output = {
        "run_timestamp":   run_ts,
        "formula":         formula,
        "is_grad_aware":   is_grad_aware,
        "args":            vars(args),
        "total_elapsed_s": total_elapsed,
        "device":          str(device),
        "results": [r.to_dict() for r in all_results],
        "ranking": [r.name for r in sorted_results],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"step2_mnist_{run_ts}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    console.print(
        f"\n  [dim]Results saved -> {out_path}[/dim]"
        f"\n  [dim]Total runtime: {int(total_elapsed//60)}m {int(total_elapsed%60):02d}s[/dim]\n"
    )


if __name__ == "__main__":
    main()
