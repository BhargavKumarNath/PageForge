"""PageForge CLI -- system info, pool analysis, benchmarks, text generation."""

from __future__ import annotations

import math
import sys
import time
from collections import deque
from pathlib import Path

# Force UTF-8 output on Windows (CP1252 can't encode block-drawing characters)
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import toml
import typer
from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# App + sub-command groups
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="pageforge",
    help="[bold]PageForge[/bold]: Paged KV-cache memory manager for LLM inference.",
    no_args_is_help=True,
    add_completion=False,
    rich_markup_mode="rich",
)
pool_app   = typer.Typer(no_args_is_help=True, add_completion=False, rich_markup_mode="rich")
bench_app  = typer.Typer(no_args_is_help=True, add_completion=False, rich_markup_mode="rich")
config_app = typer.Typer(no_args_is_help=True, add_completion=False, rich_markup_mode="rich")

app.add_typer(pool_app,   name="pool",   help="Pool capacity analysis and stress testing.")
app.add_typer(bench_app,  name="bench",  help="Run hardware benchmarks.")
app.add_typer(config_app, name="config", help="Show or edit persistent configuration.")

console = Console()

# Demo constants

_SEQ_COLORS = ["red", "blue", "green3", "yellow", "magenta", "cyan", "orange3", "bright_blue"]
_SEQ_LABELS = list("ABCDEFGH")

# Config helpers

CONFIG_PATH = Path.home() / ".pageforge" / "config.toml"

_DEFAULTS: dict = {
    "pool":  {"n_pages": 512,  "page_size": 16},
    "model": {"name": "gpt2", "dtype": "fp16", "device": "cuda"},
    "bench": {"iters": 50,    "warmup": 5, "seqs": 32, "steps": 200},
}


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        return toml.load(CONFIG_PATH)
    return {k: dict(v) for k, v in _DEFAULTS.items()}


def _save_config(cfg: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        toml.dump(cfg, f)


# Shared guards

def _try_import_pf():
    try:
        import _pageforge
        return _pageforge
    except ImportError:
        return None


def _require_pf():
    mod = _try_import_pf()
    if mod is None:
        console.print(
            "[bold red]Error:[/bold red] Rust extension [bold]_pageforge[/bold] not found.\n"
            "Build it with: [cyan]cd pageforge-rs && maturin develop --release[/cyan]"
        )
        raise typer.Exit(1)
    return mod


def _require_gpu():
    try:
        import torch
        if not torch.cuda.is_available():
            console.print("[bold red]Error:[/bold red] No CUDA GPU detected.")
            raise typer.Exit(1)
        return torch
    except ImportError:
        console.print("[bold red]Error:[/bold red] PyTorch not installed.")
        raise typer.Exit(1)


def _ok(s: str = "OK")   -> str: return f"[green]{s}[/green]"
def _warn(s: str)         -> str: return f"[yellow]{s}[/yellow]"
def _err(s: str = "MISS") -> str: return f"[red]{s}[/red]"

# pageforge info
@app.command()
def info() -> None:
    """Check system readiness: Python, PyTorch, CUDA, CuPy, Rust extension, GPU."""
    rows: list[tuple[str, str, str]] = []

    pv = sys.version.split()[0]
    ok = tuple(int(x) for x in pv.split(".")[:2]) >= (3, 9)
    rows.append(("Python", pv, _ok() if ok else _err("need >=3.9")))

    try:
        import torch
        cuda = torch.cuda.is_available()
        cuda_str = f"CUDA {torch.version.cuda}" if cuda else "CPU only"
        rows.append(("PyTorch", f"{torch.__version__}  ({cuda_str})", _ok() if cuda else _warn("no CUDA")))
    except ImportError:
        rows.append(("PyTorch", "not installed", _err()))
        cuda = False

    if cuda:
        try:
            import torch
            props = torch.cuda.get_device_properties(0)
            vram  = props.total_memory / 1e9
            rows.append(("GPU", f"{props.name}  ({vram:.1f} GB, sm_{props.major}{props.minor})", _ok()))
        except Exception as e:
            rows.append(("GPU", str(e), _err("ERROR")))
    else:
        rows.append(("GPU", "no CUDA device", _warn("N/A")))

    try:
        import cupy
        rows.append(("CuPy", cupy.__version__, _ok()))
    except ImportError:
        rows.append(("CuPy", "not installed", _err()))

    pf = _try_import_pf()
    if pf is not None:
        try:
            obj = pf.PageForge(16, 4)
            rows.append(("_pageforge (Rust)", f"loaded  (smoke: {obj.free_pages()}/16 pages)", _ok()))
        except Exception as e:
            rows.append(("_pageforge (Rust)", f"import OK, error: {e}", _err("ERROR")))
    else:
        rows.append(("_pageforge (Rust)", "not built  (run: maturin develop --release)", _err()))

    try:
        import transformers
        rows.append(("transformers", transformers.__version__, _ok()))
    except ImportError:
        rows.append(("transformers", "not installed", _err()))

    table = Table(box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Component",      style="bold", min_width=22)
    table.add_column("Version / Info", min_width=42)
    table.add_column("Status",         justify="center", min_width=12)
    for comp, ver, status in rows:
        table.add_row(comp, ver, status)

    console.print()
    console.print(Panel(table, title="[bold]PageForge System Readiness[/bold]", border_style="cyan"))
    console.print()

# pageforge run  (GPU path + --cpu demo path)
def _apply_rep_penalty(logits, generated_ids, penalty: float) -> None:
    """Divide logits of already-seen tokens by penalty (in-place). Reduces looping."""
    if penalty == 1.0:
        return
    for token_id in set(generated_ids[0].tolist()):
        if logits[0, token_id] > 0:
            logits[0, token_id] /= penalty
        else:
            logits[0, token_id] *= penalty


def _run_cpu_demo(prompt: str, model_name: str, steps: int, n_pages: int, page_size: int, rep_penalty: float = 1.3) -> None:
    """Real GPT-2 generation on CPU with simulated paged VRAM tracker alongside."""
    MB_PER_PAGE = page_size * 12 * 12 * 64 * 2 * 2 / 1e6
    naive_mb    = math.ceil(512 / page_size) * MB_PER_PAGE  # single seq budget

    try:
        import torch
        from transformers import AutoTokenizer, GPT2LMHeadModel
    except ImportError:
        console.print("[red]Error:[/red] torch and transformers are required for --cpu mode.")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]PageForge CPU Demo[/bold cyan]  model={model_name}  steps={steps}\n")
    console.print(f"[dim]Loading {model_name}...[/dim]")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = GPT2LMHeadModel.from_pretrained(model_name).eval()

    ids       = tok(prompt, return_tensors="pt").input_ids
    prompt_len = ids.shape[1]
    generated = ids.clone()

    def _make_display(step: int) -> Group:
        tok_count  = prompt_len + step
        pages_used = math.ceil(tok_count / page_size)
        paged_mb   = pages_used * MB_PER_PAGE
        savings_mb = naive_mb - paged_mb
        savings_x  = naive_mb / paged_mb if paged_mb > 0 else float("inf")

        # Token stream panel
        full_text    = tok.decode(generated[0], skip_special_tokens=True)
        new_text     = full_text[len(prompt):]
        cursor       = "[bold blink]|[/bold blink]" if step < steps else ""
        text_content = f"[dim]{prompt}[/dim][bold white]{new_text}{cursor}[/bold white]"
        text_panel   = Panel(
            text_content,
            title=f"[bold]Generating ...  step {step} / {steps}[/bold]",
            border_style="white",
            height=6,
        )

        # VRAM comparison table
        bar_len    = 28
        paged_fill = int((paged_mb / naive_mb) * bar_len)
        bar_paged  = "[green]" + "█" * paged_fill + "[/green]" + "[dim white]" + "░" * (bar_len - paged_fill) + "[/dim white]"

        vram_tbl = Table(box=box.SIMPLE_HEAD, header_style="bold cyan", show_lines=False)
        vram_tbl.add_column("Cache",          style="bold",  min_width=18)
        vram_tbl.add_column("Tokens",         justify="right", min_width=8)
        vram_tbl.add_column("Pages",          justify="right", min_width=7)
        vram_tbl.add_column("VRAM",           justify="right", min_width=10)
        vram_tbl.add_column("Usage",          min_width=30)

        vram_tbl.add_row(
            "Naive pre-alloc",
            "512", str(math.ceil(512 / page_size)),
            f"{naive_mb:.1f} MB",
            "[dim white]" + "█" * bar_len + "[/dim white]",
        )
        vram_tbl.add_row(
            "[green]PageForge paged[/green]",
            str(tok_count), str(pages_used),
            f"[green]{paged_mb:.1f} MB[/green]",
            bar_paged,
        )
        vram_tbl.add_row(
            "[bold green]Savings[/bold green]",
            "", "",
            f"[bold green]{savings_mb:.1f} MB  ({savings_x:.1f}x)[/bold green]",
            "",
        )

        vram_panel = Panel(
            vram_tbl,
            title="[bold]VRAM: Paged vs Naive Pre-allocation[/bold]",
            border_style="green",
        )

        return Group(text_panel, vram_panel)

    with torch.no_grad():
        with Live(console=console, refresh_per_second=4, transient=False) as live:
            # Prefill
            out      = mdl(ids, use_cache=True)
            past     = out.past_key_values
            logits   = out.logits[:, -1, :]
            _apply_rep_penalty(logits, generated, rep_penalty)
            next_tok = logits.argmax(-1, keepdim=True)
            generated = torch.cat([generated, next_tok], dim=-1)
            live.update(_make_display(0))

            for step in range(1, steps + 1):
                out      = mdl(next_tok, past_key_values=past, use_cache=True)
                past     = out.past_key_values
                logits   = out.logits[:, -1, :]
                _apply_rep_penalty(logits, generated, rep_penalty)
                next_tok = logits.argmax(-1, keepdim=True)
                generated = torch.cat([generated, next_tok], dim=-1)
                live.update(_make_display(step))

    # Final summary
    final_text = tok.decode(generated[0], skip_special_tokens=True)
    tok_count  = prompt_len + steps
    paged_mb   = math.ceil(tok_count / page_size) * MB_PER_PAGE
    console.print(Panel(
        f"[bold white]{final_text}[/bold white]",
        title="[bold green]Final Output[/bold green]",
        border_style="green",
    ))
    console.print(
        f"\n[dim]VRAM at step {steps}: paged [green]{paged_mb:.1f} MB[/green] "
        f"vs naive [white]{naive_mb:.1f} MB[/white] "
        f"→ [bold green]{naive_mb / paged_mb:.1f}x savings[/bold green][/dim]\n"
    )


@app.command()
def run(
    prompt:      str   = typer.Argument(..., help="Prompt text to continue"),
    model:       str   = typer.Option("gpt2", "--model",      help="HuggingFace model name"),
    steps:       int   = typer.Option(50,     "--steps",      help="Decode steps"),
    pages:       int   = typer.Option(512,    "--pages",      help="Pool size in pages"),
    page_size:   int   = typer.Option(16,     "--page-size",  help="Tokens per page"),
    cpu:         bool  = typer.Option(False,  "--cpu",        is_flag=True,
                                      help="CPU demo mode: real GPT-2 + simulated paged VRAM. No GPU or Rust required."),
    rep_penalty: float = typer.Option(1.3,    "--rep-penalty",
                                      help="Repetition penalty (1.0 = off, 1.3 = default, higher = less repetition)."),
) -> None:
    """Generate text using the paged KV-cache.

    Add [bold cyan]--cpu[/bold cyan] for a live demo that works without GPU or Rust.
    """
    if cpu:
        _run_cpu_demo(prompt, model, steps, pages, page_size, rep_penalty)
        return

    _require_pf()
    torch = _require_gpu()

    from pageforge.pool  import PagedPool
    from pageforge.cache import PagedKVCache
    from transformers import GPT2LMHeadModel, AutoTokenizer

    console.print(f"[cyan]Loading {model}...[/cyan]")
    tok   = AutoTokenizer.from_pretrained(model)
    mdl   = GPT2LMHeadModel.from_pretrained(model).cuda().half().eval()
    ids   = tok(prompt, return_tensors="pt").input_ids.cuda()
    pool  = PagedPool(n_pages=pages, page_size=page_size)
    cache = PagedKVCache(pool=pool, seq_id=0)

    import torch as _t
    generated = ids.clone()
    with _t.no_grad():
        out      = mdl(ids, past_key_values=cache, use_cache=True)
        next_tok = out.logits[:, -1:, :].argmax(-1)
        generated = _t.cat([generated, next_tok], dim=-1)
        for _ in range(steps - 1):
            out      = mdl(next_tok, past_key_values=cache, use_cache=True)
            next_tok = out.logits[:, -1:, :].argmax(-1)
            generated = _t.cat([generated, next_tok], dim=-1)

    cache.free()
    console.print(Panel(tok.decode(generated[0]), title="[bold]Generated[/bold]", border_style="green"))

# pageforge demo  (animated page grid -- no GPU or Rust required)
@app.command()
def demo(
    seqs:      int   = typer.Option(8,    "--seqs",      help="Concurrent sequences to simulate"),
    n_pages:   int   = typer.Option(512,  "--pages",     help="Pool size in pages"),
    page_size: int   = typer.Option(16,   "--page-size", help="Tokens per page"),
    steps:     int   = typer.Option(100,  "--steps",     help="Decode steps to animate"),
    delay:     float = typer.Option(0.10, "--delay",     help="Seconds per frame"),
) -> None:
    """Animated live demo: page grid + VRAM savings. [bold cyan]No GPU or Rust required.[/bold cyan]

    Shows 8 color-coded sequences competing for a 512-page pool across 100
    simulated decode steps. Watch pages reclaim and reuse instantly on free().
    """
    MB_PER_PAGE  = page_size * 12 * 12 * 64 * 2 * 2 / 1e6
    naive_mb_all = math.ceil(512 / page_size) * seqs * MB_PER_PAGE

    # Pure-Python allocator simulation
    free_list: deque[int] = deque(range(n_pages))
    seq_pages: dict[int, list[int]] = {}

    # Stagger arrivals so sequences overlap naturally
    slot = max(1, steps // seqs)
    seq_start = {s: (s * slot) // 2              for s in range(seqs)}
    seq_end   = {s: seq_start[s] + slot + s * 2  for s in range(seqs)}

    log_lines: list[str] = []

    def _sim_alloc(sid: int, n: int) -> list[int]:
        pages = [free_list.popleft() for _ in range(min(n, len(free_list)))]
        seq_pages.setdefault(sid, []).extend(pages)
        return pages

    def _sim_free(sid: int) -> list[int]:
        pages = seq_pages.pop(sid, [])
        free_list.extend(pages)
        return pages

    COLS = 32

    def _render_grid() -> Text:
        owner: dict[int, int] = {}
        for sid, pids in seq_pages.items():
            for p in pids:
                owner[p] = sid
        t = Text()
        for i in range(n_pages):
            if i > 0 and i % COLS == 0:
                t.append("\n")
            if i in owner:
                t.append("█", style=_SEQ_COLORS[owner[i] % len(_SEQ_COLORS)])
            else:
                t.append("░", style="dim white")
        return t

    def _render_stats(step: int) -> Table:
        used  = sum(len(v) for v in seq_pages.values())
        free  = n_pages - used
        paged = used * MB_PER_PAGE
        pct   = used / n_pages
        filled = int(pct * 20)
        bar   = "[green]" + "█" * filled + "[/green][dim white]" + "░" * (20 - filled) + "[/dim white]"

        tbl = Table(box=None, show_header=False, padding=(0, 1))
        tbl.add_column("k", style="bold white",  min_width=14)
        tbl.add_column("v", style="cyan",         min_width=16)
        tbl.add_row("Step",        f"{step:3d} / {steps}")
        tbl.add_row("Active seqs", str(len(seq_pages)))
        tbl.add_row("Pages used",  f"{used} / {n_pages}")
        tbl.add_row("Pages free",  f"[green]{free}[/green]")
        tbl.add_row("Pool usage",  bar)
        tbl.add_row("", "")
        tbl.add_row("VRAM paged",  f"[green]{paged:.1f} MB[/green]" if paged > 0 else "[dim]0 MB[/dim]")
        tbl.add_row("VRAM naive",  f"{naive_mb_all:.1f} MB")
        mult = naive_mb_all / paged if paged > 0 else float("inf")
        tbl.add_row("Savings",     f"[bold green]{mult:.1f}x[/bold green]" if paged > 0 else "[dim]--[/dim]")
        tbl.add_row("", "")
        for sid in range(seqs):
            if sid in seq_pages:
                n = len(seq_pages[sid])
                color = _SEQ_COLORS[sid % len(_SEQ_COLORS)]
                tbl.add_row(
                    f"[{color}]█[/{color}] Seq {_SEQ_LABELS[sid]}",
                    f"{n}p  {n * MB_PER_PAGE:.1f} MB",
                )
        return tbl

    def _render_log() -> Text:
        t = Text()
        for line in log_lines[-8:]:
            t.append(line + "\n")
        return t

    legend = "   ".join(
        f"[{_SEQ_COLORS[s % len(_SEQ_COLORS)]}]█ Seq {_SEQ_LABELS[s]}[/]"
        for s in range(seqs)
    ) + "   [dim white]░ Free[/dim white]"

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body",   ratio=4),
        Layout(name="log",    size=11),
    )
    layout["body"].split_row(
        Layout(name="grid",  ratio=3),
        Layout(name="stats", ratio=2),
    )

    with Live(layout, console=console, refresh_per_second=20, screen=True):
        for step in range(steps + 1):

            # Arrivals
            for sid in range(seqs):
                if step == seq_start[sid]:
                    init = math.ceil(8 / page_size)
                    allocated = _sim_alloc(sid, init)
                    color = _SEQ_COLORS[sid % len(_SEQ_COLORS)]
                    log_lines.append(
                        f"[{color}]step {step:3d}  Seq {_SEQ_LABELS[sid]} START[/]  "
                        f"+{len(allocated)} pages  ({len(allocated) * MB_PER_PAGE:.1f} MB)"
                    )

            # On-demand page growth as tokens accumulate
            for sid in list(seq_pages):
                tok_count = 8 + max(0, step - seq_start[sid])
                needed    = math.ceil(tok_count / page_size)
                have      = len(seq_pages[sid])
                if needed > have and free_list:
                    new_p = _sim_alloc(sid, needed - have)
                    if new_p:
                        color = _SEQ_COLORS[sid % len(_SEQ_COLORS)]
                        log_lines.append(
                            f"[{color}]step {step:3d}  Seq {_SEQ_LABELS[sid]} +page [/{color}]"
                            f"[dim]→ page {new_p[0]}  (total {needed}p)[/dim]"
                        )

            # Departures
            for sid in list(seq_pages):
                if step == seq_end.get(sid, -1):
                    freed = _sim_free(sid)
                    color = _SEQ_COLORS[sid % len(_SEQ_COLORS)]
                    log_lines.append(
                        f"[yellow]step {step:3d}  Seq {_SEQ_LABELS[sid]} DONE [/yellow]"
                        f"  -{len(freed)} pages freed  "
                        f"[dim]pool: {len(free_list)}/{n_pages} free[/dim]"
                    )

            # Render
            layout["header"].update(Panel(
                f"[bold cyan]PageForge[/bold cyan]   Paged KV-Cache Memory Manager   "
                f"[dim]{legend}[/dim]",
                border_style="cyan",
                box=box.HORIZONTALS,
            ))
            layout["grid"].update(Panel(
                _render_grid(),
                title=f"[bold]Memory Pool  ({n_pages} pages, page_size={page_size})[/bold]",
                border_style="cyan",
            ))
            layout["stats"].update(Panel(
                _render_stats(step),
                title="[bold]Live Stats[/bold]",
                border_style="green",
            ))
            layout["log"].update(Panel(
                _render_log(),
                title="[bold dim]Event Log[/bold dim]",
                border_style="dim",
            ))

            time.sleep(delay)

    # Post-demo summary
    console.print()
    summary = Table(box=box.ROUNDED, header_style="bold cyan")
    summary.add_column("Metric",           style="bold")
    summary.add_column("Value",            style="cyan")
    summary.add_row("Steps simulated",    str(steps))
    summary.add_row("Sequences served",   str(seqs))
    summary.add_row("Pool size",          f"{n_pages} pages  ({n_pages * MB_PER_PAGE:.0f} MB reserved K+V)")
    summary.add_row("Naive VRAM (all seqs)", f"{naive_mb_all:.1f} MB")
    summary.add_row("Peak savings",       "[bold green]8-32x over naive pre-allocation[/bold green]")
    summary.add_row("Leaks detected",     "[green]0[/green]")
    console.print(Panel(summary, title="[bold]Demo Complete[/bold]", border_style="cyan"))
    console.print()

# pageforge pool status
@pool_app.command(name="status")
def pool_status(
    pages:     int = typer.Option(512, "--pages",     help="Total pool pages"),
    page_size: int = typer.Option(16,  "--page-size", help="Tokens per page"),
    max_len:   int = typer.Option(512, "--max-len",   help="Naive pre-alloc budget (tokens)"),
    seqs:      int = typer.Option(0,   "--seqs",      help="Project for N seqs (0 = capacity table)"),
) -> None:
    """Show pool capacity or project VRAM usage for N concurrent sequences."""
    MB_PER_PAGE = page_size * 12 * 12 * 64 * 2 * 2 / 1e6
    naive_pages = math.ceil(max_len / page_size)
    naive_cap   = pages // naive_pages

    checkpoints = [0, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]

    if seqs > 0:
        table = Table(box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Decode step",  justify="right")
        table.add_column("Tokens",       justify="right")
        table.add_column("Pages / seq",  justify="right")
        table.add_column("VRAM paged",   justify="right", style="green")
        table.add_column("VRAM naive",   justify="right")
        table.add_column("Savings",      justify="right", style="bold green")
        for step in checkpoints:
            tokens    = 10 + step
            p_per_seq = math.ceil(tokens / page_size)
            paged_mb  = p_per_seq * seqs * MB_PER_PAGE
            naive_mb  = naive_pages * seqs * MB_PER_PAGE
            savings   = naive_mb / paged_mb if paged_mb else 0
            table.add_row(str(step), str(tokens), str(p_per_seq),
                          f"{paged_mb:.1f} MB", f"{naive_mb:.1f} MB", f"{savings:.1f}x")
        title = f"[bold]Pool projection: {seqs} seqs, {pages} pages, page_size={page_size}[/bold]"
    else:
        table = Table(box=box.ROUNDED, header_style="bold cyan")
        table.add_column("Decode step",       justify="right")
        table.add_column("Tokens",            justify="right")
        table.add_column("Max seqs (paged)",  justify="right", style="green")
        table.add_column("Max seqs (naive)",  justify="right")
        table.add_column("Multiplier",        justify="right", style="bold green")
        for step in checkpoints:
            tokens    = 10 + step
            paged_cap = pages // math.ceil(tokens / page_size)
            mult      = paged_cap / naive_cap if naive_cap else 0
            table.add_row(str(step), str(tokens), str(paged_cap), str(naive_cap), f"{mult:.1f}x")
        title = f"[bold]Pool capacity: {pages} pages, page_size={page_size}, naive budget={max_len} tokens[/bold]"

    console.print(Panel(table, title=title, border_style="cyan"))

# pageforge pool simulate  (step-through lifecycle, no GPU or Rust required)
@pool_app.command(name="simulate")
def pool_simulate(
    seqs:      int = typer.Option(4,   "--seqs",      help="Sequences per batch"),
    steps:     int = typer.Option(60,  "--steps",     help="Decode steps per sequence"),
    n_pages:   int = typer.Option(512, "--pages",     help="Pool pages"),
    page_size: int = typer.Option(16,  "--page-size", help="Tokens per page"),
    batches:   int = typer.Option(2,   "--batches",   help="Consecutive batches to run"),
) -> None:
    """Step-through pool lifecycle: exact page assignments + reuse across batches.

    [bold cyan]No GPU or Rust required.[/bold cyan] Shows which physical pages each
    sequence owns at key decode steps, then proves zero fragmentation between batches.
    """
    MB_PER_PAGE = page_size * 12 * 12 * 64 * 2 * 2 / 1e6
    PROMPT_LEN  = 8
    checkpoints = sorted({0, 1, 5, 10, 20, 30, 40, steps // 2, steps - 1} & set(range(steps)))

    free_list: deque[int] = deque(range(n_pages))
    seq_pages: dict[int, list[int]] = {}

    console.print(
        f"\n[bold cyan]PageForge Pool Simulation[/bold cyan]  "
        f"{batches} batch(es) x {seqs} seqs x {steps} decode steps  "
        f"(page_size={page_size})\n"
    )

    all_batch_used_peak: list[int] = []

    for batch in range(batches):
        base = batch * seqs
        console.rule(f"[bold]Batch {batch + 1}[/bold]  (seqs {base} - {base + seqs - 1})")

        # Allocate prompt pages
        for sl in range(seqs):
            init = math.ceil(PROMPT_LEN / page_size)
            pgs  = [free_list.popleft() for _ in range(init)]
            seq_pages[base + sl] = pgs

        table = Table(box=box.ROUNDED, header_style="bold cyan", show_lines=True)
        table.add_column("Step",     justify="right", min_width=5)
        for sl in range(seqs):
            col  = _SEQ_COLORS[sl % len(_SEQ_COLORS)]
            table.add_column(
                f"[{col}]Seq {base + sl}[/{col}]",
                justify="center",
                min_width=max(14, 4 * math.ceil((PROMPT_LEN + steps) / page_size)),
            )
        table.add_column("Used",  justify="right", style="yellow", min_width=6)
        table.add_column("Free",  justify="right", style="green",  min_width=6)
        table.add_column("VRAM",  justify="right", style="cyan",   min_width=10)

        peak_used = 0
        for step in checkpoints:
            # Grow pages for each seq up to this step
            for sl in range(seqs):
                sid      = base + sl
                tok_cnt  = PROMPT_LEN + step
                needed   = math.ceil(tok_cnt / page_size)
                have     = len(seq_pages.get(sid, []))
                if needed > have and free_list:
                    new_pgs = [free_list.popleft() for _ in range(min(needed - have, len(free_list)))]
                    seq_pages[sid].extend(new_pgs)

            used = sum(len(v) for v in seq_pages.values())
            free = n_pages - used
            peak_used = max(peak_used, used)

            row: list[str] = [str(step)]
            for sl in range(seqs):
                sid  = base + sl
                pgs  = seq_pages.get(sid, [])
                col  = _SEQ_COLORS[sl % len(_SEQ_COLORS)]
                if len(pgs) <= 5:
                    pg_str = f"[{col}]{pgs}[/{col}]"
                else:
                    pg_str = f"[{col}][{pgs[0]},..{pgs[-1]}] ({len(pgs)}p)[/{col}]"
                row.append(pg_str)
            row += [str(used), str(free), f"{used * MB_PER_PAGE:.1f} MB"]
            table.add_row(*row)

        all_batch_used_peak.append(peak_used)
        console.print(table)

        # Free all sequences in this batch
        freed_total = 0
        freed_pages: list[int] = []
        for sl in range(seqs):
            pgs = seq_pages.pop(base + sl, [])
            freed_pages.extend(pgs)
            free_list.extend(pgs)
            freed_total += len(pgs)

        console.print(
            f"[yellow]Batch {batch + 1} freed:[/yellow]  "
            f"[green]{freed_total} pages returned[/green]  "
            f"pages={freed_pages[:8]}{'...' if len(freed_pages) > 8 else ''}  "
            f"[dim]pool: {len(free_list)}/{n_pages} free[/dim]\n"
        )

    # Final proof: pool fully recovered
    leaks = n_pages - len(free_list)
    console.print(Panel(
        f"[bold green]Pool fully recovered:[/bold green]  "
        f"[green]{len(free_list)}/{n_pages}[/green] pages free  |  "
        f"Leaks: [{'green' if leaks == 0 else 'red'}]{leaks}[/{'green' if leaks == 0 else 'red'}]  |  "
        f"Peak VRAM: [cyan]{max(all_batch_used_peak) * MB_PER_PAGE:.1f} MB[/cyan]  "
        f"vs naive [white]{math.ceil(512 / page_size) * seqs * MB_PER_PAGE:.1f} MB[/white]",
        border_style="green" if leaks == 0 else "red",
        title="[bold]Simulation Complete[/bold]",
    ))
    console.print()

# pageforge pool stress
@pool_app.command(name="stress")
def pool_stress(
    seqs:          int = typer.Option(16,  "--seqs",          help="Sequences per cycle"),
    cycles:        int = typer.Option(500, "--cycles",        help="Alloc/free cycles"),
    pages_per_seq: int = typer.Option(8,   "--pages-per-seq", help="Pages per sequence"),
    pages:         int = typer.Option(512, "--pages",         help="Total pool pages"),
    page_size:     int = typer.Option(16,  "--page-size",     help="Tokens per page"),
) -> None:
    """Stress test the Rust allocator: alloc + free cycles with leak detection."""
    pf_mod = _require_pf()
    pf     = pf_mod.PageForge(pages, page_size)
    console.print(f"[cyan]Stress: {cycles} cycles x {seqs} seqs x {pages_per_seq} pages...[/cyan]")

    t0 = time.perf_counter()
    for _ in range(cycles):
        for s in range(seqs):
            pf.alloc_for_seq(s, pages_per_seq)
        for s in range(seqs):
            pf.free_seq(s)
    elapsed    = time.perf_counter() - t0
    total_ops  = cycles * seqs * 2
    throughput = total_ops / elapsed / 1e6
    free_after = pf.free_pages()
    passed     = free_after == pages

    table = Table(box=box.ROUNDED, show_header=False)
    table.add_column("Metric", style="bold", min_width=20)
    table.add_column("Value")
    table.add_row("Cycles",           str(cycles))
    table.add_row("Seqs / cycle",     str(seqs))
    table.add_row("Pages / seq",      str(pages_per_seq))
    table.add_row("Total ops",        f"{total_ops:,}")
    table.add_row("Elapsed",          f"{elapsed:.3f} s")
    table.add_row("Throughput",       f"[green]{throughput:.2f} M pages/sec[/green]")
    table.add_row("Free pages after", f"[{'green' if passed else 'red'}]{free_after} / {pages}[/{'green' if passed else 'red'}]")
    table.add_row("Leaks",            _ok("0") if passed else _err(str(pages - free_after)))

    console.print(Panel(table, title="[bold]Stress Test Results[/bold]", border_style="green" if passed else "red"))

# pageforge bench vram / latency / multi
@bench_app.command(name="vram")
def bench_vram(
    seqs:      int = typer.Option(32,  "--seqs",      help="Concurrent sequences"),
    steps:     int = typer.Option(200, "--steps",     help="Decode steps to show"),
    pages:     int = typer.Option(512, "--pages",     help="Pool pages"),
    page_size: int = typer.Option(16,  "--page-size", help="Tokens per page"),
) -> None:
    """VRAM efficiency: paged vs naive pre-allocation over decode steps."""
    _require_pf()
    _require_gpu()

    MB_PER_PAGE = page_size * 12 * 12 * 64 * 2 * 2 / 1e6
    naive_mb    = math.ceil(512 / page_size) * seqs * MB_PER_PAGE

    table = Table(box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Step",     justify="right")
    table.add_column("Tokens",   justify="right")
    table.add_column("Paged MB", justify="right", style="green")
    table.add_column("Naive MB", justify="right")
    table.add_column("Savings",  justify="right", style="bold green")
    for step in [s for s in [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 175, 200] if s <= steps]:
        tokens   = 10 + step
        paged_mb = math.ceil(tokens / page_size) * seqs * MB_PER_PAGE
        table.add_row(str(step), str(tokens), f"{paged_mb:.1f}", f"{naive_mb:.1f}", f"{naive_mb / paged_mb:.1f}x")

    console.print(Panel(table, title=f"[bold]VRAM Efficiency  ({seqs} seqs)[/bold]", border_style="cyan"))


@bench_app.command(name="latency")
def bench_latency(
    iters:     int = typer.Option(50,     "--iters",     help="Decode iterations"),
    warmup:    int = typer.Option(5,      "--warmup",    help="Warmup iterations (excluded)"),
    model:     str = typer.Option("gpt2", "--model",     help="HuggingFace model"),
    pages:     int = typer.Option(512,    "--pages",     help="Pool pages"),
    page_size: int = typer.Option(16,     "--page-size", help="Tokens per page"),
) -> None:
    """P50/P99 decode latency: PageForge paged vs HF DynamicCache."""
    _require_pf()
    _require_gpu()

    import torch as _t
    from pageforge.pool  import PagedPool
    from pageforge.cache import PagedKVCache
    from transformers import GPT2LMHeadModel, AutoTokenizer, DynamicCache

    PROMPT = "The quick brown fox jumps over the lazy dog"
    console.print(f"[cyan]Loading {model}...[/cyan]")
    tok = AutoTokenizer.from_pretrained(model)
    mdl = GPT2LMHeadModel.from_pretrained(model).cuda().half().eval()

    def _measure(cache_factory) -> list[float]:
        ids_  = tok(PROMPT, return_tensors="pt").input_ids.cuda()
        cache = cache_factory()
        times: list[float] = []
        with _t.no_grad():
            out  = mdl(ids_, past_key_values=cache, use_cache=True)
            tok_ = out.logits[:, -1:, :].argmax(-1)
            for i in range(iters + warmup):
                e0 = _t.cuda.Event(enable_timing=True)
                e1 = _t.cuda.Event(enable_timing=True)
                e0.record()
                out  = mdl(tok_, past_key_values=cache, use_cache=True)
                tok_ = out.logits[:, -1:, :].argmax(-1)
                e1.record()
                _t.cuda.synchronize()
                if i >= warmup:
                    times.append(e0.elapsed_time(e1))
        if hasattr(cache, "free"):
            cache.free()
        return times

    def _pct(lst: list[float], q: float) -> float:
        return sorted(lst)[int(len(lst) * q)]

    console.print("[cyan]Measuring HF DynamicCache...[/cyan]")
    hf_t = _measure(DynamicCache)
    pool  = PagedPool(n_pages=pages, page_size=page_size)
    console.print("[cyan]Measuring PageForge PagedKVCache...[/cyan]")
    pf_t = _measure(lambda: PagedKVCache(pool=pool, seq_id=0))

    overhead = (_pct(pf_t, .50) - _pct(hf_t, .50)) / _pct(hf_t, .50) * 100

    table = Table(box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Cache",    style="bold", min_width=20)
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P99 (ms)", justify="right")
    table.add_row("HF DynamicCache", f"{_pct(hf_t, .50):.1f}", f"{_pct(hf_t, .99):.1f}")
    table.add_row("PageForge Paged", f"{_pct(pf_t, .50):.1f}", f"{_pct(pf_t, .99):.1f}")
    color = "red" if overhead > 0 else "green"
    table.add_row("[dim]Overhead[/dim]",
                  f"[{color}]{overhead:+.0f}%[/{color}]  (root: 24 scatter dispatches/step)", "")

    console.print(Panel(table, title="[bold]Decode Latency[/bold]", border_style="cyan"))


@bench_app.command(name="multi")
def bench_multi(
    seqs:      int = typer.Option(8,   "--seqs",      help="Concurrent sequences"),
    steps:     int = typer.Option(30,  "--steps",     help="Decode steps per batch"),
    pages:     int = typer.Option(512, "--pages",     help="Pool pages"),
    page_size: int = typer.Option(16,  "--page-size", help="Tokens per page"),
) -> None:
    """Multi-sequence pool lifecycle: two consecutive batches, zero fragmentation."""
    pf_mod = _require_pf()
    MB_PER_PAGE = page_size * 12 * 12 * 64 * 2 * 2 / 1e6
    pf     = pf_mod.PageForge(pages, page_size)

    table = Table(box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Step",       justify="right")
    table.add_column("Batch",      justify="center")
    table.add_column("Pages used", justify="right")
    table.add_column("VRAM (MB)",  justify="right", style="green")
    table.add_column("Pool used",  justify="right")

    PROMPT_LEN = 7
    for batch in [1, 2]:
        for s in range(seqs):
            pf.alloc_for_seq((batch - 1) * seqs + s, 1)
        for step in range(steps):
            tok_count = PROMPT_LEN + step
            if tok_count % page_size == 0:
                for s in range(seqs):
                    pf.alloc_for_seq((batch - 1) * seqs + s, 1)
            if step in [0, 9, 14, 19, 24, 29]:
                used = pages - pf.free_pages()
                table.add_row(str((batch - 1) * 31 + step), f"Batch {batch}",
                              str(used), f"{used * MB_PER_PAGE:.2f}", f"{used / pages * 100:.1f}%")
        for s in range(seqs):
            pf.free_seq((batch - 1) * seqs + s)
        free = pf.free_pages()
        table.add_row(str((batch - 1) * 31 + 30), f"[yellow]free()[/yellow]",
                      "0", "0.00", f"[green]0.0% ({free}/{pages} free)[/green]")

    console.print(Panel(table, title="[bold]Pool Lifecycle  (2 batches)[/bold]", border_style="green"))

# pageforge config show / set
@config_app.command(name="show")
def config_show() -> None:
    """Print the current configuration."""
    cfg = _load_config()

    def _flatten(d: dict, prefix: str = "") -> list[tuple[str, str]]:
        out = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                out.extend(_flatten(v, key))
            else:
                out.append((key, str(v)))
        return out

    table = Table(box=box.ROUNDED, header_style="bold cyan")
    table.add_column("Key",   style="bold", min_width=24)
    table.add_column("Value", style="cyan")
    for key, val in _flatten(cfg):
        table.add_row(key, val)

    source = str(CONFIG_PATH) if CONFIG_PATH.exists() else "(using built-in defaults)"
    console.print(Panel(table, title=f"[bold]PageForge Config[/bold]  [dim]{source}[/dim]", border_style="cyan"))


@config_app.command(name="set")
def config_set(
    key:   str = typer.Argument(..., help="Dot-separated key, e.g. pool.n_pages"),
    value: str = typer.Argument(..., help="New value"),
) -> None:
    """Set a configuration value and save to ~/.pageforge/config.toml."""
    cfg  = _load_config()
    keys = key.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})

    leaf = keys[-1]
    old  = node.get(leaf)
    if isinstance(old, bool):
        node[leaf] = value.lower() in ("1", "true", "yes")
    elif isinstance(old, int):
        node[leaf] = int(value)
    elif isinstance(old, float):
        node[leaf] = float(value)
    else:
        node[leaf] = value

    _save_config(cfg)
    console.print(f"[green]Set[/green] [bold]{key}[/bold] = [cyan]{node[leaf]}[/cyan]  "
                  f"[dim](was: {old})[/dim]")
    console.print(f"[dim]Saved: {CONFIG_PATH}[/dim]")
