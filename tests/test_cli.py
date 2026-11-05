"""CLI commands that need neither a GPU nor a model download."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

pytest.importorskip("_pageforge")

from pageforge.cli import app  # noqa: E402

runner = CliRunner()


def test_info_reports_the_rust_extension():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0, result.output
    assert "_pageforge (Rust)" in result.output and "loaded" in result.output


def test_pool_stress_reports_zero_leaks():
    result = runner.invoke(app, ["pool", "stress", "--cycles", "20", "--seqs", "4"])
    assert result.exit_code == 0, result.output
    assert "512 / 512" in result.output


def test_pool_simulate_recovers_the_pool():
    result = runner.invoke(app, ["pool", "simulate", "--seqs", "2", "--steps", "20"])
    assert result.exit_code == 0, result.output
    assert "Leaks: 0" in result.output


def test_bench_multi_runs_two_batches():
    result = runner.invoke(app, ["bench", "multi", "--seqs", "2", "--steps", "30"])
    assert result.exit_code == 0, result.output
    assert "free()" in result.output
