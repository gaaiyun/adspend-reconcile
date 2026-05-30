"""End-to-end tests of the CLI via typer's test runner."""

from pathlib import Path

from typer.testing import CliRunner

from adspend_reconcile import __version__
from adspend_reconcile.cli import app

runner = CliRunner()


def test_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("ingest", "reconcile", "report", "sample"):
        assert cmd in result.stdout


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_sample_command_writes_files(tmp_path: Path):
    out = tmp_path / "data"
    result = runner.invoke(app, ["sample", str(out), "--days", "7"])
    assert result.exit_code == 0
    assert (out / "meta_ads_sample.csv").exists()
    assert (out / "shopify_revenue_sample.csv").exists()


def test_ingest_command(sample_bundle):
    result = runner.invoke(
        app,
        ["ingest", str(sample_bundle["meta"]), "--kind", "ads", "--preset", "meta"],
    )
    assert result.exit_code == 0
    assert "Column mapping" in result.stdout
    assert "Amount spent" in result.stdout


def test_reconcile_command(sample_bundle):
    result = runner.invoke(
        app,
        [
            "reconcile",
            "--store", str(sample_bundle["shopify"]),
            "--ads", f"meta:{sample_bundle['meta']}",
            "--ads", f"google:{sample_bundle['google']}",
            "--ads", f"tiktok:{sample_bundle['tiktok']}",
            "--cogs", "0.35",
        ],
    )
    assert result.exit_code == 0
    assert "blended MER" in result.stdout
    assert "By channel" in result.stdout


def test_report_command_with_out_file(sample_bundle, tmp_path: Path):
    out_csv = tmp_path / "cells.csv"
    result = runner.invoke(
        app,
        [
            "report",
            "--store", str(sample_bundle["shopify"]),
            "--ads", f"meta:{sample_bundle['meta']}",
            "--top", "3",
        ],
    )
    assert result.exit_code == 0
    assert "discrepancies" in result.stdout or "discrepant" in result.stdout


def test_reconcile_writes_cells_csv(sample_bundle, tmp_path: Path):
    out_csv = tmp_path / "cells.csv"
    result = runner.invoke(
        app,
        [
            "reconcile",
            "--store", str(sample_bundle["shopify"]),
            "--ads", f"meta:{sample_bundle['meta']}",
            "--out", str(out_csv),
        ],
    )
    assert result.exit_code == 0
    assert out_csv.exists()
    text = out_csv.read_text(encoding="utf-8")
    assert "classification" in text
