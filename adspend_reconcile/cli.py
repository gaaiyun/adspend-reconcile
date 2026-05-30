"""Command-line interface for adspend-reconcile.

Subcommands
-----------
* ``ingest``    - normalise one ad-platform or store CSV into the canonical
                  schema; prints the resolved column mapping so you can audit
                  the fuzzy matching.
* ``reconcile`` - join one or more ad CSVs against a store CSV by date x channel,
                  classify matched / discrepant / unattributed money, and apply
                  a profit model for contribution-margin ROAS and blended MER.
* ``report``    - a fuller reconcile run: classification split, per-channel
                  profit table, and the largest discrepancies.
* ``sample``    - write the bundled, offline sample dataset to a directory.

Everything is local; nothing here touches the network or any ad API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .ingest import PRESETS, ingest_csv
from .reconcile import ProfitModel, reconcile
from .samples import write_samples

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Local-first ad-spend reconciliation: align Meta/Google/TikTok exports "
    "with real store revenue by date x channel.",
)
# A roomy fixed width keeps tables readable when output is piped (rich would
# otherwise fall back to 80 columns and crush the per-channel table).
console = Console(width=120)


def _parse_map(pairs: list[str] | None) -> dict[str, str]:
    """Parse repeated ``field=Column`` options into a dict."""
    out: dict[str, str] = {}
    for item in pairs or []:
        if "=" not in item:
            raise typer.BadParameter(f"--map expects field=Column, got {item!r}")
        field, col = item.split("=", 1)
        out[field.strip()] = col.strip()
    return out


def _df_to_table(df, title: str, max_rows: int = 50) -> Table:
    table = Table(title=title, show_lines=False, header_style="bold")
    for col in df.columns:
        justify = "right" if df[col].dtype.kind in "fi" else "left"
        table.add_column(str(col), justify=justify)
    for _, row in df.head(max_rows).iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if isinstance(val, float):
                cells.append(f"{val:,.2f}")
            else:
                cells.append(str(val))
        table.add_row(*cells)
    return table


@app.command()
def version() -> None:
    """Print the installed version."""
    console.print(f"adspend-reconcile {__version__}")


@app.command()
def sample(
    out_dir: Path = typer.Argument(
        Path("sample_data"), help="Directory to write the sample CSV bundle into."
    ),
    days: int = typer.Option(14, help="Number of days of data to generate."),
) -> None:
    """Write the bundled offline sample dataset (Meta/Google/TikTok + Shopify)."""
    written = write_samples(out_dir, days=days)
    console.print(f"[bold green]Wrote {len(written)} sample files to {out_dir}:[/]")
    for name, path in written.items():
        console.print(f"  {name:8s} -> {path}")


@app.command()
def ingest(
    csv_path: Path = typer.Argument(..., exists=True, help="CSV file to normalise."),
    kind: str = typer.Option(..., "--kind", help="'ads' or 'store'."),
    preset: Optional[str] = typer.Option(
        None, "--preset", help=f"Platform preset: {', '.join(sorted(PRESETS))}."
    ),
    channel: Optional[str] = typer.Option(
        None, "--channel", help="Constant channel label when the file has no channel column."
    ),
    map_: Optional[list[str]] = typer.Option(
        None, "--map", help="Explicit column override, repeatable: field=ColumnName."
    ),
    threshold: float = typer.Option(0.55, help="Fuzzy column-match acceptance threshold."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write the normalised CSV here."),
) -> None:
    """Normalise one CSV into the canonical schema and show the column mapping."""
    result = ingest_csv(
        csv_path,
        kind=kind,
        preset=preset,
        overrides=_parse_map(map_),
        default_channel=channel,
        threshold=threshold,
    )
    console.print(_df_to_table(result.mapping_table(), f"Column mapping ({kind})"))
    console.print(
        f"[dim]{len(result.frame)} normalised rows, "
        f"{result.frame['channel'].nunique()} channel(s).[/]"
    )
    console.print(_df_to_table(result.frame, "Normalised data (first rows)", max_rows=12))

    if out is not None:
        result.frame.to_csv(out, index=False)
        console.print(f"[green]Wrote normalised CSV -> {out}[/]")


def _load_ads(
    ads_specs: list[str], threshold: float
):
    """Load and stack one or more ads CSVs.

    Each spec is ``preset:path`` (e.g. ``meta:sample_data/meta_ads_sample.csv``)
    or just ``path`` to rely on fuzzy matching alone.
    """
    import pandas as pd

    frames = []
    for spec in ads_specs:
        if ":" in spec and not Path(spec).exists():
            preset, _, path = spec.partition(":")
        else:
            preset, path = None, spec
        res = ingest_csv(path, kind="ads", preset=preset or None, threshold=threshold)
        frames.append(res.frame)
    return pd.concat(frames, ignore_index=True) if frames else None


def _profit_model(
    cogs: float, shipping: float, fee_rate: float, fee_flat: float
) -> ProfitModel:
    return ProfitModel(
        cogs_rate=cogs,
        shipping_rate=shipping,
        processing_fee_rate=fee_rate,
        processing_fee_flat=fee_flat,
    )


def _summary_table(summary: dict) -> Table:
    table = Table(title="Account summary", header_style="bold", show_lines=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    order = [
        ("date range", f"{summary['date_min']} -> {summary['date_max']}"),
        ("channels", ", ".join(summary["channels"])),
        ("cells (date x channel)", str(summary["cells_total"])),
        ("  matched", str(summary["cells_matched"])),
        ("  discrepant", str(summary["cells_discrepant"])),
        ("  unattributed", str(summary["cells_unattributed"])),
        ("total ad spend", f"{summary['total_spend']:,.2f}"),
        ("platform-reported revenue", f"{summary['total_platform_revenue']:,.2f}"),
        ("store revenue (actual)", f"{summary['total_store_revenue']:,.2f}"),
        ("platform vs store gap", f"{summary['platform_vs_store_gap']:,.2f}"),
        ("contribution margin", f"{summary['total_contribution_margin']:,.2f}"),
        ("blended MER", f"{summary['blended_mer']:.3f}"),
        ("blended gross ROAS", f"{summary['blended_gross_roas']:.3f}"),
        ("blended CM ROAS", f"{summary['blended_cm_roas']:.3f}"),
        ("account marginal profit", f"{summary['account_marginal_profit']:,.2f}"),
    ]
    for k, v in order:
        table.add_row(k, v)
    return table


@app.command(name="reconcile")
def reconcile_cmd(
    store: Path = typer.Option(..., "--store", exists=True, help="Store/Stripe revenue CSV."),
    ads: list[str] = typer.Option(
        ..., "--ads", help="Ads CSV as 'preset:path' or 'path', repeatable."
    ),
    cogs: float = typer.Option(0.0, help="COGS as a fraction of revenue (0.35 = 35%)."),
    shipping: float = typer.Option(0.0, help="Shipping cost as a fraction of revenue."),
    fee_rate: float = typer.Option(0.0, help="Processing fee rate (0.029 = 2.9%)."),
    fee_flat: float = typer.Option(0.0, help="Flat processing fee per order."),
    tolerance: float = typer.Option(
        0.15, help="Relative gap still counted as matched (0.15 = within 15%)."
    ),
    threshold: float = typer.Option(0.55, help="Fuzzy column-match threshold for ingest."),
    out: Optional[Path] = typer.Option(None, "--out", help="Write the per-cell table as CSV."),
) -> None:
    """Reconcile ad spend against store revenue by date x channel."""
    store_res = ingest_csv(store, kind="store", preset="shopify", threshold=threshold)
    ads_frame = _load_ads(ads, threshold)
    result = reconcile(
        ads_frame,
        store_res.frame,
        _profit_model(cogs, shipping, fee_rate, fee_flat),
        tolerance=tolerance,
    )

    console.print(_summary_table(result.summary))
    console.print(_df_to_table(result.by_classification(), "By classification"))
    console.print(_df_to_table(result.by_channel(), "By channel"))

    if out is not None:
        result.cells.to_csv(out, index=False)
        console.print(f"[green]Wrote per-cell reconciliation -> {out}[/]")


@app.command()
def report(
    store: Path = typer.Option(..., "--store", exists=True, help="Store/Stripe revenue CSV."),
    ads: list[str] = typer.Option(
        ..., "--ads", help="Ads CSV as 'preset:path' or 'path', repeatable."
    ),
    cogs: float = typer.Option(0.0, help="COGS as a fraction of revenue."),
    shipping: float = typer.Option(0.0, help="Shipping as a fraction of revenue."),
    fee_rate: float = typer.Option(0.0, help="Processing fee rate."),
    fee_flat: float = typer.Option(0.0, help="Flat processing fee per order."),
    tolerance: float = typer.Option(0.15, help="Relative gap still counted as matched."),
    threshold: float = typer.Option(0.55, help="Fuzzy column-match threshold."),
    top: int = typer.Option(10, help="How many top discrepancies to show."),
) -> None:
    """Full report: summary, classification split, per-channel profit, top gaps."""
    store_res = ingest_csv(store, kind="store", preset="shopify", threshold=threshold)
    ads_frame = _load_ads(ads, threshold)
    result = reconcile(
        ads_frame,
        store_res.frame,
        _profit_model(cogs, shipping, fee_rate, fee_flat),
        tolerance=tolerance,
    )

    console.print(_summary_table(result.summary))
    console.print(_df_to_table(result.by_classification(), "By classification"))
    console.print(_df_to_table(result.by_channel(), "By channel"))

    discrepant = result.cells[result.cells["classification"] == "discrepant"].copy()
    if len(discrepant):
        discrepant["abs_gap"] = discrepant["revenue_gap"].abs()
        top_gaps = discrepant.sort_values("abs_gap", ascending=False).head(top)[
            ["date", "channel", "spend", "platform_revenue", "store_revenue", "revenue_gap"]
        ]
        console.print(
            _df_to_table(top_gaps, f"Top {top} discrepancies (platform vs store revenue)")
        )
    else:
        console.print("[dim]No discrepant cells at the current tolerance.[/]")


def main() -> None:  # pragma: no cover - thin entry point
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
