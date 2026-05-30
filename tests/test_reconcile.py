"""Tests for the date x channel reconciliation engine."""

import math

import pytest

from adspend_reconcile.reconcile import (
    DISCREPANT,
    MATCHED,
    UNATTRIBUTED,
    ProfitModel,
    reconcile,
)


def _cell(result, channel):
    row = result.cells[result.cells["channel"] == channel]
    assert len(row) == 1, f"expected exactly one row for {channel}"
    return row.iloc[0]


def test_classification_buckets(tiny_ads, tiny_store):
    result = reconcile(tiny_ads, tiny_store, tolerance=0.15)

    # google: platform 600 vs store 590 -> within 15% -> matched
    assert _cell(result, "google")["classification"] == MATCHED
    # meta: platform 500 vs store 300 -> 40% gap -> discrepant
    assert _cell(result, "meta")["classification"] == DISCREPANT
    # tiktok: spend but no store revenue -> unattributed
    assert _cell(result, "tiktok")["classification"] == UNATTRIBUTED
    # organic: store revenue but no spend -> unattributed
    assert _cell(result, "organic")["classification"] == UNATTRIBUTED


def test_summary_counts(tiny_ads, tiny_store):
    s = reconcile(tiny_ads, tiny_store, tolerance=0.15).summary
    assert s["cells_total"] == 4
    assert s["cells_matched"] == 1
    assert s["cells_discrepant"] == 1
    assert s["cells_unattributed"] == 2


def test_full_outer_join_keeps_both_sides(tiny_ads, tiny_store):
    # tiktok exists only in ads, organic only in store; both must survive.
    result = reconcile(tiny_ads, tiny_store)
    channels = set(result.cells["channel"])
    assert {"meta", "google", "tiktok", "organic"} == channels


def test_blended_mer_and_gross_roas(tiny_ads, tiny_store):
    s = reconcile(tiny_ads, tiny_store).summary
    total_spend = 100 + 200 + 50          # 350
    total_store = 300 + 590 + 250         # 1140 (tiktok store rev = 0)
    assert s["total_spend"] == pytest.approx(total_spend)
    assert s["total_store_revenue"] == pytest.approx(total_store)
    assert s["blended_mer"] == pytest.approx(total_store / total_spend, abs=1e-4)


def test_platform_vs_store_gap(tiny_ads, tiny_store):
    s = reconcile(tiny_ads, tiny_store).summary
    total_platform = 500 + 600 + 80       # 1180
    total_store = 1140
    assert s["platform_vs_store_gap"] == pytest.approx(total_platform - total_store)


def test_profit_model_reduces_contribution_margin(tiny_ads, tiny_store):
    no_costs = reconcile(tiny_ads, tiny_store).summary
    with_costs = reconcile(
        tiny_ads,
        tiny_store,
        ProfitModel(cogs_rate=0.4, shipping_rate=0.1, processing_fee_rate=0.03),
    ).summary
    assert with_costs["total_contribution_margin"] < no_costs["total_contribution_margin"]


def test_contribution_margin_math_is_exact(tiny_ads, tiny_store):
    # meta store revenue 300, orders 4, refunds 0; cogs 40%, shipping 10%,
    # fee 2.9% + 0.30/order -> costs = 120 + 30 + (8.7 + 1.20) = 159.9
    # CM = 300 - 159.9 = 140.1
    pm = ProfitModel(cogs_rate=0.4, shipping_rate=0.1, processing_fee_rate=0.029,
                     processing_fee_flat=0.30)
    result = reconcile(tiny_ads, tiny_store, pm)
    meta = _cell(result, "meta")
    assert meta["cogs"] == pytest.approx(120.0)
    assert meta["shipping"] == pytest.approx(30.0)
    assert meta["fees"] == pytest.approx(8.7 + 1.20, abs=0.01)
    assert meta["contribution_margin"] == pytest.approx(140.1, abs=0.01)


def test_by_channel_metrics(tiny_ads, tiny_store):
    by_ch = reconcile(tiny_ads, tiny_store).by_channel().set_index("channel")
    # google: store 590 / spend 200 = 2.95 gross ROAS
    assert by_ch.loc["google", "gross_roas"] == pytest.approx(2.95, abs=0.01)
    # tiktok: spend 50, store revenue 0 -> negative marginal profit
    assert by_ch.loc["tiktok", "marginal_profit"] == pytest.approx(-50.0)
    # organic: spend 0 -> gross ROAS undefined (nan), not inf
    g = by_ch.loc["organic", "gross_roas"]
    assert math.isnan(g)


def test_by_classification_rollup_sums_match_totals(tiny_ads, tiny_store):
    result = reconcile(tiny_ads, tiny_store)
    rollup = result.by_classification()
    assert rollup["spend"].sum() == pytest.approx(result.summary["total_spend"])
    assert rollup["store_revenue"].sum() == pytest.approx(
        result.summary["total_store_revenue"]
    )


def test_tolerance_controls_matched_vs_discrepant(tiny_ads, tiny_store):
    # meta gap is 40%. At tolerance 0.5 it becomes matched; at 0.15 discrepant.
    loose = reconcile(tiny_ads, tiny_store, tolerance=0.5)
    assert _cell(loose, "meta")["classification"] == MATCHED
    strict = reconcile(tiny_ads, tiny_store, tolerance=0.15)
    assert _cell(strict, "meta")["classification"] == DISCREPANT


def test_sample_bundle_reconciles(sample_bundle):
    # End-to-end on the shipped sample data, going through ingest.
    from adspend_reconcile.ingest import ingest_csv
    import pandas as pd

    store = ingest_csv(sample_bundle["shopify"], kind="store", preset="shopify").frame
    ads = pd.concat(
        [
            ingest_csv(sample_bundle[p], kind="ads", preset=p).frame
            for p in ("meta", "google", "tiktok")
        ],
        ignore_index=True,
    )
    s = reconcile(ads, store, ProfitModel(cogs_rate=0.35)).summary
    assert s["total_spend"] > 0
    assert s["total_store_revenue"] > 0
    assert s["cells_discrepant"] > 0  # the sample data is built to disagree
    assert "tiktok" in s["channels"]
