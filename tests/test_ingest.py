"""Tests for CSV ingestion, column matching, and normalisation."""

from pathlib import Path

import pandas as pd
import pytest

from adspend_reconcile import ingest


def _write(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


def test_meta_preset_exact_mapping(sample_bundle):
    res = ingest.ingest_csv(sample_bundle["meta"], kind="ads", preset="meta")
    methods = {m.field: m.method for m in res.matches.values()}
    assert methods["spend"] == "preset"
    assert methods["platform_revenue"] == "preset"
    # ads channel is the platform label, not a file column
    assert set(res.frame["channel"].unique()) == {"meta"}


def test_fuzzy_matching_without_preset(tmp_path):
    # Off-spec headers; no preset -> must fall back to fuzzy matching. "Ad Spend"
    # does not normalise to exactly "spend", so it exercises the fuzzy path.
    df = pd.DataFrame(
        {
            "Reporting Date": ["2024-01-01", "2024-01-02"],
            "Ad Spend ($)": ["1,000.50", "$900"],
            "Purchases": [10, 9],
            "Revenue Amount": [4000, 3600],
        }
    )
    path = _write(tmp_path, "weird.csv", df)
    res = ingest.ingest_csv(path, kind="ads", default_channel="customchan", threshold=0.5)
    assert res.matches["spend"].source_header == "Ad Spend ($)"
    assert res.matches["spend"].method == "fuzzy"
    # currency symbols / thousands separators stripped
    assert res.frame["spend"].tolist() == [1000.50, 900.0]
    assert set(res.frame["channel"]) == {"customchan"}


def test_explicit_override_wins(tmp_path):
    df = pd.DataFrame(
        {
            "d": ["2024-01-01"],
            "money_out": [123.0],
            "val": [400.0],
        }
    )
    path = _write(tmp_path, "ov.csv", df)
    res = ingest.ingest_csv(
        path,
        kind="ads",
        default_channel="x",
        overrides={"date": "d", "spend": "money_out", "platform_revenue": "val"},
    )
    assert res.matches["spend"].method == "override"
    assert res.frame["spend"].iloc[0] == 123.0


def test_store_channel_from_column(sample_bundle):
    res = ingest.ingest_csv(sample_bundle["shopify"], kind="store", preset="shopify")
    assert "organic" in set(res.frame["channel"].unique())
    assert {"date", "channel", "revenue", "orders", "refunds"}.issubset(res.frame.columns)


def test_daily_aggregation_sums_duplicate_rows(tmp_path):
    # Two rows same day/channel should be summed into one.
    df = pd.DataFrame(
        {
            "day": ["2024-01-01", "2024-01-01"],
            "channel": ["meta", "meta"],
            "total sales": [100.0, 50.0],
            "orders": [1, 1],
            "returns": [0, 0],
        }
    )
    path = _write(tmp_path, "dup.csv", df)
    res = ingest.ingest_csv(path, kind="store", preset="shopify")
    assert len(res.frame) == 1
    assert res.frame["revenue"].iloc[0] == 150.0


def test_missing_required_field_raises(tmp_path):
    df = pd.DataFrame({"foo": [1], "bar": [2]})
    path = _write(tmp_path, "bad.csv", df)
    with pytest.raises(ValueError, match="required field"):
        ingest.ingest_csv(path, kind="ads", default_channel="x")


def test_ads_without_channel_or_preset_raises(tmp_path):
    df = pd.DataFrame({"date": ["2024-01-01"], "spend": [10.0]})
    path = _write(tmp_path, "nochan.csv", df)
    with pytest.raises(ValueError, match="channel label"):
        ingest.ingest_csv(path, kind="ads")


def test_unknown_preset_raises(sample_bundle):
    with pytest.raises(ValueError, match="unknown preset"):
        ingest.ingest_csv(sample_bundle["meta"], kind="ads", preset="nope")


def test_preset_kind_mismatch_raises(sample_bundle):
    with pytest.raises(ValueError, match="is for kind"):
        ingest.ingest_csv(sample_bundle["meta"], kind="store", preset="meta")


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        ingest.ingest_csv("does_not_exist.csv", kind="ads", preset="meta")


def test_similarity_is_symmetric_and_bounded():
    s = ingest._similarity("Amount spent (USD)", "amount spent")
    assert 0.0 <= s <= 1.0
    assert ingest._similarity("Cost", "Cost") == 1.0
