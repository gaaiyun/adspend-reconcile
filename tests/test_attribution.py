"""Tests for the rule-based attribution module."""

import pandas as pd
import pytest

from adspend_reconcile import attribution


@pytest.fixture
def paths() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"touchpoints": ["meta", "google", "meta"], "conversion_value": 100.0},
            {"touchpoints": ["google"], "conversion_value": 50.0},
            {"touchpoints": ["tiktok", "google"], "conversion_value": 60.0},
        ]
    )


def _credit(df: pd.DataFrame) -> dict[str, float]:
    return dict(zip(df["channel"], df["credited_value"]))


def test_first_click(paths):
    res = attribution.attribute(paths, "first")
    credit = _credit(res)
    # firsts: meta(100), google(50), tiktok(60)
    assert credit["meta"] == 100.0
    assert credit["google"] == 50.0
    assert credit["tiktok"] == 60.0


def test_last_click(paths):
    res = attribution.attribute(paths, "last")
    credit = _credit(res)
    # lasts: meta(100), google(50), google(60) -> google 110
    assert credit["meta"] == 100.0
    assert credit["google"] == 110.0
    assert "tiktok" not in credit  # tiktok was never a last touch


def test_linear_split(paths):
    res = attribution.attribute(paths, "linear")
    credit = _credit(res)
    # path1: 100/3 each to meta,google,meta -> meta 66.67, google 33.33
    # path2: google 50
    # path3: tiktok 30, google 30
    assert credit["meta"] == pytest.approx(66.67, abs=0.01)
    assert credit["google"] == pytest.approx(33.33 + 50 + 30, abs=0.01)
    assert credit["tiktok"] == pytest.approx(30.0, abs=0.01)


def test_time_decay_weights_last_touch_more(paths):
    res = attribution.attribute(paths, "time_decay", decay_factor=0.5)
    credit = _credit(res)
    # path1 ["meta","google","meta"], weights 0.25/0.5/1 -> norm 1/7,2/7,4/7
    # meta gets (1/7 + 4/7)*100 = 71.43, google (2/7)*100 = 28.57
    # path2 google 50; path3 ["tiktok","google"] weights .5,1 -> tiktok 20, google 40
    assert credit["meta"] == pytest.approx(71.43, abs=0.05)
    assert credit["google"] == pytest.approx(28.57 + 50 + 40, abs=0.05)
    assert credit["tiktok"] == pytest.approx(20.0, abs=0.05)


def test_value_is_conserved_across_rules(paths):
    total_in = paths["conversion_value"].sum()
    for rule in attribution.VALID_RULES:
        res = attribution.attribute(paths, rule)
        assert res["credited_value"].sum() == pytest.approx(total_in, abs=0.05)
        assert res["share"].sum() == pytest.approx(1.0, abs=1e-6)


def test_unknown_rule_raises(paths):
    with pytest.raises(ValueError, match="unknown attribution rule"):
        attribution.attribute(paths, "shapley")


def test_invalid_decay_factor_raises(paths):
    with pytest.raises(ValueError, match="decay_factor"):
        attribution.attribute(paths, "time_decay", decay_factor=0.0)


def test_empty_paths_are_skipped():
    df = pd.DataFrame([{"touchpoints": [], "conversion_value": 100.0}])
    res = attribution.attribute(df, "first")
    assert len(res) == 0


def test_compare_rules_pivot(paths):
    pivot = attribution.compare_rules(paths)
    assert set(pivot.columns) == set(attribution.VALID_RULES)
    assert set(pivot.index) == {"meta", "google", "tiktok"}
    # every rule column should sum to the same grand total
    totals = pivot.sum(axis=0)
    assert totals.nunique() == 1 or totals.std() < 0.05
