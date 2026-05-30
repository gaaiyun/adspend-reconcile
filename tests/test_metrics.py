"""Tests for the core marketing-efficiency arithmetic."""

import math

import pytest

from adspend_reconcile import metrics


def test_safe_div_normal():
    assert metrics.safe_div(10, 2) == 5.0


def test_safe_div_zero_denominator_returns_nan_by_default():
    assert math.isnan(metrics.safe_div(10, 0))


def test_safe_div_custom_default():
    assert metrics.safe_div(10, 0, default=0.0) == 0.0


def test_roas_basic():
    assert metrics.roas(1000, 250) == 4.0


def test_roas_zero_spend_is_nan_not_inf():
    result = metrics.roas(1000, 0)
    assert math.isnan(result)
    assert not math.isinf(result)


def test_mer_blended():
    # total revenue 4200 over total spend 1200 -> 3.5
    assert metrics.mer(4200, 1200) == 3.5


def test_roi_fraction():
    # (1500 - 1000) / 1000 = 0.5
    assert metrics.roi(1500, 1000) == 0.5


def test_roi_negative_when_underwater():
    assert metrics.roi(800, 1000) == pytest.approx(-0.2)


def test_contribution_margin_full_stack():
    # 1000 revenue, 350 COGS, 80 shipping, 30 fees, 50 refunds
    cm = metrics.contribution_margin(1000, cogs=350, shipping=80, fees=30, refunds=50)
    assert cm == 490.0


def test_contribution_margin_defaults_to_revenue():
    assert metrics.contribution_margin(1000) == 1000.0


def test_contribution_margin_roas():
    assert metrics.contribution_margin_roas(500, 250) == 2.0


def test_marginal_profit_positive_and_negative():
    assert metrics.marginal_profit(500, 200) == 300.0
    assert metrics.marginal_profit(150, 200) == -50.0


def test_break_even_roas():
    # 40% contribution margin -> need 2.5x gross ROAS to break even
    assert metrics.break_even_roas(0.4) == pytest.approx(2.5)


def test_break_even_roas_zero_margin_is_nan():
    assert math.isnan(metrics.break_even_roas(0.0))
