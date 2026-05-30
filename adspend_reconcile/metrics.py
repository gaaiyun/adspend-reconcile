"""Core marketing-efficiency arithmetic.

These are deliberately small, pure functions with explicit zero handling so the
reconciliation layer never silently produces ``inf`` or ``NaN`` in a report.

Definitions used throughout the project
----------------------------------------
ROAS (return on ad spend)
    revenue attributable to a channel / ad spend on that channel.
MER (marketing efficiency ratio, a.k.a. blended ROAS)
    total store revenue / total ad spend, across *all* channels. MER is the
    honest top-line number because it does not double-count overlapping
    platform attribution.
Contribution margin
    revenue - COGS - shipping - processing fees - refunds. This is the money
    left to cover fixed costs and profit *after* the variable cost of fulfilling
    an order, which is the number a ROAS should really be measured against.
Contribution-margin ROAS (a.k.a. profit ROAS)
    contribution margin / ad spend.
"""

from __future__ import annotations

import math

# A break-even-style sentinel returned when spend is zero but revenue is not.
# We never return math.inf so downstream formatting / comparisons stay sane.
UNDEFINED = float("nan")


def safe_div(numerator: float, denominator: float, default: float = UNDEFINED) -> float:
    """Divide, returning ``default`` when the denominator is zero / missing."""
    if denominator is None or numerator is None:
        return default
    if denominator == 0 or (isinstance(denominator, float) and math.isnan(denominator)):
        return default
    return numerator / denominator


def roas(revenue: float, spend: float) -> float:
    """Return on ad spend = revenue / spend."""
    return safe_div(revenue, spend)


def mer(total_revenue: float, total_spend: float) -> float:
    """Marketing efficiency ratio (blended ROAS) = total revenue / total spend."""
    return safe_div(total_revenue, total_spend)


def roi(revenue: float, spend: float) -> float:
    """Return on investment as a fraction = (revenue - spend) / spend.

    ROI of 0.5 means 50% return on top of the spend recovered.
    """
    return safe_div(revenue - spend, spend)


def contribution_margin(
    revenue: float,
    cogs: float = 0.0,
    shipping: float = 0.0,
    fees: float = 0.0,
    refunds: float = 0.0,
) -> float:
    """Contribution margin = revenue - COGS - shipping - fees - refunds."""
    return revenue - cogs - shipping - fees - refunds


def contribution_margin_roas(contribution: float, spend: float) -> float:
    """Profit ROAS = contribution margin / ad spend."""
    return safe_div(contribution, spend)


def marginal_profit(contribution: float, spend: float) -> float:
    """Channel marginal profit = contribution margin - ad spend.

    Positive means the channel pays for itself after variable costs; negative
    means every dollar of spend is currently destroying margin.
    """
    return contribution - spend


def break_even_roas(margin_rate: float) -> float:
    """Minimum gross ROAS needed to break even given a contribution-margin rate.

    If a product carries a 40% contribution margin (``margin_rate=0.4``), spend
    breaks even at a gross ROAS of 1 / 0.4 = 2.5.
    """
    return safe_div(1.0, margin_rate)
