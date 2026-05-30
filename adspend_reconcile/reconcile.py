"""Date x channel reconciliation between ad spend and real store revenue.

This is the core of the tool. Given a tidy *ads* frame (spend + the revenue
each platform *claims* it drove) and a tidy *store* frame (the revenue that
actually landed, e.g. Shopify / Stripe), it:

1. Full-outer-joins them on ``date x channel`` (via DuckDB).
2. Classifies every cell:

   * ``matched``      - spend and store revenue both present, and the platform's
                        self-reported revenue agrees with the store within a
                        tolerance.
   * ``discrepant``   - both present, but platform-reported revenue diverges
                        from store revenue beyond the tolerance. This is the
                        "why don't the numbers line up" bucket.
   * ``unattributed`` - store revenue with no matching spend (organic / direct),
                        or spend with no store revenue traced to that channel.

3. Applies a profit model (COGS / shipping / processing fees / refunds) to the
   *store* revenue to get contribution margin, and from there
   contribution-margin ROAS, gross ROAS, channel marginal profit, and a single
   blended MER for the whole account.

The output is intentionally boring and auditable: one row per date x channel
with every input and derived number side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import pandas as pd

from . import metrics

MATCHED = "matched"
DISCREPANT = "discrepant"
UNATTRIBUTED = "unattributed"


@dataclass
class ProfitModel:
    """Variable-cost assumptions used to turn revenue into contribution margin.

    All rates are fractions of revenue (0.30 == 30%). ``processing_fee_rate``
    plus ``processing_fee_flat`` model a Stripe-style "2.9% + $0.30" per order.
    Refunds are taken from the store data directly, not assumed here.
    """

    cogs_rate: float = 0.0
    shipping_rate: float = 0.0
    processing_fee_rate: float = 0.0
    processing_fee_flat: float = 0.0

    def variable_costs(self, revenue: float, orders: float) -> dict[str, float]:
        cogs = revenue * self.cogs_rate
        shipping = revenue * self.shipping_rate
        fees = revenue * self.processing_fee_rate + orders * self.processing_fee_flat
        return {"cogs": round(cogs, 2), "shipping": round(shipping, 2), "fees": round(fees, 2)}


@dataclass
class ReconcileResult:
    """Per-cell reconciliation table plus account-level summary."""

    cells: pd.DataFrame
    summary: dict = field(default_factory=dict)

    def by_classification(self) -> pd.DataFrame:
        """Aggregate spend / revenue / margin grouped by classification bucket."""
        agg = (
            self.cells.groupby("classification", as_index=False)
            .agg(
                cells=("classification", "size"),
                spend=("spend", "sum"),
                platform_revenue=("platform_revenue", "sum"),
                store_revenue=("store_revenue", "sum"),
                contribution_margin=("contribution_margin", "sum"),
            )
            .sort_values("classification", ignore_index=True)
        )
        for col in ("spend", "platform_revenue", "store_revenue", "contribution_margin"):
            agg[col] = agg[col].round(2)
        return agg

    def by_channel(self) -> pd.DataFrame:
        """Aggregate the full profit picture per channel."""
        grp = self.cells.groupby("channel", as_index=False).agg(
            spend=("spend", "sum"),
            platform_revenue=("platform_revenue", "sum"),
            store_revenue=("store_revenue", "sum"),
            refunds=("refunds", "sum"),
            contribution_margin=("contribution_margin", "sum"),
        )
        grp["gross_roas"] = grp.apply(
            lambda r: metrics.roas(r["store_revenue"], r["spend"]), axis=1
        )
        grp["cm_roas"] = grp.apply(
            lambda r: metrics.contribution_margin_roas(r["contribution_margin"], r["spend"]),
            axis=1,
        )
        grp["marginal_profit"] = grp.apply(
            lambda r: metrics.marginal_profit(r["contribution_margin"], r["spend"]), axis=1
        )
        for col in (
            "spend",
            "platform_revenue",
            "store_revenue",
            "refunds",
            "contribution_margin",
            "gross_roas",
            "cm_roas",
            "marginal_profit",
        ):
            grp[col] = grp[col].round(2)
        return grp.sort_values("spend", ascending=False, ignore_index=True)


def _classify(
    spend: float,
    platform_revenue: float,
    store_revenue: float,
    tolerance: float,
    min_spend: float,
    min_revenue: float,
) -> str:
    """Bucket a single cell. ``tolerance`` is a relative gap fraction."""
    has_spend = spend > min_spend
    has_store = store_revenue > min_revenue

    if has_spend and has_store:
        # Compare what the platform claimed vs what the store actually booked.
        denom = max(store_revenue, platform_revenue, 1e-9)
        rel_gap = abs(platform_revenue - store_revenue) / denom
        return MATCHED if rel_gap <= tolerance else DISCREPANT

    # Spend without store revenue, or store revenue without spend.
    return UNATTRIBUTED


def reconcile(
    ads: pd.DataFrame,
    store: pd.DataFrame,
    profit_model: ProfitModel | None = None,
    *,
    tolerance: float = 0.15,
    min_spend: float = 0.0,
    min_revenue: float = 0.0,
) -> ReconcileResult:
    """Reconcile ad spend against store revenue by date x channel.

    Args:
        ads: tidy ads frame (columns: date, channel, spend, platform_conversions,
            platform_revenue). Extra columns are ignored.
        store: tidy store frame (columns: date, channel, revenue, orders,
            refunds). Extra columns are ignored.
        profit_model: variable-cost assumptions; defaults to all-zero (so
            contribution margin equals net revenue minus refunds).
        tolerance: relative gap between platform-reported and store revenue that
            still counts as ``matched`` (0.15 == within 15%).
        min_spend: spend at or below this is treated as "no spend".
        min_revenue: store revenue at or below this is treated as "no revenue".

    Returns:
        :class:`ReconcileResult`.
    """
    profit_model = profit_model or ProfitModel()

    ads = ads.copy()
    store = store.copy()
    for col in ("platform_conversions", "platform_revenue"):
        if col not in ads.columns:
            ads[col] = 0.0
    for col in ("orders", "refunds"):
        if col not in store.columns:
            store[col] = 0.0

    # Full outer join on date x channel using DuckDB.
    con = duckdb.connect(":memory:")
    try:
        con.register("ads", ads)
        con.register("store", store)
        joined = con.execute(
            """
            SELECT
                COALESCE(a.date, s.date)                       AS date,
                COALESCE(a.channel, s.channel)                 AS channel,
                COALESCE(a.spend, 0.0)                         AS spend,
                COALESCE(a.platform_conversions, 0.0)          AS platform_conversions,
                COALESCE(a.platform_revenue, 0.0)              AS platform_revenue,
                COALESCE(s.revenue, 0.0)                       AS store_revenue,
                COALESCE(s.orders, 0.0)                        AS orders,
                COALESCE(s.refunds, 0.0)                       AS refunds
            FROM ads AS a
            FULL OUTER JOIN store AS s
              ON a.date = s.date AND a.channel = s.channel
            ORDER BY date, channel
            """
        ).fetchdf()
    finally:
        con.close()

    # Profit model -> contribution margin, applied to store revenue.
    cogs, shipping, fees, contrib, classes = [], [], [], [], []
    for row in joined.itertuples(index=False):
        costs = profit_model.variable_costs(row.store_revenue, row.orders)
        cm = metrics.contribution_margin(
            revenue=row.store_revenue,
            cogs=costs["cogs"],
            shipping=costs["shipping"],
            fees=costs["fees"],
            refunds=row.refunds,
        )
        cogs.append(costs["cogs"])
        shipping.append(costs["shipping"])
        fees.append(costs["fees"])
        contrib.append(round(cm, 2))
        classes.append(
            _classify(
                row.spend,
                row.platform_revenue,
                row.store_revenue,
                tolerance,
                min_spend,
                min_revenue,
            )
        )

    joined["cogs"] = cogs
    joined["shipping"] = shipping
    joined["fees"] = fees
    joined["contribution_margin"] = contrib
    joined["classification"] = classes
    joined["revenue_gap"] = (joined["platform_revenue"] - joined["store_revenue"]).round(2)

    result = ReconcileResult(cells=joined)
    result.summary = _summarize(joined, tolerance)
    return result


def _summarize(cells: pd.DataFrame, tolerance: float) -> dict:
    """Account-level rollup: totals, blended MER, classification split."""
    total_spend = float(cells["spend"].sum())
    total_store_rev = float(cells["store_revenue"].sum())
    total_platform_rev = float(cells["platform_revenue"].sum())
    total_cm = float(cells["contribution_margin"].sum())

    counts = cells["classification"].value_counts().to_dict()

    return {
        "tolerance": tolerance,
        "date_min": str(cells["date"].min()) if len(cells) else None,
        "date_max": str(cells["date"].max()) if len(cells) else None,
        "channels": sorted(cells["channel"].unique().tolist()),
        "cells_total": int(len(cells)),
        "cells_matched": int(counts.get(MATCHED, 0)),
        "cells_discrepant": int(counts.get(DISCREPANT, 0)),
        "cells_unattributed": int(counts.get(UNATTRIBUTED, 0)),
        "total_spend": round(total_spend, 2),
        "total_platform_revenue": round(total_platform_rev, 2),
        "total_store_revenue": round(total_store_rev, 2),
        "total_contribution_margin": round(total_cm, 2),
        # Platform self-report vs reality: the headline reconciliation number.
        "platform_vs_store_gap": round(total_platform_rev - total_store_rev, 2),
        "blended_mer": round(metrics.mer(total_store_rev, total_spend), 4),
        "blended_gross_roas": round(metrics.roas(total_store_rev, total_spend), 4),
        "blended_cm_roas": round(
            metrics.contribution_margin_roas(total_cm, total_spend), 4
        ),
        "account_marginal_profit": round(
            metrics.marginal_profit(total_cm, total_spend), 2
        ),
    }
