"""Deterministic sample-data generators.

Produces small, realistic-looking ad-platform and store exports so the CLI is
fully runnable offline and the test-suite has fixtures. The data is seeded, so
output is byte-stable across runs. Channel names, currency formatting and a few
deliberate discrepancies (platforms over-reporting revenue, an organic channel
with revenue but no spend, one channel with spend but no sales) are baked in so
that ``reconcile`` produces a non-trivial, instructive breakdown.

All numbers are fictional. No real account data is included.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# A short, fixed date window keeps sample files tiny and diff-friendly.
_DEFAULT_DAYS = 14
_START = pd.Timestamp("2024-01-01")

# Per-channel behaviour knobs: daily spend level, the platform's revenue
# over-reporting factor (platforms tend to over-claim), and the true store
# conversion strength. "organic" has no ad spend; "tiktok" spends but barely
# converts in-store (a classic "looks busy, sells nothing" channel).
_CHANNELS = {
    "meta": {"spend": 320.0, "platform_overreport": 1.35, "store_strength": 3.1},
    "google": {"spend": 410.0, "platform_overreport": 1.12, "store_strength": 4.2},
    "tiktok": {"spend": 180.0, "platform_overreport": 1.80, "store_strength": 0.7},
    "organic": {"spend": 0.0, "platform_overreport": 0.0, "store_strength": 0.0},
}


def _dates(days: int) -> pd.DatetimeIndex:
    return pd.date_range(_START, periods=days, freq="D")


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def build_ads_frame(channel: str, days: int = _DEFAULT_DAYS, seed: int = 7) -> pd.DataFrame:
    """Build a per-platform ads frame (canonical-ish columns) for one channel."""
    cfg = _CHANNELS[channel]
    rng = _rng(seed + hash(channel) % 1000)
    rows = []
    for d in _dates(days):
        if cfg["spend"] <= 0:
            continue  # organic has no ad account
        spend = round(cfg["spend"] * rng.uniform(0.7, 1.3), 2)
        # platform-reported conversions and the revenue the platform claims
        conv = int(round(spend / 25.0 * rng.uniform(0.8, 1.2)))
        true_store_rev = spend * cfg["store_strength"] * rng.uniform(0.85, 1.15)
        platform_rev = round(true_store_rev * cfg["platform_overreport"], 2)
        rows.append(
            {
                "date": d.date().isoformat(),
                "campaign": f"{channel}_prospecting",
                "spend": spend,
                "conversions": conv,
                "conversion_value": platform_rev,
            }
        )
    return pd.DataFrame(rows)


def build_store_frame(days: int = _DEFAULT_DAYS, seed: int = 7) -> pd.DataFrame:
    """Build a Shopify-style store-revenue frame across all channels."""
    rows = []
    for channel, cfg in _CHANNELS.items():
        rng = _rng(seed * 3 + hash(channel) % 1000)
        for d in _dates(days):
            if channel == "organic":
                revenue = round(rng.uniform(150, 400), 2)  # free traffic still sells
            elif cfg["spend"] <= 0:
                revenue = 0.0
            else:
                spend = cfg["spend"] * rng.uniform(0.7, 1.3)
                revenue = round(spend * cfg["store_strength"] * rng.uniform(0.85, 1.15), 2)
            if revenue <= 0:
                continue
            orders = max(1, int(round(revenue / 80.0)))
            refunds = round(revenue * rng.uniform(0.0, 0.06), 2)
            rows.append(
                {
                    "day": d.date().isoformat(),
                    "channel": channel,
                    "total sales": revenue,
                    "orders": orders,
                    "returns": refunds,
                }
            )
    return pd.DataFrame(rows)


# --- Platform-specific header styling --------------------------------------
# Rename canonical columns to each platform's real-world export headers so the
# ingest fuzzy-matcher / presets get genuinely exercised.

_META_HEADERS = {
    "date": "Day",
    "campaign": "Campaign name",
    "spend": "Amount spent (USD)",
    "conversions": "Results",
    "conversion_value": "Purchases conversion value",
}
_GOOGLE_HEADERS = {
    "date": "Day",
    "campaign": "Campaign",
    "spend": "Cost",
    "conversions": "Conversions",
    "conversion_value": "Conv. value",
}
_TIKTOK_HEADERS = {
    "date": "Date",
    "campaign": "Campaign name",
    "spend": "Cost",
    "conversions": "Conversions",
    "conversion_value": "Total complete payment value",
}
_PLATFORM_HEADERS = {
    "meta": _META_HEADERS,
    "google": _GOOGLE_HEADERS,
    "tiktok": _TIKTOK_HEADERS,
}


def write_samples(out_dir: str | Path, days: int = _DEFAULT_DAYS, seed: int = 7) -> dict[str, Path]:
    """Write the full sample bundle to ``out_dir`` and return the file map."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    for platform, headers in _PLATFORM_HEADERS.items():
        df = build_ads_frame(platform, days=days, seed=seed).rename(columns=headers)
        path = out_dir / f"{platform}_ads_sample.csv"
        df.to_csv(path, index=False)
        written[platform] = path

    store_path = out_dir / "shopify_revenue_sample.csv"
    build_store_frame(days=days, seed=seed).to_csv(store_path, index=False)
    written["shopify"] = store_path

    return written


def create_sample_conversion_paths(n_samples: int = 200, seed: int = 42) -> pd.DataFrame:
    """Sample multi-touch conversion paths for the attribution module.

    One row per conversion: ``touchpoints`` (ordered channel list) and
    ``conversion_value``.
    """
    rng = _rng(seed)
    channels = ["meta", "google", "tiktok", "organic", "email"]
    rows = []
    for i in range(n_samples):
        n_touch = int(rng.integers(1, 5))
        touchpoints = rng.choice(channels, size=n_touch, replace=True).tolist()
        value = round(float(rng.exponential(scale=90) + 10), 2)
        rows.append(
            {
                "customer_id": f"C{i:05d}",
                "touchpoints": touchpoints,
                "conversion_value": value,
            }
        )
    return pd.DataFrame(rows)
