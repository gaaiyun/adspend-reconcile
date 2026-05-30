"""Shared fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from adspend_reconcile import samples


@pytest.fixture
def sample_bundle(tmp_path: Path) -> dict[str, Path]:
    """Write the full sample CSV bundle into a temp dir and return the file map."""
    return samples.write_samples(tmp_path, days=14, seed=7)


@pytest.fixture
def tiny_ads() -> pd.DataFrame:
    """A hand-built ads frame with exactly known numbers."""
    return pd.DataFrame(
        [
            # meta over-reports revenue badly vs the store
            {"date": "2024-01-01", "channel": "meta", "spend": 100.0,
             "platform_conversions": 4, "platform_revenue": 500.0},
            # google reports almost exactly what the store booked
            {"date": "2024-01-01", "channel": "google", "spend": 200.0,
             "platform_conversions": 8, "platform_revenue": 600.0},
            # tiktok spends but drives no traceable store revenue
            {"date": "2024-01-01", "channel": "tiktok", "spend": 50.0,
             "platform_conversions": 1, "platform_revenue": 80.0},
        ]
    )


@pytest.fixture
def tiny_store() -> pd.DataFrame:
    """Store revenue that intentionally disagrees with the ads frame."""
    return pd.DataFrame(
        [
            {"date": "2024-01-01", "channel": "meta", "revenue": 300.0,
             "orders": 4, "refunds": 0.0},     # store << meta's claim -> discrepant
            {"date": "2024-01-01", "channel": "google", "revenue": 590.0,
             "orders": 7, "refunds": 0.0},     # within tolerance -> matched
            {"date": "2024-01-01", "channel": "organic", "revenue": 250.0,
             "orders": 3, "refunds": 0.0},     # no spend -> unattributed
            # tiktok has spend but no store revenue row -> unattributed
        ]
    )
