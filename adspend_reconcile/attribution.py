"""Rule-based touch attribution.

Four deterministic, well-understood rules for splitting a conversion's value
across the channels in its touch path:

* ``first`` - 100% to the first touch.
* ``last`` - 100% to the last touch.
* ``linear`` - split evenly across every touch.
* ``time_decay`` - exponential weighting toward touches nearer the conversion.

These are heuristics, not causal claims. They are useful for sanity-checking
how much a *rule choice* alone moves per-channel credit, which is exactly the
kind of thing that makes platform-reported numbers disagree. A statistical /
Shapley / MMM-style attribution is intentionally out of scope (see README).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Sequence

import pandas as pd

VALID_RULES = ("first", "last", "linear", "time_decay")


def _credit_one_path(
    touchpoints: Sequence[str],
    value: float,
    rule: str,
    decay_factor: float,
) -> dict[str, float]:
    """Return {channel: credited_value} for a single conversion path."""
    if not touchpoints:
        return {}

    if rule == "first":
        return {touchpoints[0]: value}

    if rule == "last":
        return {touchpoints[-1]: value}

    if rule == "linear":
        share = value / len(touchpoints)
        credited: dict[str, float] = defaultdict(float)
        for ch in touchpoints:
            credited[ch] += share
        return dict(credited)

    if rule == "time_decay":
        n = len(touchpoints)
        # Touch i gets weight decay_factor ** (distance from the conversion).
        weights = [decay_factor ** (n - i - 1) for i in range(n)]
        total_weight = sum(weights)
        credited = defaultdict(float)
        for ch, w in zip(touchpoints, weights):
            credited[ch] += value * (w / total_weight)
        return dict(credited)

    raise ValueError(f"unknown attribution rule: {rule!r} (expected one of {VALID_RULES})")


def attribute(
    paths: pd.DataFrame,
    rule: str = "last",
    *,
    touch_col: str = "touchpoints",
    value_col: str = "conversion_value",
    decay_factor: float = 0.5,
) -> pd.DataFrame:
    """Attribute conversion value to channels under a single rule.

    Args:
        paths: one row per conversion; ``touch_col`` holds an ordered list of
            channel names (earliest touch first), ``value_col`` the conversion
            value.
        rule: one of :data:`VALID_RULES`.
        decay_factor: weight multiplier per step away from the conversion, only
            used by ``time_decay`` (0 < factor <= 1; smaller decays faster).

    Returns:
        DataFrame with columns ``channel``, ``credited_value``, ``share`` (a
        fraction of total credited value) and ``rule``, sorted by credited
        value descending.
    """
    if rule not in VALID_RULES:
        raise ValueError(f"unknown attribution rule: {rule!r} (expected one of {VALID_RULES})")
    if not 0 < decay_factor <= 1:
        raise ValueError("decay_factor must be in the interval (0, 1]")

    totals: dict[str, float] = defaultdict(float)
    for touchpoints, value in zip(paths[touch_col], paths[value_col]):
        for ch, credit in _credit_one_path(list(touchpoints), float(value), rule, decay_factor).items():
            totals[ch] += credit

    grand_total = sum(totals.values())
    rows = [
        {
            "channel": ch,
            "credited_value": round(v, 2),
            "share": (v / grand_total) if grand_total else 0.0,
            "rule": rule,
        }
        for ch, v in totals.items()
    ]
    out = pd.DataFrame(rows, columns=["channel", "credited_value", "share", "rule"])
    return out.sort_values("credited_value", ascending=False, ignore_index=True)


def compare_rules(
    paths: pd.DataFrame,
    rules: Iterable[str] = VALID_RULES,
    *,
    touch_col: str = "touchpoints",
    value_col: str = "conversion_value",
    decay_factor: float = 0.5,
) -> pd.DataFrame:
    """Pivot table of per-channel credited value across several rules.

    Index is ``channel``; one column per rule. This is the view that shows how
    much the *choice of rule* alone shifts credit between channels.
    """
    frames = []
    for rule in rules:
        res = attribute(
            paths,
            rule,
            touch_col=touch_col,
            value_col=value_col,
            decay_factor=decay_factor,
        )
        frames.append(res.set_index("channel")["credited_value"].rename(rule))

    pivot = pd.concat(frames, axis=1).fillna(0.0)
    pivot.index.name = "channel"
    return pivot.sort_index()
