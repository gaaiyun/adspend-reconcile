"""CSV ingestion with platform presets and fuzzy column matching.

Real ad-platform exports never agree on column names: Meta calls it
``Amount spent (USD)``, Google calls it ``Cost``, TikTok calls it ``Cost``,
and your Shopify export calls revenue ``Total sales``. This module normalises
all of them into one tidy schema:

    date | channel | spend | platform_conversions | platform_revenue   (ads)
    date | channel | revenue | orders | refunds                        (store)

Matching strategy, in priority order:
1. An explicit ``--map`` override from the caller.
2. The platform preset (exact, case-insensitive header match).
3. Fuzzy matching against a set of known aliases (token overlap + difflib).

Nothing here calls the network. You hand it a file, it hands you a DataFrame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

# Canonical target fields for each kind of file.
#
# Ads exports are single-platform, so their `channel` is the platform itself
# (set from the preset's default_channel or --channel) rather than a column in
# the file. The campaign name, if present, is kept as a separate audit column.
AD_FIELDS = ("date", "spend", "platform_conversions", "platform_revenue")
STORE_FIELDS = ("date", "channel", "revenue", "orders", "refunds")


@dataclass
class Preset:
    """A named set of header aliases for one source kind."""

    name: str
    kind: str  # "ads" or "store"
    # canonical field -> list of known source-header aliases (lower-cased)
    aliases: dict[str, list[str]] = field(default_factory=dict)
    # a constant channel label when the file is single-platform and has no channel column
    default_channel: str | None = None


# --- Platform presets -------------------------------------------------------

PRESETS: dict[str, Preset] = {
    "meta": Preset(
        name="meta",
        kind="ads",
        default_channel="meta",
        aliases={
            "date": ["day", "date", "reporting starts", "reporting_starts"],
            "campaign": ["campaign name", "account name", "ad set name"],
            "spend": ["amount spent (usd)", "amount spent", "spend", "cost"],
            "platform_conversions": [
                "results",
                "purchases",
                "website purchases",
                "conversions",
            ],
            "platform_revenue": [
                "purchases conversion value",
                "website purchases conversion value",
                "conversion value",
                "revenue",
            ],
        },
    ),
    "google": Preset(
        name="google",
        kind="ads",
        default_channel="google",
        aliases={
            "date": ["day", "date"],
            "campaign": ["campaign", "campaign name"],
            "spend": ["cost", "spend", "amount"],
            "platform_conversions": ["conversions", "conv.", "all conv."],
            "platform_revenue": [
                "conv. value",
                "conversion value",
                "total conv. value",
                "revenue",
            ],
        },
    ),
    "tiktok": Preset(
        name="tiktok",
        kind="ads",
        default_channel="tiktok",
        aliases={
            "date": ["date", "stat time day", "day"],
            "campaign": ["campaign name", "ad group name"],
            "spend": ["cost", "spend", "total cost"],
            "platform_conversions": [
                "conversions",
                "conversion",
                "complete payment",
                "purchases",
            ],
            "platform_revenue": [
                "total complete payment value",
                "complete payment value",
                "purchase value",
                "revenue",
            ],
        },
    ),
    "shopify": Preset(
        name="shopify",
        kind="store",
        aliases={
            "date": ["day", "date", "order date"],
            "channel": ["channel", "referring channel", "marketing channel", "source"],
            "revenue": ["total sales", "net sales", "gross sales", "revenue", "sales"],
            "orders": ["orders", "order count", "net orders"],
            "refunds": ["returns", "refunds", "refund amount"],
        },
    ),
    "stripe": Preset(
        name="stripe",
        kind="store",
        aliases={
            "date": ["created (utc)", "created", "date"],
            "channel": ["channel", "metadata.channel", "source"],
            "revenue": ["amount", "gross", "net", "revenue"],
            "orders": ["count", "orders"],
            "refunds": ["refunded amount", "amount refunded", "refunds"],
        },
    ),
}


def _norm(text: str) -> str:
    """Lower-case and collapse non-alphanumerics to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def _similarity(a: str, b: str) -> float:
    """Blend token-overlap (Jaccard) with character ratio for header matching."""
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    ta, tb = set(na.split()), set(nb.split())
    jaccard = len(ta & tb) / len(ta | tb) if (ta | tb) else 0.0
    ratio = SequenceMatcher(None, na, nb).ratio()
    return max(jaccard, ratio)


@dataclass
class ColumnMatch:
    """The resolved source header for one canonical field."""

    field: str
    source_header: str | None
    score: float
    method: str  # "override" | "preset" | "exact" | "fuzzy" | "missing"


def resolve_columns(
    headers: list[str],
    target_fields: tuple[str, ...],
    preset: Preset | None,
    overrides: dict[str, str] | None = None,
    threshold: float = 0.55,
) -> dict[str, ColumnMatch]:
    """Map each canonical field to the best source header.

    Args:
        headers: the actual column names found in the file.
        target_fields: canonical fields we want to fill.
        preset: optional platform preset providing aliases / default channel.
        overrides: explicit ``{canonical_field: source_header}`` mapping that
            always wins.
        threshold: minimum similarity for a fuzzy match to be accepted.
    """
    overrides = overrides or {}
    norm_headers = {h: _norm(h) for h in headers}
    matches: dict[str, ColumnMatch] = {}

    for fld in target_fields:
        # 1. explicit override
        if fld in overrides:
            matches[fld] = ColumnMatch(fld, overrides[fld], 1.0, "override")
            continue

        candidates_alias = preset.aliases.get(fld, []) if preset else []
        norm_aliases = {_norm(a) for a in candidates_alias}

        # 2. exact (normalised) match against the canonical name or a preset alias
        exact_hit, exact_method = None, None
        for header, nh in norm_headers.items():
            if nh in norm_aliases:
                exact_hit, exact_method = header, "preset"
                break
            if nh == fld:
                exact_hit, exact_method = header, "exact"
                break
        if exact_hit is not None:
            matches[fld] = ColumnMatch(fld, exact_hit, 1.0, exact_method)
            continue

        # 3. fuzzy against (field name + aliases)
        targets = [fld] + candidates_alias
        best_header, best_score = None, 0.0
        for header in headers:
            score = max(_similarity(header, t) for t in targets)
            if score > best_score:
                best_header, best_score = header, score

        if best_header is not None and best_score >= threshold:
            matches[fld] = ColumnMatch(fld, best_header, round(best_score, 3), "fuzzy")
        else:
            matches[fld] = ColumnMatch(fld, None, round(best_score, 3), "missing")

    return matches


def _to_numeric(series: pd.Series) -> pd.Series:
    """Strip currency symbols / thousands separators, coerce to float."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": None, "-": None})
    )
    return pd.to_numeric(cleaned, errors="coerce").fillna(0.0)


@dataclass
class IngestResult:
    """Tidy frame plus the column mapping used to produce it (for auditing)."""

    frame: pd.DataFrame
    matches: dict[str, ColumnMatch]
    kind: str
    preset: str | None

    def mapping_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "field": m.field,
                    "source_column": m.source_header,
                    "method": m.method,
                    "score": m.score,
                }
                for m in self.matches.values()
            ]
        )


def ingest_csv(
    path: str | Path,
    kind: str,
    preset: str | None = None,
    overrides: dict[str, str] | None = None,
    default_channel: str | None = None,
    threshold: float = 0.55,
) -> IngestResult:
    """Read a CSV into the canonical ads/store schema.

    Args:
        path: CSV file path.
        kind: ``"ads"`` or ``"store"``.
        preset: a key in :data:`PRESETS` (e.g. ``"meta"``). Optional; without it
            matching relies on fuzzy header similarity alone.
        overrides: explicit ``{canonical_field: source_header}`` pairs.
        default_channel: channel label to use when the file has no channel
            column (overrides any preset default).
        threshold: fuzzy-match acceptance threshold.

    Returns:
        :class:`IngestResult` with a normalised, daily-grained frame.

    Raises:
        ValueError: if ``kind`` is invalid or a required field cannot be mapped.
        FileNotFoundError: if ``path`` does not exist.
    """
    if kind not in ("ads", "store"):
        raise ValueError(f"kind must be 'ads' or 'store', got {kind!r}")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"no such CSV: {path}")

    preset_obj = PRESETS.get(preset) if preset else None
    if preset and preset_obj is None:
        raise ValueError(f"unknown preset {preset!r}; known: {sorted(PRESETS)}")
    if preset_obj and preset_obj.kind != kind:
        raise ValueError(
            f"preset {preset!r} is for kind={preset_obj.kind!r}, not {kind!r}"
        )

    target_fields = AD_FIELDS if kind == "ads" else STORE_FIELDS
    # For ads we also try to locate (but don't require) a campaign column.
    resolve_fields = target_fields + (("campaign",) if kind == "ads" else ())

    raw = pd.read_csv(path)
    raw.columns = [str(c).strip() for c in raw.columns]

    matches = resolve_columns(
        list(raw.columns), resolve_fields, preset_obj, overrides, threshold
    )

    # Determine the channel fallback. For ads this is the canonical channel;
    # for store it is only used when the file lacks a channel column.
    fallback_channel = default_channel or (preset_obj.default_channel if preset_obj else None)

    # Required fields: date + the money field.
    money_field = "spend" if kind == "ads" else "revenue"
    for required in ("date", money_field):
        if matches[required].source_header is None:
            raise ValueError(
                f"could not map required field {required!r} for a {kind} file. "
                f"Found columns: {list(raw.columns)}. "
                f"Use --map {required}=<your column> to set it explicitly."
            )

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(raw[matches["date"].source_header], errors="coerce").dt.date

    if kind == "ads":
        # Ad exports are single-platform: channel is the platform label.
        if not fallback_channel:
            raise ValueError(
                "an ads file needs a channel label. Pass a preset (e.g. "
                "--preset meta) or set --channel <name>."
            )
        out["channel"] = fallback_channel
    else:
        ch_src = matches["channel"].source_header
        if ch_src is not None:
            out["channel"] = raw[ch_src].astype(str).str.strip().str.lower()
        elif fallback_channel:
            out["channel"] = fallback_channel
        else:
            raise ValueError(
                "no channel column found in the store file and no --channel set."
            )

    for fld in target_fields:
        if fld in ("date", "channel"):
            continue
        src = matches[fld].source_header
        out[fld] = _to_numeric(raw[src]) if src is not None else 0.0

    # Drop rows whose date failed to parse, then aggregate to daily x channel.
    out = out.dropna(subset=["date"])
    numeric_cols = [c for c in out.columns if c not in ("date", "channel")]
    out = (
        out.groupby(["date", "channel"], as_index=False)[numeric_cols]
        .sum()
        .sort_values(["date", "channel"], ignore_index=True)
    )

    return IngestResult(frame=out, matches=matches, kind=kind, preset=preset)
