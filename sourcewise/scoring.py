"""Weighted supplier scorecard.

Steps:
    1. **Hard filter** — suppliers that do not meet the required capability are
       removed from ranking (no amount of low cost compensates for a garment
       that can't be made to spec).
    2. **Normalise** each factor to 0–1 across the *eligible* suppliers with
       min-max scaling, inverting factors where lower is better.
    3. **Weight** — weights are rescaled to sum to 1, and each supplier's score
       is ``Σ normalised_factor × weight × 100`` (a score out of 100).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Factor:
    """A scorecard factor and how to read it.

    Attributes:
        key: Short identifier used for weights.
        column: Supplier column holding the raw value.
        label: Name shown in the UI.
        higher_is_better: False for cost-like factors that are inverted.
        help: One-line explanation for a non-technical reader.
    """

    key: str
    column: str
    label: str
    higher_is_better: bool
    help: str


FACTORS: tuple[Factor, ...] = (
    Factor("quality", "sample_quality", "Quality", True,
           "How good the samples were (1–5). Higher is better."),
    Factor("capability", "capability_score", "Capability", True,
           "Stitching, finishing and silhouette capability (1–5). Higher is better."),
    Factor("cost", "unit_cost", "Cost", False,
           "Price per piece in ₹. Lower is better."),
    Factor("moq", "moq_per_style", "MOQ", False,
           "Minimum units you must order per style. Lower means less inventory risk."),
    Factor("lead_time", "lead_time_days", "Lead time", False,
           "Days to deliver the first production run. Lower is better."),
    Factor("reliability", "reliability_score", "Reliability", True,
           "Communication, responsiveness and meeting deadlines (1–5). Higher is better."),
    Factor("transparency", "costing_transparency", "Costing transparency", True,
           "How openly they break down their costing (1–5). Higher is better."),
    Factor("advance", "advance_payment_pct", "Advance payment", False,
           "Share of the order paid upfront. Lower protects your cash flow."),
)

FACTOR_BY_KEY: dict[str, Factor] = {f.key: f for f in FACTORS}

#: Default slider positions (relative importance, 0–10). Quality, capability
#: and reliability matter most for a small fashion brand; cost and MOQ next.
DEFAULT_WEIGHTS: dict[str, float] = {
    "quality": 8,
    "capability": 7,
    "cost": 7,
    "moq": 6,
    "lead_time": 4,
    "reliability": 7,
    "transparency": 3,
    "advance": 3,
}

EXCLUSION_REASON = "Does not meet the required capability (hard requirement)"


@dataclass
class ScorecardResult:
    """Output of :func:`score_suppliers`.

    Attributes:
        ranked: Eligible suppliers sorted best-first, with ``rank``, ``score``,
            ``norm_<key>`` (0–1) and ``pts_<key>`` (points out of 100) columns.
        excluded: Suppliers removed by hard filters, with an ``exclusion_reason``.
        weights: The normalised weights actually used (sum to 1).
    """

    ranked: pd.DataFrame
    excluded: pd.DataFrame
    weights: dict[str, float] = field(default_factory=dict)

    def contributions(self) -> pd.DataFrame:
        """Long-format points per supplier and factor, for stacked bar charts."""
        cols = [f"pts_{f.key}" for f in FACTORS]
        long = self.ranked.melt(
            id_vars=["rank", "supplier_name", "score"], value_vars=cols,
            var_name="factor", value_name="points",
        )
        long["factor"] = long["factor"].str.removeprefix("pts_").map(lambda k: FACTOR_BY_KEY[k].label)
        return long


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    """Rescale weights so they sum to 1.

    Missing factors get weight 0. Unknown keys are rejected so a typo can't
    silently drop a factor.

    Raises:
        ValueError: if a weight is negative, a key is unknown, or all are zero.
    """
    unknown = set(weights) - set(FACTOR_BY_KEY)
    if unknown:
        raise ValueError(f"Unknown weight factor(s): {', '.join(sorted(unknown))}")
    full = {f.key: float(weights.get(f.key, 0.0)) for f in FACTORS}
    if any(w < 0 for w in full.values()):
        raise ValueError("Weights cannot be negative.")
    total = sum(full.values())
    if total <= 0:
        raise ValueError("At least one weight must be greater than zero.")
    return {k: w / total for k, w in full.items()}


def min_max_normalize(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    """Scale values to 0–1 with min-max normalisation.

    ``(x - min) / (max - min)`` when higher is better, and
    ``(max - x) / (max - min)`` when lower is better, so 1 is always best.

    If every value is identical (including a single supplier) the factor does
    not separate anyone, so all get 1.0 — "as good as the best available" —
    instead of dividing by zero.
    """
    values = values.astype(float)
    low, high = values.min(), values.max()
    if pd.isna(low) or high == low:
        return pd.Series(1.0, index=values.index)
    scaled = (values - low) / (high - low)
    return scaled if higher_is_better else 1.0 - scaled


def apply_hard_filters(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split suppliers into (eligible, excluded) using hard requirements.

    Currently the only hard requirement is ``meets_required_capability``.
    Excluded rows get an ``exclusion_reason`` column.
    """
    passes = df["meets_required_capability"].astype(bool)
    eligible = df[passes].copy()
    excluded = df[~passes].copy()
    excluded["exclusion_reason"] = EXCLUSION_REASON
    return eligible, excluded


def score_suppliers(df: pd.DataFrame, weights: dict[str, float] | None = None) -> ScorecardResult:
    """Apply hard filters, normalise factors and compute weighted scores.

    Normalisation uses only eligible suppliers, so an excluded supplier (e.g.
    a very cheap one that can't make the garment) does not stretch the scale.

    Ties are broken by lower unit cost, then supplier name, so ranks are stable.

    Args:
        df: A validated supplier DataFrame.
        weights: Relative importance per factor key; defaults to DEFAULT_WEIGHTS.

    Returns:
        A :class:`ScorecardResult`.
    """
    norm_weights = normalize_weights(weights if weights is not None else DEFAULT_WEIGHTS)
    eligible, excluded = apply_hard_filters(df)

    ranked = eligible.copy()
    ranked["score"] = 0.0
    for factor in FACTORS:
        norm = min_max_normalize(ranked[factor.column], factor.higher_is_better)
        ranked[f"norm_{factor.key}"] = norm
        ranked[f"pts_{factor.key}"] = norm * norm_weights[factor.key] * 100
        ranked["score"] += ranked[f"pts_{factor.key}"]

    ranked = ranked.sort_values(
        ["score", "unit_cost", "supplier_name"], ascending=[False, True, True], kind="mergesort"
    ).reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ScorecardResult(ranked=ranked, excluded=excluded.reset_index(drop=True), weights=norm_weights)
