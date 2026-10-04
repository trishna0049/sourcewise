"""Final recommendation: combine scorecard, risk model and scaling check.

Decision score (out of 100, per candidate supplier)::

    risk_score     = 100 × (½ · norm(expected_profit) + ½ · norm_inverted(low_capital_at_risk))
    decision_score = w_scorecard × scorecard_score + w_risk × risk_score − scaling_penalty

where ``norm`` is the same min-max scaling used by the scorecard (computed
across the compared suppliers), ``w_scorecard + w_risk = 1`` and the scaling
penalty is 0 / 10 / 25 points for Green / Amber / Red.

The module also writes the plain-English trade-off, warnings and the
runner-up explanation, and exports everything to a small Excel workbook.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import pandas as pd

from sourcewise.formatting import indian_group, inr
from sourcewise.models import DISPLAY_LABELS, OPTIONAL_COLUMNS
from sourcewise.scoring import FACTORS, ScorecardResult, min_max_normalize


@dataclass(frozen=True)
class DecisionWeights:
    """How the three analyses are blended into one decision score.

    Attributes:
        scorecard: Relative weight of the weighted scorecard score.
        risk: Relative weight of the risk score (expected profit + downside).
        amber_penalty: Points deducted for an Amber scaling status.
        red_penalty: Points deducted for a Red scaling status.
    """

    scorecard: float = 0.6
    risk: float = 0.4
    amber_penalty: float = 10.0
    red_penalty: float = 25.0

    def normalised(self) -> tuple[float, float]:
        """Return (scorecard, risk) weights rescaled to sum to 1."""
        if self.scorecard < 0 or self.risk < 0:
            raise ValueError("Decision weights cannot be negative.")
        total = self.scorecard + self.risk
        if total <= 0:
            raise ValueError("At least one decision weight must be greater than zero.")
        return self.scorecard / total, self.risk / total

    def penalty(self, status: object) -> float:
        """Points deducted for a scaling status (unknown status → no penalty)."""
        return {"Amber": self.amber_penalty, "Red": self.red_penalty}.get(str(status), 0.0)


@dataclass(frozen=True)
class WarningThresholds:
    """Limits above which the recommended supplier gets a warning."""

    max_lead_time_days: int = 30
    high_advance_pct: float = 50.0
    low_reliability: float = 2.0


@dataclass
class Recommendation:
    """The final answer shown to the decision-maker."""

    recommended: str
    decision_score: float
    scorecard_score: float
    scorecard_rank: int
    trade_off: str
    warnings: list[str] = field(default_factory=list)
    runner_up: str | None = None
    runner_up_reason: str | None = None
    decision_table: pd.DataFrame = field(default_factory=pd.DataFrame)


def build_decision_table(
    scorecard: ScorecardResult,
    risk_summary: pd.DataFrame,
    scaling: pd.DataFrame,
    weights: DecisionWeights = DecisionWeights(),
) -> pd.DataFrame:
    """Merge the three analyses and compute the decision score per candidate.

    Candidates are the suppliers present in ``risk_summary`` that are also
    eligible in the scorecard (hard-filter failures can never be recommended).

    Returns:
        DataFrame sorted best-first with ``decision_rank``, ``decision_score``,
        ``risk_score``, ``scaling_penalty`` and all scorecard / risk / scaling
        fields. Ties fall back to the scorecard rank.
    """
    w_score, w_risk = weights.normalised()
    table = scorecard.ranked.merge(risk_summary, on="supplier_name", how="inner",
                                   suffixes=("", "_risk"))
    if table.empty:
        return table
    scaling_cols = ["supplier_name", "status", "explanation", "capacity_ok", "reorder_ok"]
    if not scaling.empty:
        table = table.merge(scaling[scaling_cols].rename(columns={"explanation": "scaling_explanation"}),
                            on="supplier_name", how="left")
    else:
        table["status"], table["scaling_explanation"] = None, None
        table["capacity_ok"], table["reorder_ok"] = True, True

    profit_norm = min_max_normalize(table["expected_profit"], higher_is_better=True)
    downside_norm = min_max_normalize(table["low_capital_at_risk"], higher_is_better=False)
    table["risk_score"] = 100 * (0.5 * profit_norm + 0.5 * downside_norm)
    table["scaling_penalty"] = table["status"].map(weights.penalty).astype(float)
    table["decision_score"] = (w_score * table["score"] + w_risk * table["risk_score"]
                               - table["scaling_penalty"])
    table = table.sort_values(["decision_score", "rank"], ascending=[False, True],
                              kind="mergesort").reset_index(drop=True)
    table.insert(0, "decision_rank", range(1, len(table) + 1))
    return table


def _factor_edges(winner: pd.Series, loser: pd.Series, top_n: int = 2) -> list[str]:
    """Scorecard factors where ``winner`` earned the most extra points over ``loser``."""
    gaps = {f.label: winner[f"pts_{f.key}"] - loser[f"pts_{f.key}"] for f in FACTORS}
    best = sorted(((g, label) for label, g in gaps.items() if g > 0.05), reverse=True)[:top_n]
    return [label.lower() for _, label in best]


def _join(items: list[str]) -> str:
    """Join with commas and a final 'and'."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def describe_trade_off(chosen: pd.Series, other: pd.Series) -> str:
    """Plain-English price-vs-risk comparison of two suppliers.

    Example: "Manufacturer B costs ₹30 more per unit than Manufacturer A, but
    has ₹14,400 less capital at risk if demand is low (MOQ 50 vs 100)."
    """
    name, other_name = chosen["supplier_name"], other["supplier_name"]
    cost_gap = chosen["unit_cost"] - other["unit_cost"]
    risk_gap = other["low_capital_at_risk"] - chosen["low_capital_at_risk"]  # > 0: chosen safer
    moq = f"(MOQ {indian_group(chosen['moq_per_style'])} vs {indian_group(other['moq_per_style'])})"

    if cost_gap > 0:
        if risk_gap > 0:
            text = (f"{name} costs {inr(cost_gap)} more per unit than {other_name}, but has "
                    f"{inr(risk_gap)} less capital at risk if demand is low {moq}.")
        else:
            edges = _factor_edges(chosen, other)
            why = f" — it wins on {_join(edges)}" if edges else ""
            text = (f"{name} costs {inr(cost_gap)} more per unit than {other_name} and does not "
                    f"reduce low-demand risk{why}.")
    else:
        price = (f"is {inr(-cost_gap)} cheaper per unit than {other_name}" if cost_gap < 0
                 else f"has the same unit cost as {other_name}")
        if risk_gap > 0:
            text = f"{name} {price} and has {inr(risk_gap)} less capital at risk if demand is low {moq}."
        elif risk_gap < 0:
            text = (f"{name} {price}, but ties up {inr(-risk_gap)} more capital if demand is low "
                    f"{moq}.")
        else:
            text = f"{name} {price} with the same capital at risk if demand is low."

    profit_gap = chosen["expected_profit"] - other["expected_profit"]
    if abs(profit_gap) >= 1:
        direction = "higher" if profit_gap > 0 else "lower"
        text += (f" Expected profit per style is {inr(abs(profit_gap))} {direction} "
                 f"({inr(chosen['expected_profit'])} vs {inr(other['expected_profit'])}).")
    else:
        text += f" Expected profit per style is about the same ({inr(chosen['expected_profit'])})."
    return text


def _warnings(row: pd.Series, thresholds: WarningThresholds, planned_quantity: int | None,
              low_demand: int | None) -> list[str]:
    """Risks the decision-maker should plan around for the chosen supplier."""
    out: list[str] = []
    failed = [row.get(c) for c in ("capacity_ok", "reorder_ok")]
    if any(pd.notna(v) and not bool(v) for v in failed):
        out.append(f"Scaling ({row['status']}): {row['scaling_explanation']}")
    if row["lead_time_days"] > thresholds.max_lead_time_days:
        out.append(f"Lead time: first production takes {int(row['lead_time_days'])} days, over your "
                   f"{thresholds.max_lead_time_days}-day limit.")
    if row["advance_payment_pct"] >= thresholds.high_advance_pct:
        out.append(f"High advance: {row['advance_payment_pct']:g}% upfront — {inr(row['advance_amount'])} "
                   f"of a {inr(row['order_value'])} order before production starts.")
    if row["reliability_score"] <= thresholds.low_reliability:
        out.append(f"Reliability is only {row['reliability_score']:g}/5 — build buffer into deadlines "
                   "and follow up often.")
    if row["low_profit"] < 0:
        demand = f" ({indian_group(low_demand)} units)" if low_demand is not None else ""
        out.append(f"Loses {inr(-row['low_profit'])} on this style if demand is low{demand}.")
    if planned_quantity is not None and row["moq_per_style"] > planned_quantity:
        extra = int(row["moq_per_style"]) - planned_quantity
        out.append(f"MOQ of {indian_group(row['moq_per_style'])} is above your planned "
                   f"{indian_group(planned_quantity)} units — you must buy {indian_group(extra)} extra "
                   f"units per style.")
    return out


def _runner_up_reason(winner: pd.Series, runner: pd.Series) -> str:
    """Why the second-placed supplier lost."""
    parts: list[str] = []
    score_gap = winner["score"] - runner["score"]
    if score_gap > 0.05:
        edges = _factor_edges(winner, runner)
        main = f", mainly on {_join(edges)}" if edges else ""
        parts.append(f"scored {score_gap:.1f} points lower on the scorecard{main}")
    elif score_gap < -0.05:
        parts.append(f"scored {-score_gap:.1f} points higher on the scorecard but lost on the "
                     "risk model or scaling check")
    if runner["risk_score"] < winner["risk_score"] - 0.05:
        parts.append(f"has a weaker risk profile (expected profit {inr(runner['expected_profit'])} vs "
                     f"{inr(winner['expected_profit'])}; {inr(runner['low_capital_at_risk'])} vs "
                     f"{inr(winner['low_capital_at_risk'])} at risk if demand is low)")
    elif runner["risk_score"] > winner["risk_score"] + 0.05:
        parts.append(f"that outweighed its stronger risk profile (expected profit "
                     f"{inr(runner['expected_profit'])} vs {inr(winner['expected_profit'])}; "
                     f"{inr(runner['low_capital_at_risk'])} vs {inr(winner['low_capital_at_risk'])} "
                     "at risk if demand is low)")
    if runner["scaling_penalty"] > winner["scaling_penalty"]:
        parts.append(f"has a {runner['status']} scaling status: {runner['scaling_explanation']}")
    detail = "; ".join(parts) if parts else "came out marginally behind on the combined score"
    return (f"{runner['supplier_name']} came second (decision score {runner['decision_score']:.1f} vs "
            f"{winner['decision_score']:.1f}). It {detail}.")


def build_recommendation(
    scorecard: ScorecardResult,
    risk_summary: pd.DataFrame,
    scaling: pd.DataFrame,
    weights: DecisionWeights = DecisionWeights(),
    thresholds: WarningThresholds = WarningThresholds(),
    planned_quantity: int | None = None,
    low_demand: int | None = None,
) -> Recommendation:
    """Pick the best supplier and explain the decision.

    The key trade-off compares the recommended supplier with the *cheapest*
    compared supplier (the classic "pay more per unit for a lower MOQ"
    question). If the recommended supplier is already the cheapest, it is
    compared with the runner-up instead.

    Raises:
        ValueError: if there is no eligible supplier to recommend.
    """
    table = build_decision_table(scorecard, risk_summary, scaling, weights)
    if table.empty:
        raise ValueError("No eligible suppliers to recommend. Check the hard filter and the "
                         "suppliers selected for comparison.")
    best = table.iloc[0]
    runner = table.iloc[1] if len(table) > 1 else None

    others = table.iloc[1:]
    cheapest = others.loc[others["unit_cost"].idxmin()] if not others.empty else None
    if cheapest is not None and cheapest["unit_cost"] < best["unit_cost"]:
        trade_off = describe_trade_off(best, cheapest)
    elif runner is not None:
        trade_off = describe_trade_off(best, runner)
    else:
        trade_off = f"{best['supplier_name']} is the only supplier being compared."

    return Recommendation(
        recommended=str(best["supplier_name"]),
        decision_score=float(best["decision_score"]),
        scorecard_score=float(best["score"]),
        scorecard_rank=int(best["rank"]),
        trade_off=trade_off,
        warnings=_warnings(best, thresholds, planned_quantity, low_demand),
        runner_up=None if runner is None else str(runner["supplier_name"]),
        runner_up_reason=None if runner is None else _runner_up_reason(best, runner),
        decision_table=table,
    )


# --- Excel export ----------------------------------------------------------

_RISK_LABELS = {
    "units_bought": "Units bought", "order_value": "Order value (₹)",
    "advance_amount": "Advance (₹)", "breakeven_units": "Break-even units",
    "expected_profit": "Expected profit (₹)", "expected_capital_at_risk": "Expected capital at risk (₹)",
    "low_profit": "Profit if demand low (₹)", "low_capital_at_risk": "Capital at risk if demand low (₹)",
    "low_unsold_units": "Unsold units if demand low", "scenario": "Scenario", "demand": "Demand",
    "probability": "Probability (%)", "units_sold": "Units sold", "revenue": "Revenue (₹)",
    "cost": "Cost (₹)", "profit": "Profit (₹)", "unsold_units": "Unsold units",
    "capital_at_risk": "Capital at risk (₹)",
}
_SCALING_LABELS = {
    "usable_capacity": "Usable capacity", "required_monthly_units": "Required units / month",
    "capacity_utilisation_pct": "Utilisation (%)", "capacity_ok": "Capacity OK",
    "reorder_slack_days": "Reorder slack (days)", "reorder_ok": "Reorder OK",
    "status": "Status", "explanation": "Explanation",
}


def _labelled(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={**DISPLAY_LABELS, **_RISK_LABELS, **_SCALING_LABELS})


def scorecard_export_table(scorecard: ScorecardResult) -> pd.DataFrame:
    """Scorecard with rank, descriptive columns, score and points per factor."""
    ranked = scorecard.ranked
    cols = ["rank", "supplier_name"] + [c for c in OPTIONAL_COLUMNS if c in ranked.columns] + ["score"]
    out = ranked[cols].rename(columns={"rank": "Rank", "score": "Score (/100)"})
    for f in FACTORS:
        out[f"{f.label} pts"] = ranked[f"pts_{f.key}"].round(1)
    out["Score (/100)"] = out["Score (/100)"].round(1)
    return _labelled(out)


def summary_to_excel(
    recommendation: Recommendation,
    scorecard: ScorecardResult,
    risk_results: pd.DataFrame,
    risk_summary: pd.DataFrame,
    scaling: pd.DataFrame,
    assumptions: dict[str, object] | None = None,
) -> bytes:
    """Write a short Excel summary and return the file contents.

    Sheets: Recommendation, Scorecard, Excluded, MOQ Risk, Scenario detail, Scaling.
    """
    rows: list[tuple[str, object]] = [
        ("Recommended supplier", recommendation.recommended),
        ("Decision score (/100)", round(recommendation.decision_score, 1)),
        ("Scorecard score (/100)", round(recommendation.scorecard_score, 1)),
        ("Scorecard rank", recommendation.scorecard_rank),
        ("Key trade-off", recommendation.trade_off),
    ]
    rows += [("Warning", w) for w in recommendation.warnings] or [("Warnings", "None")]
    rows += [("Runner-up", recommendation.runner_up or "—"),
             ("Why the runner-up lost", recommendation.runner_up_reason or "—")]
    if assumptions:
        rows.append(("", ""))
        rows += [(f"Assumption: {k}", v) for k, v in assumptions.items()]
    rec_sheet = pd.DataFrame(rows, columns=["Item", "Detail"])

    decision_cols = ["decision_rank", "supplier_name", "decision_score", "score", "risk_score",
                     "status", "scaling_penalty"]
    decision = recommendation.decision_table[decision_cols].rename(columns={
        "decision_rank": "Decision rank", "decision_score": "Decision score", "score": "Scorecard score",
        "risk_score": "Risk score", "scaling_penalty": "Scaling penalty"}).round(1)

    excluded_cols = ["supplier_name"] + [c for c in OPTIONAL_COLUMNS if c in scorecard.excluded.columns]
    excluded = scorecard.excluded[excluded_cols + ["exclusion_reason"]].rename(
        columns={"exclusion_reason": "Reason"})

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        sheets = {
            "Recommendation": rec_sheet,
            "Decision": _labelled(decision),
            "Scorecard": scorecard_export_table(scorecard),
            "Excluded": _labelled(excluded),
            "MOQ Risk": _labelled(risk_summary.round(0)),
            "Scenario detail": _labelled(risk_results),
            "Scaling": _labelled(scaling.round(1)),
        }
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            for idx, column in enumerate(frame.columns, start=1):
                values = [str(column)] + [str(v) for v in frame[column].head(50)]
                width = min(max(len(v) for v in values) + 2, 90)
                sheet.column_dimensions[sheet.cell(row=1, column=idx).column_letter].width = width
    return buffer.getvalue()
