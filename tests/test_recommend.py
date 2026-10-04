"""Tests for the combined recommendation, trade-off text and Excel export.

Clozet example with default scorecard weights (sum 45) and the default risk
inputs (₹899, planned 50, demand 30/60/120 at 30/50/20%):

    Scorecard: between A and B, A only wins on cost (weight 7/45)
        A = 15.56, B = 84.44
    Risk score: B has higher expected profit (23,056 vs 23,041) and lower
        low-demand capital at risk (6,600 vs 21,000) -> B = 100, A = 0
    Scaling (800 units × 4, 20-day reorder): A Amber (25-day reorder), B Green
    Decision = 0.6 × scorecard + 0.4 × risk − penalty
        B = 0.6 × 84.44 + 40 = 90.67
        A = 0.6 × 15.56 + 0 − 10 = −0.67
"""

from __future__ import annotations

import io

import pandas as pd
import pytest

from sourcewise.capacity import ScalingInputs, check_scaling
from sourcewise.models import validate_suppliers
from sourcewise.moq_risk import DEFAULT_SCENARIOS, run_risk_model, summarize_risk
from sourcewise.recommend import (
    DecisionWeights,
    build_decision_table,
    build_recommendation,
    describe_trade_off,
    summary_to_excel,
)
from sourcewise.scoring import ScorecardResult, score_suppliers
from tests.conftest import make_supplier

PRICE, PLANNED = 899.0, 50


def _pipeline(df: pd.DataFrame, weights: dict[str, float] | None = None,
              compare: list[str] | None = None) -> tuple[ScorecardResult, pd.DataFrame, pd.DataFrame,
                                                         pd.DataFrame]:
    scorecard = score_suppliers(df, weights)
    chosen = df if compare is None else df[df["supplier_name"].isin(compare)]
    results = run_risk_model(chosen, PRICE, PLANNED, DEFAULT_SCENARIOS)
    summary = summarize_risk(results, chosen, DEFAULT_SCENARIOS, PRICE)
    scaling = check_scaling(scorecard.ranked, ScalingInputs(800, 4.0, 20))
    return scorecard, results, summary, scaling


def test_clozet_recommends_b_over_cheaper_a(clozet_df: pd.DataFrame) -> None:
    scorecard, _, summary, scaling = _pipeline(clozet_df)
    rec = build_recommendation(scorecard, summary, scaling, planned_quantity=PLANNED, low_demand=30)

    assert rec.recommended == "Manufacturer B"
    assert rec.runner_up == "Manufacturer A"
    assert rec.scorecard_rank == 1
    assert rec.scorecard_score == pytest.approx(100 - 7 / 45 * 100)
    assert rec.decision_score == pytest.approx(0.6 * (100 - 7 / 45 * 100) + 40)
    a = rec.decision_table.set_index("supplier_name").loc["Manufacturer A"]
    assert a["decision_score"] == pytest.approx(0.6 * 7 / 45 * 100 - 10)

    assert "costs ₹30 more per unit than Manufacturer A" in rec.trade_off
    assert "₹14,400 less capital at risk if demand is low (MOQ 50 vs 100)" in rec.trade_off
    assert "₹15 higher" in rec.trade_off
    assert rec.warnings == []
    assert "lower on the scorecard" in rec.runner_up_reason
    assert "Amber" in rec.runner_up_reason


def test_excluded_supplier_is_never_recommended(clozet_df: pd.DataFrame) -> None:
    # C is cheapest with the lowest MOQ but fails the capability requirement.
    scorecard, _, summary, scaling = _pipeline(clozet_df, {"cost": 1, "moq": 1})
    assert "Manufacturer C" in summary["supplier_name"].tolist()
    rec = build_recommendation(scorecard, summary, scaling)
    assert "Manufacturer C" not in rec.decision_table["supplier_name"].tolist()
    assert rec.recommended != "Manufacturer C"


def test_cost_focused_weights_pick_a_with_warnings(clozet_df: pd.DataFrame) -> None:
    # Scorecard on cost only: A = 100, B = 0. Decision: A = 60 − 10 = 50, B = 0 + 40 = 40.
    scorecard, _, summary, scaling = _pipeline(clozet_df, {"cost": 1})
    rec = build_recommendation(scorecard, summary, scaling, planned_quantity=PLANNED, low_demand=30)
    assert rec.recommended == "Manufacturer A"
    assert rec.decision_score == pytest.approx(50.0)
    assert "is ₹30 cheaper per unit than Manufacturer B" in rec.trade_off
    assert "ties up ₹14,400 more capital if demand is low" in rec.trade_off

    text = " | ".join(rec.warnings)
    assert "Scaling (Amber)" in text
    assert "35 days" in text                      # lead time over 30
    assert "50% upfront — ₹15,000" in text        # high advance
    assert "Loses ₹3,030" in text                 # negative low-demand profit
    assert "buy 50 extra units" in text           # MOQ 100 vs planned 50
    assert "higher on the scorecard" not in rec.runner_up_reason
    assert "lower on the scorecard" in rec.runner_up_reason


def test_risk_only_weights_and_penalties(clozet_df: pd.DataFrame) -> None:
    scorecard, _, summary, scaling = _pipeline(clozet_df, {"cost": 1})
    table = build_decision_table(scorecard, summary, scaling,
                                 DecisionWeights(scorecard=0, risk=1, amber_penalty=0, red_penalty=0))
    assert table["supplier_name"].tolist() == ["Manufacturer B", "Manufacturer A"]
    assert table["decision_score"].tolist() == pytest.approx([100.0, 0.0])
    assert table["decision_rank"].tolist() == [1, 2]


def test_ties_fall_back_to_scorecard_rank() -> None:
    df = validate_suppliers(pd.DataFrame([make_supplier("Alpha"), make_supplier("Beta")]))
    scorecard, _, summary, scaling = _pipeline(df)
    table = build_decision_table(scorecard, summary, scaling)
    assert table["decision_score"].nunique() == 1
    assert table["supplier_name"].tolist() == scorecard.ranked["supplier_name"].tolist()


def test_single_candidate_has_no_runner_up(clozet_df: pd.DataFrame) -> None:
    scorecard, _, summary, scaling = _pipeline(clozet_df, compare=["Manufacturer B"])
    rec = build_recommendation(scorecard, summary, scaling)
    assert rec.recommended == "Manufacturer B"
    assert rec.runner_up is None and rec.runner_up_reason is None
    assert "only supplier" in rec.trade_off


def test_no_candidates_raises(clozet_df: pd.DataFrame) -> None:
    scorecard, _, summary, scaling = _pipeline(clozet_df, compare=["Manufacturer C"])
    with pytest.raises(ValueError, match="No eligible suppliers"):
        build_recommendation(scorecard, summary, scaling)


def test_invalid_decision_weights() -> None:
    with pytest.raises(ValueError):
        DecisionWeights(scorecard=0, risk=0).normalised()


def test_describe_trade_off_cheaper_and_safer() -> None:
    chosen = pd.Series({"supplier_name": "X", "unit_cost": 300, "moq_per_style": 50,
                        "low_capital_at_risk": 6000, "expected_profit": 20000})
    other = pd.Series({"supplier_name": "Y", "unit_cost": 320, "moq_per_style": 100,
                       "low_capital_at_risk": 22400, "expected_profit": 20000})
    text = describe_trade_off(chosen, other)
    assert text.startswith("X is ₹20 cheaper per unit than Y and has ₹16,400 less capital at risk")
    assert "about the same" in text


def test_sample_data_end_to_end(sample_df: pd.DataFrame) -> None:
    scorecard = score_suppliers(sample_df)
    top5 = scorecard.ranked.head(5)
    results = run_risk_model(top5, PRICE, PLANNED, DEFAULT_SCENARIOS)
    summary = summarize_risk(results, top5, DEFAULT_SCENARIOS, PRICE)
    scaling = check_scaling(scorecard.ranked, ScalingInputs(800, 4.0, 20))
    rec = build_recommendation(scorecard, summary, scaling, planned_quantity=PLANNED, low_demand=30)
    assert rec.recommended in top5["supplier_name"].tolist()
    assert len(rec.decision_table) == 5
    assert rec.decision_table["decision_score"].is_monotonic_decreasing
    assert rec.runner_up is not None


def test_excel_export_contains_all_sheets(clozet_df: pd.DataFrame) -> None:
    scorecard, results, summary, scaling = _pipeline(clozet_df)
    rec = build_recommendation(scorecard, summary, scaling)
    data = summary_to_excel(rec, scorecard, results, summary, scaling,
                            assumptions={"Selling price (₹)": PRICE})
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None)
    assert list(sheets) == ["Recommendation", "Decision", "Scorecard", "Excluded", "MOQ Risk",
                            "Scenario detail", "Scaling"]
    rec_sheet = sheets["Recommendation"]
    assert rec_sheet.loc[0, "Detail"] == "Manufacturer B"
    assert "Assumption: Selling price (₹)" in rec_sheet["Item"].tolist()
    assert sheets["Excluded"]["Supplier"].tolist() == ["Manufacturer C"]
    assert len(sheets["Scenario detail"]) == 9
