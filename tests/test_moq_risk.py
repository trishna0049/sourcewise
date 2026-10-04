"""Tests for the MOQ vs unit-cost risk model, using hand-calculated examples.

Worked example (selling price ₹899, planned 50 units, demand 30 / 60 / 120):

Manufacturer A — ₹300, MOQ 100 -> buys 100 units, cost ₹30,000
    Low (30):       revenue 26,970  profit  -3,030  unsold 70  at risk 21,000
    Expected (60):  revenue 53,940  profit  23,940  unsold 40  at risk 12,000
    High (120):     revenue 89,900  profit  59,900  unsold  0  at risk      0
    Expected profit = 0.3(-3,030) + 0.5(23,940) + 0.2(59,900) = 23,041

Manufacturer B — ₹330, MOQ 50 -> buys 50 units, cost ₹16,500
    Low (30):       revenue 26,970  profit  10,470  unsold 20  at risk  6,600
    Expected (60):  revenue 44,950  profit  28,450  unsold  0  at risk      0
    High (120):     revenue 44,950  profit  28,450  unsold  0  at risk      0
    Expected profit = 0.3(10,470) + 0.5(28,450) + 0.2(28,450) = 23,056
"""

from __future__ import annotations

import pandas as pd
import pytest

from sourcewise.moq_risk import (
    DEFAULT_SCENARIOS,
    Scenario,
    cheaper_but_riskier,
    low_scenario,
    run_risk_model,
    scenario_outcome,
    summarize_risk,
    validate_scenarios,
)
from sourcewise.scoring import apply_hard_filters

PRICE, PLANNED = 899.0, 50


@pytest.fixture
def ab(clozet_df: pd.DataFrame) -> pd.DataFrame:
    eligible, _ = apply_hard_filters(clozet_df)
    return eligible


def test_scenario_outcome_moq_forces_overbuying() -> None:
    out = scenario_outcome(unit_cost=300, moq=100, planned_quantity=50, demand=30, selling_price=PRICE)
    assert out.units_bought == 100
    assert out.units_sold == 30
    assert out.revenue == 26_970
    assert out.cost == 30_000
    assert out.profit == -3_030
    assert out.unsold_units == 70
    assert out.capital_at_risk == 21_000


def test_scenario_outcome_planned_above_moq_and_demand_above_stock() -> None:
    out = scenario_outcome(unit_cost=330, moq=50, planned_quantity=80, demand=200, selling_price=PRICE)
    assert out.units_bought == 80 and out.units_sold == 80
    assert out.profit == 80 * 899 - 80 * 330
    assert out.unsold_units == 0 and out.capital_at_risk == 0


def test_zero_demand_everything_at_risk() -> None:
    out = scenario_outcome(unit_cost=330, moq=50, planned_quantity=10, demand=0, selling_price=PRICE)
    assert out.revenue == 0 and out.profit == -16_500 and out.capital_at_risk == 16_500


def test_run_risk_model_long_format(ab: pd.DataFrame) -> None:
    results = run_risk_model(ab, PRICE, PLANNED, DEFAULT_SCENARIOS)
    assert len(results) == 2 * 3
    a = results[results["supplier_name"] == "Manufacturer A"].set_index("scenario")
    assert a.loc["Low", "profit"] == -3_030
    assert a.loc["Expected", "profit"] == 23_940
    assert a.loc["High", "profit"] == 59_900
    assert a.loc["Expected", "capital_at_risk"] == 12_000


def test_summary_expected_values(ab: pd.DataFrame) -> None:
    results = run_risk_model(ab, PRICE, PLANNED, DEFAULT_SCENARIOS)
    summary = summarize_risk(results, ab, DEFAULT_SCENARIOS, PRICE).set_index("supplier_name")
    a, b = summary.loc["Manufacturer A"], summary.loc["Manufacturer B"]

    assert a["expected_profit"] == pytest.approx(23_041)
    assert b["expected_profit"] == pytest.approx(23_056)
    assert a["expected_capital_at_risk"] == pytest.approx(0.3 * 21_000 + 0.5 * 12_000)
    assert b["expected_capital_at_risk"] == pytest.approx(0.3 * 6_600)

    assert a["low_profit"] == -3_030 and b["low_profit"] == 10_470
    assert a["low_capital_at_risk"] == 21_000 and b["low_capital_at_risk"] == 6_600
    assert a["low_unsold_units"] == 70 and b["low_unsold_units"] == 20

    assert a["order_value"] == 30_000 and b["order_value"] == 16_500
    assert a["advance_amount"] == 15_000 and b["advance_amount"] == 6_600  # 50% and 40%
    assert a["breakeven_units"] == 34 and b["breakeven_units"] == 19      # ceil(cost / 899)
    assert a["profit_High"] == 59_900


def test_cheaper_supplier_flagged_as_riskier(ab: pd.DataFrame) -> None:
    results = run_risk_model(ab, PRICE, PLANNED, DEFAULT_SCENARIOS)
    pairs = cheaper_but_riskier(summarize_risk(results, ab, DEFAULT_SCENARIOS, PRICE))
    assert len(pairs) == 1
    row = pairs.iloc[0]
    assert row["cheaper_supplier"] == "Manufacturer A"
    assert row["dearer_supplier"] == "Manufacturer B"
    assert row["unit_cost_saving"] == 30
    assert row["extra_capital_at_risk"] == 14_400


def test_no_flag_when_cheaper_supplier_is_also_safer(ab: pd.DataFrame) -> None:
    # Planned 100 >= both MOQs: both buy 100 units, so the cheaper one risks less.
    results = run_risk_model(ab, PRICE, 100, DEFAULT_SCENARIOS)
    assert cheaper_but_riskier(summarize_risk(results, ab, DEFAULT_SCENARIOS, PRICE)).empty


def test_low_scenario_is_lowest_demand() -> None:
    scenarios = [Scenario("Boom", 500, 10), Scenario("Bust", 5, 40), Scenario("Base", 50, 50)]
    assert low_scenario(scenarios).name == "Bust"


@pytest.mark.parametrize(
    "scenarios",
    [
        [Scenario("Low", 30, 30), Scenario("Expected", 60, 50), Scenario("High", 120, 30)],  # 110%
        [Scenario("Low", 30, 50), Scenario("Low", 60, 50)],                                   # dup name
        [Scenario("Low", -1, 100)],                                                           # negative
        [],
    ],
)
def test_invalid_scenarios_rejected(scenarios: list[Scenario]) -> None:
    with pytest.raises(ValueError):
        validate_scenarios(scenarios)


def test_probabilities_summing_to_100_accepted() -> None:
    validate_scenarios([Scenario("Low", 30, 33.33), Scenario("Expected", 60, 33.33),
                        Scenario("High", 120, 33.34)])


@pytest.mark.parametrize(("price", "qty"), [(0, 50), (-10, 50), (899, 0)])
def test_invalid_price_or_quantity_rejected(ab: pd.DataFrame, price: float, qty: int) -> None:
    with pytest.raises(ValueError):
        run_risk_model(ab, price, qty, DEFAULT_SCENARIOS)
