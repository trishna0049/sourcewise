"""MOQ vs unit-cost risk model.

A cheaper supplier with a high minimum order quantity (MOQ) forces you to buy
more units than you may be able to sell. This module runs each supplier
through demand scenarios and reports profit and *capital at risk* — money tied
up in unsold stock.

For each supplier × scenario::

    units_bought    = max(moq_per_style, planned_quantity)
    units_sold      = min(units_bought, demand)
    revenue         = units_sold × selling_price
    cost            = units_bought × unit_cost
    profit          = revenue − cost
    unsold_units    = units_bought − units_sold
    capital_at_risk = unsold_units × unit_cost

Expected values weight each scenario by its probability.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import pandas as pd

#: Tolerance when checking that probabilities add up to 100%.
PROBABILITY_TOLERANCE = 0.01


@dataclass(frozen=True)
class Scenario:
    """A demand scenario for one style.

    Attributes:
        name: Label such as "Low", "Expected", "High".
        demand: Units customers would buy in this scenario.
        probability: Likelihood in percent (0–100).
    """

    name: str
    demand: int
    probability: float


DEFAULT_SELLING_PRICE = 899.0
DEFAULT_PLANNED_QUANTITY = 50
DEFAULT_SCENARIOS: tuple[Scenario, ...] = (
    Scenario("Low", 30, 30.0),
    Scenario("Expected", 60, 50.0),
    Scenario("High", 120, 20.0),
)


@dataclass(frozen=True)
class ScenarioOutcome:
    """Result of buying from one supplier under one demand scenario."""

    units_bought: int
    units_sold: int
    revenue: float
    cost: float
    profit: float
    unsold_units: int
    capital_at_risk: float


def validate_scenarios(scenarios: list[Scenario] | tuple[Scenario, ...]) -> None:
    """Check scenarios are usable.

    Raises:
        ValueError: if there are no scenarios, names repeat, a demand or
            probability is negative, or probabilities do not sum to 100%.
    """
    if not scenarios:
        raise ValueError("Add at least one demand scenario.")
    names = [s.name for s in scenarios]
    if len(set(names)) != len(names):
        raise ValueError("Scenario names must be unique.")
    for s in scenarios:
        if s.demand < 0:
            raise ValueError(f"Demand for '{s.name}' cannot be negative.")
        if s.probability < 0:
            raise ValueError(f"Probability for '{s.name}' cannot be negative.")
    total = sum(s.probability for s in scenarios)
    if abs(total - 100.0) > PROBABILITY_TOLERANCE:
        raise ValueError(f"Scenario probabilities must add up to 100% (currently {total:g}%).")


def scenario_outcome(
    unit_cost: float, moq: int, planned_quantity: int, demand: int, selling_price: float
) -> ScenarioOutcome:
    """Compute profit and capital at risk for one supplier in one scenario."""
    units_bought = max(int(moq), int(planned_quantity))
    units_sold = min(units_bought, int(demand))
    revenue = units_sold * selling_price
    cost = units_bought * unit_cost
    unsold = units_bought - units_sold
    return ScenarioOutcome(
        units_bought=units_bought,
        units_sold=units_sold,
        revenue=revenue,
        cost=cost,
        profit=revenue - cost,
        unsold_units=unsold,
        capital_at_risk=unsold * unit_cost,
    )


def low_scenario(scenarios: list[Scenario] | tuple[Scenario, ...]) -> Scenario:
    """Return the scenario with the lowest demand (the downside case)."""
    return min(scenarios, key=lambda s: s.demand)


def _check_inputs(selling_price: float, planned_quantity: int) -> None:
    if selling_price <= 0:
        raise ValueError("Selling price must be greater than zero.")
    if planned_quantity < 1:
        raise ValueError("Planned order quantity must be at least 1 unit.")


def run_risk_model(
    suppliers: pd.DataFrame,
    selling_price: float,
    planned_quantity: int,
    scenarios: list[Scenario] | tuple[Scenario, ...] = DEFAULT_SCENARIOS,
) -> pd.DataFrame:
    """Evaluate every supplier under every scenario.

    Args:
        suppliers: Validated supplier rows to compare.
        selling_price: Retail price per unit (₹).
        planned_quantity: Units you would order if there were no MOQ.
        scenarios: Demand scenarios with probabilities summing to 100%.

    Returns:
        Long-format DataFrame, one row per supplier × scenario, with the
        columns of :class:`ScenarioOutcome` plus supplier, scenario, demand,
        probability, unit_cost and moq_per_style.
    """
    _check_inputs(selling_price, planned_quantity)
    validate_scenarios(scenarios)
    rows = []
    for _, sup in suppliers.iterrows():
        for sc in scenarios:
            outcome = scenario_outcome(sup["unit_cost"], sup["moq_per_style"], planned_quantity,
                                       sc.demand, selling_price)
            rows.append({
                "supplier_name": sup["supplier_name"],
                "scenario": sc.name,
                "demand": sc.demand,
                "probability": sc.probability,
                "unit_cost": float(sup["unit_cost"]),
                "moq_per_style": int(sup["moq_per_style"]),
                **asdict(outcome),
            })
    return pd.DataFrame(rows)


def summarize_risk(
    results: pd.DataFrame,
    suppliers: pd.DataFrame,
    scenarios: list[Scenario] | tuple[Scenario, ...] = DEFAULT_SCENARIOS,
    selling_price: float = DEFAULT_SELLING_PRICE,
) -> pd.DataFrame:
    """One row per supplier with expected and downside figures.

    Columns:
        order_value: Cash needed for the order (units_bought × unit_cost).
        advance_amount: Upfront payment (order_value × advance_payment_pct).
        breakeven_units: Units you must sell to recover the order cost.
        expected_profit / expected_capital_at_risk: probability-weighted.
        low_profit / low_capital_at_risk / low_unsold_units: the lowest-demand scenario.
        profit_<scenario>: profit in each scenario.
    """
    weighted = results.assign(
        w_profit=results["profit"] * results["probability"] / 100,
        w_car=results["capital_at_risk"] * results["probability"] / 100,
    )
    summary = weighted.groupby("supplier_name", sort=False).agg(
        unit_cost=("unit_cost", "first"),
        moq_per_style=("moq_per_style", "first"),
        units_bought=("units_bought", "first"),
        expected_profit=("w_profit", "sum"),
        expected_capital_at_risk=("w_car", "sum"),
    )
    downside = results[results["scenario"] == low_scenario(scenarios).name].set_index("supplier_name")
    summary["low_profit"] = downside["profit"]
    summary["low_capital_at_risk"] = downside["capital_at_risk"]
    summary["low_unsold_units"] = downside["unsold_units"]
    for sc in scenarios:
        part = results[results["scenario"] == sc.name].set_index("supplier_name")
        summary[f"profit_{sc.name}"] = part["profit"]

    summary["order_value"] = summary["units_bought"] * summary["unit_cost"]
    advance = suppliers.set_index("supplier_name")["advance_payment_pct"]
    summary["advance_amount"] = summary["order_value"] * advance.reindex(summary.index).fillna(0) / 100
    summary["breakeven_units"] = (summary["order_value"] / selling_price).map(math.ceil).astype(int)
    return summary.reset_index()


def cheaper_but_riskier(summary: pd.DataFrame) -> pd.DataFrame:
    """Find pairs where the cheaper supplier carries *more* low-demand risk.

    A pair (cheaper, dearer) is reported when ``cheaper.unit_cost <
    dearer.unit_cost`` but ``cheaper.low_capital_at_risk >
    dearer.low_capital_at_risk`` — usually because of a higher MOQ.

    Returns:
        DataFrame with cheaper_supplier, dearer_supplier, unit_cost_saving,
        extra_capital_at_risk, cheaper_moq, dearer_moq — largest extra risk first.
    """
    columns = ["cheaper_supplier", "dearer_supplier", "unit_cost_saving",
               "extra_capital_at_risk", "cheaper_moq", "dearer_moq"]
    rows = []
    records = summary.to_dict("records")
    for cheap in records:
        for dear in records:
            if (cheap["unit_cost"] < dear["unit_cost"]
                    and cheap["low_capital_at_risk"] > dear["low_capital_at_risk"]):
                rows.append({
                    "cheaper_supplier": cheap["supplier_name"],
                    "dearer_supplier": dear["supplier_name"],
                    "unit_cost_saving": dear["unit_cost"] - cheap["unit_cost"],
                    "extra_capital_at_risk": cheap["low_capital_at_risk"] - dear["low_capital_at_risk"],
                    "cheaper_moq": cheap["moq_per_style"],
                    "dearer_moq": dear["moq_per_style"],
                })
    if not rows:
        return pd.DataFrame(columns=columns)
    return (pd.DataFrame(rows, columns=columns)
            .sort_values("extra_capital_at_risk", ascending=False, kind="mergesort")
            .reset_index(drop=True))
