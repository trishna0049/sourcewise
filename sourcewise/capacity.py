"""Scaling and reorder feasibility check.

Answers two questions per supplier if a style takes off:

1. **Capacity** — can ``monthly_capacity`` (times the share of it you can
   realistically book) cover ``current_monthly_units × growth_multiplier``?
2. **Reorder speed** — is ``reorder_time_days`` within the number of days
   you can wait for a restock?

Status: Green = both OK, Amber = one fails, Red = both fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from sourcewise.formatting import indian_group


class ScalingStatus(str, Enum):
    """Traffic-light result of the scaling check."""

    GREEN = "Green"
    AMBER = "Amber"
    RED = "Red"


STATUS_ORDER: dict[str, int] = {ScalingStatus.GREEN.value: 0, ScalingStatus.AMBER.value: 1,
                                ScalingStatus.RED.value: 2}

DEFAULT_CURRENT_MONTHLY_UNITS = 800
DEFAULT_GROWTH_MULTIPLIER = 4.0
DEFAULT_REORDER_DEADLINE_DAYS = 20
DEFAULT_MAX_CAPACITY_SHARE = 1.0


@dataclass(frozen=True)
class ScalingInputs:
    """Assumptions for the scaling check.

    Attributes:
        current_monthly_units: Units you order per month today (all styles).
        growth_multiplier: Expected demand growth, e.g. 3.0 for 3×.
        reorder_deadline_days: Longest acceptable wait for a restock.
        max_capacity_share: Fraction of a supplier's capacity you can count on
            (1.0 = all of it; 0.5 if you expect to get only half their lines).
    """

    current_monthly_units: int = DEFAULT_CURRENT_MONTHLY_UNITS
    growth_multiplier: float = DEFAULT_GROWTH_MULTIPLIER
    reorder_deadline_days: int = DEFAULT_REORDER_DEADLINE_DAYS
    max_capacity_share: float = DEFAULT_MAX_CAPACITY_SHARE

    @property
    def required_monthly_units(self) -> float:
        """Monthly volume the supplier must handle after growth."""
        return self.current_monthly_units * self.growth_multiplier

    def validate(self) -> None:
        """Raise ValueError for impossible inputs."""
        if self.current_monthly_units < 0:
            raise ValueError("Current monthly volume cannot be negative.")
        if self.growth_multiplier <= 0:
            raise ValueError("Growth multiplier must be greater than zero.")
        if self.reorder_deadline_days < 0:
            raise ValueError("Reorder deadline cannot be negative.")
        if not 0 < self.max_capacity_share <= 1:
            raise ValueError("Usable capacity share must be between 0% and 100%.")


def scaling_status(capacity_ok: bool, reorder_ok: bool) -> ScalingStatus:
    """Map the two checks to a traffic light: both OK → Green, one → Amber, none → Red."""
    passed = int(capacity_ok) + int(reorder_ok)
    return {2: ScalingStatus.GREEN, 1: ScalingStatus.AMBER, 0: ScalingStatus.RED}[passed]


def _explain(capacity_ok: bool, reorder_ok: bool, usable: float, required: float,
             reorder_days: int, deadline: int) -> str:
    """Short plain-English explanation, e.g. 'Capacity OK (6,000 vs 3,200 needed); reorder OK (15 vs 20 days).'"""
    cap = (f"Capacity OK ({indian_group(usable)} vs {indian_group(required)} needed)"
           if capacity_ok else
           f"Capacity short by {indian_group(required - usable)} units/month "
           f"({indian_group(usable)} vs {indian_group(required)} needed)")
    reo = (f"reorder OK ({reorder_days} vs {deadline} days)"
           if reorder_ok else
           f"reorder {reorder_days - deadline} days too slow ({reorder_days} vs {deadline} days)")
    joiner = "; " if capacity_ok == reorder_ok else ", but "
    return f"{cap}{joiner}{reo}."


def check_scaling(suppliers: pd.DataFrame, inputs: ScalingInputs = ScalingInputs()) -> pd.DataFrame:
    """Run the capacity and reorder checks for every supplier.

    Returns:
        One row per supplier with ``usable_capacity``, ``required_monthly_units``,
        ``capacity_utilisation_pct``, ``capacity_ok``, ``reorder_time_days``,
        ``reorder_slack_days`` (negative = late), ``reorder_ok``, ``status``
        and ``explanation``, ordered Green → Amber → Red.
    """
    inputs.validate()
    required = inputs.required_monthly_units
    rows = []
    for _, sup in suppliers.iterrows():
        usable = float(sup["monthly_capacity"]) * inputs.max_capacity_share
        capacity_ok = usable >= required
        reorder_days = int(sup["reorder_time_days"])
        reorder_ok = reorder_days <= inputs.reorder_deadline_days
        status = scaling_status(capacity_ok, reorder_ok)
        rows.append({
            "supplier_name": sup["supplier_name"],
            "monthly_capacity": int(sup["monthly_capacity"]),
            "usable_capacity": usable,
            "required_monthly_units": required,
            "capacity_utilisation_pct": (required / usable * 100) if usable > 0 else float("inf"),
            "capacity_ok": capacity_ok,
            "reorder_time_days": reorder_days,
            "reorder_slack_days": inputs.reorder_deadline_days - reorder_days,
            "reorder_ok": reorder_ok,
            "status": status.value,
            "explanation": _explain(capacity_ok, reorder_ok, usable, required, reorder_days,
                                    inputs.reorder_deadline_days),
        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["_order"] = result["status"].map(STATUS_ORDER)
    return (result.sort_values("_order", kind="mergesort").drop(columns="_order")
            .reset_index(drop=True))
