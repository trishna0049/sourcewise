"""Tests for the scaling / reorder feasibility check."""

from __future__ import annotations

import pandas as pd
import pytest

from sourcewise.capacity import ScalingInputs, ScalingStatus, check_scaling, scaling_status
from sourcewise.models import validate_suppliers
from tests.conftest import make_supplier


@pytest.mark.parametrize(
    ("capacity_ok", "reorder_ok", "expected"),
    [(True, True, ScalingStatus.GREEN), (True, False, ScalingStatus.AMBER),
     (False, True, ScalingStatus.AMBER), (False, False, ScalingStatus.RED)],
)
def test_status_logic(capacity_ok: bool, reorder_ok: bool, expected: ScalingStatus) -> None:
    assert scaling_status(capacity_ok, reorder_ok) is expected


@pytest.fixture
def four() -> pd.DataFrame:
    return validate_suppliers(pd.DataFrame([
        make_supplier("Red", monthly_capacity=2000, reorder_time_days=30),
        make_supplier("AmberCap", monthly_capacity=2000, reorder_time_days=10),
        make_supplier("Green", monthly_capacity=5000, reorder_time_days=15),
        make_supplier("AmberReorder", monthly_capacity=5000, reorder_time_days=25),
    ]))


def test_check_scaling_statuses_and_order(four: pd.DataFrame) -> None:
    # Need 800 × 4 = 3,200 units/month and reorder within 20 days.
    result = check_scaling(four, ScalingInputs(800, 4.0, 20)).set_index("supplier_name")
    assert result.loc["Green", "status"] == "Green"
    assert result.loc["AmberCap", "status"] == "Amber"
    assert result.loc["AmberReorder", "status"] == "Amber"
    assert result.loc["Red", "status"] == "Red"
    assert result.loc["Green", "capacity_utilisation_pct"] == pytest.approx(64.0)
    assert result.loc["AmberReorder", "reorder_slack_days"] == -5
    ordered = check_scaling(four, ScalingInputs(800, 4.0, 20))["status"].tolist()
    assert ordered == ["Green", "Amber", "Amber", "Red"]


def test_boundaries_count_as_pass() -> None:
    df = validate_suppliers(pd.DataFrame([make_supplier("Edge", monthly_capacity=3200,
                                                         reorder_time_days=20)]))
    row = check_scaling(df, ScalingInputs(800, 4.0, 20)).iloc[0]
    assert row["capacity_ok"] and row["reorder_ok"] and row["status"] == "Green"


def test_capacity_share_reduces_usable_capacity(four: pd.DataFrame) -> None:
    # Only half of a 5,000 capacity is usable -> 2,500 < 3,200.
    result = check_scaling(four, ScalingInputs(800, 4.0, 20, max_capacity_share=0.5))
    green = result.set_index("supplier_name").loc["Green"]
    assert green["usable_capacity"] == 2500 and not green["capacity_ok"]
    assert green["status"] == "Amber"


def test_explanations_are_plain_english(four: pd.DataFrame) -> None:
    result = check_scaling(four, ScalingInputs(800, 4.0, 20)).set_index("supplier_name")
    assert result.loc["Green", "explanation"] == ("Capacity OK (5,000 vs 3,200 needed); "
                                                    "reorder OK (15 vs 20 days).")
    assert "Capacity short by 1,200 units/month" in result.loc["Red", "explanation"]
    assert ", but reorder 5 days too slow (25 vs 20 days)" in result.loc["AmberReorder", "explanation"]


@pytest.mark.parametrize("inputs", [ScalingInputs(-1, 2, 20), ScalingInputs(100, 0, 20),
                                    ScalingInputs(100, 2, -1), ScalingInputs(100, 2, 20, 0)])
def test_invalid_inputs_rejected(four: pd.DataFrame, inputs: ScalingInputs) -> None:
    with pytest.raises(ValueError):
        check_scaling(four, inputs)


def test_clozet_a_vs_b(clozet_df: pd.DataFrame) -> None:
    result = check_scaling(clozet_df, ScalingInputs(800, 4.0, 20)).set_index("supplier_name")
    assert result.loc["Manufacturer B", "status"] == "Green"   # 6,000 cap, 15-day reorder
    assert result.loc["Manufacturer A", "status"] == "Amber"   # 25-day reorder is too slow
