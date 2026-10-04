"""Shared fixtures for the SourceWise test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sourcewise.models import load_suppliers, validate_suppliers

SAMPLE_CSV = Path(__file__).resolve().parents[1] / "data" / "sample_suppliers.csv"


def make_supplier(name: str, **overrides: object) -> dict[str, object]:
    """Return one valid supplier row; keyword arguments override fields."""
    row: dict[str, object] = {
        "supplier_name": name,
        "unit_cost": 300,
        "moq_per_style": 100,
        "sample_quality": 4,
        "capability_score": 4,
        "meets_required_capability": True,
        "lead_time_days": 30,
        "reorder_time_days": 20,
        "monthly_capacity": 5000,
        "reliability_score": 4,
        "costing_transparency": 4,
        "advance_payment_pct": 40,
    }
    row.update(overrides)
    return row


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """The provided 30-supplier sample file, validated."""
    return load_suppliers(SAMPLE_CSV)


@pytest.fixture
def clozet_df() -> pd.DataFrame:
    """The three real Clozet manufacturers (A, B, C) from the sample file."""
    return validate_suppliers(
        pd.DataFrame(
            [
                make_supplier("Manufacturer A", unit_cost=300, moq_per_style=100, sample_quality=4,
                              capability_score=4, lead_time_days=35, reorder_time_days=25,
                              monthly_capacity=8000, reliability_score=3, costing_transparency=3,
                              advance_payment_pct=50),
                make_supplier("Manufacturer B", unit_cost=330, moq_per_style=50, sample_quality=5,
                              capability_score=5, lead_time_days=25, reorder_time_days=15,
                              monthly_capacity=6000, reliability_score=5, costing_transparency=5,
                              advance_payment_pct=40),
                make_supplier("Manufacturer C", unit_cost=280, moq_per_style=30, sample_quality=3,
                              capability_score=2, meets_required_capability=False, lead_time_days=20,
                              reorder_time_days=12, monthly_capacity=3000, reliability_score=3,
                              costing_transparency=2, advance_payment_pct=50),
            ]
        )
    )
