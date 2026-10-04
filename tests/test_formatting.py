"""Tests for rupee / unit formatting."""

from __future__ import annotations

import pytest

from sourcewise.formatting import indian_group, inr, units


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0"), (999, "999"), (1000, "1,000"), (100000, "1,00,000"),
     (1234567, "12,34,567"), (-21000, "-21,000")],
)
def test_indian_group(value: float, expected: str) -> None:
    assert indian_group(value) == expected


def test_inr_and_units() -> None:
    assert inr(21000) == "₹21,000"
    assert inr(-3030) == "-₹3,030"
    assert inr(12.5, decimals=2) == "₹12.50"
    assert inr(float("nan")) == "—"
    assert units(3200) == "3,200 units"
