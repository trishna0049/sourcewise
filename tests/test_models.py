"""Tests for CSV loading and validation."""

from __future__ import annotations

import io

import pandas as pd
import pytest

from sourcewise.models import (
    OPTIONAL_COLUMNS,
    REQUIRED_COLUMNS,
    Supplier,
    SupplierDataError,
    load_suppliers,
    to_suppliers,
    validate_suppliers,
)
from tests.conftest import SAMPLE_CSV, make_supplier


def _errors(df: pd.DataFrame) -> list[str]:
    with pytest.raises(SupplierDataError) as excinfo:
        validate_suppliers(df)
    return excinfo.value.errors


def test_sample_file_is_accepted() -> None:
    df = load_suppliers(SAMPLE_CSV)
    assert len(df) == 30
    assert set(REQUIRED_COLUMNS) <= set(df.columns)
    assert set(OPTIONAL_COLUMNS) <= set(df.columns)
    assert df["meets_required_capability"].dtype == bool
    assert df["moq_per_style"].dtype == "int64"
    # Rows 1-3 are the real Clozet manufacturers.
    a, b, c = df.iloc[0], df.iloc[1], df.iloc[2]
    assert (a.unit_cost, a.moq_per_style) == (300, 100)
    assert (b.unit_cost, b.moq_per_style) == (330, 50)
    assert c.moq_per_style == 30 and not c.meets_required_capability


def test_optional_columns_may_be_missing() -> None:
    df = validate_suppliers(pd.DataFrame([make_supplier("X"), make_supplier("Y")]))
    assert not set(OPTIONAL_COLUMNS) & set(df.columns)
    assert len(df) == 2


def test_headers_are_normalised() -> None:
    raw = pd.DataFrame([make_supplier("X")])
    raw.columns = [f" {c.upper().replace('_', ' ')} " for c in raw.columns]
    df = validate_suppliers(raw)
    assert "unit_cost" in df.columns


def test_missing_required_column_is_reported() -> None:
    raw = pd.DataFrame([make_supplier("X")]).drop(columns=["unit_cost", "reliability_score"])
    errors = _errors(raw)
    assert len(errors) == 1
    assert "unit_cost" in errors[0] and "reliability_score" in errors[0]


def test_missing_column_and_bad_values_reported_together() -> None:
    raw = pd.DataFrame([make_supplier("X", unit_cost="abc", sample_quality=7)]).drop(
        columns=["costing_transparency"])
    errors = _errors(raw)
    assert len(errors) == 3
    assert "costing_transparency" in errors[0]


def test_empty_file_is_rejected() -> None:
    raw = pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    assert _errors(raw) == ["The file has no supplier rows."]


@pytest.mark.parametrize(
    ("field", "value", "fragment"),
    [
        ("unit_cost", 0, "greater than 0"),
        ("unit_cost", "abc", "must be a number"),
        ("moq_per_style", 0, "at least 1"),
        ("moq_per_style", 12.5, "whole number"),
        ("sample_quality", 6, "at most 5"),
        ("reliability_score", 0, "at least 1"),
        ("advance_payment_pct", 120, "at most 100"),
        ("advance_payment_pct", -5, "at least 0"),
        ("lead_time_days", -1, "at least 0"),
        ("meets_required_capability", "maybe", "TRUE or FALSE"),
    ],
)
def test_out_of_range_values_are_reported(field: str, value: object, fragment: str) -> None:
    raw = pd.DataFrame([make_supplier("Ok"), make_supplier("Bad", **{field: value})])
    errors = _errors(raw)
    assert len(errors) == 1
    assert fragment in errors[0]
    assert "row 3" in errors[0]  # header is row 1, "Ok" is row 2


def test_all_errors_are_collected_at_once() -> None:
    raw = pd.DataFrame([make_supplier("X", unit_cost=-1, sample_quality=9, advance_payment_pct=150)])
    assert len(_errors(raw)) == 3


def test_blank_and_duplicate_names_are_rejected() -> None:
    raw = pd.DataFrame([make_supplier("Same"), make_supplier("Same"), make_supplier("  ")])
    errors = " ".join(_errors(raw))
    assert "blank" in errors and "duplicate" in errors


@pytest.mark.parametrize(("text", "expected"), [("TRUE", True), ("yes", True), ("Y", True),
                                                 ("false", False), ("No", False), ("0", False)])
def test_boolean_spellings(text: str, expected: bool) -> None:
    df = validate_suppliers(pd.DataFrame([make_supplier("X", meets_required_capability=text)]))
    assert bool(df.loc[0, "meets_required_capability"]) is expected


def test_fully_blank_rows_are_ignored() -> None:
    raw = pd.DataFrame([make_supplier("X"), {c: None for c in REQUIRED_COLUMNS}])
    assert len(validate_suppliers(raw)) == 1


def test_unparseable_csv_raises_supplier_error() -> None:
    with pytest.raises(SupplierDataError):
        load_suppliers(io.StringIO(""))


def test_to_suppliers_builds_dataclasses(sample_df: pd.DataFrame) -> None:
    suppliers = to_suppliers(sample_df)
    assert len(suppliers) == 30
    first = suppliers[0]
    assert isinstance(first, Supplier)
    assert first.city == "Surat" and first.moq_per_style == 100
    assert first.to_dict()["supplier_name"] == "Manufacturer A (Surat)"
