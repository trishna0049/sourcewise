"""Tests for normalisation, hard filters and the weighted scorecard."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from sourcewise.models import validate_suppliers
from sourcewise.scoring import (
    DEFAULT_WEIGHTS,
    EXCLUSION_REASON,
    FACTORS,
    apply_hard_filters,
    min_max_normalize,
    normalize_weights,
    score_suppliers,
)
from tests.conftest import make_supplier


# --- min-max normalisation -------------------------------------------------

def test_normalize_higher_is_better() -> None:
    result = min_max_normalize(pd.Series([1, 3, 5]))
    assert result.tolist() == [0.0, 0.5, 1.0]


def test_normalize_lower_is_better_is_inverted() -> None:
    result = min_max_normalize(pd.Series([280, 300, 330]), higher_is_better=False)
    assert result.tolist() == pytest.approx([1.0, 0.6, 0.0])


def test_normalize_all_equal_values_no_divide_by_zero() -> None:
    for direction in (True, False):
        result = min_max_normalize(pd.Series([4, 4, 4]), higher_is_better=direction)
        assert result.tolist() == [1.0, 1.0, 1.0]


def test_normalize_single_value() -> None:
    assert min_max_normalize(pd.Series([42.0])).tolist() == [1.0]


def test_normalize_preserves_index() -> None:
    series = pd.Series([10, 20], index=[7, 9])
    assert list(min_max_normalize(series).index) == [7, 9]


# --- weights ---------------------------------------------------------------

def test_weights_are_rescaled_to_sum_to_one() -> None:
    weights = normalize_weights({"quality": 3, "cost": 1})
    assert math.isclose(sum(weights.values()), 1.0)
    assert weights["quality"] == pytest.approx(0.75)
    assert weights["moq"] == 0.0  # missing factors get 0


def test_default_weights_cover_every_factor() -> None:
    assert set(DEFAULT_WEIGHTS) == {f.key for f in FACTORS}


@pytest.mark.parametrize("bad", [{"quality": 0, "cost": 0}, {"quality": -1, "cost": 2}, {"colour": 1}])
def test_invalid_weights_raise(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        normalize_weights(bad)


# --- hard filters ----------------------------------------------------------

def test_hard_filter_excludes_capability_failures(clozet_df: pd.DataFrame) -> None:
    eligible, excluded = apply_hard_filters(clozet_df)
    assert eligible["supplier_name"].tolist() == ["Manufacturer A", "Manufacturer B"]
    assert excluded["supplier_name"].tolist() == ["Manufacturer C"]
    assert excluded["exclusion_reason"].iloc[0] == EXCLUSION_REASON


def test_excluded_suppliers_are_not_ranked_and_do_not_affect_scale(clozet_df: pd.DataFrame) -> None:
    result = score_suppliers(clozet_df, {"cost": 1})
    assert "Manufacturer C" not in result.ranked["supplier_name"].tolist()
    # C (₹280) is excluded, so A (₹300) is the cheapest eligible and gets the full cost score.
    a = result.ranked.set_index("supplier_name").loc["Manufacturer A"]
    assert a["norm_cost"] == 1.0 and a["score"] == pytest.approx(100.0)


def test_all_suppliers_fail_hard_filter() -> None:
    df = validate_suppliers(pd.DataFrame([make_supplier("X", meets_required_capability=False)]))
    result = score_suppliers(df)
    assert result.ranked.empty and len(result.excluded) == 1


# --- weighted score --------------------------------------------------------

def test_hand_calculated_scores(clozet_df: pd.DataFrame) -> None:
    # Two eligible suppliers: each factor normalises to exactly 0 or 1.
    # Weights quality 1, cost 1 -> 50% each.
    # A: quality 4 (worse -> 0), cost 300 (cheaper -> 1)  => 50
    # B: quality 5 (better -> 1), cost 330 (dearer -> 0)  => 50 ; tie -> cheaper A ranks first.
    result = score_suppliers(clozet_df, {"quality": 1, "cost": 1})
    scores = result.ranked.set_index("supplier_name")["score"]
    assert scores["Manufacturer A"] == pytest.approx(50.0)
    assert scores["Manufacturer B"] == pytest.approx(50.0)
    assert result.ranked["supplier_name"].tolist()[0] == "Manufacturer A"

    # Weight quality 3 : cost 1 -> B gets 75, A gets 25.
    result = score_suppliers(clozet_df, {"quality": 3, "cost": 1})
    scores = result.ranked.set_index("supplier_name")["score"]
    assert scores["Manufacturer B"] == pytest.approx(75.0)
    assert scores["Manufacturer A"] == pytest.approx(25.0)
    assert result.ranked.iloc[0]["rank"] == 1 and result.ranked.iloc[0]["supplier_name"] == "Manufacturer B"


def test_three_supplier_intermediate_normalisation() -> None:
    df = validate_suppliers(pd.DataFrame([
        make_supplier("Cheap", unit_cost=200),
        make_supplier("Mid", unit_cost=250),
        make_supplier("Dear", unit_cost=400),
    ]))
    result = score_suppliers(df, {"cost": 1})
    scores = result.ranked.set_index("supplier_name")["score"]
    assert scores["Mid"] == pytest.approx(75.0)  # (400 - 250) / (400 - 200)


def test_contribution_points_sum_to_score(sample_df: pd.DataFrame) -> None:
    result = score_suppliers(sample_df)
    pts = result.ranked[[f"pts_{f.key}" for f in FACTORS]].sum(axis=1)
    assert (pts - result.ranked["score"]).abs().max() < 1e-9
    assert result.ranked["score"].between(0, 100).all()
    assert result.ranked["score"].is_monotonic_decreasing
    long = result.contributions()
    assert len(long) == len(result.ranked) * len(FACTORS)


def test_sample_data_ranking(sample_df: pd.DataFrame) -> None:
    result = score_suppliers(sample_df)
    assert len(result.ranked) == 22 and len(result.excluded) == 8
    assert "Manufacturer C (Surat)" in result.excluded["supplier_name"].tolist()
    ranks = result.ranked.set_index("supplier_name")["rank"]
    # B (better quality, reliability, MOQ, lead time) beats A under default weights.
    assert ranks["Manufacturer B (Ahmedabad)"] < ranks["Manufacturer A (Surat)"]
