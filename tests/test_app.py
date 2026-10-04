"""Smoke tests: the Streamlit app runs end-to-end without errors."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_starts_with_sample_data(app: AppTest) -> None:
    assert not app.exception, app.exception
    assert not app.error
    assert [t.label for t in app.tabs] == ["📋 Suppliers", "🏆 Scorecard", "📦 MOQ Risk",
                                           "📈 Scaling", "✅ Recommendation"]
    successes = " ".join(s.value for s in app.success)
    assert "Recommended:" in successes


def test_changing_selling_price_reruns_cleanly(app: AppTest) -> None:
    # The first number input on the page is the selling price in the MOQ Risk tab.
    app.number_input[0].set_value(999).run()
    assert not app.exception
    assert any("Recommended:" in s.value for s in app.success)


def test_filters_and_weights_rerun_cleanly(app: AppTest) -> None:
    app.sidebar.multiselect[0].set_value(["Gujarat"]).run()  # state filter
    assert not app.exception
    app.sidebar.slider[0].set_value(0).run()                  # quality weight
    assert not app.exception
    assert any("Recommended:" in s.value for s in app.success)


def test_all_weights_zero_shows_error(app: AppTest) -> None:
    for slider in app.sidebar.slider:
        slider.set_value(0)
    app.run()
    assert not app.exception
    assert any("at least one scorecard weight" in e.value for e in app.error)
