"""
Regression tests for the HTML report's Prediction Profiler section.

``_build_profiler_section`` in ``src/ui/utils/export.py`` recomputes trace
predictions at export time. The original implementation called
``results.fitted_model.predict(...)`` on dataframes that contained raw
categorical strings — statsmodels needs the patsy dummy-column design matrix,
so injecting the raw string produced NaN predictions for the entire row and
``json.dumps(..., allow_nan=False)`` then raised
"Out of range float values are not JSON compliant: nan". Categorical factors
also crashed the x-axis construction (``np.array(factor.levels, dtype=float)``).

The fix predicts via ``results.predict_from_settings(encode_settings_dict(...))``
and renders categorical factors as bar charts. These tests lock that behaviour
in without a running Streamlit session (the module's ``st.session_state`` is
faked).
"""

import types

import numpy as np
import pandas as pd
import pytest

import src.ui.utils.export as export
from src.core.analysis import ANOVAAnalysis
from src.core.factors import ChangeabilityLevel, Factor, FactorType
from src.core.full_factorial import full_factorial


class _FakeSessionState:
    """Minimal stand-in for ``st.session_state`` supporting ``.get``."""

    def __init__(self, data):
        self._data = data

    def get(self, key, default=None):
        return self._data.get(key, default)


class ProfilerFixture:
    """Factory for a fitted model with a categorical + continuous factors."""

    LEVELS = ["nitrogen", "oxygen", "air"]

    @staticmethod
    def make_factors():
        return [
            Factor(
                "Atmosphere",
                FactorType.CATEGORICAL,
                ChangeabilityLevel.EASY,
                levels=ProfilerFixture.LEVELS,
            ),
            Factor(
                "feed_rate",
                FactorType.CONTINUOUS,
                ChangeabilityLevel.EASY,
                levels=[10, 30],
            ),
            Factor(
                "Temperature",
                FactorType.CONTINUOUS,
                ChangeabilityLevel.EASY,
                levels=[100, 200],
            ),
        ]

    @staticmethod
    def build_fitted():
        factors = ProfilerFixture.make_factors()
        design = full_factorial(factors, n_replicates=2, randomize=False)
        rng = np.random.default_rng(0)
        means = design["Atmosphere"].map(
            {"nitrogen": 10.0, "oxygen": 12.0, "air": 11.0}
        )
        response = (
            means
            + 0.1 * design["feed_rate"]
            + 0.05 * design["Temperature"]
            + rng.normal(0, 0.5, len(design))
        )
        ana = ANOVAAnalysis(design, response, factors, response_name="yid")
        results = ana.fit(["Atmosphere", "feed_rate", "Temperature"])
        return results, factors

    @staticmethod
    def build_html(results, factors):
        fake = _FakeSessionState(
            {
                "profiler_settings": {
                    "Atmosphere": "nitrogen",
                    "feed_rate": 20.0,
                    "Temperature": 150.0,
                }
            }
        )
        original = export.st.session_state
        export.st.session_state = fake
        try:
            return export._build_profiler_section("yid", results, factors)
        finally:
            export.st.session_state = original


class TestProfilerSectionPredictions:
    def test_prediction_value_is_a_real_float(self):
        results, factors = ProfilerFixture.build_fitted()
        html = ProfilerFixture.build_html(results, factors)
        assert "Prediction at displayed settings: nan" not in html
        assert "Prediction at displayed settings: -0" != html  # sanity guard
        assert "Prediction at displayed settings: " in html

    def test_no_nan_and_no_failed_markers(self):
        results, factors = ProfilerFixture.build_fitted()
        html = ProfilerFixture.build_html(results, factors)
        assert "nan" not in html.lower()
        assert "failed:" not in html
        assert "failed" not in html.lower()

    def test_categorical_rendered_as_bar_trace(self):
        results, factors = ProfilerFixture.build_fitted()
        html = ProfilerFixture.build_html(results, factors)
        assert '"type":"bar"' in html or '"type": "bar"' in html
        assert "nitrogen" in html
        assert "oxygen" in html
        assert "air" in html

    def test_continuous_rendered_as_scatter_trace(self):
        results, factors = ProfilerFixture.build_fitted()
        html = ProfilerFixture.build_html(results, factors)
        assert '"type":"scatter"' in html or '"type": "scatter"' in html

    def test_no_error_when_profiler_never_run(self):
        # If profiler_settings is missing the section renders a muted note.
        fake = _FakeSessionState({})
        original = export.st.session_state
        export.st.session_state = fake
        try:
            results, factors = ProfilerFixture.build_fitted()
            html = export._build_profiler_section("yid", results, factors)
        finally:
            export.st.session_state = original
        assert "Profiler not yet used" in html
        assert "Prediction at displayed settings" not in html