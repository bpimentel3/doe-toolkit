"""
Tests for missing-response-value handling in ANOVA analysis.

``statsmodels`` ``ols``/``mixedlm`` silently drop rows whose response value is
missing, so the fitted model uses fewer observations than the analysis frame
contains.  These tests pin down the observation-usage metadata exposed via
``ANOVAResults`` — total/used/excluded counts, excluded-run labels, and the
row-alignment arrays used by the diagnostic plots.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from src.core.analysis import ANOVAAnalysis
from src.core.factors import Factor, FactorType, ChangeabilityLevel


def _factors_ff() -> list:
    return [
        Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
    ]


def _design_ff(with_runorder: bool = True) -> pd.DataFrame:
    design = pd.DataFrame({
        "A": [-1, -1, 1, 1, -1, -1, 1, 1],
        "B": [-1, 1, -1, 1, -1, 1, -1, 1],
    })
    if with_runorder:
        design["RunOrder"] = [11, 12, 13, 14, 15, 16, 17, 18]
        design["StdOrder"] = [1, 2, 3, 4, 5, 6, 7, 8]
    return design


def _response_with_missing() -> np.ndarray:
    return np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0])


def _fit_ff(design: pd.DataFrame, response: np.ndarray):
    analysis = ANOVAAnalysis(
        design=design, response=response, factors=_factors_ff(), response_name="Y"
    )
    return analysis.fit(['1', 'A', 'B'])


class TestMissingResponseUsage:
    """Observation-usage metadata for fits with missing response values."""

    def test_usage_counts(self):
        results = _fit_ff(_design_ff(), _response_with_missing())
        assert results.n_obs_total == 8
        assert results.n_obs_used == 6
        assert results.n_obs_excluded == 2

    def test_used_row_indices_missing_mask(self):
        response = _response_with_missing()
        results = _fit_ff(_design_ff(), response)
        expected = np.flatnonzero(pd.notna(response))
        np.testing.assert_array_equal(results.used_row_indices, expected)

    def test_excluded_labels_use_run_order(self):
        results = _fit_ff(_design_ff(), _response_with_missing())
        assert results.excluded_obs_labels == ["Run 13", "Run 16"]

    def test_excluded_labels_fallback_without_meta_columns(self):
        results = _fit_ff(_design_ff(with_runorder=False), _response_with_missing())
        assert results.excluded_obs_labels == ["Row 3", "Row 6"]

    def test_used_response_alignment(self):
        response = _response_with_missing()
        results = _fit_ff(_design_ff(), response)
        used_expected = response[pd.notna(response)]
        np.testing.assert_allclose(results.used_response, used_expected)
        np.testing.assert_allclose(
            results.residuals,
            np.asarray(results.used_response) - np.asarray(results.fitted_values),
            atol=1e-8,
        )

    def test_complete_response_reports_zero_excluded(self):
        response = np.arange(1.0, 9.0)
        results = _fit_ff(_design_ff(), response)
        assert results.n_obs_total == 8
        assert results.n_obs_used == 8
        assert results.n_obs_excluded == 0
        assert results.excluded_obs_labels == []
        np.testing.assert_allclose(results.used_response, response)

    def test_categorical_and_discrete_numeric_factors(self):
        """Egg-yolk-style frame: categorical + discrete-numeric factors."""
        factors = [
            Factor("Egg_lot", FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
                   levels=["A", "B", "C", "D"]),
            Factor("Egg_percent", FactorType.DISCRETE_NUMERIC,
                   ChangeabilityLevel.EASY, levels=[1.2, 1.8]),
        ]
        design = pd.DataFrame({
            "Egg_lot": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "Egg_percent": [1.2, 1.8, 1.2, 1.8, 1.2, 1.8, 1.2, 1.8],
            "RunOrder": [1, 2, 3, 4, 5, 6, 7, 8],
        })
        response = np.array([1.0, np.nan, 3.0, 4.0, 5.0, np.nan, 7.0, 8.0])
        analysis = ANOVAAnalysis(
            design=design, response=response, factors=factors, response_name="Y"
        )
        results = analysis.fit(['1', 'Egg_lot', 'Egg_percent'])
        assert results.n_obs_total == 8
        assert results.n_obs_used == 6
        assert results.n_obs_excluded == 2
        assert results.excluded_obs_labels == ["Run 2", "Run 6"]
        np.testing.assert_array_equal(
            results.used_row_indices, np.flatnonzero(pd.notna(response))
        )

    def test_transform_term_path_uses_natural_data(self):
        """Transform terms fit on the natural-data frame but must still report
        the same observation usage."""
        factors = [
            Factor("X", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                   levels=[1.0, 3.0]),
            Factor("Z", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                   levels=[10.0, 30.0]),
        ]
        design = pd.DataFrame({
            "X": [1.0, 3.0, 1.0, 3.0, 1.0, 3.0, 1.0, 3.0],
            "Z": [10.0, 10.0, 20.0, 20.0, 30.0, 30.0, 10.0, 10.0],
            "RunOrder": [9, 8, 7, 6, 5, 4, 3, 2],
        })
        response = np.array([1.0, 2.0, np.nan, 4.0, 5.0, np.nan, 7.0, 8.0])
        analysis = ANOVAAnalysis(
            design=design, response=response, factors=factors, response_name="Y"
        )
        results = analysis.fit(['1', 'np.log(X)', 'Z'])
        assert results.n_obs_total == 8
        assert results.n_obs_used == 6
        assert results.n_obs_excluded == 2
        assert results.excluded_obs_labels == ["Run 7", "Run 4"]
        np.testing.assert_array_equal(
            results.used_row_indices, np.flatnonzero(pd.notna(response))
        )

    def test_blocked_design_missing_response(self):
        factors = _factors_ff()
        design = _design_ff()
        design["Block"] = ["I", "I", "II", "II", "III", "III", "IV", "IV"]
        response = _response_with_missing()
        analysis = ANOVAAnalysis(
            design=design, response=response, factors=factors, response_name="Y"
        )
        results = analysis.fit(['1', 'A', 'B'])
        assert results.n_obs_total == 8
        assert results.n_obs_used == 6
        assert results.n_obs_excluded == 2
        assert results.excluded_obs_labels == ["Run 13", "Run 16"]

    def test_split_plot_usage_reported(self):
        """Split-plot fits must report usage from the subplot model."""
        from pathlib import Path

        csv_path = (
            Path(__file__).parent.parent / "test_data" / "test_case_5_split_plot.csv"
        )
        raw = pd.read_csv(csv_path, comment="#")
        response = raw["Yield"].values.copy()
        response[2] = np.nan
        response[7] = np.nan

        factors = [
            Factor("Temperature", FactorType.CONTINUOUS,
                   ChangeabilityLevel.HARD, levels=[125, 175]),
            Factor("Pressure", FactorType.CONTINUOUS,
                   ChangeabilityLevel.HARD, levels=[25, 75]),
            Factor("Time", FactorType.CONTINUOUS,
                   ChangeabilityLevel.EASY, levels=[0, 20]),
            Factor("Catalyst", FactorType.CATEGORICAL,
                   ChangeabilityLevel.EASY, levels=["A", "B"]),
        ]
        analysis = ANOVAAnalysis(
            design=raw, response=response, factors=factors, response_name="Yield"
        )
        results = analysis.fit(
            ['Temperature', 'Pressure', 'Time', 'Catalyst'],
            enforce_hierarchy_flag=False,
        )
        assert results.is_split_plot is True
        assert results.n_obs_total == len(raw)
        assert results.n_obs_used == len(raw) - 2
        assert results.n_obs_excluded == 2
        assert len(results.excluded_obs_labels) == 2
        np.testing.assert_allclose(
            results.residuals,
            np.asarray(results.used_response) - np.asarray(results.fitted_values),
            atol=1e-8,
        )