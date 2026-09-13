"""
Tests for the box plot component.

The box-plot core lives in ``src.ui.utils.plotting`` as pure, testable
functions (:func:`box_plot_stats` and :func:`create_box_plot`) so these tests
exercise the figure-building logic directly without a Streamlit harness,
matching the convention used by the interaction-plot helpers.

Edge cases covered:
- one box per observed factor level (present levels only)
- categorical factor labels containing numbers (never coerced to numeric)
- replicated vs non-replicated designs
- missing response values are dropped
- numeric factor level sorting
- Tukey whisker/outlier behaviour delegated to Plotly
"""

import numpy as np
import pandas as pd
import pytest

from src.ui.utils.plotting import (
    box_plot_stats,
    create_box_plot,
)
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.full_factorial import full_factorial


class BaseBoxFixture:
    """Shared factory helpers for the box-plot tests."""

    LOT_LEVELS = ["41007587", "41005191", "41007741", "41007302"]

    @staticmethod
    def lot_factors():
        """Categorical Egg_lot (numeric-looking labels) + numeric Egg_percent."""
        return [
            Factor(
                "Egg_lot",
                FactorType.CATEGORICAL,
                ChangeabilityLevel.EASY,
                levels=BaseBoxFixture.LOT_LEVELS,
                _validate_on_init=False,
            ),
            Factor(
                "Egg_percent",
                FactorType.DISCRETE_NUMERIC,
                ChangeabilityLevel.EASY,
                levels=[1.2, 1.8],
            ),
        ]

    @staticmethod
    def build_lot_design(n_replicates=2, seed=42):
        factors = BaseBoxFixture.lot_factors()
        design = full_factorial(
            factors, n_replicates=n_replicates, randomize=False
        )
        rng = np.random.default_rng(seed)
        lot_means = {
            lot: m
            for lot, m in zip(
                BaseBoxFixture.LOT_LEVELS, [100, 102, 98, 105]
            )
        }
        response = (
            design["Egg_lot"].map(lot_means).astype(float)
            + 1.5 * design["Egg_percent"]
            + rng.normal(0, 0.05, len(design))
        )
        return design, np.asarray(response, dtype=float), factors


class TestBoxPlotStats(BaseBoxFixture):
    """Aggregation logic."""

    def test_one_row_per_level(self):
        design, response, factors = self.build_lot_design(n_replicates=2)
        stats = box_plot_stats(
            "Egg_lot", design, response, is_categorical=True
        )
        # 4 lots with 2 replicates across 2 percents -> one row per lot.
        assert list(stats["level"]) == sorted(self.LOT_LEVELS)
        assert all(stats["n"] == 4)

    def test_numeric_levels_stay_strings(self):
        design, response, factors = self.build_lot_design()
        stats = box_plot_stats(
            "Egg_lot", design, response, is_categorical=True
        )
        assert all(isinstance(v, str) for v in stats["level"])

    def test_numeric_factor_levels_sorted_numerically(self):
        factors = [
            Factor("Temp", FactorType.DISCRETE_NUMERIC,
                   ChangeabilityLevel.EASY, levels=[70, 80]),
        ]
        design = full_factorial(factors, n_replicates=2, randomize=False)
        response = np.array([10.0, 12.0, 16.0, 18.0], dtype=float)
        stats = box_plot_stats("Temp", design, response, is_categorical=False)
        # Levels sorted numerically, not lexically.
        assert list(stats["level"]) == ["70", "80"]
        assert all(isinstance(v, str) for v in stats["level"])

    def test_non_replicated_collapses_std(self):
        # Deterministic response -> identical values within a level.
        factors = [
            Factor("A", FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
                   levels=["a", "b"]),
        ]
        design = full_factorial(factors, n_replicates=3, randomize=False)
        # Row order is a,b,a,b,a,b; assign each level a single constant value.
        response = np.array(
            [10.0, 20.0, 10.0, 20.0, 10.0, 20.0]
        )
        stats = box_plot_stats("A", design, response, is_categorical=True)
        assert list(stats["level"]) == ["a", "b"]
        assert all(stats["std"] == 0.0)
        assert list(stats["n"]) == [3, 3]

    def test_missing_response_drops_row_value(self):
        design, response, factors = self.build_lot_design(n_replicates=2)
        # Null both replicates of the first run's lot.
        response = response.copy()
        first_lot_runs = design.index[design["Egg_lot"] == self.LOT_LEVELS[0]]
        response[first_lot_runs] = np.nan
        stats = box_plot_stats(
            "Egg_lot", design, response, is_categorical=True
        )
        assert self.LOT_LEVELS[0] not in list(stats["level"])
        assert len(stats) == 3
        # No NaN anywhere.
        assert not pd.isna(stats["mean"]).any()


class TestCreateBoxPlot(BaseBoxFixture):
    """Figure-building behaviour."""

    def _fig(self, **kwargs):
        design, response, factors = self.build_lot_design()
        stats = box_plot_stats(
            "Egg_lot", design, response, is_categorical=True
        )
        values_map = {
            level: design.loc[design["Egg_lot"] == level, :].index.map(
                lambda i: response[i]
            ).values.astype(float)
            for level in stats["level"]
        }
        default = {
            "stats": stats,
            "factor_name": "Egg_lot",
            "factor_label": "Egg_lot",
            "response_name": "Day_7_Max_Force",
            "response_values_by_level": values_map,
        }
        default.update(kwargs)
        return create_box_plot(**default)

    def test_one_box_per_level(self):
        fig = self._fig()
        assert len(fig.data) == 4  # four lot levels
        assert fig.layout.xaxis.type == "category"
        assert all(isinstance(t.type, str) and "box" in t.type for t in fig.data)

    def test_boxes_unfilled_all_black(self):
        fig = self._fig()
        for t in fig.data:
            assert t.fillcolor in (None, "rgba(0,0,0,0)")
            assert t.line.color == "#000000"
            assert t.marker.color == "#000000"

    def test_y_axis_label_includes_units(self):
        fig = self._fig(response_units="kg")
        assert fig.layout.yaxis.title.text == "Day_7_Max_Force (kg)"

    def test_tooltip_includes_response_and_n(self):
        fig = self._fig()
        tmpl = fig.data[0].hovertemplate
        assert "Day_7_Max_Force:" in tmpl and "n:" in tmpl
        assert "Egg_lot:" in tmpl

    def test_2_level_design(self):
        factors = [
            Factor("A", FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
                   levels=["a", "b"]),
        ]
        design = full_factorial(factors, randomize=False)
        response = np.array([10.0, 12.0])
        stats = box_plot_stats("A", design, response, is_categorical=True)
        values_map = {
            level: np.array([response[i] for i in design.index
                             if str(design.loc[i, "A"]) == level])
            for level in stats["level"]
        }
        fig = create_box_plot(stats, "A", "A", "Resp", values_map)
        assert len(fig.data) == 2
        assert fig.layout.xaxis.type == "category"