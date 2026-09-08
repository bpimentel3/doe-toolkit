"""
Tests for the interaction plot component.

The interaction-plot core lives in ``src.ui.utils.plotting`` as pure,
testable functions (:func:`interaction_stats` and
:func:`create_interaction_plot`) so these tests exercise the figure-building
logic directly without a Streamlit harness, matching the convention used by
the existing pure plotting helpers.

Edge cases covered:
- 2x2 designs
- 4x2 designs (Egg_lot x Egg_percent)
- categorical factor labels containing numbers (never coerced to numeric)
- replicated vs non-replicated designs
- missing response values
- error-bar modes (mean only / +/- SD / +/- CI)
- interaction p-value overlay (present, marginal, significant, absent)
- numeric factor level sorting
"""

import numpy as np
import pandas as pd
import pytest

from src.ui.utils.plotting import (
    interaction_stats,
    create_interaction_plot,
    QUALITATIVE_COLORS,
    SEQUENTIAL_COLORS,
)
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.full_factorial import full_factorial
from src.core.analysis import ANOVAAnalysis


class BaseInteractionFixture:
    """Shared factory helpers for the interaction-plot tests."""

    LOT_LEVELS = ["41007587", "41005191", "41007741", "41007302"]

    @staticmethod
    def lot_factors():
        """Categorical Egg_lot (numeric-looking labels) + numeric Egg_percent."""
        return [
            Factor(
                "Egg_lot",
                FactorType.CATEGORICAL,
                ChangeabilityLevel.EASY,
                levels=BaseInteractionFixture.LOT_LEVELS,
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
        factors = BaseInteractionFixture.lot_factors()
        design = full_factorial(
            factors, n_replicates=n_replicates, randomize=False
        )
        rng = np.random.default_rng(seed)
        lot_means = {
            lot: m
            for lot, m in zip(
                BaseInteractionFixture.LOT_LEVELS, [100, 102, 98, 105]
            )
        }
        response = (
            design["Egg_lot"].map(lot_means).astype(float)
            + 1.5 * design["Egg_percent"]
            + rng.normal(0, 0.05, len(design))
        )
        return design, np.asarray(response, dtype=float), factors


class TestInteractionStats(BaseInteractionFixture):
    """Aggregation logic."""

    def test_mean_and_n_correct(self):
        design, response, factors = self.build_lot_design(n_replicates=2)
        stats = interaction_stats("Egg_lot", "Egg_percent", design, response)

        # 4 lots x 2 percents = 8 combinations, each with 2 replicates.
        assert len(stats) == 8
        assert all(stats["n"] == 2)

        # Egg_percent 1.8 should have higher mean than 1.2 for the same lot.
        p12 = stats[stats["Egg_percent"] == 1.2]
        p18 = stats[stats["Egg_percent"] == 1.8]
        assert (p18["mean"].values > p12["mean"].values).all()
        assert all(stats["std"] > 0)
        assert all(stats["ci_lower"] < stats["mean"])
        assert all(stats["mean"] < stats["ci_upper"])

    def test_categorical_levels_stay_strings(self):
        design, response, factors = self.build_lot_design()
        stats = interaction_stats("Egg_lot", "Egg_percent", design, response)
        assert all(isinstance(v, str) for v in stats["Egg_lot"])

    def test_non_replicated_collapses_spread(self):
        design, response, factors = self.build_lot_design(n_replicates=1)
        # Deterministic response -> identical means within a combo.
        lot_means = {lot: m for lot, m in
                     zip(self.LOT_LEVELS, [100, 102, 98, 105])}
        response = (
            design["Egg_lot"].map(lot_means).astype(float)
            + 1.5 * design["Egg_percent"]
        )
        stats = interaction_stats("Egg_lot", "Egg_percent", design, response)
        assert all(stats["n"] == 1)
        assert all(stats["std"] == 0.0)
        assert all(stats["sem"] == 0.0)
        assert np.allclose(stats["ci_lower"], stats["mean"])
        assert np.allclose(stats["ci_upper"], stats["mean"])

    def test_missing_response_drops_row(self):
        design, response, factors = self.build_lot_design(n_replicates=2)
        # Null one replicate of the first run.
        response = response.copy()
        response[0] = np.nan
        stats = interaction_stats("Egg_lot", "Egg_percent", design, response)

        # The affected combo has n==1 remaining; all other combos keep n==2.
        assert len(stats) == 8
        dropped_lot = design["Egg_lot"].iloc[0]
        dropped_percent = design["Egg_percent"].iloc[0]
        affected = stats[
            (stats["Egg_lot"] == str(dropped_lot))
            & (stats["Egg_percent"] == dropped_percent)
        ].iloc[0]
        assert affected["n"] == 1
        # No NaN means anywhere.
        assert not stats["mean"].isna().any()


class TestCreateInteractionPlot(BaseInteractionFixture):
    """Figure-building behaviour."""

    def _fig(self, **kwargs):
        design, response, factors = self.build_lot_design()
        stats = interaction_stats("Egg_lot", "Egg_percent", design, response)
        default = {
            "stats": stats,
            "f1_name": "Egg_lot",
            "f2_name": "Egg_percent",
            "f1_is_categorical": True,
            "f2_is_categorical": False,
            "response_name": "Day_7_Max_Force",
        }
        default.update(kwargs)
        return create_interaction_plot(**default), stats

    def test_numeric_line_factor_has_sequential_palette(self):
        fig, _ = self._fig()
        # Egg_lot on x (categorical) -> category axis.
        assert fig.layout.xaxis.type == "category"
        assert len(fig.data) == 2  # two Egg_percent levels
        assert fig.data[0].line.color == SEQUENTIAL_COLORS[0]
        assert fig.data[1].line.color == SEQUENTIAL_COLORS[-1]

    def test_categorical_line_factor_has_qualitative_palette(self):
        fig, stats = self._fig(
            f1_name="Egg_percent",
            f2_name="Egg_lot",
            f1_is_categorical=False,
            f2_is_categorical=True,
        )
        assert fig.layout.xaxis.type == "linear"
        # Four lot levels -> four lines, coloured from the qualitative palette.
        assert len(fig.data) == 4
        colors = [t.line.color for t in fig.data]
        assert colors == QUALITATIVE_COLORS[:4]

    def test_numeric_x_levels_sorted(self):
        fig, _ = self._fig(
            f1_name="Egg_percent",
            f2_name="Egg_lot",
            f1_is_categorical=False,
            f2_is_categorical=True,
        )
        xs = list(fig.data[0].x)
        assert xs == sorted(xs)

    def test_error_mode_none_has_no_error_bars(self):
        fig, _ = self._fig(error_mode="none")
        for t in fig.data:
            # Plotly materialises an empty error_y object when unset.
            assert len(t.error_y.to_plotly_json()) == 0

    def test_error_mode_sd_sets_error_y(self):
        fig, _ = self._fig(error_mode="sd")
        for t in fig.data:
            assert "array" in t.error_y.to_plotly_json()

    def test_error_mode_ci_sets_error_y(self):
        fig, _ = self._fig(error_mode="ci")
        for t in fig.data:
            assert "array" in t.error_y.to_plotly_json()

    def test_interaction_significant_subtitle(self):
        fig, _ = self._fig(p_value=0.012)
        sub = fig.layout.title.subtitle.text
        assert "Significant interaction" in sub
        assert "0.012" in sub

    def test_interaction_marginal_subtitle(self):
        fig, _ = self._fig(p_value=0.07)
        assert "Marginal interaction" in fig.layout.title.subtitle.text

    def test_interaction_not_significant_subtitle(self):
        fig, _ = self._fig(p_value=0.5)
        assert "No significant interaction" in fig.layout.title.subtitle.text

    def test_interaction_absent_marks_not_in_model(self):
        fig, _ = self._fig(interaction_present=False)
        assert "not in model" in fig.layout.title.subtitle.text

    def test_tooltip_includes_sd_and_n(self):
        fig, _ = self._fig()
        tmpl = fig.data[0].hovertemplate
        assert "SD:" in tmpl and "n:" in tmpl and "Day_7_Max_Force:" in tmpl
        # The x-labels appear in the tooltip.
        assert "Egg_lot:" in tmpl and "Egg_percent:" in tmpl

    def test_2x2_design(self):
        factors = [
            Factor("A", FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
                   levels=["a", "b"]),
            Factor("B", FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
                   levels=["x", "y"]),
        ]
        design = full_factorial(factors, randomize=False)
        response = np.array([10.0, 20.0, 12.0, 22.0])
        stats = interaction_stats("A", "B", design, response)
        assert len(stats) == 4
        fig = create_interaction_plot(
            stats, "A", "B", True, True, "Resp"
        )
        assert len(fig.data) == 2  # two B levels
        assert fig.layout.xaxis.type == "category"

    def test_integer_numeric_factors_render_lines(self):
        """Integer-valued discrete-numeric factors must draw visible lines.

        Regression: ``_sorted_levels`` returned floats (1.0, 2.0) while the
        stats columns were stringified from the original ints ('1', '2'),
        so ``str(1.0)`` never matched '1' and every y-value was None.
        """
        factors = [
            Factor("Temperature", FactorType.DISCRETE_NUMERIC,
                   ChangeabilityLevel.EASY, levels=[70, 80]),
            Factor("Catalyst", FactorType.DISCRETE_NUMERIC,
                   ChangeabilityLevel.EASY, levels=[1, 2]),
        ]
        design = full_factorial(factors, n_replicates=2, randomize=False)
        # Deterministic response: each temperature/catalyst combo distinct.
        response = np.array(
            [10.0, 12.0, 16.0, 18.0] * 2, dtype=float
        )
        assert design["Catalyst"].dtype.kind in "iu"  # integer column

        stats = interaction_stats("Temperature", "Catalyst", design, response)
        assert len(stats) == 4  # 2 temperatures x 2 catalysts

        # Catalyst on the x-axis (integer numeric): lines drawn, not empty.
        fig = create_interaction_plot(
            stats, "Catalyst", "Temperature", False, False, "Yield"
        )
        assert len(fig.data) == 2
        ys = sorted(tuple(t.y) for t in fig.data)
        assert ys == [(10.0, 12.0), (16.0, 18.0)]
        assert fig.layout.xaxis.type == "linear"
        assert list(fig.data[0].x) == [1.0, 2.0]

        # Temperature on the x-axis (integer numeric): lines drawn too.
        fig2 = create_interaction_plot(
            stats, "Temperature", "Catalyst", False, False, "Yield"
        )
        assert len(fig2.data) == 2
        ys2 = sorted(tuple(t.y) for t in fig2.data)
        assert ys2 == [(10.0, 16.0), (12.0, 18.0)]
        assert list(fig2.data[0].x) == [70.0, 80.0]


class TestIntegrationWithANOVA(BaseInteractionFixture):
    """End-to-end: fit ANOVA, read interaction p-value, build plot."""

    def test_pvalue_extracted_from_fitted_model(self):
        design, response, factors = self.build_lot_design()
        analysis = ANOVAAnalysis(design, response, factors)
        results = analysis.fit(
            ["Egg_lot", "Egg_percent", "Egg_lot*Egg_percent"]
        )

        key = "Egg_lot:Egg_percent"
        col = "PR(>F)" if "PR(>F)" in results.anova_table.columns else "P"
        p_value = float(results.anova_table.loc[key, col])
        assert 0.0 < p_value < 1.0

        stats = interaction_stats("Egg_lot", "Egg_percent", design, response)
        fig = create_interaction_plot(
            stats,
            "Egg_lot",
            "Egg_percent",
            True,
            False,
            "Day_7_Max_Force",
            p_value=p_value,
            interaction_present=True,
        )
        assert "interaction p-value" in fig.layout.title.subtitle.text
