"""Tests for the two complementary effect charts and their canonical tables.

- ``ANOVAResults.coefficient_significance``: coefficient-level tests taken
  from the fitted model (``fitted_model.pvalues``), preserved unchanged.
- ``ANOVAResults.anova_effect_summary``: term-level tests sourced from the
  displayed ANOVA table (``PR(>F)`` / ``P``), with standardized statistics
  ``|t| = sqrt(F)`` (one-Df) or ``omnibus sqrt(F)`` (multi-Df).

See the repository docs for the verified Yield example: Temperature has an
ANOVA F = 88.5088 (p = 1.08e-9) while its fitted-model coefficient test has a
very different p-value.  The two views intentionally answer different
questions and must not be merged.
"""
import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from src.core.analysis import ANOVAAnalysis
from src.core.analysis_base import (
    BONFERRONI_EXCLUDE_BLOCK,
    attach_critical_limits,
    coefficient_logworth,
)
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.ui.utils.plotting import (
    create_coefficient_significance_plot,
    create_standardized_effects_plot,
)

DATA_DIR = Path(__file__).parent.parent / "test_data"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _blocked_fixture():
    """2^4 blocked full factorial (16 runs), all-easy continuous factors."""
    rows = []
    for i in range(16):
        bits = [(i >> k) & 1 for k in range(4)]
        rows.append({
            "A": 40.0 + 20.0 * bits[0],
            "B": 80.0 + 40.0 * bits[1],
            "C": 2.5 + 5.0 * bits[2],
            "D": 150.0 + 100.0 * bits[3],
            "Block": 1 + (i % 2),
        })
    design = pd.DataFrame(rows)
    rng = np.random.RandomState(11)
    a = np.where(design["A"] == 60.0, 1.0, -1.0)
    b = np.where(design["B"] == 120.0, 1.0, -1.0)
    c = np.where(design["C"] == 7.5, 1.0, -1.0)
    d = np.where(design["D"] == 250.0, 1.0, -1.0)
    block = np.where(design["Block"] == 2, 1.0, -1.0)
    design["Yield"] = (
        70.0
        + 4.0 * a + 3.0 * b - 2.0 * c + 5.0 * d
        + 2.0 * c * d + 1.5 * block
        + rng.normal(0.0, 0.4, len(design))
    )
    factors = [
        Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[40.0, 60.0]),
        Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[80.0, 120.0]),
        Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[2.5, 7.5]),
        Factor("D", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[150.0, 250.0]),
    ]
    return design, factors


def _fit_blocked(model_terms=None):
    design, factors = _blocked_fixture()
    analysis = ANOVAAnalysis(
        design=design, response=design["Yield"].values,
        factors=factors, response_name="Yield",
    )
    terms = model_terms or ["A", "B", "C", "D", "C*D"]
    return analysis.fit(terms, enforce_hierarchy_flag=False), analysis


def _categorical_fixture():
    """4-level 'numeric-looking' categorical (reference-code meets patsy C())."""
    levels = [41007587, 41007666, 41005191, 41007741]
    design = pd.DataFrame({
        "Cata": pd.Series(levels * 8, dtype="category"),
        "X": np.tile([10.0, 20.0] * 16, 1),
        "Block": np.tile([1, 1, 2, 2] * 8, 1),
    })
    rng = np.random.RandomState(5)
    mu = np.array([0.0, 4.0, 7.0, 9.0])
    design["Yield"] = (
        100.0
        + mu[np.array(design["Cata"].cat.codes) % 4]
        + 2.0 * np.where(design["X"] == 20.0, 1.0, -1.0)
        + 1.0 * np.where(design["Block"] == 2, 1.0, -1.0)
        + rng.normal(0.0, 1.0, len(design))
    )
    factors = [
        Factor("Cata", FactorType.CATEGORICAL, ChangeabilityLevel.EASY, levels=levels),
        Factor("X", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[10.0, 20.0]),
    ]
    return design, factors


# ---------------------------------------------------------------------------
# Coefficient-level chart: fitted-model p-values preserved
# ---------------------------------------------------------------------------

class TestCoefficientSignificance:
    def test_p_values_match_fitted_model(self):
        results, _ = _fit_blocked()
        cs = results.coefficient_significance
        assert cs is not None and not cs.empty
        unwrap = {k.replace("C(", "").replace(")", ""): v
                  for k, v in results.fitted_model.pvalues.items()}
        for _, row in cs.iterrows():
            assert row["coefficient_name"] in unwrap, row["coefficient_name"]
            assert row["p_value"] == pytest.approx(
                unwrap[row["coefficient_name"]]
            )
            assert row["source"] == "Fitted-model coefficient test"

    def test_logworth_formula(self):
        results, _ = _fit_blocked()
        cs = results.coefficient_significance
        for _, row in cs.iterrows():
            assert row["logworth"] == pytest.approx(
                coefficient_logworth(row["p_value"])
            )

    def test_intercept_excluded(self):
        results, _ = _fit_blocked()
        names = results.coefficient_significance["coefficient_name"].tolist()
        assert "Intercept" not in names
        assert "const" not in names

    def test_block_rows_flagged(self):
        results, _ = _fit_blocked()
        cs = results.coefficient_significance
        block_rows = cs[cs["is_block"].fillna(False)]
        assert len(block_rows) >= 1
        assert all("Block" in n for n in block_rows["coefficient_name"])

    def test_preserves_coefficient_and_t(self):
        results, _ = _fit_blocked()
        cs = results.coefficient_significance
        assert cs["coefficient_estimate"].notna().all()
        assert cs["t_value"].notna().all()
        assert not cs["p_value"].isna().any()


# ---------------------------------------------------------------------------
# ANOVA-level chart: term-level ANOVA table is source of truth
# ---------------------------------------------------------------------------

class TestStandardizedEffects:
    def test_p_values_match_displayed_anova_table(self):
        results, _ = _fit_blocked()
        aes = results.anova_effect_summary
        for term, row in aes.iterrows():
            anova_row = results.anova_table.loc[term]
            assert row["p_value"] == pytest.approx(anova_row["PR(>F)"])
            assert row["source"] == "displayed term-level ANOVA table"

    def test_one_df_t_is_sqrt_F(self):
        results, _ = _fit_blocked()
        aes = results.anova_effect_summary
        for term, row in aes.iterrows():
            if row["df"] != 1:
                continue
            assert row["standardized_statistic_type"] == "|t| = sqrt(F)"
            assert row["standardized_statistic"] ** 2 == pytest.approx(row["F"])

    def test_residual_df_comes_from_residual_row(self):
        results, _ = _fit_blocked()
        aes = results.anova_effect_summary
        residual_df = results.anova_table.loc["Residual", "df"]
        assert (aes["residual_df"] == residual_df).all()

    def test_effect_sign_matches_coefficient(self):
        results, _ = _fit_blocked()
        aes = results.anova_effect_summary
        for term, row in aes.iterrows():
            if row["df"] != 1:
                continue
            if np.isnan(row["effect_estimate"]):
                continue
            assert row["effect_sign"] == np.sign(row["effect_estimate"])

    def test_critical_limits_bonferroni_stricter(self):
        results, _ = _fit_blocked()
        aes = attach_critical_limits(results.anova_effect_summary.copy(), alpha=0.05)
        finite = aes[aes["t_critical"].notna()]
        assert not finite.empty
        assert (finite["bonferroni_limit"] > finite["t_critical"]).all()

    def test_block_bonferroni_policy(self):
        """Block excluded from the Bonferroni family by default."""
        assert BONFERRONI_EXCLUDE_BLOCK is True
        results, _ = _fit_blocked()
        aes = results.anova_effect_summary
        assert bool(aes.loc["Block", "is_block"])
        n_eligible = int(np.sum(~aes["is_block"].astype(bool)))
        assert n_eligible == len(aes) - 1


class TestMultiDfOmnibus:
    def test_numeric_categorical_is_multi_df(self):
        design, factors = _categorical_fixture()
        analysis = ANOVAAnalysis(
            design=design, response=design["Yield"].values,
            factors=factors, response_name="Yield",
        )
        results = analysis.fit(["Cata", "X"], enforce_hierarchy_flag=False)
        aes = results.anova_effect_summary
        assert aes.loc["Cata", "df"] == 3
        assert aes.loc["Cata", "standardized_statistic_type"] == "omnibus sqrt(F)"
        assert np.isnan(aes.loc["Cata", "effect_sign"])
        assert aes.loc["Cata", "standardized_statistic"] ** 2 == pytest.approx(
            aes.loc["Cata", "F"]
        )

    def test_dummy_rows_map_to_parent_omnibus_term(self):
        design, factors = _categorical_fixture()
        analysis = ANOVAAnalysis(
            design=design, response=design["Yield"].values,
            factors=factors, response_name="Yield",
        )
        results = analysis.fit(["Cata", "X"], enforce_hierarchy_flag=False)
        cs = results.coefficient_significance
        dummies = cs[cs["parent_anova_term"] == "Cata"]
        assert len(dummies) == 3
        assert dummies["parent_anova_df"].eq(3).all()
        assert dummies["coefficient_name"].str.contains("Cata").all()

    def test_anova_p_not_spread_to_dummy_rows(self):
        """The multi-Df omnibus p never overwrites per-dummy fitted p."""
        design, factors = _categorical_fixture()
        analysis = ANOVAAnalysis(
            design=design, response=design["Yield"].values,
            factors=factors, response_name="Yield",
        )
        results = analysis.fit(["Cata", "X"], enforce_hierarchy_flag=False)
        omnibus_p = results.anova_table.loc["Cata", "PR(>F)"]
        cs = results.coefficient_significance
        dummy_p = cs.loc[cs["parent_anova_term"] == "Cata", "p_value"].values
        assert (omnibus_p != dummy_p).all()


# ---------------------------------------------------------------------------
# View separation + rendering
# ---------------------------------------------------------------------------

class TestViewsAndPlots:
    def test_coefficient_and_anova_views_distinct_sources(self):
        """Each view carries its own source label; ANOVA never feeds the coeff chart."""
        design, factors = _categorical_fixture()
        analysis = ANOVAAnalysis(
            design=design, response=design["Yield"].values,
            factors=factors, response_name="Yield",
        )
        results = analysis.fit(["Cata", "X"], enforce_hierarchy_flag=False)
        assert (
            results.coefficient_significance["source"]
            == "Fitted-model coefficient test"
        ).all()
        assert (
            results.anova_effect_summary["source"]
            == "displayed term-level ANOVA table"
        ).all()

    def test_block_toggle_is_display_only(self):
        results, _ = _fit_blocked()
        total_bars = lambda fig: sum(len(t.y) for t in fig.data)
        full = create_coefficient_significance_plot(results.coefficient_significance)
        filtered = create_coefficient_significance_plot(
            results.coefficient_significance, show_block=False
        )
        assert total_bars(full) > total_bars(filtered)
        full_anova = create_standardized_effects_plot(results.anova_effect_summary)
        filtered_anova = create_standardized_effects_plot(
            results.anova_effect_summary, show_block=False
        )
        assert total_bars(full_anova) > total_bars(filtered_anova)

    def test_uniform_residual_df_gets_critical_lines(self):
        results, _ = _fit_blocked()
        fig = create_standardized_effects_plot(results.anova_effect_summary)
        n_vlines = sum(
            1 for s in fig.layout.shapes if s.type == "line" and s.xref == "x"
        )
        assert n_vlines >= 2

    def test_split_plot_no_universal_critical_line(self):
        """Split-plot terms use different error strata -> no single critical line."""
        analysis = ANOVAAnalysis(
            design=pd.read_csv(DATA_DIR / "test_case_5_split_plot.csv", comment='#'),
            response=pd.read_csv(
                DATA_DIR / "test_case_5_split_plot.csv", comment='#'
            )["Yield"].values,
            factors=[
                Factor("Temperature", FactorType.CONTINUOUS, ChangeabilityLevel.HARD,
                       levels=[125, 175]),
                Factor("Pressure", FactorType.CONTINUOUS, ChangeabilityLevel.HARD,
                       levels=[25, 75]),
                Factor("Time", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                       levels=[0, 20]),
                Factor("Catalyst", FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
                       levels=["A", "B"]),
            ],
            response_name="Yield",
        )
        results = analysis.fit(
            ["Temperature", "Pressure", "Time", "Catalyst"],
            enforce_hierarchy_flag=False,
        )
        assert results.anova_effect_summary is not None
        assert results.coefficient_significance is not None
        assert "Actual_Coefficient" in results.effect_estimates.columns
        assert "Actual_Std_Error" in results.effect_estimates.columns
        # Whole-plot and sub-plot terms must carry their own residual DF.
        wp_df = results.anova_effect_summary.loc["Temperature", "residual_df"]
        sp_df = results.anova_effect_summary.loc["Time", "residual_df"]
        assert wp_df != sp_df

    def test_figures_render_for_split_plot(self):
        analysis = ANOVAAnalysis(
            design=pd.read_csv(DATA_DIR / "test_case_5_split_plot.csv", comment='#'),
            response=pd.read_csv(
                DATA_DIR / "test_case_5_split_plot.csv", comment='#'
            )["Yield"].values,
            factors=[
                Factor("Temperature", FactorType.CONTINUOUS, ChangeabilityLevel.HARD,
                       levels=[125, 175]),
                Factor("Pressure", FactorType.CONTINUOUS, ChangeabilityLevel.HARD,
                       levels=[25, 75]),
                Factor("Time", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                       levels=[0, 20]),
                Factor("Catalyst", FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
                       levels=["A", "B"]),
            ],
            response_name="Yield",
        )
        results = analysis.fit(
            ["Temperature", "Pressure", "Time", "Catalyst"],
            enforce_hierarchy_flag=False,
        )
        assert create_coefficient_significance_plot(
            results.coefficient_significance
        ) is not None
        assert create_standardized_effects_plot(
            results.anova_effect_summary
        ) is not None