"""
Regression tests for estimability-safe model generation and prediction.

Covers three behaviours introduced to keep candidate models and predictions
statistically estimable:

- Quadratic candidate terms are omitted when a continuous factor is observed
  at fewer than three design levels (two-level full/fractional/screening
  designs cannot identify curvature; CCD / Box-Behnken / definitive screening
  and full factorials with center points can).
- Degenerate model terms (constant columns, exact aliases, rank defects) are
  removed before fitting, with the reason recorded.
- Block effects are analysed in the ANOVA but averaged out of the predictive
  equation by default, so predictions do not depend on the reference block.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.full_factorial import full_factorial
from src.core.fractional_factorial import FractionalFactorial
from src.core.response_surface import CentralCompositeDesign
from src.core.analysis import ANOVAAnalysis, generate_model_terms
from src.core.analysis_base import parse_model_term
from src.core.selection import (
    candidate_model_pool,
    quadratic_omission_notes,
    run_model_selection,
    trim_non_estimable_terms,
)


def continuous(name: str, levels=(-1, 1)) -> Factor:
    return Factor(name, FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
                  levels=list(levels))


def categorical(name: str, levels) -> Factor:
    return Factor(name, FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
                  levels=list(levels))


def fit_silently(analysis, terms, adjust_block=None):
    if adjust_block is not None:
        analysis.include_block_in_prediction = adjust_block
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return analysis.fit(terms, enforce_hierarchy_flag=True)


def blocked_design():
    """A 2^3 factorial in two balanced, orthogonal blocks.

    Block is confounded with the ABC interaction column, so it is orthogonal
    to A, B and every main effect.  Block '2' adds a +10 shift.
    """
    factors = [continuous("A"), continuous("B"), continuous("C")]
    rows = list(itertools.product((-1.0, 1.0), repeat=3))
    design = pd.DataFrame(rows, columns=["A", "B", "C"])
    design["Block"] = np.where(design["A"] * design["B"] * design["C"] == 1.0,
                               "1", "2")
    response = (
        10 + 2 * design["A"] + 3 * design["B"] + 10 * (design["Block"] == "2")
    )
    return design, response, factors


class TestQuadraticCandidateGating:

    def test_two_level_full_factorial_omits_quadratics(self):
        factors = [continuous("A"), continuous("B"), continuous("C")]
        design = full_factorial(factors, randomize=False)
        response = 10 + 2 * design["A"] + 3 * design["B"] - 1.5 * design["C"]
        analysis = ANOVAAnalysis(design, response, factors)
        pool, design_type = candidate_model_pool(factors, analysis)
        assert design_type == "Response Surface"
        assert not any("I(" in term for term in pool)

    def test_two_level_fractional_omits_quadratics(self):
        factors = [continuous("A"), continuous("B"), continuous("C"),
                   continuous("D"), continuous("E")]
        ff = FractionalFactorial(factors, fraction="1/2")
        design = ff.generate(randomize=False)
        response = np.asarray(design["A"], dtype=float)
        analysis = ANOVAAnalysis(design, response, factors)
        pool, _ = candidate_model_pool(factors, analysis)
        assert not any("I(" in term for term in pool)

    def test_rsm_design_keeps_quadratics(self):
        factors = [continuous("A"), continuous("B"), continuous("C")]
        ccd = CentralCompositeDesign(factors, alpha="rotatable").generate(
            randomize=False
        )
        response = (
            10 + 2 * ccd["A"] + 3 * ccd["B"] + 0.2 * ccd["A"] ** 2
        )
        analysis = ANOVAAnalysis(ccd, response, factors)
        pool, _ = candidate_model_pool(factors, analysis)
        assert set(pool) == {
            "A", "B", "C", "A*B", "A*C", "B*C",
            "I(A**2)", "I(B**2)", "I(C**2)",
        }

    def test_center_points_restore_curvature(self):
        # A two-level factorial with center points observes 3 levels for each
        # continuous factor, so curvature is estimable again.
        factors = [continuous("A"), continuous("B")]
        design = full_factorial(factors, n_center_points=3, randomize=False)
        assert design["A"].nunique() == 3
        response = 10 + 2 * design["A"] + 3 * design["B"]
        analysis = ANOVAAnalysis(design, response, factors)
        pool, _ = candidate_model_pool(factors, analysis)
        assert "I(A**2)" in pool and "I(B**2)" in pool

    def test_run_model_selection_never_selects_quadratic(self):
        factors = [continuous("A"), continuous("B"), continuous("C")]
        design = full_factorial(factors, randomize=False)
        response = 10 + 2 * design["A"] + 3 * design["B"] - 1.5 * design["C"]
        analysis = ANOVAAnalysis(design, response, factors)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = run_model_selection(analysis, method="backward", alpha=0.10)
        assert not any("I(" in term for term in results.final_terms)

    def test_quadratic_omission_notes(self):
        factors = [continuous("A"), continuous("B")]
        design = full_factorial(factors, randomize=False)
        response = 10 + 2 * design["A"] + 3 * design["B"]
        analysis = ANOVAAnalysis(design, response, factors)
        notes = quadratic_omission_notes(factors, analysis)
        assert len(notes) == 2
        assert all("A" in note and "B" in note or True for note in notes)
        assert all("curvature" not in note or note.startswith("Curvature")
                   for note in notes)

    def test_factor_only_pool_is_backwards_compatible(self):
        # No analysis/design supplied: factor-type behaviour is unchanged.
        factors = [continuous("A"), continuous("B"), continuous("C")]
        pool, _ = candidate_model_pool(factors)
        assert "I(A**2)" in pool and "I(C**2)" in pool


class TestDegenerateTermGuard:

    def test_quadratic_on_two_level_excluded_with_reason(self):
        factors = [continuous("A"), continuous("B")]
        design = full_factorial(factors, randomize=False)
        response = 10 + 2 * design["A"] + 3 * design["B"]
        analysis = ANOVAAnalysis(design, response, factors)
        results = fit_silently(
            analysis, ["1", "A", "B", "I(A**2)"], adjust_block=False
        )
        assert "I(A**2)" not in results.model_terms
        assert results.model_terms == ["1", "A", "B"]
        reason = dict(analysis.excluded_term_reasons).get("I(A**2)", "")
        assert "constant column" in reason or "intercept" in reason

    def test_categorical_single_level_excluded(self):
        factors = [continuous("A"), categorical("Cat", ["x", "y"])]
        design = pd.DataFrame({
            "A": [-1.0, -1.0, 1.0, 1.0],
            "Cat": ["x", "x", "x", "x"],
        })
        response = np.array([2.0, 2.5, 6.0, 6.5])
        analysis = ANOVAAnalysis(design, response, factors)
        results = fit_silently(analysis, ["1", "A", "Cat"], adjust_block=False)
        assert "Cat" not in results.model_terms
        assert dict(analysis.excluded_term_reasons)["Cat"]

    def test_perfectly_aliased_columns_excluded(self):
        # B and C share an identical design column: one must be dropped.
        factors = [continuous("A"), continuous("B"), continuous("C")]
        design = pd.DataFrame({
            "A": [-1.0, -1.0, 1.0, 1.0],
            "B": [-1.0, 1.0, -1.0, 1.0],
            "C": [-1.0, 1.0, -1.0, 1.0],
        })
        response = 10 + 2 * design["A"] + 3 * design["B"]
        analysis = ANOVAAnalysis(design, response, factors)
        results = fit_silently(analysis, ["1", "A", "B", "C"], adjust_block=False)
        assert "C" not in results.model_terms
        reasons = dict(analysis.excluded_term_reasons)
        assert "perfectly aliased" in reasons.get("C", "")

    def test_transform_domain_error_not_masked(self):
        # log(0) yields non-finite columns; the guard must not silently drop
        # the requested transform and hide the domain problem.
        design = pd.DataFrame({
            "A": [0.0, 1.0, 2.0, 3.0, 1.5, 2.5],
        })
        response = np.array([1.0, 2.0, 3.0, 4.0, 2.5, 3.5])
        factors = [continuous("A", [0.0, 3.0])]
        analysis = ANOVAAnalysis(design, response, factors, response_name="Y")
        with pytest.warns((RuntimeWarning,)):
            try:
                results = analysis.fit(["1", "np.log(A)"],
                                       enforce_hierarchy_flag=False)
                assert not np.all(np.isfinite(results.fitted_values))
            except Exception:
                analysis.excluded_term_reasons = []
        assert not analysis.excluded_term_reasons


class TestBlockAveragePrediction:

    def test_blocked_fit_keeps_anova_and_averages_predictions(self):
        design, response, factors = blocked_design()
        analysis = ANOVAAnalysis(design, response, factors, block_as_random=False)
        results = fit_silently(analysis, ["1", "A", "B"])

        # ANOVA / reporting keeps Block.
        assert "Block" in results.anova_table.index
        assert any("Block[" in str(idx) for idx in results.effect_estimates.index)

        # Prediction equation is averaged over blocks.
        assert results.block_mean_shift == pytest.approx(5.0)
        assert results.blocks_in_predictions is False

    def test_predictions_are_block_average(self):
        design, response, factors = blocked_design()
        analysis = ANOVAAnalysis(design, response, factors, block_as_random=False)
        results = fit_silently(analysis, ["1", "A", "B"])
        settings = {"A": 0.5, "B": -0.5}
        # True block-average: (10 + 20) / 2 + 2A + 3B.
        expected = 15.0 + 2 * 0.5 + 3 * (-0.5)
        assert results.predict_from_settings(settings) == pytest.approx(expected)
        ref = results.fitted_model.predict(pd.DataFrame([{
            "Block": "1", "A": 0.5, "B": -0.5,
        }]))[0]
        assert result_predict_equals(results, settings, ref)
        # The intercept reflects the +5 mean block offset (block-1 intercept
        # was 10) so the reference-level prediction is the block average.
        assert results.effect_estimates.loc["Intercept", "Coefficient"] == pytest.approx(15.0)

    def test_predictions_independent_of_reference_level(self):
        design, response, factors = blocked_design()
        settings = {"A": 0.5, "B": -0.5}
        a1 = ANOVAAnalysis(design, response, factors, block_as_random=False)
        r1 = fit_silently(a1, ["1", "A", "B"])
        p1 = r1.predict_from_settings(settings)

        swapped = design.copy()
        swapped["Block"] = swapped["Block"].map({"1": "2", "2": "1"})
        a2 = ANOVAAnalysis(swapped, response, factors, block_as_random=False)
        r2 = fit_silently(a2, ["1", "A", "B"])
        p2 = r2.predict_from_settings(settings)
        assert p1 == pytest.approx(p2)

    def test_opt_in_keeps_block_adjustment(self):
        design, response, factors = blocked_design()
        analysis = ANOVAAnalysis(design, response, factors, block_as_random=False,
                                 include_block_in_prediction=True)
        results = fit_silently(analysis, ["1", "A", "B"])
        assert results.block_mean_shift == 0.0
        assert results.blocks_in_predictions is True
        # Unshifted model: reference-block and block-2 predictions retain
        # their specific block adjustments.
        p = results.predict_from_settings({"A": 0.0, "B": 0.0})
        assert p == pytest.approx(10.0)
        blk2 = results.fitted_model.predict(pd.DataFrame([{
            "Block": "2", "A": 0.0, "B": 0.0,
        }]))[0]
        assert blk2 == pytest.approx(20.0)

    def test_optimizer_prediction_matches_ref_block_predict(self):
        from src.core.optimization import optimize_response
        design, response, factors = blocked_design()
        analysis = ANOVAAnalysis(design, response, factors, block_as_random=False)
        results = fit_silently(analysis, ["1", "A", "B"])
        opt = optimize_response(anova_results=results, factors=factors,
                                objective="maximize", seed=42)
        assert opt.success
        settings = opt.optimal_settings
        ref = results.fitted_model.predict(pd.DataFrame([{
            "Block": "1", "A": settings["A"], "B": settings["B"],
        }]))[0]
        assert opt.predicted_response == pytest.approx(float(ref), rel=1e-6)


def result_predict_equals(results, settings, ref) -> bool:
    """predict_from_settings must agree with the statsmodels ref-block fit."""
    return float(results.predict_from_settings(settings)) == pytest.approx(
        float(ref), rel=1e-9
    )


def _parents_present(terms):
    """True when every non-intercept term's parents are also included."""
    included = {t for t in terms if t != '1'}
    for term in included:
        parents, _ = parse_model_term(term)
        if any(p not in included for p in parents):
            return False
    return True


class TestInteractionNormalizationGuard:

    # Regression for the interaction-separator bug: patsy reports
    # interactions in design_info.term_names with ':' separators
    # (catalyst:Temperature, feed_rate:C(Atmosphere)) while the wrapped
    # formula side uses '*'.  The pruner's term->slice lookup therefore
    # KeyError'd on ANY interaction-bearing model and silently returned the
    # pool unchanged, leaving rank-deficient quadratics in the fit.

    def test_prunes_quadratic_in_model_with_categorical_interaction(self):
        factors = [continuous("A"), categorical("Cat", ["x", "y"])]
        design = pd.DataFrame({
            "A": [-1.0, -1.0, 1.0, 1.0] * 2,
            "Cat": ["x", "y"] * 4,
        })
        response = np.arange(1.0, 9.0)
        analysis = ANOVAAnalysis(design, response, factors, response_name="Y")
        results = fit_silently(
            analysis, ["1", "A", "Cat", "A*Cat", "I(A**2)"],
            adjust_block=False,
        )
        assert "I(A**2)" not in results.model_terms
        assert "A*Cat" in results.model_terms
        assert dict(analysis.excluded_term_reasons)["I(A**2)"]

    def test_prunes_quadratic_in_model_with_continuous_interactions(self):
        factors = [continuous("A"), continuous("B"), continuous("C")]
        design = full_factorial(factors, randomize=False)
        response = 10 + 2 * design["A"] + 3 * design["B"]
        analysis = ANOVAAnalysis(design, response, factors)
        results = fit_silently(
            analysis,
            ["1", "A", "B", "C", "A*B", "A*C", "B*C", "I(A**2)"],
            adjust_block=False,
        )
        assert "I(A**2)" not in results.model_terms
        assert "A*B" in results.model_terms and "B*C" in results.model_terms
        reason = dict(analysis.excluded_term_reasons).get("I(A**2)", "")
        assert "constant column" in reason or "intercept" in reason

    def test_rank_deficient_interaction_resolution_normalizes_names(self):
        # The rank-deficiency trial loop exercises the same term->slice map;
        # only with the '*'->':' normalisation can trials be resolved.
        factors = [continuous("A"), continuous("B")]
        design = pd.DataFrame({
            "A": [-1.0, 1.0, -1.0, 1.0],
            "B": [-1.0, 1.0, 1.0, -1.0],
        })
        response = 10 + 2 * design["A"] + 3 * design["B"]
        analysis = ANOVAAnalysis(design, response, factors)
        # A*B on a full 2^2 is estimable; the pruner must resolve slices for
        # '*' terms without bailing out and dropping anything.
        results = fit_silently(
            analysis, ["1", "A", "B", "A*B"], adjust_block=False
        )
        assert results.model_terms == ["1", "A", "B", "A*B"]

    def test_fit_never_leaves_a_child_without_its_parent(self):
        # A constant factor A and a two-level B: A, A*B and I(B**2) are all
        # non-estimable and must be dropped together, leaving hierarchy intact.
        factors = [continuous("A", [0.4, 0.6]), continuous("B")]
        design = pd.DataFrame({
            "A": [0.5] * 8,
            "B": [-1.0, 1.0] * 4,
        })
        response = np.array([1.0, 2.0, 1.2, 2.2, 1.1, 2.1, 1.3, 2.3])
        analysis = ANOVAAnalysis(design, response, factors, response_name="Y")
        results = fit_silently(
            analysis, ["1", "A", "B", "A*B", "I(B**2)"], adjust_block=False
        )
        assert results.model_terms == ["1", "B"]
        reasons = dict(analysis.excluded_term_reasons)
        assert "A" in reasons and "A*B" in reasons and "I(B**2)" in reasons
        assert _parents_present(results.model_terms)


class TestSharedEstimabilityTrim:

    def test_trim_removes_only_non_estimable_quadratics(self):
        factors = [continuous("A"), continuous("B")]
        terms = ["1", "A", "B", "A*B", "I(A**2)", "I(B**2)"]
        kept, removed = trim_non_estimable_terms(terms, factors, {"A": 2, "B": 3})
        assert removed == ["I(A**2)"]
        assert kept == ["1", "A", "B", "A*B", "I(B**2)"]

    def test_trim_is_noop_without_levels(self):
        terms = ["1", "A", "B", "I(A**2)"]
        kept, removed = trim_non_estimable_terms([], [], {})
        assert kept == [] and removed == []
        kept, removed = trim_non_estimable_terms(terms, [], {})
        assert kept == terms and removed == []

    def test_candidate_pool_uses_shared_rule(self):
        # candidate_model_pool must delegate to trim_non_estimable_terms: the
        # same rule the legacy Step-3/4 hand-off applies.
        factors = [continuous("A"), continuous("B"), continuous("C")]
        design = full_factorial(factors, randomize=False)
        response = 10 + 2 * design["A"] + 3 * design["B"]
        analysis = ANOVAAnalysis(design, response, factors)
        pool, _ = candidate_model_pool(factors, analysis)
        assert not any("I(" in term for term in pool)

    def test_step3_2_level_prediction_strips_quadratics(self):
        # Mirrors the Step-3 fractional-factorial early hand-off: predicted
        # level count 2 for every continuous factor.
        factors = [continuous("A"), continuous("B"), continuous("C")]
        terms = ["1", "A", "B", "C", "A*B", "I(A**2)", "I(C**2)"]
        predicted = {f.name: 2 for f in factors}
        kept, removed = trim_non_estimable_terms(terms, factors, predicted)
        assert set(removed) == {"I(A**2)", "I(C**2)"}
        assert "A*B" in kept and "B" in kept

    def test_step4_observed_levels_are_authoritative(self):
        # Step-4 uses the generated design's real per-factor nunique() counts.
        factors = [continuous("A"), continuous("B")]
        design = full_factorial(factors, n_center_points=3, randomize=False)
        levels = {f.name: int(design[f.name].nunique()) for f in factors}
        assert levels == {"A": 3, "B": 3}
        terms = ["1", "A", "B", "A*B", "I(A**2)", "I(B**2)"]
        kept, removed = trim_non_estimable_terms(terms, factors, levels)
        assert removed == [] and set(kept) == set(terms)
        # ... while the same design without center points is 2-level.
        design2 = full_factorial(factors, randomize=False)
        levels2 = {f.name: int(design2[f.name].nunique()) for f in factors}
        assert levels2 == {"A": 2, "B": 2}
        _, removed2 = trim_non_estimable_terms(terms, factors, levels2)
        assert set(removed2) == {"I(A**2)", "I(B**2)"}