"""
Tests for Automatic Model Selection (p-value based, strong heredity).

Covers forward / backward / stepwise selection, hierarchy and parent
protection, mixed continuous + categorical designs, term-table reporting and
model-quality diagnostics.
"""
import warnings

import numpy as np
import pandas as pd
import pytest

from src.core.factors import Factor, FactorType
from src.core.analysis import ANOVAAnalysis, generate_model_terms
from src.core.analysis_base import enforce_hierarchy, parse_model_term
from src.core.selection import (
    LOF_UNAVAILABLE_TEXT,
    ModelSelectionResults,
    SelectionStep,
    backward_elimination,
    candidate_model_pool,
    classify_model_type,
    compute_model_quality,
    compute_press,
    factor_display_map,
    format_selection_summary,
    forward_selection,
    friendly_factor_name,
    map_term_display,
    near_perfect_fit_message,
    predicted_r2_warning,
    required_by_hierarchy,
    run_model_selection,
    stepwise_selection,
    term_p_value,
)

ALLOWED_REASONS = {
    'Significant',
    'Not significant',
    'Kept for hierarchy',
    'Removed by backward elimination',
    'Added during forward selection',
    'Added during stepwise selection',
    'Excluded (model saturated)',
    'Added during stepwise (BIC) selection',
    'Removed during stepwise (BIC) selection',
    'Selected by BIC stepwise',
    'Not selected',
}


def continuous(name: str) -> Factor:
    return Factor(name, FactorType.CONTINUOUS, levels=[-1, 1])


def categorical(name: str, levels) -> Factor:
    return Factor(name, FactorType.CATEGORICAL, levels=list(levels))


def run_silently(analysis, method='forward', alpha=0.10, bic_threshold=2.0):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return run_model_selection(
            analysis, method=method, alpha=alpha, bic_threshold=bic_threshold
        )


def make_analysis(design, y, factors):
    return ANOVAAnalysis(design, y, factors)


def ccd_dataset(seed=7, n=60):
    """3-factor dataset: A, B, A*B and A² active; C is pure noise."""
    rng = np.random.default_rng(seed)
    A = np.array([-1.0, 1.0] * int(np.ceil(n / 2)))[:n] + rng.normal(0, 0.05, n)
    B = rng.uniform(-1, 1, n)
    C = rng.uniform(-1, 1, n)
    y = 2 + 0.4 * A + 1.5 * B + 0.8 * A * B + 0.5 * A ** 2 + rng.normal(0, 0.6, n)
    factors = [continuous('A'), continuous('B'), continuous('C')]
    return pd.DataFrame({'A': A, 'B': B, 'C': C}), y, factors


def proxy_dataset(seed=14, n=80):
    """D is a noisy copy of B; stepwise should add D then remove it."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, n)
    B = rng.uniform(-1, 1, n)
    C = rng.uniform(-1, 1, n)
    D = B + rng.normal(0, 0.15, n)
    y = 1 + 2 * A + 2 * B + rng.normal(0, 0.6, n)
    factors = [continuous('A'), continuous('B'), continuous('C'), continuous('D')]
    return pd.DataFrame({'A': A, 'B': B, 'C': C, 'D': D}), y, factors


def mixed_dataset(seed=3, n=90):
    """Continuous A, B plus a 3-level categorical 'Cat'."""
    rng = np.random.default_rng(seed)
    A = rng.uniform(-1, 1, n)
    Cat = rng.choice(['lo', 'mid', 'hi'], n, p=[0.34, 0.33, 0.33])
    B = rng.uniform(-1, 1, n)
    y = (
        2 + 0.5 * A + (np.array(Cat) == 'hi').astype(float) * 1.5
        + 0.4 * A * (np.array(Cat) == 'mid').astype(float)
        + rng.normal(0, 0.5, n)
    )
    factors = [continuous('A'), categorical('Cat', ['lo', 'mid', 'hi']), continuous('B')]
    return pd.DataFrame({'A': A, 'Cat': Cat, 'B': B}), y, factors


def categorical_only_dataset(seed=11, n=120):
    """3 categorical factors; CatA and the CatA*CatB interaction are active."""
    rng = np.random.default_rng(seed)
    CatA = rng.choice(['x', 'y', 'z'], n, p=[0.33, 0.33, 0.34])
    CatB = rng.choice(['lo', 'hi'], n, p=[0.5, 0.5])
    CatC = rng.choice(['p', 'q'], n, p=[0.5, 0.5])
    coef = {
        ('x', 'lo'): -0.5, ('x', 'hi'): 0.5,
        ('y', 'lo'): 1.0, ('y', 'hi'): -1.0,
        ('z', 'lo'): 0.2, ('z', 'hi'): 0.8,
    }
    eta = np.array([coef[(a, b)] for a, b in zip(CatA, CatB)])
    y = 2.0 + eta + rng.normal(0, 0.4, n)
    factors = [
        categorical('CatA', ['x', 'y', 'z']),
        categorical('CatB', ['lo', 'hi']),
        categorical('CatC', ['p', 'q']),
    ]
    return pd.DataFrame({'CatA': CatA, 'CatB': CatB, 'CatC': CatC}), y, factors


class TestForwardSelection:

    def test_discovers_known_effects(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward')
        final = set(result.final_terms)
        assert {'A', 'B', 'A*B'} <= final
        assert 'C' not in final
        assert '1' in final

    def test_final_model_is_hierarchical(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward')
        factor_names = [f.name for f in factors]
        enforced, added = enforce_hierarchy(result.final_terms, factor_names)
        assert len(added) == 0
        assert set(enforced) == set(result.final_terms)

    def test_hierarchy_kept_on_every_add(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward')
        present = {'1'}
        for step in result.selection_path:
            if step.action == 'add':
                factor_list, op = parse_model_term(step.term)
                if op in ('*', '**'):
                    assert all(f in present for f in factor_list), step.term
            present = set(step.current_terms)

    def test_first_added_term_is_a_main_effect(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward')
        assert result.selection_path
        first = result.selection_path[0]
        assert first.action == 'add'
        _, op = parse_model_term(first.term)
        assert op == ''

    def test_borderline_term_respects_alpha(self):
        # A*C (full-model p ≈ 0.082) and I(A**2) (≈ 0.096) straddle
        # alpha = 0.05 / 0.10, so they must appear only at the looser alpha.
        rng = np.random.default_rng(17)
        n = 60
        A = rng.uniform(-1, 1, n)
        B = rng.uniform(-1, 1, n)
        C = rng.uniform(-1, 1, n)
        y = 2 + 0.6 * A + 1.2 * B + 0.25 * A * B + 0.2 * A ** 2 + rng.normal(0, 1.0, n)
        factors = [continuous('A'), continuous('B'), continuous('C')]
        analysis = make_analysis(pd.DataFrame({'A': A, 'B': B, 'C': C}), y, factors)

        strict = run_silently(analysis, method='forward', alpha=0.05)
        relaxed = run_silently(analysis, method='forward', alpha=0.10)

        strict_set = set(strict.final_terms)
        relaxed_set = set(relaxed.final_terms)
        assert strict_set < relaxed_set
        assert {'A*C', 'I(A**2)'} <= relaxed_set
        assert 'A*C' not in strict_set and 'I(A**2)' not in strict_set
        # C is only present as the parent of the significant A*C at alpha 0.10
        assert 'C' not in strict_set and 'C' in relaxed_set

    def test_started_with_intercept_only(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward')
        assert result.started_with_intercept_only is True
        assert result.model_quality['model_p_value'] is not None

    def test_invalid_method_rejected(self):
        design, y, factors = ccd_dataset()
        analysis = make_analysis(design, y, factors)
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                run_model_selection(analysis, method='branch-and-bound')


class TestBackwardElimination:

    def test_removes_noise_and_keeps_active(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='backward')
        final = set(result.final_terms)
        assert {'A', 'B', 'A*B'} <= final
        assert 'C' not in final
        assert '1' in final
        assert set(enforce_hierarchy(result.final_terms, [f.name for f in factors])[0]) == final

    def test_dependents_removed_before_parents(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='backward')
        assert result.selection_path
        # Dependents (interactions / quadratics) must be queued up and removed
        # before their parent mains.
        first = result.selection_path[0]
        assert first.action == 'remove'
        _, op = parse_model_term(first.term)
        assert op in ('*', '**')
        for step in result.selection_path:
            if step.action == 'remove':
                factor_list, _ = parse_model_term(step.term)
                for f in factor_list:
                    assert f in set(step.current_terms) or f == step.term

    def test_parent_protection(self):
        rng = np.random.default_rng(42)
        n = 100
        A = rng.uniform(-1, 1, n)
        B = rng.uniform(-1, 1, n)
        y = 0.05 * A + 0.05 * B + 0.8 * A * B + rng.normal(0, 0.5, n)
        factors = [continuous('A'), continuous('B')]
        analysis = make_analysis(pd.DataFrame({'A': A, 'B': B}), y, factors)

        result = run_silently(analysis, method='backward', alpha=0.10)
        final = set(result.final_terms)
        assert {'A', 'B', 'A*B'} <= final

        table = result.term_table.set_index('Term')
        # The weak parent (B) must be kept for heredity even though p > alpha.
        assert table.loc['A*B', 'Reason'] == 'Significant'
        assert table.loc['B', 'Reason'] == 'Kept for hierarchy'
        reasons = set(table.loc[['A', 'B'], 'Reason'])
        assert 'Kept for hierarchy' in reasons


class TestStepwiseSelection:

    def test_converges_to_full_model_significant_set(self):
        design, y, factors = proxy_dataset()
        finals = {}
        for method in ('forward', 'backward', 'stepwise'):
            result = run_silently(make_analysis(design, y, factors), method=method, alpha=0.10)
            finals[method] = set(result.final_terms)
        assert finals['forward'] == finals['backward'] == finals['stepwise']
        final = finals['forward']
        assert {'A', 'B', 'B*D', 'I(A**2)', 'I(B**2)'} <= final
        assert 'C' not in final
        assert '1' in final

    def test_parent_protection_keeps_nonsignificant_parent(self):
        # D (full-model p ≈ 0.77) is retained only as the parent of the
        # significant B*D interaction.
        design, y, factors = proxy_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward', alpha=0.10)
        table = result.term_table.set_index('Term')
        assert 'D' in set(result.final_terms)
        assert table.loc['D', 'Reason'] == 'Kept for hierarchy'
        assert table.loc['B*D', 'Reason'] == 'Significant'

    def test_no_midrun_removals_under_full_model_criterion(self):
        design, y, factors = proxy_dataset()
        result = run_silently(make_analysis(design, y, factors), method='stepwise', alpha=0.10)
        assert all(s.action == 'add' for s in result.selection_path)

    def test_final_terms_hierarchical(self):
        design, y, factors = proxy_dataset()
        result = run_silently(make_analysis(design, y, factors), method='stepwise', alpha=0.10)
        enforced, added = enforce_hierarchy(result.final_terms, [f.name for f in factors])
        assert added == []
        assert set(enforced) == set(result.final_terms)


class TestMixedDesigns:

    def test_pool_has_no_categorical_quadratic(self):
        factors = [continuous('A'), categorical('Cat', ['lo', 'mid', 'hi'])]
        pool = generate_model_terms(factors, 'quadratic')
        assert 'I(Cat**2)' not in pool
        assert 'Cat' in pool and 'A*Cat' in pool
        assert 'I(A**2)' in pool

    def test_categorical_term_selected_via_anova_pvalue(self):
        design, y, factors = mixed_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward')
        assert 'Cat' in set(result.final_terms)
        table = result.term_table.set_index('Term')
        assert table.loc['Cat', 'p-value'] is not None
        # multi-DF terms have no representative coefficient
        assert pd.isna(table.loc['Cat', 'Coefficient'])

    def test_interaction_with_categorical_coefficient_marked_missing(self):
        design, y, factors = mixed_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward')
        table = result.term_table.set_index('Term')
        assert pd.isna(table.loc['A*Cat', 'Coefficient'])
        assert table.loc['A*Cat', 'p-value'] is not None


class TestDesignAwarePool:

    def test_pool_response_surface(self):
        factors = [continuous('A'), continuous('B'), continuous('C')]
        pool, design_type = candidate_model_pool(factors)
        assert design_type == 'Response Surface'
        assert set(pool) == {'A', 'B', 'C', 'A*B', 'A*C', 'B*C',
                             'I(A**2)', 'I(B**2)', 'I(C**2)'}

    def test_pool_categorical_factorial(self):
        factors = [categorical('CatA', ['x', 'y', 'z']),
                   categorical('CatB', ['lo', 'hi'])]
        pool, design_type = candidate_model_pool(factors)
        assert design_type == 'Categorical Factorial'
        assert 'I(' not in ' '.join(pool)
        assert pool == ['CatA', 'CatB', 'CatA*CatB']

    def test_pool_mixed(self):
        factors = [continuous('A'), categorical('Cat', ['lo', 'hi'])]
        pool, design_type = candidate_model_pool(factors)
        assert design_type == 'Mixed'
        assert set(pool) == {'A', 'Cat', 'A*Cat', 'I(A**2)'}

    def test_pool_categorical_only_never_raises_quadratic_valueerror(self):
        factors = [categorical('CatA', ['x', 'y']), categorical('CatB', ['p', 'q'])]
        pool, design_type = candidate_model_pool(factors)
        assert design_type == 'Categorical Factorial'
        assert 'I(' not in ' '.join(pool)


class TestCategoricalOnlySelection:

    def test_backward_no_crash_and_hierarchical(self):
        design, y, factors = categorical_only_dataset()
        result = run_silently(make_analysis(design, y, factors), method='backward', alpha=0.10)
        assert '1' in result.final_terms
        assert all('I(' not in t for t in result.final_terms)
        enforced, added = enforce_hierarchy(result.final_terms, [f.name for f in factors])
        assert added == []

    def test_backward_discovers_active_interaction(self):
        design, y, factors = categorical_only_dataset()
        result = run_silently(make_analysis(design, y, factors), method='backward', alpha=0.10)
        final = set(result.final_terms)
        assert 'CatA' in final
        assert 'CatA*CatB' in final or 'CatB' in final

    def test_no_quadratic_terms_in_table(self):
        design, y, factors = categorical_only_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward', alpha=0.10)
        assert all('I(' not in t for t in result.term_table['Term'])
        assert set(result.term_table['Reason']) <= ALLOWED_REASONS

    def test_bic_selection_categorical_only(self):
        design, y, factors = categorical_only_dataset()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = run_model_selection(
                make_analysis(design, y, factors), method='stepwise_bic', alpha=0.10,
                bic_threshold=2.0,
            )
        assert '1' in result.final_terms
        assert all('I(' not in t for t in result.final_terms)
        enforced, added = enforce_hierarchy(result.final_terms, [f.name for f in factors])
        assert added == []


class TestBicAdapter:

    def test_runs_hierarchically(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='stepwise_bic', bic_threshold=2.0)
        assert result.method == 'stepwise_bic'
        assert '1' in result.final_terms
        enforced, added = enforce_hierarchy(result.final_terms, [f.name for f in factors])
        assert added == []

    def test_bic_threshold_stored(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='stepwise_bic', bic_threshold=3.0)
        assert result.bic_threshold == 3.0
        assert result.alpha == 0.10

    def test_steps_record_bic(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='stepwise_bic', bic_threshold=2.0)
        assert result.selection_path
        assert all(isinstance(s, SelectionStep) for s in result.selection_path)
        assert all(s.bic is not None for s in result.selection_path)
        assert all(s.delta_bic is not None for s in result.selection_path)
        numbers = [s.step_number for s in result.selection_path]
        assert numbers == sorted(numbers)

    def test_term_table_and_reasons(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='stepwise_bic', bic_threshold=2.0)
        table = result.term_table
        assert list(table.columns) == ['Term', 'Coefficient', 'p-value', 'Included', 'Reason']
        assert set(table['Reason']) <= ALLOWED_REASONS
        final_set = set(result.final_terms)
        for _, row in table.iterrows():
            assert (row['Included'] == 'Yes') == (row['Term'] in final_set)

    def test_summary_reports_bic_metadata_and_path(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='stepwise_bic', bic_threshold=2.0)
        summary = format_selection_summary(result, "Yield")
        assert '**Selection Method:** Stepwise (BIC)' in summary
        assert '**BIC Threshold:** 2' in summary
        assert '**Alpha:**' not in summary
        assert 'BIC =' in summary and 'ΔBIC =' in summary
        assert 'Final Model' in summary

    def test_bic_threshold_must_be_positive(self):
        design, y, factors = ccd_dataset()
        with pytest.raises(ValueError):
            run_model_selection(
                make_analysis(design, y, factors), method='stepwise_bic', alpha=0.10,
                bic_threshold=0.0,
            )


class TestReporting:

    def test_term_table_columns_and_flags(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward')
        table = result.term_table
        assert list(table.columns) == ['Term', 'Coefficient', 'p-value', 'Included', 'Reason']
        assert '1' not in set(table['Term'])
        pool_size = len([t for t in generate_model_terms(factors, 'quadratic') if t != '1'])
        assert len(table) == pool_size
        assert set(table['Included']) <= {'Yes', 'No'}
        assert set(table['Reason']) <= ALLOWED_REASONS
        final_set = set(result.final_terms)
        for _, row in table.iterrows():
            assert (row['Included'] == 'Yes') == (row['Term'] in final_set)

    def test_reasons_reflect_method(self):
        design, y, factors = ccd_dataset()
        fwd = run_silently(make_analysis(design, y, factors), method='forward')
        bwd = run_silently(make_analysis(design, y, factors), method='backward')
        assert 'Not significant' in set(fwd.term_table['Reason'])
        assert 'Removed by backward elimination' in set(bwd.term_table['Reason'])
        assert 'Not significant' not in set(bwd.term_table['Reason'])

    def test_format_selection_summary(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward')
        summary = format_selection_summary(result, "Yield")
        assert 'Automatic Model Selection Results' in summary
        assert '**Selection Method:** Forward' in summary
        assert '**Alpha:**' in summary
        assert 'Selection Path' in summary
        assert 'Final Model' in summary and 'Yield =' in summary
        assert 'Added' in summary

    def test_format_selection_summary_metadata(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='backward')
        pool = [t for t in generate_model_terms(factors, 'quadratic') if t != '1']
        summary = format_selection_summary(result, "Length", candidate_pool=pool)
        non_intercept = len([t for t in result.final_terms if t != '1'])
        assert f"**Selected Terms:** {non_intercept}" in summary
        assert "**Model Type:**" in summary
        assert "**Hierarchy Status:** ✓ Strong heredity satisfied" in summary
        assert classify_model_type(result.final_terms, pool) in {
            'Linear', 'Interaction', 'Reduced Quadratic', 'Full Quadratic',
            'Linear (intercept only)',
        }

    def test_format_selection_summary_friendly_names(self):
        factors = [continuous('A'), continuous('B'), continuous('f_C')]
        design, y, _ = ccd_dataset()
        design = design.rename(columns={'C': 'f_C'})
        result = run_silently(make_analysis(design, y, factors), method='backward')
        dmap = factor_display_map(factors)
        assert dmap['f_C'] == 'C'
        summary = format_selection_summary(result, "Length", factor_display_map=dmap)
        assert 'f_C' not in summary
        assert 'C' in summary


class TestEdgeCases:

    def test_oversaturated_no_crash(self):
        design = pd.DataFrame({'A': [-1.0, -0.5, 0.0, 0.5, 1.0],
                               'B': [1.0, -1.0, 1.0, -1.0, 1.0]})
        y = 2 + 0.5 * design['A'] + 0.3 * design['B']
        factors = [continuous('A'), continuous('B')]
        analysis = make_analysis(design, y, factors)
        for method in ('forward', 'backward'):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                result = run_model_selection(analysis, method=method, alpha=0.10)
            assert 'Excluded (model saturated)' in set(result.term_table['Reason'])
            assert 'I(A**2)' in set(result.term_table['Term'])
            table = result.term_table.set_index('Term')
            assert pd.isna(table.loc['I(A**2)', 'Coefficient'])
            assert pd.isna(table.loc['I(A**2)', 'p-value'])

    def test_backward_prefers_fit_able_start(self):
        design = pd.DataFrame({'A': [-1.0, -0.5, 0.0, 0.5, 1.0],
                               'B': [1.0, -1.0, 1.0, -1.0, 1.0]})
        y = 2 + 0.5 * design['A'] + 0.3 * design['B']
        factors = [continuous('A'), continuous('B')]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = run_model_selection(make_analysis(design, y, factors), method='backward', alpha=0.10)
        assert '1' in result.final_terms


class TestQualityDiagnostics:

    def test_press_and_predicted_r2(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward')
        press = compute_press(result.final_results)
        assert press is not None and press > 0
        q = result.model_quality
        assert q['press'] == pytest.approx(press)
        assert q['predicted_r_squared'] is not None
        assert q['predicted_r_squared'] <= 1.0
        assert q['r_squared'] is not None and 0 < q['r_squared'] <= 1.0
        assert q['adj_r_squared'] is not None
        assert q['model_p_value'] is not None and q['model_p_value'] < 0.05

    def test_lack_of_fit_with_replicates(self):
        rng = np.random.default_rng(1)
        u = rng.uniform(-1, 1, 15)
        v = rng.uniform(-1, 1, 15)
        A = np.repeat(u, 2)
        B = np.repeat(v, 2)
        y = 2 + 0.7 * A + 0.3 * B + rng.normal(0, 0.4, len(A))
        factors = [continuous('A'), continuous('B')]
        result = run_silently(make_analysis(pd.DataFrame({'A': A, 'B': B}), y, factors), method='forward')
        assert result.model_quality['lack_of_fit_p_value'] is not None
        assert isinstance(result.model_quality['lack_of_fit_p_value'], float)

    def test_lack_of_fit_absent_without_replicates(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward')
        assert result.model_quality['lack_of_fit_p_value'] is None

    def test_predicted_r2_warning(self):
        assert isinstance(predicted_r2_warning(0.90, 0.60), str)
        assert predicted_r2_warning(0.90, 0.85) is None
        assert predicted_r2_warning(None, 0.60) is None
        assert predicted_r2_warning(0.90, None) is None


class TestPresentationHelpers:

    def test_friendly_factor_name_unwraps_escapes(self):
        assert friendly_factor_name('f_C') == 'C'
        assert friendly_factor_name('f_for') == 'for'
        assert friendly_factor_name('f_2nd_run') == '2nd_run'
        assert friendly_factor_name('Temperature') == 'Temperature'
        assert friendly_factor_name('pressure') == 'pressure'
        assert friendly_factor_name('f_') == 'f_'

    def test_factor_display_map_prefers_original_names(self):
        factors = [continuous('f_C')]
        assert factor_display_map(factors) == {'f_C': 'C'}
        rename_map = {'Flow C': 'f_C'}
        assert factor_display_map(factors, rename_map) == {'f_C': 'Flow C'}

    def test_map_term_display_replaces_factor_tokens(self):
        dmap = {'f_C': 'C'}
        assert map_term_display('f_C', dmap) == 'C'
        assert map_term_display('f_C*f_C_B', dmap) == 'C*f_C_B'
        assert map_term_display('I(f_C**2)', dmap) == 'I(C**2)'
        assert map_term_display('np.log(f_C)', dmap) == 'np.log(C)'
        assert map_term_display('A*B', None) == 'A*B'

    def test_map_term_display_longest_first(self):
        dmap = {'A': 'Alpha', 'AB': 'Beta'}
        assert map_term_display('AB*C', dmap) == 'Beta*C'
        assert map_term_display('A*AB', dmap) == 'Alpha*Beta'

    def test_classify_model_type(self):
        pool = ['A', 'B', 'C', 'A*B', 'A*C', 'B*C', 'I(A**2)', 'I(B**2)', 'I(C**2)']
        assert classify_model_type(['1'], ['1'] + pool) == 'Linear (intercept only)'
        assert classify_model_type(['1', 'A', 'B'], ['1'] + pool) == 'Linear'
        assert classify_model_type(['1', 'A', 'B', 'A*B'], ['1'] + pool) == 'Interaction'
        reduced = ['1', 'A', 'B', 'A*B', 'I(A**2)']
        assert classify_model_type(reduced, ['1'] + pool) == 'Reduced Quadratic'
        assert classify_model_type(['1'] + pool, ['1'] + pool) == 'Full Quadratic'

    def test_classify_model_type_detects_all_quadratics(self):
        pool = ['A', 'B', 'I(A**2)', 'I(B**2)']
        # every pool term selected, in a different order, still full
        final = ['1', 'I(A**2)', 'A', 'B', 'I(B**2)']
        assert classify_model_type(final, ['1'] + pool) == 'Full Quadratic'

    def test_classify_model_type_full_interaction_pool(self):
        pool = ['CatA', 'CatB', 'CatA*CatB']
        final = ['1', 'CatA', 'CatB', 'CatA*CatB']
        assert classify_model_type(final, ['1'] + pool) == 'Full Interaction'
        reduced = ['1', 'CatA', 'CatB']
        assert classify_model_type(reduced, ['1'] + pool) == 'Linear'

    def test_near_perfect_fit_message_thresholds(self):
        assert near_perfect_fit_message(0.9990, 0.9991, 0.9992)
        assert near_perfect_fit_message(0.999, 0.999, 0.999)
        assert near_perfect_fit_message(0.9989, 0.999, 0.999) is None
        assert near_perfect_fit_message(0.999, 0.999, None) is None
        assert near_perfect_fit_message(None, 0.999, 0.999) is None
        assert near_perfect_fit_message(0.999, 0.999, 0.999) is not False


class TestCurvatureResponseSurfaces:
    """
    Regression for the reported issue: marginal p-to-enter made forward
    selection return no model for a curvature-dominated CCD even though the
    full quadratic model is highly significant.  Under the full-model
    significance criterion all methods must find the active effects.
    """

    CCD_ROWS = [
        (130.0, 15.0, 3.0, -151.36),
        (85.0, 0.0, 5.045378491522287, 4.02),
        (85.0, -25.226892457611434, 0.0, -530.9),
        (130.0, -15.0, -3.0, -197.91),
        (160.6806773728343, 0.0, 0.0, 6.34),
        (85.0, 25.226892457611434, 0.0, -479.46),
        (85.0, 0.0, 0.0, 3.92),
        (85.0, 0.0, 0.0, 3.92),
        (40.0, -15.0, -3.0, -184.97),
        (85.0, 0.0, 0.0, 3.92),
        (130.0, -15.0, 3.0, -198.166),
        (40.0, -15.0, 3.0, -184.468),
        (85.0, 0.0, 0.0, 3.92),
        (130.0, 15.0, -3.0, -151.114),
        (85.0, 0.0, -5.045378491522287, 3.81),
        (40.0, 15.0, -3.0, -170.57),
        (85.0, 0.0, 0.0, 3.92),
        (40.0, 15.0, 3.0, -170.068),
        (9.31932262716569, 0.0, 0.0, 1.498),
    ]

    def _analysis(self):
        rows = np.array(self.CCD_ROWS, dtype=float)
        factors = [
            Factor('A', FactorType.CONTINUOUS, levels=[40.0, 130.0]),
            Factor('B', FactorType.CONTINUOUS, levels=[-15.0, 15.0]),
            Factor('f_C', FactorType.CONTINUOUS, levels=[-3.0, 3.0]),
        ]
        design = pd.DataFrame(rows[:, :3], columns=['A', 'B', 'f_C'])
        return make_analysis(design, rows[:, 3], factors)

    ACTIVE = {'1', 'A', 'B', 'f_C', 'A*B', 'A*f_C', 'I(B**2)'}

    def test_all_methods_find_curvature_dominated_effects(self):
        analysis = self._analysis()
        for method in ('forward', 'backward', 'stepwise'):
            result = run_silently(analysis, method=method, alpha=0.10)
            assert set(result.final_terms) == self.ACTIVE, method
            assert len(result.selection_path) > 0

    def test_table_consistent_with_selection(self):
        analysis = self._analysis()
        result = run_silently(analysis, method='forward', alpha=0.10)
        table = result.term_table.set_index('Term')
        for term in ('A', 'B', 'f_C', 'A*B', 'A*f_C', 'I(B**2)'):
            assert table.loc[term, 'Reason'] == 'Significant'
        for term in ('B*f_C', 'I(A**2)', 'I(f_C**2)'):
            assert table.loc[term, 'Reason'] == 'Not significant'

    def test_marginal_fits_would_be_nonsignificant(self):
        # Guards the premise of the bug: no single main effect is significant
        # on its own, which is why the old marginal criterion stalled.
        analysis = self._analysis()
        for f in analysis.factors:
            r = analysis.fit(['1', f.name])
            assert (term_p_value(r, f.name) or 1.0) > 0.10


class TestHelpers:

    def test_required_by_hierarchy(self):
        fnames = ['A', 'B']
        assert required_by_hierarchy(['1', 'A', 'B', 'A*B'], fnames, 'A')
        assert required_by_hierarchy(['1', 'A', 'B', 'A*B'], fnames, 'B')
        assert not required_by_hierarchy(['1', 'A', 'B', 'A*B'], fnames, 'A*B')
        assert not required_by_hierarchy(['1', 'A'], fnames, 'A')
        assert not required_by_hierarchy(['1', 'A', 'I(A**2)'], fnames[:1], '1')

    def test_term_p_value_from_anova_summary(self):
        design = pd.DataFrame({'A': [0.5, 1.0, 1.5, 2.0, 2.5],
                               'B': [-1, 1, -1, 1, -1]})
        y = 3 * design['A'] + 0.2 * design['B']
        factors = [continuous('A'), continuous('B')]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            analysis = make_analysis(design, y, factors)
            results = analysis.fit(['1', 'A', 'B'])
        p = term_p_value(results, 'A')
        assert p is not None and 0 < p < 0.05

    def test_steps_record_current_terms(self):
        design, y, factors = ccd_dataset()
        result = run_silently(make_analysis(design, y, factors), method='forward')
        for step in result.selection_path:
            assert isinstance(step, SelectionStep)
            assert all(t in step.current_terms for t in ('1',))
            assert step.step_number > 0
        numbers = [s.step_number for s in result.selection_path]
        assert numbers == sorted(numbers)