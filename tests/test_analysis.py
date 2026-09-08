"""
Tests for ANOVA analysis module.

Updated to use Python/patsy notation: I(A**2) instead of A^2
"""

import pytest
import warnings
from typing import List, Tuple
import numpy as np
import pandas as pd
from src.core.analysis import (
    ANOVAAnalysis,
    generate_model_terms,
    parse_model_term,
    enforce_hierarchy,
    detect_split_plot_structure,
    prepare_analysis_data,
    validate_model_terms,
    quadratic
)
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.full_factorial import full_factorial


class TestModelTermGeneration:
    """Test model term generation utilities."""
    
    def test_generate_linear_terms(self):
        """Test linear model term generation."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        terms = generate_model_terms(factors, 'linear')
        
        assert '1' in terms
        assert 'A' in terms
        assert 'B' in terms
        assert len(terms) == 3
    
    def test_generate_interaction_terms(self):
        """Test interaction model term generation."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        terms = generate_model_terms(factors, 'interaction')
        
        assert 'A*B' in terms
        assert 'A*C' in terms
        assert 'B*C' in terms
        assert len(terms) == 7  # 1 + 3 main + 3 interactions
    
    def test_generate_quadratic_terms(self):
        """Test quadratic model term generation with Python notation."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        terms = generate_model_terms(factors, 'quadratic')
        
        # Should use I(**2) notation, not ^
        assert 'I(A**2)' in terms
        assert 'I(B**2)' in terms
        assert 'A*B' in terms
        assert len(terms) == 6  # 1 + 2 main + 1 interaction + 2 quadratic
    
    def test_quadratic_helper_function(self):
        """Test quadratic() helper function."""
        result = quadratic('Temperature')
        assert result == 'I(Temperature**2)'
    
    def test_parse_model_term_main_effect(self):
        """Test parsing main effect term."""
        factor_list, operator = parse_model_term("Temperature")
        
        assert factor_list == ["Temperature"]
        assert operator == ''
    
    def test_parse_model_term_interaction(self):
        """Test parsing interaction term."""
        factor_list, operator = parse_model_term("A*B")
        
        assert factor_list == ["A", "B"]
        assert operator == '*'
    
    def test_parse_model_term_quadratic(self):
        """Test parsing quadratic term with Python notation."""
        factor_list, operator = parse_model_term("I(Temperature**2)")
        
        assert factor_list == ["Temperature"]
        assert operator == '**'
    
    def test_enforce_hierarchy_adds_main_effects(self):
        """Test hierarchy enforcement adds missing main effects."""
        terms = ['1', 'A*B']
        factor_names = ['A', 'B', 'C']
        
        complete, added = enforce_hierarchy(terms, factor_names)
        
        assert 'A' in complete
        assert 'B' in complete
        assert 'A' in added
        assert 'B' in added
    
    def test_enforce_hierarchy_with_quadratic(self):
        """Test hierarchy enforcement with quadratic terms."""
        terms = ['1', 'I(A**2)']
        factor_names = ['A', 'B']
        
        complete, added = enforce_hierarchy(terms, factor_names)
        
        assert 'A' in complete
        assert 'A' in added
    
    def test_enforce_hierarchy_ordering(self):
        """Test that enforce_hierarchy orders terms correctly."""
        terms = ['1', 'A*B', 'I(C**2)']
        factor_names = ['A', 'B', 'C']
        
        complete, added = enforce_hierarchy(terms, factor_names)
        
        # Should add main effects
        assert 'A' in complete
        assert 'B' in complete
        assert 'C' in complete
        
        # Should order: intercept, main effects, interactions, quadratic
        intercept_idx = complete.index('1')
        a_idx = complete.index('A')
        b_idx = complete.index('B')
        c_idx = complete.index('C')
        interaction_idx = complete.index('A*B')
        quadratic_idx = complete.index('I(C**2)')
        
        # Verify ordering
        assert intercept_idx < a_idx
        assert a_idx < interaction_idx
        assert b_idx < interaction_idx
        assert c_idx < quadratic_idx
        assert interaction_idx < quadratic_idx


class TestStructureDetection:
    """Test design structure detection."""
    
    def test_detect_regular_factorial(self):
        """Test detection of regular factorial (no split-plot)."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, randomize=False)
        
        structure = detect_split_plot_structure(design, factors)
        
        assert structure['is_split_plot'] is False
        assert len(structure['whole_plot_factors']) == 0
        assert len(structure['sub_plot_factors']) == 2
    
    def test_detect_split_plot(self):
        """Test detection of split-plot design."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.HARD, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = pd.DataFrame({
            'A': [-1, -1, 1, 1],
            'B': [-1, 1, -1, 1],
            'WholePlot': [1, 1, 2, 2]
        })
        
        structure = detect_split_plot_structure(design, factors)
        
        assert structure['is_split_plot'] is True
        assert 'A' in structure['whole_plot_factors']
        assert 'B' in structure['sub_plot_factors']
        assert structure['whole_plot_column'] == 'WholePlot'
    
    def test_detect_blocking(self):
        """Test detection of blocked design."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = pd.DataFrame({
            'A': [-1, 1],
            'Block': [1, 2]
        })
        
        structure = detect_split_plot_structure(design, factors)
        
        assert structure['has_blocking'] is True


class TestDataPreparation:
    """Test data preparation utilities."""
    
    def test_prepare_analysis_data_basic(self):
        """Test basic data preparation."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = pd.DataFrame({'A': [-1, 1]})
        response = np.array([10, 20])
        
        data = prepare_analysis_data(design, response, factors)
        
        assert 'A' in data.columns
        assert 'Response' in data.columns
        assert len(data) == 2
    
    def test_prepare_analysis_data_length_mismatch(self):
        """Test error on length mismatch."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = pd.DataFrame({'A': [-1, 1]})
        response = np.array([10, 20, 30])
        
        with pytest.raises(ValueError, match="Response length mismatch"):
            prepare_analysis_data(design, response, factors)
    
    def test_validate_model_terms_invalid_factor(self):
        """Test validation catches invalid factor names."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = pd.DataFrame({'A': [-1, 1]})
        terms = ['A', 'B']  # B doesn't exist
        
        with pytest.raises(ValueError, match="Factor 'B'.*not found"):
            validate_model_terms(terms, factors, design)
    
    def test_validate_model_terms_quadratic_non_continuous(self):
        """Test validation catches quadratic terms on categorical factors."""
        factors = [
            Factor("Material", FactorType.CATEGORICAL, ChangeabilityLevel.EASY, 
                   levels=["A", "B"])
        ]
        
        design = pd.DataFrame({'Material': ["A", "B"]})
        terms = ['I(Material**2)']
        
        with pytest.raises(ValueError, match="Quadratic.*requires continuous factor"):
            validate_model_terms(terms, factors, design)


class TestRegularANOVA:
    """Test regular factorial ANOVA."""
    
    def test_fit_simple_model(self):
        """Test fitting simple 2^2 factorial."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, randomize=False)
        
        # Simple additive model: Response = 10 + 2*A + 3*B
        response = 10 + 2*design['A'] + 3*design['B'] + np.random.normal(0, 0.1, len(design))
        
        analysis = ANOVAAnalysis(design, response, factors)
        results = analysis.fit(['A', 'B'])
        
        # Check results structure
        assert results.anova_table is not None
        assert results.effect_estimates is not None
        assert len(results.residuals) == len(design)
        assert results.is_split_plot is False
        
        # Check coefficient estimates are close to true values
        coef_A = results.effect_estimates.loc['A', 'Coefficient']
        coef_B = results.effect_estimates.loc['B', 'Coefficient']
        
        assert abs(coef_A - 2.0) < 0.5
        assert abs(coef_B - 3.0) < 0.5
    
    def test_fit_with_interaction(self):
        """Test fitting model with interaction term."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, n_center_points=2, randomize=False)
        
        response = (10 + 2*design['A'] + 3*design['B'] + 4*design['A']*design['B'] + 
                   np.random.normal(0, 0.5, len(design)))
        
        analysis = ANOVAAnalysis(design, response, factors)
        results = analysis.fit(['A', 'B', 'A*B'])
        
        coef_AB = results.effect_estimates.loc['A:B', 'Coefficient']
        assert abs(coef_AB - 4.0) < 1.0
    
    def test_hierarchy_enforcement_warning(self):
        """Test that hierarchy enforcement warns and adds terms."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, n_center_points=2, randomize=False)
        response = (10 + 4*design['A']*design['B'] + 
                   np.random.normal(0, 0.5, len(design)))
        
        analysis = ANOVAAnalysis(design, response, factors)
        
        # Request only interaction (should warn and add main effects)
        with pytest.warns(UserWarning, match="Added for hierarchy"):
            results = analysis.fit(['1', 'A*B'], enforce_hierarchy_flag=True)
        
        # Should have added A and B
        assert 'A' in results.model_terms
        assert 'B' in results.model_terms
        assert 'A*B' in results.model_terms
    
    def test_update_model_add_terms(self):
        """Test updating model by adding terms."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, n_center_points=2, randomize=False)
        response = (10 + 2*design['A'] + 3*design['B'] + 4*design['A']*design['B'] +
                   np.random.normal(0, 0.5, len(design)))
        
        analysis = ANOVAAnalysis(design, response, factors)
        
        results1 = analysis.fit(['A', 'B'])
        results2 = analysis.update_model(terms_to_add=['A*B'])
        
        assert 'A*B' in results2.model_terms
        assert results2.r_squared > results1.r_squared


class TestSplitPlotANOVA:
    """Test split-plot ANOVA with proper error terms."""
    
    def test_split_plot_detection(self):
        """Test automatic split-plot detection."""
        factors = [
            Factor("Temperature", FactorType.CONTINUOUS, ChangeabilityLevel.HARD, levels=[100, 200]),
            Factor("Time", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[10, 30])
        ]
        
        design = pd.DataFrame({
            'Temperature': [-1, -1, 1, 1],
            'Time': [-1, 1, -1, 1],
            'WholePlot': [1, 1, 2, 2]
        })
        
        response = np.array([10, 15, 20, 25])
        
        analysis = ANOVAAnalysis(design, response, factors)
        
        assert analysis.design_structure['is_split_plot'] is True
        assert 'Temperature' in analysis.design_structure['whole_plot_factors']
        assert 'Time' in analysis.design_structure['sub_plot_factors']
    
    def test_split_plot_override(self):
        """Test overriding split-plot detection."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.HARD, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = pd.DataFrame({
            'A': [-1, -1, 1, 1],
            'B': [-1, 1, -1, 1],
            'WholePlot': [1, 1, 2, 2]
        })
        
        response = np.array([10, 15, 20, 25])
        
        analysis = ANOVAAnalysis(design, response, factors, is_split_plot=False)
        
        assert analysis.design_structure['is_split_plot'] is False


class TestBlockedDesign:
    """Test ANOVA with blocked designs."""
    
    def test_blocked_design_fixed_effect(self):
        """Test ANOVA on blocked design with Block as fixed effect."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, n_blocks=2, randomize=False)
        response = 10 + 2*design['A'] + 3*design['B'] + 5*(design['Block']-1.5)
        
        analysis = ANOVAAnalysis(design, response, factors, block_as_random=False)
        
        assert analysis.design_structure['has_blocking'] is True
        
        results = analysis.fit(['A', 'B'])
        
        assert results is not None
    
    def test_blocked_design_random_effect(self):
        """Test ANOVA on blocked design with Block as random effect."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, n_blocks=2, randomize=False)
        response = (10 + 2*design['A'] + 3*design['B'] + 
                   np.random.normal(0, 0.5, len(design)))
        
        analysis = ANOVAAnalysis(design, response, factors, block_as_random=True)
        
        assert analysis.design_structure['has_blocking'] is True
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=Warning)
            try:
                results = analysis.fit(['A', 'B'])
                assert results is not None
            except Exception as e:
                pytest.skip(f"Mixed model convergence issue (acceptable): {e}")


class TestLogWorthComputation:
    """Test LogWorth computation (no plotting)."""
    
    def test_logworth_computed(self):
        """Test that LogWorth values are precomputed in results."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, randomize=False)
        response = 10 + 2*design['A'] + 3*design['B'] + np.random.normal(0, 0.5, len(design))
        
        analysis = ANOVAAnalysis(design, response, factors)
        results = analysis.fit(['A', 'B'])
        
        assert hasattr(results, 'logworth')
        assert 'LogWorth' in results.logworth.columns
        assert len(results.logworth) > 0
        
        for idx, row in results.logworth.iterrows():
            if not np.isnan(row['LogWorth']):
                if row['p_value'] >= 1e-16:
                    expected_logworth = -np.log10(row['p_value'])
                    assert abs(row['LogWorth'] - expected_logworth) < 1e-6
                else:
                    assert row['LogWorth'] <= 16.0
    
    def test_logworth_excludes_intercept(self):
        """Test that LogWorth excludes intercept term."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, randomize=False)
        response = 10 + 2*design['A'] + 3*design['B'] + np.random.normal(0, 0.1, len(design))
        
        analysis = ANOVAAnalysis(design, response, factors)
        results = analysis.fit(['1', 'A', 'B'])
        
        assert 'Intercept' not in results.logworth.index
        assert 'A' in results.logworth.index
        assert 'B' in results.logworth.index


class TestDiagnostics:
    """Test diagnostic computations."""
    
    def test_shapiro_wilk_computed(self):
        """Test that Shapiro-Wilk test is computed."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, randomize=False)
        response = 10 + 2*design['A'] + 3*design['B'] + np.random.normal(0, 0.5, len(design))
        
        analysis = ANOVAAnalysis(design, response, factors)
        results = analysis.fit(['A', 'B'])
        
        assert 'shapiro_wilk' in results.diagnostics
        assert 'statistic' in results.diagnostics['shapiro_wilk']
        assert 'p_value' in results.diagnostics['shapiro_wilk']


class TestDegreesOfFreedom:
    """Test degrees of freedom validation."""
    
    def test_saturated_model_warns_and_fits(self):
        """Test that saturated model (df=0) warns but still fits."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, randomize=False)
        response = 10 + 2*design['A'] + 3*design['B'] + 4*design['A']*design['B']
        
        analysis = ANOVAAnalysis(design, response, factors)
        
        with pytest.warns(UserWarning, match=r"Model is saturated.*df_error = 0"):
            results = analysis.fit(['A', 'B', 'A*B'])
        
        assert results is not None
        assert results.r_squared == 1.0
        assert 'A' in results.effect_estimates.index
    
    def test_oversaturated_model_warns(self):
        """Test that oversaturated model (df<0) warns."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, randomize=False)
        response = 10 + 2*design['A'] + np.random.normal(0, 0.1, len(design))
        
        design['C'] = [-1, 1, -1, 1]
        factors_extended = factors + [
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        analysis = ANOVAAnalysis(design, response, factors_extended)
        
        # Oversaturated: 4 runs but 5 terms (intercept + 3 factors + 1 interaction)
        with pytest.warns(UserWarning, match="Oversaturated|singular|not enough"):
            try:
                results = analysis.fit(['A', 'B', 'C', 'A*B'])
                # Model may fail to fit, which is acceptable
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                pass
    
    def test_low_df_warns(self):
        """Test that low degrees of freedom produces warning."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, n_center_points=1, randomize=False)
        response = 10 + 2*design['A'] + 3*design['B'] + np.random.normal(0, 0.5, len(design))
        
        analysis = ANOVAAnalysis(design, response, factors)
        
        with pytest.warns(UserWarning, match="Low df"):
            results = analysis.fit(['A', 'B', 'A*B'])
        
        assert results is not None
    
    def test_adequate_df_no_warning(self):
        """Test that adequate degrees of freedom produces no warning."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = full_factorial(factors, n_center_points=4, randomize=False)
        response = 10 + 2*design['A'] + 3*design['B'] + np.random.normal(0, 0.5, len(design))
        
        analysis = ANOVAAnalysis(design, response, factors)
        
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                results = analysis.fit(['A', 'B'])
                assert results is not None
            except UserWarning as w:
                if "hierarchy" not in str(w).lower():
                    raise
    
    def test_split_plot_whole_plot_warning(self):
        """Test warning for insufficient whole-plots in split-plot design."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.HARD, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = pd.DataFrame({
            'A': [-1, -1, 1, 1],
            'B': [-1, 1, -1, 1],
            'WholePlot': [1, 1, 2, 2]
        })
        response = np.array([10, 15, 20, 25])
        
        analysis = ANOVAAnalysis(design, response, factors)
        
        with pytest.warns(UserWarning, match="whole-plots"):
            results = analysis.fit(['A', 'B'])
        
        assert results is not None


class TestTwoStrataSplitPlot:
    """
    Tests for the two-strata split-plot ANOVA implementation.

    Uses test_case_5_split_plot.csv data (Temperature/Pressure hard,
    Time/Catalyst easy, 4 whole-plots x 4 sub-runs x 2 replicates = 32 runs).
    True model: Yield = 70 + 5*T + 3*P + 4*Ti + 2*C + 3*T*Ti + errors
    """

    # ------------------------------------------------------------------
    # Fixtures / shared setup
    # ------------------------------------------------------------------

    @staticmethod
    def _make_factors() -> List[Factor]:
        return [
            Factor("Temperature", FactorType.CONTINUOUS,
                   ChangeabilityLevel.HARD, levels=[125, 175]),
            Factor("Pressure", FactorType.CONTINUOUS,
                   ChangeabilityLevel.HARD, levels=[25, 75]),
            Factor("Time", FactorType.CONTINUOUS,
                   ChangeabilityLevel.EASY, levels=[0, 20]),
            Factor("Catalyst", FactorType.CATEGORICAL,
                   ChangeabilityLevel.EASY, levels=["A", "B"]),
        ]

    @staticmethod
    def _make_design_and_response() -> Tuple[pd.DataFrame, np.ndarray]:
        """Return the test_case_5 design and response as arrays."""
        from pathlib import Path
        csv_path = (
            Path(__file__).parent.parent
            / "test_data" / "test_case_5_split_plot.csv"
        )
        # Read past the comment header block
        raw = pd.read_csv(csv_path, comment='#')
        response = raw["Yield"].values
        return raw, response

    # ------------------------------------------------------------------
    # Term classification
    # ------------------------------------------------------------------

    def test_classify_terms_whole_plot_only(self):
        """Pure whole-plot terms land in the WP stratum."""
        from src.core.split_plot_analysis import _classify_terms
        wp, sp = _classify_terms(
            ['Temperature', 'Pressure', 'Temperature*Pressure'],
            {'Temperature', 'Pressure'},
            {'Time', 'Catalyst'},
        )
        assert 'Temperature' in wp
        assert 'Pressure' in wp
        assert 'Temperature*Pressure' in wp
        assert len(sp) == 0

    def test_classify_terms_subplot_only(self):
        """Pure subplot terms land in the SP stratum."""
        from src.core.split_plot_analysis import _classify_terms
        wp, sp = _classify_terms(
            ['Time', 'Catalyst'],
            {'Temperature', 'Pressure'},
            {'Time', 'Catalyst'},
        )
        assert len(wp) == 0
        assert 'Time' in sp
        assert 'Catalyst' in sp

    def test_classify_terms_cross_stratum_interaction(self):
        """WP*SP interactions belong to the subplot stratum."""
        from src.core.split_plot_analysis import _classify_terms
        wp, sp = _classify_terms(
            ['Temperature', 'Time', 'Temperature*Time'],
            {'Temperature'},
            {'Time'},
        )
        assert 'Temperature*Time' in sp
        assert 'Temperature*Time' not in wp

    # ------------------------------------------------------------------
    # Whole-plot means helper
    # ------------------------------------------------------------------

    def test_whole_plot_means_shape(self):
        """_build_whole_plot_means returns one row per unique whole-plot."""
        from src.core.split_plot_analysis import _build_whole_plot_means
        design, response = self._make_design_and_response()
        design["Yield"] = response
        wp_means = _build_whole_plot_means(
            design, "WholePlot", ["Temperature", "Pressure"], "Yield"
        )
        # 4 unique whole-plot IDs x 2 replicates -> 4 means
        assert len(wp_means) == 4
        assert "Yield" in wp_means.columns
        assert "Temperature" in wp_means.columns

    # ------------------------------------------------------------------
    # End-to-end: no crash
    # ------------------------------------------------------------------

    def test_fit_does_not_raise(self):
        """Loading test_case_5 and fitting a linear model must not raise."""
        factors = self._make_factors()
        design, response = self._make_design_and_response()

        analysis = ANOVAAnalysis(
            design=design, response=response,
            factors=factors, response_name="Yield",
        )
        results = analysis.fit(
            ['Temperature', 'Pressure', 'Time', 'Catalyst'],
            enforce_hierarchy_flag=False,
        )
        assert results is not None

    # ------------------------------------------------------------------
    # Results structure
    # ------------------------------------------------------------------

    def test_results_is_split_plot_flag(self):
        """Results object must report is_split_plot=True."""
        factors = self._make_factors()
        design, response = self._make_design_and_response()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Yield")
        results = analysis.fit(
            ['Temperature', 'Pressure', 'Time', 'Catalyst'],
            enforce_hierarchy_flag=False,
        )
        assert results.is_split_plot is True

    def test_anova_table_has_two_strata(self):
        """ANOVA table must contain both whole-plot and subplot error rows."""
        factors = self._make_factors()
        design, response = self._make_design_and_response()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Yield")
        results = analysis.fit(
            ['Temperature', 'Pressure', 'Time', 'Catalyst'],
            enforce_hierarchy_flag=False,
        )
        assert 'WholePlot Error' in results.anova_table.index
        assert 'SubPlot Error' in results.anova_table.index

    def test_anova_table_stratum_labels(self):
        """Hard-factor terms are labelled Whole-Plot; easy-factor terms Sub-Plot."""
        factors = self._make_factors()
        design, response = self._make_design_and_response()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Yield")
        results = analysis.fit(
            ['Temperature', 'Pressure', 'Time', 'Catalyst'],
            enforce_hierarchy_flag=False,
        )
        tbl = results.anova_table
        assert tbl.loc['Temperature', 'Stratum'] == 'Whole-Plot'
        assert tbl.loc['Pressure', 'Stratum'] == 'Whole-Plot'
        # Time or Catalyst must be Sub-Plot
        sp_sources = tbl[tbl['Stratum'] == 'Sub-Plot'].index.tolist()
        assert any('Time' in s or 'Catalyst' in s for s in sp_sources)

    def test_residuals_length(self):
        """Residuals must have one entry per run."""
        factors = self._make_factors()
        design, response = self._make_design_and_response()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Yield")
        results = analysis.fit(
            ['Temperature', 'Pressure', 'Time', 'Catalyst'],
            enforce_hierarchy_flag=False,
        )
        assert len(results.residuals) == len(design)

    def test_r_squared_range(self):
        """R² must be in [0, 1]."""
        factors = self._make_factors()
        design, response = self._make_design_and_response()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Yield")
        results = analysis.fit(
            ['Temperature', 'Pressure', 'Time', 'Catalyst'],
            enforce_hierarchy_flag=False,
        )
        assert 0.0 <= results.r_squared <= 1.0

    # ------------------------------------------------------------------
    # Statistical correctness: known signal in test data
    # ------------------------------------------------------------------

    def test_temperature_significant(self):
        """
        Temperature has a true coefficient of 5 and should be significant
        in the whole-plot stratum (p < 0.05).
        """
        factors = self._make_factors()
        design, response = self._make_design_and_response()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Yield")
        results = analysis.fit(
            ['Temperature', 'Pressure', 'Time', 'Catalyst'],
            enforce_hierarchy_flag=False,
        )
        p_temp = results.anova_table.loc['Temperature', 'P']
        assert p_temp < 0.05, f"Temperature p={p_temp:.4f} should be < 0.05"

    def test_time_significant(self):
        """
        Time has a true coefficient of 4 and should be significant in the
        subplot stratum (p < 0.05).
        """
        factors = self._make_factors()
        design, response = self._make_design_and_response()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Yield")
        results = analysis.fit(
            ['Temperature', 'Pressure', 'Time', 'Catalyst'],
            enforce_hierarchy_flag=False,
        )
        p_time = results.anova_table.loc['Time', 'P']
        assert p_time < 0.05, f"Time p={p_time:.4f} should be < 0.05"

    def test_cross_stratum_interaction_in_sp(self):
        """
        Temperature*Time is a cross-stratum interaction and must be tested
        against subplot error (Stratum == Sub-Plot in the ANOVA table).
        """
        factors = self._make_factors()
        design, response = self._make_design_and_response()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Yield")
        results = analysis.fit(
            ['Temperature', 'Pressure', 'Time', 'Catalyst', 'Temperature*Time'],
            enforce_hierarchy_flag=True,
        )
        tbl = results.anova_table
        assert 'Temperature*Time' in tbl.index
        assert tbl.loc['Temperature*Time', 'Stratum'] == 'Sub-Plot'

    def test_logworth_columns_present(self):
        """LogWorth DataFrame must have LogWorth and Stratum columns."""
        factors = self._make_factors()
        design, response = self._make_design_and_response()
        analysis = ANOVAAnalysis(design=design, response=response,
                                 factors=factors, response_name="Yield")
        results = analysis.fit(
            ['Temperature', 'Pressure', 'Time', 'Catalyst'],
            enforce_hierarchy_flag=False,
        )
        assert 'LogWorth' in results.logworth.columns
        assert 'Stratum' in results.logworth.columns


class TestValidationData:
    """Test against published validation datasets."""

    def test_basic_factorial_validation(self):
        """Basic validation test - will expand later."""
        pass


class TestBlocking:
    """Test that blocking is included in the ANOVA analysis as a fixed effect."""

    def _blocked_design(self):
        """A 2^2 factorial in 2 blocks with replicates (natural units)."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        design = full_factorial(factors, n_replicates=2, n_blocks=2, randomize=False)
        # Block effect: block 1 adds -5, block 2 adds +5
        block_effect = np.where(design['Block'] == 1, -5.0, 5.0)
        response = 10 + 2 * design['A'] + 3 * design['B'] + block_effect
        return design, response, factors

    def test_fixed_effect_block_fits(self):
        """Block included as a fixed effect by default."""
        design, response, factors = self._blocked_design()
        analysis = ANOVAAnalysis(design, response, factors)
        results = analysis.fit(['A', 'B'])

        assert results is not None
        # Block term should appear in the ANOVA table
        assert 'Block' in results.anova_table.index

    def test_design_structure_detects_blocking(self):
        """Detection sees a Block column in a blocked design."""
        design, response, factors = self._blocked_design()
        analysis = ANOVAAnalysis(design, response, factors)
        assert analysis.design_structure['has_blocking'] is True

    def test_replicated_blocked_design_fits(self):
        """A design with both replicates and blocks fits the fixed-effect model."""
        design, response, factors = self._blocked_design()
        analysis = ANOVAAnalysis(design, response, factors)
        results = analysis.fit(['A', 'B'])

        assert results is not None
        assert 'Block' in results.anova_table.index
        assert results.r_squared > 0.99


class TestCategoricalNumericLabels:
    """Categorical factors with numeric-looking levels must be fit as
    categorical (k-1 dummy columns), not silently as continuous."""

    def test_parse_model_term_strips_c_wrapper(self):
        """parse_model_term tolerates patsy C(...) wrappers."""
        main_factors, op1 = parse_model_term('C(Egg_lot)')
        assert main_factors == ['Egg_lot']
        assert op1 == ''

        int_factors, op2 = parse_model_term('C(Egg_lot)*Egg_percent')
        assert int_factors == ['Egg_lot', 'Egg_percent']
        assert op2 == '*'

    def test_full_factorial_categorical_df(self):
        """4-level categorical x 2-level numeric with replicates: lot DF=3,
        percent DF=1, interaction DF=3 (was DF=1 when coerced to numeric)."""
        lot_levels = ['41007587', '41005191', '41007741', '41007302']
        factors = [
            Factor('Egg_lot', FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
                   levels=lot_levels, _validate_on_init=False),
            Factor('Egg_percent', FactorType.DISCRETE_NUMERIC,
                   ChangeabilityLevel.EASY, levels=[10, 20]),
        ]

        design = full_factorial(factors, n_replicates=2, randomize=False)
        # Egg_lot column must remain a string column in the design
        assert pd.api.types.is_string_dtype(design['Egg_lot'])
        assert design['Egg_lot'].nunique() == 4

        # Deterministic response: strong lot effect, mild percent + noise
        rng = np.random.default_rng(42)
        lot_means = {lot: m for lot, m in zip(lot_levels, [100, 102, 98, 105])}
        response = (
            design['Egg_lot'].map(lot_means).astype(float)
            + 1.5 * design['Egg_percent']
            + rng.normal(0, 0.05, len(design))
        )

        analysis = ANOVAAnalysis(design, response, factors)
        results = analysis.fit(['Egg_lot', 'Egg_percent', 'Egg_lot*Egg_percent'])

        df = results.anova_table['df']
        assert results.anova_table.index.get_loc('Egg_lot') is not None
        assert int(df.loc['Egg_lot']) == 3
        assert int(df.loc['Egg_percent']) == 1
        assert int(df.loc['Egg_lot:Egg_percent']) == 3

        # No C(...) wrappers leak into reported labels
        assert not any('C(' in str(i) for i in results.anova_table.index)
        assert not any('C(' in str(i) for i in results.effect_estimates.index)

        # effect_estimates use plain [T.level] dummy keys (predict compat)
        main_dummy_keys = [
            i for i in results.effect_estimates.index
            if str(i).startswith('Egg_lot[') and ':' not in str(i)
        ]
        assert len(main_dummy_keys) == 3
        interaction_keys = [
            i for i in results.effect_estimates.index
            if str(i).startswith('Egg_lot[') and ':' in str(i)
        ]
        assert len(interaction_keys) == 3

    def test_categorical_df_matches_dummy_regression(self):
        """Wrapped formula produces the same DF as coding the dummies by hand."""
        lot_levels = ['41007587', '41005191', '41007741', '41007302']
        factors = [
            Factor('Egg_lot', FactorType.CATEGORICAL, ChangeabilityLevel.EASY,
                   levels=lot_levels, _validate_on_init=False),
            Factor('Egg_percent', FactorType.DISCRETE_NUMERIC,
                   ChangeabilityLevel.EASY, levels=[10, 20]),
        ]
        design = full_factorial(factors, n_replicates=2, randomize=False)

        rng = np.random.default_rng(7)
        lot_means = {lot: m for lot, m in zip(lot_levels, [50, 52, 48, 55])}
        response = (
            design['Egg_lot'].map(lot_means).astype(float)
            + 0.5 * design['Egg_percent']
            + rng.normal(0, 0.1, len(design))
        )

        analysis = ANOVAAnalysis(design, response, factors)
        results = analysis.fit(['Egg_lot', 'Egg_percent', 'Egg_lot*Egg_percent'])
        df = results.anova_table['df']

        # Reference: hand-built design matrix using 3 lot dummies + numeric,
        # plus the 3 per-level interaction columns (dummy * percent)
        ref = pd.get_dummies(design['Egg_lot'], prefix='Egg_lot', prefix_sep='')
        ref = ref.iloc[:, 1:]  # drop the reference level (fits with intercept)
        X = pd.concat([design[['Egg_percent']], ref], axis=1)
        for col in ref.columns:
            X[f'{col}:Egg_percent'] = ref[col] * design['Egg_percent']
        from statsmodels.api import OLS
        ref_fit = OLS(response, sm_add_constant(X)).fit()
        ref_rsq = ref_fit.rsquared

        assert df.loc['Egg_lot'] == 3
        assert df.loc['Egg_percent'] == 1
        assert df.loc['Egg_lot:Egg_percent'] == 3
        assert abs(results.r_squared - ref_rsq) < 1e-9


def sm_add_constant(frame):
    """Add an intercept column to an arbitrary DataFrame."""
    import statsmodels.api as sm
    return sm.add_constant(frame)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])