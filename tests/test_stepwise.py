"""
Tests for Stepwise Regression with BIC.
"""
import pytest
import numpy as np
import pandas as pd
from src.core.factors import Factor, FactorType
from src.core.analysis import ANOVAAnalysis
from src.core.stepwise import (
    stepwise_selection,
    compute_bic,
    get_candidate_terms_forward,
    get_candidate_terms_backward,
    format_stepwise_summary
)


class TestBICComputation:
    """Test BIC calculation."""
    
    def test_bic_basic(self):
        """Test BIC computation on simple linear model."""
        # Create simple dataset
        np.random.seed(42)
        n = 20
        x = np.linspace(-1, 1, n)
        y = 2 + 3*x + np.random.normal(0, 0.5, n)
        
        design = pd.DataFrame({'A': x})
        factors = [Factor('A', FactorType.CONTINUOUS, levels=[-1, 1])]
        
        analysis = ANOVAAnalysis(design, y, factors)
        results = analysis.fit(['1', 'A'])
        
        bic = compute_bic(results, n)
        
        # BIC should be positive and finite
        assert bic > 0
        assert np.isfinite(bic)
    
    def test_bic_decreases_with_true_effect(self):
        """BIC should decrease when adding a true effect."""
        np.random.seed(42)
        n = 30
        x1 = np.linspace(-1, 1, n)
        x2 = np.linspace(-1, 1, n)
        # True model: y = 1 + 2*x1 + 3*x2
        y = 1 + 2*x1 + 3*x2 + np.random.normal(0, 0.3, n)
        
        design = pd.DataFrame({'A': x1, 'B': x2})
        factors = [
            Factor('A', FactorType.CONTINUOUS, levels=[-1, 1]),
            Factor('B', FactorType.CONTINUOUS, levels=[-1, 1])
        ]
        
        analysis = ANOVAAnalysis(design, y, factors)
        
        # Model with just intercept
        results_int = analysis.fit(['1'])
        bic_int = compute_bic(results_int, n)
        
        # Model with intercept + A
        results_a = analysis.fit(['1', 'A'])
        bic_a = compute_bic(results_a, n)
        
        # Model with intercept + A + B
        results_ab = analysis.fit(['1', 'A', 'B'])
        bic_ab = compute_bic(results_ab, n)
        
        # BIC should decrease as we add true effects
        assert bic_a < bic_int  # Adding A improves model
        assert bic_ab < bic_a   # Adding B improves model


class TestCandidateGeneration:
    """Test candidate term generation."""
    
    def test_forward_candidates_basic(self):
        """Test forward candidate generation."""
        current_terms = ['1', 'A']
        all_terms = ['1', 'A', 'B', 'A*B', 'I(A**2)']
        factor_names = ['A', 'B']
        
        candidates = get_candidate_terms_forward(
            current_terms, all_terms, factor_names
        )
        
        # Should be able to add B (no hierarchy issues)
        assert 'B' in candidates
        # Should be able to add A^2 (A is in model)
        assert 'I(A**2)' in candidates
        # Should NOT be able to add A*B (B not in model - hierarchy)
        assert 'A*B' not in candidates
    
    def test_backward_candidates_basic(self):
        """Test backward candidate generation."""
        current_terms = ['1', 'A', 'B', 'A*B']
        factor_names = ['A', 'B']
        mandatory_terms = ['1']
        
        candidates = get_candidate_terms_backward(
            current_terms, factor_names, mandatory_terms
        )
        
        # Should NOT be able to remove A or B (interaction depends on them)
        assert 'A' not in candidates
        assert 'B' not in candidates
        # Should be able to remove A*B
        assert 'A*B' in candidates
        # Should NOT include intercept (mandatory)
        assert '1' not in candidates
    
    def test_backward_respects_hierarchy(self):
        """Test that backward elimination respects hierarchy."""
        current_terms = ['1', 'A', 'B', 'C', 'A*B']
        factor_names = ['A', 'B', 'C']
        mandatory_terms = ['1']
        
        candidates = get_candidate_terms_backward(
            current_terms, factor_names, mandatory_terms
        )
        
        # C can be removed (no dependencies)
        assert 'C' in candidates
        # A*B can be removed
        assert 'A*B' in candidates
        # A and B cannot be removed (A*B depends on them)
        assert 'A' not in candidates
        assert 'B' not in candidates


class TestStepwiseSelection:
    """Test full stepwise selection algorithm."""
    
    def test_stepwise_simple_linear(self):
        """Test stepwise on simple 2-factor linear model."""
        np.random.seed(42)
        n = 40
        x1 = np.linspace(-1, 1, n)
        x2 = np.linspace(-1, 1, n)
        # True model: y = 1 + 3*x1 (x2 is noise)
        y = 1 + 3*x1 + np.random.normal(0, 0.5, n)
        
        design = pd.DataFrame({'A': x1, 'B': x2})
        factors = [
            Factor('A', FactorType.CONTINUOUS, levels=[-1, 1]),
            Factor('B', FactorType.CONTINUOUS, levels=[-1, 1])
        ]
        
        analysis = ANOVAAnalysis(design, y, factors)
        
        all_terms = ['1', 'A', 'B', 'A*B']
        
        results = stepwise_selection(
            anova_analysis=analysis,
            all_possible_terms=all_terms,
            starting_terms=['1'],
            mandatory_terms=['1'],
            max_iterations=10,
            bic_threshold=2.0
        )
        
        # Should select A (true effect)
        assert 'A' in results.final_terms
        # Should have intercept
        assert '1' in results.final_terms
        # Likely should NOT select B or A*B (noise)
        # (though with small sample, might occasionally select B)
        
        # Should have some iterations
        assert results.n_iterations > 0
        assert results.n_iterations <= 10
        
        # BIC should improve
        assert results.improvement >= 0  # At least no worse
    
    def test_stepwise_with_interaction(self):
        """Test stepwise selects interaction when present."""
        np.random.seed(42)
        n = 50
        x1 = np.random.uniform(-1, 1, n)
        x2 = np.random.uniform(-1, 1, n)
        # True model: y = 1 + 2*x1 + 3*x2 + 4*x1*x2
        y = 1 + 2*x1 + 3*x2 + 4*x1*x2 + np.random.normal(0, 0.5, n)
        
        design = pd.DataFrame({'A': x1, 'B': x2})
        factors = [
            Factor('A', FactorType.CONTINUOUS, levels=[-1, 1]),
            Factor('B', FactorType.CONTINUOUS, levels=[-1, 1])
        ]
        
        analysis = ANOVAAnalysis(design, y, factors)
        
        all_terms = ['1', 'A', 'B', 'A*B']
        
        results = stepwise_selection(
            anova_analysis=analysis,
            all_possible_terms=all_terms,
            max_iterations=10,
            bic_threshold=2.0
        )
        
        # Should select A, B, A*B (all true effects)
        assert 'A' in results.final_terms
        assert 'B' in results.final_terms
        assert 'A*B' in results.final_terms
        
        # Should have multiple steps
        assert results.n_iterations >= 3  # At least 3 additions
    
    def test_stepwise_convergence(self):
        """Test that stepwise converges."""
        np.random.seed(42)
        n = 30
        x = np.linspace(-1, 1, n)
        y = 2 + 3*x + np.random.normal(0, 0.5, n)
        
        design = pd.DataFrame({'A': x})
        factors = [Factor('A', FactorType.CONTINUOUS, levels=[-1, 1])]
        
        analysis = ANOVAAnalysis(design, y, factors)
        
        all_terms = ['1', 'A', 'I(A**2)', 'I(A**3)']
        
        results = stepwise_selection(
            anova_analysis=analysis,
            all_possible_terms=all_terms,
            max_iterations=20,
            bic_threshold=2.0
        )
        
        # Should converge before max_iterations
        assert results.n_iterations < 20
        # Should have convergence reason
        assert 'improvement' in results.convergence_reason.lower() or \
               'possible' in results.convergence_reason.lower()
    
    def test_stepwise_respects_mandatory_terms(self):
        """Test that mandatory terms are never removed."""
        np.random.seed(42)
        n = 30
        x = np.linspace(-1, 1, n)
        y = np.random.normal(0, 1, n)  # Pure noise, no relationship
        
        design = pd.DataFrame({'A': x})
        factors = [Factor('A', FactorType.CONTINUOUS, levels=[-1, 1])]
        
        analysis = ANOVAAnalysis(design, y, factors)
        
        all_terms = ['1', 'A']
        
        results = stepwise_selection(
            anova_analysis=analysis,
            all_possible_terms=all_terms,
            starting_terms=['1'],
            mandatory_terms=['1'],
            max_iterations=10,
            bic_threshold=2.0
        )
        
        # Intercept should always be in final model (mandatory)
        assert '1' in results.final_terms


class TestStepwiseSummary:
    """Test summary formatting."""
    
    def test_format_summary(self):
        """Test that summary formats without error."""
        np.random.seed(42)
        n = 30
        x = np.linspace(-1, 1, n)
        y = 2 + 3*x + np.random.normal(0, 0.5, n)
        
        design = pd.DataFrame({'A': x})
        factors = [Factor('A', FactorType.CONTINUOUS, levels=[-1, 1])]
        
        analysis = ANOVAAnalysis(design, y, factors)
        
        all_terms = ['1', 'A']
        
        results = stepwise_selection(
            anova_analysis=analysis,
            all_possible_terms=all_terms,
            max_iterations=5,
            bic_threshold=2.0
        )
        
        summary = format_stepwise_summary(results)
        
        # Should be non-empty string
        assert isinstance(summary, str)
        assert len(summary) > 0
        # Should contain key information
        assert 'BIC' in summary
        assert 'Convergence' in summary
        assert 'Final Model' in summary
        
        # Should not error when printed
        print(summary)  # Just verify it doesn't crash


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
