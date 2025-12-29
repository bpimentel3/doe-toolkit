"""
Tests for diagnostic infrastructure.

Tests variance diagnostics, estimability checks, and summary generation.
"""

import pytest
import numpy as np
import pandas as pd

from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.diagnostics import (
    ResponseDiagnostics,
    DesignDiagnosticSummary,
    DesignQualityReport,
    Issue
)
from src.core.diagnostics.variance import (
    build_model_matrix,
    compute_prediction_variance,
    prediction_variance_stats,
    identify_high_variance_regions,
    compute_scaled_prediction_variance,
    assess_variance_uniformity
)
from src.core.diagnostics.estimability import (
    compute_vif,
    check_collinearity,
    compute_condition_number,
    assess_estimability,
    compute_leverage,
    identify_high_leverage_points
)
from src.core.diagnostics.summary import (
    compute_response_diagnostics,
    compute_design_diagnostic_summary,
    generate_quality_report
)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def simple_factors():
    """Create simple 2-factor design for testing."""
    return [
        Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
    ]


@pytest.fixture
def simple_design(simple_factors):
    """Create 2^2 factorial design."""
    design = pd.DataFrame({
        'A': [-1, 1, -1, 1],
        'B': [-1, -1, 1, 1]
    })
    return design


@pytest.fixture
def three_factor_design():
    """Create 2^3 factorial design."""
    factors = [
        Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
    ]
    
    design = pd.DataFrame({
        'A': [-1, 1, -1, 1, -1, 1, -1, 1],
        'B': [-1, -1, 1, 1, -1, -1, 1, 1],
        'C': [-1, -1, -1, -1, 1, 1, 1, 1]
    })
    
    return factors, design


@pytest.fixture
def mock_fitted_model():
    """Create mock fitted model for testing."""
    class MockModel:
        def __init__(self):
            self.rsquared = 0.85
            self.rsquared_adj = 0.82
            self.resid = np.array([0.5, -0.3, 0.2, -0.4])
            self.pvalues = pd.Series({
                'Intercept': 0.5,
                'A': 0.01,
                'B': 0.03,
                'A*B': 0.15
            })
    
    return MockModel()


# ============================================================
# TESTS: build_model_matrix
# ============================================================


class TestBuildModelMatrix:
    """Test model matrix construction."""
    
    def test_linear_terms(self, simple_design, simple_factors):
        """Test main effects only."""
        X = build_model_matrix(simple_design, simple_factors, ['1', 'A', 'B'])
        
        assert X.shape == (4, 3)
        assert np.allclose(X[:, 0], 1)  # Intercept
        assert np.allclose(X[:, 1], simple_design['A'])
        assert np.allclose(X[:, 2], simple_design['B'])
    
    def test_interaction_terms(self, simple_design, simple_factors):
        """Test interaction term."""
        X = build_model_matrix(simple_design, simple_factors, ['1', 'A', 'B', 'A*B'])
        
        assert X.shape == (4, 4)
        expected_interaction = simple_design['A'] * simple_design['B']
        assert np.allclose(X[:, 3], expected_interaction)
    
    def test_quadratic_terms(self, simple_design, simple_factors):
        """Test quadratic term."""
        X = build_model_matrix(simple_design, simple_factors, ['1', 'A', 'A^2'])
        
        assert X.shape == (4, 3)
        expected_quadratic = simple_design['A'] ** 2
        assert np.allclose(X[:, 2], expected_quadratic)


# ============================================================
# TESTS: Prediction Variance
# ============================================================


class TestPredictionVariance:
    """Test prediction variance computations."""
    
    def test_compute_prediction_variance(self, simple_design, simple_factors):
        """Test variance computation."""
        pred_var = compute_prediction_variance(
            simple_design, simple_factors, ['1', 'A', 'B'], sigma_squared=1.0
        )
        
        assert len(pred_var) == 4
        assert np.all(pred_var > 0)
    
    def test_variance_stats(self, simple_design, simple_factors):
        """Test variance statistics."""
        stats = prediction_variance_stats(
            simple_design, simple_factors, ['1', 'A', 'B'], sigma_squared=1.0
        )
        
        assert 'min' in stats
        assert 'max' in stats
        assert 'mean' in stats
        assert 'std' in stats
        assert 'max_ratio' in stats
        
        assert stats['min'] > 0
        assert stats['max'] >= stats['min']
        assert stats['max_ratio'] >= 1.0
    
    def test_identify_high_variance_regions(self, simple_design, simple_factors):
        """Test high variance region identification."""
        regions = identify_high_variance_regions(
            simple_design, simple_factors, ['1', 'A', 'B'], threshold=0.5
        )
        
        # Should be a list (may be empty or have entries)
        assert isinstance(regions, list)
        
        # If regions exist, check structure
        if regions:
            for region in regions:
                assert 'run_index' in region
                assert 'variance' in region
                assert 'variance_ratio' in region
    
    def test_scaled_prediction_variance(self, simple_design, simple_factors):
        """Test SPV computation."""
        spv = compute_scaled_prediction_variance(
            simple_design, simple_factors, ['1', 'A', 'B']
        )
        
        assert len(spv) == 4
        assert np.all(spv > 0)
        
        # For perfect design, mean SPV ≈ p (number of parameters)
        p = 3
        mean_spv = np.mean(spv)
        assert mean_spv >= p  # SPV >= p always
    
    def test_assess_variance_uniformity(self, simple_design, simple_factors):
        """Test variance uniformity assessment."""
        is_uniform, message = assess_variance_uniformity(
            simple_design, simple_factors, ['1', 'A', 'B']
        )
        
        assert isinstance(is_uniform, bool)
        assert isinstance(message, str)
        assert len(message) > 0


# ============================================================
# TESTS: VIF and Estimability
# ============================================================


class TestVIFAndEstimability:
    """Test VIF and estimability diagnostics."""
    
    def test_compute_vif_orthogonal_design(self, simple_design, simple_factors):
        """Test VIF on orthogonal design (should all be ~1)."""
        vif = compute_vif(simple_design, simple_factors, ['1', 'A', 'B'])
        
        assert 'A' in vif
        assert 'B' in vif
        
        # Orthogonal design → VIF ≈ 1
        assert abs(vif['A'] - 1.0) < 0.1
        assert abs(vif['B'] - 1.0) < 0.1
    
    def test_compute_vif_saturated_design(self, simple_design, simple_factors):
        """Test VIF correctly returns NaN for saturated design."""
        vif = compute_vif(simple_design, simple_factors, ['1', 'A', 'B', 'A*B'])
        
        # Saturated design → VIF cannot be computed
        assert all(np.isnan(v) for v in vif.values())
    
    def test_check_collinearity(self):
        """Test collinearity detection."""
        vif_values = {'A': 2.0, 'B': 15.0, 'C': 3.0}
        problematic = check_collinearity(vif_values, threshold=10.0)
        
        assert 'B' in problematic
        assert 'A' not in problematic
        assert 'C' not in problematic
    
    def test_condition_number(self, simple_design, simple_factors):
        """Test condition number computation."""
        kappa = compute_condition_number(
            simple_design, simple_factors, ['1', 'A', 'B']
        )
        
        assert kappa > 0
        assert not np.isinf(kappa)
        
        # Well-conditioned design should have κ < 100
        assert kappa < 100
    
    def test_assess_estimability_saturated(self):
        """Test estimability with saturated design."""
        # 3 runs, 3 parameters → saturated
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = pd.DataFrame({'A': [-1, 0, 1]})
        
        estimable, issues = assess_estimability(
            design, factors, ['1', 'A', 'A^2']
        )
        
        # Saturated but still estimable
        assert len(issues) > 0
        assert 'saturated' in issues[0].lower()
    
    def test_assess_estimability_supersaturated(self):
        """Test estimability with supersaturated design."""
        # 2 runs, 3 parameters → supersaturated
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        design = pd.DataFrame({'A': [-1, 1]})
        
        estimable, issues = assess_estimability(
            design, factors, ['1', 'A', 'A^2']
        )
        
        assert not estimable
        assert len(issues) > 0
        assert 'supersaturated' in issues[0].lower()
    
    def test_compute_leverage(self, simple_design, simple_factors):
        """Test leverage computation."""
        leverage = compute_leverage(
            simple_design, simple_factors, ['1', 'A', 'B']
        )
        
        assert len(leverage) == 4
        assert np.all(leverage > 0)
        assert np.all(leverage <= 1)
        
        # Sum of leverage = p (number of parameters)
        p = 3
        assert abs(np.sum(leverage) - p) < 0.01
    
    def test_identify_high_leverage_points(self, simple_design, simple_factors):
        """Test high leverage point identification."""
        high_lev = identify_high_leverage_points(
            simple_design, simple_factors, ['1', 'A', 'B']
        )
        
        # Should return list of indices
        assert isinstance(high_lev, list)


# ============================================================
# TESTS: Response Diagnostics
# ============================================================


class TestResponseDiagnostics:
    """Test response-level diagnostics."""
    
    def test_response_diagnostics_structure(self):
        """Test ResponseDiagnostics data structure."""
        diag = ResponseDiagnostics(response_name='Yield')
        
        assert diag.response_name == 'Yield'
        assert diag.needs_augmentation is False
        assert len(diag.issues) == 0
    
    def test_add_issue(self):
        """Test adding issues to diagnostics."""
        diag = ResponseDiagnostics(response_name='Yield')
        
        diag.add_issue(
            severity='critical',
            category='aliasing',
            description='Temperature aliased',
            affected_terms=['Temperature'],
            recommended_action='Foldover'
        )
        
        assert len(diag.issues) == 1
        assert diag.needs_augmentation is True
        assert len(diag.augmentation_reasons) == 1
    
    def test_get_primary_issue(self):
        """Test primary issue identification."""
        diag = ResponseDiagnostics(response_name='Yield')
        
        diag.add_issue('warning', 'precision', 'Low precision', [], 'Add runs')
        diag.add_issue('critical', 'aliasing', 'Aliased', [], 'Foldover')
        
        primary = diag.get_primary_issue()
        assert primary is not None
        assert primary.severity == 'critical'


# ============================================================
# TESTS: Design Diagnostic Summary
# ============================================================


class TestDesignDiagnosticSummary:
    """Test design-level diagnostic summary."""
    
    def test_summary_structure(self, simple_factors, simple_design):
        """Test DesignDiagnosticSummary structure."""
        diag1 = ResponseDiagnostics(response_name='Yield')
        diag2 = ResponseDiagnostics(response_name='Purity')
        
        summary = DesignDiagnosticSummary(
            design_type='factorial',
            n_runs=4,
            n_factors=2,
            response_diagnostics={'Yield': diag1, 'Purity': diag2},
            factors=simple_factors,
            original_design=simple_design
        )
        
        assert summary.design_type == 'factorial'
        assert summary.n_runs == 4
        assert len(summary.response_diagnostics) == 2
    
    def test_get_priority_response(self):
        """Test priority response identification."""
        diag1 = ResponseDiagnostics(response_name='Yield')
        diag1.add_issue('warning', 'precision', 'Low precision', [], 'Add runs')
        
        diag2 = ResponseDiagnostics(response_name='Purity')
        diag2.add_issue('critical', 'aliasing', 'Aliased', [], 'Foldover')
        
        summary = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': diag1, 'Purity': diag2}
        )
        
        priority = summary.get_priority_response()
        assert priority == 'Purity'  # Has critical issue
    
    def test_needs_any_augmentation(self):
        """Test augmentation need detection."""
        diag1 = ResponseDiagnostics(response_name='Yield')
        diag1.needs_augmentation = True
        
        diag2 = ResponseDiagnostics(response_name='Purity')
        diag2.needs_augmentation = False
        
        summary = DesignDiagnosticSummary(
            design_type='factorial',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': diag1, 'Purity': diag2}
        )
        
        assert summary.needs_any_augmentation() is True
    
    def test_get_unified_recommendation(self):
        """Test unified recommendation generation."""
        diag1 = ResponseDiagnostics(response_name='Yield')
        diag1.add_issue('critical', 'aliasing', 'Aliased', [], 'Foldover')
        
        summary = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': diag1}
        )
        
        recommendation = summary.get_unified_recommendation()
        assert 'foldover' in recommendation.lower()


# ============================================================
# TESTS: Quality Report Generation
# ============================================================


class TestQualityReport:
    """Test quality report generation."""
    
    def test_generate_quality_report(self, simple_factors, simple_design):
        """Test complete quality report generation."""
        diag = ResponseDiagnostics(response_name='Yield')
        diag.r_squared = 0.85
        diag.adj_r_squared = 0.82
        diag.rmse = 1.5
        diag.add_issue('warning', 'precision', 'Low precision', [], 'Add runs')
        
        summary = DesignDiagnosticSummary(
            design_type='factorial',
            n_runs=4,
            n_factors=2,
            response_diagnostics={'Yield': diag},
            factors=simple_factors,
            original_design=simple_design
        )
        
        report = generate_quality_report(summary)
        
        assert isinstance(report, DesignQualityReport)
        assert 'Yield' in report.response_quality
        assert len(report.warnings) > 0
    
    def test_report_markdown_format(self, simple_factors, simple_design):
        """Test markdown formatting."""
        diag = ResponseDiagnostics(response_name='Yield')
        diag.r_squared = 0.90
        
        summary = DesignDiagnosticSummary(
            design_type='factorial',
            n_runs=4,
            n_factors=2,
            response_diagnostics={'Yield': diag},
            factors=simple_factors,
            original_design=simple_design
        )
        
        report = generate_quality_report(summary)
        markdown = report.to_markdown()
        
        assert isinstance(markdown, str)
        assert '# Design Quality Report' in markdown
        assert 'Yield' in markdown
    
    def test_report_to_dict(self, simple_factors, simple_design):
        """Test dictionary serialization."""
        diag = ResponseDiagnostics(response_name='Yield')
        diag.r_squared = 0.85
        
        summary = DesignDiagnosticSummary(
            design_type='factorial',
            n_runs=4,
            n_factors=2,
            response_diagnostics={'Yield': diag},
            factors=simple_factors,
            original_design=simple_design
        )
        
        report = generate_quality_report(summary)
        report_dict = report.to_dict()
        
        assert isinstance(report_dict, dict)
        assert 'design_type' in report_dict
        assert 'response_quality' in report_dict


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestDiagnosticsIntegration:
    """Integration tests for complete diagnostic workflow."""
    
    def test_complete_diagnostic_workflow(self, three_factor_design, mock_fitted_model):
        """Test complete diagnostic computation workflow."""
        factors, design = three_factor_design
        
        # Simulate response
        response = np.array([10, 12, 11, 15, 9, 13, 10, 14])
        
        # Create metadata
        metadata = {
            'design_type': 'factorial',
            'is_split_plot': False,
            'has_blocking': False,
            'has_center_points': False
        }
        
        # Compute diagnostics
        diag = compute_response_diagnostics(
            design=design,
            response=response,
            response_name='Yield',
            factors=factors,
            model_terms=['1', 'A', 'B', 'C'],
            fitted_model=mock_fitted_model,
            design_metadata=metadata
        )
        
        assert diag.response_name == 'Yield'
        assert diag.r_squared > 0
        assert diag.rmse > 0
        assert len(diag.vif_values) > 0
    
    def test_multi_response_summary(self, three_factor_design):
        """Test multi-response diagnostic summary."""
        factors, design = three_factor_design
        
        # Simulate two responses
        responses = {
            'Yield': np.array([10, 12, 11, 15, 9, 13, 10, 14]),
            'Purity': np.array([95, 92, 94, 90, 96, 93, 95, 91])
        }
        
        # Create mock models
        class MockModel:
            def __init__(self, r2):
                self.rsquared = r2
                self.rsquared_adj = r2 - 0.05
                self.resid = np.random.normal(0, 1, 8)
                self.pvalues = pd.Series({
                    'Intercept': 0.5,
                    'A': 0.01,
                    'B': 0.05,
                    'C': 0.20
                })
        
        fitted_models = {
            'Yield': MockModel(0.85),
            'Purity': MockModel(0.75)
        }
        
        model_terms = {
            'Yield': ['1', 'A', 'B', 'C'],
            'Purity': ['1', 'A', 'B', 'C']
        }
        
        metadata = {
            'design_type': 'factorial',
            'is_split_plot': False,
            'has_blocking': False,
            'has_center_points': False
        }
        
        summary = compute_design_diagnostic_summary(
            design=design,
            responses=responses,
            fitted_models=fitted_models,
            factors=factors,
            model_terms_per_response=model_terms,
            design_metadata=metadata
        )
        
        assert len(summary.response_diagnostics) == 2
        assert 'Yield' in summary.response_diagnostics
        assert 'Purity' in summary.response_diagnostics