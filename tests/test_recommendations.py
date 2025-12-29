"""
Tests for Augmentation Recommendation Engine.

Tests cover:
- Foldover strategy recommendation
- D-optimal augmentation recommendation
- Plan ranking by utility score
- Multi-response conflict resolution
- Edge cases and validation
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.diagnostics import (
    DesignDiagnosticSummary,
    ResponseDiagnostics,
    Issue
)
from src.core.augmentation.recommendations import (
    recommend_foldover_strategy,
    recommend_optimal_augmentation,
    recommend_augmentation_plans,
    rank_plans,
    resolve_multi_response_conflicts,
    assess_augmentation_necessity
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def factors_3():
    """Three continuous factors."""
    return [
        Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
        Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
        Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1])
    ]


@pytest.fixture
def factors_5():
    """Five continuous factors."""
    return [
        Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
        Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
        Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
        Factor("D", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
        Factor("E", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1])
    ]


@pytest.fixture
def fractional_design_8run():
    """8-run fractional factorial design."""
    return pd.DataFrame({
        'A': [-1, 1, -1, 1, -1, 1, -1, 1],
        'B': [-1, -1, 1, 1, -1, -1, 1, 1],
        'C': [-1, -1, -1, -1, 1, 1, 1, 1]
    })


@pytest.fixture
def fractional_design_16run():
    """16-run fractional factorial design."""
    from itertools import product
    
    # 2^4 full factorial
    points = list(product([-1, 1], repeat=4))
    
    return pd.DataFrame(points, columns=['A', 'B', 'C', 'D'])


# ============================================================================
# Test Foldover Recommendations
# ============================================================================

class TestFoldoverRecommendations:
    """Tests for foldover strategy recommendation."""
    
    def test_no_recommendation_for_non_fractional(self, factors_3, fractional_design_8run):
        """Should not recommend foldover for non-fractional designs."""
        diagnostics = DesignDiagnosticSummary(
            design_type='full_factorial',
            n_runs=8,
            n_factors=3,
            response_diagnostics={},
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        plan = recommend_foldover_strategy(diagnostics, {})
        
        assert plan is None
    
    def test_no_recommendation_without_generators(self, factors_3, fractional_design_8run):
        """Should not recommend foldover without generators."""
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={},
            generators=None,
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        plan = recommend_foldover_strategy(diagnostics, {})
        
        assert plan is None
    
    def test_single_factor_foldover_recommendation(self, factors_5, fractional_design_16run):
        """Should recommend single-factor foldover when one effect is aliased."""
        # Create diagnostic with one significant aliased effect
        response_diag = ResponseDiagnostics(response_name='Yield')
        response_diag.resolution = 3
        response_diag.significant_effects = ['A'] # Only A, not A and B
        response_diag.aliased_effects = {
            'A': ['BC', 'DE'],
            'B': ['AC', 'DE']
        }
        response_diag.add_issue(
            severity='critical',
            category='aliasing',
            description='A aliased with BC',
            affected_terms=['A'],
            recommended_action='Single-factor foldover on A'
        )
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=16,
            n_factors=5,
            response_diagnostics={'Yield': response_diag},
            generators=[('E', 'ABCD')],
            original_design=fractional_design_16run,
            factors=[
                Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("D", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("E", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1])
            ]
        )
        
        plan = recommend_foldover_strategy(diagnostics, {})
        
        assert plan is not None
        assert plan.strategy == 'foldover'
        assert plan.strategy_config.foldover_type == 'single_factor'
        assert plan.strategy_config.factor_to_fold in ['A', 'B']
        assert plan.n_runs_to_add == 16
    
    def test_full_foldover_recommendation(self, factors_5, fractional_design_16run):
        """Should recommend full foldover when multiple effects are aliased."""
        # Create diagnostic with multiple significant aliased effects
        response_diag = ResponseDiagnostics(response_name='Yield')
        response_diag.resolution = 3
        response_diag.significant_effects = ['A', 'B', 'C']
        response_diag.aliased_effects = {
            'A': ['BC'],
            'B': ['AC'],
            'C': ['AB']
        }
        response_diag.add_issue(
            severity='critical',
            category='aliasing',
            description='Multiple main effects aliased',
            affected_terms=['A', 'B', 'C'],
            recommended_action='Full foldover'
        )
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=16,
            n_factors=5,
            response_diagnostics={'Yield': response_diag},
            generators=[('E', 'ABCD')],
            original_design=fractional_design_16run,
            factors=[
                Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("D", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("E", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1])
            ]
        )
        
        plan = recommend_foldover_strategy(diagnostics, {})
        
        assert plan is not None
        assert plan.strategy == 'foldover'
        assert plan.strategy_config.foldover_type == 'full'
        assert plan.n_runs_to_add == 16
        assert 'resolution' in plan.expected_improvements
    
    def test_full_foldover_for_low_resolution(self, factors_5, fractional_design_16run):
        """Should recommend full foldover for Resolution III even without significant effects."""
        # Create diagnostic with low resolution but no significant effects
        response_diag = ResponseDiagnostics(response_name='Yield')
        response_diag.resolution = 3
        response_diag.significant_effects = []
        response_diag.aliased_effects = {'A': ['BC']}
        response_diag.add_issue(
            severity='warning',
            category='aliasing',
            description='Low resolution design',
            affected_terms=['A'],
            recommended_action='Consider foldover'
        )
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=16,
            n_factors=5,
            response_diagnostics={'Yield': response_diag},
            generators=[('E', 'ABCD')],
            original_design=fractional_design_16run,
            factors=[
                Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("D", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("E", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1])
            ]
        )
        
        plan = recommend_foldover_strategy(diagnostics, {})
        
        assert plan is not None
        assert plan.strategy_config.foldover_type == 'full'


# ============================================================================
# Test D-Optimal Recommendations
# ============================================================================

class TestOptimalRecommendations:
    """Tests for D-optimal augmentation recommendation."""
    
    def test_no_recommendation_when_fit_good(self, factors_3, fractional_design_8run):
        """Should not recommend optimal augmentation when model fits well."""
        response_diag = ResponseDiagnostics(response_name='Yield')
        response_diag.r_squared = 0.95
        response_diag.lack_of_fit_p_value = 0.45
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': response_diag},
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        plan = recommend_optimal_augmentation(diagnostics, {})
        
        assert plan is None
    
    def test_recommend_for_lack_of_fit(self, factors_3, fractional_design_8run):
        """Should recommend D-optimal when lack of fit detected."""
        response_diag = ResponseDiagnostics(response_name='Yield')
        response_diag.r_squared = 0.75
        response_diag.lack_of_fit_p_value = 0.008
        response_diag.add_issue(
            severity='critical',
            category='lack_of_fit',
            description='Lack of fit detected',
            affected_terms=[],
            recommended_action='Add quadratic terms'
        )
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': response_diag},
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        plan = recommend_optimal_augmentation(diagnostics, {})
        
        assert plan is not None
        assert plan.strategy == 'd_optimal'
        assert plan.n_runs_to_add > 0
        assert 'quadratic' in plan.expected_improvements.get('capability', '').lower()
    
    def test_recommend_for_poor_r_squared(self, factors_3, fractional_design_8run):
        """Should recommend D-optimal when R² is poor."""
        response_diag = ResponseDiagnostics(response_name='Yield')
        response_diag.r_squared = 0.55
        response_diag.lack_of_fit_p_value = None
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': response_diag},
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        plan = recommend_optimal_augmentation(diagnostics, {})
        
        assert plan is not None
        assert plan.strategy == 'd_optimal'
    
    def test_no_recommendation_without_continuous_factors(self, fractional_design_8run):
        """Should not recommend quadratic extension without continuous factors."""
        categorical_factors = [
            Factor("A", FactorType.CATEGORICAL, ChangeabilityLevel.EASY, ['Low', 'High']),
            Factor("B", FactorType.CATEGORICAL, ChangeabilityLevel.EASY, ['X', 'Y']),
            Factor("C", FactorType.CATEGORICAL, ChangeabilityLevel.EASY, ['1', '2'])
        ]
        
        response_diag = ResponseDiagnostics(response_name='Yield')
        response_diag.r_squared = 0.55
        response_diag.lack_of_fit_p_value = 0.01
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': response_diag},
            original_design=fractional_design_8run,
            factors=categorical_factors
        )
        
        plan = recommend_optimal_augmentation(diagnostics, {})
        
        assert plan is None


# ============================================================================
# Test Plan Ranking
# ============================================================================

class TestPlanRanking:
    """Tests for augmentation plan ranking algorithm."""
    
    def test_rank_by_utility_score(self, factors_3, fractional_design_8run):
        """Should rank plans by utility score."""
        from src.core.augmentation.plan import (
            AugmentationPlan,
            FoldoverConfig,
            OptimalAugmentConfig,
            create_plan_id
        )
        
        # Create multiple plans
        plan1 = AugmentationPlan(
            plan_id=create_plan_id(),
            plan_name="Full Foldover",
            strategy='foldover',
            strategy_config=FoldoverConfig('full'),
            original_design=fractional_design_8run,
            factors=factors_3,
            n_runs_to_add=8,
            total_runs_after=16,
            expected_improvements={'resolution': '3 → 4'},
            benefits_responses=['Yield'],
            primary_beneficiary='Yield',
            experimental_cost=8.0,
            utility_score=0.0,
            rank=1
        )
        
        plan2 = AugmentationPlan(
            plan_id=create_plan_id(),
            plan_name="D-Optimal",
            strategy='d_optimal',
            strategy_config=OptimalAugmentConfig(['A', 'B'], 6),
            original_design=fractional_design_8run,
            factors=factors_3,
            n_runs_to_add=6,
            total_runs_after=14,
            expected_improvements={'model': 'extended'},
            benefits_responses=['Yield', 'Purity'],
            primary_beneficiary='Yield',
            experimental_cost=6.0,
            utility_score=0.0,
            rank=1
        )
        
        # Create diagnostics with critical issues
        response_diag = ResponseDiagnostics(response_name='Yield')
        response_diag.add_issue(
            severity='critical',
            category='aliasing',
            description='Critical aliasing',
            affected_terms=['A'],
            recommended_action='Foldover'
        )
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': response_diag, 'Purity': response_diag},
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        # Rank plans
        ranked = rank_plans([plan1, plan2], diagnostics)
        
        assert len(ranked) == 2
        assert ranked[0].utility_score > 0
        assert ranked[1].utility_score > 0
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2
    
    def test_higher_severity_ranks_higher(self, factors_3, fractional_design_8run):
        """Plans addressing critical issues should rank higher than warnings."""
        from src.core.augmentation.plan import (
            AugmentationPlan,
            FoldoverConfig,
            create_plan_id
        )
        
        plan_critical = AugmentationPlan(
            plan_id=create_plan_id(),
            plan_name="Addresses Critical",
            strategy='foldover',
            strategy_config=FoldoverConfig('full'),
            original_design=fractional_design_8run,
            factors=factors_3,
            n_runs_to_add=8,
            total_runs_after=16,
            expected_improvements={},
            benefits_responses=['Critical'],
            primary_beneficiary='Critical',
            experimental_cost=8.0,
            utility_score=0.0,
            rank=1
        )
        
        plan_warning = AugmentationPlan(
            plan_id=create_plan_id(),
            plan_name="Addresses Warning",
            strategy='foldover',
            strategy_config=FoldoverConfig('full'),
            original_design=fractional_design_8run,
            factors=factors_3,
            n_runs_to_add=8,
            total_runs_after=16,
            expected_improvements={},
            benefits_responses=['Warning'],
            primary_beneficiary='Warning',
            experimental_cost=8.0,
            utility_score=0.0,
            rank=1
        )
        
        # Critical issue
        diag_critical = ResponseDiagnostics(response_name='Critical')
        diag_critical.add_issue('critical', 'aliasing', 'Critical', [], 'Fix')
        
        # Warning issue
        diag_warning = ResponseDiagnostics(response_name='Warning')
        diag_warning.add_issue('warning', 'precision', 'Warning', [], 'Improve')
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={
                'Critical': diag_critical,
                'Warning': diag_warning
            },
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        ranked = rank_plans([plan_warning, plan_critical], diagnostics)
        
        # Critical should rank first
        assert ranked[0].benefits_responses == ['Critical']
        assert ranked[0].utility_score > ranked[1].utility_score


# ============================================================================
# Test Main Recommendation Function
# ============================================================================

class TestRecommendAugmentationPlans:
    """Tests for main recommendation function."""
    
    def test_no_plans_when_no_issues(self, factors_3, fractional_design_8run):
        """Should return empty list when no augmentation needed."""
        response_diag = ResponseDiagnostics(response_name='Yield')
        response_diag.r_squared = 0.95
        response_diag.needs_augmentation = False
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': response_diag},
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        plans = recommend_augmentation_plans(diagnostics, {})
        
        assert len(plans) == 0
    
    def test_returns_ranked_plans(self, factors_5, fractional_design_16run):
        """Should return ranked plans addressing issues."""
        # Create diagnostics with both aliasing and LOF
        response_diag = ResponseDiagnostics(response_name='Yield')
        response_diag.resolution = 3
        response_diag.significant_effects = ['A']
        response_diag.aliased_effects = {'A': ['BC']}
        response_diag.lack_of_fit_p_value = 0.01
        response_diag.r_squared = 0.72
        response_diag.needs_augmentation = True
        
        response_diag.add_issue(
            severity='critical',
            category='aliasing',
            description='A aliased',
            affected_terms=['A'],
            recommended_action='Foldover'
        )
        response_diag.add_issue(
            severity='warning',
            category='lack_of_fit',
            description='LOF detected',
            affected_terms=[],
            recommended_action='Add terms'
        )
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=16,
            n_factors=5,
            response_diagnostics={'Yield': response_diag},
            generators=[('E', 'ABCD')],
            original_design=fractional_design_16run,
            factors=[
                Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("D", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("E", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1])
            ]
        )
        
        plans = recommend_augmentation_plans(diagnostics, {})
        
        assert len(plans) > 0
        assert len(plans) <= 3
        assert all(p.utility_score > 0 for p in plans)
        assert plans[0].rank == 1
    
    def test_respects_budget_constraint(self, factors_5, fractional_design_16run):
        """Should filter plans exceeding budget constraint."""
        response_diag = ResponseDiagnostics(response_name='Yield')
        response_diag.resolution = 3
        response_diag.significant_effects = ['A', 'B']
        response_diag.aliased_effects = {'A': ['BC']}
        response_diag.needs_augmentation = True
        response_diag.add_issue('critical', 'aliasing', 'Aliased', ['A'], 'Fix')
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=16,
            n_factors=5,
            response_diagnostics={'Yield': response_diag},
            generators=[('E', 'ABCD')],
            original_design=fractional_design_16run,
            factors=[
                Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("D", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
                Factor("E", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1])
            ]
        )
        
        # Budget only allows 10 runs (foldover needs 16)
        plans = recommend_augmentation_plans(diagnostics, {}, budget_constraint=10)
        
        # Should filter out plans requiring > 10 runs
        assert all(p.n_runs_to_add <= 10 for p in plans)


# ============================================================================
# Test Multi-Response Conflict Resolution
# ============================================================================

class TestConflictResolution:
    """Tests for multi-response conflict resolution."""
    
    def test_no_conflict_message(self, factors_3, fractional_design_8run):
        """Should indicate no conflict when responses agree."""
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={},
            conflicting_recommendations=False,
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        message = resolve_multi_response_conflicts(diagnostics)
        
        assert 'no conflict' in message.lower()
    
    def test_critical_issue_priority(self, factors_3, fractional_design_8run):
        """Should prioritize critical issues in conflict resolution."""
        diag1 = ResponseDiagnostics(response_name='Yield')
        diag1.add_issue('critical', 'aliasing', 'Critical aliasing', ['A'], 'Foldover')
        
        diag2 = ResponseDiagnostics(response_name='Purity')
        diag2.add_issue('warning', 'lack_of_fit', 'LOF', [], 'Add terms')
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': diag1, 'Purity': diag2},
            conflicting_recommendations=True,
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        message = resolve_multi_response_conflicts(diagnostics)
        
        assert 'aliasing' in message.lower()
        assert 'critical' in message.lower()


# ============================================================================
# Test Augmentation Necessity Assessment
# ============================================================================

class TestAugmentationNecessity:
    """Tests for assessing whether augmentation is necessary."""
    
    def test_necessary_with_critical_issues(self, factors_3, fractional_design_8run):
        """Should assess augmentation as necessary with critical issues."""
        diag = ResponseDiagnostics(response_name='Yield')
        diag.add_issue('critical', 'aliasing', 'Critical', ['A'], 'Fix')
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': diag},
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        necessary, reason = assess_augmentation_necessity(diagnostics)
        
        assert necessary is True
        assert 'critical' in reason.lower()
    
    def test_necessary_with_multiple_warnings(self, factors_3, fractional_design_8run):
        """Should assess augmentation as necessary with 2+ warnings."""
        diag = ResponseDiagnostics(response_name='Yield')
        diag.add_issue('warning', 'precision', 'Warning 1', [], 'Fix')
        diag.add_issue('warning', 'estimability', 'Warning 2', [], 'Fix')
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': diag},
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        necessary, reason = assess_augmentation_necessity(diagnostics)
        
        assert necessary is True
        assert 'warning' in reason.lower()
    
    def test_not_necessary_without_issues(self, factors_3, fractional_design_8run):
        """Should assess augmentation as not necessary without issues."""
        diag = ResponseDiagnostics(response_name='Yield')
        diag.r_squared = 0.95
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': diag},
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        necessary, reason = assess_augmentation_necessity(diagnostics)
        
        assert necessary is False
        assert 'acceptable' in reason.lower()
    
    def test_necessary_with_poor_fit(self, factors_3, fractional_design_8run):
        """Should assess augmentation as necessary with poor R²."""
        diag = ResponseDiagnostics(response_name='Yield')
        diag.r_squared = 0.55
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=8,
            n_factors=3,
            response_diagnostics={'Yield': diag},
            original_design=fractional_design_8run,
            factors=factors_3
        )
        
        necessary, reason = assess_augmentation_necessity(diagnostics)
        
        assert necessary is True
        assert 'poor' in reason.lower() or 'r²' in reason.lower()


# ============================================================================
# Integration Tests
# ============================================================================

class TestRecommendationIntegration:
    """Integration tests for complete recommendation workflow."""
    
    def test_complete_workflow_fractional_design(self, factors_5):
        """Test complete workflow from diagnostics to ranked plans."""
        # Create realistic fractional factorial scenario
        from itertools import product
        
        design_points = list(product([-1, 1], repeat=4))
        design = pd.DataFrame(design_points, columns=['A', 'B', 'C', 'D'])
        
        # Add aliasing issue
        diag_yield = ResponseDiagnostics(response_name='Yield')
        diag_yield.resolution = 3
        diag_yield.significant_effects = ['A', 'B']
        diag_yield.aliased_effects = {'A': ['BC'], 'B': ['AC']}
        diag_yield.r_squared = 0.82
        diag_yield.needs_augmentation = True
        diag_yield.add_issue(
            'critical', 'aliasing',
            'Multiple effects aliased',
            ['A', 'B'],
            'Full foldover'
        )
        
        # Add LOF issue
        diag_purity = ResponseDiagnostics(response_name='Purity')
        diag_purity.lack_of_fit_p_value = 0.002
        diag_purity.r_squared = 0.68
        diag_purity.needs_augmentation = True
        diag_purity.add_issue(
            'warning', 'lack_of_fit',
            'Lack of fit detected',
            [],
            'Add quadratic terms'
        )
        
        diagnostics = DesignDiagnosticSummary(
            design_type='fractional',
            n_runs=16,
            n_factors=4,
            response_diagnostics={
                'Yield': diag_yield,
                'Purity': diag_purity
            },
            generators=[('D', 'ABC')],
            original_design=design,
            factors=factors_5[:4],  # Use first 4 factors
            conflicting_recommendations=True
        )
        
        # Get recommendations
        plans = recommend_augmentation_plans(diagnostics, {})
        
        # Verify we get ranked plans
        assert len(plans) > 0
        assert plans[0].rank == 1
        assert plans[0].utility_score > 0
        
        # Verify plans address issues
        strategies = {p.strategy for p in plans}
        assert 'foldover' in strategies or 'd_optimal' in strategies
        
        # Get conflict resolution advice
        advice = resolve_multi_response_conflicts(diagnostics)
        assert len(advice) > 0
        assert 'aliasing' in advice.lower() or 'critical' in advice.lower()