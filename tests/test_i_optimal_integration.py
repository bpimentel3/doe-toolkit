"""
Integration Tests for I-Optimal Augmentation.

Tests the augmentation system with I-optimality including:
- I-optimal augmentation plans
- Goal-driven I-optimal recommendations
- OptimalAugmentConfig with I-criterion
"""

import pytest
import numpy as np
import pandas as pd
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.augmentation.optimal import augment_for_model_extension
from src.core.augmentation.plan import OptimalAugmentConfig
from src.core.augmentation.goals import AugmentationGoal
from src.core.augmentation.goal_driven import (
    GoalDrivenContext,
    recommend_from_goal,
    _create_strategy_config,
)
from src.core.diagnostics import DesignDiagnosticSummary


@pytest.fixture
def simple_factors():
    """Create simple 3-factor setup."""
    return [
        Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
    ]


@pytest.fixture
def simple_design():
    """Create simple factorial design."""
    return pd.DataFrame({
        'A': [-1, 1, -1, 1, -1, 1, -1, 1],
        'B': [-1, -1, 1, 1, -1, -1, 1, 1],
        'C': [-1, -1, -1, -1, 1, 1, 1, 1],
    })


class TestOptimalAugmentConfig:
    """Test OptimalAugmentConfig with I-optimality."""
    
    def test_config_defaults_to_d_optimal(self):
        """Config should default to D-optimal."""
        config = OptimalAugmentConfig(
            new_model_terms=['1', 'A', 'B'],
            n_runs_to_add=5
        )
        
        assert config.criterion == 'D'
        assert config.prediction_grid_config is None
    
    def test_config_accepts_i_optimal(self):
        """Config should accept I-optimal criterion."""
        config = OptimalAugmentConfig(
            new_model_terms=['1', 'A', 'B'],
            n_runs_to_add=5,
            criterion='I',
            prediction_grid_config={'n_points_per_dim': 7}
        )
        
        assert config.criterion == 'I'
        assert config.prediction_grid_config == {'n_points_per_dim': 7}


class TestAugmentForModelExtension:
    """Test augment_for_model_extension with I-optimality."""
    
    def test_d_optimal_augmentation_still_works(self, simple_design, simple_factors):
        """D-optimal augmentation should still work (backward compat)."""
        current_terms = ['1', 'A', 'B', 'C']
        new_terms = ['1', 'A', 'B', 'C', 'A*B', 'A*C', 'B*C']
        
        augmented = augment_for_model_extension(
            original_design=simple_design,
            factors=simple_factors,
            current_model_terms=current_terms,
            new_model_terms=new_terms,
            n_runs_to_add=6,
            criterion='D',
            seed=42
        )
        
        assert augmented.n_runs_added == 6
        assert augmented.n_runs_total == 14  # 8 original + 6 new
    
    def test_i_optimal_augmentation(self, simple_design, simple_factors):
        """I-optimal augmentation should work."""
        current_terms = ['1', 'A', 'B', 'C']
        new_terms = ['1', 'A', 'B', 'C', 'A*B', 'A*C', 'B*C']
        
        augmented = augment_for_model_extension(
            original_design=simple_design,
            factors=simple_factors,
            current_model_terms=current_terms,
            new_model_terms=new_terms,
            n_runs_to_add=6,
            criterion='I',
            prediction_grid_config={'n_points_per_dim': 5},
            seed=42
        )
        
        assert augmented.n_runs_added == 6
        assert augmented.n_runs_total == 14
    
    def test_d_vs_i_augmentation_differ(self, simple_design, simple_factors):
        """D and I augmentation should produce different run selections."""
        current_terms = ['1', 'A', 'B', 'C']
        new_terms = ['1', 'A', 'B', 'C', 'I(A**2)', 'I(B**2)', 'I(C**2)']
        
        seed = 42
        
        augmented_d = augment_for_model_extension(
            original_design=simple_design,
            factors=simple_factors,
            current_model_terms=current_terms,
            new_model_terms=new_terms,
            n_runs_to_add=8,
            criterion='D',
            seed=seed
        )
        
        augmented_i = augment_for_model_extension(
            original_design=simple_design,
            factors=simple_factors,
            current_model_terms=current_terms,
            new_model_terms=new_terms,
            n_runs_to_add=8,
            criterion='I',
            prediction_grid_config={'n_points_per_dim': 5},
            seed=seed
        )
        
        # Extract new runs
        new_runs_d = augmented_d.new_runs_only[['A', 'B', 'C']].values
        new_runs_i = augmented_i.new_runs_only[['A', 'B', 'C']].values
        
        # Should produce different selections
        assert not np.allclose(new_runs_d, new_runs_i), \
            "D and I augmentation should select different runs"


class TestGoalDrivenIOptimal:
    """Test goal-driven augmentation with I-optimality."""
    
    def test_create_i_optimal_strategy_config(self, simple_factors):
        """_create_strategy_config should create I-optimal config."""
        # Mock diagnostics
        class MockDiagnostics:
            factors = simple_factors
            n_factors = len(simple_factors)
            n_runs = 8
            design_type = 'full_factorial'
            response_diagnostics = {
                'Y': type('obj', (object,), {
                    'significant_effects': ['A', 'B'],
                    'marginally_significant': ['C']
                })()
            }
        
        config = _create_strategy_config(
            strategy_name='i_optimal',
            goal=AugmentationGoal.IMPROVE_PREDICTION,
            n_runs=10,
            diagnostics=MockDiagnostics(),
            user_adjustments={}
        )
        
        assert isinstance(config, OptimalAugmentConfig)
        assert config.criterion == 'I'
        assert config.prediction_grid_config is not None
        assert 'n_points_per_dim' in config.prediction_grid_config
        assert config.n_runs_to_add == 10
    
    def test_improve_prediction_goal_uses_i_optimal(self, simple_factors):
        """IMPROVE_PREDICTION goal should recommend I-optimal."""
        # Mock diagnostics
        class MockResponseDiag:
            significant_effects = ['A', 'B']
            marginally_significant = ['C']
            aliased_effects = {}
            resolution = None
            lack_of_fit_p_value = None
            prediction_variance_stats = {'max': 1.0, 'mean': 0.3}
            issues = []
        
        class MockDiagnostics:
            def __init__(self):
                self.factors = simple_factors
                self.n_factors = len(simple_factors)
                self.n_runs = 8
                self.design_type = 'full_factorial'
                self.has_center_points = False
                self.response_diagnostics = {'Y': MockResponseDiag()}
                self.original_design = pd.DataFrame({
                    'A': [-1, 1, -1, 1, -1, 1, -1, 1],
                    'B': [-1, -1, 1, 1, -1, -1, 1, 1],
                    'C': [-1, -1, -1, -1, 1, 1, 1, 1],
                })
        
        context = GoalDrivenContext(
            selected_goal=AugmentationGoal.IMPROVE_PREDICTION,
            design_diagnostics=MockDiagnostics()
        )
        
        plans = recommend_from_goal(context)
        
        # Should return plans
        assert len(plans) > 0
        
        # Primary plan should be I-optimal
        primary_plan = plans[0]
        config = primary_plan.strategy_config
        
        assert isinstance(config, OptimalAugmentConfig)
        assert config.criterion == 'I'


class TestPredictionGridConfigDefaults:
    """Test automatic prediction grid configuration."""
    
    def test_factorial_grid_for_few_factors(self):
        """Should use factorial grid for k <= 4."""
        factors_3 = [
            Factor(f"X{i}", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(1, 4)
        ]
        
        class MockDiag:
            factors = factors_3
            n_factors = len(factors_3)
            response_diagnostics = {
                'Y': type('obj', (object,), {
                    'significant_effects': ['X1'],
                    'marginally_significant': []
                })()
            }
        
        config = _create_strategy_config(
            strategy_name='i_optimal',
            goal=AugmentationGoal.IMPROVE_PREDICTION,
            n_runs=10,
            diagnostics=MockDiag(),
            user_adjustments={}
        )
        
        assert config.prediction_grid_config['grid_type'] == 'factorial'
    
    def test_lhs_grid_for_many_factors(self):
        """Should use LHS grid for k > 4."""
        factors_6 = [
            Factor(f"X{i}", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(1, 7)
        ]
        
        class MockDiag:
            factors = factors_6
            n_factors = len(factors_6)
            response_diagnostics = {
                'Y': type('obj', (object,), {
                    'significant_effects': ['X1'],
                    'marginally_significant': []
                })()
            }
        
        config = _create_strategy_config(
            strategy_name='i_optimal',
            goal=AugmentationGoal.IMPROVE_PREDICTION,
            n_runs=10,
            diagnostics=MockDiag(),
            user_adjustments={}
        )
        
        assert config.prediction_grid_config['grid_type'] == 'lhs'
    
    def test_user_override_prediction_grid(self):
        """User adjustments should override defaults."""
        factors_3 = [
            Factor(f"X{i}", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(1, 4)
        ]
        
        class MockDiag:
            factors = factors_3
            n_factors = len(factors_3)
            response_diagnostics = {
                'Y': type('obj', (object,), {
                    'significant_effects': ['X1'],
                    'marginally_significant': []
                })()
            }
        
        user_adjustments = {
            'prediction_grid_config': {
                'n_points_per_dim': 7,
                'grid_type': 'lhs'
            }
        }
        
        config = _create_strategy_config(
            strategy_name='i_optimal',
            goal=AugmentationGoal.IMPROVE_PREDICTION,
            n_runs=10,
            diagnostics=MockDiag(),
            user_adjustments=user_adjustments
        )
        
        assert config.prediction_grid_config['n_points_per_dim'] == 7
        assert config.prediction_grid_config['grid_type'] == 'lhs'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
