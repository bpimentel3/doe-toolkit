"""
Tests for I-Optimality Core Functionality - FIXED VERSION

Tests the fundamental I-optimality implementation including:
- Prediction grid generation
- I-criterion computation
- I-optimal design generation
- Backward compatibility with D-optimal
"""

import pytest
import numpy as np
import pandas as pd
from itertools import product
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.optimal import (
    generate_d_optimal_design,
    OptimizationResult,
    LinearConstraint,
    DOptimalityCriterion,
    IOptimalityCriterion,
    create_optimality_criterion,
    create_polynomial_builder,
)
from src.core.optimal.criteria import generate_prediction_grid
from src.core.optimal.design_generation import (
    generate_optimal_design,
    compute_i_efficiency,
)
from src.core.diagnostics.variance import (
    compute_i_criterion,
    compute_design_quality_metrics,
)


@pytest.fixture
def simple_factors():
    """Create simple 3-factor setup for testing."""
    return [
        Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
    ]


@pytest.fixture
def many_factors():
    """Create 6-factor setup for testing LHS grid."""
    return [
        Factor(f"X{i}", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        for i in range(1, 7)
    ]


class TestPredictionGrid:
    """Test prediction grid generation for I-optimality."""
    
    def test_factorial_grid_default(self, simple_factors):
        """Default grid should be factorial for 3 factors."""
        grid = generate_prediction_grid(simple_factors)
        
        # Default: 5 points per dimension, 3 factors -> 5^3 = 125 points
        assert grid.shape[1] == 3  # 3 factors
        assert grid.shape[0] == 125  # 5^3 points
        
        # All points should be in [-1, 1]
        assert np.all(grid >= -1)
        assert np.all(grid <= 1)
    
    def test_factorial_grid_custom_points(self, simple_factors):
        """Custom number of points per dimension."""
        config = {'n_points_per_dim': 3}
        grid = generate_prediction_grid(simple_factors, config)
        
        # 3^3 = 27 points
        assert grid.shape == (27, 3)
    
    def test_lhs_grid_many_factors(self, many_factors):
        """LHS should be auto-selected for k > 4."""
        grid = generate_prediction_grid(many_factors)
        
        # Should use LHS, not factorial (5^6 = 15625 would be too many)
        assert grid.shape[1] == 6  # 6 factors
        # LHS uses n^2 points by default (5^2 = 25)
        # Plus vertices and center point
        assert grid.shape[0] > 25  # At least 25 from LHS
        assert grid.shape[0] < 200  # But not factorial
    
    def test_explicit_lhs_grid(self, simple_factors):
        """Explicitly request LHS grid."""
        config = {'grid_type': 'lhs', 'n_points_per_dim': 5}
        grid = generate_prediction_grid(simple_factors, config)
        
        assert grid.shape[1] == 3
        # Should have LHS points plus vertices plus center
        assert grid.shape[0] > 25  # At least 5^2
        assert grid.shape[0] < 125  # Less than factorial
    
    def test_grid_includes_vertices(self, simple_factors):
        """Grid should include corner points."""
        config = {'include_vertices': True, 'n_points_per_dim': 3}
        grid = generate_prediction_grid(simple_factors, config)
        
        # Check that all 8 vertices are included (may have duplicates removed)
        vertices = np.array([
            [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]
        ])
        
        for vertex in vertices:
            # Check if this vertex exists in grid (within tolerance)
            distances = np.linalg.norm(grid - vertex, axis=1)
            assert np.min(distances) < 0.01, f"Vertex {vertex} not in grid"
    
    def test_grid_includes_center(self, simple_factors):
        """Grid should include center point."""
        config = {'include_center': True}
        grid = generate_prediction_grid(simple_factors, config)
        
        # Check for center point [0, 0, 0]
        center = np.zeros(3)
        distances = np.linalg.norm(grid - center, axis=1)
        assert np.min(distances) < 0.01, "Center point not in grid"


class TestIOptimalCriterion:
    """Test I-optimality criterion object."""
    
    def test_criterion_creation(self, simple_factors):
        """Create I-optimal criterion object."""
        model_builder = create_polynomial_builder(simple_factors, 'linear')
        prediction_points = generate_prediction_grid(simple_factors)
        
        criterion = IOptimalityCriterion(
            prediction_points=prediction_points,
            model_builder=model_builder
        )
        
        assert criterion is not None
        assert hasattr(criterion, 'M')  # Moment matrix should be precomputed
        assert criterion.M.shape[0] == 4  # 1 + 3 factors for linear model
    
    def test_objective_computation(self, simple_factors):
        """I-criterion objective should be computable."""
        model_builder = create_polynomial_builder(simple_factors, 'linear')
        prediction_points = generate_prediction_grid(simple_factors)
        
        criterion = IOptimalityCriterion(
            prediction_points=prediction_points,
            model_builder=model_builder
        )
        
        # Create a simple design
        design = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
        ])
        
        X_model = model_builder(design)
        objective = criterion.objective(X_model)
        
        # Objective should be negative (we return -I for maximization)
        assert objective < 0
        # Should be finite
        assert np.isfinite(objective)
    
    def test_criterion_factory(self, simple_factors):
        """Factory function should create correct criterion type."""
        model_builder = create_polynomial_builder(simple_factors, 'linear')
        
        # Create I-optimal criterion
        criterion_i = create_optimality_criterion(
            'I', model_builder, simple_factors
        )
        
        assert isinstance(criterion_i, IOptimalityCriterion)
        assert criterion_i.name == "I-optimal"


class TestBackwardCompatibility:
    """Test that D-optimal functionality is unchanged."""
    
    def test_d_optimal_still_works(self, simple_factors):
        """Old D-optimal API should still work."""
        design = generate_d_optimal_design(
            factors=simple_factors,
            model_type='linear',
            n_runs=12,
            seed=42
        )
        
        assert design.criterion_type == 'D-optimal'
        assert design.d_efficiency_vs_benchmark > 85
        assert design.design_actual.shape[0] == 12
    
    def test_generate_optimal_design_defaults_to_d(self, simple_factors):
        """New API should default to D-optimal."""
        design = generate_optimal_design(
            factors=simple_factors,
            model_type='linear',
            n_runs=12,
            seed=42
        )
        
        assert design.criterion_type == 'D-optimal'
        assert design.d_efficiency_vs_benchmark > 0
    
    def test_deprecated_function_warns(self, simple_factors):
        """Deprecated function should warn but still work."""
        # This test would check for deprecation warnings if we implement them
        # For now, just verify it works
        design = generate_d_optimal_design(
            factors=simple_factors,
            model_type='linear',
            n_runs=8,
            seed=42
        )
        
        assert design is not None


class TestDvsIDifference:
    """Test that D-optimal and I-optimal produce different designs."""
    
    def test_different_designs_quadratic(self, simple_factors):
        """D and I-optimal should differ for quadratic models."""
        seed = 42
        
        design_d = generate_optimal_design(
            factors=simple_factors,
            model_type='quadratic',
            n_runs=20,
            criterion='D',
            seed=seed
        )
        
        design_i = generate_optimal_design(
            factors=simple_factors,
            model_type='quadratic',
            n_runs=20,
            criterion='I',
            seed=seed
        )
        
        # Designs should be different
        assert not np.allclose(design_d.design_coded, design_i.design_coded)
        
        # Both should be valid
        assert design_d.criterion_type == 'D-optimal'
        assert design_i.criterion_type == 'I-optimal'
    
    def test_i_optimal_has_i_metrics(self, simple_factors):
        """I-optimal result should have I-optimality metrics."""
        design_i = generate_optimal_design(
            factors=simple_factors,
            model_type='linear',
            n_runs=12,
            criterion='I',
            seed=42
        )
        
        # Should have I-optimality fields
        assert design_i.i_criterion is not None
        assert design_i.i_efficiency_vs_benchmark is not None
        assert design_i.i_criterion > 0  # I-criterion should be positive


class TestIEfficiency:
    """Test I-efficiency computation."""
    
    def test_compute_i_efficiency(self, simple_factors):
        """I-efficiency should be computed correctly."""
        # Create a simple design
        design_coded = np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [-1, 1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [-1, 1, 1],
            [1, 1, 1],
            [0, 0, 0],
        ])
        
        model_builder = create_polynomial_builder(simple_factors, 'linear')
        
        efficiency = compute_i_efficiency(
            design_coded=design_coded,
            factors=simple_factors,
            model_type='linear',
            model_builder=model_builder,
            prediction_grid_config={'n_points_per_dim': 3}
        )
        
        # Should be a reasonable efficiency value
        assert 0 <= efficiency <= 200
        assert np.isfinite(efficiency)
    
    def test_i_efficiency_caps_at_200(self, simple_factors):
        """I-efficiency should cap at 200%."""
        # Create a very efficient design (full factorial)
        design_coded = np.array(list(product([-1, 1], repeat=3)))
        
        model_builder = create_polynomial_builder(simple_factors, 'linear')
        
        efficiency = compute_i_efficiency(
            design_coded=design_coded,
            factors=simple_factors,
            model_type='linear',
            model_builder=model_builder,
            prediction_grid_config={'n_points_per_dim': 3}
        )
        
        # Should be capped at 200%
        assert efficiency <= 200.0


class TestDiagnosticsIntegration:
    """Test I-optimality in diagnostics module."""
    
    def test_compute_i_criterion_diagnostic(self, simple_factors):
        """Diagnostics module should compute I-criterion."""
        # Create simple design
        design_df = pd.DataFrame({
            'A': [-1, 1, -1, 1, -1, 1, -1, 1],
            'B': [-1, -1, 1, 1, -1, -1, 1, 1],
            'C': [-1, -1, -1, -1, 1, 1, 1, 1],
        })
        
        model_terms = ['1', 'A', 'B', 'C']
        
        # Compute I-criterion with correct signature
        i_value = compute_i_criterion(design_df, simple_factors, model_terms)
        
        assert i_value > 0
        assert np.isfinite(i_value)
    
    def test_design_quality_metrics(self, simple_factors):
        """Quality metrics should include both D and I criteria."""
        design_df = pd.DataFrame({
            'A': [-1, 1, -1, 1, -1, 1, -1, 1],
            'B': [-1, -1, 1, 1, -1, -1, 1, 1],
            'C': [-1, -1, -1, -1, 1, 1, 1, 1],
        })
        
        model_terms = ['1', 'A', 'B', 'C']
        
        metrics = compute_design_quality_metrics(
            design_df, simple_factors, model_terms, include_i_optimal=True
        )
        
        # Should have both D and I metrics
        assert 'd_efficiency' in metrics
        assert 'condition_number' in metrics
        assert 'i_criterion' in metrics
        assert 'avg_prediction_variance' in metrics
        
        # All should be finite
        assert np.isfinite(metrics['i_criterion'])


class TestConstrainedIOptimal:
    """Test I-optimal with constraints."""
    
    def test_i_optimal_with_constraints(self, simple_factors):
        """I-optimal should work with linear constraints."""
        # Simple constraint: A + B <= 0.5
        constraints = [
            LinearConstraint(
                coefficients={'A': 1, 'B': 1},
                bound=0.5,
                constraint_type='le'
            )
        ]
        
        design = generate_optimal_design(
            factors=simple_factors,
            model_type='linear',
            n_runs=10,
            criterion='I',
            constraints=constraints,
            seed=42
        )
        
        # Verify constraint is satisfied
        for i in range(len(design.design_actual)):
            a = design.design_actual.iloc[i]['A']
            b = design.design_actual.iloc[i]['B']
            assert a + b <= 0.5 + 1e-6, f"Constraint violated at run {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
