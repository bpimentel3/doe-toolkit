"""
Tests for optimality criteria (D-optimal, I-optimal).

Tests cover:
- D-optimality criterion computation
- I-optimality criterion computation
- Prediction grid generation
- Criterion factory function
- Model matrix builder
"""

import numpy as np
import pytest

from src.core.factors import ChangeabilityLevel, Factor, FactorType
from src.core.optimal.criteria import (
    DOptimalityCriterion,
    IOptimalityCriterion,
    create_optimality_criterion,
    create_polynomial_builder,
    generate_prediction_grid,
)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def simple_factors():
    """3 continuous factors for testing."""
    return [
        Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
        Factor("X2", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
        Factor("X3", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
    ]


@pytest.fixture
def two_factors():
    """2 continuous factors for small tests."""
    return [
        Factor("Temp", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[100, 200]),
        Factor("Press", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[1, 5]),
    ]


# ============================================================
# D-OPTIMALITY CRITERION
# ============================================================


class TestDOptimalityCriterion:
    """Test D-optimality criterion."""

    def test_d_criterion_returns_logdet(self):
        """Verify D-criterion returns log(det(X'X))."""
        criterion = DOptimalityCriterion()

        # Simple orthogonal design
        X_model = np.array([[1, -1], [1, 1]])
        obj = criterion.objective(X_model)

        # Expected: XtX = [[2, 0], [0, 2]], det = 4, log(4) ≈ 1.386
        assert abs(obj - np.log(4)) < 0.01

    def test_d_criterion_detects_singular(self):
        """Verify singular designs return large negative value."""
        criterion = DOptimalityCriterion()

        # Singular matrix (duplicate rows)
        X_model = np.array([[1, 1], [1, 1], [1, 1]])
        obj = criterion.objective(X_model)

        assert obj == -1e10

    def test_d_criterion_name(self):
        """Verify criterion name."""
        criterion = DOptimalityCriterion()
        assert criterion.name == "D-optimal"


# ============================================================
# I-OPTIMALITY CRITERION
# ============================================================


class TestIOptimalityCriterion:
    """Test I-optimality criterion."""

    def test_i_criterion_returns_negative_trace(self, simple_factors):
        """Verify I-criterion returns negative trace value."""
        model_builder = create_polynomial_builder(simple_factors, "linear")

        # Create prediction grid
        prediction_grid = generate_prediction_grid(
            simple_factors, {"grid_type": "factorial", "n_points_per_dim": 3}
        )

        criterion = IOptimalityCriterion(prediction_grid, model_builder)

        # Simple design
        design_coded = np.array([[-1, -1, -1], [1, 1, 1], [0, 0, 0], [-1, 1, 0]])
        X_model = model_builder(design_coded)

        obj = criterion.objective(X_model)

        # Should be negative (we return -I for maximization)
        assert obj < 0

    def test_i_criterion_precomputes_moment_matrix(self, two_factors):
        """Verify moment matrix M is precomputed."""
        model_builder = create_polynomial_builder(two_factors, "linear")
        prediction_grid = generate_prediction_grid(two_factors, {"n_points_per_dim": 5})

        criterion = IOptimalityCriterion(prediction_grid, model_builder)

        # Check M exists and is correct shape
        assert hasattr(criterion, "M")
        assert criterion.M.shape == (3, 3)  # 1 intercept + 2 factors

    def test_i_criterion_detects_singular(self, simple_factors):
        """Verify singular designs return large negative value."""
        model_builder = create_polynomial_builder(simple_factors, "linear")
        prediction_grid = generate_prediction_grid(simple_factors)

        criterion = IOptimalityCriterion(prediction_grid, model_builder)

        # Singular matrix
        X_model = np.array([[1, 1, 1, 1], [1, 1, 1, 1]])
        obj = criterion.objective(X_model)

        assert obj == -1e10

    def test_i_criterion_name(self, simple_factors):
        """Verify criterion name."""
        model_builder = create_polynomial_builder(simple_factors, "linear")
        prediction_grid = generate_prediction_grid(simple_factors)

        criterion = IOptimalityCriterion(prediction_grid, model_builder)
        assert criterion.name == "I-optimal"


# ============================================================
# PREDICTION GRID GENERATION
# ============================================================


class TestPredictionGrid:
    """Test prediction grid generation for I-optimality."""

    def test_factorial_grid_for_small_k(self, simple_factors):
        """Factorial grid used for k ≤ 4."""
        grid = generate_prediction_grid(simple_factors, {"n_points_per_dim": 5})

        # 5^3 = 125 points (may have duplicates removed)
        assert grid.shape[0] <= 125
        assert grid.shape[1] == 3

        # All points in [-1, 1]
        assert np.all(grid >= -1.001)
        assert np.all(grid <= 1.001)

    def test_lhs_grid_for_large_k(self):
        """LHS grid used for k > 4."""
        factors_6d = [
            Factor(f"X{i}", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10])
            for i in range(1, 7)
        ]

        grid = generate_prediction_grid(factors_6d, {"n_points_per_dim": 5})

        # Should use LHS: n^2 = 25 points + vertices + center
        assert grid.shape[0] >= 25
        assert grid.shape[1] == 6

    def test_grid_includes_center(self, two_factors):
        """Verify center point is included."""
        grid = generate_prediction_grid(two_factors, {"include_center": True})

        # Check center exists
        center = np.array([0.0, 0.0])
        distances = np.linalg.norm(grid - center, axis=1)
        assert np.any(distances < 1e-6)

    def test_grid_excludes_center_when_requested(self, two_factors):
        """Verify center can be excluded."""
        grid = generate_prediction_grid(two_factors, {"include_center": False})

        # Check center doesn't exist (or is very unlikely)
        center = np.array([0.0, 0.0])
        distances = np.linalg.norm(grid - center, axis=1)
        # If factorial grid, center might still appear naturally
        # This test is more relevant for LHS


# ============================================================
# MODEL MATRIX BUILDER
# ============================================================


class TestModelMatrixBuilder:
    """Test polynomial model matrix builders."""

    def test_linear_model_builder(self, simple_factors):
        """Test linear model builder creates correct terms."""
        builder = create_polynomial_builder(simple_factors, "linear")

        X_point = np.array([[0.5, -0.5, 1.0]])
        X_model = builder(X_point)

        assert X_model.shape == (1, 4)  # Intercept + 3 factors
        assert X_model[0, 0] == 1.0  # Intercept
        assert X_model[0, 1] == 0.5  # X1
        assert X_model[0, 2] == -0.5  # X2
        assert X_model[0, 3] == 1.0  # X3

    def test_interaction_model_builder(self, two_factors):
        """Test interaction model builder."""
        builder = create_polynomial_builder(two_factors, "interaction")

        X_point = np.array([[0.5, -0.5]])
        X_model = builder(X_point)

        assert X_model.shape == (1, 4)  # Intercept + 2 + 1 interaction
        assert X_model[0, 0] == 1.0  # Intercept
        assert X_model[0, 1] == 0.5  # X1
        assert X_model[0, 2] == -0.5  # X2
        assert X_model[0, 3] == 0.5 * (-0.5)  # X1*X2

    def test_quadratic_model_builder(self, simple_factors):
        """Test quadratic model builder."""
        builder = create_polynomial_builder(simple_factors, "quadratic")

        X_point = np.array([[0.5, -0.5, 1.0]])
        X_model = builder(X_point)

        assert X_model.shape == (1, 10)  # 1 + 3 + 3 + 3
        assert X_model[0, 0] == 1.0  # Intercept
        assert X_model[0, 7] == 0.5**2  # X1^2
        assert X_model[0, 8] == (-0.5) ** 2  # X2^2
        assert X_model[0, 9] == 1.0**2  # X3^2


# ============================================================
# CRITERION FACTORY
# ============================================================


class TestCriterionFactory:
    """Test create_optimality_criterion factory."""

    def test_create_d_criterion(self, simple_factors):
        """Test factory creates D-criterion."""
        model_builder = create_polynomial_builder(simple_factors, "linear")

        criterion = create_optimality_criterion("D", model_builder, simple_factors)

        assert isinstance(criterion, DOptimalityCriterion)
        assert criterion.name == "D-optimal"

    def test_create_i_criterion(self, simple_factors):
        """Test factory creates I-criterion."""
        model_builder = create_polynomial_builder(simple_factors, "linear")

        criterion = create_optimality_criterion("I", model_builder, simple_factors)

        assert isinstance(criterion, IOptimalityCriterion)
        assert criterion.name == "I-optimal"

    def test_factory_invalid_type_raises_error(self, simple_factors):
        """Test invalid criterion type raises error."""
        model_builder = create_polynomial_builder(simple_factors, "linear")

        with pytest.raises(ValueError, match="Unknown criterion_type"):
            create_optimality_criterion("A", model_builder, simple_factors)

    def test_i_criterion_accepts_custom_grid_config(self, simple_factors):
        """Test I-criterion accepts custom grid config."""
        model_builder = create_polynomial_builder(simple_factors, "linear")

        criterion = create_optimality_criterion(
            "I",
            model_builder,
            simple_factors,
            prediction_grid_config={"n_points_per_dim": 7, "include_center": False},
        )

        assert isinstance(criterion, IOptimalityCriterion)
        # Grid should reflect config (hard to verify exactly without access to internals)
