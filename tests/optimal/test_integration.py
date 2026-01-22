"""
Integration tests for complete D-optimal design generation workflow.

Tests cover end-to-end scenarios:
- Basic design generation
- Constraint handling
- Design quality metrics
- Realistic usage patterns
"""

import numpy as np
import pytest

from src.core.factors import ChangeabilityLevel, Factor, FactorType
from src.core.optimal import (
    CandidatePoolConfig,
    LinearConstraint,
    OptimizerConfig,
    OptimizationResult,
    generate_d_optimal_design,
)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def simple_factors():
    """3 continuous factors for basic testing."""
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


@pytest.fixture
def fast_config():
    """Fast optimizer config for testing."""
    return OptimizerConfig(
        max_iterations=50,
        relative_improvement_tolerance=1e-3,
        stability_window=10,
        n_random_starts=2,
        max_candidates_per_row=30,
    )


# ============================================================
# BASIC FUNCTIONALITY
# ============================================================


class TestBasicDesignGeneration:
    """Test basic D-optimal design generation."""

    def test_linear_model_no_constraints(self, simple_factors):
        """Generate simple linear model design."""
        result = generate_d_optimal_design(
            factors=simple_factors, model_type="linear", n_runs=8, seed=42
        )

        assert isinstance(result, OptimizationResult)
        assert result.n_runs == 8
        assert result.n_parameters == 4  # 1 + 3 factors
        assert result.design_coded.shape == (8, 3)
        assert result.design_actual.shape[0] == 8
        assert "StdOrder" in result.design_actual.columns
        assert "RunOrder" in result.design_actual.columns

        # All coded values in [-1, 1]
        assert np.all(result.design_coded >= -1.001)
        assert np.all(result.design_coded <= 1.001)

        # Actual values in factor bounds
        for i, factor in enumerate(simple_factors):
            assert np.all(result.design_actual[factor.name] >= factor.min_value - 1e-6)
            assert np.all(result.design_actual[factor.name] <= factor.max_value + 1e-6)

    def test_quadratic_model(self, simple_factors):
        """Generate quadratic model design."""
        result = generate_d_optimal_design(
            factors=simple_factors, model_type="quadratic", n_runs=15, seed=42
        )

        assert result.n_parameters == 10  # 1 + 3 + 3 + 3
        assert result.model_matrix.shape == (15, 10)

        # Quadratic terms should be >= 0 (squares)
        X = result.model_matrix
        for col_idx in range(7, 10):
            assert np.all(X[:, col_idx] >= -1e-10)

    def test_reproducibility_with_seed(self, two_factors):
        """Same seed produces identical designs."""
        result1 = generate_d_optimal_design(
            factors=two_factors, model_type="linear", n_runs=6, seed=12345
        )

        result2 = generate_d_optimal_design(
            factors=two_factors, model_type="linear", n_runs=6, seed=12345
        )

        np.testing.assert_array_almost_equal(
            result1.design_coded, result2.design_coded, decimal=10
        )
        assert result1.final_objective == result2.final_objective


# ============================================================
# CONSTRAINT HANDLING
# ============================================================


class TestConstraintHandling:
    """Test designs with linear constraints."""

    def test_simple_sum_constraint(self, simple_factors):
        """Test: X1 + X2 <= 15 in actual units."""
        constraint = LinearConstraint(
            coefficients={"X1": 1.0, "X2": 1.0}, bound=15.0, constraint_type="le"
        )

        result = generate_d_optimal_design(
            factors=simple_factors,
            model_type="linear",
            n_runs=8,
            constraints=[constraint],
            seed=42,
        )

        for idx in range(result.n_runs):
            x1 = result.design_actual.iloc[idx]["X1"]
            x2 = result.design_actual.iloc[idx]["X2"]
            assert x1 + x2 <= 15.0 + 1e-6

    def test_multiple_constraints(self, simple_factors):
        """Test multiple simultaneous constraints."""
        constraints = [
            LinearConstraint(
                coefficients={"X1": 1.0, "X2": 1.0}, bound=12.0, constraint_type="le"
            ),
            LinearConstraint(coefficients={"X1": 1.0}, bound=3.0, constraint_type="ge"),
            LinearConstraint(coefficients={"X3": 1.0}, bound=8.0, constraint_type="le"),
        ]

        result = generate_d_optimal_design(
            factors=simple_factors,
            model_type="linear",
            n_runs=10,
            constraints=constraints,
            seed=42,
        )

        for idx in range(result.n_runs):
            x1 = result.design_actual.iloc[idx]["X1"]
            x2 = result.design_actual.iloc[idx]["X2"]
            x3 = result.design_actual.iloc[idx]["X3"]

            assert x1 + x2 <= 12.0 + 1e-6
            assert x1 >= 3.0 - 1e-6
            assert x3 <= 8.0 + 1e-6

    def test_too_restrictive_constraints(self, two_factors):
        """Test that overly restrictive constraints raise error."""
        constraints = [
            LinearConstraint(coefficients={"Temp": 1.0}, bound=180.0, constraint_type="ge"),
            LinearConstraint(coefficients={"Temp": 1.0}, bound=120.0, constraint_type="le"),
        ]

        with pytest.raises(ValueError, match="feasible candidates"):
            generate_d_optimal_design(
                factors=two_factors,
                model_type="linear",
                n_runs=6,
                constraints=constraints,
                seed=42,
            )


# ============================================================
# INPUT VALIDATION
# ============================================================


class TestInputValidation:
    """Test input validation and error handling."""

    def test_insufficient_runs(self, simple_factors):
        """Test: n_runs < n_parameters."""
        with pytest.raises(ValueError, match="n_runs.*must be.*n_parameters"):
            generate_d_optimal_design(
                factors=simple_factors, model_type="linear", n_runs=3, seed=42
            )

    def test_saturated_design(self, two_factors):
        """Test: n_runs == n_parameters (saturated)."""
        result = generate_d_optimal_design(
            factors=two_factors, model_type="linear", n_runs=3, seed=42
        )

        assert result.n_runs == 3
        assert result.n_parameters == 3

    def test_non_continuous_factors_raises_error(self):
        """Test that categorical/discrete factors raise error."""
        factors = [
            Factor(
                "Material",
                FactorType.CATEGORICAL,
                ChangeabilityLevel.EASY,
                levels=["A", "B", "C"],
            ),
            Factor("Temp", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[100, 200]),
        ]

        with pytest.raises(ValueError, match="must be continuous"):
            generate_d_optimal_design(factors=factors, model_type="linear", n_runs=6, seed=42)


# ============================================================
# DESIGN QUALITY
# ============================================================


class TestDesignQuality:
    """Test that designs meet quality standards."""

    def test_determinant_positive(self, simple_factors):
        """Verify |X'X| > 0 (invertible)."""
        result = generate_d_optimal_design(
            factors=simple_factors, model_type="linear", n_runs=8, seed=42
        )

        det = np.exp(result.final_objective)
        assert det > 0

    def test_model_matrix_full_rank(self, two_factors):
        """Verify X has full column rank."""
        result = generate_d_optimal_design(
            factors=two_factors, model_type="interaction", n_runs=8, seed=42
        )

        rank = np.linalg.matrix_rank(result.model_matrix)
        assert rank == result.n_parameters

    def test_d_efficiency_in_valid_range(self, two_factors):
        """Verify D-efficiency is reasonable percentage."""
        result = generate_d_optimal_design(
            factors=two_factors, model_type="linear", n_runs=6, seed=42
        )

        assert 0 <= result.d_efficiency_vs_benchmark <= 200
        assert result.benchmark_design_name is not None

    def test_efficiency_vs_full_factorial_for_linear(self, two_factors):
        """Linear model should achieve >90% efficiency vs Full Factorial."""
        result = generate_d_optimal_design(
            factors=two_factors, model_type="linear", n_runs=4, seed=42  # Same as 2^2 factorial
        )

        assert result.d_efficiency_vs_benchmark >= 90
        assert "Full Factorial" in result.benchmark_design_name


# ============================================================
# REALISTIC SCENARIOS
# ============================================================


class TestRealisticScenarios:
    """Test realistic usage scenarios."""

    def test_typical_screening_experiment(self):
        """Test typical 5-factor screening with 12 runs."""
        factors = [
            Factor(f"X{i}", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10])
            for i in range(1, 6)
        ]

        result = generate_d_optimal_design(
            factors=factors, model_type="linear", n_runs=12, seed=42
        )

        assert result.n_runs == 12
        assert result.n_parameters == 6
        assert result.d_efficiency_vs_benchmark > 20
        assert result.condition_number < 1000

    def test_response_surface_with_constraints(self):
        """Test quadratic model with sum constraint."""
        factors = [
            Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 5]),
            Factor("X2", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 5]),
            Factor("X3", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 5]),
        ]

        constraint = LinearConstraint(
            coefficients={"X1": 1.0, "X2": 1.0, "X3": 1.0}, bound=10.0, constraint_type="le"
        )

        result = generate_d_optimal_design(
            factors=factors,
            model_type="quadratic",
            n_runs=20,
            constraints=[constraint],
            seed=42,
        )

        for idx in range(result.n_runs):
            x1 = result.design_actual.iloc[idx]["X1"]
            x2 = result.design_actual.iloc[idx]["X2"]
            x3 = result.design_actual.iloc[idx]["X3"]
            assert x1 + x2 + x3 <= 10.0 + 1e-6

        assert result.d_efficiency_vs_benchmark > 20

    def test_process_optimization_scenario(self):
        """Test realistic process optimization with multiple constraints."""
        factors = [
            Factor("Temperature", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[150, 250]),
            Factor("Pressure", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[10, 50]),
            Factor("Time", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[30, 120]),
        ]

        constraints = [
            LinearConstraint(
                coefficients={"Temperature": 1.0, "Pressure": 2.0},
                bound=350.0,
                constraint_type="le",
            ),
            LinearConstraint(
                coefficients={"Temperature": -1.0, "Time": 1.0},
                bound=-100.0,
                constraint_type="ge",
            ),
        ]

        result = generate_d_optimal_design(
            factors=factors,
            model_type="interaction",
            n_runs=15,
            constraints=constraints,
            seed=42,
        )

        assert result.n_runs == 15

        for idx in range(result.n_runs):
            temp = result.design_actual.iloc[idx]["Temperature"]
            press = result.design_actual.iloc[idx]["Pressure"]
            time = result.design_actual.iloc[idx]["Time"]

            assert temp + 2 * press <= 350.0 + 1e-6
            assert -temp + time >= -100.0 - 1e-6
