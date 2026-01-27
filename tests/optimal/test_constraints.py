"""
Tests for constraint handling and feasibility filtering.

Tests cover:
- Linear constraint evaluation
- Feasibility predicate creation
- Candidate filtering
- Rejection sampling augmentation
"""

import numpy as np
import pytest

from src.core.factors import ChangeabilityLevel, Factor, FactorType
from src.core.optimal.candidates import CandidatePoolConfig, generate_candidate_pool
from src.core.optimal.constraints import (
    LinearConstraint,
    augment_constrained_candidates,
    create_linear_constraint_predicate,
    filter_feasible_candidates,
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


# ============================================================
# LINEAR CONSTRAINT EVALUATION
# ============================================================


class TestLinearConstraint:
    """Test LinearConstraint dataclass and evaluation."""

    def test_le_constraint_satisfied(self):
        """Test ≤ constraint satisfaction."""
        constraint = LinearConstraint(
            coefficients={"X1": 1.0, "X2": 2.0}, bound=10.0, constraint_type="le"
        )

        point_satisfied = {"X1": 2.0, "X2": 3.0}
        assert constraint.evaluate(point_satisfied)  # 2 + 2*3 = 8 ≤ 10

        point_boundary = {"X1": 2.0, "X2": 4.0}
        assert constraint.evaluate(point_boundary)  # 2 + 2*4 = 10 ≤ 10

    def test_le_constraint_violated(self):
        """Test ≤ constraint violation."""
        constraint = LinearConstraint(
            coefficients={"X1": 1.0, "X2": 2.0}, bound=10.0, constraint_type="le"
        )

        point_violated = {"X1": 5.0, "X2": 4.0}
        assert not constraint.evaluate(point_violated)  # 5 + 2*4 = 13 > 10

    def test_ge_constraint(self):
        """Test ≥ constraint."""
        constraint = LinearConstraint(
            coefficients={"Temp": 1.0}, bound=150.0, constraint_type="ge"
        )

        assert constraint.evaluate({"Temp": 180.0})  # 180 ≥ 150
        assert constraint.evaluate({"Temp": 150.0})  # 150 ≥ 150
        assert not constraint.evaluate({"Temp": 120.0})  # 120 < 150

    def test_eq_constraint(self):
        """Test = constraint."""
        constraint = LinearConstraint(
            coefficients={"X1": 1.0, "X2": 1.0, "X3": 1.0},
            bound=1.0,
            constraint_type="eq",
        )

        assert constraint.evaluate({"X1": 0.3, "X2": 0.3, "X3": 0.4})  # ≈ 1.0
        assert not constraint.evaluate({"X1": 0.5, "X2": 0.3, "X3": 0.3})  # = 1.1

    def test_constraint_with_missing_coefficients(self):
        """Test constraint when factor not in coefficients."""
        constraint = LinearConstraint(
            coefficients={"X1": 1.0}, bound=5.0, constraint_type="le"
        )

        # X2 missing from coefficients -> coefficient is 0
        point = {"X1": 3.0, "X2": 100.0}
        assert constraint.evaluate(point)  # 3 ≤ 5

    def test_invalid_constraint_type_raises_error(self):
        """Test invalid constraint type."""
        constraint = LinearConstraint(
            coefficients={"X1": 1.0},
            bound=5.0,
            constraint_type="invalid",  # type: ignore
        )

        with pytest.raises(ValueError, match="Unknown constraint type"):
            constraint.evaluate({"X1": 3.0})


# ============================================================
# FEASIBILITY PREDICATE
# ============================================================


class TestFeasibilityPredicate:
    """Test feasibility predicate creation and usage."""

    def test_predicate_checks_coded_points(self, simple_factors):
        """Verify predicate decodes points before checking."""
        constraint = LinearConstraint(
            coefficients={"X1": 1.0, "X2": 1.0}, bound=12.0, constraint_type="le"
        )

        is_feasible = create_linear_constraint_predicate(simple_factors, [constraint])

        # Coded point [0, 0, 0] -> actual [5, 5, 5]
        center_coded = np.array([0.0, 0.0, 0.0])
        assert is_feasible(center_coded)  # 5 + 5 = 10 ≤ 12

        # Coded point [1, 1, 0] -> actual [10, 10, 5]
        corner_coded = np.array([1.0, 1.0, 0.0])
        assert not is_feasible(corner_coded)  # 10 + 10 = 20 > 12

    def test_predicate_multiple_constraints(self, simple_factors):
        """Test predicate with multiple constraints."""
        constraints = [
            LinearConstraint(
                coefficients={"X1": 1.0, "X2": 1.0}, bound=12.0, constraint_type="le"
            ),
            LinearConstraint(coefficients={"X3": 1.0}, bound=7.0, constraint_type="le"),
        ]

        is_feasible = create_linear_constraint_predicate(simple_factors, constraints)

        # Check various points
        assert is_feasible(np.array([0.0, 0.0, 0.0]))  # [5, 5, 5] OK
        assert not is_feasible(np.array([1.0, 1.0, 1.0]))  # [10, 10, 10] both violated


# ============================================================
# CANDIDATE FILTERING
# ============================================================


class TestCandidateFiltering:
    """Test filtering candidates by feasibility."""

    def test_filter_removes_infeasible_points(self, simple_factors):
        """Test that filtering removes infeasible candidates."""
        constraint = LinearConstraint(
            coefficients={"X1": 1.0, "X2": 1.0}, bound=12.0, constraint_type="le"
        )
        is_feasible = create_linear_constraint_predicate(simple_factors, [constraint])

        # Generate candidate pool
        config = CandidatePoolConfig(lhs_multiplier=2)
        candidates = generate_candidate_pool(simple_factors, n_runs=10, config=config, seed=42)

        feasible = filter_feasible_candidates(candidates, is_feasible)

        # Should have fewer candidates after filtering
        assert len(feasible) <= len(candidates)

        # All feasible candidates should satisfy constraint
        for point in feasible:
            assert is_feasible(point)

    def test_filter_preserves_feasible_points(self, simple_factors):
        """Test that feasible points are preserved."""
        # Very permissive constraint (almost everything feasible)
        constraint = LinearConstraint(
            coefficients={"X1": 1.0}, bound=9.9, constraint_type="le"
        )
        is_feasible = create_linear_constraint_predicate(simple_factors, [constraint])

        config = CandidatePoolConfig(lhs_multiplier=2)
        candidates = generate_candidate_pool(simple_factors, n_runs=10, config=config, seed=42)

        feasible = filter_feasible_candidates(candidates, is_feasible)

        # Most points should remain (X1 ≤ 9.9 in actual, which is almost max)
        # At least some points should survive
        assert len(feasible) > 0


# ============================================================
# REJECTION SAMPLING AUGMENTATION
# ============================================================


class TestAugmentation:
    """Test rejection sampling augmentation for constrained pools."""

    def test_augmentation_increases_pool_size(self, simple_factors):
        """Test that augmentation finds more feasible points."""
        constraint = LinearConstraint(
            coefficients={"X1": 1.0, "X2": 1.0, "X3": 1.0},
            bound=15.0,
            constraint_type="le",
        )
        is_feasible = create_linear_constraint_predicate(simple_factors, [constraint])

        # Start with small initial pool
        existing = np.array([[0.0, 0.0, 0.0], [-0.5, -0.5, -0.5]])

        augmented = augment_constrained_candidates(
            factors=simple_factors,
            existing_candidates=existing,
            is_feasible=is_feasible,
            target_size=20,
            seed=42,
            max_attempts=5000,
        )

        # Should find more feasible points
        assert len(augmented) > len(existing)

    def test_augmentation_respects_constraints(self, simple_factors):
        """Test that augmented points satisfy constraints."""
        constraint = LinearConstraint(
            coefficients={"X1": 1.0, "X2": 1.0}, bound=12.0, constraint_type="le"
        )
        is_feasible = create_linear_constraint_predicate(simple_factors, [constraint])

        existing = np.array([[0.0, 0.0, 0.0]])

        augmented = augment_constrained_candidates(
            factors=simple_factors,
            existing_candidates=existing,
            is_feasible=is_feasible,
            target_size=15,
            seed=42,
            max_attempts=3000,
        )

        # All points should be feasible
        for point in augmented:
            assert is_feasible(point)

    def test_augmentation_warns_on_low_acceptance(self, simple_factors):
        """Test warning when feasible region is very small."""
        # Very restrictive constraint
        constraint = LinearConstraint(
            coefficients={"X1": 1.0, "X2": 1.0, "X3": 1.0},
            bound=3.0,  # Only allows very small corner
            constraint_type="le",
        )
        is_feasible = create_linear_constraint_predicate(simple_factors, [constraint])

        existing = np.array([[-0.9, -0.9, -0.9]])  # One feasible point

        with pytest.warns(UserWarning, match="Very low feasible region"):
            augmented = augment_constrained_candidates(
                factors=simple_factors,
                existing_candidates=existing,
                is_feasible=is_feasible,
                target_size=20,
                seed=42,
                max_attempts=1000,  # Limited attempts to trigger warning
            )

        # Should still return existing candidates even if augmentation fails
        assert len(augmented) >= len(existing)

    def test_augmentation_returns_existing_if_target_met(self, simple_factors):
        """Test that augmentation does nothing if target already met."""
        constraint = LinearConstraint(
            coefficients={"X1": 1.0}, bound=9.0, constraint_type="le"
        )
        is_feasible = create_linear_constraint_predicate(simple_factors, [constraint])

        # Already have enough candidates
        existing = np.random.uniform(-1, 1, size=(25, 3))

        augmented = augment_constrained_candidates(
            factors=simple_factors,
            existing_candidates=existing,
            is_feasible=is_feasible,
            target_size=20,  # Already exceeded
            seed=42,
        )

        # Should return exactly the existing candidates
        assert len(augmented) == len(existing)
        np.testing.assert_array_equal(augmented, existing)
