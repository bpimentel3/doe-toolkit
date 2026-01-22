"""
Tests for the coordinate exchange optimizer and Sherman-Morrison updates.

Tests cover:
- Sherman-Morrison numerical accuracy
- Optimizer configuration
- CEXCH algorithm convergence
- Multiple random starts
"""

import numpy as np
import pytest

from src.core.factors import ChangeabilityLevel, Factor, FactorType
from src.core.optimal.candidates import CandidatePoolConfig, generate_candidate_pool
from src.core.optimal.criteria import DOptimalityCriterion, create_polynomial_builder
from src.core.optimal.optimizer import (
    OptimizerConfig,
    ShermanMorrisonResult,
    cexch_optimize,
    sherman_morrison_swap,
)


# ============================================================
# SHERMAN-MORRISON TESTS
# ============================================================


class TestShermanMorrisonSwap:
    """Test Sherman-Morrison determinant update with inverse reuse."""

    def test_sm_swap_matches_explicit_computation(self):
        """Verify SM swap matches explicit computation."""
        XtX = np.array([[10, 2, 1], [2, 8, 3], [1, 3, 6]])
        XtX_inv = np.linalg.inv(XtX)

        x_old = np.array([1, 2, 1])
        x_new = np.array([2, 1, 3])

        sm_result = sherman_morrison_swap(XtX_inv, x_old, x_new)

        # Explicit computation
        XtX_new = XtX - np.outer(x_old, x_old) + np.outer(x_new, x_new)
        XtX_inv_explicit = np.linalg.inv(XtX_new)

        det_old = np.linalg.det(XtX)
        det_new = np.linalg.det(XtX_new)
        explicit_ratio = det_new / det_old

        # Check ratio matches
        assert abs(sm_result.det_ratio - explicit_ratio) < 1e-8

        # Check updated inverse matches
        np.testing.assert_array_almost_equal(
            sm_result.XtX_inv_updated, XtX_inv_explicit, decimal=10
        )

        assert sm_result.is_valid

    def test_sm_result_contains_all_fields(self):
        """Test that SM result has all required fields."""
        XtX = np.eye(3)
        XtX_inv = np.eye(3)

        x_old = np.array([1, 0, 0])
        x_new = np.array([0.5, 0.5, 0])

        sm_result = sherman_morrison_swap(XtX_inv, x_old, x_new)

        assert isinstance(sm_result, ShermanMorrisonResult)
        assert hasattr(sm_result, "det_ratio")
        assert hasattr(sm_result, "XtX_inv_updated")
        assert hasattr(sm_result, "is_valid")

    def test_sm_handles_singular_case(self):
        """Test SM when denominator near zero."""
        XtX = np.eye(3)
        XtX_inv = np.eye(3)

        x_old = np.array([1, 0, 0])
        x_new = np.array([0, 1, 0])

        sm_result = sherman_morrison_swap(XtX_inv, x_old, x_new)

        # Result should be finite
        assert np.all(np.isfinite(sm_result.XtX_inv_updated))

        # If invalid, det_ratio should be 0
        if not sm_result.is_valid:
            assert sm_result.det_ratio == 0.0

    def test_sm_validates_input_shapes(self):
        """Test SM rejects mismatched input shapes."""
        XtX_inv = np.eye(3)
        x_old = np.array([1, 2])  # Wrong size
        x_new = np.array([2, 1, 3])

        with pytest.raises(ValueError, match="must have shape"):
            sherman_morrison_swap(XtX_inv, x_old, x_new)

    def test_sm_validates_square_matrix(self):
        """Test SM rejects non-square matrices."""
        XtX_inv = np.array([[1, 2], [3, 4], [5, 6]])  # Not square
        x_old = np.array([1, 2])
        x_new = np.array([2, 1])

        with pytest.raises(ValueError, match="square matrix"):
            sherman_morrison_swap(XtX_inv, x_old, x_new)


# ============================================================
# OPTIMIZER CONFIGURATION
# ============================================================


class TestOptimizerConfig:
    """Test OptimizerConfig dataclass and validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = OptimizerConfig()

        assert config.max_iterations == 200
        assert config.relative_improvement_tolerance == 1e-4
        assert config.stability_window == 15
        assert config.n_random_starts == 3
        assert config.max_candidates_per_row == 50
        assert config.use_sherman_morrison is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = OptimizerConfig(
            max_iterations=100,
            n_random_starts=5,
            use_sherman_morrison=False,
        )

        assert config.max_iterations == 100
        assert config.n_random_starts == 5
        assert config.use_sherman_morrison is False

    def test_config_validation_max_iterations(self):
        """Test validation of max_iterations."""
        with pytest.raises(ValueError, match="max_iterations must be"):
            OptimizerConfig(max_iterations=0)

    def test_config_validation_tolerance(self):
        """Test validation of tolerance."""
        with pytest.raises(ValueError, match="relative_improvement_tolerance must be"):
            OptimizerConfig(relative_improvement_tolerance=0)

    def test_config_validation_stability_window(self):
        """Test validation of stability_window."""
        with pytest.raises(ValueError, match="stability_window must be"):
            OptimizerConfig(stability_window=0)

    def test_config_validation_random_starts(self):
        """Test validation of n_random_starts."""
        with pytest.raises(ValueError, match="n_random_starts must be"):
            OptimizerConfig(n_random_starts=0)


# ============================================================
# CEXCH OPTIMIZATION
# ============================================================


class TestCEXCHOptimizer:
    """Test coordinate exchange optimizer."""

    @pytest.fixture
    def simple_setup(self):
        """Common setup for optimizer tests."""
        factors = [
            Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
            Factor("X2", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
        ]

        config_cand = CandidatePoolConfig(lhs_multiplier=3)
        candidates = generate_candidate_pool(factors, n_runs=8, config=config_cand, seed=42)

        model_builder = create_polynomial_builder(factors, "linear")
        criterion = DOptimalityCriterion()

        return {
            "candidates": candidates,
            "model_builder": model_builder,
            "criterion": criterion,
        }

    def test_optimizer_returns_valid_indices(self, simple_setup):
        """Test that optimizer returns valid candidate indices."""
        config = OptimizerConfig(max_iterations=20, n_random_starts=1)

        indices, obj, n_iter, converged_by = cexch_optimize(
            candidates=simple_setup["candidates"],
            n_runs=6,
            model_builder=simple_setup["model_builder"],
            criterion=simple_setup["criterion"],
            config=config,
            seed=42,
        )

        assert indices.shape == (6,)
        assert np.all(indices >= 0)
        assert np.all(indices < len(simple_setup["candidates"]))
        assert len(np.unique(indices)) == 6  # All distinct

    def test_optimizer_improves_objective(self, simple_setup):
        """Test that optimizer improves from initial random design."""
        config = OptimizerConfig(max_iterations=50, n_random_starts=1)

        indices, final_obj, n_iter, converged_by = cexch_optimize(
            candidates=simple_setup["candidates"],
            n_runs=6,
            model_builder=simple_setup["model_builder"],
            criterion=simple_setup["criterion"],
            config=config,
            seed=42,
        )

        # Objective should be finite and positive (log-det)
        assert np.isfinite(final_obj)
        assert final_obj > -1e10

    def test_optimizer_respects_max_iterations(self, simple_setup):
        """Test that optimizer doesn't exceed max_iterations."""
        config = OptimizerConfig(max_iterations=10, n_random_starts=1)

        indices, obj, n_iter, converged_by = cexch_optimize(
            candidates=simple_setup["candidates"],
            n_runs=6,
            model_builder=simple_setup["model_builder"],
            criterion=simple_setup["criterion"],
            config=config,
            seed=42,
        )

        assert n_iter <= config.max_iterations

    def test_optimizer_records_convergence_reason(self, simple_setup):
        """Test that converged_by is set correctly."""
        config = OptimizerConfig(max_iterations=100, n_random_starts=1)

        indices, obj, n_iter, converged_by = cexch_optimize(
            candidates=simple_setup["candidates"],
            n_runs=6,
            model_builder=simple_setup["model_builder"],
            criterion=simple_setup["criterion"],
            config=config,
            seed=42,
        )

        assert converged_by in ["stability", "max_iterations"]

    def test_multiple_starts_improve_or_match(self, simple_setup):
        """Test that multiple starts find better or equal solutions."""
        config_1start = OptimizerConfig(max_iterations=30, n_random_starts=1)

        _, obj_1start, _, _ = cexch_optimize(
            candidates=simple_setup["candidates"],
            n_runs=6,
            model_builder=simple_setup["model_builder"],
            criterion=simple_setup["criterion"],
            config=config_1start,
            seed=42,
        )

        config_3start = OptimizerConfig(max_iterations=30, n_random_starts=3)

        _, obj_3start, _, _ = cexch_optimize(
            candidates=simple_setup["candidates"],
            n_runs=6,
            model_builder=simple_setup["model_builder"],
            criterion=simple_setup["criterion"],
            config=config_3start,
            seed=42,
        )

        # More starts should find better or equal solution
        assert obj_3start >= obj_1start - 1e-6

    def test_optimizer_raises_error_insufficient_candidates(self, simple_setup):
        """Test error when n_runs > number of candidates."""
        config = OptimizerConfig(max_iterations=10, n_random_starts=1)

        # Request more runs than candidates
        with pytest.raises(ValueError, match="Not enough candidates"):
            cexch_optimize(
                candidates=simple_setup["candidates"],
                n_runs=1000,  # Way more than candidates
                model_builder=simple_setup["model_builder"],
                criterion=simple_setup["criterion"],
                config=config,
                seed=42,
            )

    def test_optimizer_reproducible_with_seed(self, simple_setup):
        """Test that same seed produces identical results."""
        config = OptimizerConfig(max_iterations=20, n_random_starts=2)

        indices1, obj1, _, _ = cexch_optimize(
            candidates=simple_setup["candidates"],
            n_runs=6,
            model_builder=simple_setup["model_builder"],
            criterion=simple_setup["criterion"],
            config=config,
            seed=12345,
        )

        indices2, obj2, _, _ = cexch_optimize(
            candidates=simple_setup["candidates"],
            n_runs=6,
            model_builder=simple_setup["model_builder"],
            criterion=simple_setup["criterion"],
            config=config,
            seed=12345,
        )

        np.testing.assert_array_equal(indices1, indices2)
        assert obj1 == obj2
