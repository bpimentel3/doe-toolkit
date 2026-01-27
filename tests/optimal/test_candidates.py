"""
Tests for candidate pool generation.

Tests cover:
- Vertex generation
- Axial point generation
- Candidate pool assembly
- Configuration handling
"""

import numpy as np
import pytest

from src.core.factors import ChangeabilityLevel, Factor, FactorType
from src.core.optimal.candidates import (
    CandidatePoolConfig,
    generate_axial_points,
    generate_candidate_pool,
    generate_vertices,
)


# ============================================================
# VERTEX GENERATION
# ============================================================


class TestVertexGeneration:
    """Test factorial vertex generation."""

    def test_vertices_2d(self):
        """Test 2^2 vertices."""
        vertices = generate_vertices(2)

        assert vertices.shape == (4, 2)
        expected = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
        np.testing.assert_array_equal(vertices, expected)

    def test_vertices_3d(self):
        """Test 2^3 vertices."""
        vertices = generate_vertices(3)

        assert vertices.shape == (8, 3)
        # Check all corners present
        assert np.all(np.abs(vertices) == 1)

    def test_vertices_all_combinations(self):
        """Verify all -1, +1 combinations present."""
        k = 4
        vertices = generate_vertices(k)

        assert vertices.shape == (2**k, k)

        # Each vertex should be unique
        unique_vertices = np.unique(vertices, axis=0)
        assert len(unique_vertices) == 2**k


# ============================================================
# AXIAL POINT GENERATION
# ============================================================


class TestAxialGeneration:
    """Test axial (star) point generation."""

    def test_axial_2d(self):
        """Test 2k axial points for k=2."""
        axial = generate_axial_points(2, alpha=1.0)

        assert axial.shape == (4, 2)
        expected = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        np.testing.assert_array_almost_equal(axial, expected)

    def test_axial_3d(self):
        """Test 2k axial points for k=3."""
        axial = generate_axial_points(3, alpha=1.0)

        assert axial.shape == (6, 3)

        # Check each point has exactly one non-zero coordinate
        nonzero_counts = np.count_nonzero(axial, axis=1)
        assert np.all(nonzero_counts == 1)

    def test_axial_custom_alpha(self):
        """Test axial points with custom alpha."""
        alpha = 1.414
        axial = generate_axial_points(2, alpha=alpha)

        # First two points should be [±alpha, 0]
        assert abs(axial[0, 0] - alpha) < 1e-10
        assert abs(axial[1, 0] + alpha) < 1e-10


# ============================================================
# CANDIDATE POOL GENERATION
# ============================================================


class TestCandidatePoolGeneration:
    """Test complete candidate pool generation."""

    def test_basic_pool_includes_structured_points(self):
        """Verify structured points are included."""
        factors = [
            Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
            Factor("X2", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
        ]

        config = CandidatePoolConfig(lhs_multiplier=2)
        candidates = generate_candidate_pool(factors, n_runs=10, config=config, seed=42)

        k = 2
        # Should have at least vertices (2^k) + axial (2k) + center (1) + LHS (n*mult)
        min_expected = 2**k + 2 * k + 1
        assert len(candidates) >= min_expected

    def test_pool_contains_vertices(self):
        """Verify all vertices are in pool."""
        factors = [
            Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
            Factor("X2", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
        ]

        config = CandidatePoolConfig()
        candidates = generate_candidate_pool(factors, n_runs=10, config=config, seed=42)

        # Check corners present
        vertices = generate_vertices(2)
        for vertex in vertices:
            # Find if this vertex is in candidates
            distances = np.linalg.norm(candidates - vertex, axis=1)
            assert np.any(distances < 1e-6), f"Vertex {vertex} not in pool"

    def test_pool_contains_center(self):
        """Verify center point is included."""
        factors = [
            Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10])
        ]

        config = CandidatePoolConfig(include_center=True)
        candidates = generate_candidate_pool(factors, n_runs=5, config=config, seed=42)

        # Check center [0] present
        distances = np.abs(candidates[:, 0])
        assert np.any(distances < 1e-6)

    def test_pool_excludes_center_when_requested(self):
        """Verify center can be excluded."""
        factors = [
            Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10])
        ]

        config = CandidatePoolConfig(include_center=False, include_vertices=False, include_axial=False)
        candidates = generate_candidate_pool(factors, n_runs=5, config=config, seed=42)

        # Center should not be present (with high confidence for LHS)
        if len(candidates) > 0:
            distances = np.abs(candidates[:, 0])
            assert not np.any(distances < 1e-6)

    def test_lhs_multiplier_affects_pool_size(self):
        """Verify LHS multiplier increases pool size."""
        factors = [
            Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
            Factor("X2", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
        ]

        config_small = CandidatePoolConfig(lhs_multiplier=2)
        pool_small = generate_candidate_pool(factors, n_runs=10, config=config_small, seed=42)

        config_large = CandidatePoolConfig(lhs_multiplier=10)
        pool_large = generate_candidate_pool(factors, n_runs=10, config=config_large, seed=42)

        assert len(pool_large) > len(pool_small)

    def test_all_candidates_in_bounds(self):
        """Verify all candidates in [-1, 1]^k."""
        factors = [
            Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
            Factor("X2", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
            Factor("X3", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
        ]

        config = CandidatePoolConfig(lhs_multiplier=5)
        candidates = generate_candidate_pool(factors, n_runs=15, config=config, seed=42)

        assert np.all(candidates >= -1.001)
        assert np.all(candidates <= 1.001)

    def test_reproducibility_with_seed(self):
        """Verify same seed produces same pool."""
        factors = [
            Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10])
        ]

        config = CandidatePoolConfig()

        pool1 = generate_candidate_pool(factors, n_runs=10, config=config, seed=123)
        pool2 = generate_candidate_pool(factors, n_runs=10, config=config, seed=123)

        np.testing.assert_array_equal(pool1, pool2)


# ============================================================
# CONFIGURATION
# ============================================================


class TestCandidatePoolConfig:
    """Test configuration dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CandidatePoolConfig()

        assert config.lhs_multiplier == 5
        assert config.include_vertices is True
        assert config.include_axial is True
        assert config.include_center is True
        assert config.alpha_axial == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = CandidatePoolConfig(
            lhs_multiplier=10, include_vertices=False, alpha_axial=1.414
        )

        assert config.lhs_multiplier == 10
        assert config.include_vertices is False
        assert config.alpha_axial == 1.414
