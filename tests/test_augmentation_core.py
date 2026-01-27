"""
Tests for augmentation core functionality.

Tests candidate generation, foldover, and D-optimal augmentation.
"""

import pytest
import numpy as np
import pandas as pd

from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.candidates.generators import (
    generate_vertices,
    generate_axial_points,
    generate_candidate_pool,
    generate_augmentation_candidates,
    CandidatePoolConfig
)
from src.core.augmentation.plan import (
    AugmentationPlan,
    AugmentedDesign,
    FoldoverConfig,
    OptimalAugmentConfig,
    create_plan_id
)
from src.core.augmentation.foldover import (
    augment_full_foldover,
    augment_single_factor_foldover,
    recommend_foldover_factor
)
from src.core.augmentation.optimal import (
    augment_for_model_extension
)
from src.core.augmentation.validation import (
    validate_augmented_design
)


# ============================================================
# FIXTURES
# ============================================================


@pytest.fixture
def simple_factors():
    """Create simple 2-factor design."""
    return [
        Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
    ]


@pytest.fixture
def three_factors():
    """Create 3-factor list."""
    return [
        Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
    ]


@pytest.fixture
def fractional_design(three_factors):
    """Create 2^(3-1) fractional factorial design."""
    design = pd.DataFrame({
        'A': [-1, 1, -1, 1],
        'B': [-1, -1, 1, 1],
        'C': [1, -1, -1, 1]  # C = AB
    })
    generators = [('C', 'AB')]
    return design, generators, three_factors


# ============================================================
# TESTS: Candidate Generation
# ============================================================


class TestCandidateGeneration:
    """Test candidate pool generation."""
    
    def test_generate_vertices(self):
        """Test factorial vertex generation."""
        vertices = generate_vertices(3)
        
        assert vertices.shape == (8, 3)
        assert np.all(np.isin(vertices, [-1, 1]))
        
        # Check uniqueness
        assert len(np.unique(vertices, axis=0)) == 8
    
    def test_generate_axial_points(self):
        """Test axial point generation."""
        axial = generate_axial_points(3, alpha=1.5)
        
        assert axial.shape == (6, 3)
        
        # Each point should have one non-zero coordinate
        for point in axial:
            assert np.sum(point != 0) == 1
            assert np.max(np.abs(point)) == 1.5
    
    def test_generate_candidate_pool(self, three_factors):
        """Test complete candidate pool generation."""
        config = CandidatePoolConfig(
            include_vertices=True,
            include_axial=True,
            include_center=True,
            lhs_multiplier=5
        )
        
        candidates = generate_candidate_pool(
            factors=three_factors,
            n_runs=10,
            config=config,
            seed=42
        )
        
        # Should have vertices (8) + axial (6) + center (1) + LHS (50)
        # After deduplication, at least 50+
        assert len(candidates) >= 50
        assert candidates.shape[1] == 3
    
    def test_generate_candidate_pool_excludes_existing(self, simple_factors):
        """Test that existing design points are excluded."""
        existing = pd.DataFrame({
            'A': [-1, 1, -1, 1],
            'B': [-1, -1, 1, 1]
        })
        
        config = CandidatePoolConfig(exclude_existing_runs=True)
        
        candidates = generate_candidate_pool(
            factors=simple_factors,
            n_runs=4,
            config=config,
            existing_design=existing,
            seed=42
        )
        
        # Check that no candidate matches existing runs
        for _, row in existing.iterrows():
            point = row[['A', 'B']].values
            distances = np.linalg.norm(candidates - point, axis=1)
            assert np.all(distances > 0.001)  # All should be far
    
    def test_generate_augmentation_candidates(self, simple_factors):
        """Test augmentation-specific candidate generation."""
        existing = pd.DataFrame({
            'A': [-1, 1, -1, 1],
            'B': [-1, -1, 1, 1]
        })
        
        candidates = generate_augmentation_candidates(
            factors=simple_factors,
            original_design=existing,
            n_candidates=20,
            seed=42
        )
        
        assert len(candidates) <= 20
        assert candidates.shape[1] == 2


# ============================================================
# TESTS: Augmentation Plans
# ============================================================


class TestAugmentationPlan:
    """Test augmentation plan data structures."""
    
    def test_foldover_config(self):
        """Test foldover configuration."""
        config = FoldoverConfig(
            foldover_type='full'
        )
        
        assert config.foldover_type == 'full'
        assert config.factor_to_fold is None
    
    def test_optimal_config(self):
        """Test optimal augmentation configuration."""
        config = OptimalAugmentConfig(
            new_model_terms=['1', 'A', 'B', 'I(A**2)', 'I(B**2)'],
            n_runs_to_add=10
        )
        
        assert len(config.new_model_terms) == 5
        assert config.n_runs_to_add == 10
        assert config.criterion == 'D'
    
    def test_create_plan_id(self):
        """Test unique plan ID generation."""
        id1 = create_plan_id()
        id2 = create_plan_id()
        
        assert id1 != id2
        assert id1.startswith('plan_')
    
    def test_plan_validation_valid(self, fractional_design):
        """Test plan validation with valid plan."""
        design, generators, factors = fractional_design
        
        config = FoldoverConfig(foldover_type='full')
        
        plan = AugmentationPlan(
            plan_id='test_1',
            plan_name='Test Plan',
            strategy='foldover',
            strategy_config=config,
            original_design=design,
            factors=factors,
            n_runs_to_add=4,
            total_runs_after=8,
            expected_improvements={},
            benefits_responses=['Yield'],
            primary_beneficiary='Yield',
            experimental_cost=4.0,
            utility_score=90.0,
            rank=1,
            metadata={'generators': generators}
        )
        
        result = plan.validate()
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_plan_validation_invalid(self, fractional_design):
        """Test plan validation catches errors."""
        design, generators, factors = fractional_design
        
        # Invalid: single-factor foldover without factor specified
        config = FoldoverConfig(
            foldover_type='single_factor',
            factor_to_fold=None  # Missing!
        )
        
        plan = AugmentationPlan(
            plan_id='test_2',
            plan_name='Invalid Plan',
            strategy='foldover',
            strategy_config=config,
            original_design=design,
            factors=factors,
            n_runs_to_add=4,
            total_runs_after=8,
            expected_improvements={},
            benefits_responses=['Yield'],
            primary_beneficiary='Yield',
            experimental_cost=4.0,
            utility_score=90.0,
            rank=1
        )
        
        result = plan.validate()
        assert not result.is_valid
        assert len(result.errors) > 0


# ============================================================
# TESTS: Foldover Augmentation
# ============================================================


class TestFoldoverAugmentation:
    """Test foldover augmentation methods."""
    
    def test_full_foldover(self, fractional_design):
        """Test full foldover doubles runs and improves resolution."""
        design, generators, factors = fractional_design
        
        augmented = augment_full_foldover(
            original_design=design,
            factors=factors,
            generators=generators,
            randomize=False,
            seed=42
        )
        
        # Check structure
        assert augmented.n_runs_original == 4
        assert augmented.n_runs_added == 4
        assert augmented.n_runs_total == 8
        assert 'Phase' in augmented.combined_design.columns
        
        # Check phases
        phases = augmented.combined_design['Phase'].value_counts()
        assert phases[1] == 4  # Original
        assert phases[2] == 4  # Foldover
        
        # Check resolution improved (full factorial convention: resolution = k+1)
        # Full foldover of 2^(3-1) gives 2^3 full factorial with resolution > k
        assert augmented.resolution > 3
    
    def test_full_foldover_flips_signs(self, fractional_design):
        """Test that foldover correctly flips all factor signs."""
        design, generators, factors = fractional_design
        
        augmented = augment_full_foldover(
            original_design=design,
            factors=factors,
            generators=generators,
            randomize=False
        )
        
        # Extract original and foldover runs
        original_runs = augmented.combined_design[augmented.combined_design['Phase'] == 1]
        foldover_runs = augmented.combined_design[augmented.combined_design['Phase'] == 2]
        
        factor_names = [f.name for f in factors]
        
        # Check that foldover = -original
        original_values = original_runs[factor_names].values
        foldover_values = foldover_runs[factor_names].values
        
        assert np.allclose(foldover_values, -original_values)
    
    def test_single_factor_foldover(self, fractional_design):
        """Test single-factor foldover."""
        design, generators, factors = fractional_design
        
        augmented = augment_single_factor_foldover(
            original_design=design,
            factors=factors,
            generators=generators,
            factor_to_fold='A',
            randomize=False
        )
        
        # Check structure
        assert augmented.n_runs_total == 8
        assert 'Phase' in augmented.combined_design.columns
        
        # Check that only factor A was flipped
        original_runs = augmented.combined_design[augmented.combined_design['Phase'] == 1]
        foldover_runs = augmented.combined_design[augmented.combined_design['Phase'] == 2]
        
        # A should be flipped
        assert np.allclose(
            foldover_runs['A'].values,
            -original_runs['A'].values
        )
        
        # B and C should be same
        assert np.allclose(
            foldover_runs['B'].values,
            original_runs['B'].values
        )
        assert np.allclose(
            foldover_runs['C'].values,
            original_runs['C'].values
        )
    
    def test_recommend_foldover_factor(self, fractional_design):
        """Test foldover factor recommendation."""
        design, generators, factors = fractional_design
        
        # Simulate A being significant
        significant_effects = ['A', 'B']
        
        recommended = recommend_foldover_factor(
            original_design=design,
            factors=factors,
            generators=generators,
            significant_effects=significant_effects
        )
        
        # Should recommend a factor that's aliased
        assert recommended in ['A', 'B', 'C']


# ============================================================
# TESTS: D-Optimal Augmentation
# ============================================================


class TestOptimalAugmentation:
    """Test D-optimal augmentation."""
    
    def test_model_extension_basic(self, simple_factors):
        """Test basic model extension augmentation."""
        # Original design: 2^2 factorial
        original = pd.DataFrame({
            'A': [-1, 1, -1, 1],
            'B': [-1, -1, 1, 1]
        })
        
        current_terms = ['1', 'A', 'B', 'A*B']
        new_terms = ['1', 'A', 'B', 'A*B', 'I(A**2)', 'I(B**2)']
        
        augmented = augment_for_model_extension(
            original_design=original,
            factors=simple_factors,
            current_model_terms=current_terms,
            new_model_terms=new_terms,
            n_runs_to_add=6,
            seed=42
        )
        
        # Check structure
        assert augmented.n_runs_original == 4
        assert augmented.n_runs_added == 6
        assert augmented.n_runs_total == 10
        assert 'Phase' in augmented.combined_design.columns
    
    def test_model_extension_supersaturated_error(self, simple_factors):
        """Test error when augmentation would be supersaturated."""
        original = pd.DataFrame({
            'A': [-1, 1],
            'B': [-1, -1]
        })
        
        current_terms = ['1', 'A', 'B']
        new_terms = ['1', 'A', 'B', 'I(A**2)', 'I(B**2)', 'A*B']  # 6 terms
        
        with pytest.raises(ValueError, match="supersaturated"):
            augment_for_model_extension(
                original_design=original,
                factors=simple_factors,
                current_model_terms=current_terms,
                new_model_terms=new_terms,
                n_runs_to_add=2,  # 2+2=4 runs, 6 terms → supersaturated
                seed=42
            )


# ============================================================
# TESTS: Validation
# ============================================================


class TestValidation:
    """Test augmented design validation."""
    
    def test_validate_augmented_design(self, fractional_design):
        """Test comprehensive augmented design validation."""
        design, generators, factors = fractional_design
        
        augmented = augment_full_foldover(
            original_design=design,
            factors=factors,
            generators=generators
        )
        
        result = validate_augmented_design(
            augmented=augmented,
            factors=factors,
            model_terms=['1', 'A', 'B', 'C']
        )
        
        # Should be valid
        assert result.is_valid
        assert len(result.errors) == 0
        
        # Should have metrics
        assert 'n_runs_total' in result.metrics
        assert 'condition_number' in result.metrics


# ============================================================
# INTEGRATION TESTS
# ============================================================


class TestAugmentationIntegration:
    """Integration tests for complete augmentation workflow."""
    
    def test_full_foldover_workflow(self, fractional_design):
        """Test complete foldover workflow via plan."""
        design, generators, factors = fractional_design
        
        # Create plan
        config = FoldoverConfig(foldover_type='full')
        
        plan = AugmentationPlan(
            plan_id=create_plan_id(),
            plan_name='Full Foldover',
            strategy='foldover',
            strategy_config=config,
            original_design=design,
            factors=factors,
            n_runs_to_add=4,
            total_runs_after=8,
            expected_improvements={'resolution': '3 → 4'},
            benefits_responses=['Yield'],
            primary_beneficiary='Yield',
            experimental_cost=4.0,
            utility_score=95.0,
            rank=1,
            metadata={'generators': generators}
        )
        
        # Execute plan
        augmented = plan.execute()
        
        # Verify result
        assert augmented.n_runs_total == 8
        assert augmented.resolution >= 4
        assert augmented.plan_executed is plan
    
    def test_d_optimal_workflow(self, simple_factors):
        """Test D-optimal augmentation workflow."""
        original = pd.DataFrame({
            'A': [-1, 1, -1, 1],
            'B': [-1, -1, 1, 1]
        })
        
        config = OptimalAugmentConfig(
            new_model_terms=['1', 'A', 'B', 'I(A**2)', 'I(B**2)'],
            n_runs_to_add=4
        )
        
        plan = AugmentationPlan(
            plan_id=create_plan_id(),
            plan_name='Add Quadratic Terms',
            strategy='d_optimal',
            strategy_config=config,
            original_design=original,
            factors=simple_factors,
            n_runs_to_add=4,
            total_runs_after=8,
            expected_improvements={'model_terms': '4 → 5'},
            benefits_responses=['Yield'],
            primary_beneficiary='Yield',
            experimental_cost=4.0,
            utility_score=85.0,
            rank=1,
            metadata={'current_model_terms': ['1', 'A', 'B', 'A*B']}
        )
        
        # Execute
        augmented = plan.execute()
        
        # Verify
        assert augmented.n_runs_total == 8
        assert augmented.d_efficiency is not None