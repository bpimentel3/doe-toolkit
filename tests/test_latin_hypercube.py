"""
Tests for Latin Hypercube Sampling module.
"""

import pytest
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import sys
sys.path.insert(0, 'src/core')

from latin_hypercube import (
    generate_latin_hypercube,
    augment_latin_hypercube,
    LHSDesign
)
from factors import Factor, FactorType


class TestGenerateLatinHypercube:
    """Test Latin Hypercube generation with various factor types."""
    
    def test_basic_continuous_factors(self):
        """Test LHS generation with continuous factors."""
        factors = [
            Factor(name='Temperature', type=FactorType.CONTINUOUS, min=100, max=200),
            Factor(name='Pressure', type=FactorType.CONTINUOUS, min=1, max=5)
        ]
        
        design = generate_latin_hypercube(factors, n_runs=20, seed=42)
        
        assert isinstance(design, LHSDesign)
        assert len(design.design) == 20
        assert design.n_runs == 20
        assert list(design.design.columns) == ['StdOrder', 'RunOrder', 'Temperature', 'Pressure']
        
        # Check bounds
        assert design.design['Temperature'].min() >= 100
        assert design.design['Temperature'].max() <= 200
        assert design.design['Pressure'].min() >= 1
        assert design.design['Pressure'].max() <= 5
    
    def test_discrete_numeric_factors(self):
        """Test LHS with discrete numeric factors (should round to allowed levels)."""
        factors = [
            Factor(name='RPM', type=FactorType.DISCRETE, 
                   levels=[100, 150, 200, 250, 300])
        ]
        
        design = generate_latin_hypercube(factors, n_runs=15, seed=42)
        
        # All values should be from allowed levels
        allowed_levels = {100, 150, 200, 250, 300}
        assert set(design.design['RPM'].unique()).issubset(allowed_levels)
    
    def test_categorical_factors(self):
        """Test LHS with categorical factors (stratified sampling)."""
        factors = [
            Factor(name='Material', type=FactorType.CATEGORICAL,
                   levels=['A', 'B', 'C'])
        ]
        
        design = generate_latin_hypercube(factors, n_runs=30, seed=42)
        
        # Should have balanced representation (10 of each for 30 runs / 3 levels)
        counts = design.design['Material'].value_counts()
        assert all(counts == 10)
    
    def test_categorical_uneven_distribution(self):
        """Test categorical with runs not evenly divisible by levels."""
        factors = [
            Factor(name='Supplier', type=FactorType.CATEGORICAL,
                   levels=['X', 'Y', 'Z'])
        ]
        
        design = generate_latin_hypercube(factors, n_runs=20, seed=42)
        
        # 20 runs / 3 levels = 6 remainder 2
        # Should get: two levels with 7, one with 6
        counts = design.design['Supplier'].value_counts().sort_values(ascending=False)
        assert counts.iloc[0] == 7
        assert counts.iloc[1] == 7
        assert counts.iloc[2] == 6
    
    def test_mixed_factor_types(self):
        """Test LHS with continuous, discrete, and categorical factors."""
        factors = [
            Factor(name='Temp', type=FactorType.CONTINUOUS, min=100, max=200),
            Factor(name='Speed', type=FactorType.DISCRETE, levels=[10, 20, 30]),
            Factor(name='Tool', type=FactorType.CATEGORICAL, levels=['A', 'B'])
        ]
        
        design = generate_latin_hypercube(factors, n_runs=20, seed=42)
        
        assert len(design.design) == 20
        assert 'Temp' in design.design.columns
        assert 'Speed' in design.design.columns
        assert 'Tool' in design.design.columns
        
        # Check continuous bounds
        assert design.design['Temp'].min() >= 100
        assert design.design['Temp'].max() <= 200
        
        # Check discrete levels
        assert set(design.design['Speed'].unique()).issubset({10, 20, 30})
        
        # Check categorical balance (20 runs / 2 levels = 10 each)
        tool_counts = design.design['Tool'].value_counts()
        assert all(tool_counts == 10)
    
    def test_coded_levels(self):
        """Test that coded levels are in [-1, 1] range."""
        factors = [
            Factor(name='X1', type=FactorType.CONTINUOUS, min=0, max=100),
            Factor(name='X2', type=FactorType.CONTINUOUS, min=50, max=150)
        ]
        
        design = generate_latin_hypercube(factors, n_runs=15, seed=42)
        
        # Coded levels should be in [-1, 1]
        assert design.design_coded['X1'].min() >= -1.0
        assert design.design_coded['X1'].max() <= 1.0
        assert design.design_coded['X2'].min() >= -1.0
        assert design.design_coded['X2'].max() <= 1.0
        
        # Check center point coding (actual=50 should code to 0 for X1)
        # X1: center = 50, half_range = 50, so 50 codes to 0
        # Find point closest to actual value of 50
        closest_idx = (design.design['X1'] - 50).abs().idxmin()
        coded_value = design.design_coded.loc[closest_idx, 'X1']
        assert abs(coded_value) < 0.5  # Should be close to 0
    
    def test_maximin_criterion(self):
        """Test that maximin criterion selects design with good space-filling."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1),
            Factor(name='Y', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        design = generate_latin_hypercube(
            factors, n_runs=10, criterion='maximin', n_candidates=5, seed=42
        )
        
        assert design.criterion == 'maximin'
        assert design.criterion_value > 0  # Should have positive minimum distance
    
    def test_correlation_criterion(self):
        """Test correlation criterion for factor independence."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1),
            Factor(name='Y', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        design = generate_latin_hypercube(
            factors, n_runs=20, criterion='correlation', n_candidates=5, seed=42
        )
        
        assert design.criterion == 'correlation'
        # Criterion value is negative max correlation, so should be negative
        assert design.criterion_value <= 0
        
        # Check actual correlation is low
        corr = pearsonr(design.design_coded['X'], design.design_coded['Y'])[0]
        assert abs(corr) < 0.3  # Should have low correlation
    
    def test_single_continuous_factor(self):
        """Test with single continuous factor (edge case for correlation)."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        # Should work for both criteria
        design_maximin = generate_latin_hypercube(
            factors, n_runs=10, criterion='maximin', seed=42
        )
        design_corr = generate_latin_hypercube(
            factors, n_runs=10, criterion='correlation', seed=42
        )
        
        assert len(design_maximin.design) == 10
        assert len(design_corr.design) == 10
    
    def test_only_categorical_factors(self):
        """Test with only categorical factors (no numeric scoring)."""
        factors = [
            Factor(name='Material', type=FactorType.CATEGORICAL, levels=['A', 'B', 'C']),
            Factor(name='Process', type=FactorType.CATEGORICAL, levels=['P1', 'P2'])
        ]
        
        design = generate_latin_hypercube(factors, n_runs=18, seed=42)
        
        assert len(design.design) == 18
        # Should have balanced representation
        assert all(design.design['Material'].value_counts() == 6)
        assert all(design.design['Process'].value_counts() == 9)
    
    def test_latin_property(self):
        """Test that design has Latin property (one sample per interval)."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        n_runs = 10
        design = generate_latin_hypercube(factors, n_runs=n_runs, seed=42)
        
        # Divide [0, 1] into n_runs intervals
        # Each interval should contain exactly one point
        intervals = np.linspace(0, 1, n_runs + 1)
        counts = np.histogram(design.design['X'], bins=intervals)[0]
        
        # Each interval should have exactly one point
        assert all(counts == 1)
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same design."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1),
            Factor(name='Y', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        design1 = generate_latin_hypercube(factors, n_runs=10, seed=42)
        design2 = generate_latin_hypercube(factors, n_runs=10, seed=42)
        
        pd.testing.assert_frame_equal(design1.design, design2.design)
    
    def test_different_seeds_produce_different_designs(self):
        """Test that different seeds produce different designs."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1),
            Factor(name='Y', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        design1 = generate_latin_hypercube(factors, n_runs=10, seed=42)
        design2 = generate_latin_hypercube(factors, n_runs=10, seed=123)
        
        # Designs should be different
        assert not design1.design.equals(design2.design)


class TestLHSValidation:
    """Test input validation and error handling."""
    
    def test_n_runs_too_small(self):
        """Test that n_runs < 2 raises error."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        with pytest.raises(ValueError, match="n_runs must be at least 2"):
            generate_latin_hypercube(factors, n_runs=1)
    
    def test_n_candidates_too_small(self):
        """Test that n_candidates < 1 raises error."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        with pytest.raises(ValueError, match="n_candidates must be at least 1"):
            generate_latin_hypercube(factors, n_runs=10, n_candidates=0)
    
    def test_no_factors(self):
        """Test that empty factor list raises error."""
        with pytest.raises(ValueError, match="At least one factor must be provided"):
            generate_latin_hypercube([], n_runs=10)
    
    def test_invalid_criterion(self):
        """Test that invalid criterion raises error."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        with pytest.raises(ValueError, match="Unknown criterion"):
            generate_latin_hypercube(factors, n_runs=10, criterion='invalid')


class TestAugmentLatinHypercube:
    """Test design augmentation functionality."""
    
    def test_basic_augmentation(self):
        """Test adding runs to existing design."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1),
            Factor(name='Y', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        # Generate initial design
        initial = generate_latin_hypercube(factors, n_runs=10, seed=42)
        initial_clean = initial.design.drop(columns=['StdOrder', 'RunOrder'])
        
        # Augment
        augmented = augment_latin_hypercube(
            existing_design=initial_clean,
            factors=factors,
            n_additional_runs=5,
            seed=42
        )
        
        assert len(augmented.design) == 15  # 10 + 5
        assert augmented.n_runs == 15
        
        # First 10 rows should match initial design
        initial_values = initial_clean.values
        augmented_first_10 = augmented.design.drop(columns=['StdOrder', 'RunOrder']).iloc[:10].values
        np.testing.assert_array_almost_equal(initial_values, augmented_first_10)
    
    def test_augmentation_maintains_bounds(self):
        """Test that augmented runs respect factor bounds."""
        factors = [
            Factor(name='Temp', type=FactorType.CONTINUOUS, min=100, max=200)
        ]
        
        initial = generate_latin_hypercube(factors, n_runs=10, seed=42)
        initial_clean = initial.design.drop(columns=['StdOrder', 'RunOrder'])
        
        augmented = augment_latin_hypercube(
            existing_design=initial_clean,
            factors=factors,
            n_additional_runs=5,
            seed=42
        )
        
        assert augmented.design['Temp'].min() >= 100
        assert augmented.design['Temp'].max() <= 200
    
    def test_augmentation_with_categorical(self):
        """Test augmentation with mixed factor types."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1),
            Factor(name='Cat', type=FactorType.CATEGORICAL, levels=['A', 'B', 'C'])
        ]
        
        initial = generate_latin_hypercube(factors, n_runs=9, seed=42)
        initial_clean = initial.design.drop(columns=['StdOrder', 'RunOrder'])
        
        augmented = augment_latin_hypercube(
            existing_design=initial_clean,
            factors=factors,
            n_additional_runs=6,
            seed=42
        )
        
        assert len(augmented.design) == 15
        # Categorical should still be balanced
        # 15 total / 3 levels = 5 each
        assert all(augmented.design['Cat'].value_counts() == 5)
    
    def test_augmentation_invalid_runs(self):
        """Test that n_additional_runs < 1 raises error."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        initial = generate_latin_hypercube(factors, n_runs=10, seed=42)
        initial_clean = initial.design.drop(columns=['StdOrder', 'RunOrder'])
        
        with pytest.raises(ValueError, match="n_additional_runs must be at least 1"):
            augment_latin_hypercube(
                existing_design=initial_clean,
                factors=factors,
                n_additional_runs=0
            )
    
    def test_augmentation_mismatched_factors(self):
        """Test that mismatched factors raise error."""
        factors1 = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        factors2 = [
            Factor(name='Y', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        initial = generate_latin_hypercube(factors1, n_runs=10, seed=42)
        initial_clean = initial.design.drop(columns=['StdOrder', 'RunOrder'])
        
        with pytest.raises(ValueError, match="Factor names must match"):
            augment_latin_hypercube(
                existing_design=initial_clean,
                factors=factors2,
                n_additional_runs=5
            )
    
    def test_augmentation_improves_space_filling(self):
        """Test that augmentation maintains good space-filling properties."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1),
            Factor(name='Y', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        # Start with small design
        initial = generate_latin_hypercube(factors, n_runs=5, seed=42)
        initial_clean = initial.design.drop(columns=['StdOrder', 'RunOrder'])
        
        # Augment with maximin criterion
        augmented = augment_latin_hypercube(
            existing_design=initial_clean,
            factors=factors,
            n_additional_runs=5,
            criterion='maximin',
            n_candidates=5,
            seed=42
        )
        
        # Augmented design should have reasonable space-filling
        assert augmented.criterion_value > 0


class TestLHSDesignObject:
    """Test LHSDesign dataclass properties."""
    
    def test_design_object_attributes(self):
        """Test that LHSDesign has all required attributes."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        design = generate_latin_hypercube(factors, n_runs=10, seed=42)
        
        assert hasattr(design, 'design')
        assert hasattr(design, 'design_coded')
        assert hasattr(design, 'n_runs')
        assert hasattr(design, 'criterion')
        assert hasattr(design, 'criterion_value')
        
        assert isinstance(design.design, pd.DataFrame)
        assert isinstance(design.design_coded, pd.DataFrame)
        assert isinstance(design.n_runs, int)
        assert isinstance(design.criterion, str)
        assert isinstance(design.criterion_value, (int, float))


class TestSpaceFillingQuality:
    """Test space-filling quality metrics."""
    
    def test_better_than_random(self):
        """Test that LHS has better space coverage than random sampling."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1),
            Factor(name='Y', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        # Generate LHS
        lhs = generate_latin_hypercube(factors, n_runs=20, criterion='maximin', seed=42)
        
        # Generate random design
        rng = np.random.default_rng(42)
        random_design = pd.DataFrame({
            'X': rng.uniform(0, 1, 20),
            'Y': rng.uniform(0, 1, 20)
        })
        
        # Compute minimum distances
        from scipy.spatial.distance import pdist
        lhs_min_dist = np.min(pdist(lhs.design_coded[['X', 'Y']].values))
        random_min_dist = np.min(pdist(random_design.values))
        
        # LHS should have better (larger) minimum distance
        assert lhs_min_dist > random_min_dist
    
    def test_coverage_across_space(self):
        """Test that LHS covers the design space well."""
        factors = [
            Factor(name='X', type=FactorType.CONTINUOUS, min=0, max=1)
        ]
        
        n_runs = 20
        design = generate_latin_hypercube(factors, n_runs=n_runs, seed=42)
        
        # Divide space into quartiles
        # Each quartile should have roughly n_runs/4 points
        quartiles = [0, 0.25, 0.5, 0.75, 1.0]
        counts = np.histogram(design.design['X'], bins=quartiles)[0]
        
        # Each quartile should have 4-6 points (20/4 = 5 ± 1)
        assert all(3 <= c <= 7 for c in counts)