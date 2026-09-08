"""
Unit tests for full factorial design generation.
"""

import pytest
import pandas as pd
import numpy as np
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.full_factorial import (
    full_factorial,
    decode_design,
    get_design_summary
)


class TestFullFactorial:
    """Test full factorial design generation."""
    
    def test_2_factor_2_level(self):
        """Test basic 2^2 factorial design."""
        temp = Factor("Temperature", FactorType.CONTINUOUS, 
                     ChangeabilityLevel.EASY, levels=[150, 200])
        pressure = Factor("Pressure", FactorType.CONTINUOUS,
                         ChangeabilityLevel.EASY, levels=[50, 100])
        
        design = full_factorial([temp, pressure], randomize=False)
        
        # Should have 4 runs
        assert len(design) == 4
        
        # Should have correct columns
        assert 'StdOrder' in design.columns
        assert 'RunOrder' in design.columns
        assert 'Temperature' in design.columns
        assert 'Pressure' in design.columns
        
        # Should have all combinations of -1 and 1
        temp_levels = set(design['Temperature'].unique())
        pressure_levels = set(design['Pressure'].unique())
        assert temp_levels == {-1, 1}
        assert pressure_levels == {-1, 1}
    
    def test_3_factor_2_level(self):
        """Test 2^3 factorial design."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[10, 20]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[5, 15]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[100, 200])
        ]
        
        design = full_factorial(factors, randomize=False)
        
        # Should have 2^3 = 8 runs
        assert len(design) == 8
        
        # Each factor should have 2 levels
        for factor in factors:
            levels = set(design[factor.name].unique())
            assert levels == {-1, 1}
    
    def test_with_center_points(self):
        """Test factorial with center points."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        pressure = Factor("Pressure", FactorType.CONTINUOUS,
                         ChangeabilityLevel.EASY, levels=[50, 100])
        
        design = full_factorial([temp, pressure], n_center_points=3, randomize=False)
        
        # Should have 4 factorial runs + 3 center points = 7 total
        assert len(design) == 7
        
        # Should have center points (coded as 0)
        temp_levels = set(design['Temperature'].unique())
        pressure_levels = set(design['Pressure'].unique())
        assert 0 in temp_levels
        assert 0 in pressure_levels
        
        # Should have exactly 3 center point runs
        center_runs = design[(design['Temperature'] == 0) & (design['Pressure'] == 0)]
        assert len(center_runs) == 3
    
    def test_randomization(self):
        """Test that randomization changes run order."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        pressure = Factor("Pressure", FactorType.CONTINUOUS,
                         ChangeabilityLevel.EASY, levels=[50, 100])
        
        # Generate without randomization
        design_not_random = full_factorial([temp, pressure], randomize=False, random_seed=42)
        
        # Generate with randomization
        design_random = full_factorial([temp, pressure], randomize=True, random_seed=42)
        
        # Run orders should be different
        assert not design_not_random.equals(design_random)
        
        # But should have same rows (just different order)
        assert len(design_not_random) == len(design_random)
    
    def test_random_seed_reproducibility(self):
        """Test that same seed gives same randomization."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        pressure = Factor("Pressure", FactorType.CONTINUOUS,
                         ChangeabilityLevel.EASY, levels=[50, 100])
        
        design1 = full_factorial([temp, pressure], randomize=True, random_seed=123)
        design2 = full_factorial([temp, pressure], randomize=True, random_seed=123)
        
        # Should be identical
        pd.testing.assert_frame_equal(design1, design2)
    
    def test_with_blocking(self):
        """Test factorial with blocking."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        pressure = Factor("Pressure", FactorType.CONTINUOUS,
                         ChangeabilityLevel.EASY, levels=[50, 100])
        
        design = full_factorial([temp, pressure], n_blocks=2, randomize=False)
        
        # Should have Block column
        assert 'Block' in design.columns
        
        # Should have 2 blocks
        assert design['Block'].nunique() == 2
        
        # Each block should have 2 runs (4 runs / 2 blocks)
        block_sizes = design['Block'].value_counts()
        assert all(block_sizes == 2)
    
    def test_categorical_factor(self):
        """Test factorial with categorical factor."""
        material = Factor("Material", FactorType.CATEGORICAL,
                         ChangeabilityLevel.EASY, levels=["A", "B", "C"])
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        
        design = full_factorial([material, temp], randomize=False)
        
        # Should have 3 materials × 2 temps = 6 runs
        assert len(design) == 6
        
        # Material should have all 3 levels
        materials = set(design['Material'].unique())
        assert materials == {"A", "B", "C"}
        
        # Temperature should have 2 coded levels
        temps = set(design['Temperature'].unique())
        assert temps == {-1, 1}
    
    def test_discrete_numeric_factor(self):
        """Test factorial with discrete numeric factor."""
        rpm = Factor("RPM", FactorType.DISCRETE_NUMERIC,
                    ChangeabilityLevel.EASY, levels=[100, 150, 200])
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        
        design = full_factorial([rpm, temp], randomize=False)
        
        # Should have 3 RPMs × 2 temps = 6 runs
        assert len(design) == 6
        
        # RPM should have actual values
        rpms = set(design['RPM'].unique())
        assert rpms == {100, 150, 200}
    
    def test_mixed_factors(self):
        """Test factorial with mixed factor types."""
        factors = [
            Factor("Material", FactorType.CATEGORICAL, 
                  ChangeabilityLevel.EASY, levels=["A", "B"]),
            Factor("Temperature", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[150, 200]),
            Factor("RPM", FactorType.DISCRETE_NUMERIC,
                  ChangeabilityLevel.EASY, levels=[100, 200])
        ]
        
        design = full_factorial(factors, randomize=False)
        
        # Should have 2 × 2 × 2 = 8 runs
        assert len(design) == 8
        
        # Check each factor has correct levels
        assert set(design['Material'].unique()) == {"A", "B"}
        assert set(design['Temperature'].unique()) == {-1, 1}
        assert set(design['RPM'].unique()) == {100, 200}
    
    def test_empty_factors_raises_error(self):
        """Test that empty factor list raises error."""
        with pytest.raises(ValueError, match="At least one factor must be provided"):
            full_factorial([])
    
    def test_too_many_blocks_raises_error(self):
        """Test that too many blocks raises error."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        
        design = full_factorial([temp], randomize=False)
        n_runs = len(design)
        
        with pytest.raises(ValueError, match="cannot exceed number of runs"):
            full_factorial([temp], n_blocks=n_runs + 1)

    def test_with_replicates(self):
        """Test factorial with replicates."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        pressure = Factor("Pressure", FactorType.CONTINUOUS,
                         ChangeabilityLevel.EASY, levels=[50, 100])
        
        design = full_factorial([temp, pressure], n_replicates=2, randomize=False)
        
        # 4 factorial runs x 2 replicates = 8 runs
        assert len(design) == 8
        
        # Should have a Replicate column
        assert 'Replicate' in design.columns
        
        # Each replicate should appear 4 times
        replicate_counts = design['Replicate'].value_counts().sort_index()
        assert list(replicate_counts.index) == [1, 2]
        assert all(replicate_counts == 4)
        
        # Each factor combination should appear exactly twice (once per replicate)
        combo_counts = design.groupby(['Temperature', 'Pressure']).size()
        assert all(combo_counts == 2)

    def test_with_replicates_and_center_points(self):
        """Test factorial with replicates and center points."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        
        design = full_factorial([temp], n_center_points=2, n_replicates=3, randomize=False)
        
        # (2 factorial + 2 center) x 3 replicates = 12 runs
        assert len(design) == 12
        
        # Replicate column present with values 1, 2, 3
        assert set(design['Replicate'].unique()) == {1, 2, 3}
        
        # Each replicate has 4 runs (2 factorial + 2 center)
        replicate_counts = design['Replicate'].value_counts().sort_index()
        assert all(replicate_counts == 4)

    def test_replicates_run_order_sequential(self):
        """Test that RunOrder is sequential across all replicates."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        pressure = Factor("Pressure", FactorType.CONTINUOUS,
                         ChangeabilityLevel.EASY, levels=[50, 100])
        
        design = full_factorial([temp, pressure], n_replicates=3, randomize=False)
        
        # RunOrder should be 1 through 12
        assert list(design['RunOrder']) == list(range(1, 13))

    def test_single_replicate_no_column(self):
        """Test that n_replicates=1 (default) produces no Replicate column."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        
        design = full_factorial([temp], randomize=False)
        
        assert 'Replicate' not in design.columns
        assert len(design) == 2

    def test_invalid_replicates_raises_error(self):
        """Test that n_replicates < 1 raises error."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        
        with pytest.raises(ValueError, match="Number of replicates"):
            full_factorial([temp], n_replicates=0)

    def test_with_blocking_and_replicates(self):
        """Test factorial with both blocking and replicates."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        pressure = Factor("Pressure", FactorType.CONTINUOUS,
                         ChangeabilityLevel.EASY, levels=[50, 100])
        
        design = full_factorial(
            [temp, pressure], n_replicates=2, n_blocks=2, randomize=False
        )
        
        # 4 factorial points x 2 replicates = 8 runs
        assert len(design) == 8
        
        # Both Replicate and Block columns present
        assert 'Replicate' in design.columns
        assert 'Block' in design.columns
        
        # 2 replicates, 2 blocks
        assert set(design['Replicate'].unique()) == {1, 2}
        assert set(design['Block'].unique()) == {1, 2}
        
        # Blocks are balanced: 4 runs each
        block_counts = design['Block'].value_counts().sort_index()
        assert all(block_counts == 4)

    def test_blocking_balanced_sizes(self):
        """Test that blocks are evenly sized."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        pressure = Factor("Pressure", FactorType.CONTINUOUS,
                         ChangeabilityLevel.EASY, levels=[50, 100])
        rpm = Factor("RPM", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[100, 200])
        
        # 2^3 = 8 runs into 4 blocks = 2 runs each
        design = full_factorial([temp, pressure, rpm], n_blocks=4, randomize=False)
        
        assert 'Block' in design.columns
        block_sizes = design['Block'].value_counts().sort_index()
        assert all(block_sizes == 2)

    def test_unblocked_has_no_block_column(self):
        """Test that default (n_blocks=1) produces no Block column."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        
        design = full_factorial([temp], randomize=False)
        
        assert 'Block' not in design.columns


class TestDecodeDesign:
    """Test design decoding."""
    
    def test_decode_continuous(self):
        """Test decoding continuous factors."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        pressure = Factor("Pressure", FactorType.CONTINUOUS,
                         ChangeabilityLevel.EASY, levels=[50, 100])
        
        design = full_factorial([temp, pressure], randomize=False)
        decoded = decode_design(design, [temp, pressure])
        
        # Temperature should be decoded to actual values
        temp_values = set(decoded['Temperature'].unique())
        assert temp_values == {150, 200}
        
        # Pressure should be decoded
        pressure_values = set(decoded['Pressure'].unique())
        assert pressure_values == {50, 100}
    
    def test_decode_with_center_points(self):
        """Test decoding with center points."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        
        design = full_factorial([temp], n_center_points=2, randomize=False)
        decoded = decode_design(design, [temp])
        
        # Should have low (150), high (200), and center (175)
        temp_values = set(decoded['Temperature'].unique())
        assert temp_values == {150, 175, 200}
    
    def test_decode_categorical_unchanged(self):
        """Test that categorical factors remain unchanged."""
        material = Factor("Material", FactorType.CATEGORICAL,
                         ChangeabilityLevel.EASY, levels=["A", "B"])
        
        design = full_factorial([material], randomize=False)
        decoded = decode_design(design, [material])
        
        # Should be unchanged
        assert set(decoded['Material'].unique()) == {"A", "B"}


class TestDesignSummary:
    """Test design summary generation."""
    
    def test_basic_summary(self):
        """Test basic design summary."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        pressure = Factor("Pressure", FactorType.CONTINUOUS,
                         ChangeabilityLevel.EASY, levels=[50, 100])
        
        design = full_factorial([temp, pressure], randomize=False)
        summary = get_design_summary(design, [temp, pressure])
        
        assert summary['n_runs'] == 4
        assert summary['n_factors'] == 2
        assert summary['n_continuous'] == 2
        assert summary['n_discrete'] == 0
        assert summary['n_categorical'] == 0
        assert summary['has_center_points'] == False
        assert summary['n_blocks'] == 1
    
    def test_summary_with_center_points(self):
        """Test summary with center points."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        
        design = full_factorial([temp], n_center_points=3, randomize=False)
        summary = get_design_summary(design, [temp])
        
        assert summary['n_runs'] == 5  # 2 factorial + 3 center
        assert summary['has_center_points'] == True
    
    def test_summary_with_blocks(self):
        """Test summary with blocking."""
        temp = Factor("Temperature", FactorType.CONTINUOUS,
                     ChangeabilityLevel.EASY, levels=[150, 200])
        pressure = Factor("Pressure", FactorType.CONTINUOUS,
                         ChangeabilityLevel.EASY, levels=[50, 100])
        
        design = full_factorial([temp, pressure], n_blocks=2, randomize=False)
        summary = get_design_summary(design, [temp, pressure])
        
        assert summary['n_blocks'] == 2
    
    def test_summary_mixed_factors(self):
        """Test summary with mixed factor types."""
        factors = [
            Factor("Material", FactorType.CATEGORICAL,
                  ChangeabilityLevel.EASY, levels=["A", "B"]),
            Factor("Temperature", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[150, 200]),
            Factor("RPM", FactorType.DISCRETE_NUMERIC,
                  ChangeabilityLevel.EASY, levels=[100, 200])
        ]
        
        design = full_factorial(factors, randomize=False)
        summary = get_design_summary(design, factors)
        
        assert summary['n_factors'] == 3
        assert summary['n_continuous'] == 1
        assert summary['n_discrete'] == 1
        assert summary['n_categorical'] == 1