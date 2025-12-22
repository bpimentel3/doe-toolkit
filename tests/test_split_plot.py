"""
Tests for Split-Plot Design module.
"""

import pytest
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, 'src/core')

from split_plot import (
    generate_split_plot_design,
    evaluate_split_plot_design,
    SplitPlotDesign
)
from factors import Factor, FactorType, ChangeabilityLevel


class TestBasicSplitPlot:
    """Test basic split-plot design generation."""
    
    def test_simple_two_level_split_plot(self):
        """Test basic 2x2 split-plot with one hard and one easy factor."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS, 
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(factors, seed=42)
        
        assert isinstance(design, SplitPlotDesign)
        # 2 whole-plots (Temp: low, high) × 2 sub-plots (Time: low, high) = 4 runs
        assert design.n_runs == 4
        assert design.n_whole_plots == 2
        assert design.n_sub_plots_per_whole_plot == 2
        assert design.whole_plot_factors == ['Temp']
        assert design.sub_plot_factors == ['Time']
    
    def test_multiple_hard_factors(self):
        """Test split-plot with multiple hard-to-change factors."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Pressure', factor_type=FactorType.CONTINUOUS,
                   levels=[1, 5], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(factors, seed=42)
        
        # 4 whole-plots (2×2 for Temp, Pressure) × 2 sub-plots (Time) = 8 runs
        assert design.n_runs == 8
        assert design.n_whole_plots == 4
        assert design.n_sub_plots_per_whole_plot == 2
        assert len(design.whole_plot_factors) == 2
        assert len(design.sub_plot_factors) == 1
    
    def test_multiple_easy_factors(self):
        """Test split-plot with multiple easy-to-change factors."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY),
            Factor(name='Speed', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(factors, seed=42)
        
        # 2 whole-plots (Temp) × 4 sub-plots (2×2 for Time, Speed) = 8 runs
        assert design.n_runs == 8
        assert design.n_whole_plots == 2
        assert design.n_sub_plots_per_whole_plot == 4
        assert len(design.whole_plot_factors) == 1
        assert len(design.sub_plot_factors) == 2
    
    def test_three_level_nesting_very_hard(self):
        """Test split-plot with very-hard, hard, and easy factors."""
        factors = [
            Factor(name='Machine', factor_type=FactorType.CATEGORICAL,
                   levels=['M1', 'M2'], changeability=ChangeabilityLevel.VERY_HARD),
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(factors, seed=42)
        
        # 4 whole-plots (2 machines × 2 temps) × 2 sub-plots (time) = 8 runs
        assert design.n_runs == 8
        assert design.n_whole_plots == 4
        assert design.has_very_hard_factors
        assert 'VeryHardPlot' in design.design.columns
        assert 'Machine' in design.whole_plot_factors
        assert 'Temp' in design.whole_plot_factors
        assert 'Time' in design.sub_plot_factors


class TestReplicatesAndCenterPoints:
    """Test replicates and center point functionality."""
    
    def test_replicates(self):
        """Test that replicates multiply the design."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(factors, n_replicates=3, seed=42)
        
        # Base: 2 WP × 2 SP = 4 runs
        # With 3 replicates: 4 × 3 = 12 runs
        assert design.n_runs == 12
        assert design.n_whole_plots == 6  # 2 WP × 3 replicates
        
        # Check replicate column
        assert 'Replicate' in design.design.columns
        assert design.design['Replicate'].min() == 1
        assert design.design['Replicate'].max() == 3
    
    def test_center_points_at_subplot_level(self):
        """Test that center points are added at sub-plot level only."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(
            factors, n_center_points=2, seed=42
        )
        
        # Base: 2 WP × 2 SP = 4 runs
        # Center points: 2 per WP × 2 WP = 4 additional runs
        # Total: 8 runs
        assert design.n_runs == 8
        
        # Find center point runs (Time should be at center)
        center_time = (10 + 30) / 2
        center_runs = design.design[
            np.isclose(design.design['Time'], center_time)
        ]
        assert len(center_runs) == 4  # 2 per whole-plot
        
        # Temp should NOT be at center for center point runs
        # (whole-plot factors stay at their WP values)
        center_temp = (100 + 200) / 2
        assert not all(np.isclose(center_runs['Temp'], center_temp))
    
    def test_center_points_with_categorical(self):
        """Test center points with categorical sub-plot factors."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY),
            Factor(name='Catalyst', factor_type=FactorType.CATEGORICAL,
                   levels=['A', 'B'], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(
            factors, n_center_points=1, seed=42
        )
        
        # Should not crash with categorical factors
        assert design.n_runs > 0


class TestBlocking:
    """Test blocking functionality in split-plot designs."""
    
    def test_basic_blocking(self):
        """Test that blocking divides whole-plots correctly."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(
            factors, n_replicates=2, n_blocks=2, seed=42
        )
        
        # 2 WP × 2 replicates = 4 whole-plots, divided into 2 blocks
        assert design.n_blocks == 2
        assert 'Block' in design.design.columns
        
        # Each block should have 2 whole-plots
        block_wp_counts = design.design.groupby('Block')['WholePlot'].nunique()
        assert all(block_wp_counts == 2)
    
    def test_blocking_preserves_structure(self):
        """Test that blocking doesn't break sub-plot structure."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(
            factors, n_replicates=4, n_blocks=2, seed=42
        )
        
        # Within each block, each whole-plot should have same sub-plots
        for block in design.design['Block'].unique():
            block_data = design.design[design.design['Block'] == block]
            for wp in block_data['WholePlot'].unique():
                wp_data = block_data[block_data['WholePlot'] == wp]
                # Should have 2 sub-plots (Time: low, high)
                assert len(wp_data) == 2


class TestRandomization:
    """Test randomization options."""
    
    def test_no_randomization(self):
        """Test design with no randomization."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(
            factors,
            randomize_whole_plots=False,
            randomize_sub_plots=False,
            seed=42
        )
        
        # Without randomization, should get systematic order
        # First WP: Temp=low, Second WP: Temp=high
        wp1_data = design.design[design.design['WholePlot'] == 1]
        wp2_data = design.design[design.design['WholePlot'] == 2]
        
        assert all(wp1_data['Temp'] == 100)
        assert all(wp2_data['Temp'] == 200)
    
    def test_whole_plot_randomization_only(self):
        """Test randomizing whole-plots but not sub-plots."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(
            factors,
            randomize_whole_plots=True,
            randomize_sub_plots=False,
            seed=42
        )
        
        # Whole-plot order should be randomized
        # But within each WP, sub-plots should be systematic
        assert design.n_runs == 4
    
    def test_subplot_randomization_only(self):
        """Test randomizing sub-plots but not whole-plots."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(
            factors,
            randomize_whole_plots=False,
            randomize_sub_plots=True,
            seed=42
        )
        
        # Whole-plots should be in order
        # Sub-plots within each WP should be randomized
        assert design.n_runs == 4
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same design."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design1 = generate_split_plot_design(factors, seed=42)
        design2 = generate_split_plot_design(factors, seed=42)
        
        pd.testing.assert_frame_equal(design1.design, design2.design)


class TestFactorTypes:
    """Test different factor type combinations."""
    
    def test_discrete_factors(self):
        """Test split-plot with discrete numeric factors."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.DISCRETE_NUMERIC,
                   levels=[100, 150, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.DISCRETE_NUMERIC,
                   levels=[10, 20, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(factors, seed=42)
        
        # 3 WP × 3 SP = 9 runs
        assert design.n_runs == 9
        assert design.n_whole_plots == 3
        
        # All values should be from allowed levels
        assert set(design.design['Temp'].unique()) == {100, 150, 200}
        assert set(design.design['Time'].unique()) == {10, 20, 30}
    
    def test_categorical_hard_factor(self):
        """Test split-plot with categorical hard-to-change factor."""
        factors = [
            Factor(name='Machine', factor_type=FactorType.CATEGORICAL,
                   levels=['A', 'B', 'C'], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(factors, seed=42)
        
        # 3 WP (machines) × 2 SP (time) = 6 runs
        assert design.n_runs == 6
        assert design.n_whole_plots == 3
        assert set(design.design['Machine'].unique()) == {'A', 'B', 'C'}
    
    def test_categorical_easy_factor(self):
        """Test split-plot with categorical easy-to-change factor."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Catalyst', factor_type=FactorType.CATEGORICAL,
                   levels=['X', 'Y'], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(factors, seed=42)
        
        # 2 WP × 2 SP = 4 runs
        assert design.n_runs == 4
        assert set(design.design['Catalyst'].unique()) == {'X', 'Y'}
    
    def test_mixed_factor_types(self):
        """Test split-plot with all factor types."""
        factors = [
            Factor(name='Line', factor_type=FactorType.CATEGORICAL,
                   levels=['L1', 'L2'], changeability=ChangeabilityLevel.VERY_HARD),
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Speed', factor_type=FactorType.DISCRETE_NUMERIC,
                   levels=[10, 20], changeability=ChangeabilityLevel.EASY),
            Factor(name='Tool', factor_type=FactorType.CATEGORICAL,
                   levels=['A', 'B'], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(factors, seed=42)
        
        # 4 WP (2 lines × 2 temps) × 4 SP (2 speeds × 2 tools) = 16 runs
        assert design.n_runs == 16
        assert design.n_whole_plots == 4
        assert design.has_very_hard_factors


class TestCodedLevels:
    """Test coded level calculations."""
    
    def test_coded_levels_continuous(self):
        """Test coding of continuous factors."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(factors, seed=42)
        
        # Low level should code to -1
        low_temp_runs = design.design[design.design['Temp'] == 100]
        assert all(design.design_coded.loc[low_temp_runs.index, 'Temp'] == -1.0)
        
        # High level should code to +1
        high_temp_runs = design.design[design.design['Temp'] == 200]
        assert all(design.design_coded.loc[high_temp_runs.index, 'Temp'] == 1.0)
    
    def test_coded_levels_discrete(self):
        """Test coding of discrete factors."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.DISCRETE_NUMERIC,
                   levels=[100, 150, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(factors, seed=42)
        
        # Min (100) should code to -1
        assert design.design_coded[design.design['Temp'] == 100]['Temp'].iloc[0] == -1.0
        
        # Max (200) should code to +1
        assert design.design_coded[design.design['Temp'] == 200]['Temp'].iloc[0] == 1.0
        
        # Middle (150) should code to 0
        assert design.design_coded[design.design['Temp'] == 150]['Temp'].iloc[0] == 0.0
    
    def test_categorical_not_coded(self):
        """Test that categorical factors are not numerically coded."""
        factors = [
            Factor(name='Machine', factor_type=FactorType.CATEGORICAL,
                   levels=['A', 'B'], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        design = generate_split_plot_design(factors, seed=42)
        
        # Categorical should remain unchanged in coded design
        assert design.design_coded['Machine'].dtype == object
        assert set(design.design_coded['Machine'].unique()) == {'A', 'B'}


class TestValidation:
    """Test input validation and error handling."""
    
    def test_no_factors(self):
        """Test that empty factor list raises error."""
        with pytest.raises(ValueError, match="At least one factor must be provided"):
            generate_split_plot_design([])
    
    def test_all_easy_factors(self):
        """Test that all-EASY factors raises error."""
        factors = [
            Factor(name='A', factor_type=FactorType.CONTINUOUS,
                   levels=[0, 1], changeability=ChangeabilityLevel.EASY),
            Factor(name='B', factor_type=FactorType.CONTINUOUS,
                   levels=[0, 1], changeability=ChangeabilityLevel.EASY)
        ]
        
        with pytest.raises(ValueError, match="Split-plot requires at least one HARD"):
            generate_split_plot_design(factors)
    
    def test_insufficient_subplots(self):
        """Test warning for too few sub-plots (no easy factors)."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            # No Easy factors, so sub-plots per whole-plot = 1
        ]
        
        with pytest.raises(ValueError, match="Need at least 2 sub-plots"):
            generate_split_plot_design(factors)
    
    def test_invalid_replicates(self):
        """Test that n_replicates < 1 raises error."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        with pytest.raises(ValueError, match="n_replicates must be at least 1"):
            generate_split_plot_design(factors, n_replicates=0)
    
    def test_invalid_center_points(self):
        """Test that negative center points raises error."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        with pytest.raises(ValueError, match="n_center_points cannot be negative"):
            generate_split_plot_design(factors, n_center_points=-1)
    
    def test_too_many_blocks(self):
        """Test that more blocks than whole-plots raises error."""
        factors = [
            Factor(name='Temp', factor_type=FactorType.CONTINUOUS,
                   levels=[100, 200], changeability=ChangeabilityLevel.HARD),
            Factor(name='Time', factor_type=FactorType.CONTINUOUS,
                   levels=[10, 30], changeability=ChangeabilityLevel.EASY)
        ]
        
        with pytest.raises(ValueError, match="Cannot have more blocks"):
            generate_split_plot_design(factors, n_blocks=5)