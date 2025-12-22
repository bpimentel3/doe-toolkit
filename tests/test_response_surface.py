"""
Unit tests for response surface designs.
"""

import pytest
import pandas as pd
import numpy as np
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.response_surface import CentralCompositeDesign, BoxBehnkenDesign


class TestCentralCompositeDesign:
    """Test Central Composite Design generation."""
    
    def test_rotatable_ccd_3_factors(self):
        """Test rotatable CCD with 3 factors."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[10, 20]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[50, 100]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[1, 5])
        ]
        
        ccd = CentralCompositeDesign(factors, alpha="rotatable", center_points=6)
        
        # Check properties
        assert ccd.k == 3
        assert ccd.n_factorial == 8  # 2^3
        assert ccd.n_axial == 6  # 2*3
        assert ccd.n_center == 6
        assert ccd.n_total == 20  # 8+6+6
        
        # Alpha for rotatable 3-factor: (8)^0.25 = 1.682
        assert abs(ccd.alpha - 1.682) < 0.01
        
        design = ccd.generate(randomize=False)
        
        # Check total runs
        assert len(design) == 20
        
        # Check point types
        assert (design['PointType'] == 'Factorial').sum() == 8
        assert (design['PointType'] == 'Axial').sum() == 6
        assert (design['PointType'] == 'Center').sum() == 6
    
    def test_face_centered_ccd(self):
        """Test face-centered CCD (alpha=1)."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        ccd = CentralCompositeDesign(factors, alpha="face", center_points=5)
        
        # Alpha should be 1.0
        assert ccd.alpha == 1.0
        
        design = ccd.generate(randomize=False)
        
        # Check that axial points are at ±1
        axial_points = design[design['PointType'] == 'Axial']
        max_value = axial_points[['A', 'B']].abs().max().max()
        assert max_value == 1.0
    
    def test_custom_alpha(self):
        """Test CCD with custom alpha."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        custom_alpha = 1.5
        ccd = CentralCompositeDesign(factors, alpha=custom_alpha, center_points=3)
        
        assert ccd.alpha == custom_alpha
        assert ccd.design_type == "custom"
        
        design = ccd.generate(randomize=False)
        
        # Check that axial points use custom alpha
        axial_points = design[design['PointType'] == 'Axial']
        max_value = axial_points[['A', 'B']].abs().max().max()
        assert abs(max_value - custom_alpha) < 0.001
    
    def test_factorial_points_correct(self):
        """Test that factorial points are correct."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        ccd = CentralCompositeDesign(factors, alpha=1, center_points=0)
        design = ccd.generate(randomize=False)
        
        # Extract factorial points
        factorial = design[design['PointType'] == 'Factorial'][['A', 'B']]
        
        # Should have all 2^2 = 4 combinations of ±1
        expected_combinations = {(-1, -1), (-1, 1), (1, -1), (1, 1)}
        actual_combinations = set(factorial.itertuples(index=False, name=None))
        
        assert actual_combinations == expected_combinations
    
    def test_axial_points_correct(self):
        """Test that axial points are correct."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        alpha = 1.414
        ccd = CentralCompositeDesign(factors, alpha=alpha, center_points=0)
        design = ccd.generate(randomize=False)
        
        # Extract axial points
        axial = design[design['PointType'] == 'Axial'][['A', 'B']]
        
        # Should have 2*k = 4 points
        assert len(axial) == 4
        
        # Each point should be along an axis
        for idx, row in axial.iterrows():
            # One coordinate should be ±alpha, other should be 0
            non_zero = row[abs(row) > 0.01]
            assert len(non_zero) == 1
            assert abs(abs(non_zero.iloc[0]) - alpha) < 0.01
    
    def test_center_points_correct(self):
        """Test that center points are at origin."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        n_center = 5
        ccd = CentralCompositeDesign(factors, alpha=1, center_points=n_center)
        design = ccd.generate(randomize=False)
        
        # Extract center points
        center = design[design['PointType'] == 'Center'][['A', 'B']]
        
        # Should have n_center points
        assert len(center) == n_center
        
        # All should be at (0, 0)
        assert (center == 0).all().all()
    
    def test_randomization_reproducibility(self):
        """Test that same seed gives same randomization."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        ccd1 = CentralCompositeDesign(factors, alpha=1, center_points=3)
        design1 = ccd1.generate(randomize=True, random_seed=42)
        
        ccd2 = CentralCompositeDesign(factors, alpha=1, center_points=3)
        design2 = ccd2.generate(randomize=True, random_seed=42)
        
        pd.testing.assert_frame_equal(design1, design2)
    
    def test_get_design_properties(self):
        """Test design properties method."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        ccd = CentralCompositeDesign(factors, alpha="rotatable", center_points=6)
        props = ccd.get_design_properties()
        
        assert props['design_type'] == 'Central Composite Design'
        assert props['n_factors'] == 3
        assert props['n_total_runs'] == 20
        assert props['rotatable'] == True


class TestBoxBehnkenDesign:
    """Test Box-Behnken Design generation."""
    
    def test_3_factor_bbd(self):
        """Test Box-Behnken design with 3 factors."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[10, 20]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[50, 100]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[1, 5])
        ]
        
        bbd = BoxBehnkenDesign(factors, center_points=3)
        
        # Check properties
        assert bbd.k == 3
        assert bbd.n_factorial == 12  # 2*3*(3-1)
        assert bbd.n_center == 3
        assert bbd.n_total == 15
        
        design = bbd.generate(randomize=False)
        
        # Check total runs
        assert len(design) == 15
        
        # Check point types
        assert (design['PointType'] == 'Factorial').sum() == 12
        assert (design['PointType'] == 'Center').sum() == 3
    
    def test_4_factor_bbd(self):
        """Test Box-Behnken design with 4 factors."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[0, 100])
            for i in range(4)
        ]
        
        bbd = BoxBehnkenDesign(factors, center_points=3)
        
        # 4 factors: 2*4*(4-1) = 24 factorial points
        assert bbd.n_factorial == 24
        assert bbd.n_total == 27
        
        design = bbd.generate(randomize=False)
        assert len(design) == 27
    
    def test_factorial_points_structure(self):
        """Test that factorial points have correct structure."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        bbd = BoxBehnkenDesign(factors, center_points=0)
        design = bbd.generate(randomize=False)
        
        # Extract factorial points
        factorial = design[design['PointType'] == 'Factorial'][['A', 'B', 'C']]
        
        # Each row should have exactly two non-zero values (±1) and one zero
        for idx, row in factorial.iterrows():
            non_zero = row[abs(row) > 0.01]
            zero_values = row[abs(row) < 0.01]
            
            assert len(non_zero) == 2, f"Row {idx} should have 2 non-zero values"
            assert len(zero_values) == 1, f"Row {idx} should have 1 zero value"
            
            # Non-zero values should be ±1
            assert all(abs(abs(val) - 1.0) < 0.01 for val in non_zero)
    
    def test_no_extreme_corners(self):
        """Test that Box-Behnken has no corner points."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        bbd = BoxBehnkenDesign(factors, center_points=0)
        design = bbd.generate(randomize=False)
        
        # Extract factorial points
        factorial = design[design['PointType'] == 'Factorial'][['A', 'B', 'C']]
        
        # Check that no row has all values at ±1 (no corners)
        for idx, row in factorial.iterrows():
            # At least one value should be 0
            has_zero = any(abs(val) < 0.01 for val in row)
            assert has_zero, f"Row {idx} appears to be a corner point"
    
    def test_center_points_correct(self):
        """Test that center points are at origin."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        n_center = 5
        bbd = BoxBehnkenDesign(factors, center_points=n_center)
        design = bbd.generate(randomize=False)
        
        # Extract center points
        center = design[design['PointType'] == 'Center'][['A', 'B', 'C']]
        
        # Should have n_center points
        assert len(center) == n_center
        
        # All should be at (0, 0, 0)
        assert (center == 0).all().all()
    
    def test_randomization_reproducibility(self):
        """Test that same seed gives same randomization."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        bbd1 = BoxBehnkenDesign(factors, center_points=3)
        design1 = bbd1.generate(randomize=True, random_seed=42)
        
        bbd2 = BoxBehnkenDesign(factors, center_points=3)
        design2 = bbd2.generate(randomize=True, random_seed=42)
        
        pd.testing.assert_frame_equal(design1, design2)
    
    def test_get_design_properties(self):
        """Test design properties method."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        bbd = BoxBehnkenDesign(factors, center_points=3)
        props = bbd.get_design_properties()
        
        assert props['design_type'] == 'Box-Behnken Design'
        assert props['n_factors'] == 3
        assert props['n_total_runs'] == 15
        assert props['spherical'] == True
        assert props['no_extreme_points'] == True


class TestValidation:
    """Test input validation for response surface designs."""
    
    def test_ccd_too_few_factors(self):
        """Test that CCD rejects < 2 factors."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        with pytest.raises(ValueError, match="at least 2 factors"):
            CentralCompositeDesign(factors)
    
    def test_bbd_too_few_factors(self):
        """Test that BBD rejects < 3 factors."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        with pytest.raises(ValueError, match="at least 3 factors"):
            BoxBehnkenDesign(factors)
    
    def test_categorical_factor_rejected(self):
        """Test that categorical factors are rejected."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CATEGORICAL, ChangeabilityLevel.EASY, levels=["X", "Y"])
        ]
        
        with pytest.raises(ValueError, match="must be continuous"):
            CentralCompositeDesign(factors)
    
    def test_too_many_factors_warning(self):
        """Test that too many factors gives warning."""
        factors = [
            Factor(f"Factor_{i}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[0, 100])
            for i in range(11)
        ]
        
        with pytest.raises(ValueError, match="impractical"):
            CentralCompositeDesign(factors)


class TestDesignComparison:
    """Compare CCD and BBD properties."""
    
    def test_run_efficiency_3_factors(self):
        """Compare run counts for 3 factors."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100]),
            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 100])
        ]
        
        ccd = CentralCompositeDesign(factors, alpha="rotatable", center_points=6)
        bbd = BoxBehnkenDesign(factors, center_points=3)
        
        # BBD should be more efficient for 3 factors
        assert bbd.n_total < ccd.n_total
        # BBD: 15 runs, CCD: 20 runs
        assert bbd.n_total == 15
        assert ccd.n_total == 20