"""
Unit tests for fractional factorial designs.
"""

import pytest
import pandas as pd
import numpy as np
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.fractional_factorial import FractionalFactorial


class TestFractionalFactorialCreation:
    """Test fractional factorial design creation."""
    
    def test_2_5_1_design(self):
        """Test 2^(5-1) fractional factorial (Resolution V)."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(5)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2", resolution=5)
        
        assert ff.k == 5
        assert ff.p == 1
        assert ff.resolution == 5
        
        design = ff.generate(randomize=False)
        
        # Should have 2^(5-1) = 16 runs
        assert len(design) == 16
        
        # Should have all 5 factors
        for factor in factors:
            assert factor.name in design.columns
    
    def test_2_4_1_design(self):
        """Test 2^(4-1) fractional factorial (Resolution IV)."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2", resolution=4)
        
        assert ff.k == 4
        assert ff.p == 1
        assert ff.resolution == 4
        
        design = ff.generate(randomize=False)
        
        # Should have 2^(4-1) = 8 runs
        assert len(design) == 8
    
    def test_2_7_3_design(self):
        """Test 2^(7-3) fractional factorial (Resolution IV)."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(7)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/8", resolution=4)
        
        assert ff.k == 7
        assert ff.p == 3
        
        design = ff.generate(randomize=False)
        
        # Should have 2^(7-3) = 16 runs
        assert len(design) == 16
    
    def test_custom_generators(self):
        """Test fractional factorial with custom generators."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(5)
        ]
        
        ff = FractionalFactorial(
            factors,
            fraction="1/2",
            generators=["E=ABCD"]
        )
        
        # Check generator was parsed correctly
        assert ff.generators == [("E", "ABCD")]
    
    def test_with_blocking(self):
        """Test fractional factorial with blocking."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2")
        design = ff.generate(randomize=False, n_blocks=2)
        
        # Should have Block column
        assert 'Block' in design.columns
        
        # Should have 2 blocks
        assert design['Block'].nunique() == 2
        
        # Each block should have 4 runs (8 runs / 2 blocks)
        assert all(design['Block'].value_counts() == 4)
    
    def test_randomization_reproducibility(self):
        """Test that same seed gives same design."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        ff1 = FractionalFactorial(factors, fraction="1/2")
        design1 = ff1.generate(randomize=True, random_seed=42)
        
        ff2 = FractionalFactorial(factors, fraction="1/2")
        design2 = ff2.generate(randomize=True, random_seed=42)
        
        pd.testing.assert_frame_equal(design1, design2)


class TestDefiningRelation:
    """Test defining relation calculation."""
    
    def test_2_5_1_defining_relation(self):
        """Test defining relation for 2^(5-1) design."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(5)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2", resolution=5)
        
        # For E=ABCD, defining relation should be I=ABCDE
        assert "I" in ff.defining_relation
        assert "ABCDE" in ff.defining_relation
    
    def test_2_4_1_defining_relation(self):
        """Test defining relation for 2^(4-1) design."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2", resolution=4)
        
        # For D=ABC, defining relation should be I=ABCD
        assert "I" in ff.defining_relation
        assert "ABCD" in ff.defining_relation


class TestAliasStructure:
    """Test alias structure calculation."""
    
    def test_resolution_v_main_effects_clear(self):
        """Test that Resolution V has clear main effects and 2FI."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(5)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2", resolution=5)
        
        # Main effects should be clear (aliased with 4FI or higher)
        for letter in "ABCDE":
            if letter in ff.alias_structure:
                aliases = ff.alias_structure[letter]
                # Should not be aliased with any main effects or 2FI
                for alias in aliases:
                    assert len(alias) >= 4, f"{letter} aliased with {alias}"
    
    def test_resolution_iv_2fi_aliased(self):
        """Test that Resolution IV has 2FI aliased with other 2FI."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2", resolution=4)
        
        # Main effects should be clear
        for letter in "ABCD":
            if letter in ff.alias_structure:
                aliases = ff.alias_structure[letter]
                for alias in aliases:
                    assert len(alias) >= 3, f"Main effect {letter} aliased with {alias}"
    
    def test_alias_summary_dataframe(self):
        """Test that alias summary returns valid DataFrame."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2")
        summary = ff.get_alias_summary()
        
        # Should be a DataFrame
        assert isinstance(summary, pd.DataFrame)
        
        # Should have required columns
        assert 'Effect' in summary.columns
        assert 'Aliased_With' in summary.columns
        
        # Should have entries
        assert len(summary) > 0


class TestValidation:
    """Test input validation."""
    
    def test_too_few_factors(self):
        """Test that < 3 factors raises error."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        ]
        
        with pytest.raises(ValueError, match="at least 3 factors"):
            FractionalFactorial(factors, fraction="1/2")
    
    def test_categorical_factor_rejected(self):
        """Test that categorical factors are rejected."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("C", FactorType.CATEGORICAL, ChangeabilityLevel.EASY, levels=["X", "Y"])
        ]
        
        with pytest.raises(ValueError, match="must be continuous or discrete numeric"):
            FractionalFactorial(factors, fraction="1/2")
    
    def test_non_2_level_discrete_rejected(self):
        """Test that non-2-level discrete factors are rejected."""
        factors = [
            Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
            Factor("C", FactorType.DISCRETE_NUMERIC, ChangeabilityLevel.EASY, 
                  levels=[100, 150, 200])
        ]
        
        with pytest.raises(ValueError, match="must have exactly 2 levels"):
            FractionalFactorial(factors, fraction="1/2")
    
    def test_invalid_fraction_format(self):
        """Test that invalid fraction format raises error."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        with pytest.raises(ValueError, match="must be in format"):
            FractionalFactorial(factors, fraction="1/3")
    
    def test_too_large_fraction(self):
        """Test that fraction >= k raises error."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        with pytest.raises(ValueError, match="must be less than number of factors"):
            FractionalFactorial(factors, fraction="1/16")


class TestDesignGeneration:
    """Test the actual design matrix generation."""
    
    def test_generator_multiplication(self):
        """Test that generated factors follow generator rule."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(4)
        ]
        
        # D = ABC
        ff = FractionalFactorial(factors, fraction="1/2", generators=["D=ABC"])
        design = ff.generate(randomize=False)
        
        # Verify D = A * B * C for all rows
        factor_names = [f.name for f in factors]
        for idx, row in design.iterrows():
            expected_d = row[factor_names[0]] * row[factor_names[1]] * row[factor_names[2]]
            actual_d = row[factor_names[3]]
            assert expected_d == actual_d, f"Row {idx}: D should equal A*B*C"
    
    def test_all_runs_unique(self):
        """Test that all runs are unique."""
        factors = [
            Factor(f"Factor_{chr(65+i)}", FactorType.CONTINUOUS,
                  ChangeabilityLevel.EASY, levels=[-1, 1])
            for i in range(5)
        ]
        
        ff = FractionalFactorial(factors, fraction="1/2")
        design = ff.generate(randomize=False)
        
        # Get factor columns only
        factor_cols = [f.name for f in factors]
        
        # Check for duplicates
        n_unique = design[factor_cols].drop_duplicates().shape[0]
        assert n_unique == len(design), "Design contains duplicate runs"