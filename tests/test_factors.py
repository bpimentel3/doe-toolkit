"""
Unit tests for factors module.
"""

import pytest
from src.core.factors import Factor, FactorType, ChangeabilityLevel


class TestFactorCreation:
    """Test factor creation and validation."""
    
    def test_continuous_factor(self):
        """Test creating a continuous factor."""
        temp = Factor(
            name="Temperature",
            factor_type=FactorType.CONTINUOUS,
            changeability=ChangeabilityLevel.HARD,
            levels=[150, 200],
            units="°C"
        )
        
        assert temp.name == "Temperature"
        assert temp.is_continuous()
        assert not temp.is_categorical()
        assert not temp.is_discrete_numeric()
        assert temp.get_n_levels() == 2
        assert temp.levels == [150, 200]
        assert temp.units == "°C"
    
    def test_categorical_factor(self):
        """Test creating a categorical factor."""
        material = Factor(
            name="Material",
            factor_type=FactorType.CATEGORICAL,
            changeability=ChangeabilityLevel.EASY,
            levels=["A", "B", "C"]
        )
        
        assert material.name == "Material"
        assert material.is_categorical()
        assert not material.is_continuous()
        assert material.get_n_levels() == 3
        assert material.levels == ["A", "B", "C"]
    
    def test_discrete_numeric_factor(self):
        """Test creating a discrete numeric factor."""
        rpm = Factor(
            name="RPM",
            factor_type=FactorType.DISCRETE_NUMERIC,
            changeability=ChangeabilityLevel.EASY,
            levels=[100, 150, 200, 250]
        )
        
        assert rpm.name == "RPM"
        assert rpm.is_discrete_numeric()
        assert not rpm.is_continuous()
        assert not rpm.is_categorical()
        assert rpm.get_n_levels() == 4
        assert rpm.levels == [100, 150, 200, 250]


class TestFactorValidation:
    """Test factor validation."""
    
    def test_empty_name_raises_error(self):
        """Test that empty name raises ValueError."""
        with pytest.raises(ValueError, match="Factor name cannot be empty"):
            Factor(
                name="",
                factor_type=FactorType.CONTINUOUS,
                changeability=ChangeabilityLevel.EASY,
                levels=[0, 100]
            )
    
    def test_no_levels_raises_error(self):
        """Test that missing levels raises ValueError."""
        with pytest.raises(ValueError, match="must have levels defined"):
            Factor(
                name="Temperature",
                factor_type=FactorType.CONTINUOUS,
                changeability=ChangeabilityLevel.EASY,
                levels=[]
            )
    
    def test_continuous_wrong_number_levels(self):
        """Test that continuous factor with wrong number of levels raises error."""
        with pytest.raises(ValueError, match="must have exactly 2 levels"):
            Factor(
                name="Temperature",
                factor_type=FactorType.CONTINUOUS,
                changeability=ChangeabilityLevel.EASY,
                levels=[100, 150, 200]
            )
    
    def test_continuous_min_greater_than_max(self):
        """Test that continuous factor with min > max raises error."""
        with pytest.raises(ValueError, match="min.*must be less than max"):
            Factor(
                name="Temperature",
                factor_type=FactorType.CONTINUOUS,
                changeability=ChangeabilityLevel.EASY,
                levels=[200, 150]
            )
    
    def test_discrete_numeric_non_numeric_levels(self):
        """Test that discrete numeric with non-numeric levels raises error."""
        with pytest.raises(ValueError, match="levels must be numeric"):
            Factor(
                name="Speed",
                factor_type=FactorType.DISCRETE_NUMERIC,
                changeability=ChangeabilityLevel.EASY,
                levels=[100, "fast", 200]
            )
    
    def test_discrete_numeric_duplicate_levels(self):
        """Test that discrete numeric with duplicates raises error."""
        with pytest.raises(ValueError, match="duplicate levels"):
            Factor(
                name="Speed",
                factor_type=FactorType.DISCRETE_NUMERIC,
                changeability=ChangeabilityLevel.EASY,
                levels=[100, 150, 150, 200]
            )
    
    def test_categorical_too_few_levels(self):
        """Test that categorical with only 1 level raises error."""
        with pytest.raises(ValueError, match="must have at least 2 levels"):
            Factor(
                name="Material",
                factor_type=FactorType.CATEGORICAL,
                changeability=ChangeabilityLevel.EASY,
                levels=["A"]
            )
    
    def test_categorical_duplicate_levels(self):
        """Test that categorical with duplicates raises error."""
        with pytest.raises(ValueError, match="duplicate levels"):
            Factor(
                name="Material",
                factor_type=FactorType.CATEGORICAL,
                changeability=ChangeabilityLevel.EASY,
                levels=["A", "B", "A"]
            )


class TestFactorMethods:
    """Test factor methods."""
    
    def test_to_dict(self):
        """Test converting factor to dictionary."""
        temp = Factor(
            name="Temperature",
            factor_type=FactorType.CONTINUOUS,
            changeability=ChangeabilityLevel.HARD,
            levels=[150, 200],
            units="°C"
        )
        
        result = temp.to_dict()
        
        assert result['name'] == "Temperature"
        assert result['type'] == "continuous"
        assert result['changeability'] == "hard"
        assert result['levels'] == [150, 200]
        assert result['units'] == "°C"
        assert result['n_levels'] == 2
    
    def test_repr(self):
        """Test string representation."""
        temp = Factor(
            name="Temperature",
            factor_type=FactorType.CONTINUOUS,
            changeability=ChangeabilityLevel.HARD,
            levels=[150, 200],
            units="°C"
        )
        
        result = repr(temp)
        
        assert "Temperature" in result
        assert "continuous" in result
        assert "hard" in result
        assert "150-200" in result
        assert "°C" in result