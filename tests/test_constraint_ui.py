"""
Test constraint builder UI component

Tests the constraint UI functions independently.
"""
import pytest
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.optimal_design import LinearConstraint
from src.ui.components.constraint_builder import (
    format_constraint_preview,
    validate_constraints
)


def test_format_constraint_preview_simple():
    """Test formatting simple constraint."""
    coefficients = {'Temperature': 1.0}
    bound = 100.0
    constraint_type = 'le'
    
    result = format_constraint_preview(coefficients, bound, constraint_type)
    
    assert 'Temperature' in result
    assert '≤' in result
    assert '100' in result


def test_format_constraint_preview_multiple_factors():
    """Test formatting constraint with multiple factors."""
    coefficients = {'Temperature': 1.0, 'Pressure': 0.5}
    bound = 150.0
    constraint_type = 'le'
    
    result = format_constraint_preview(coefficients, bound, constraint_type)
    
    assert 'Temperature' in result
    assert '0.5*Pressure' in result
    assert '≤' in result
    assert '150' in result


def test_format_constraint_preview_negative_coefficient():
    """Test formatting constraint with negative coefficient."""
    coefficients = {'Time': 1.0, 'Temperature': -2.0}
    bound = 50.0
    constraint_type = 'ge'
    
    result = format_constraint_preview(coefficients, bound, constraint_type)
    
    assert 'Time' in result
    assert '-2*Temperature' in result
    assert '≥' in result
    assert '50' in result


def test_format_constraint_preview_equality():
    """Test formatting equality constraint."""
    coefficients = {'A': 1.0, 'B': 1.0}
    bound = 100.0
    constraint_type = 'eq'
    
    result = format_constraint_preview(coefficients, bound, constraint_type)
    
    assert '=' in result
    assert '100' in result


def test_validate_constraints_valid():
    """Test validation with valid constraints."""
    factors = [
        Factor(
            name='Temperature',
            factor_type=FactorType.CONTINUOUS,
            changeability=ChangeabilityLevel.EASY,
            levels=[100, 200]
        ),
        Factor(
            name='Pressure',
            factor_type=FactorType.CONTINUOUS,
            changeability=ChangeabilityLevel.EASY,
            levels=[1, 5]
        )
    ]
    
    constraints = [
        LinearConstraint(
            coefficients={'Temperature': 1.0, 'Pressure': 0.5},
            bound=150.0,
            constraint_type='le'
        )
    ]
    
    is_valid, warnings = validate_constraints(constraints, factors)
    
    assert is_valid is True
    assert len(warnings) == 0


def test_validate_constraints_unknown_factor():
    """Test validation catches unknown factor."""
    factors = [
        Factor(
            name='Temperature',
            factor_type=FactorType.CONTINUOUS,
            changeability=ChangeabilityLevel.EASY,
            levels=[100, 200]
        )
    ]
    
    constraints = [
        LinearConstraint(
            coefficients={'Temperature': 1.0, 'UnknownFactor': 0.5},
            bound=150.0,
            constraint_type='le'
        )
    ]
    
    is_valid, warnings = validate_constraints(constraints, factors)
    
    assert is_valid is False
    assert len(warnings) > 0
    assert 'unknown' in warnings[0].lower()


def test_validate_constraints_categorical_factor():
    """Test validation rejects categorical factors in constraints."""
    factors = [
        Factor(
            name='Temperature',
            factor_type=FactorType.CONTINUOUS,
            changeability=ChangeabilityLevel.EASY,
            levels=[100, 200]
        ),
        Factor(
            name='Material',
            factor_type=FactorType.CATEGORICAL,
            changeability=ChangeabilityLevel.EASY,
            levels=['A', 'B', 'C']
        )
    ]
    
    constraints = [
        LinearConstraint(
            coefficients={'Temperature': 1.0, 'Material': 0.5},
            bound=150.0,
            constraint_type='le'
        )
    ]
    
    is_valid, warnings = validate_constraints(constraints, factors)
    
    # Should be invalid because Material is categorical
    assert is_valid is False
    assert any('Material' in w for w in warnings)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
