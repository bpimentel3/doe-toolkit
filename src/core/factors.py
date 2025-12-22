"""
Factor definitions for DOE-Toolkit.

This module handles the definition and validation of experimental factors
including continuous, discrete numeric, and categorical types.
"""

from enum import Enum
from typing import List, Optional, Union, Dict, Any
from dataclasses import dataclass, field
import numpy as np


class FactorType(Enum):
    """Type of experimental factor."""
    CONTINUOUS = "continuous"
    DISCRETE_NUMERIC = "discrete_numeric"
    CATEGORICAL = "categorical"


class ChangeabilityLevel(Enum):
    """
    How difficult it is to change a factor during experimentation.
    
    This affects the experimental design structure:
    - EASY: Can be changed for every run (sub-plot factors)
    - HARD: Changed infrequently, defines whole-plots
    - VERY_HARD: Changed rarely or never, defines whole-whole-plots
    """
    EASY = "easy"
    HARD = "hard"
    VERY_HARD = "very_hard"


@dataclass
class Factor:
    """
    Represents an experimental factor.
    
    Parameters
    ----------
    name : str
        Factor name (e.g., "Temperature", "Pressure")
    factor_type : FactorType
        Type of factor (continuous, discrete_numeric, or categorical)
    changeability : ChangeabilityLevel, optional
        How easy it is to change this factor (default: EASY)
    levels : list, optional
        For discrete_numeric or categorical: list of possible values
        For continuous: [min, max] range
    units : str, optional
        Units of measurement (e.g., "°C", "psi", "minutes")
    
    Examples
    --------
    >>> # Continuous factor
    >>> temp = Factor(
    ...     name="Temperature",
    ...     factor_type=FactorType.CONTINUOUS,
    ...     changeability=ChangeabilityLevel.HARD,
    ...     levels=[150, 200],
    ...     units="°C"
    ... )
    
    >>> # Categorical factor
    >>> material = Factor(
    ...     name="Material",
    ...     factor_type=FactorType.CATEGORICAL,
    ...     changeability=ChangeabilityLevel.EASY,
    ...     levels=["A", "B", "C"]
    ... )
    
    >>> # Discrete numeric factor
    >>> rpm = Factor(
    ...     name="RPM",
    ...     factor_type=FactorType.DISCRETE_NUMERIC,
    ...     changeability=ChangeabilityLevel.EASY,
    ...     levels=[100, 150, 200, 250]
    ... )
    """
    name: str
    factor_type: FactorType
    changeability: ChangeabilityLevel = field(default=ChangeabilityLevel.EASY)
    levels: Optional[List[Union[float, int, str]]] = None
    units: Optional[str] = None
    
    def __post_init__(self):
        """Validate factor definition after initialization."""
        self.validate()
    
    def validate(self) -> None:
        """
        Validate factor definition.
        
        Raises
        ------
        ValueError
            If factor definition is invalid
        """
        # Name must be non-empty
        if not self.name or not self.name.strip():
            raise ValueError("Factor name cannot be empty")
        
        # Levels must be provided
        if self.levels is None or len(self.levels) == 0:
            raise ValueError(f"Factor '{self.name}' must have levels defined")
        
        # Validate based on factor type
        if self.factor_type == FactorType.CONTINUOUS:
            self._validate_continuous()
        elif self.factor_type == FactorType.DISCRETE_NUMERIC:
            self._validate_discrete_numeric()
        elif self.factor_type == FactorType.CATEGORICAL:
            self._validate_categorical()
    
    def _validate_continuous(self) -> None:
        """Validate continuous factor."""
        if len(self.levels) != 2:
            raise ValueError(
                f"Continuous factor '{self.name}' must have exactly 2 levels "
                f"[min, max], got {len(self.levels)}"
            )
        
        min_val, max_val = self.levels
        
        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            raise ValueError(
                f"Continuous factor '{self.name}' levels must be numeric"
            )
        
        if min_val >= max_val:
            raise ValueError(
                f"Continuous factor '{self.name}': min ({min_val}) must be "
                f"less than max ({max_val})"
            )
    
    def _validate_discrete_numeric(self) -> None:
        """Validate discrete numeric factor."""
        if len(self.levels) < 2:
            raise ValueError(
                f"Discrete numeric factor '{self.name}' must have at least 2 levels"
            )
        
        for level in self.levels:
            if not isinstance(level, (int, float)):
                raise ValueError(
                    f"Discrete numeric factor '{self.name}' levels must be numeric, "
                    f"got {type(level).__name__}"
                )
        
        # Check for duplicates
        if len(self.levels) != len(set(self.levels)):
            raise ValueError(
                f"Discrete numeric factor '{self.name}' has duplicate levels"
            )
    
    def _validate_categorical(self) -> None:
        """Validate categorical factor."""
        if len(self.levels) < 2:
            raise ValueError(
                f"Categorical factor '{self.name}' must have at least 2 levels"
            )
        
        # Check for duplicates
        if len(self.levels) != len(set(self.levels)):
            raise ValueError(
                f"Categorical factor '{self.name}' has duplicate levels"
            )
        
        # Convert all to strings
        self.levels = [str(level) for level in self.levels]
    
    def get_n_levels(self) -> int:
        """
        Get number of levels.
        
        Returns
        -------
        int
            Number of discrete levels (2 for continuous factors representing min/max)
        """
        return len(self.levels)
    
    def is_continuous(self) -> bool:
        """Check if factor is continuous."""
        return self.factor_type == FactorType.CONTINUOUS
    
    def is_categorical(self) -> bool:
        """Check if factor is categorical."""
        return self.factor_type == FactorType.CATEGORICAL
    
    def is_discrete_numeric(self) -> bool:
        """Check if factor is discrete numeric."""
        return self.factor_type == FactorType.DISCRETE_NUMERIC
    
    @property
    def min(self) -> Optional[float]:
        """
        Get minimum value for continuous/discrete numeric factors.
        
        Returns
        -------
        float or None
            Minimum value, or None for categorical factors
        """
        if self.is_continuous() or self.is_discrete_numeric():
            return min(self.levels) if self.levels else None
        return None
    
    @property
    def max(self) -> Optional[float]:
        """
        Get maximum value for continuous/discrete numeric factors.
        
        Returns
        -------
        float or None
            Maximum value, or None for categorical factors
        """
        if self.is_continuous() or self.is_discrete_numeric():
            return max(self.levels) if self.levels else None
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert factor to dictionary representation.
        
        Returns
        -------
        dict
            Factor definition as dictionary
        """
        return {
            'name': self.name,
            'type': self.factor_type.value,
            'changeability': self.changeability.value,
            'levels': self.levels,
            'units': self.units,
            'n_levels': self.get_n_levels()
        }
    
    def __repr__(self) -> str:
        """String representation of factor."""
        levels_str = f"{self.levels[0]}-{self.levels[1]}" if self.is_continuous() else str(self.levels)
        units_str = f" {self.units}" if self.units else ""
        return (
            f"Factor('{self.name}', {self.factor_type.value}, "
            f"{self.changeability.value}, levels={levels_str}{units_str})"
        )