"""
Factor definitions for DOE-Toolkit.

This module handles the definition and validation of experimental factors
including continuous, discrete numeric, and categorical types.
"""

from enum import Enum
from typing import List, Optional, Union, Dict, Any, Tuple, Set
from dataclasses import dataclass, field
import numpy as np

import re
import keyword


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
    _validate_on_init: bool = field(default=True, repr=False, compare=False)
    
    def __post_init__(self):
        """Validate factor definition after initialization."""
        if self._validate_on_init:
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
    def min_value(self) -> Optional[float]:
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
    def max_value(self) -> Optional[float]:
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
    
    """
Factor Name Sanitization for Patsy Formula Compatibility.

This module provides functions to sanitize factor names to ensure they are
safe for use in patsy formulas and Python expressions.

References
----------
.. [1] Patsy formula syntax: https://patsy.readthedocs.io/en/latest/formulas.html
"""



# Patsy reserved words that have special meaning
PATSY_RESERVED: Set[str] = {
    'I',     # Identity function (prevents interpretation)
    'C',     # Categorical encoding
    'Q',     # Quote function (backtick alternative)
    'T',     # Treatment encoding (deprecated but still reserved)
}

# Common numpy/pandas names that might cause confusion
COMMON_RESERVED: Set[str] = {
    'np', 'pd', 'df', 'pd', 'sm',  # Module abbreviations
}

# Combine all reserved words
ALL_RESERVED: Set[str] = PATSY_RESERVED | COMMON_RESERVED


def sanitize_factor_name(name: str) -> Tuple[str, bool]:
    """
    Sanitize factor name for patsy formula compatibility.
    
    This function ensures factor names are safe for use in patsy formulas by:
    1. Removing/replacing unsafe characters
    2. Ensuring valid Python identifier format
    3. Avoiding Python keywords and patsy reserved words
    
    Parameters
    ----------
    name : str
        Raw factor name from user input
    
    Returns
    -------
    sanitized : str
        Safe factor name that can be used in patsy formulas
    was_modified : bool
        Whether the name was changed during sanitization
    
    Examples
    --------
    >>> sanitize_factor_name("Temperature (°C)")
    ('Temperature_C', True)
    
    >>> sanitize_factor_name("Pressure*Time")
    ('Pressure_Time', True)
    
    >>> sanitize_factor_name("1X")
    ('F_1X', True)
    
    >>> sanitize_factor_name("For")
    ('For_var', True)
    
    >>> sanitize_factor_name("I()")
    ('I_factor', True)
    
    >>> sanitize_factor_name("(TEST)")
    ('TEST', True)
    
    >>> sanitize_factor_name("Temperature")
    ('Temperature', False)
    
    Notes
    -----
    Sanitization rules:
    - Only alphanumeric characters and underscores are allowed
    - Spaces are replaced with underscores
    - Must start with letter or underscore (not digit)
    - Cannot be a Python keyword (for, if, class, etc.)
    - Cannot be a patsy reserved word (I, C, Q, T)
    - Empty strings default to 'Factor_1'
    """
    original = name
    
    # Strip leading/trailing whitespace
    name = name.strip()
    
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    
    # Remove all non-alphanumeric characters except underscores
    # This handles: *, +, -, /, ^, (), [], {}, quotes, special symbols, etc.
    name = re.sub(r'[^\w]', '_', name)
    
    # Remove consecutive underscores
    name = re.sub(r'_+', '_', name)
    
    # Remove leading/trailing underscores
    name = name.strip('_')
    
    # Ensure starts with letter or underscore (not digit)
    if name and name[0].isdigit():
        name = 'F_' + name
    
    # Check for Python keywords (case-sensitive match for Python)
    # But also check lowercase for common mistakes
    if keyword.iskeyword(name) or keyword.iskeyword(name.lower()):
        name = name + '_var'
    
    # Check for patsy/common reserved words (case-sensitive)
    if name in ALL_RESERVED:
        name = name + '_factor'
    
    # Ensure not empty after all transformations
    if not name:
        name = 'Factor_1'
    
    was_modified = (name != original)
    
    return name, was_modified


def validate_factor_name(name: str) -> Tuple[bool, str]:
    """
    Validate factor name without modification.
    
    Use this to check if a name is valid before accepting it.
    
    Parameters
    ----------
    name : str
        Factor name to validate
    
    Returns
    -------
    is_valid : bool
        Whether the name is valid as-is
    reason : str
        Explanation of why invalid (empty if valid)
    
    Examples
    --------
    >>> validate_factor_name("Temperature")
    (True, '')
    
    >>> validate_factor_name("1X")
    (False, 'Factor name cannot start with a digit')
    
    >>> validate_factor_name("Pressure*Time")
    (False, 'Factor name contains invalid characters: *')
    """
    # Empty
    if not name or not name.strip():
        return False, "Factor name cannot be empty"
    
    name = name.strip()
    
    # Check for invalid characters
    invalid_chars = re.findall(r'[^\w]', name)
    if invalid_chars:
        # Filter to unique, excluding underscore and space
        unique_invalid = set(invalid_chars) - {'_', ' '}
        if unique_invalid:
            chars_str = ', '.join(sorted(unique_invalid))
            return False, f"Factor name contains invalid characters: {chars_str}"
    
    # Check for spaces (allowed but will be replaced)
    if ' ' in name:
        return False, "Factor name contains spaces (use underscores instead)"
    
    # Starts with digit
    if name[0].isdigit():
        return False, "Factor name cannot start with a digit"
    
    # Python keyword
    if keyword.iskeyword(name) or keyword.iskeyword(name.lower()):
        return False, f"'{name}' is a Python keyword"
    
    # Reserved word
    if name in ALL_RESERVED:
        return False, f"'{name}' is reserved by patsy/pandas"
    
    return True, ''


def suggest_alternative_names(name: str, existing_names: Set[str]) -> list:
    """
    Suggest alternative factor names if sanitized name conflicts.
    
    Parameters
    ----------
    name : str
        Original name
    existing_names : Set[str]
        Names already in use
    
    Returns
    -------
    List[str]
        Up to 3 alternative names
    
    Examples
    --------
    >>> suggest_alternative_names("Temp", {"Temp", "Temp_1"})
    ['Temp_2', 'Temp_v2', 'Temperature']
    """
    sanitized, _ = sanitize_factor_name(name)
    
    suggestions = []
    
    # Try numbered suffixes
    for i in range(1, 10):
        candidate = f"{sanitized}_{i}"
        if candidate not in existing_names:
            suggestions.append(candidate)
            if len(suggestions) >= 3:
                return suggestions
    
    # Try common variations
    variations = [
        f"{sanitized}_v2",
        f"{sanitized}_alt",
        f"{sanitized}_factor",
        f"Factor_{sanitized}",
    ]
    
    for var in variations:
        if var not in existing_names:
            suggestions.append(var)
            if len(suggestions) >= 3:
                return suggestions
    
    return suggestions[:3]


def get_sanitization_report(name: str) -> dict:
    """
    Get detailed report of sanitization changes.
    
    Parameters
    ----------
    name : str
        Original name
    
    Returns
    -------
    dict
        Report with keys:
        - 'original': Original name
        - 'sanitized': Sanitized name
        - 'was_modified': Whether changed
        - 'changes': List of specific changes made
    
    Examples
    --------
    >>> report = get_sanitization_report("Temp (°C)")
    >>> print(report['changes'])
    ['Removed special character: (', 'Removed special character: °', 
     'Removed special character: )']
    """
    original = name
    sanitized, was_modified = sanitize_factor_name(name)
    
    changes = []
    
    # Track specific transformations
    if not name:
        changes.append("Empty name replaced with 'Factor_1'")
    
    # Check for spaces
    if ' ' in original:
        changes.append("Spaces replaced with underscores")
    
    # Check for special characters
    special_chars = set(re.findall(r'[^\w\s]', original))
    for char in sorted(special_chars):
        changes.append(f"Removed special character: {char}")
    
    # Check for leading digit
    if original and original.strip() and original.strip()[0].isdigit():
        changes.append("Added 'F_' prefix (name started with digit)")
    
    # Check for keyword
    cleaned = re.sub(r'[^\w]', '_', original).strip('_')
    if keyword.iskeyword(cleaned) or keyword.iskeyword(cleaned.lower()):
        changes.append(f"Added '_var' suffix ('{cleaned}' is Python keyword)")
    
    # Check for reserved
    if cleaned in ALL_RESERVED:
        changes.append(f"Added '_factor' suffix ('{cleaned}' is reserved by patsy)")
    
    return {
        'original': original,
        'sanitized': sanitized,
        'was_modified': was_modified,
        'changes': changes if was_modified else []
    }