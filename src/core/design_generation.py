"""
Design generation algorithms for DOE-Toolkit.

This module contains functions to generate various experimental designs including
full factorial, fractional factorial, and response surface designs.
"""

import itertools
from typing import List, Dict, Optional, Union
import pandas as pd
import numpy as np

from src.core.factors import Factor, FactorType


def full_factorial(
    factors: List[Factor],
    n_center_points: int = 0,
    randomize: bool = True,
    random_seed: Optional[int] = None,
    n_blocks: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate a full factorial design.
    
    A full factorial design includes all possible combinations of factor levels.
    For k factors with levels L₁, L₂, ..., Lₖ, the total number of runs is
    L₁ × L₂ × ... × Lₖ.
    
    Parameters
    ----------
    factors : List[Factor]
        List of factors to include in the design
    n_center_points : int, optional
        Number of center point runs to add (only for continuous factors), default 0
    randomize : bool, optional
        Whether to randomize run order, default True
    random_seed : int, optional
        Random seed for reproducibility, default None
    n_blocks : int, optional
        Number of blocks to divide runs into, default None (no blocking)
    
    Returns
    -------
    pd.DataFrame
        Design matrix with columns for each factor, plus 'StdOrder', 'RunOrder',
        and 'Block' (if n_blocks specified)
    
    Notes
    -----
    This implementation uses itertools.product to generate the Cartesian product
    of all factor levels. For continuous factors, the design uses coded levels
    (-1, +1) corresponding to the low and high values specified in factor.levels.
    
    Center points (coded as 0) are added for continuous factors only. They provide
    an independent estimate of pure error and allow testing for curvature.
    
    Examples
    --------
    >>> from src.core.factors import Factor, FactorType, ChangeabilityLevel
    >>> 
    >>> # Define factors
    >>> temp = Factor("Temperature", FactorType.CONTINUOUS, 
    ...               ChangeabilityLevel.EASY, levels=[150, 200])
    >>> pressure = Factor("Pressure", FactorType.CONTINUOUS,
    ...                   ChangeabilityLevel.EASY, levels=[50, 100])
    >>> 
    >>> # Generate 2^2 factorial with 3 center points
    >>> design = full_factorial([temp, pressure], n_center_points=3)
    >>> print(design)
       StdOrder  RunOrder  Temperature  Pressure
    0         1         2         -1.0      -1.0
    1         2         4          1.0      -1.0
    2         3         1         -1.0       1.0
    3         4         3          1.0       1.0
    4         5         5          0.0       0.0
    5         6         7          0.0       0.0
    6         7         6          0.0       0.0
    
    References
    ----------
    .. [1] Box, G. E. P., Hunter, J. S., and Hunter, W. G. (2005).
           Statistics for Experimenters, 2nd Ed. Wiley.
    .. [2] Montgomery, D. C. (2017). Design and Analysis of Experiments, 9th Ed.
           Wiley.
    """
    if not factors:
        raise ValueError("At least one factor must be provided")
    
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate factorial points
    design_matrix = _generate_factorial_points(factors)
    
    # Add center points if requested
    if n_center_points > 0:
        center_points = _generate_center_points(factors, n_center_points)
        design_matrix = pd.concat([design_matrix, center_points], ignore_index=True)
    
    # Add standard order (before randomization)
    design_matrix.insert(0, 'StdOrder', range(1, len(design_matrix) + 1))
    
    # Randomize if requested
    if randomize:
        design_matrix = design_matrix.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    # Add run order (after randomization)
    design_matrix.insert(1, 'RunOrder', range(1, len(design_matrix) + 1))
    
    # Add blocking if requested
    if n_blocks is not None:
        if n_blocks < 1:
            raise ValueError("Number of blocks must be at least 1")
        if n_blocks > len(design_matrix):
            raise ValueError(f"Number of blocks ({n_blocks}) cannot exceed number of runs ({len(design_matrix)})")
        
        design_matrix = _assign_blocks(design_matrix, n_blocks, randomize, random_seed)
    
    return design_matrix


def _generate_factorial_points(factors: List[Factor]) -> pd.DataFrame:
    """
    Generate factorial design points (all combinations of factor levels).
    
    Parameters
    ----------
    factors : List[Factor]
        List of factors
    
    Returns
    -------
    pd.DataFrame
        Design matrix with one column per factor
    """
    factor_levels = []
    factor_names = []
    
    for factor in factors:
        factor_names.append(factor.name)
        
        if factor.is_continuous():
            # For continuous factors, use coded levels: -1 (low), +1 (high)
            factor_levels.append([-1, 1])
        elif factor.is_discrete_numeric():
            # For discrete numeric, use actual values
            factor_levels.append(factor.levels)
        elif factor.is_categorical():
            # For categorical, use the level labels
            factor_levels.append(factor.levels)
    
    # Generate all combinations using Cartesian product
    combinations = list(itertools.product(*factor_levels))
    
    # Create DataFrame
    design = pd.DataFrame(combinations, columns=factor_names)
    
    return design


def _generate_center_points(factors: List[Factor], n_center_points: int) -> pd.DataFrame:
    """
    Generate center point runs.
    
    Center points are only meaningful for continuous factors (coded as 0).
    For discrete numeric and categorical factors, we use the middle level
    if odd number of levels, otherwise skip.
    
    Parameters
    ----------
    factors : List[Factor]
        List of factors
    n_center_points : int
        Number of center point replicates
    
    Returns
    -------
    pd.DataFrame
        Center point runs
    """
    center_values = []
    factor_names = []
    
    has_continuous = False
    
    for factor in factors:
        factor_names.append(factor.name)
        
        if factor.is_continuous():
            # Continuous: center is coded as 0
            center_values.append(0)
            has_continuous = True
        elif factor.is_discrete_numeric():
            # Discrete numeric: use middle level if possible
            n_levels = len(factor.levels)
            if n_levels % 2 == 1:  # Odd number of levels
                middle_idx = n_levels // 2
                center_values.append(factor.levels[middle_idx])
            else:
                # Even number of levels - use lower middle
                middle_idx = n_levels // 2 - 1
                center_values.append(factor.levels[middle_idx])
        elif factor.is_categorical():
            # Categorical: use first level (arbitrary choice for center point)
            center_values.append(factor.levels[0])
    
    if not has_continuous:
        # No continuous factors - center points don't make sense
        # Return empty DataFrame
        return pd.DataFrame(columns=factor_names)
    
    # Replicate center point n_center_points times
    center_points = pd.DataFrame([center_values] * n_center_points, columns=factor_names)
    
    return center_points


def _assign_blocks(
    design: pd.DataFrame,
    n_blocks: int,
    randomize_within_blocks: bool,
    random_seed: Optional[int]
) -> pd.DataFrame:
    """
    Assign runs to blocks.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix
    n_blocks : int
        Number of blocks
    randomize_within_blocks : bool
        Whether runs were randomized (affects block assignment strategy)
    random_seed : int, optional
        Random seed
    
    Returns
    -------
    pd.DataFrame
        Design matrix with 'Block' column added
    """
    n_runs = len(design)
    
    # Calculate runs per block (as evenly as possible)
    base_runs_per_block = n_runs // n_blocks
    extra_runs = n_runs % n_blocks
    
    # Create block assignments
    blocks = []
    for block_num in range(1, n_blocks + 1):
        # First 'extra_runs' blocks get one extra run
        runs_in_this_block = base_runs_per_block + (1 if block_num <= extra_runs else 0)
        blocks.extend([block_num] * runs_in_this_block)
    
    # If randomized, blocks are already mixed - just assign sequentially
    # If not randomized, we want to distribute runs across blocks
    if not randomize_within_blocks:
        # Interleave blocks: run 1 -> block 1, run 2 -> block 2, etc.
        blocks = [((i % n_blocks) + 1) for i in range(n_runs)]
    
    design.insert(2, 'Block', blocks)
    
    return design


def decode_design(
    design: pd.DataFrame,
    factors: List[Factor]
) -> pd.DataFrame:
    """
    Decode a design matrix from coded levels to actual values.
    
    For continuous factors, converts coded levels (-1, 0, +1) back to actual
    values specified in factor.levels.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with coded levels
    factors : List[Factor]
        List of factors with level definitions
    
    Returns
    -------
    pd.DataFrame
        Design matrix with actual (decoded) values
    
    Examples
    --------
    >>> # Design has Temperature coded as -1, 0, +1
    >>> # Factor specifies levels=[150, 200]
    >>> decoded = decode_design(design, [temp_factor])
    >>> # Temperature now shows 150, 175, 200
    """
    decoded = design.copy()
    
    factor_dict = {f.name: f for f in factors}
    
    for col in decoded.columns:
        if col in factor_dict:
            factor = factor_dict[col]
            
            if factor.is_continuous():
                # Decode: -1 -> low, 0 -> middle, +1 -> high
                low, high = factor.levels
                middle = (low + high) / 2
                
                # Map coded to actual
                decoded[col] = decoded[col].map({
                    -1: low,
                    0: middle,
                    1: high
                })
            # Discrete numeric and categorical already have actual values
    
    return decoded


def get_design_summary(design: pd.DataFrame, factors: List[Factor]) -> Dict[str, Union[int, str]]:
    """
    Get summary statistics for a design.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix
    factors : List[Factor]
        List of factors
    
    Returns
    -------
    dict
        Summary information including number of runs, factors, blocks, etc.
    """
    summary = {
        'n_runs': len(design),
        'n_factors': len(factors),
        'n_continuous': sum(1 for f in factors if f.is_continuous()),
        'n_discrete': sum(1 for f in factors if f.is_discrete_numeric()),
        'n_categorical': sum(1 for f in factors if f.is_categorical()),
        'has_center_points': (design[factors[0].name] == 0).any() if factors[0].is_continuous() else False,
        'n_blocks': design['Block'].nunique() if 'Block' in design.columns else 1,
        'randomized': 'RunOrder' in design.columns
    }
    
    return summary