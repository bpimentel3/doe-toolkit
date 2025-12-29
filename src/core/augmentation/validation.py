"""
Validation Utilities for Augmented Designs.

This module provides validation functions to check augmented design quality.
"""

from typing import List
import numpy as np
import pandas as pd

from src.core.factors import Factor
from src.core.augmentation.plan import AugmentedDesign, ValidationResult


def validate_augmented_design(
    augmented: AugmentedDesign,
    factors: List[Factor],
    model_terms: List[str]
) -> ValidationResult:
    """
    Comprehensive validation of augmented design.
    
    Checks:
    - Run count correctness
    - Phase/Block structure
    - Model matrix properties
    - Duplicates
    - Condition number
    
    Parameters
    ----------
    augmented : AugmentedDesign
        Augmented design to validate
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Model terms for analysis
    
    Returns
    -------
    ValidationResult
        Validation result with errors/warnings
    
    Examples
    --------
    >>> result = validate_augmented_design(augmented, factors, model_terms)
    >>> if not result.is_valid:
    ...     print("Errors:", result.errors)
    >>> for warning in result.warnings:
    ...     print(f"Warning: {warning}")
    """
    errors = []
    warnings = []
    metrics = {}
    
    # Check 1: Run count
    expected_total = augmented.n_runs_original + augmented.n_runs_added
    actual_total = len(augmented.combined_design)
    
    if actual_total != expected_total:
        errors.append(
            f"Run count mismatch: expected {expected_total}, got {actual_total}"
        )
    
    metrics['n_runs_total'] = actual_total
    
    # Check 2: Phase column exists and is valid
    if augmented.block_column not in augmented.combined_design.columns:
        errors.append(f"Block column '{augmented.block_column}' missing")
    else:
        phases = augmented.combined_design[augmented.block_column].unique()
        if not all(p in [1, 2] for p in phases):
            warnings.append(f"Unexpected phase values: {phases}")
    
    # Check 3: Model matrix properties
    from src.core.diagnostics.variance import build_model_matrix
    from src.core.diagnostics.estimability import compute_condition_number
    
    factor_names = [f.name for f in factors]
    
    try:
        # Build model matrix
        X = build_model_matrix(
            augmented.combined_design[factor_names],
            factors,
            model_terms
        )
        
        n, p = X.shape
        metrics['n_parameters'] = p
        metrics['df_error'] = n - p
        
        # Check rank
        rank = np.linalg.matrix_rank(X)
        metrics['rank'] = rank
        
        if rank < p:
            errors.append(
                f"Model matrix is rank-deficient: rank={rank}, expected {p}"
            )
        
        # Condition number
        kappa = compute_condition_number(
            augmented.combined_design[factor_names],
            factors,
            model_terms
        )
        metrics['condition_number'] = kappa
        
        if kappa > 1000:
            errors.append(f"Severely ill-conditioned: κ = {kappa:.1e}")
        elif kappa > 100:
            warnings.append(f"High condition number: κ = {kappa:.1f}")
    
    except Exception as e:
        errors.append(f"Model matrix construction failed: {e}")
    
    # Check 4: Duplicates
    factor_cols = [f.name for f in factors if f.name in augmented.combined_design.columns]
    
    if len(factor_cols) > 0:
        design_points = augmented.combined_design[factor_cols].round(6)
        n_duplicates = design_points.duplicated().sum()
        
        if n_duplicates > 0:
            warnings.append(f"Found {n_duplicates} duplicate run(s)")
        
        metrics['n_duplicates'] = n_duplicates
    
    # Check 5: Balance
    if augmented.block_column in augmented.combined_design.columns:
        phase_counts = augmented.combined_design[augmented.block_column].value_counts()
        
        if len(phase_counts) > 1:
            imbalance = phase_counts.max() - phase_counts.min()
            metrics['phase_imbalance'] = imbalance
            
            if imbalance > len(augmented.combined_design) * 0.2:
                warnings.append(
                    f"Unbalanced phases: {dict(phase_counts)}"
                )
    
    is_valid = len(errors) == 0
    
    return ValidationResult(
        is_valid=is_valid,
        errors=errors,
        warnings=warnings,
        metrics=metrics
    )


def check_resolution_improvement(
    original_resolution: int,
    new_resolution: int,
    foldover_type: str
) -> bool:
    """
    Verify that foldover achieved expected resolution improvement.
    
    Parameters
    ----------
    original_resolution : int
        Original design resolution
    new_resolution : int
        New design resolution
    foldover_type : str
        'full' or 'single_factor'
    
    Returns
    -------
    bool
        Whether improvement matches expectation
    """
    if foldover_type == 'full':
        # Full foldover should increase resolution by at least 1
        return new_resolution >= original_resolution + 1
    else:
        # Single-factor foldover doesn't necessarily change overall resolution
        return True


def check_combined_design_orthogonality(
    design: pd.DataFrame,
    factors: List[Factor],
    tolerance: float = 0.1
) -> bool:
    """
    Check if combined design maintains reasonable orthogonality.
    
    Parameters
    ----------
    design : pd.DataFrame
        Combined design
    factors : List[Factor]
        Factor definitions
    tolerance : float
        Max acceptable correlation
    
    Returns
    -------
    bool
        Whether design is acceptably orthogonal
    """
    factor_names = [f.name for f in factors]
    X = design[factor_names].values
    
    # Compute correlation matrix
    if X.shape[1] > 1:
        corr = np.corrcoef(X.T)
        
        # Check off-diagonal elements
        np.fill_diagonal(corr, 0)
        max_corr = np.max(np.abs(corr))
        
        return max_corr < tolerance
    
    return True