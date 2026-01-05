"""
D-Optimal Augmentation for Model Extension.

This module implements D-optimal augmentation to add runs that improve
model precision or enable estimation of additional model terms.
"""

from typing import List, Optional
import pandas as pd
import numpy as np

from src.core.factors import Factor
from src.core.augmentation.plan import (
    AugmentationPlan,
    AugmentedDesign,
    OptimalAugmentConfig
)
from src.core.candidates.generators import (
    generate_augmentation_candidates,
    CandidatePoolConfig
)


def augment_for_model_extension(
    original_design: pd.DataFrame,
    factors: List[Factor],
    current_model_terms: List[str],
    new_model_terms: List[str],
    n_runs_to_add: int,
    seed: Optional[int] = None
) -> AugmentedDesign:
    """
    Add runs optimized for extended model.
    
    Use case: Model needs additional terms (e.g., linear → quadratic).
    Strategy: Fix existing runs, optimize new runs to maximize D-efficiency
    of extended model.
    
    Parameters
    ----------
    original_design : pd.DataFrame
        Original design matrix (coded levels)
    factors : List[Factor]
        Factor definitions
    current_model_terms : List[str]
        Terms in current model
    new_model_terms : List[str]
        Complete set of terms for extended model (includes current)
    n_runs_to_add : int
        Number of runs to add
    seed : int, optional
        Random seed
    
    Returns
    -------
    AugmentedDesign
        Combined design optimized for extended model
    
    Examples
    --------
    >>> # Original model: linear
    >>> current_terms = ['1', 'A', 'B', 'C']
    >>> 
    >>> # Extended model: quadratic
    >>> new_terms = ['1', 'A', 'B', 'C', 'A*B', 'A*C', 'B*C', 
    ...              'I(A**2)', 'I(B**2)', 'I(C**2)']
    >>> 
    >>> augmented = augment_for_model_extension(
    ...     design, factors, current_terms, new_terms, n_runs_to_add=10
    ... )
    
    Notes
    -----
    The algorithm:
    1. Build model matrix for existing runs with new terms
    2. Generate candidate pool (excluding existing runs)
    3. Use coordinate exchange to select new runs maximizing det([X_old; X_new])
    4. Combine original and new runs
    
    This reuses the CEXCH optimizer from optimal_design.py but with
    existing runs fixed.
    """
    if seed is not None:
        np.random.seed(seed)
    
    from src.core.diagnostics.variance import build_model_matrix
    
    # Extract factor columns
    factor_names = [f.name for f in factors]
    X_original = original_design[factor_names].values
    
    # Build model matrix for original runs with new terms
    X_model_original = build_model_matrix(
        original_design[factor_names], factors, new_model_terms
    )
    
    n_original = len(original_design)
    p = len(new_model_terms)
    
    # Check that augmented design will have enough runs
    if n_original + n_runs_to_add < p:
        raise ValueError(
            f"Augmented design will be supersaturated: "
            f"{n_original + n_runs_to_add} runs for {p} parameters"
        )
    
    # Generate candidate pool (excluding existing runs)
    config = CandidatePoolConfig(
        include_vertices=True,
        include_axial=True,
        include_center=True,
        alpha_axial=1.0,
        lhs_multiplier=5,
        exclude_existing_runs=True,
        min_distance=0.01
    )
    
    candidates = generate_augmentation_candidates(
        factors=factors,
        original_design=original_design[factor_names],
        n_candidates=max(n_runs_to_add * 10, 200),
        seed=seed
    )
    
    if len(candidates) < n_runs_to_add:
        raise ValueError(
            f"Insufficient candidates: found {len(candidates)}, "
            f"need {n_runs_to_add}"
        )
    
    # Select best new runs using augmented coordinate exchange
    new_run_indices = _augmented_coordinate_exchange(
        X_model_original=X_model_original,
        candidates=candidates,
        factors=factors,
        model_terms=new_model_terms,
        n_runs_to_add=n_runs_to_add,
        seed=seed
    )
    
    new_runs_coded = candidates[new_run_indices]
    
    # Create new runs DataFrame
    new_runs = pd.DataFrame(new_runs_coded, columns=factor_names)
    new_runs['Phase'] = 2
    
    # Combine original and new
    original_with_phase = original_design.copy()
    original_with_phase['Phase'] = 1
    
    combined = pd.concat([original_with_phase, new_runs], ignore_index=True)
    
    # Add standard order and run order
    combined.insert(0, 'StdOrder', range(1, len(combined) + 1))
    combined.insert(1, 'RunOrder', range(1, len(combined) + 1))
    
    # Compute D-efficiency
    X_model_combined = build_model_matrix(
        combined[factor_names], factors, new_model_terms
    )
    
    XtX = X_model_combined.T @ X_model_combined
    try:
        det_combined = np.linalg.det(XtX)
        d_efficiency = 100.0  # Relative to itself (could compare to ideal)
    except:
        det_combined = 0.0
        d_efficiency = 0.0
    
    # Compute condition number
    try:
        condition_number = np.linalg.cond(XtX)
    except:
        condition_number = np.inf
    
    # Build result
    augmented = AugmentedDesign(
        combined_design=combined,
        new_runs_only=new_runs,
        block_column='Phase',
        n_runs_original=n_original,
        n_runs_added=n_runs_to_add,
        n_runs_total=len(combined),
        achieved_improvements={
            'model_terms': f'{len(current_model_terms)} → {len(new_model_terms)}',
            'n_runs': f'{n_original} → {len(combined)}'
        },
        d_efficiency=d_efficiency,
        condition_number=condition_number
    )
    
    return augmented


def _augmented_coordinate_exchange(
    X_model_original: np.ndarray,
    candidates: np.ndarray,
    factors: List[Factor],
    model_terms: List[str],
    n_runs_to_add: int,
    seed: Optional[int],
    max_iterations: int = 100
) -> np.ndarray:
    """
    Coordinate exchange for augmented design.
    
    This is a simplified version of the CEXCH algorithm that:
    1. Fixes existing runs (X_model_original)
    2. Optimizes selection of new runs from candidates
    3. Maximizes det([X_original; X_new])
    
    Parameters
    ----------
    X_model_original : np.ndarray
        Model matrix for existing runs
    candidates : np.ndarray
        Candidate points (coded space)
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Model terms
    n_runs_to_add : int
        Number of runs to select
    seed : int, optional
        Random seed
    max_iterations : int
        Maximum iterations
    
    Returns
    -------
    np.ndarray
        Indices of selected candidates
    """
    from src.core.diagnostics.variance import build_model_matrix
    
    rng = np.random.default_rng(seed)
    N_cand = len(candidates)
    p = X_model_original.shape[1]
    
    # Initialize with random selection
    selected_indices = rng.choice(N_cand, size=n_runs_to_add, replace=False)
    
    # Build model matrix for candidates
    candidates_df = pd.DataFrame(candidates, columns=[f.name for f in factors])
    X_model_candidates = build_model_matrix(candidates_df, factors, model_terms)
    
    # Compute initial X'X
    X_current = X_model_candidates[selected_indices]
    X_combined = np.vstack([X_model_original, X_current])
    XtX = X_combined.T @ X_combined
    
    try:
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(p))
        _, logdet = np.linalg.slogdet(XtX)
    except:
        return selected_indices
    
    # Coordinate exchange
    for iteration in range(max_iterations):
        improved = False
        
        for i in range(n_runs_to_add):
            current_idx = selected_indices[i]
            best_idx = current_idx
            best_logdet = logdet
            
            # Try swapping with candidates
            candidate_subset = rng.choice(N_cand, size=min(50, N_cand), replace=False)
            
            for cand_idx in candidate_subset:
                if cand_idx in selected_indices:
                    continue
                
                # Swap
                trial_indices = selected_indices.copy()
                trial_indices[i] = cand_idx
                
                X_trial = X_model_candidates[trial_indices]
                X_trial_combined = np.vstack([X_model_original, X_trial])
                XtX_trial = X_trial_combined.T @ X_trial_combined
                
                try:
                    sign, logdet_trial = np.linalg.slogdet(XtX_trial)
                    if sign > 0 and logdet_trial > best_logdet:
                        best_logdet = logdet_trial
                        best_idx = cand_idx
                        improved = True
                except:
                    continue
            
            # Apply best swap
            if best_idx != current_idx:
                selected_indices[i] = best_idx
                X_current = X_model_candidates[selected_indices]
                X_combined = np.vstack([X_model_original, X_current])
                XtX = X_combined.T @ X_combined
                logdet = best_logdet
        
        if not improved:
            break
    
    return selected_indices


def execute_optimal_plan(plan: AugmentationPlan) -> AugmentedDesign:
    """
    Execute D-optimal augmentation plan.
    
    Called by AugmentationPlan.execute() for d_optimal strategies.
    
    Parameters
    ----------
    plan : AugmentationPlan
        Plan to execute
    
    Returns
    -------
    AugmentedDesign
        Result of execution
    """
    config = plan.strategy_config
    
    if not isinstance(config, OptimalAugmentConfig):
        raise TypeError(f"Expected OptimalAugmentConfig, got {type(config)}")
    
    # Validate required metadata
    if 'current_model_terms' not in plan.metadata:
        raise ValueError(
            "Plan metadata missing 'current_model_terms'. "
            "Cannot perform model extension without current model specification."
        )
    
    current_model_terms = plan.metadata['current_model_terms']
    
    if not current_model_terms:
        raise ValueError("current_model_terms cannot be empty")

    #####
    
    augmented = augment_for_model_extension(
        original_design=plan.original_design,
        factors=plan.factors,
        current_model_terms=current_model_terms,
        new_model_terms=config.new_model_terms,
        n_runs_to_add=config.n_runs_to_add,
        seed=plan.metadata.get('seed')
    )
    
    # Attach plan provenance
    augmented.plan_executed = plan
    
    return augmented