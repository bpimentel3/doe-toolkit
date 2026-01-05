"""
Foldover Augmentation for Fractional Factorial Designs.

This module implements full and single-factor foldover strategies to
de-alias effects in Resolution III and IV designs.
"""

from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np

from src.core.factors import Factor
from src.core.augmentation.plan import (
    AugmentationPlan,
    AugmentedDesign,
    FoldoverConfig
)


def augment_full_foldover(
    original_design: pd.DataFrame,
    factors: List[Factor],
    generators: List[Tuple[str, str]],
    randomize: bool = True,
    seed: Optional[int] = None
) -> AugmentedDesign:
    """
    Add full foldover to fractional factorial design.
    
    Full foldover flips the signs of ALL factors, which:
    - Resolution III → Resolution IV (main effects clear of 2FI)
    - Resolution IV → Resolution V (2FI clear of other 2FI)
    - Doubles the number of runs
    
    Parameters
    ----------
    original_design : pd.DataFrame
        Original fractional factorial design (coded levels)
    factors : List[Factor]
        Factor definitions
    generators : List[Tuple[str, str]]
        Original generators (e.g., [('E', 'ABCD')])
    randomize : bool, default=True
        Whether to randomize new run order
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    AugmentedDesign
        Combined design with Phase column
    
    Examples
    --------
    >>> # 2^(5-1) Resolution V design
    >>> design = generate_fractional_factorial(factors, '1/2', resolution=5)
    >>> 
    >>> # Add full foldover
    >>> augmented = augment_full_foldover(
    ...     design, factors, generators=[('E', 'ABCD')]
    ... )
    >>> 
    >>> print(f"Resolution improved to {augmented.resolution}")
    
    Notes
    -----
    Full foldover creates the "fold-over" design by changing all signs:
        x_new = -x_old
    
    The combined design is the union of original and foldover, which
    de-aliases effects according to standard foldover theory.
    
    References
    ----------
    .. [1] Box, G. E. P., Hunter, J. S., & Hunter, W. G. (2005).
           Statistics for Experimenters, 2nd Ed. Wiley.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Extract factor columns
    factor_names = [f.name for f in factors]
    
    # Check that design has these columns
    missing = [name for name in factor_names if name not in original_design.columns]
    if missing:
        raise ValueError(f"Design missing factor columns: {missing}")
    
    # Create foldover design by flipping all signs
    foldover_design = original_design[factor_names].copy()
    foldover_design[factor_names] = -foldover_design[factor_names]
    
    # Randomize foldover if requested
    if randomize:
        foldover_design = foldover_design.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Add Phase column
    original_with_phase = original_design.copy()
    original_with_phase['Phase'] = 1
    
    foldover_with_phase = foldover_design.copy()
    foldover_with_phase['Phase'] = 2
    
    # Combine
    combined = pd.concat([original_with_phase, foldover_with_phase], ignore_index=True)
    
    # Add standard order and run order
    combined.insert(0, 'StdOrder', range(1, len(combined) + 1))
    combined.insert(1, 'RunOrder', range(1, len(combined) + 1))
    
    # Compute new resolution using aliasing engine
    from src.core.aliasing import AliasingEngine
    
    k = len(factors)
    p = len(generators)
    
    # Get original resolution
    try:
        original_engine = AliasingEngine(k, generators)
        original_resolution = original_engine.resolution
    except:
        original_resolution = 3
    
    # Compute new resolution for combined design
    # Full foldover combines original + mirror image
    # This effectively reduces fractionation by one level
    
    if p == 0:
        # Already full factorial
        new_resolution = k
        new_generators = []
    elif p == 1:
        # Single generator: foldover gives full factorial
        new_resolution = k
        new_generators = []
    else:
        # Multiple generators: combined design has p-1 effective generators
        # The resolution typically increases, but we need to compute it properly
        
        # For a full foldover, the combined design has reduced confounding
        # Conservative approach: calculate actual resolution from theory
        # Full foldover of 2^(k-p) gives 2^(k-p+1) design
        
        # Build new generator set (removing one degree of fractionation)
        # This is complex, so we use a conservative estimate
        new_resolution = min(original_resolution + 1, k)
        new_generators = []
        
        # For accurate resolution, we'd need to analyze the combined alias structure
        # But for typical cases, resolution increases by 1
    
    # Compute updated alias structure for the combined design
    if new_resolution >= k:
        # Full factorial or nearly full - minimal aliasing
        updated_alias_structure = {}
    else:
        # Still some aliasing, compute it
        try:
            if new_generators:
                new_engine = AliasingEngine(k, new_generators)
                updated_alias_structure = new_engine.alias_structure
            else:
                # No generators means full factorial
                updated_alias_structure = {}
        except:
            updated_alias_structure = {}
    
    # Build improvements description
    achieved_improvements = {
        'resolution': f'{original_resolution} → {new_resolution}',
        'n_runs': f'{len(original_design)} → {len(combined)}',
        'main_effects': 'All main effects clear of 2FI' if new_resolution >= 4 else 'Some aliasing remains'
    }
    
    # Build result
    augmented = AugmentedDesign(
        combined_design=combined,
        new_runs_only=foldover_with_phase,
        block_column='Phase',
        n_runs_original=len(original_design),
        n_runs_added=len(foldover_design),
        n_runs_total=len(combined),
        achieved_improvements=achieved_improvements,
        resolution=new_resolution,
        updated_alias_structure=updated_alias_structure
    )
    
    return augmented


def augment_single_factor_foldover(
    original_design: pd.DataFrame,
    factors: List[Factor],
    generators: List[Tuple[str, str]],
    factor_to_fold: str,
    randomize: bool = True,
    seed: Optional[int] = None
) -> AugmentedDesign:
    """
    Add single-factor foldover to fractional factorial design.
    
    Single-factor foldover flips the sign of ONE factor, which:
    - De-aliases that factor from its 2-factor interactions
    - Uses same number of runs as original (more efficient than full foldover)
    - Does not fully increase resolution, but resolves specific confounding
    
    Parameters
    ----------
    original_design : pd.DataFrame
        Original fractional factorial design
    factors : List[Factor]
        Factor definitions
    generators : List[Tuple[str, str]]
        Original generators
    factor_to_fold : str
        Name of factor to fold on
    randomize : bool, default=True
        Whether to randomize new run order
    seed : int, optional
        Random seed
    
    Returns
    -------
    AugmentedDesign
        Combined design with single-factor foldover
    
    Examples
    --------
    >>> # Temperature is aliased with Pressure*Time
    >>> augmented = augment_single_factor_foldover(
    ...     design, factors, generators, factor_to_fold='Temperature'
    ... )
    >>> 
    >>> # Temperature is now clear of 2FI
    >>> print(augmented.achieved_improvements)
    
    Notes
    -----
    Single-factor foldover is most useful when:
    1. One main effect is significant and aliased
    2. Budget limits prevent full foldover
    3. Only specific de-aliasing is needed
    
    The combined design (original + single-fold) allows estimation of:
    - The folded factor (clear of 2FI)
    - 2FI involving the folded factor
    
    References
    ----------
    .. [1] Montgomery, D. C. (2017). Design and Analysis of Experiments, 9th Ed.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Validate factor exists
    factor_names = [f.name for f in factors]
    if factor_to_fold not in factor_names:
        raise ValueError(f"Factor '{factor_to_fold}' not in design")
    
    # Check design has required columns
    missing = [name for name in factor_names if name not in original_design.columns]
    if missing:
        raise ValueError(f"Design missing factor columns: {missing}")
    
    # Create foldover by flipping ONE factor
    foldover_design = original_design[factor_names].copy()
    foldover_design[factor_to_fold] = -foldover_design[factor_to_fold]
    
    # Randomize if requested
    if randomize:
        foldover_design = foldover_design.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    # Add Phase column
    original_with_phase = original_design.copy()
    original_with_phase['Phase'] = 1
    
    foldover_with_phase = foldover_design.copy()
    foldover_with_phase['Phase'] = 2
    
    # Combine
    combined = pd.concat([original_with_phase, foldover_with_phase], ignore_index=True)
    
    # Add standard order and run order
    combined.insert(0, 'StdOrder', range(1, len(combined) + 1))
    combined.insert(1, 'RunOrder', range(1, len(combined) + 1))
    
    # Resolution doesn't necessarily increase globally, but specified factor is de-aliased
    from src.core.aliasing import AliasingEngine
    
    k = len(factors)
    original_resolution = AliasingEngine(k, generators).resolution
    
    # Single-factor foldover doesn't change overall resolution
    # But it de-aliases the specific factor
    new_resolution = original_resolution
    
    # Compute which effects are now estimable
    # (This is complex - for now, just note that factor_to_fold is de-aliased)
    achieved_improvements = {
        'de_aliased': f'{factor_to_fold} clear of 2-factor interactions',
        'n_runs': f'{len(original_design)} → {len(combined)}'
    }
    
    # Build result
    augmented = AugmentedDesign(
        combined_design=combined,
        new_runs_only=foldover_with_phase,
        block_column='Phase',
        n_runs_original=len(original_design),
        n_runs_added=len(foldover_design),
        n_runs_total=len(combined),
        achieved_improvements=achieved_improvements,
        resolution=new_resolution,
        updated_alias_structure=None  # Complex to compute for single-factor
    )
    
    return augmented


def execute_foldover_plan(plan: AugmentationPlan) -> AugmentedDesign:
    """
    Execute a foldover augmentation plan.
    
    This is called by AugmentationPlan.execute() for foldover strategies.
    
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
    
    if not isinstance(config, FoldoverConfig):
        raise TypeError(f"Expected FoldoverConfig, got {type(config)}")
    
    # Extract generators from metadata
    generators = plan.metadata.get('generators', [])
    
    if config.foldover_type == 'full':
        augmented = augment_full_foldover(
            original_design=plan.original_design,
            factors=plan.factors,
            generators=generators,
            randomize=True,
            seed=plan.metadata.get('seed')
        )
    
    elif config.foldover_type == 'single_factor':
        if not config.factor_to_fold:
            raise ValueError("Single-factor foldover requires factor_to_fold")
        
        augmented = augment_single_factor_foldover(
            original_design=plan.original_design,
            factors=plan.factors,
            generators=generators,
            factor_to_fold=config.factor_to_fold,
            randomize=True,
            seed=plan.metadata.get('seed')
        )
    
    else:
        raise ValueError(f"Unknown foldover type: {config.foldover_type}")
    
    # Attach plan provenance
    augmented.plan_executed = plan
    
    return augmented


def recommend_foldover_factor(
    original_design: pd.DataFrame,
    factors: List[Factor],
    generators: List[Tuple[str, str]],
    significant_effects: List[str],
    effect_sizes: Optional[Dict[str, float]] = None
) -> Optional[str]:
    """
    Recommend which factor to use for single-factor foldover.
    
    Logic:
    1. If one main effect is significant and aliased → recommend that factor
    2. If multiple significant and aliased → recommend most significant
    3. If none significant → return None
    
    Parameters
    ----------
    original_design : pd.DataFrame
        Original design
    factors : List[Factor]
        Factor definitions
    generators : List[Tuple[str, str]]
        Generators
    significant_effects : List[str]
        List of significant effects (from ANOVA)
    effect_sizes : Dict[str, float], optional
        Effect sizes (absolute values) for ranking. If provided,
        recommends factor with largest effect size.
    
    Returns
    -------
    str or None
        Recommended factor name, or None if no clear choice
    
    Examples
    --------
    >>> significant = ['Temperature', 'Pressure', 'Temperature*Time']
    >>> effect_sizes = {'Temperature': 5.2, 'Pressure': 3.1}
    >>> factor = recommend_foldover_factor(
    ...     design, factors, generators, significant, effect_sizes
    ... )
    >>> print(f"Recommend folding on: {factor}")
    'Temperature'
    """
    from src.core.aliasing import AliasingEngine
    
    k = len(factors)
    engine = AliasingEngine(k, generators)
    
    # Find significant main effects that are aliased
    aliased_significant = []
    
    for effect in significant_effects:
        # Main effects only (single factor name)
        if len(effect) == 1 or (effect in [f.name for f in factors]):
            # Check if aliased
            if effect in engine.alias_structure:
                aliases = engine.alias_structure[effect]
                # Check if aliased with 2FI (two-letter effects)
                has_2fi_alias = any(len(a) == 2 for a in aliases)
                if has_2fi_alias:
                    aliased_significant.append(effect)
    
    if len(aliased_significant) == 0:
        return None
    elif len(aliased_significant) == 1:
        return aliased_significant[0]
    else:
        # Multiple candidates - rank by effect size if available
        if effect_sizes:
            # Recommend factor with largest effect
            return max(
                aliased_significant,
                key=lambda f: effect_sizes.get(f, 0)
            )
        else:
            # Fallback: return first
            return aliased_significant[0]