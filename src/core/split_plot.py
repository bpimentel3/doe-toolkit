"""
Split-Plot Design Generation for Design of Experiments.

This module generates split-plot designs that properly handle factors with
different levels of changeability (hard-to-change vs. easy-to-change factors).
Split-plot designs are essential when some factors are expensive or time-consuming
to change, requiring a nested experimental structure.
"""

import numpy as np
import pandas as pd
from typing import List, Literal, Optional, Tuple
from dataclasses import dataclass
from itertools import product

from src.core.factors import Factor, FactorType, ChangeabilityLevel


@dataclass
class SplitPlotDesign:
    """
    Container for split-plot design and metadata.
    
    Attributes
    ----------
    design : pd.DataFrame
        Design matrix with factor columns, plot IDs, and run orders
    design_coded : pd.DataFrame
        Design matrix with continuous factors in coded [-1, 1] scale
    n_runs : int
        Total number of experimental runs
    n_whole_plots : int
        Number of whole-plots (unique combinations of hard-to-change factors)
    n_sub_plots_per_whole_plot : int
        Number of sub-plots within each whole-plot
    whole_plot_factors : List[str]
        Names of whole-plot (hard-to-change) factors
    sub_plot_factors : List[str]
        Names of sub-plot (easy-to-change) factors
    has_very_hard_factors : bool
        Whether design includes very-hard-to-change factors
    n_blocks : int
        Number of blocks (if blocking is used)
    """
    design: pd.DataFrame
    design_coded: pd.DataFrame
    n_runs: int
    n_whole_plots: int
    n_sub_plots_per_whole_plot: int
    whole_plot_factors: List[str]
    sub_plot_factors: List[str]
    has_very_hard_factors: bool
    n_blocks: int


def generate_split_plot_design(
    factors: List[Factor],
    n_replicates: int = 1,
    n_center_points: int = 0,
    n_blocks: int = 1,
    randomize_whole_plots: bool = True,
    randomize_sub_plots: bool = True,
    seed: int = None
) -> SplitPlotDesign:
    """
    Generate split-plot design respecting factor changeability.
    
    Automatically infers plot structure from factor changeability levels:
    - VERY_HARD factors define whole-whole-plots (outermost level)
    - HARD factors define whole-plots (middle level)
    - EASY factors define sub-plots (innermost level, fully randomized)
    
    Currently supports full factorial structure only.
    
    Parameters
    ----------
    factors : List[Factor]
        List of factor definitions with changeability specified
    n_replicates : int, default=1
        Number of complete replicates of the design
    n_center_points : int, default=0
        Number of center point runs (added at sub-plot level only)
    n_blocks : int, default=1
        Number of blocks to divide whole-plots into
    randomize_whole_plots : bool, default=True
        Whether to randomize order of whole-plots
    randomize_sub_plots : bool, default=True
        Whether to randomize order of sub-plots within each whole-plot
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    SplitPlotDesign
        Object containing design matrix, plot structure, and metadata
    
    Raises
    ------
    ValueError
        If no factors provided, invalid changeability, or insufficient plots
    
    Notes
    -----
    Split-plot designs are essential when factors have different costs or
    difficulty of changing. The proper structure ensures:
    
    1. **Whole-plot factors** are held constant within a whole-plot
    2. **Sub-plot factors** vary within each whole-plot
    3. **Proper randomization** respects restrictions
    4. **Correct error structure** for subsequent ANOVA analysis
    
    The minimum requirements for a valid split-plot design:
    - At least 2 whole-plots (for whole-plot error estimation)
    - At least 2 sub-plots per whole-plot (for sub-plot error estimation)
    
    Center points, if requested, are added only at the sub-plot level since
    whole-plot factors cannot be easily changed to center values.
    
    Blocking can be applied to whole-plots to account for nuisance variables
    that might affect groups of runs conducted together.
    
    Examples
    --------
    >>> from factors import Factor, FactorType, ChangeabilityLevel
    >>> 
    >>> factors = [
    ...     Factor(name='Temperature', factor_type=FactorType.CONTINUOUS, 
    ...            levels=[100, 200], changeability=ChangeabilityLevel.HARD),
    ...     Factor(name='Pressure', factor_type=FactorType.CONTINUOUS,
    ...            levels=[1, 5], changeability=ChangeabilityLevel.HARD),
    ...     Factor(name='Time', factor_type=FactorType.CONTINUOUS,
    ...            levels=[10, 30], changeability=ChangeabilityLevel.EASY),
    ...     Factor(name='Catalyst', factor_type=FactorType.CATEGORICAL,
    ...            levels=['A', 'B'], changeability=ChangeabilityLevel.EASY)
    ... ]
    >>> 
    >>> design = generate_split_plot_design(
    ...     factors=factors,
    ...     n_replicates=2,
    ...     n_center_points=3,
    ...     randomize_whole_plots=True,
    ...     seed=42
    ... )
    >>> 
    >>> print(f"Whole-plots: {design.n_whole_plots}")
    >>> print(f"Sub-plots per whole-plot: {design.n_sub_plots_per_whole_plot}")
    >>> print(design.design.head())
    
    References
    ----------
    .. [1] Box, G. E., Hunter, W. G., & Hunter, J. S. (2005).
           Statistics for experimenters: design, innovation, and discovery.
           Wiley-Interscience.
    .. [2] Montgomery, D. C. (2017). Design and analysis of experiments.
           John Wiley & Sons.
    """
    # Validation
    if not factors:
        raise ValueError("At least one factor must be provided")
    if n_replicates < 1:
        raise ValueError("n_replicates must be at least 1")
    if n_center_points < 0:
        raise ValueError("n_center_points cannot be negative")
    if n_blocks < 1:
        raise ValueError("n_blocks must be at least 1")
    
    # Separate factors by changeability
    very_hard_factors = [f for f in factors if f.changeability == ChangeabilityLevel.VERY_HARD]
    hard_factors = [f for f in factors if f.changeability == ChangeabilityLevel.HARD]
    easy_factors = [f for f in factors if f.changeability == ChangeabilityLevel.EASY]
    
    # Validate changeability assignments
    if not hard_factors and not easy_factors:
        raise ValueError("Must have at least one HARD or EASY factor for split-plot design")
    
    # Determine plot structure
    if very_hard_factors:
        # Three-level nesting: very_hard -> hard -> easy
        whole_whole_plot_factors = very_hard_factors
        whole_plot_factors = hard_factors
        sub_plot_factors = easy_factors
        has_very_hard = True
    else:
        # Two-level nesting: hard -> easy
        whole_whole_plot_factors = []
        whole_plot_factors = hard_factors if hard_factors else []
        sub_plot_factors = easy_factors if easy_factors else []
        has_very_hard = False
    
    # If all factors are EASY, treat as regular factorial (no split-plot)
    if not whole_plot_factors and not whole_whole_plot_factors:
        raise ValueError(
            "All factors are EASY. Use generate_full_factorial() instead. "
            "Split-plot requires at least one HARD or VERY_HARD factor."
        )
    
    # Generate base factorial design
    rng = np.random.default_rng(seed)
    
    # Step 1: Generate all combinations
    design_dict = {}
    
    # Very hard factors (whole-whole-plot level)
    if very_hard_factors:
        vh_combinations = _generate_factor_combinations(very_hard_factors)
        n_vh_combos = len(vh_combinations)
    else:
        vh_combinations = [{}]
        n_vh_combos = 1
    
    # Hard factors (whole-plot level)
    if whole_plot_factors:
        h_combinations = _generate_factor_combinations(whole_plot_factors)
        n_h_combos = len(h_combinations)
    else:
        h_combinations = [{}]
        n_h_combos = 1
    
    # Easy factors (sub-plot level)
    if sub_plot_factors:
        e_combinations = _generate_factor_combinations(sub_plot_factors)
        n_e_combos = len(e_combinations)
    else:
        e_combinations = [{}]
        n_e_combos = 1
    
    # Calculate structure
    n_whole_plots_per_replicate = n_vh_combos * n_h_combos
    n_sub_plots_per_whole_plot = n_e_combos
    n_runs_per_replicate = n_whole_plots_per_replicate * n_sub_plots_per_whole_plot
    
    # Validate minimum structure
    if n_whole_plots_per_replicate < 2:
        raise ValueError(
            f"Design has only {n_whole_plots_per_replicate} whole-plot(s). "
            "Need at least 2 whole-plots for proper error estimation. "
            "Add more levels to HARD factors or use replicates."
        )
    if n_sub_plots_per_whole_plot < 2 and n_center_points == 0:
        raise ValueError(
            f"Design has only {n_sub_plots_per_whole_plot} sub-plot(s) per whole-plot. "
            "Need at least 2 sub-plots for proper error estimation. "
            "Add more levels to EASY factors, use center points, or use replicates."
        )
    
    # Step 2: Build complete design
    all_runs = []
    whole_plot_counter = 1
    vh_plot_counter = 1
    
    for replicate in range(n_replicates):
        # Optionally randomize order of VH combinations
        vh_order = list(range(n_vh_combos))
        if randomize_whole_plots and very_hard_factors:
            rng.shuffle(vh_order)
        
        for vh_idx in vh_order:
            vh_combo = vh_combinations[vh_idx]
            
            # Optionally randomize order of H combinations within this VH combo
            h_order = list(range(n_h_combos))
            if randomize_whole_plots:
                rng.shuffle(h_order)
            
            for h_idx in h_order:
                h_combo = h_combinations[h_idx]
                
                # This defines one whole-plot
                # Optionally randomize order of E combinations within this whole-plot
                e_order = list(range(n_e_combos))
                if randomize_sub_plots:
                    rng.shuffle(e_order)
                
                for e_idx in e_order:
                    e_combo = e_combinations[e_idx]
                    
                    # Combine all factor settings for this run
                    run = {
                        **vh_combo,
                        **h_combo,
                        **e_combo,
                        'Replicate': replicate + 1,
                        'WholePlot': whole_plot_counter
                    }
                    
                    if very_hard_factors:
                        run['VeryHardPlot'] = vh_plot_counter
                    
                    all_runs.append(run)
                
                whole_plot_counter += 1
            
            if very_hard_factors:
                vh_plot_counter += 1
    
    # Step 3: Add center points (at sub-plot level only)
    if n_center_points > 0:
        # Center points added to each whole-plot
        center_runs = []
        for wp in range(1, whole_plot_counter):
            # Find a run from this whole-plot to get its whole-plot factor settings
            wp_run = next(r for r in all_runs if r['WholePlot'] == wp)
            
            for _ in range(n_center_points):
                center_run = {
                    'Replicate': wp_run['Replicate'],
                    'WholePlot': wp
                }
                
                if very_hard_factors:
                    center_run['VeryHardPlot'] = wp_run['VeryHardPlot']
                
                # Whole-plot factors stay at their whole-plot values
                for f in very_hard_factors + whole_plot_factors:
                    center_run[f.name] = wp_run[f.name]
                
                # Sub-plot factors go to center
                for f in sub_plot_factors:
                    if f.factor_type in (FactorType.CONTINUOUS, FactorType.DISCRETE_NUMERIC):
                        center_run[f.name] = (f.max + f.min) / 2
                    else:
                        # For categorical, use first level (arbitrary choice)
                        center_run[f.name] = f.levels[0]
                
                center_runs.append(center_run)
        
        all_runs.extend(center_runs)
    
    # Step 4: Add blocking if requested
    if n_blocks > 1:
        if n_blocks > n_whole_plots_per_replicate * n_replicates:
            raise ValueError(
                f"Cannot have more blocks ({n_blocks}) than whole-plots "
                f"({n_whole_plots_per_replicate * n_replicates})"
            )
        
        # Assign blocks to whole-plots
        whole_plots_per_block = (whole_plot_counter - 1) / n_blocks
        for run in all_runs:
            run['Block'] = int((run['WholePlot'] - 1) / whole_plots_per_block) + 1
    
    # Step 5: Create DataFrame
    design = pd.DataFrame(all_runs)
    
    # Add standard order (before any randomization)
    design['StdOrder'] = range(1, len(design) + 1)
    
    # Add run order (actual execution order, already randomized above)
    design['RunOrder'] = range(1, len(design) + 1)
    
    # Reorder columns
    order_cols = ['StdOrder', 'RunOrder']
    if n_blocks > 1:
        order_cols.append('Block')
    order_cols.append('Replicate')
    if very_hard_factors:
        order_cols.append('VeryHardPlot')
    order_cols.append('WholePlot')
    
    factor_cols = [f.name for f in factors]
    design = design[order_cols + factor_cols]
    
    # Step 6: Create coded design
    design_coded = design.copy()
    for factor in factors:
        if factor.factor_type in (FactorType.CONTINUOUS, FactorType.DISCRETE_NUMERIC):
            center = (factor.max_value + factor.min_value) / 2
            half_range = (factor.max_value - factor.min_value) / 2
            design_coded[factor.name] = (design[factor.name] - center) / half_range
    
    return SplitPlotDesign(
        design=design,
        design_coded=design_coded,
        n_runs=len(design),
        n_whole_plots=whole_plot_counter - 1,
        n_sub_plots_per_whole_plot=n_sub_plots_per_whole_plot,
        whole_plot_factors=[f.name for f in whole_plot_factors + very_hard_factors],
        sub_plot_factors=[f.name for f in sub_plot_factors],
        has_very_hard_factors=has_very_hard,
        n_blocks=n_blocks
    )


def _generate_factor_combinations(factors: List[Factor]) -> List[dict]:
    """
    Generate all factorial combinations for a list of factors.
    
    Parameters
    ----------
    factors : List[Factor]
        Factors to generate combinations for
    
    Returns
    -------
    List[dict]
        List of dictionaries, each representing one factor combination
    """
    if not factors:
        return [{}]
    
    # Get levels for each factor
    factor_levels = []
    for factor in factors:
        if factor.factor_type == FactorType.CONTINUOUS:
            # Use low and high for continuous factors
            levels = [factor.min_value, factor.max_value]
        elif factor.factor_type == FactorType.DISCRETE_NUMERIC:
            levels = factor.levels
        elif factor.factor_type == FactorType.CATEGORICAL:
            levels = factor.levels
        else:
            raise ValueError(f"Unknown factor type: {factor.factor_type}")
        
        factor_levels.append(levels)
    
    # Generate all combinations
    combinations = []
    for combo in product(*factor_levels):
        combo_dict = {factors[i].name: combo[i] for i in range(len(factors))}
        combinations.append(combo_dict)
    
    return combinations


def evaluate_split_plot_design(design: SplitPlotDesign) -> dict:
    """
    Evaluate the quality and validity of a split-plot design.
    
    Parameters
    ----------
    design : SplitPlotDesign
        Design to evaluate
    
    Returns
    -------
    dict
        Dictionary containing evaluation metrics and warnings
    """
    metrics = {
        'n_runs': design.n_runs,
        'n_whole_plots': design.n_whole_plots,
        'n_sub_plots_per_whole_plot': design.n_sub_plots_per_whole_plot,
        'n_blocks': design.n_blocks,
        'whole_plot_factors': design.whole_plot_factors,
        'sub_plot_factors': design.sub_plot_factors,
        'warnings': []
    }
    
    # Check for sufficient whole-plots
    if design.n_whole_plots < 3:
        metrics['warnings'].append(
            f"Only {design.n_whole_plots} whole-plots. Consider adding replicates "
            "for better whole-plot error estimation (recommended: ≥3)."
        )
    
    # Check for sufficient sub-plots
    if design.n_sub_plots_per_whole_plot < 3:
        metrics['warnings'].append(
            f"Only {design.n_sub_plots_per_whole_plot} sub-plots per whole-plot. "
            "Consider adding center points or more factor levels "
            "for better sub-plot error estimation (recommended: ≥3)."
        )
    
    # Check balance
    wp_counts = design.design['WholePlot'].value_counts()
    if wp_counts.std() > 0:
        metrics['warnings'].append(
            "Design is unbalanced (different numbers of sub-plots per whole-plot). "
            "This complicates analysis."
        )
    else:
        metrics['balanced'] = True
    
    # Check blocking
    if design.n_blocks > 1:
        block_counts = design.design.groupby('Block')['WholePlot'].nunique()
        if block_counts.std() > 0:
            metrics['warnings'].append(
                "Blocks have different numbers of whole-plots. "
                "Prefer balanced blocking for simpler analysis."
            )
    
    return metrics