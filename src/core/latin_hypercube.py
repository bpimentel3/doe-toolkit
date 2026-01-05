"""
Latin Hypercube Sampling for Design of Experiments.

This module provides space-filling designs using Latin Hypercube Sampling (LHS),
which is particularly useful for initial exploration and computer experiments.
"""

from typing import List, Literal, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
from scipy.spatial.distance import pdist

from src.core.factors import Factor, FactorType


@dataclass
class LHSDesign:
    """
    Container for Latin Hypercube Sampling design and metadata.
    
    Attributes
    ----------
    design : pd.DataFrame
        Design matrix with factor columns in actual levels
    design_coded : pd.DataFrame
        Design matrix with continuous factors in coded [-1, 1] scale
    n_runs : int
        Number of experimental runs
    criterion : str
        Selection criterion used ('maximin' or 'correlation')
    criterion_value : float
        Value of the criterion for selected design
    """
    design: pd.DataFrame
    design_coded: pd.DataFrame
    n_runs: int
    criterion: str
    criterion_value: float


def generate_latin_hypercube(
    factors: List[Factor],
    n_runs: int,
    criterion: Literal['maximin', 'correlation'] = 'maximin',
    n_candidates: int = 10,
    seed: int = None
) -> LHSDesign:
    """
    Generate Latin Hypercube Sample with optimized space-filling properties.
    
    This function generates multiple candidate LHS designs and selects the best
    according to the specified criterion. Continuous and discrete numeric factors
    use LHS, while categorical factors use stratified random sampling.
    
    Parameters
    ----------
    factors : List[Factor]
        List of factor definitions
    n_runs : int
        Number of experimental runs to generate
    criterion : {'maximin', 'correlation'}, default='maximin'
        Selection criterion:
        - 'maximin': Maximize minimum distance between points (best space-filling)
        - 'correlation': Minimize maximum absolute correlation between factors
    n_candidates : int, default=10
        Number of candidate designs to generate and evaluate
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    LHSDesign
        Object containing design matrix, coded design, and metadata
    
    Raises
    ------
    ValueError
        If n_runs < 2, n_candidates < 1, or no factors provided
    
    Notes
    -----
    Latin Hypercube Sampling ensures that each factor's range is divided into
    n equal intervals, with exactly one sample point in each interval. This
    provides better space coverage than pure random sampling.
    
    For discrete numeric factors, continuous LHS values are rounded to the
    nearest allowed level. For categorical factors, levels are assigned to
    ensure balanced representation.
    
    The maximin criterion produces designs with better space-filling properties,
    while the correlation criterion produces designs with more independent factors.
    
    Examples
    --------
    >>> from src.core.factors import Factor, FactorType
    >>> factors = [
    ...     Factor(name='Temperature', factor_type=FactorType.CONTINUOUS, 
    ...            levels=[100, 200], units='°C'),
    ...     Factor(name='Pressure', factor_type=FactorType.CONTINUOUS,
    ...            levels=[1, 5], units='bar')
    ... ]
    >>> design = generate_latin_hypercube(factors, n_runs=20, criterion='maximin')
    >>> print(design.design.head())
    
    References
    ----------
    .. [1] McKay, M. D., Beckman, R. J., & Conover, W. J. (1979).
           A comparison of three methods for selecting values of input
           variables in the analysis of output from a computer code.
           Technometrics, 21(2), 239-245.
    """
    # Validation
    if n_runs < 2:
        raise ValueError("n_runs must be at least 2")
    if n_candidates < 1:
        raise ValueError("n_candidates must be at least 1")
    if not factors:
        raise ValueError("At least one factor must be provided")
    
    # Separate factors by type
    continuous_factors = [f for f in factors 
                         if f.factor_type in (FactorType.CONTINUOUS, FactorType.DISCRETE_NUMERIC)]
    categorical_factors = [f for f in factors if f.factor_type == FactorType.CATEGORICAL]
    
    # Set up random state
    rng = np.random.default_rng(seed)
    
    # Generate candidate designs
    candidates = []
    candidate_scores = []
    
    for i in range(n_candidates):
        # Generate design for this candidate
        candidate_seed = None if seed is None else seed + i
        design_dict = {}
        design_coded_dict = {}
        
        # LHS for continuous/discrete factors
        if continuous_factors:
            n_continuous = len(continuous_factors)
            sampler = LatinHypercube(d=n_continuous, seed=candidate_seed)
            lhs_samples = sampler.random(n=n_runs)  # Returns [0, 1] scaled samples
            
            for idx, factor in enumerate(continuous_factors):
                # Scale from [0, 1] to [min, max]
                scaled = factor.min_value + lhs_samples[:, idx] * (factor.max_value - factor.min_value)
                
                if factor.factor_type == FactorType.DISCRETE_NUMERIC:
                    # Round to nearest allowed level
                    scaled = np.array([min(factor.levels, key=lambda x: abs(x - val)) 
                                      for val in scaled])
                
                design_dict[factor.name] = scaled
                
                # Coded levels: [-1, 1] scale
                center = (factor.max_value + factor.min_value) / 2
                half_range = (factor.max_value - factor.min_value) / 2
                coded = (scaled - center) / half_range
                design_coded_dict[factor.name] = coded
        
        # Stratified sampling for categorical factors
        for factor in categorical_factors:
            # Ensure balanced representation across levels
            n_levels = len(factor.levels)
            
            # Calculate how many times each level should appear
            base_count = n_runs // n_levels
            remainder = n_runs % n_levels
            
            # Create level assignments
            level_assignments = []
            for level_idx in range(n_levels):
                count = base_count + (1 if level_idx < remainder else 0)
                level_assignments.extend([factor.levels[level_idx]] * count)
            
            # Randomize order
            rng.shuffle(level_assignments)
            
            design_dict[factor.name] = level_assignments
            # Categorical factors don't have coded values in traditional sense
            design_coded_dict[factor.name] = level_assignments
        
        # Create DataFrames
        design = pd.DataFrame(design_dict)
        design_coded = pd.DataFrame(design_coded_dict)
        
        # Score this candidate
        if continuous_factors:
            # Extract only continuous/discrete numeric columns for scoring
            numeric_cols = [f.name for f in continuous_factors]
            numeric_data = design_coded[numeric_cols].values
            
            if criterion == 'maximin':
                # Maximize minimum distance between points
                distances = pdist(numeric_data)
                score = np.min(distances)  # Higher is better
            elif criterion == 'correlation':
                # Minimize maximum absolute correlation
                if len(continuous_factors) > 1:
                    corr_matrix = np.corrcoef(numeric_data.T)
                    # Get upper triangle (exclude diagonal)
                    upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                    score = -np.max(np.abs(upper_triangle))  # More negative is better (lower correlation)
                else:
                    score = 0.0  # Single factor, no correlation
            else:
                raise ValueError(f"Unknown criterion: {criterion}")
        else:
            # Only categorical factors, use dummy score
            score = 0.0
        
        candidates.append((design, design_coded))
        candidate_scores.append(score)
    
    # Select best candidate
    best_idx = np.argmax(candidate_scores)
    best_design, best_design_coded = candidates[best_idx]
    best_score = candidate_scores[best_idx]
    
    # Add standard run order columns
    best_design.insert(0, 'StdOrder', range(1, n_runs + 1))
    best_design.insert(1, 'RunOrder', range(1, n_runs + 1))
    
    best_design_coded.insert(0, 'StdOrder', range(1, n_runs + 1))
    best_design_coded.insert(1, 'RunOrder', range(1, n_runs + 1))
    
    return LHSDesign(
        design=best_design,
        design_coded=best_design_coded,
        n_runs=n_runs,
        criterion=criterion,
        criterion_value=best_score
    )

def _generate_candidate_augmentations(
    existing_clean: pd.DataFrame,
    factors: List[Factor],
    n_additional_runs: int,
    n_candidates: int,
    criterion: str,
    seed: Optional[int]
) -> Tuple[pd.DataFrame, pd.DataFrame, float]:
    """
    Generate and evaluate candidate augmentations.
    
    Parameters
    ----------
    existing_clean : pd.DataFrame
        Existing design without StdOrder/RunOrder
    factors : List[Factor]
        Factor definitions
    n_additional_runs : int
        Number of runs to add
    n_candidates : int
        Number of candidates to evaluate
    criterion : str
        'maximin' or 'correlation'
    seed : int, optional
        Random seed
    
    Returns
    -------
    best_additional_design : pd.DataFrame
        Best new runs (actual levels)
    best_additional_coded : pd.DataFrame
        Best new runs (coded levels)
    best_score : float
        Score of best augmentation
    """
    best_score = -np.inf
    best_additional_design = None
    best_additional_coded = None
    
    for i in range(n_candidates):
        # Generate new runs
        candidate_seed = None if seed is None else seed + i
        new_design = generate_latin_hypercube(
            factors=factors,
            n_runs=n_additional_runs,
            criterion=criterion,
            n_candidates=1,
            seed=candidate_seed
        )
        
        # Remove StdOrder/RunOrder for combination
        new_clean = new_design.design.drop(columns=['StdOrder', 'RunOrder'])
        new_coded_clean = new_design.design_coded.drop(columns=['StdOrder', 'RunOrder'])
        
        # Score this augmentation
        score = _score_augmentation(
            existing_clean=existing_clean,
            new_clean=new_clean,
            new_coded_clean=new_coded_clean,
            factors=factors,
            criterion=criterion
        )
        
        # Track best
        if score > best_score:
            best_score = score
            best_additional_design = new_clean
            best_additional_coded = new_coded_clean
    
    return best_additional_design, best_additional_coded, best_score


def _score_augmentation(
    existing_clean: pd.DataFrame,
    new_clean: pd.DataFrame,
    new_coded_clean: pd.DataFrame,
    factors: List[Factor],
    criterion: str
) -> float:
    """
    Score an augmentation candidate.
    
    Parameters
    ----------
    existing_clean : pd.DataFrame
        Existing design (actual levels)
    new_clean : pd.DataFrame
        New runs (actual levels)
    new_coded_clean : pd.DataFrame
        New runs (coded levels)
    factors : List[Factor]
        Factor definitions
    criterion : str
        'maximin' or 'correlation'
    
    Returns
    -------
    float
        Score (higher is better)
    """
    continuous_factors = [f for f in factors 
                         if f.factor_type in (FactorType.CONTINUOUS, FactorType.DISCRETE_NUMERIC)]
    
    if not continuous_factors:
        return 0.0
    
    # Code existing design for scoring
    numeric_cols = [f.name for f in continuous_factors]
    existing_coded = _code_design(existing_clean[numeric_cols], continuous_factors)
    combined_coded = pd.concat([existing_coded, new_coded_clean[numeric_cols]], ignore_index=True)
    
    numeric_data = combined_coded.values
    
    if criterion == 'maximin':
        distances = pdist(numeric_data)
        score = np.min(distances)
    elif criterion == 'correlation':
        if len(continuous_factors) > 1:
            corr_matrix = np.corrcoef(numeric_data.T)
            upper_triangle = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            score = -np.max(np.abs(upper_triangle))
        else:
            score = 0.0
    else:
        raise ValueError(f"Unknown criterion: {criterion}")
    
    return score


def _build_combined_design(
    existing_clean: pd.DataFrame,
    best_additional_design: pd.DataFrame,
    best_additional_coded: pd.DataFrame,
    factors: List[Factor],
    total_runs: int,
    best_score: float,
    criterion: str
) -> LHSDesign:
    """
    Build final combined design with proper coding.
    
    Parameters
    ----------
    existing_clean : pd.DataFrame
        Existing design without StdOrder/RunOrder
    best_additional_design : pd.DataFrame
        Best new runs (actual levels)
    best_additional_coded : pd.DataFrame
        Best new runs (coded levels)
    factors : List[Factor]
        Factor definitions
    total_runs : int
        Total number of runs
    best_score : float
        Final criterion value
    criterion : str
        Criterion used
    
    Returns
    -------
    LHSDesign
        Combined design
    """
    # Combine designs
    combined_design = pd.concat([existing_clean, best_additional_design], ignore_index=True)
    
    # Code existing design properly for output
    continuous_factors = [f for f in factors 
                         if f.factor_type in (FactorType.CONTINUOUS, FactorType.DISCRETE_NUMERIC)]
    categorical_factors = [f for f in factors if f.factor_type == FactorType.CATEGORICAL]
    
    combined_coded_dict = {}
    if continuous_factors:
        numeric_cols = [f.name for f in continuous_factors]
        existing_coded = _code_design(existing_clean[numeric_cols], continuous_factors)
        combined_coded_numeric = pd.concat([existing_coded, best_additional_coded[numeric_cols]], ignore_index=True)
        for col in numeric_cols:
            combined_coded_dict[col] = combined_coded_numeric[col]
    
    for factor in categorical_factors:
        combined_coded_dict[factor.name] = combined_design[factor.name]
    
    combined_coded = pd.DataFrame(combined_coded_dict)
    
    # Add run order columns
    combined_design.insert(0, 'StdOrder', range(1, total_runs + 1))
    combined_design.insert(1, 'RunOrder', range(1, total_runs + 1))
    
    combined_coded.insert(0, 'StdOrder', range(1, total_runs + 1))
    combined_coded.insert(1, 'RunOrder', range(1, total_runs + 1))
    
    return LHSDesign(
        design=combined_design,
        design_coded=combined_coded,
        n_runs=total_runs,
        criterion=criterion,
        criterion_value=best_score
    )


def augment_latin_hypercube(
    existing_design: pd.DataFrame,
    factors: List[Factor],
    n_additional_runs: int,
    criterion: Literal['maximin', 'correlation'] = 'maximin',
    n_candidates: int = 10,
    seed: int = None
) -> LHSDesign:
    """
    Add runs to an existing Latin Hypercube design.
    
    This function generates additional runs that maintain good space-filling
    properties when combined with the existing design.
    
    [Keep existing docstring parameters/returns/examples...]
    """
    # Validate inputs
    if n_additional_runs < 1:
        raise ValueError("n_additional_runs must be at least 1")
    
    # Remove StdOrder/RunOrder if present
    existing_clean = existing_design.copy()
    if 'StdOrder' in existing_clean.columns:
        existing_clean = existing_clean.drop(columns=['StdOrder', 'RunOrder'])
    
    # Validate that factors match existing design
    factor_names = [f.name for f in factors]
    if not all(name in existing_clean.columns for name in factor_names):
        raise ValueError("Factor names must match existing design columns")
    
    n_existing = len(existing_clean)
    total_runs = n_existing + n_additional_runs
    
    # Generate and evaluate candidate augmentations
    best_additional_design, best_additional_coded, best_score = _generate_candidate_augmentations(
        existing_clean=existing_clean,
        factors=factors,
        n_additional_runs=n_additional_runs,
        n_candidates=n_candidates,
        criterion=criterion,
        seed=seed
    )
    
    # Build final combined design
    return _build_combined_design(
        existing_clean=existing_clean,
        best_additional_design=best_additional_design,
        best_additional_coded=best_additional_coded,
        factors=factors,
        total_runs=total_runs,
        best_score=best_score,
        criterion=criterion
    )

def _code_design(design: pd.DataFrame, factors: List[Factor]) -> pd.DataFrame:
    """
    Convert actual levels to coded [-1, 1] scale.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design with actual levels
    factors : List[Factor]
        Factor definitions for coding
    
    Returns
    -------
    pd.DataFrame
        Design with coded levels
    """
    coded_dict = {}
    for factor in factors:
        center = (factor.max_value + factor.min_value) / 2
        half_range = (factor.max_value - factor.min_value) / 2
        coded = (design[factor.name] - center) / half_range
        coded_dict[factor.name] = coded
    
    return pd.DataFrame(coded_dict)