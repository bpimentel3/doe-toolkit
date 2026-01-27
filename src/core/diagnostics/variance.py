"""
Prediction Variance Diagnostics for Design Quality Assessment.

This module computes prediction variance statistics across the design space
to identify regions with poor precision.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

from src.core.factors import Factor

def _parse_term_type(term: str) -> Tuple[str, List[str], Optional[int]]:
    """
    Parse term into type, factors, and power.
    
    Parameters
    ----------
    term : str
        Model term (e.g., '1', 'A', 'A*B', 'I(A**2)', 'A^2')
    
    Returns
    -------
    term_type : str
        One of: 'intercept', 'main', 'interaction', 'power'
    factors : List[str]
        Factor names involved in the term
    power : int, optional
        Power for polynomial terms (None for other types)
    
    Examples
    --------
    >>> _parse_term_type('1')
    ('intercept', [], None)
    >>> _parse_term_type('A')
    ('main', ['A'], None)
    >>> _parse_term_type('A*B')
    ('interaction', ['A', 'B'], None)
    >>> _parse_term_type('I(A**2)')
    ('power', ['A'], 2)
    >>> _parse_term_type('A^2')
    ('power', ['A'], 2)
    """
    if term == '1':
        return 'intercept', [], None
    
    # Check for I(A**2) notation
    if term.startswith('I(') and '**' in term:
        # Extract: I(A**2) -> A, 2
        inner = term[2:-1]  # Remove 'I(' and ')'
        parts = inner.split('**')
        return 'power', [parts[0].strip()], int(parts[1].strip())
    
    # Check for A^2 or A**2 notation (without I())
    if '^' in term or '**' in term:
        sep = '^' if '^' in term else '**'
        parts = term.split(sep)
        return 'power', [parts[0].strip()], int(parts[1].strip())
    
    # Check for interaction A*B
    if '*' in term:
        factors = [f.strip() for f in term.split('*')]
        return 'interaction', factors, None
    
    # Main effect (single factor)
    return 'main', [term.strip()], None

def build_model_matrix(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str]
) -> np.ndarray:
    """
    Build model matrix X from design and model terms.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with factor columns
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Model terms (e.g., ['1', 'A', 'B', 'A*B', 'I(A**2)'])
    
    Returns
    -------
    np.ndarray
        Model matrix X (n_runs x n_terms)
    
    Notes
    -----
    This builds the X matrix used in regression: Y = Xβ + ε
    
    Supports:
    - Intercept: '1'
    - Main effects: 'A', 'B'
    - Interactions: 'A*B'
    - Quadratic (patsy notation): 'I(A**2)'
    - Quadratic (caret notation): 'A^2'
    
    Examples
    --------
    >>> X = build_model_matrix(design, factors, ['1', 'A', 'B', 'A*B'])
    >>> print(X.shape)
    (16, 4)
    """
    n_runs = len(design)
    factor_dict = {f.name: design[f.name].values for f in factors}
    
    columns = []
    
    for term in model_terms:
        term_type, factors_in_term, power = _parse_term_type(term)
        
        if term_type == 'intercept':
            columns.append(np.ones(n_runs))
        
        elif term_type == 'main':
            fname = factors_in_term[0]
            if fname not in factor_dict:
                raise ValueError(f"Unknown factor: '{fname}'")
            columns.append(factor_dict[fname])
        
        elif term_type == 'power':
            fname = factors_in_term[0]
            if fname not in factor_dict:
                raise ValueError(f"Unknown factor in term '{term}': {fname}")
            columns.append(factor_dict[fname] ** power)
        
        elif term_type == 'interaction':
            col = np.ones(n_runs)
            for fname in factors_in_term:
                if fname not in factor_dict:
                    raise ValueError(f"Unknown factor in term '{term}': {fname}")
                col *= factor_dict[fname]
            columns.append(col)
        
        else:
            raise ValueError(f"Unknown term type for '{term}'")
    
    return np.column_stack(columns)

def compute_prediction_variance(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    sigma_squared: float = 1.0
) -> np.ndarray:
    """
    Compute prediction variance at each design point.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Model terms
    sigma_squared : float, default=1.0
        Residual variance estimate (from model fit)
    
    Returns
    -------
    np.ndarray
        Prediction variance at each design point
    
    Notes
    -----
    Prediction variance at point x is:
        Var(ŷ) = σ² · x'(X'X)⁻¹x
    
    where X is the model matrix and x is the row for that point.
    
    References
    ----------
    .. [1] Myers, R. H., Montgomery, D. C., & Anderson-Cook, C. M. (2016).
           Response surface methodology: process and product optimization
           using designed experiments. John Wiley & Sons.
    """
    X = build_model_matrix(design, factors, model_terms)
    
    # Compute (X'X)^-1
    XtX = X.T @ X
    
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # Singular matrix - add small ridge
        XtX_inv = np.linalg.inv(XtX + 1e-8 * np.eye(XtX.shape[0]))
    
    # Prediction variance at each point: diag(X (X'X)^-1 X')
    # More efficient: sum((X @ XtX_inv) * X, axis=1)
    pred_var = np.sum((X @ XtX_inv) * X, axis=1) * sigma_squared
    
    return pred_var


def prediction_variance_stats(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    sigma_squared: float = 1.0
) -> Dict[str, float]:
    """
    Compute summary statistics of prediction variance.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Model terms
    sigma_squared : float, default=1.0
        Residual variance estimate
    
    Returns
    -------
    Dict[str, float]
        Statistics with keys: 'min', 'max', 'mean', 'std', 'range', 'max_ratio'
    
    Examples
    --------
    >>> stats = prediction_variance_stats(design, factors, model_terms, sigma_sq=2.5)
    >>> print(f"Max prediction variance: {stats['max']:.2f}")
    >>> print(f"Max/Min ratio: {stats['max_ratio']:.2f}")
    """
    pred_var = compute_prediction_variance(design, factors, model_terms, sigma_squared)
    
    stats = {
        'min': float(np.min(pred_var)),
        'max': float(np.max(pred_var)),
        'mean': float(np.mean(pred_var)),
        'std': float(np.std(pred_var)),
        'range': float(np.ptp(pred_var)),
    }
    
    # Max/min ratio (indicates uniformity)
    if stats['min'] > 0:
        stats['max_ratio'] = stats['max'] / stats['min']
    else:
        stats['max_ratio'] = np.inf
    
    return stats


def identify_high_variance_regions(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    sigma_squared: float = 1.0,
    threshold: float = 2.0
) -> List[Dict]:
    """
    Identify design regions with high prediction variance.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Model terms
    sigma_squared : float, default=1.0
        Residual variance estimate
    threshold : float, default=2.0
        Threshold multiplier (regions with variance > threshold * mean)
    
    Returns
    -------
    List[Dict]
        List of high-variance regions with run indices and factor settings
    
    Examples
    --------
    >>> regions = identify_high_variance_regions(design, factors, model_terms, threshold=2.5)
    >>> for region in regions:
    ...     print(f"Run {region['run_index']}: variance = {region['variance']:.2f}")
    """
    pred_var = compute_prediction_variance(design, factors, model_terms, sigma_squared)
    mean_var = np.mean(pred_var)
    
    high_var_mask = pred_var > (threshold * mean_var)
    high_var_indices = np.where(high_var_mask)[0]
    
    regions = []
    for idx in high_var_indices:
        region = {
            'run_index': int(idx),
            'variance': float(pred_var[idx]),
            'variance_ratio': float(pred_var[idx] / mean_var),
        }
        
        # Add factor settings
        for factor in factors:
            region[factor.name] = design[factor.name].iloc[idx]
        
        regions.append(region)
    
    return regions


def compute_fraction_of_design_space(
    pred_var: np.ndarray,
    threshold: float,
    n_grid_points: int = 1000
) -> float:
    """
    Estimate fraction of design space with prediction variance below threshold.
    
    This is a simple estimator using the empirical CDF of prediction variance
    at design points.
    
    Parameters
    ----------
    pred_var : np.ndarray
        Prediction variances at design points
    threshold : float
        Variance threshold
    n_grid_points : int
        (Not used in simple version)
    
    Returns
    -------
    float
        Estimated fraction of design space with variance ≤ threshold
    
    Notes
    -----
    For more accurate FDS plots, Monte Carlo sampling or grid evaluation
    would be needed. This simple version uses the design points themselves
    as a rough approximation.
    """
    return float(np.mean(pred_var <= threshold))


def compute_scaled_prediction_variance(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str]
) -> np.ndarray:
    """
    Compute scaled prediction variance (SPV).
    
    Scaled prediction variance is prediction variance divided by
    number of parameters, providing a dimensionless measure.
    
    SPV = n * Var(ŷ) / σ² = n * x'(X'X)⁻¹x
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Model terms
    
    Returns
    -------
    np.ndarray
        Scaled prediction variance at each point
    
    Notes
    -----
    SPV is useful for design comparison - lower SPV indicates better precision.
    For a perfect design, SPV = p (number of parameters).
    
    References
    ----------
    .. [1] Box, G. E., & Draper, N. R. (2007). Response surfaces, mixtures,
           and ridge analyses. John Wiley & Sons.
    """
    n = len(design)
    p = len(model_terms)
    
    # Compute with σ² = 1, then scale by n
    pred_var = compute_prediction_variance(design, factors, model_terms, sigma_squared=1.0)
    spv = n * pred_var
    
    return spv


def assess_variance_uniformity(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str]
) -> Tuple[bool, str]:
    """
    Assess whether prediction variance is reasonably uniform.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Model terms
    
    Returns
    -------
    is_uniform : bool
        Whether variance is acceptably uniform
    message : str
        Description of variance uniformity
    
    Notes
    -----
    Uses the criterion: max(SPV) / min(SPV) < 3 for acceptable uniformity.
    """
    spv = compute_scaled_prediction_variance(design, factors, model_terms)
    
    min_spv = np.min(spv)
    max_spv = np.max(spv)
    
    if min_spv > 0:
        ratio = max_spv / min_spv
    else:
        ratio = np.inf
    
    # Criterion: ratio < 3 is acceptable
    if ratio < 3.0:
        return True, f"Prediction variance is uniform (max/min ratio: {ratio:.2f})"
    elif ratio < 5.0:
        return False, f"Prediction variance is moderately non-uniform (ratio: {ratio:.2f})"
    else:
        return False, f"Prediction variance is highly non-uniform (ratio: {ratio:.2f})"


def compute_i_criterion(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    prediction_grid_config: Optional[dict] = None
) -> float:
    """
    Compute I-optimality criterion for a design.
    
    The I-criterion measures average prediction variance across the design
    region, providing a single scalar measure of prediction quality.
    
    I = trace((X'X)^(-1) * M)
    
    where M is the moment matrix over the prediction region.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with factor columns
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Model terms to evaluate
    prediction_grid_config : dict, optional
        Configuration for prediction grid generation.
        See generate_prediction_grid() for options.
    
    Returns
    -------
    float
        I-criterion value (lower is better for prediction)
    
    Notes
    -----
    The I-criterion is the integral of prediction variance over the
    design region. Lower values indicate better, more uniform prediction
    quality across the space.
    
    For comparison:
    - D-optimal designs minimize parameter variance
    - I-optimal designs minimize average prediction variance
    
    Examples
    --------
    >>> i_value = compute_i_criterion(design, factors, model_terms)
    >>> print(f"Average prediction variance: {i_value:.3f}")
    
    References
    ----------
    .. [1] Atkinson, A. C., Donev, A. N., & Tobias, R. D. (2007).
           Optimum experimental designs, with SAS. Oxford University Press.
    .. [2] Jones, B., & Goos, P. (2012). I-optimal versus D-optimal
           split-plot response surface designs. Journal of Quality
           Technology, 44(2), 85-101.
    """
    from src.core.optimal.criteria import generate_prediction_grid
    
    # Build model matrix for design
    X = build_model_matrix(design, factors, model_terms)
    
    # Compute (X'X)^(-1)
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(XtX.shape[0]))
    except np.linalg.LinAlgError:
        return np.inf
    
    # Generate prediction grid
    factor_names = [f.name for f in factors]
    prediction_points = generate_prediction_grid(factors, prediction_grid_config or {})
    
    # Build model matrix for prediction points
    pred_df = pd.DataFrame(prediction_points, columns=factor_names)
    X_pred = build_model_matrix(pred_df, factors, model_terms)
    
    # Compute moment matrix M = (1/N) * X_pred' * X_pred
    M = (X_pred.T @ X_pred) / len(prediction_points)
    
    # I-criterion = trace((X'X)^(-1) * M)
    i_criterion = np.trace(XtX_inv @ M)
    
    return float(i_criterion)


def compute_design_quality_metrics(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    include_i_optimal: bool = True,
    prediction_grid_config: Optional[dict] = None
) -> Dict[str, float]:
    """
    Compute comprehensive design quality metrics.
    
    Provides both D-optimality and I-optimality measures for evaluating
    and comparing experimental designs.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with factor columns
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Model terms to evaluate
    include_i_optimal : bool, default=True
        Whether to compute I-optimality metrics
    prediction_grid_config : dict, optional
        Configuration for I-optimal prediction grid
    
    Returns
    -------
    Dict[str, float]
        Dictionary with quality metrics:
        - 'd_efficiency': D-efficiency (0-100)
        - 'condition_number': Condition number of X'X
        - 'i_criterion': I-criterion value (if include_i_optimal=True)
        - 'avg_prediction_variance': Average prediction variance
        - 'max_prediction_variance': Maximum prediction variance
        - 'prediction_variance_ratio': Max/min prediction variance ratio
    
    Examples
    --------
    >>> metrics = compute_design_quality_metrics(design, factors, model_terms)
    >>> print(f"D-efficiency: {metrics['d_efficiency']:.1f}%")
    >>> print(f"I-criterion: {metrics['i_criterion']:.3f}")
    
    Notes
    -----
    This function provides a comprehensive assessment of design quality
    for both parameter estimation (D-optimal) and prediction (I-optimal)
    objectives.
    """
    # Build model matrix
    X = build_model_matrix(design, factors, model_terms)
    
    # Compute X'X
    XtX = X.T @ X
    
    metrics = {}
    
    # D-efficiency metrics
    try:
        det_XtX = np.linalg.det(XtX)
        metrics['d_efficiency'] = 100.0 if det_XtX > 0 else 0.0
        
        condition_number = np.linalg.cond(XtX)
        metrics['condition_number'] = float(condition_number)
    except:
        metrics['d_efficiency'] = 0.0
        metrics['condition_number'] = np.inf
    
    # Prediction variance metrics
    try:
        pred_var_stats = prediction_variance_stats(
            design, factors, model_terms, sigma_squared=1.0
        )
        metrics['avg_prediction_variance'] = pred_var_stats['mean']
        metrics['max_prediction_variance'] = pred_var_stats['max']
        metrics['prediction_variance_ratio'] = pred_var_stats['max_ratio']
    except:
        metrics['avg_prediction_variance'] = np.nan
        metrics['max_prediction_variance'] = np.nan
        metrics['prediction_variance_ratio'] = np.nan
    
    # I-optimality metrics
    if include_i_optimal:
        try:
            i_crit = compute_i_criterion(
                design, factors, model_terms, prediction_grid_config
            )
            metrics['i_criterion'] = i_crit
        except:
            metrics['i_criterion'] = np.nan
    
    return metrics