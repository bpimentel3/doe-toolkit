"""
Estimability Diagnostics for Design Quality Assessment.

This module computes Variance Inflation Factors (VIF), identifies collinearity
issues, and assesses model matrix conditioning.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import warnings

from src.core.factors import Factor


def compute_vif(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str]
) -> Dict[str, float]:
    """
    Compute Variance Inflation Factors for model terms.
    
    VIF measures multicollinearity by regressing each predictor on all others.
    VIF = 1/(1-R²) where R² comes from auxiliary regression.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with factor columns
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Model terms
    
    Returns
    -------
    Dict[str, float]
        VIF value for each term (excluding intercept)
        VIF > 10 indicates problematic collinearity
    
    Notes
    -----
    The auxiliary regression for each predictor X_j is:
        X_j ~ X_-j
    where X_-j contains all other predictors INCLUDING the intercept.
    
    The VIF cannot be computed if:
    - Design is saturated (n ≤ p)
    - Auxiliary regression is singular
    
    Examples
    --------
    >>> vif = compute_vif(design, factors, ['1', 'A', 'B', 'A*B'])
    >>> print(vif['A*B'])
    2.5
    """
    from src.core.diagnostics.variance import build_model_matrix

    # Build full model matrix INCLUDING intercept
    X = build_model_matrix(design, factors, model_terms)

    # Terms for which we compute VIF (exclude intercept)
    terms_no_intercept = [t for t in model_terms if t != '1']

    # Map each term to its correct column index in X
    col_indices = [model_terms.index(t) for t in terms_no_intercept]

    if X.shape[1] == 0:
        return {}

    n, p = X.shape

    if n <= p:
        warnings.warn(
            f"Cannot compute VIF: n_runs ({n}) ≤ n_terms ({p}). "
            "Design is saturated or supersaturated."
        )
        return {term: np.nan for term in terms_no_intercept}

    vif_values = {}

    for term, col_idx in zip(terms_no_intercept, col_indices):
        # Predictor column to explain
        X_j = X[:, col_idx].reshape(-1, 1)

        # All other predictors (includes intercept if it was in model_terms)
        X_not_j = np.delete(X, col_idx, axis=1)

        try:
            # Regress X_j on all other columns using least squares
            # Note: X_not_j already includes intercept (first column of X)
            beta, residuals, rank, s = np.linalg.lstsq(X_not_j, X_j, rcond=None)
            
            # Calculate R² from auxiliary regression
            y_pred = X_not_j @ beta
            ss_res = np.sum((X_j.flatten() - y_pred.flatten()) ** 2)
            ss_tot = np.sum((X_j.flatten() - np.mean(X_j)) ** 2)

            if ss_tot > 0:
                r_squared = 1 - ss_res / ss_tot
            else:
                r_squared = 0
            
            # VIF = 1 / (1 - R²)
            # If R² ≈ 1, VIF → ∞ (perfect collinearity)
            if r_squared >= 0.9999:
                vif = np.inf
            elif r_squared < 0:  # Can happen with poor fit
                vif = 1.0
            else:
                vif = 1.0 / (1.0 - r_squared)

            vif_values[term] = float(vif)

        except (np.linalg.LinAlgError, ValueError):
            # Singular matrix or other numerical issue
            vif_values[term] = np.inf

    return vif_values

def check_collinearity(
    vif_values: Dict[str, float],
    threshold: float = 10.0
) -> List[str]:
    """Identify terms with VIF exceeding threshold."""
    problematic = []

    for term, vif in vif_values.items():
        if not np.isnan(vif) and not np.isinf(vif) and vif > threshold:
            problematic.append(term)

    return problematic


def compute_condition_number(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str]
) -> float:
    """Compute condition number of model matrix."""

    from src.core.diagnostics.variance import build_model_matrix

    X = build_model_matrix(design, factors, model_terms)

    try:
        singular_values = np.linalg.svd(X, compute_uv=False)

        if singular_values[-1] > 0:
            return float(singular_values[0] / singular_values[-1])
        else:
            return np.inf

    except (np.linalg.LinAlgError, ValueError):
        return np.inf


def assess_estimability(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str]
) -> Tuple[bool, List[str]]:
    """Assess whether all model terms are estimable."""

    from src.core.diagnostics.variance import build_model_matrix

    X = build_model_matrix(design, factors, model_terms)
    n, p = X.shape

    issues = []

    # Supersaturated
    if n < p:
        issues.append(
            f"Design is supersaturated: {n} runs for {p} terms. "
            "Cannot estimate all effects."
        )
        return False, issues

    # Saturated
    if n == p:
        issues.append(
            f"Design is saturated: {n} runs for {p} terms. "
            "No degrees of freedom for error. Cannot test effects."
        )

    # Rank deficiency
    rank = np.linalg.matrix_rank(X)
    if rank < p:
        issues.append(
            f"Model matrix is rank-deficient: rank = {rank}, expected {p}. "
            f"{p - rank} term(s) are linearly dependent."
        )
        return False, issues

    # Severe collinearity
    vif_values = compute_vif(design, factors, model_terms)
    high_vif = [term for term, vif in vif_values.items() if vif > 50]

    if high_vif:
        issues.append(
            f"Severe collinearity detected (VIF > 50): {', '.join(high_vif)}. "
            "Coefficient estimates may be unreliable."
        )

    # Condition number
    kappa = compute_condition_number(design, factors, model_terms)
    if kappa > 1000:
        issues.append(
            f"Severely ill-conditioned design (κ = {kappa:.1e}). "
            "Numerical instability likely."
        )

    all_estimable = len(issues) == 0 or (
        len(issues) == 1 and 'saturated' in issues[0]
    )

    return all_estimable, issues


def identify_redundant_terms(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    tolerance: float = 1e-6
) -> List[str]:
    """Identify model terms that are linearly dependent."""

    from src.core.diagnostics.variance import build_model_matrix

    X = build_model_matrix(design, factors, model_terms)

    try:
        Q, R = np.linalg.qr(X)
        diag_R = np.abs(np.diag(R))
        dependent_indices = np.where(diag_R < tolerance)[0]

        return [model_terms[i] for i in dependent_indices]

    except (np.linalg.LinAlgError, ValueError):
        return []


def compute_leverage(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str]
) -> np.ndarray:
    """Compute leverage (hat values) for each observation."""

    from src.core.diagnostics.variance import build_model_matrix

    X = build_model_matrix(design, factors, model_terms)
    n, p = X.shape

    try:
        XtX = X.T @ X
        XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(p))

        leverage = np.sum((X @ XtX_inv) * X, axis=1)
        return leverage

    except (np.linalg.LinAlgError, ValueError):
        return np.full(n, np.nan)


def identify_high_leverage_points(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    threshold_multiplier: float = 2.0
) -> List[int]:
    """Identify observations with high leverage."""

    leverage = compute_leverage(design, factors, model_terms)

    if np.any(np.isnan(leverage)):
        return []

    n = len(design)
    p = len(model_terms)
    avg_leverage = p / n

    threshold = threshold_multiplier * avg_leverage
    return list(np.where(leverage > threshold)[0])
