"""
Stepwise Regression with BIC for Model Selection.

Implements bidirectional stepwise selection using Bayesian Information Criterion
with mandatory hierarchy enforcement.
"""

from typing import List, Dict, Optional, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
import warnings

from src.core.analysis import (
    ANOVAAnalysis,
    ANOVAResults,
    enforce_hierarchy,
    parse_model_term
)
from src.core.factors import Factor


@dataclass
class StepwiseStep:
    """
    Record of a single stepwise iteration.
    
    Parameters
    ----------
    step_number : int
        Iteration number (1-indexed)
    action : str
        'add' or 'remove'
    term : str
        Term that was added or removed
    bic : float
        BIC after this step
    delta_bic : float
        Change in BIC from previous step (negative is improvement)
    current_terms : List[str]
        Model terms after this step
    """
    step_number: int
    action: str
    term: str
    bic: float
    delta_bic: float
    current_terms: List[str]


@dataclass
class StepwiseResults:
    """
    Results from stepwise regression procedure.
    
    Parameters
    ----------
    final_terms : List[str]
        Selected model terms
    final_bic : float
        BIC of final model
    initial_bic : float
        BIC of starting model
    improvement : float
        Total BIC improvement (initial - final, positive is better)
    steps : List[StepwiseStep]
        Record of each iteration
    n_iterations : int
        Total number of iterations performed
    convergence_reason : str
        Why the procedure stopped
    final_anova_results : ANOVAResults
        ANOVA results for final model
    """
    final_terms: List[str]
    final_bic: float
    initial_bic: float
    improvement: float
    steps: List[StepwiseStep]
    n_iterations: int
    convergence_reason: str
    final_anova_results: ANOVAResults


def compute_bic(anova_results: ANOVAResults, n_obs: int) -> float:
    """
    Compute Bayesian Information Criterion.
    
    BIC = n*ln(RSS/n) + k*ln(n)
    
    Where:
    - n = number of observations
    - RSS = residual sum of squares
    - k = number of parameters (including intercept)
    
    Lower BIC is better.
    
    Parameters
    ----------
    anova_results : ANOVAResults
        Fitted model results
    n_obs : int
        Number of observations
    
    Returns
    -------
    float
        BIC value
    
    Notes
    -----
    For mixed models (split-plot), uses marginal log-likelihood if available,
    otherwise falls back to RSS-based calculation.
    
    References
    ----------
    .. [1] Schwarz, G. (1978). "Estimating the dimension of a model."
           Annals of Statistics, 6(2), 461-464.
    """
    # Number of parameters (including intercept)
    k = len(anova_results.model_terms)
    
    # Try to get log-likelihood for mixed models
    if hasattr(anova_results.fitted_model, 'llf'):
        # statsmodels provides log-likelihood directly
        log_likelihood = anova_results.fitted_model.llf
        bic = -2 * log_likelihood + k * np.log(n_obs)
    else:
        # RSS-based calculation for OLS
        rss = np.sum(anova_results.residuals ** 2)
        
        # Avoid log(0) or log(negative)
        if rss <= 0:
            warnings.warn("RSS ≤ 0 detected, returning infinite BIC")
            return np.inf
        
        bic = n_obs * np.log(rss / n_obs) + k * np.log(n_obs)
    
    return bic


def get_candidate_terms_forward(
    current_terms: List[str],
    all_possible_terms: List[str],
    factor_names: List[str]
) -> List[str]:
    """
    Get candidate terms for forward addition.
    
    Returns terms not in current model that respect hierarchy.
    
    Parameters
    ----------
    current_terms : List[str]
        Terms currently in model
    all_possible_terms : List[str]
        All available terms to consider
    factor_names : List[str]
        Names of all factors
    
    Returns
    -------
    List[str]
        Terms that can be added while respecting hierarchy
    """
    candidates = []
    current_set = set(current_terms)
    
    for term in all_possible_terms:
        if term in current_set or term == '1':
            continue
        
        # Check if adding this term would violate hierarchy
        test_terms = current_terms + [term]
        enforced_terms, _ = enforce_hierarchy(test_terms, factor_names)
        
        # If enforce_hierarchy added new terms beyond our candidate,
        # then this term requires prerequisites
        if set(enforced_terms) - current_set == {term}:
            candidates.append(term)
    
    return candidates


def get_candidate_terms_backward(
    current_terms: List[str],
    factor_names: List[str],
    mandatory_terms: List[str]
) -> List[str]:
    """
    Get candidate terms for backward elimination.
    
    Returns terms that can be removed without violating hierarchy
    or removing mandatory terms.
    
    Parameters
    ----------
    current_terms : List[str]
        Terms currently in model
    factor_names : List[str]
        Names of all factors
    mandatory_terms : List[str]
        Terms that cannot be removed (e.g., '1' for intercept)
    
    Returns
    -------
    List[str]
        Terms that can be removed while respecting hierarchy
    """
    candidates = []
    mandatory_set = set(mandatory_terms)
    
    for term in current_terms:
        if term in mandatory_set:
            continue
        
        # Try removing this term
        test_terms = [t for t in current_terms if t != term]
        
        # Check hierarchy
        enforced_terms, added = enforce_hierarchy(test_terms, factor_names)
        
        # If nothing was added back, we can safely remove this term
        if len(added) == 0:
            candidates.append(term)
    
    return candidates


def stepwise_selection(
    anova_analysis: ANOVAAnalysis,
    all_possible_terms: List[str],
    starting_terms: Optional[List[str]] = None,
    mandatory_terms: Optional[List[str]] = None,
    max_iterations: int = 50,
    bic_threshold: float = 2.0,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> StepwiseResults:
    """
    Perform bidirectional stepwise model selection using BIC.
    
    Starts with a minimal model and iteratively adds/removes terms to
    minimize BIC while respecting model hierarchy.
    
    Parameters
    ----------
    anova_analysis : ANOVAAnalysis
        Analysis object with data and factors
    all_possible_terms : List[str]
        All terms to consider for selection (candidate pool)
    starting_terms : List[str], optional
        Initial model terms. If None, starts with ['1'] (intercept only)
    mandatory_terms : List[str], optional
        Terms that must remain in model. If None, defaults to ['1']
    max_iterations : int, default=50
        Maximum number of stepwise iterations
    bic_threshold : float, default=2.0
        Minimum BIC improvement to continue (ΔBIC must exceed this)
    progress_callback : Callable[[int, int], None], optional
        Callback function called as progress_callback(current_step, total_steps)
        for progress tracking
    
    Returns
    -------
    StepwiseResults
        Complete stepwise selection results
    
    Notes
    -----
    Algorithm:
    1. Fit current model, compute BIC
    2. Try adding each candidate term (forward step)
    3. Try removing each removable term (backward step)
    4. Select action with best BIC improvement
    5. If improvement < threshold, stop
    6. Repeat until convergence or max_iterations
    
    Hierarchy enforcement ensures that:
    - Interactions require their constituent main effects
    - Quadratic terms require the main effect
    
    Examples
    --------
    >>> analysis = ANOVAAnalysis(design, response, factors)
    >>> all_terms = ['1', 'A', 'B', 'C', 'A*B', 'A*C', 'B*C', 'I(A**2)']
    >>> results = stepwise_selection(analysis, all_terms)
    >>> print(results.final_terms)
    ['1', 'A', 'B', 'A*B']
    
    References
    ----------
    .. [1] Hocking, R. R. (1976). "A Biometrics invited paper. The analysis
           and selection of variables in linear regression." Biometrics, 32(1), 1-49.
    """
    # Initialize
    if starting_terms is None:
        starting_terms = ['1']
    
    if mandatory_terms is None:
        mandatory_terms = ['1']
    
    factor_names = [f.name for f in anova_analysis.factors]
    current_terms = starting_terms.copy()
    
    # Enforce hierarchy on starting model
    current_terms, _ = enforce_hierarchy(current_terms, factor_names)
    
    # Fit initial model
    initial_results = anova_analysis.fit(current_terms, enforce_hierarchy_flag=True)
    n_obs = len(anova_analysis.data)
    current_bic = compute_bic(initial_results, n_obs)
    initial_bic = current_bic
    
    steps = []
    iteration = 0
    
    # Main stepwise loop
    for iteration in range(max_iterations):
        if progress_callback:
            progress_callback(iteration + 1, max_iterations)
        
        best_bic = current_bic
        best_action = None
        best_term = None
        best_terms = None
        
        # === FORWARD STEP: Try adding terms ===
        forward_candidates = get_candidate_terms_forward(
            current_terms, all_possible_terms, factor_names
        )
        
        for candidate in forward_candidates:
            try:
                # Try adding this term
                test_terms = current_terms + [candidate]
                test_terms, _ = enforce_hierarchy(test_terms, factor_names)
                
                # Fit model
                test_results = anova_analysis.fit(test_terms, enforce_hierarchy_flag=True)
                test_bic = compute_bic(test_results, n_obs)
                
                # Track best improvement
                if test_bic < best_bic:
                    best_bic = test_bic
                    best_action = 'add'
                    best_term = candidate
                    best_terms = test_terms
            
            except Exception as e:
                # Model fitting failed, skip this candidate
                warnings.warn(f"Failed to fit model with term '{candidate}': {e}")
                continue
        
        # === BACKWARD STEP: Try removing terms ===
        backward_candidates = get_candidate_terms_backward(
            current_terms, factor_names, mandatory_terms
        )
        
        for candidate in backward_candidates:
            try:
                # Try removing this term
                test_terms = [t for t in current_terms if t != candidate]
                
                # Hierarchy should already be satisfied after removal
                # (get_candidate_terms_backward checks this)
                
                # Fit model
                test_results = anova_analysis.fit(test_terms, enforce_hierarchy_flag=True)
                test_bic = compute_bic(test_results, n_obs)
                
                # Track best improvement
                if test_bic < best_bic:
                    best_bic = test_bic
                    best_action = 'remove'
                    best_term = candidate
                    best_terms = test_terms
            
            except Exception as e:
                # Model fitting failed, skip this candidate
                warnings.warn(f"Failed to fit model without term '{candidate}': {e}")
                continue
        
        # === CHECK CONVERGENCE ===
        delta_bic = current_bic - best_bic  # Positive = improvement
        
        # No improvement found
        if best_action is None or delta_bic < bic_threshold:
            convergence_reason = (
                "No further improvement possible" if best_action is None
                else f"BIC improvement below threshold (ΔBIC = {delta_bic:.2f} < {bic_threshold})"
            )
            break
        
        # Apply best action
        current_terms = best_terms
        current_bic = best_bic
        
        # Record this step
        step = StepwiseStep(
            step_number=len(steps) + 1,
            action=best_action,
            term=best_term,
            bic=current_bic,
            delta_bic=-delta_bic,  # Negative because we report as change in BIC
            current_terms=current_terms.copy()
        )
        steps.append(step)
    
    else:
        # Loop completed without break (hit max_iterations)
        convergence_reason = f"Maximum iterations reached ({max_iterations})"
    
    # Fit final model
    final_results = anova_analysis.fit(current_terms, enforce_hierarchy_flag=True)
    final_bic = compute_bic(final_results, n_obs)
    
    return StepwiseResults(
        final_terms=current_terms,
        final_bic=final_bic,
        initial_bic=initial_bic,
        improvement=initial_bic - final_bic,
        steps=steps,
        n_iterations=len(steps),
        convergence_reason=convergence_reason,
        final_anova_results=final_results
    )


def format_stepwise_summary(results: StepwiseResults) -> str:
    """
    Format stepwise results as markdown summary.
    
    Parameters
    ----------
    results : StepwiseResults
        Stepwise selection results
    
    Returns
    -------
    str
        Markdown-formatted summary
    
    Examples
    --------
    >>> summary = format_stepwise_summary(results)
    >>> print(summary)
    """
    lines = []
    
    # Header
    lines.append("### 🔍 Stepwise Regression Summary")
    lines.append("")
    
    # Overall results
    lines.append(f"**Convergence:** {results.convergence_reason}")
    lines.append(f"**Iterations:** {results.n_iterations}")
    lines.append(f"**Initial BIC:** {results.initial_bic:.2f}")
    lines.append(f"**Final BIC:** {results.final_bic:.2f}")
    
    improvement_sign = "✓" if results.improvement > 0 else "✗"
    lines.append(f"**Improvement:** {improvement_sign} {results.improvement:.2f}")
    lines.append("")
    
    # Steps table
    if results.steps:
        lines.append("**Selection Steps:**")
        lines.append("")
        lines.append("| Step | Action | Term | BIC | ΔBIC |")
        lines.append("|------|--------|------|-----|------|")
        
        for step in results.steps:
            action_icon = "➕" if step.action == 'add' else "➖"
            lines.append(
                f"| {step.step_number} | {action_icon} {step.action} | "
                f"`{step.term}` | {step.bic:.2f} | {step.delta_bic:.2f} |"
            )
        
        lines.append("")
    
    # Final model
    lines.append("**Final Model Terms:**")
    lines.append("")
    non_intercept = [t for t in results.final_terms if t != '1']
    if non_intercept:
        for term in non_intercept:
            lines.append(f"- `{term}`")
    else:
        lines.append("- (Intercept only)")
    
    return "\n".join(lines)
