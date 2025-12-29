"""
Augmentation Recommendation Engine.

This module is the "brain" of the augmentation system. It analyzes diagnostic
summaries across all responses and generates prioritized augmentation plans
that balance cost, benefit, and design quality improvement.
"""

from typing import List, Dict, Optional, Tuple
import numpy as np

from src.core.factors import Factor
from src.core.diagnostics import (
    DesignDiagnosticSummary,
    ResponseDiagnostics,
    DesignQualityReport,
    Issue
)
from src.core.augmentation.plan import (
    AugmentationPlan,
    FoldoverConfig,
    OptimalAugmentConfig,
    create_plan_id
)


def recommend_foldover_strategy(
    diagnostics: DesignDiagnosticSummary,
    fitted_models: Dict[str, object]
) -> Optional[AugmentationPlan]:
    """
    Recommend foldover strategy based on analysis results.
    
    Decision tree:
    1. If one main effect significant and aliased → single-factor foldover
    2. If multiple main effects significant and aliased → full foldover
    3. If no significant effects but low resolution → consider full foldover
    4. If budget very tight and one critical factor → single-factor
    
    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Complete diagnostic summary
    fitted_models : Dict[str, object]
        Fitted models for each response (name → model object)
    
    Returns
    -------
    AugmentationPlan or None
        Foldover plan if recommended, None otherwise
    
    Examples
    --------
    >>> plan = recommend_foldover_strategy(diagnostics, models)
    >>> if plan:
    ...     print(f"Recommend: {plan.plan_name}")
    ...     print(f"Runs to add: {plan.n_runs_to_add}")
    
    Notes
    -----
    Only recommends foldover for fractional factorial designs with
    aliasing issues. Will not recommend for response surface or
    optimal designs.
    """
    # Only applicable to fractional designs
    if diagnostics.design_type != 'fractional':
        return None
    
    if diagnostics.generators is None or len(diagnostics.generators) == 0:
        return None
    
    # Collect aliasing issues across all responses
    aliasing_issues = []
    for response_name, diag in diagnostics.response_diagnostics.items():
        for issue in diag.issues:
            if issue.severity in ('critical', 'warning') and issue.category == 'aliasing':
                aliasing_issues.append((response_name, issue, diag))
    
    if not aliasing_issues:
        return None
    
    # Identify significant aliased effects
    significant_aliased_factors = set()
    
    for response_name, issue, diag in aliasing_issues:
        for effect in diag.significant_effects:
            # Main effects only (single factor names)
            if effect in [f.name for f in diagnostics.factors]:
                if effect in diag.aliased_effects:
                    significant_aliased_factors.add(effect)
    
    # Decision logic
    n_runs_original = diagnostics.n_runs
    
    if len(significant_aliased_factors) == 0:
        # No significant aliased main effects, but low resolution
        # Consider full foldover if resolution ≤ III
        min_resolution = min(
            (diag.resolution for diag in diagnostics.response_diagnostics.values()
             if diag.resolution is not None),
            default=5
        )
        
        if min_resolution <= 3:
            # Full foldover for low-resolution designs
            return _create_full_foldover_plan(
                diagnostics, n_runs_original,
                reason="Low resolution design - recommend resolution upgrade"
            )
        
        return None
    
    elif len(significant_aliased_factors) == 1:
        # Single significant aliased factor → single-factor foldover
        factor_to_fold = list(significant_aliased_factors)[0]
        
        return _create_single_factor_foldover_plan(
            diagnostics, n_runs_original, factor_to_fold,
            reason=f"{factor_to_fold} is significant and aliased with 2FI"
        )
    
    else:
        # Multiple significant aliased factors → full foldover
        return _create_full_foldover_plan(
            diagnostics, n_runs_original,
            reason=f"{len(significant_aliased_factors)} significant effects aliased - full de-aliasing needed"
        )


def _create_full_foldover_plan(
    diagnostics: DesignDiagnosticSummary,
    n_runs_original: int,
    reason: str
) -> AugmentationPlan:
    """Create a full foldover augmentation plan."""
    config = FoldoverConfig(foldover_type='full')
    
    # Estimate resolution improvement
    current_resolution = min(
        (diag.resolution for diag in diagnostics.response_diagnostics.values()
         if diag.resolution is not None),
        default=3
    )
    expected_resolution = current_resolution + 1
    
    # All responses with aliasing issues benefit
    beneficiaries = [
        name for name, diag in diagnostics.response_diagnostics.items()
        if diag.resolution is not None and diag.resolution <= 4
    ]
    
    if not beneficiaries:
        beneficiaries = list(diagnostics.response_diagnostics.keys())
    
    plan = AugmentationPlan(
        plan_id=create_plan_id(),
        plan_name="Full Foldover",
        strategy='foldover',
        strategy_config=config,
        original_design=diagnostics.original_design,
        factors=diagnostics.factors,
        n_runs_to_add=n_runs_original,
        total_runs_after=n_runs_original * 2,
        expected_improvements={
            'resolution': f'{current_resolution} → {expected_resolution}',
            'aliasing': 'All main effects clear of 2-factor interactions'
        },
        benefits_responses=beneficiaries,
        primary_beneficiary=beneficiaries[0] if beneficiaries else '',
        experimental_cost=float(n_runs_original),
        utility_score=0.0,  # Will be set by rank_plans
        rank=1,
        metadata={
            'generators': diagnostics.generators,
            'reason': reason
        }
    )
    
    return plan


def _create_single_factor_foldover_plan(
    diagnostics: DesignDiagnosticSummary,
    n_runs_original: int,
    factor_to_fold: str,
    reason: str
) -> AugmentationPlan:
    """Create a single-factor foldover augmentation plan."""
    config = FoldoverConfig(
        foldover_type='single_factor',
        factor_to_fold=factor_to_fold,
        reason=reason
    )
    
    # Identify which responses benefit
    beneficiaries = []
    for name, diag in diagnostics.response_diagnostics.items():
        if factor_to_fold in diag.significant_effects:
            beneficiaries.append(name)
    
    if not beneficiaries:
        beneficiaries = list(diagnostics.response_diagnostics.keys())
    
    plan = AugmentationPlan(
        plan_id=create_plan_id(),
        plan_name=f"Single-Factor Foldover ({factor_to_fold})",
        strategy='foldover',
        strategy_config=config,
        original_design=diagnostics.original_design,
        factors=diagnostics.factors,
        n_runs_to_add=n_runs_original,
        total_runs_after=n_runs_original * 2,
        expected_improvements={
            'de_aliased': f'{factor_to_fold} clear of 2-factor interactions',
            'estimable': f'{factor_to_fold} 2FI now estimable'
        },
        benefits_responses=beneficiaries,
        primary_beneficiary=beneficiaries[0] if beneficiaries else '',
        experimental_cost=float(n_runs_original),
        utility_score=0.0,  # Will be set by rank_plans
        rank=1,
        metadata={
            'generators': diagnostics.generators,
            'factor_to_fold': factor_to_fold,
            'reason': reason
        }
    )
    
    return plan


def recommend_optimal_augmentation(
    diagnostics: DesignDiagnosticSummary,
    fitted_models: Dict[str, object]
) -> Optional[AugmentationPlan]:
    """
    Recommend D-optimal augmentation for model extension.
    
    Use cases:
    - Lack of fit detected → add quadratic terms
    - Linear model inadequate → upgrade to response surface
    - High prediction variance → add precision runs
    
    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Complete diagnostic summary
    fitted_models : Dict[str, object]
        Fitted models for each response
    
    Returns
    -------
    AugmentationPlan or None
        D-optimal plan if recommended, None otherwise
    
    Examples
    --------
    >>> plan = recommend_optimal_augmentation(diagnostics, models)
    >>> if plan:
    ...     print(f"Add {plan.n_runs_to_add} runs for model extension")
    
    Notes
    -----
    Recommends D-optimal augmentation when:
    1. Lack of fit p < 0.05
    2. R² < 0.70 (model inadequate)
    3. High prediction variance (max/mean > 3)
    """
    # Check for lack of fit or poor model fit
    lof_responses = []
    poor_fit_responses = []
    
    for response_name, diag in diagnostics.response_diagnostics.items():
        # Lack of fit
        if diag.lack_of_fit_p_value is not None and diag.lack_of_fit_p_value < 0.05:
            lof_responses.append(response_name)
        
        # Poor R²
        if diag.r_squared < 0.70:
            poor_fit_responses.append(response_name)
    
    needs_model_extension = len(lof_responses) > 0 or len(poor_fit_responses) > 0
    
    if not needs_model_extension:
        return None
    
    # Determine which terms to add
    # If continuous factors exist, add quadratic terms
    continuous_factors = [f for f in diagnostics.factors if f.is_continuous()]
    
    if len(continuous_factors) == 0:
        # Can't add quadratic terms without continuous factors
        return None
    
    # Build current model terms (simplified - assume linear)
    current_terms = ['1'] + [f.name for f in diagnostics.factors]
    
    # Extended model: add quadratic and interactions
    new_terms = current_terms.copy()
    
    # Add interactions
    factor_names = [f.name for f in diagnostics.factors]
    for i in range(len(factor_names)):
        for j in range(i + 1, len(factor_names)):
            new_terms.append(f"{factor_names[i]}*{factor_names[j]}")
    
    # Add quadratic terms for continuous factors
    for f in continuous_factors:
        new_terms.append(f"I({f.name}**2)")
    
    # Estimate runs needed: at least 1.5x the number of terms
    n_new_terms = len(new_terms)
    n_runs_to_add = max(8, int(n_new_terms * 1.5) - diagnostics.n_runs)
    
    # Beneficiaries
    beneficiaries = list(set(lof_responses + poor_fit_responses))
    if not beneficiaries:
        beneficiaries = list(diagnostics.response_diagnostics.keys())
    
    config = OptimalAugmentConfig(
        new_model_terms=new_terms,
        n_runs_to_add=n_runs_to_add,
        criterion='D'
    )
    
    plan = AugmentationPlan(
        plan_id=create_plan_id(),
        plan_name="D-Optimal Model Extension",
        strategy='d_optimal',
        strategy_config=config,
        original_design=diagnostics.original_design,
        factors=diagnostics.factors,
        n_runs_to_add=n_runs_to_add,
        total_runs_after=diagnostics.n_runs + n_runs_to_add,
        expected_improvements={
            'model_terms': f'{len(current_terms)} → {len(new_terms)}',
            'capability': 'Linear → Quadratic response surface',
            'r_squared': 'Expected improvement in model fit'
        },
        benefits_responses=beneficiaries,
        primary_beneficiary=beneficiaries[0] if beneficiaries else '',
        experimental_cost=float(n_runs_to_add),
        utility_score=0.0,  # Will be set by rank_plans
        rank=1,
        metadata={
            'current_model_terms': current_terms,
            'new_model_terms': new_terms,
            'reason': f"Lack of fit detected in {len(lof_responses)} response(s)" if lof_responses else "Poor model fit"
        }
    )
    
    return plan


def recommend_augmentation_plans(
    diagnostics: DesignDiagnosticSummary,
    fitted_models: Dict[str, object],
    budget_constraint: Optional[int] = None
) -> List[AugmentationPlan]:
    """
    Generate and rank augmentation plans based on diagnostics.
    
    This is the main entry point for the recommendation system.
    
    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Complete diagnostic summary across all responses
    fitted_models : Dict[str, object]
        Fitted models (name → model object)
    budget_constraint : int, optional
        Maximum number of additional runs allowed
    
    Returns
    -------
    List[AugmentationPlan]
        List of plans ranked by utility score (best first)
    
    Examples
    --------
    >>> plans = recommend_augmentation_plans(diagnostics, models, budget_constraint=20)
    >>> if plans:
    ...     best_plan = plans[0]
    ...     print(f"Best option: {best_plan.plan_name}")
    ...     print(f"Utility score: {best_plan.utility_score:.0f}/100")
    
    Notes
    -----
    Returns up to 3 plans, ordered by utility score. If no augmentation
    is needed, returns empty list.
    
    Plans are filtered by budget constraint if provided.
    """
    if not diagnostics.needs_any_augmentation():
        return []
    
    plans = []
    
    # 1. Check for foldover opportunities (aliasing)
    foldover_plan = recommend_foldover_strategy(diagnostics, fitted_models)
    if foldover_plan:
        plans.append(foldover_plan)
    
    # 2. Check for optimal augmentation opportunities (lack of fit)
    optimal_plan = recommend_optimal_augmentation(diagnostics, fitted_models)
    if optimal_plan:
        plans.append(optimal_plan)
    
    # 3. If both are recommended, create a combined plan
    if foldover_plan and optimal_plan:
        combined_plan = _create_combined_plan(
            diagnostics, foldover_plan, optimal_plan
        )
        if combined_plan:
            plans.append(combined_plan)
    
    # Filter by budget constraint
    if budget_constraint is not None:
        plans = [p for p in plans if p.n_runs_to_add <= budget_constraint]
    
    # Rank plans
    if plans:
        plans = rank_plans(plans, diagnostics)
    
    # Return top 3
    return plans[:3]


def _create_combined_plan(
    diagnostics: DesignDiagnosticSummary,
    foldover_plan: AugmentationPlan,
    optimal_plan: AugmentationPlan
) -> Optional[AugmentationPlan]:
    """
    Create a combined augmentation plan addressing both aliasing and LOF.
    
    Note: This is a placeholder for future enhancement. Full implementation
    would require sequential augmentation (foldover first, then optimal).
    """
    # For now, recommend addressing most critical issue first
    return None


def rank_plans(
    plans: List[AugmentationPlan],
    diagnostics: DesignDiagnosticSummary
) -> List[AugmentationPlan]:
    """
    Rank augmentation plans by utility score.
    
    Utility score (0-100) considers:
    - Severity of issues addressed (critical=100, warning=50)
    - Number of responses benefited
    - Cost (runs added)
    - Expected improvement magnitude
    
    Parameters
    ----------
    plans : List[AugmentationPlan]
        Plans to rank
    diagnostics : DesignDiagnosticSummary
        Diagnostic summary for context
    
    Returns
    -------
    List[AugmentationPlan]
        Plans sorted by utility score (descending)
    
    Examples
    --------
    >>> ranked = rank_plans(plans, diagnostics)
    >>> for i, plan in enumerate(ranked, 1):
    ...     print(f"{i}. {plan.plan_name}: {plan.utility_score:.0f}/100")
    
    Notes
    -----
    Utility formula:
        U = (severity_score × n_responses / cost) × 100
    
    Where:
    - severity_score = sum of issue severities addressed
    - n_responses = number of responses benefiting
    - cost = n_runs_to_add normalized by original design size
    """
    for plan in plans:
        # Calculate severity score
        severity_score = 0.0
        
        for response_name in plan.benefits_responses:
            if response_name in diagnostics.response_diagnostics:
                diag = diagnostics.response_diagnostics[response_name]
                for issue in diag.issues:
                    if issue.severity == 'critical':
                        severity_score += 100
                    elif issue.severity == 'warning':
                        severity_score += 50
                    else:
                        severity_score += 10
        
        # Number of responses benefited
        n_responses = len(plan.benefits_responses)
        if n_responses == 0:
            n_responses = 1
        
        # Cost factor (normalized by original design size)
        cost_normalized = plan.n_runs_to_add / max(diagnostics.n_runs, 1)
        
        # Utility score: benefit / cost
        # Higher severity × more responses = higher benefit
        # Higher cost = lower utility
        if cost_normalized > 0:
            utility = (severity_score * np.sqrt(n_responses)) / (cost_normalized * 100)
        else:
            utility = 0.0
        
        # Scale to 0-100
        utility = min(100.0, utility)
        
        plan.utility_score = float(utility)
    
    # Sort by utility (descending)
    ranked = sorted(plans, key=lambda p: p.utility_score, reverse=True)
    
    # Update ranks
    for i, plan in enumerate(ranked, 1):
        plan.rank = i
    
    return ranked


def resolve_multi_response_conflicts(
    diagnostics: DesignDiagnosticSummary
) -> str:
    """
    Generate recommendation when responses have conflicting needs.
    
    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Diagnostic summary
    
    Returns
    -------
    str
        Recommendation text for handling conflicts
    
    Examples
    --------
    >>> if diagnostics.conflicting_recommendations:
    ...     advice = resolve_multi_response_conflicts(diagnostics)
    ...     print(advice)
    
    Notes
    -----
    Conflict resolution strategy:
    1. Address critical issues first (regardless of response)
    2. If multiple critical issues, prioritize aliasing over LOF
    3. If budget allows, sequential augmentation
    4. If budget tight, choose highest-severity issue
    """
    if not diagnostics.conflicting_recommendations:
        return "No conflicts - single augmentation strategy applies"
    
    # Identify issue categories across responses
    categories = set()
    critical_issues = []
    
    for response_name, diag in diagnostics.response_diagnostics.items():
        for issue in diag.issues:
            categories.add(issue.category)
            if issue.severity == 'critical':
                critical_issues.append((response_name, issue))
    
    # Priority ranking
    priority_map = {
        'aliasing': 1,
        'lack_of_fit': 2,
        'precision': 3,
        'estimability': 4,
        'other': 5
    }
    
    if critical_issues:
        # Sort by priority
        critical_issues.sort(key=lambda x: priority_map.get(x[1].category, 99))
        
        top_issue = critical_issues[0]
        response_name, issue = top_issue
        
        return (
            f"Multiple critical issues detected. Recommended approach:\n"
            f"1. Address {issue.category} first ({response_name}: {issue.description})\n"
            f"2. Re-analyze after augmentation\n"
            f"3. Address remaining issues in subsequent augmentation if needed\n\n"
            f"Rationale: {issue.category} is highest priority and affects {response_name}."
        )
    
    # No critical issues, but warnings conflict
    return (
        f"Multiple responses have different improvement needs. "
        f"Consider sequential augmentation:\n"
        f"1. Choose augmentation that benefits most responses\n"
        f"2. Re-analyze after initial augmentation\n"
        f"3. Apply targeted augmentation for remaining issues"
    )


def assess_augmentation_necessity(
    diagnostics: DesignDiagnosticSummary
) -> Tuple[bool, str]:
    """
    Assess whether augmentation is truly necessary.
    
    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Diagnostic summary
    
    Returns
    -------
    is_necessary : bool
        Whether augmentation is recommended
    reason : str
        Explanation of assessment
    
    Examples
    --------
    >>> necessary, reason = assess_augmentation_necessity(diagnostics)
    >>> if necessary:
    ...     print(f"Augmentation needed: {reason}")
    
    Notes
    -----
    Returns True if:
    - Any response has critical issues
    - Multiple responses have warnings
    - Design quality is "Poor" or "Inadequate"
    """
    critical_count = 0
    warning_count = 0
    poor_responses = []
    
    for response_name, diag in diagnostics.response_diagnostics.items():
        for issue in diag.issues:
            if issue.severity == 'critical':
                critical_count += 1
            elif issue.severity == 'warning':
                warning_count += 1
        
        if diag.r_squared < 0.70:
            poor_responses.append(response_name)
    
    if critical_count > 0:
        return True, f"{critical_count} critical issue(s) require augmentation"
    
    if warning_count >= 2:
        return True, f"{warning_count} warning(s) suggest augmentation would improve design quality"
    
    if poor_responses:
        return True, f"{len(poor_responses)} response(s) have poor model fit (R² < 0.70)"
    
    return False, "Design quality is acceptable - augmentation optional"