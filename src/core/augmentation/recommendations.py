"""
Augmentation Recommendation Engine.

This module is the core of Mode A (diagnostics-driven augmentation),
analyzing design quality issues and recommending targeted fixes.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from src.core.diagnostics import DesignDiagnosticSummary, Issue
from src.core.augmentation.plan import (
    AugmentationPlan,
    FoldoverConfig,
    OptimalAugmentConfig,
    create_plan_id
)


@dataclass
class DiagnosticAugmentationRecommendation:
    """
    Recommendation for addressing a specific diagnostic issue.
    
    Attributes
    ----------
    issue : Issue
        The diagnostic issue being addressed
    strategy : str
        Augmentation strategy ('foldover', 'd_optimal', etc.)
    n_runs_to_add : int
        Number of runs to add
    rationale : str
        Why this strategy addresses this issue
    priority : int
        Priority ranking (1=highest)
    expected_improvement : str
        Expected improvement to metrics
    """
    
    issue: Issue
    strategy: str
    n_runs_to_add: int
    rationale: str
    priority: int
    expected_improvement: str


def recommend_from_diagnostics(
    diagnostics: DesignDiagnosticSummary,
    budget_constraint: Optional[int] = None
) -> List[AugmentationPlan]:
    """
    Generate augmentation plans to fix detected issues (Mode A).
    
    This is the diagnostics-driven mode: automatically detect problems
    and recommend targeted fixes.
    
    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Complete design diagnostics
    budget_constraint : int, optional
        Maximum number of runs to add (None = no limit)
    
    Returns
    -------
    List[AugmentationPlan]
        Ranked augmentation plans addressing detected issues
    
    Notes
    -----
    Priority order for addressing issues:
    1. Aliasing/confounding (Resolution III→IV)
    2. Insufficient rank / estimability
    3. Lack of curvature detection (no center points)
    4. Poor prediction variance
    5. Missing pure error
    6. Low efficiency
    
    Examples
    --------
    >>> plans = recommend_from_diagnostics(diagnostics, budget_constraint=16)
    >>> for plan in plans:
    ...     print(f"{plan.plan_name}: +{plan.n_runs_to_add} runs")
    """
    
    # Collect all issues across responses
    all_issues = _collect_and_prioritize_issues(diagnostics)
    
    if not all_issues:
        # No issues detected - return empty list
        return []
    
    # Generate recommendations for each issue
    recommendations = []
    for issue in all_issues:
        rec = _recommend_for_issue(issue, diagnostics)
        if rec:
            recommendations.append(rec)
    
    # Convert recommendations to plans
    plans = []
    for rec in recommendations:
        plan = _recommendation_to_plan(rec, diagnostics)
        if plan:
            # Apply budget constraint
            if budget_constraint is None or plan.n_runs_to_add <= budget_constraint:
                plans.append(plan)
    
    # Rank plans by priority and utility
    plans = _rank_plans(plans)
    
    return plans


def _collect_and_prioritize_issues(
    diagnostics: DesignDiagnosticSummary
) -> List[Issue]:
    """
    Collect all issues and prioritize by severity and category.
    
    Returns
    -------
    List[Issue]
        Issues sorted by priority (highest first)
    """
    all_issues = []
    
    for response_name, diag in diagnostics.response_diagnostics.items():
        for issue in diag.issues:
            all_issues.append(issue)
    
    # Priority order
    priority_map = {
        'aliasing': 1,
        'estimability': 2,
        'lack_of_fit': 3,
        'precision': 4,
        'pure_error': 5,
        'efficiency': 6
    }
    
    severity_map = {
        'critical': 1,
        'warning': 2,
        'info': 3
    }
    
    # Sort by severity first, then category priority
    all_issues.sort(
        key=lambda iss: (severity_map.get(iss.severity, 99), 
                        priority_map.get(iss.category, 99))
    )
    
    return all_issues


def _recommend_for_issue(
    issue: Issue,
    diagnostics: DesignDiagnosticSummary
) -> Optional[DiagnosticAugmentationRecommendation]:
    """
    Recommend augmentation strategy for a specific issue.
    
    Returns
    -------
    DiagnosticAugmentationRecommendation or None
        Recommendation, or None if no augmentation can address this issue
    """
    
    # Issue: Aliasing
    if issue.category == 'aliasing':
        return _recommend_dealias(issue, diagnostics)
    
    # Issue: Lack of fit
    elif issue.category == 'lack_of_fit':
        return _recommend_lof_fix(issue, diagnostics)
    
    # Issue: Precision (high prediction variance)
    elif issue.category == 'precision':
        return _recommend_precision_improvement(issue, diagnostics)
    
    # Issue: Estimability (collinearity, rank deficiency)
    elif issue.category == 'estimability':
        return _recommend_estimability_fix(issue, diagnostics)
    
    # Issue: Missing pure error
    elif issue.category == 'pure_error':
        return _recommend_pure_error(issue, diagnostics)
    
    else:
        return None


def _recommend_dealias(
    issue: Issue,
    diagnostics: DesignDiagnosticSummary
) -> Optional[DiagnosticAugmentationRecommendation]:
    """Recommend foldover to break alias chains."""
    
    # Determine foldover type
    if len(issue.affected_terms) == 1:
        # Single critical effect - single-factor foldover
        factor_to_fold = issue.affected_terms[0]
        strategy = 'single_factor_foldover'
        n_runs = diagnostics.n_runs  # Same as original
        rationale = (
            f"Single-factor foldover on {factor_to_fold} breaks critical alias "
            f"and confirms which effect is real."
        )
        expected_improvement = f"De-alias {factor_to_fold} from 2FI"
        
    else:
        # Multiple aliased effects - full foldover
        strategy = 'full_foldover'
        n_runs = diagnostics.n_runs  # Double the design
        rationale = (
            "Full foldover breaks all alias chains, increasing resolution "
            "and clarifying all effects."
        )
        
        # Estimate resolution improvement
        current_res = diagnostics.response_diagnostics[
            list(diagnostics.response_diagnostics.keys())[0]
        ].resolution or 3
        new_res = current_res + 1
        expected_improvement = f"Resolution {current_res} → {new_res}"
    
    return DiagnosticAugmentationRecommendation(
        issue=issue,
        strategy=strategy,
        n_runs_to_add=n_runs,
        rationale=rationale,
        priority=1,  # Highest priority
        expected_improvement=expected_improvement
    )


def _recommend_lof_fix(
    issue: Issue,
    diagnostics: DesignDiagnosticSummary
) -> Optional[DiagnosticAugmentationRecommendation]:
    """Recommend adding quadratic terms to address lack of fit."""
    
    # Check if center points already present
    if diagnostics.has_center_points:
        # Already have curvature detection - add full quadratic terms
        strategy = 'd_optimal_quadratic'
        
        # Estimate runs needed for quadratic model
        k = diagnostics.n_factors
        n_terms_quadratic = 1 + k + k * (k - 1) // 2 + k  # Intercept + main + 2FI + quadratic
        n_runs = max(n_terms_quadratic + 4, 2 * k)  # Overspecify model
        
        rationale = (
            "Add D-optimal runs to estimate quadratic terms, "
            "addressing observed lack of fit."
        )
        expected_improvement = "Capture curvature, reduce LOF p-value"
        
    else:
        # No center points - add center points first
        strategy = 'center_points'
        n_runs = 3  # 3 center points typical
        
        rationale = (
            "Add center points to test for curvature. "
            "If curvature is confirmed, further augmentation with "
            "axial points or quadratic terms may be needed."
        )
        expected_improvement = "Enable curvature detection via LOF test"
    
    return DiagnosticAugmentationRecommendation(
        issue=issue,
        strategy=strategy,
        n_runs_to_add=n_runs,
        rationale=rationale,
        priority=2,
        expected_improvement=expected_improvement
    )


def _recommend_precision_improvement(
    issue: Issue,
    diagnostics: DesignDiagnosticSummary
) -> Optional[DiagnosticAugmentationRecommendation]:
    """Recommend I-optimal augmentation for prediction variance."""
    
    strategy = 'i_optimal'
    
    # Estimate runs needed
    k = diagnostics.n_factors
    n_runs = min(2 * k, 20)  # 1-2x factors, cap at 20
    
    rationale = (
        "Add I-optimal runs to reduce prediction variance in high-variance regions, "
        "improving prediction uniformity across the design space."
    )
    
    expected_improvement = "Reduce max/mean prediction variance ratio"
    
    return DiagnosticAugmentationRecommendation(
        issue=issue,
        strategy=strategy,
        n_runs_to_add=n_runs,
        rationale=rationale,
        priority=3,
        expected_improvement=expected_improvement
    )


def _recommend_estimability_fix(
    issue: Issue,
    diagnostics: DesignDiagnosticSummary
) -> Optional[DiagnosticAugmentationRecommendation]:
    """Recommend D-optimal augmentation to fix collinearity."""
    
    strategy = 'd_optimal'
    
    # Estimate runs needed
    n_problematic = len(issue.affected_terms)
    n_runs = max(n_problematic + 2, diagnostics.n_factors)
    
    rationale = (
        f"Add D-optimal runs to orthogonalize {', '.join(issue.affected_terms)}, "
        "reducing collinearity and improving estimability."
    )
    
    expected_improvement = f"Reduce VIF for {', '.join(issue.affected_terms[:2])}"
    
    return DiagnosticAugmentationRecommendation(
        issue=issue,
        strategy=strategy,
        n_runs_to_add=n_runs,
        rationale=rationale,
        priority=4,
        expected_improvement=expected_improvement
    )


def _recommend_pure_error(
    issue: Issue,
    diagnostics: DesignDiagnosticSummary
) -> Optional[DiagnosticAugmentationRecommendation]:
    """Recommend replicates to obtain pure error estimate."""
    
    strategy = 'replicates'
    
    # Replicate 25-50% of design
    n_runs = max(3, diagnostics.n_runs // 4)
    
    rationale = (
        "Add replicate runs to obtain pure error estimate, "
        "enabling formal lack-of-fit testing."
    )
    
    expected_improvement = "Enable LOF F-test"
    
    return DiagnosticAugmentationRecommendation(
        issue=issue,
        strategy=strategy,
        n_runs_to_add=n_runs,
        rationale=rationale,
        priority=5,
        expected_improvement=expected_improvement
    )


def _recommendation_to_plan(
    rec: DiagnosticAugmentationRecommendation,
    diagnostics: DesignDiagnosticSummary
) -> Optional[AugmentationPlan]:
    """
    Convert a recommendation to an executable plan.
    
    Returns
    -------
    AugmentationPlan or None
        Executable plan, or None if conversion fails
    """
    
    # Create strategy config
    if rec.strategy in ('full_foldover', 'single_factor_foldover'):
        if rec.strategy == 'full_foldover':
            config = FoldoverConfig(foldover_type='full')
        else:
            # Extract factor from affected terms
            factor_to_fold = rec.issue.affected_terms[0] if rec.issue.affected_terms else None
            config = FoldoverConfig(
                foldover_type='single_factor',
                factor_to_fold=factor_to_fold,
                reason=rec.rationale
            )
        
        strategy_type = 'foldover'
        
    elif rec.strategy in ('d_optimal', 'd_optimal_quadratic', 'i_optimal'):
        # Determine model terms to add
        if rec.strategy == 'd_optimal_quadratic':
            # Add all quadratic terms
            factor_names = [f.name for f in diagnostics.factors]
            new_terms = [f"{f}^2" for f in factor_names]
        else:
            # Use affected terms as new model terms
            new_terms = rec.issue.affected_terms if rec.issue.affected_terms else []
        
        config = OptimalAugmentConfig(
            new_model_terms=new_terms,
            n_runs_to_add=rec.n_runs_to_add,
            criterion='D'  # TODO: support I-optimal
        )
        
        strategy_type = 'd_optimal'
        
    else:
        # Other strategies not yet implemented
        return None
    
    # Create plan
    plan = AugmentationPlan(
        plan_id=create_plan_id(),
        plan_name=_generate_plan_name(rec),
        strategy=strategy_type,
        strategy_config=config,
        original_design=diagnostics.original_design,
        factors=diagnostics.factors,
        n_runs_to_add=rec.n_runs_to_add,
        total_runs_after=diagnostics.n_runs + rec.n_runs_to_add,
        expected_improvements={
            rec.issue.category: rec.expected_improvement
        },
        benefits_responses=_get_affected_responses(rec.issue, diagnostics),
        primary_beneficiary=_get_primary_beneficiary(rec.issue, diagnostics),
        experimental_cost=float(rec.n_runs_to_add),  # Simplified cost model
        utility_score=_compute_utility_score(rec, diagnostics),
        rank=rec.priority,
        metadata={
            'issue_category': rec.issue.category,
            'issue_severity': rec.issue.severity,
            'mode': 'diagnostics_driven'
        }
    )
    
    return plan


def _generate_plan_name(rec: DiagnosticAugmentationRecommendation) -> str:
    """Generate human-readable plan name."""
    
    strategy_names = {
        'full_foldover': 'Full Foldover',
        'single_factor_foldover': 'Single-Factor Foldover',
        'd_optimal': 'D-Optimal Augmentation',
        'd_optimal_quadratic': 'Add Quadratic Terms',
        'i_optimal': 'I-Optimal Augmentation',
        'center_points': 'Add Center Points',
        'replicates': 'Add Replicates'
    }
    
    base_name = strategy_names.get(rec.strategy, rec.strategy.title())
    
    # Add context if single-factor foldover
    if rec.strategy == 'single_factor_foldover' and rec.issue.affected_terms:
        base_name += f" on {rec.issue.affected_terms[0]}"
    
    return base_name


def _get_affected_responses(
    issue: Issue,
    diagnostics: DesignDiagnosticSummary
) -> List[str]:
    """Get list of responses affected by this issue."""
    
    affected = []
    for response_name, diag in diagnostics.response_diagnostics.items():
        if issue in diag.issues:
            affected.append(response_name)
    
    return affected if affected else ['All']


def _get_primary_beneficiary(
    issue: Issue,
    diagnostics: DesignDiagnosticSummary
) -> str:
    """Get primary response that benefits from fixing this issue."""
    
    affected = _get_affected_responses(issue, diagnostics)
    
    if len(affected) == 1:
        return affected[0]
    elif affected:
        return affected[0]  # First one
    else:
        return 'All'


def _compute_utility_score(
    rec: DiagnosticAugmentationRecommendation,
    diagnostics: DesignDiagnosticSummary
) -> float:
    """
    Compute utility score (0-100) for this recommendation.
    
    Higher score = better benefit/cost ratio.
    """
    
    # Base score by severity
    severity_scores = {
        'critical': 90.0,
        'warning': 60.0,
        'info': 30.0
    }
    
    base_score = severity_scores.get(rec.issue.severity, 50.0)
    
    # Adjust for cost (fewer runs = higher utility)
    cost_factor = 1.0 - min(rec.n_runs_to_add / diagnostics.n_runs, 0.5)
    
    # Adjust for priority
    priority_factor = 1.0 - (rec.priority - 1) * 0.1
    
    utility = base_score * cost_factor * priority_factor
    
    return max(0.0, min(100.0, utility))


def _rank_plans(plans: List[AugmentationPlan]) -> List[AugmentationPlan]:
    """
    Rank plans by utility score and assign rank numbers.
    
    Returns
    -------
    List[AugmentationPlan]
        Plans sorted by utility (highest first)
    """
    
    # Sort by utility descending
    plans.sort(key=lambda p: p.utility_score, reverse=True)
    
    # Assign ranks
    for i, plan in enumerate(plans, 1):
        plan.rank = i
    
    return plans
