"""
Unified Augmentation Interface.

This module provides the single entry point for augmentation,
supporting both Mode A (diagnostics-driven) and Mode B (goal-driven).
"""

from typing import List, Dict, Optional, Literal
from dataclasses import dataclass

from src.core.diagnostics import DesignDiagnosticSummary
from src.core.augmentation.plan import AugmentationPlan
from src.core.augmentation.recommendations import recommend_from_diagnostics
from src.core.augmentation.goal_driven import (
    recommend_from_goal,
    GoalDrivenContext
)
from src.core.augmentation.goals import (
    AugmentationGoal,
    get_available_goals,
    GOAL_CATALOG
)


@dataclass
class AugmentationRequest:
    """
    Request for augmentation recommendations.
    
    Attributes
    ----------
    mode : {'fix_issues', 'enhance_design'}
        Augmentation mode
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics
    selected_goal : AugmentationGoal, optional
        User's goal (required for 'enhance_design' mode)
    budget_constraint : int, optional
        Maximum runs to add
    user_adjustments : dict, optional
        User-requested parameter adjustments
    """
    
    mode: Literal['fix_issues', 'enhance_design']
    diagnostics: DesignDiagnosticSummary
    selected_goal: Optional[AugmentationGoal] = None
    budget_constraint: Optional[int] = None
    user_adjustments: Optional[Dict] = None


def recommend_augmentation(request: AugmentationRequest) -> List[AugmentationPlan]:
    """
    Generate augmentation recommendations.
    
    This is the unified entry point that routes to either:
    - Mode A: Diagnostics-driven (fix detected issues)
    - Mode B: Goal-driven (user intent)
    
    Parameters
    ----------
    request : AugmentationRequest
        Augmentation request specifying mode and parameters
    
    Returns
    -------
    List[AugmentationPlan]
        Ranked augmentation plans
    
    Raises
    ------
    ValueError
        If mode is invalid or required parameters are missing
    
    Examples
    --------
    Mode A (Fix Issues):
    >>> request = AugmentationRequest(
    ...     mode='fix_issues',
    ...     diagnostics=diagnostics,
    ...     budget_constraint=16
    ... )
    >>> plans = recommend_augmentation(request)
    
    Mode B (Enhance Design):
    >>> request = AugmentationRequest(
    ...     mode='enhance_design',
    ...     diagnostics=diagnostics,
    ...     selected_goal=AugmentationGoal.MODEL_CURVATURE,
    ...     budget_constraint=20
    ... )
    >>> plans = recommend_augmentation(request)
    """
    
    if request.mode == 'fix_issues':
        # Mode A: Diagnostics-driven
        return recommend_from_diagnostics(
            diagnostics=request.diagnostics,
            budget_constraint=request.budget_constraint
        )
    
    elif request.mode == 'enhance_design':
        # Mode B: Goal-driven
        if request.selected_goal is None:
            raise ValueError(
                "selected_goal is required for 'enhance_design' mode"
            )
        
        context = GoalDrivenContext(
            selected_goal=request.selected_goal,
            design_diagnostics=request.diagnostics,
            budget_constraint=request.budget_constraint,
            user_adjustments=request.user_adjustments or {}
        )
        
        return recommend_from_goal(context)
    
    else:
        raise ValueError(f"Invalid mode: {request.mode}")


def get_mode_availability(
    diagnostics: DesignDiagnosticSummary
) -> Dict[str, bool]:
    """
    Determine which augmentation modes are available.
    
    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics
    
    Returns
    -------
    Dict[str, bool]
        Mode availability:
        - 'fix_issues': bool (True if issues detected)
        - 'enhance_design': bool (always True)
    
    Examples
    --------
    >>> availability = get_mode_availability(diagnostics)
    >>> if availability['fix_issues']:
    ...     print("Mode A available: Issues detected")
    >>> if availability['enhance_design']:
    ...     print("Mode B available: Can enhance design")
    """
    
    # Mode A: Available if issues detected
    has_issues = diagnostics.needs_any_augmentation()
    
    # Mode B: Always available
    can_enhance = True
    
    return {
        'fix_issues': has_issues,
        'enhance_design': can_enhance
    }


def get_mode_recommendations(
    diagnostics: DesignDiagnosticSummary
) -> Dict[str, str]:
    """
    Get user-facing recommendations for which mode to use.
    
    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics
    
    Returns
    -------
    Dict[str, str]
        Mode recommendations with descriptions
    
    Examples
    --------
    >>> recommendations = get_mode_recommendations(diagnostics)
    >>> print(recommendations['fix_issues'])
    "Critical aliasing detected - strongly recommend addressing first"
    """
    
    recommendations = {}
    
    # Analyze diagnostic severity
    has_critical = any(
        any(i.severity == 'critical' for i in diag.issues)
        for diag in diagnostics.response_diagnostics.values()
    )
    
    has_warnings = any(
        any(i.severity == 'warning' for i in diag.issues)
        for diag in diagnostics.response_diagnostics.values()
    )
    
    # Mode A recommendation
    if has_critical:
        recommendations['fix_issues'] = (
            "⚠️ **Strongly Recommended**: Critical issues detected that should be addressed."
        )
    elif has_warnings:
        recommendations['fix_issues'] = (
            "⚡ **Suggested**: Warnings detected - consider addressing before proceeding."
        )
    else:
        recommendations['fix_issues'] = (
            "✅ **Not Needed**: Design quality is satisfactory."
        )
    
    # Mode B recommendation
    if has_critical:
        recommendations['enhance_design'] = (
            "💡 **Available**: You can enhance the design, but fixing critical issues first is recommended."
        )
    else:
        recommendations['enhance_design'] = (
            "🎯 **Ready**: Design is healthy - choose a goal to enhance capabilities."
        )
    
    return recommendations


def get_available_enhancement_goals(
    diagnostics: DesignDiagnosticSummary
) -> List[Dict[str, str]]:
    """
    Get available enhancement goals for Mode B.
    
    Returns goals appropriate for the current design type,
    formatted for UI display.
    
    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics
    
    Returns
    -------
    List[Dict[str, str]]
        Goal information for UI display:
        - 'goal': goal enum value
        - 'title': display title
        - 'description': full description
        - 'when_appropriate': guidance text
        - 'example_scenario': concrete example
        - 'diagnostic_alignment': bonus info if diagnostics support this
    
    Examples
    --------
    >>> goals = get_available_enhancement_goals(diagnostics)
    >>> for goal in goals:
    ...     print(f"{goal['title']}: {goal['description']}")
    """
    
    # Get available goals
    available = get_available_goals(
        current_design_type=diagnostics.design_type,
        has_replicates=_check_has_replicates(diagnostics),
        has_center_points=diagnostics.has_center_points,
        is_fractional=(diagnostics.design_type == 'fractional')
    )
    
    # Format for UI
    goal_info = []
    for goal in available:
        desc = GOAL_CATALOG[goal]
        
        # Check diagnostic alignment
        alignment = _check_goal_alignment(goal, diagnostics)
        
        goal_info.append({
            'goal': goal.value,
            'title': desc.title,
            'description': desc.description,
            'typical_strategies': ', '.join(desc.typical_strategies),
            'when_appropriate': desc.when_appropriate,
            'example_scenario': desc.example_scenario,
            'diagnostic_alignment': alignment
        })
    
    return goal_info


def _check_has_replicates(diagnostics: DesignDiagnosticSummary) -> bool:
    """Check if design has replicate runs."""
    
    # Count unique run combinations
    design = diagnostics.original_design
    factor_cols = [f.name for f in diagnostics.factors]
    
    if not factor_cols:
        return False
    
    n_unique = design[factor_cols].drop_duplicates().shape[0]
    
    # If fewer unique than total, we have replicates
    return n_unique < len(design)


def _check_goal_alignment(
    goal: AugmentationGoal,
    diagnostics: DesignDiagnosticSummary
) -> str:
    """
    Check if diagnostics suggest this goal is particularly appropriate.
    
    Returns
    -------
    str
        Alignment message, or empty string if no special alignment
    """
    
    # Check for aliasing
    has_aliasing = any(
        diag.resolution and diag.resolution <= 4
        for diag in diagnostics.response_diagnostics.values()
    )
    
    if goal == AugmentationGoal.REDUCE_ALIASING and has_aliasing:
        return "✨ Diagnostics detected aliasing - this goal is highly relevant"
    
    # Check for lack of fit
    has_lof = any(
        diag.lack_of_fit_p_value and diag.lack_of_fit_p_value < 0.05
        for diag in diagnostics.response_diagnostics.values()
    )
    
    if goal == AugmentationGoal.MODEL_CURVATURE and has_lof:
        return "✨ Diagnostics detected lack of fit - curvature modeling recommended"
    
    # Check for high prediction variance
    has_high_var = any(
        diag.prediction_variance_stats and 
        diag.prediction_variance_stats.get('max', 0) > 3 * diag.prediction_variance_stats.get('mean', 1)
        for diag in diagnostics.response_diagnostics.values()
    )
    
    if goal == AugmentationGoal.IMPROVE_PREDICTION and has_high_var:
        return "✨ Diagnostics show uneven prediction variance - this goal is highly relevant"
    
    # Check for low R-squared (might benefit from more runs)
    has_low_r2 = any(
        diag.r_squared < 0.70
        for diag in diagnostics.response_diagnostics.values()
    )
    
    if goal == AugmentationGoal.INCREASE_CONFIDENCE and has_low_r2:
        return "💡 Model fit is marginal - additional runs may help"
    
    return ""


def create_plan_comparison_table(
    plans: List[AugmentationPlan]
) -> List[Dict]:
    """
    Create comparison table data for UI display.
    
    Parameters
    ----------
    plans : List[AugmentationPlan]
        Plans to compare
    
    Returns
    -------
    List[Dict]
        Table rows for display
    
    Examples
    --------
    >>> table = create_plan_comparison_table(plans)
    >>> df = pd.DataFrame(table)
    >>> st.dataframe(df)
    """
    
    rows = []
    
    for plan in plans:
        row = {
            'Rank': plan.rank,
            'Plan': plan.plan_name,
            'Strategy': plan.strategy.replace('_', ' ').title(),
            'Runs to Add': plan.n_runs_to_add,
            'Total After': plan.total_runs_after,
            'Utility': f"{plan.utility_score:.0f}/100",
            'Mode': plan.metadata.get('mode', 'unknown').replace('_', ' ').title()
        }
        
        rows.append(row)
    
    return rows
