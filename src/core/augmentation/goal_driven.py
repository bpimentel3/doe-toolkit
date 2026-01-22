"""
Goal-Driven Augmentation Planning (Mode B).

This module translates user intentions into augmentation plans,
using diagnostics to inform but not block user choices.
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from src.core.diagnostics import DesignDiagnosticSummary
from src.core.augmentation.goals import (
    AugmentationGoal,
    get_strategies_for_goal,
    GOAL_CATALOG
)
from src.core.augmentation.plan import (
    AugmentationPlan,
    FoldoverConfig,
    OptimalAugmentConfig,
    create_plan_id
)


@dataclass
class GoalDrivenContext:
    """
    Context information for goal-driven augmentation.
    
    Attributes
    ----------
    selected_goal : AugmentationGoal
        User's selected intention
    design_diagnostics : DesignDiagnosticSummary
        Current design diagnostics (informative, not prescriptive)
    budget_constraint : int, optional
        Maximum runs to add
    user_adjustments : dict
        User-requested adjustments (run count, regions, etc.)
    """
    
    selected_goal: AugmentationGoal
    design_diagnostics: DesignDiagnosticSummary
    budget_constraint: Optional[int] = None
    user_adjustments: Dict = None
    
    def __post_init__(self):
        if self.user_adjustments is None:
            self.user_adjustments = {}


def recommend_from_goal(
    context: GoalDrivenContext
) -> List[AugmentationPlan]:
    """
    Generate augmentation plans based on user's stated goal (Mode B).
    
    This is the intent-driven mode: user selects what they want to achieve,
    and we recommend strategies to accomplish it. Diagnostics inform the
    recommendations but don't block execution.
    
    Parameters
    ----------
    context : GoalDrivenContext
        User's goal and current design context
    
    Returns
    -------
    List[AugmentationPlan]
        Augmentation plans addressing the user's goal
    
    Notes
    -----
    Unlike Mode A (diagnostics-driven), Mode B:
    - Starts with user intention, not detected problems
    - Uses diagnostics to size/optimize the augmentation
    - Provides warnings but doesn't prevent execution
    - Offers parameter adjustment (run count, regions)
    
    Examples
    --------
    >>> context = GoalDrivenContext(
    ...     selected_goal=AugmentationGoal.MODEL_CURVATURE,
    ...     design_diagnostics=diagnostics,
    ...     budget_constraint=20
    ... )
    >>> plans = recommend_from_goal(context)
    >>> for plan in plans:
    ...     print(f"{plan.plan_name}: {plan.n_runs_to_add} runs")
    """
    
    # Get strategy mapping for this goal
    strategy_mapping = get_strategies_for_goal(
        goal=context.selected_goal,
        current_design_type=context.design_diagnostics.design_type,
        n_factors=context.design_diagnostics.n_factors,
        current_runs=context.design_diagnostics.n_runs
    )
    
    # Check if goal is applicable
    if strategy_mapping.primary_strategy == 'not_applicable':
        return []
    
    # Generate primary plan
    primary_plan = _generate_goal_plan(
        goal=context.selected_goal,
        strategy_name=strategy_mapping.primary_strategy,
        strategy_mapping=strategy_mapping,
        context=context,
        is_primary=True
    )
    
    plans = [primary_plan] if primary_plan else []
    
    # Generate alternative plans
    for alt_strategy in strategy_mapping.alternative_strategies:
        alt_plan = _generate_goal_plan(
            goal=context.selected_goal,
            strategy_name=alt_strategy,
            strategy_mapping=strategy_mapping,
            context=context,
            is_primary=False
        )
        if alt_plan:
            plans.append(alt_plan)
    
    # Use diagnostics to refine recommendations
    plans = _refine_with_diagnostics(plans, context.design_diagnostics)
    
    # Apply budget constraint
    if context.budget_constraint:
        plans = [p for p in plans if p.n_runs_to_add <= context.budget_constraint]
    
    # Apply user adjustments
    plans = _apply_user_adjustments(plans, context.user_adjustments)
    
    # Rank plans
    plans = _rank_goal_plans(plans, context)
    
    return plans


def _generate_goal_plan(
    goal: AugmentationGoal,
    strategy_name: str,
    strategy_mapping,
    context: GoalDrivenContext,
    is_primary: bool
) -> Optional[AugmentationPlan]:
    """
    Generate a single plan for a goal+strategy combination.
    
    Returns
    -------
    AugmentationPlan or None
        Plan if strategy is implementable, None otherwise
    """
    
    diagnostics = context.design_diagnostics
    
    # Determine number of runs (use strategy recommendation or user override)
    if 'n_runs' in context.user_adjustments:
        n_runs = context.user_adjustments['n_runs']
    else:
        # Use midpoint of estimated range
        n_runs = (strategy_mapping.estimated_min_runs + 
                 strategy_mapping.estimated_max_runs) // 2
    
    # Create strategy config based on strategy name
    config = _create_strategy_config(
        strategy_name=strategy_name,
        goal=goal,
        n_runs=n_runs,
        diagnostics=diagnostics,
        user_adjustments=context.user_adjustments
    )
    
    if config is None:
        return None
    
    strategy_type = _get_strategy_type(strategy_name)
    
    # Get goal description
    goal_desc = GOAL_CATALOG[goal]
    
    # Create plan
    plan = AugmentationPlan(
        plan_id=create_plan_id(),
        plan_name=_generate_goal_plan_name(goal, strategy_name, is_primary),
        strategy=strategy_type,
        strategy_config=config,
        original_design=diagnostics.original_design,
        factors=diagnostics.factors,
        n_runs_to_add=n_runs,
        total_runs_after=diagnostics.n_runs + n_runs,
        expected_improvements=_predict_improvements(
            strategy_name, goal, diagnostics, n_runs
        ),
        benefits_responses=['All'],  # Goal-driven benefits all responses
        primary_beneficiary='All',
        experimental_cost=float(n_runs),
        utility_score=_compute_goal_utility(
            goal, strategy_name, n_runs, diagnostics, is_primary
        ),
        rank=1 if is_primary else 2,
        metadata={
            'goal': goal.value,
            'goal_title': goal_desc.title,
            'mode': 'goal_driven',
            'is_primary_strategy': is_primary,
            'strategy_rationale': strategy_mapping.strategy_rationale,
            'estimated_min_runs': strategy_mapping.estimated_min_runs,
            'estimated_max_runs': strategy_mapping.estimated_max_runs
        }
    )
    
    return plan


def _create_strategy_config(
    strategy_name: str,
    goal: AugmentationGoal,
    n_runs: int,
    diagnostics: DesignDiagnosticSummary,
    user_adjustments: Dict
):
    """
    Create strategy-specific configuration object.
    
    Returns
    -------
    Union[FoldoverConfig, OptimalAugmentConfig] or None
        Configuration, or None if strategy not yet implemented
    """
    
    # Foldover strategies
    if strategy_name in ('full_foldover', 'partial_foldover'):
        if strategy_name == 'full_foldover':
            return FoldoverConfig(foldover_type='full')
        else:
            # Partial foldover - choose factor from user adjustment or diagnostics
            factor_to_fold = user_adjustments.get('factor_to_fold')
            
            if not factor_to_fold:
                # Use diagnostics to suggest a factor
                factor_to_fold = _suggest_foldover_factor(diagnostics)
            
            return FoldoverConfig(
                foldover_type='single_factor',
                factor_to_fold=factor_to_fold,
                reason=f"User goal: {goal.value}"
            )
    
    # D-optimal strategies
    elif strategy_name in ('d_optimal', 'd_optimal_quadratic', 'd_optimal_dealias',
                          'd_optimal_expansion', 'd_optimal_prediction'):
        
        # Determine model terms to add
        new_terms = _determine_model_terms(strategy_name, goal, diagnostics, user_adjustments)
        
        return OptimalAugmentConfig(
            new_model_terms=new_terms,
            n_runs_to_add=n_runs,
            criterion='D'
        )
    
    # I-optimal strategies
    elif strategy_name == 'i_optimal':
        # I-optimal for prediction - add current model terms
        current_terms = _get_current_model_terms(diagnostics)
        
        # Configure prediction grid based on number of factors
        n_factors = len(diagnostics.factors)
        prediction_grid_config = user_adjustments.get('prediction_grid_config', {})
        
        # Set defaults if not provided by user
        if 'n_points_per_dim' not in prediction_grid_config:
            prediction_grid_config['n_points_per_dim'] = 5
        if 'grid_type' not in prediction_grid_config:
            prediction_grid_config['grid_type'] = 'factorial' if n_factors <= 4 else 'lhs'
        
        return OptimalAugmentConfig(
            new_model_terms=current_terms,
            n_runs_to_add=n_runs,
            criterion='I',
            prediction_grid_config=prediction_grid_config
        )
    
    # CCD augmentation
    elif strategy_name == 'ccd_augmentation':
        # CCD adds axial points - model as D-optimal with quadratic terms
        factor_names = [f.name for f in diagnostics.factors]
        quadratic_terms = [f"{f}^2" for f in factor_names]
        
        return OptimalAugmentConfig(
            new_model_terms=quadratic_terms,
            n_runs_to_add=n_runs,
            criterion='D'
        )
    
    # Not yet implemented
    else:
        return None


def _get_strategy_type(strategy_name: str) -> str:
    """
    Map strategy name to strategy type enum.
    
    Returns
    -------
    str
        'foldover' or 'd_optimal'
    """
    
    if 'foldover' in strategy_name:
        return 'foldover'
    else:
        return 'd_optimal'


def _determine_model_terms(
    strategy_name: str,
    goal: AugmentationGoal,
    diagnostics: DesignDiagnosticSummary,
    user_adjustments: Dict
) -> List[str]:
    """
    Determine which model terms to add based on strategy and goal.
    
    Returns
    -------
    List[str]
        Model terms to estimate in augmented design
    """
    
    factor_names = [f.name for f in diagnostics.factors]
    
    # User specified terms
    if 'model_terms' in user_adjustments:
        return user_adjustments['model_terms']
    
    # Quadratic augmentation
    if 'quadratic' in strategy_name or goal == AugmentationGoal.MODEL_CURVATURE:
        # Add all quadratic terms
        quadratic = [f"{f}^2" for f in factor_names]
        return quadratic
    
    # De-aliasing
    elif 'dealias' in strategy_name or goal == AugmentationGoal.REDUCE_ALIASING:
        # Add 2-factor interactions
        two_fi = [f"{factor_names[i]}*{factor_names[j]}" 
                 for i in range(len(factor_names))
                 for j in range(i+1, len(factor_names))]
        return two_fi
    
    # Expansion or prediction
    elif goal in (AugmentationGoal.EXPAND_REGION, AugmentationGoal.IMPROVE_PREDICTION):
        # Use current model terms
        return _get_current_model_terms(diagnostics)
    
    else:
        # Default: main effects
        return factor_names


def _get_current_model_terms(diagnostics: DesignDiagnosticSummary) -> List[str]:
    """Extract current model terms from fitted models."""
    
    # Get terms from first response's diagnostics
    if diagnostics.response_diagnostics:
        first_response = list(diagnostics.response_diagnostics.values())[0]
        
        # Combine significant and marginally significant
        terms = (first_response.significant_effects + 
                first_response.marginally_significant)
        
        if terms:
            return terms
    
    # Fallback: main effects
    return [f.name for f in diagnostics.factors]


def _suggest_foldover_factor(diagnostics: DesignDiagnosticSummary) -> Optional[str]:
    """
    Suggest which factor to fold based on diagnostics.
    
    Returns
    -------
    str or None
        Factor name, or None if no good candidate
    """
    
    # Look for factors with significant aliased effects
    for diag in diagnostics.response_diagnostics.values():
        if diag.aliased_effects:
            # Find main effects with aliases
            for effect, aliases in diag.aliased_effects.items():
                if len(effect) == 1 and aliases:  # Single factor with aliases
                    # Check if this effect is significant
                    if effect in diag.significant_effects:
                        return effect
    
    # Fallback: first factor
    if diagnostics.factors:
        return diagnostics.factors[0].name
    
    return None


def _predict_improvements(
    strategy_name: str,
    goal: AugmentationGoal,
    diagnostics: DesignDiagnosticSummary,
    n_runs: int
) -> Dict[str, str]:
    """
    Predict improvements from this augmentation.
    
    Returns
    -------
    Dict[str, str]
        Metric name -> expected improvement description
    """
    
    improvements = {}
    
    # Goal-specific predictions
    if goal == AugmentationGoal.MODEL_CURVATURE:
        improvements['Model Terms'] = 'Add quadratic effects'
        improvements['Optimization'] = 'Enable response surface optimization'
    
    elif goal == AugmentationGoal.REDUCE_ALIASING:
        if diagnostics.design_type == 'fractional':
            current_res = 3  # Assume Res III if fractional
            improvements['Resolution'] = f'Increase from {current_res} to {current_res + 1}'
            improvements['Clarity'] = 'De-alias main effects from 2FI'
    
    elif goal == AugmentationGoal.IMPROVE_PREDICTION:
        improvements['Prediction Variance'] = 'Reduce by 30-50% in high-variance regions'
        improvements['Uniformity'] = 'More uniform prediction quality'
    
    elif goal == AugmentationGoal.INCREASE_CONFIDENCE:
        improvements['Pure Error'] = f'Add {n_runs} replicates for error estimation'
        improvements['LOF Test'] = 'Enable formal lack-of-fit testing'
    
    elif goal == AugmentationGoal.EXPAND_REGION:
        improvements['Coverage'] = 'Extend to broader factor ranges'
        improvements['Exploration'] = 'Discover behavior in new regions'
    
    elif goal == AugmentationGoal.ADD_ROBUSTNESS:
        improvements['Robustness'] = 'Test control factors under noise variation'
        improvements['Taguchi Analysis'] = 'Enable signal-to-noise analysis'
    
    return improvements


def _compute_goal_utility(
    goal: AugmentationGoal,
    strategy_name: str,
    n_runs: int,
    diagnostics: DesignDiagnosticSummary,
    is_primary: bool
) -> float:
    """
    Compute utility score for goal-driven plan.
    
    Returns
    -------
    float
        Utility score (0-100)
    """
    
    # Base score for primary vs alternative
    base_score = 85.0 if is_primary else 70.0
    
    # Adjust for efficiency (fewer runs = higher utility)
    efficiency_factor = 1.0 - min(n_runs / (2 * diagnostics.n_runs), 0.3)
    
    # Bonus if diagnostics support this goal
    diagnostic_bonus = _check_diagnostic_alignment(goal, diagnostics)
    
    utility = base_score * efficiency_factor + diagnostic_bonus
    
    return max(0.0, min(100.0, utility))


def _check_diagnostic_alignment(
    goal: AugmentationGoal,
    diagnostics: DesignDiagnosticSummary
) -> float:
    """
    Check if diagnostics support this goal.
    
    Returns bonus points (0-10) if diagnostics suggest this goal is appropriate.
    """
    
    bonus = 0.0
    
    # Check for aliasing issues
    has_aliasing = any(
        diag.resolution and diag.resolution <= 4
        for diag in diagnostics.response_diagnostics.values()
    )
    
    if goal == AugmentationGoal.REDUCE_ALIASING and has_aliasing:
        bonus += 5.0
    
    # Check for lack of fit
    has_lof = any(
        diag.lack_of_fit_p_value and diag.lack_of_fit_p_value < 0.05
        for diag in diagnostics.response_diagnostics.values()
    )
    
    if goal == AugmentationGoal.MODEL_CURVATURE and has_lof:
        bonus += 5.0
    
    # Check for high prediction variance
    has_high_var = any(
        diag.prediction_variance_stats and 
        diag.prediction_variance_stats.get('max', 0) > 3 * diag.prediction_variance_stats.get('mean', 1)
        for diag in diagnostics.response_diagnostics.values()
    )
    
    if goal == AugmentationGoal.IMPROVE_PREDICTION and has_high_var:
        bonus += 5.0
    
    return bonus


def _refine_with_diagnostics(
    plans: List[AugmentationPlan],
    diagnostics: DesignDiagnosticSummary
) -> List[AugmentationPlan]:
    """
    Refine plans using diagnostic information (informative, not blocking).
    
    Diagnostics are used to:
    - Adjust run count recommendations
    - Add warnings to metadata
    - Suggest parameter tweaks
    
    But NOT to:
    - Block plan execution
    - Change user-selected goals
    - Override user intent
    """
    
    for plan in plans:
        warnings = []
        suggestions = []
        
        goal = AugmentationGoal(plan.metadata['goal'])
        
        # Check: Already have what this goal provides?
        if goal == AugmentationGoal.INCREASE_CONFIDENCE:
            if diagnostics.has_center_points:
                warnings.append(
                    "Design already has center points for curvature detection"
                )
                suggestions.append(
                    "Consider adding only replicates rather than more center points"
                )
        
        if goal == AugmentationGoal.MODEL_CURVATURE:
            if diagnostics.design_type in ('response_surface', 'ccd', 'box_behnken'):
                warnings.append(
                    "Design is already a response surface - additional curvature terms may not be needed"
                )
        
        # Check: Diagnostics suggest different priority?
        critical_issues = []
        for diag in diagnostics.response_diagnostics.values():
            critical_issues.extend([i for i in diag.issues if i.severity == 'critical'])
        
        if critical_issues and goal not in (AugmentationGoal.REDUCE_ALIASING, 
                                            AugmentationGoal.MODEL_CURVATURE):
            warnings.append(
                f"Note: {len(critical_issues)} critical diagnostic issue(s) detected. "
                "Consider addressing these first."
            )
        
        # Store warnings in metadata
        if warnings:
            plan.metadata['diagnostic_warnings'] = warnings
        if suggestions:
            plan.metadata['diagnostic_suggestions'] = suggestions
    
    return plans


def _apply_user_adjustments(
    plans: List[AugmentationPlan],
    user_adjustments: Dict
) -> List[AugmentationPlan]:
    """
    Apply user-requested adjustments to plans.
    
    Supported adjustments:
    - n_runs: Override number of runs
    - factor_to_fold: Specify factor for partial foldover
    - model_terms: Specify terms to add
    - expand_ranges: New factor ranges (future)
    """
    
    # User adjustments already applied during plan creation
    # This is a hook for future dynamic adjustments
    
    return plans


def _generate_goal_plan_name(
    goal: AugmentationGoal,
    strategy_name: str,
    is_primary: bool
) -> str:
    """Generate plan name based on goal and strategy."""
    
    goal_desc = GOAL_CATALOG[goal]
    
    strategy_names = {
        'full_foldover': 'Full Foldover',
        'partial_foldover': 'Partial Foldover',
        'd_optimal': 'D-Optimal Design',
        'd_optimal_quadratic': 'Add Quadratic Terms',
        'i_optimal': 'I-Optimal Design',
        'ccd_augmentation': 'Central Composite Augmentation',
        'replicates_and_center_points': 'Replicates + Center Points',
        'space_filling': 'Space-Filling Design',
        'd_optimal_expansion': 'D-Optimal Region Expansion',
        'outer_array': 'Outer Array (Taguchi)',
    }
    
    strategy_display = strategy_names.get(strategy_name, strategy_name.replace('_', ' ').title())
    
    if is_primary:
        return f"{goal_desc.title}: {strategy_display}"
    else:
        return f"{goal_desc.title}: {strategy_display} (Alternative)"


def _rank_goal_plans(
    plans: List[AugmentationPlan],
    context: GoalDrivenContext
) -> List[AugmentationPlan]:
    """
    Rank goal-driven plans by utility.
    
    Returns
    -------
    List[AugmentationPlan]
        Plans sorted by utility (highest first)
    """
    
    # Sort by utility
    plans.sort(key=lambda p: p.utility_score, reverse=True)
    
    # Assign ranks
    for i, plan in enumerate(plans, 1):
        plan.rank = i
    
    return plans
