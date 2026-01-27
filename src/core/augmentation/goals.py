"""
User Intent Goals for Design Augmentation.

This module defines the engineer-friendly goals that drive Mode B
(user-intent driven) augmentation, mapping high-level intentions
to appropriate statistical strategies.
"""

from typing import List, Dict, Optional, Literal
from dataclasses import dataclass
from enum import Enum


class AugmentationGoal(Enum):
    """
    High-level engineering goals for design augmentation.
    
    These are phrased in engineering language rather than statistical jargon,
    making them accessible to engineers who understand experimental objectives
    but may not be statisticians.
    
    Attributes
    ----------
    INCREASE_CONFIDENCE : str
        Add replication to increase confidence in current conclusions
    IMPROVE_PREDICTION : str
        Improve prediction quality across the design space
    MODEL_CURVATURE : str
        Add capability to model curvature or search for an optimum
    REDUCE_ALIASING : str
        Clarify which effects are real by reducing aliasing
    EXPAND_REGION : str
        Expand or shift the experimental region to explore new territory
    ADD_ROBUSTNESS : str
        Add robustness checks against noise or uncontrolled factors
    """
    
    INCREASE_CONFIDENCE = "increase_confidence"
    IMPROVE_PREDICTION = "improve_prediction"
    MODEL_CURVATURE = "model_curvature"
    REDUCE_ALIASING = "reduce_aliasing"
    EXPAND_REGION = "expand_region"
    ADD_ROBUSTNESS = "add_robustness"


@dataclass
class GoalDescription:
    """
    Human-readable description of an augmentation goal.
    
    Attributes
    ----------
    goal : AugmentationGoal
        The goal enum
    title : str
        Short title for UI display
    description : str
        Detailed description explaining when this goal is appropriate
    typical_strategies : List[str]
        Common augmentation strategies for this goal
    when_appropriate : str
        When an engineer should choose this goal
    example_scenario : str
        Concrete example scenario
    """
    
    goal: AugmentationGoal
    title: str
    description: str
    typical_strategies: List[str]
    when_appropriate: str
    example_scenario: str


# Goal catalog with engineering-friendly descriptions
GOAL_CATALOG: Dict[AugmentationGoal, GoalDescription] = {
    AugmentationGoal.INCREASE_CONFIDENCE: GoalDescription(
        goal=AugmentationGoal.INCREASE_CONFIDENCE,
        title="Increase Confidence in Current Conclusions",
        description=(
            "Add replicate runs and center points to increase statistical power, "
            "reduce experimental error, and verify that your current model adequately "
            "describes the system."
        ),
        typical_strategies=["replicates", "center_points", "lack_of_fit_test"],
        when_appropriate=(
            "Use when you have promising initial results but want to be more confident "
            "before making recommendations or scaling up."
        ),
        example_scenario=(
            "Your initial 8-run screening found 3 active factors, but you want to "
            "confirm these effects are real before investing in follow-up work."
        )
    ),
    
    AugmentationGoal.IMPROVE_PREDICTION: GoalDescription(
        goal=AugmentationGoal.IMPROVE_PREDICTION,
        title="Improve Prediction Across the Design Space",
        description=(
            "Add runs optimally placed to reduce prediction variance throughout the "
            "experimental region, giving you better predictions at untested conditions."
        ),
        typical_strategies=["i_optimal", "space_filling", "uniform_precision"],
        when_appropriate=(
            "Use when you need to predict response values at many different factor "
            "settings, not just find an optimum."
        ),
        example_scenario=(
            "You need to create a prediction equation that accurately predicts yield "
            "across the full range of operating conditions for your process."
        )
    ),
    
    AugmentationGoal.MODEL_CURVATURE: GoalDescription(
        goal=AugmentationGoal.MODEL_CURVATURE,
        title="Model Curvature or Search for an Optimum",
        description=(
            "Upgrade your design to estimate quadratic effects, allowing you to "
            "find optimal factor settings or understand curvature in the response surface."
        ),
        typical_strategies=["ccd_augmentation", "box_behnken_augmentation", "d_optimal_quadratic"],
        when_appropriate=(
            "Use when screening found important factors and you're ready to optimize, "
            "or when you suspect the response has a maximum or minimum in your region."
        ),
        example_scenario=(
            "Your 2-level factorial identified temperature and pressure as important. "
            "Now you want to find the optimal temperature-pressure combination."
        )
    ),
    
    AugmentationGoal.REDUCE_ALIASING: GoalDescription(
        goal=AugmentationGoal.REDUCE_ALIASING,
        title="Reduce Aliasing or Clarify Effects",
        description=(
            "De-alias effects that are currently confounded, allowing you to determine "
            "which factors are truly active versus which are aliased artifacts."
        ),
        typical_strategies=["full_foldover", "partial_foldover", "d_optimal_dealias"],
        when_appropriate=(
            "Use when you have a fractional factorial with resolution III or IV and "
            "multiple effects show significance that could be confounded."
        ),
        example_scenario=(
            "In your Resolution IV design, both 'Temperature' and 'Pressure×Time' "
            "appear significant, but they're aliased. You need to know which is real."
        )
    ),
    
    AugmentationGoal.EXPAND_REGION: GoalDescription(
        goal=AugmentationGoal.EXPAND_REGION,
        title="Expand or Shift the Region of Interest",
        description=(
            "Add runs outside your current experimental boundaries to explore a "
            "broader or different region of the factor space."
        ),
        typical_strategies=["d_optimal_expansion", "axial_points", "region_shift"],
        when_appropriate=(
            "Use when initial results suggest interesting behavior outside your "
            "current factor ranges, or when you need to explore a different region."
        ),
        example_scenario=(
            "Your current experiments used 150-200°C, but results suggest the optimum "
            "might be above 200°C. You want to extend the temperature range to 250°C."
        )
    ),
    
    AugmentationGoal.ADD_ROBUSTNESS: GoalDescription(
        goal=AugmentationGoal.ADD_ROBUSTNESS,
        title="Add Robustness to Noise Factors",
        description=(
            "Add an outer array or noise factor combinations to understand how "
            "uncontrolled factors affect your process and find robust settings."
        ),
        typical_strategies=["outer_array", "noise_factor_augmentation", "robust_design"],
        when_appropriate=(
            "Use when you've identified good nominal settings but need to ensure "
            "they perform well despite variation in uncontrolled factors."
        ),
        example_scenario=(
            "Your process works well in the lab, but raw material variation and "
            "ambient humidity might affect production. You need robust settings."
        )
    ),
}


@dataclass
class GoalStrategyMapping:
    """
    Mapping from user goal to appropriate augmentation strategies.
    
    This is the core of Mode B - translating engineering intent into
    statistical action.
    
    Attributes
    ----------
    goal : AugmentationGoal
        The user's selected goal
    primary_strategy : str
        Primary recommended strategy
    alternative_strategies : List[str]
        Alternative strategies for this goal
    required_design_features : List[str]
        Design features needed (e.g., 'center_points', 'curvature_terms')
    estimated_min_runs : int
        Minimum additional runs typically needed
    estimated_max_runs : int
        Maximum additional runs typically recommended
    strategy_rationale : str
        Why this strategy addresses the goal
    """
    
    goal: AugmentationGoal
    primary_strategy: str
    alternative_strategies: List[str]
    required_design_features: List[str]
    estimated_min_runs: int
    estimated_max_runs: int
    strategy_rationale: str


def get_strategies_for_goal(
    goal: AugmentationGoal,
    current_design_type: str,
    n_factors: int,
    current_runs: int
) -> GoalStrategyMapping:
    """
    Map a user goal to appropriate augmentation strategies.
    
    This function implements the core logic of Mode B: translating
    engineering goals into statistical strategies while considering
    the current design context.
    
    Parameters
    ----------
    goal : AugmentationGoal
        User's selected goal
    current_design_type : str
        Current design type ('full_factorial', 'fractional', 'response_surface', etc.)
    n_factors : int
        Number of factors in design
    current_runs : int
        Current number of runs
    
    Returns
    -------
    GoalStrategyMapping
        Recommended strategies for this goal
    
    Examples
    --------
    >>> mapping = get_strategies_for_goal(
    ...     AugmentationGoal.MODEL_CURVATURE,
    ...     'fractional',
    ...     n_factors=4,
    ...     current_runs=8
    ... )
    >>> print(mapping.primary_strategy)
    'ccd_augmentation'
    >>> print(mapping.estimated_min_runs)
    8
    """
    
    # Goal: Increase Confidence
    if goal == AugmentationGoal.INCREASE_CONFIDENCE:
        # Replicates + center points for pure error and LOF testing
        min_runs = max(4, current_runs // 4)  # At least 25% more
        max_runs = current_runs  # Don't double the design
        
        return GoalStrategyMapping(
            goal=goal,
            primary_strategy="replicates_and_center_points",
            alternative_strategies=["center_points_only", "full_replication"],
            required_design_features=["pure_error_estimate", "lof_test"],
            estimated_min_runs=min_runs,
            estimated_max_runs=max_runs,
            strategy_rationale=(
                "Replicates provide pure error estimate for lack-of-fit testing. "
                "Center points check for curvature and improve precision at the center."
            )
        )
    
    # Goal: Improve Prediction
    elif goal == AugmentationGoal.IMPROVE_PREDICTION:
        # I-optimal or space-filling designs
        min_runs = n_factors + 1  # At least one run per factor
        max_runs = 2 * n_factors  # Typically 1-2x number of factors
        
        return GoalStrategyMapping(
            goal=goal,
            primary_strategy="i_optimal",
            alternative_strategies=["space_filling", "d_optimal_prediction"],
            required_design_features=["improved_prediction_variance"],
            estimated_min_runs=min_runs,
            estimated_max_runs=max_runs,
            strategy_rationale=(
                "I-optimal designs minimize average prediction variance across the "
                "design space, giving uniform prediction quality."
            )
        )
    
    # Goal: Model Curvature
    elif goal == AugmentationGoal.MODEL_CURVATURE:
        # CCD or quadratic augmentation
        if current_design_type == 'fractional':
            # Need axial points + center
            min_runs = 2 * n_factors + 3  # Axial + center
            max_runs = 2 * n_factors + 6  # Axial + multiple centers
        else:
            # Already have factorial core, just need axial
            min_runs = 2 * n_factors + 1
            max_runs = 2 * n_factors + 4
        
        return GoalStrategyMapping(
            goal=goal,
            primary_strategy="ccd_augmentation",
            alternative_strategies=["d_optimal_quadratic", "box_behnken_augmentation"],
            required_design_features=["quadratic_terms", "axial_points"],
            estimated_min_runs=min_runs,
            estimated_max_runs=max_runs,
            strategy_rationale=(
                "Central Composite Design augmentation adds axial points to estimate "
                "quadratic effects, enabling response surface modeling and optimization."
            )
        )
    
    # Goal: Reduce Aliasing
    elif goal == AugmentationGoal.REDUCE_ALIASING:
        if current_design_type != 'fractional':
            # Not applicable to non-fractional designs
            return GoalStrategyMapping(
                goal=goal,
                primary_strategy="not_applicable",
                alternative_strategies=[],
                required_design_features=[],
                estimated_min_runs=0,
                estimated_max_runs=0,
                strategy_rationale="Aliasing reduction only applies to fractional factorial designs."
            )
        
        # Foldover doubles the design
        min_runs = current_runs  # Full foldover
        max_runs = current_runs  # Full foldover (partial would be less)
        
        return GoalStrategyMapping(
            goal=goal,
            primary_strategy="full_foldover",
            alternative_strategies=["partial_foldover", "d_optimal_dealias"],
            required_design_features=["increased_resolution", "dealiased_effects"],
            estimated_min_runs=min_runs,
            estimated_max_runs=max_runs,
            strategy_rationale=(
                "Foldover augmentation creates a mirror-image design that breaks "
                "alias chains, typically increasing resolution from III to IV or IV to V."
            )
        )
    
    # Goal: Expand Region
    elif goal == AugmentationGoal.EXPAND_REGION:
        # D-optimal in expanded region
        min_runs = n_factors + 1  # Minimum for model support
        max_runs = 3 * n_factors  # More runs for broader coverage
        
        return GoalStrategyMapping(
            goal=goal,
            primary_strategy="d_optimal_expansion",
            alternative_strategies=["axial_points_extended", "uniform_expansion"],
            required_design_features=["expanded_factor_ranges"],
            estimated_min_runs=min_runs,
            estimated_max_runs=max_runs,
            strategy_rationale=(
                "D-optimal augmentation in an expanded region efficiently explores "
                "new factor space while maintaining model support."
            )
        )
    
    # Goal: Add Robustness
    elif goal == AugmentationGoal.ADD_ROBUSTNESS:
        # Outer array for noise factors
        # Multiply current design by noise factor levels
        # Assuming 2-3 noise factors with 2 levels each = 4-8x multiplier
        min_runs = current_runs * 2  # 2 noise conditions
        max_runs = current_runs * 4  # 4 noise conditions
        
        return GoalStrategyMapping(
            goal=goal,
            primary_strategy="outer_array",
            alternative_strategies=["combined_array", "noise_factor_screening"],
            required_design_features=["noise_factors", "robust_parameter_design"],
            estimated_min_runs=min_runs,
            estimated_max_runs=max_runs,
            strategy_rationale=(
                "Outer array augmentation runs each control factor setting under "
                "multiple noise factor conditions to identify robust settings."
            )
        )
    
    else:
        raise ValueError(f"Unknown goal: {goal}")


def get_available_goals(
    current_design_type: str,
    has_replicates: bool,
    has_center_points: bool,
    is_fractional: bool
) -> List[AugmentationGoal]:
    """
    Determine which goals are available given current design state.
    
    Not all goals make sense for all designs. This function filters
    the goal catalog to show only appropriate options.
    
    Parameters
    ----------
    current_design_type : str
        Current design type
    has_replicates : bool
        Whether design already has replicates
    has_center_points : bool
        Whether design already has center points
    is_fractional : bool
        Whether design is fractional factorial
    
    Returns
    -------
    List[AugmentationGoal]
        Goals that make sense for this design
    
    Examples
    --------
    >>> goals = get_available_goals(
    ...     'fractional',
    ...     has_replicates=False,
    ...     has_center_points=True,
    ...     is_fractional=True
    ... )
    >>> AugmentationGoal.REDUCE_ALIASING in goals
    True
    """
    available = []
    
    # Confidence: always available, but de-prioritize if already have replicates
    if not has_replicates:
        available.append(AugmentationGoal.INCREASE_CONFIDENCE)
    
    # Prediction: always available
    available.append(AugmentationGoal.IMPROVE_PREDICTION)
    
    # Curvature: available if not already a response surface
    if current_design_type not in ['response_surface', 'ccd', 'box_behnken']:
        available.append(AugmentationGoal.MODEL_CURVATURE)
    
    # Aliasing: only for fractional designs
    if is_fractional:
        available.append(AugmentationGoal.REDUCE_ALIASING)
    
    # Expansion: always available
    available.append(AugmentationGoal.EXPAND_REGION)
    
    # Robustness: always available (advanced feature)
    available.append(AugmentationGoal.ADD_ROBUSTNESS)
    
    return available
