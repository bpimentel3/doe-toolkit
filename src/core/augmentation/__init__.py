"""
Design Augmentation Module.

Provides dual-mode augmentation workflow:
- Mode A: Diagnostics-driven (fix detected issues automatically)
- Mode B: Goal-driven (user specifies intent, system recommends strategies)
"""

from src.core.augmentation.plan import (
    AugmentationPlan,
    AugmentedDesign,
    FoldoverConfig,
    OptimalAugmentConfig,
    ValidationResult
)

from src.core.augmentation.goals import (
    AugmentationGoal,
    GoalDescription,
    GOAL_CATALOG,
    get_strategies_for_goal,
    get_available_goals
)

from src.core.augmentation.recommendations import (
    recommend_from_diagnostics
)

from src.core.augmentation.goal_driven import (
    recommend_from_goal,
    GoalDrivenContext
)

from src.core.augmentation.interface import (
    AugmentationRequest,
    recommend_augmentation,
    get_mode_availability,
    get_mode_recommendations,
    get_available_enhancement_goals,
    create_plan_comparison_table
)

__all__ = [
    # Core data structures
    'AugmentationPlan',
    'AugmentedDesign',
    'FoldoverConfig',
    'OptimalAugmentConfig',
    'ValidationResult',
    
    # Goals (Mode B)
    'AugmentationGoal',
    'GoalDescription',
    'GOAL_CATALOG',
    'get_strategies_for_goal',
    'get_available_goals',
    
    # Mode A: Diagnostics-driven
    'recommend_from_diagnostics',
    
    # Mode B: Goal-driven
    'recommend_from_goal',
    'GoalDrivenContext',
    
    # Unified interface
    'AugmentationRequest',
    'recommend_augmentation',
    'get_mode_availability',
    'get_mode_recommendations',
    'get_available_enhancement_goals',
    'create_plan_comparison_table',
]
