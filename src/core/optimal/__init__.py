"""
Optimal Design Package.

This package provides algorithms for generating optimal experimental designs
using coordinate exchange optimization with various optimality criteria.

Main Functions
--------------
generate_d_optimal_design
    Generate D-optimal experimental design

Key Classes
-----------
OptimizationResult
    Container for design results and diagnostics
LinearConstraint
    Linear constraint definition
CandidatePoolConfig
    Configuration for candidate pool generation
OptimizerConfig
    Configuration for optimization algorithm

Examples
--------
>>> from src.core.optimal import generate_d_optimal_design, LinearConstraint
>>> from src.core.factors import Factor, FactorType, ChangeabilityLevel
>>>
>>> factors = [
...     Factor("Temp", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [100, 200]),
...     Factor("Press", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [50, 100])
... ]
>>>
>>> result = generate_d_optimal_design(
...     factors=factors,
...     model_type='quadratic',
...     n_runs=15,
...     seed=42
... )
>>> print(result.design_actual)
"""

# Main API
from src.core.optimal.design_generation import (
    OptimizationResult,
    generate_d_optimal_design,
)

# Configuration classes
from src.core.optimal.candidates import CandidatePoolConfig
from src.core.optimal.constraints import LinearConstraint
from src.core.optimal.optimizer import OptimizerConfig

# Advanced: Criteria for custom workflows
from src.core.optimal.criteria import (
    DOptimalityCriterion,
    IOptimalityCriterion,
    OptimalityCriterion,
    create_optimality_criterion,
    create_polynomial_builder,
)

__all__ = [
    # Main API
    "generate_d_optimal_design",
    "OptimizationResult",
    # Configuration
    "LinearConstraint",
    "CandidatePoolConfig",
    "OptimizerConfig",
    # Advanced
    "OptimalityCriterion",
    "DOptimalityCriterion",
    "IOptimalityCriterion",
    "create_optimality_criterion",
    "create_polynomial_builder",
]
