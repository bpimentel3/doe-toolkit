"""
Optimal Design Package.

This package provides algorithms for generating optimal experimental designs
using coordinate exchange optimization with various optimality criteria.

Main Functions
--------------
generate_d_optimal_design
    Generate D-optimal experimental design (legacy API)
generate_optimal_design
    Generate optimal design with selectable criterion (D or I)
compute_i_efficiency
    Calculate I-efficiency relative to benchmark design

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
>>> from src.core.optimal import generate_optimal_design
>>> from src.core.factors import Factor, FactorType, ChangeabilityLevel
>>>
>>> factors = [
...     Factor("Temp", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [100, 200]),
...     Factor("Press", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [50, 100])
... ]
>>>
>>> # D-optimal design (parameter estimation)
>>> result_d = generate_optimal_design(
...     factors=factors,
...     model_type='quadratic',
...     n_runs=15,
...     criterion='D',
...     seed=42
... )
>>>
>>> # I-optimal design (prediction)
>>> result_i = generate_optimal_design(
...     factors=factors,
...     model_type='quadratic',
...     n_runs=15,
...     criterion='I',
...     prediction_grid_config={'n_points_per_dim': 7},
...     seed=42
... )
"""

# Main API
from src.core.optimal.design_generation import (
    OptimizationResult,
    compute_i_efficiency,
    generate_d_optimal_design,
    generate_optimal_design,
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
    generate_prediction_grid,
)

__all__ = [
    # Main API
    "generate_optimal_design",
    "generate_d_optimal_design",
    "compute_i_efficiency",
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
    "generate_prediction_grid",
]