"""
High-Level API for Optimal Design Generation.

This module provides the main user-facing functions for generating
optimal experimental designs. It orchestrates the candidate generation,
constraint filtering, optimization, and results packaging.

Classes
-------
OptimizationResult
    Container for optimization results and diagnostics

Functions
---------
generate_d_optimal_design
    Generate D-optimal experimental design
generate_optimal_design
    Generate optimal design with selectable criterion (D or I)
compute_i_efficiency
    Calculate I-efficiency relative to benchmark design

Notes
-----
This module integrates all components:
- Criteria (D-optimal, I-optimal)
- Candidate generation
- Constraint handling
- Optimization (CEXCH)
- Benchmarking and diagnostics

References
----------
.. [1] Meyer, R. K., & Nachtsheim, C. J. (1995). The coordinate-exchange
       algorithm for constructing exact optimal experimental designs.
       Technometrics, 37(1), 60-69.
.. [2] Atkinson, A. C., Donev, A. N., & Tobias, R. D. (2007).
       Optimum experimental designs, with SAS. Oxford University Press.
"""

import warnings
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from src.core.factors import Factor, FactorType
from src.core.optimal.candidates import CandidatePoolConfig, generate_candidate_pool
from src.core.optimal.constraints import (
    LinearConstraint,
    augment_constrained_candidates,
    create_linear_constraint_predicate,
    filter_feasible_candidates,
)
from src.core.optimal.criteria import (
    DOptimalityCriterion,
    IOptimalityCriterion,
    create_optimality_criterion,
    create_polynomial_builder,
    generate_prediction_grid,
)
from src.core.optimal.optimizer import OptimizerConfig, cexch_optimize
from src.core.optimal.utils import (
    compute_benchmark_criterion,
    compute_d_efficiency_vs_benchmark,
    decode_point,
)
from src.core.diagnostics.variance import compute_i_criterion


# ============================================================
# RESULT CONTAINER
# ============================================================


@dataclass
class OptimizationResult:
    """
    Result from optimal design generation.

    Attributes
    ----------
    design_coded : np.ndarray, shape (n_runs, k)
        Design in coded [-1, 1]^k space
    design_actual : pd.DataFrame
        Design in actual factor units with StdOrder and RunOrder
    model_matrix : np.ndarray, shape (n_runs, p)
        Model matrix X for the design
    criterion_type : str
        Optimality criterion used ('D-optimal', 'I-optimal', etc.)
    final_objective : float
        Final objective value achieved
    n_iterations : int
        Number of coordinate exchange iterations performed
    converged_by : str
        Convergence reason ('stability' or 'max_iterations')
    n_runs : int
        Number of runs in design
    n_parameters : int
        Number of model parameters (p)
    condition_number : float
        Condition number of X'X (measures collinearity)
    d_efficiency_vs_benchmark : float
        D-efficiency percentage relative to benchmark design
    benchmark_design_name : str
        Name of benchmark design used for comparison
    i_criterion : float, optional
        I-optimality criterion value (avg scaled prediction variance)
        Only populated for I-optimal designs
    i_efficiency_vs_benchmark : float, optional
        I-efficiency percentage relative to benchmark design
        Only populated for I-optimal designs
    """

    design_coded: np.ndarray
    design_actual: pd.DataFrame
    model_matrix: np.ndarray
    criterion_type: str
    final_objective: float
    n_iterations: int
    converged_by: str
    n_runs: int
    n_parameters: int
    condition_number: float
    d_efficiency_vs_benchmark: float
    benchmark_design_name: str
    i_criterion: Optional[float] = None
    i_efficiency_vs_benchmark: Optional[float] = None


# ============================================================
# MAIN API FUNCTIONS
# ============================================================


def generate_optimal_design(
    factors: List[Factor],
    model_type: Literal["linear", "interaction", "quadratic"],
    n_runs: int,
    criterion: Literal["D", "I"] = "D",
    prediction_grid_config: Optional[Dict] = None,
    constraints: Optional[List[LinearConstraint]] = None,
    candidate_config: Optional[CandidatePoolConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    seed: Optional[int] = None,
) -> OptimizationResult:
    """
    Generate optimal experimental design with selectable criterion.

    This is the unified API for generating both D-optimal and I-optimal designs.
    D-optimal designs maximize det(X'X) for best parameter estimates.
    I-optimal designs minimize average prediction variance across the design space.

    Parameters
    ----------
    factors : List[Factor]
        Factor definitions (must be continuous)
    model_type : {'linear', 'interaction', 'quadratic'}
        Type of polynomial model to fit
    n_runs : int
        Number of experimental runs
    criterion : {'D', 'I'}, default='D'
        Optimality criterion:
        - 'D': D-optimal (maximize det(X'X), best for parameter estimation)
        - 'I': I-optimal (minimize avg prediction variance, best for prediction)
    prediction_grid_config : dict, optional
        Configuration for I-optimal prediction grid (ignored for D-optimal)
        Keys: 'grid_type' ('factorial' or 'lhs'), 'n_points_per_dim', etc.
    constraints : List[LinearConstraint], optional
        Linear constraints on factor combinations
    candidate_config : CandidatePoolConfig, optional
        Configuration for candidate pool generation
    optimizer_config : OptimizerConfig, optional
        Configuration for optimization algorithm
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    OptimizationResult
        Complete optimization results with design, diagnostics, and metadata

    Raises
    ------
    ValueError
        - If factors are not continuous
        - If n_runs < number of parameters
        - If constraints are too restrictive
        - If invalid criterion specified

    Examples
    --------
    >>> # D-optimal design (parameter estimation)
    >>> result_d = generate_optimal_design(
    ...     factors=factors,
    ...     model_type='quadratic',
    ...     n_runs=20,
    ...     criterion='D',
    ...     seed=42
    ... )

    >>> # I-optimal design (prediction)
    >>> result_i = generate_optimal_design(
    ...     factors=factors,
    ...     model_type='quadratic',
    ...     n_runs=20,
    ...     criterion='I',
    ...     prediction_grid_config={'n_points_per_dim': 7},
    ...     seed=42
    ... )

    References
    ----------
    .. [1] Meyer, R. K., & Nachtsheim, C. J. (1995).
    .. [2] Giovannitti-Jensen, A., & Myers, R. H. (1989). Graphical assessment
           of the prediction capability of response surface designs.
           Technometrics, 31(2), 159-171.
    """
    # Validate inputs
    if criterion not in ("D", "I"):
        raise ValueError(f"criterion must be 'D' or 'I', got '{criterion}'")

    for factor in factors:
        if factor.factor_type != FactorType.CONTINUOUS:
            raise ValueError(
                f"Factor '{factor.name}' must be continuous. "
                "Categorical/discrete not yet supported."
            )

    k = len(factors)
    model_builder = create_polynomial_builder(factors, model_type)

    # Determine number of parameters
    if model_type == "linear":
        n_params = 1 + k
    elif model_type == "interaction":
        n_params = 1 + k + k * (k - 1) // 2
    elif model_type == "quadratic":
        n_params = 1 + k + k * (k - 1) // 2 + k

    if n_runs < n_params:
        raise ValueError(f"n_runs ({n_runs}) must be >= n_parameters ({n_params})")

    # Generate candidate pool
    if candidate_config is None:
        candidate_config = CandidatePoolConfig(lhs_multiplier=5)

    candidates = generate_candidate_pool(factors, n_runs, candidate_config, seed)

    # Apply constraints if provided
    if constraints:
        is_feasible = create_linear_constraint_predicate(factors, constraints)
        candidates_initial = filter_feasible_candidates(candidates, is_feasible)
        min_candidates_needed = max(n_runs * 5, n_params * 10)

        if len(candidates_initial) < n_runs:
            warnings.warn(
                f"Only {len(candidates_initial)} feasible candidates found, "
                f"need {n_runs} runs. Attempting augmentation..."
            )
            candidates_augmented = augment_constrained_candidates(
                factors=factors,
                existing_candidates=candidates_initial,
                is_feasible=is_feasible,
                target_size=max(n_runs * 2, min_candidates_needed),
                seed=seed,
            )
            if len(candidates_augmented) < n_runs:
                raise ValueError(
                    f"Even after augmentation, only {len(candidates_augmented)} "
                    f"feasible candidates found (need {n_runs} runs)."
                )
            candidates = candidates_augmented

        elif len(candidates_initial) < min_candidates_needed:
            warnings.warn(
                f"Low candidate density: {len(candidates_initial)} feasible candidates."
            )
            candidates = candidates_initial
        else:
            candidates = candidates_initial

    # Create criterion object
    if criterion == "D":
        criterion_obj = DOptimalityCriterion()
        criterion_name = "D-optimal"
    else:  # criterion == "I"
        prediction_points = generate_prediction_grid(factors, prediction_grid_config)
        criterion_obj = IOptimalityCriterion(
            prediction_points=prediction_points, model_builder=model_builder
        )
        criterion_name = "I-optimal"

    # Optimize
    if optimizer_config is None:
        optimizer_config = OptimizerConfig()

    indices, best_objective, n_iter, converged_by = cexch_optimize(
        candidates=candidates,
        n_runs=n_runs,
        model_builder=model_builder,
        criterion=criterion_obj,
        config=optimizer_config,
        seed=seed,
    )

    # Build final design
    design_coded = candidates[indices]
    design_actual_array = np.array(
        [decode_point(design_coded[i], factors) for i in range(n_runs)]
    )
    design_actual_df = pd.DataFrame(
        design_actual_array, columns=[f.name for f in factors]
    )
    design_actual_df.insert(0, "StdOrder", range(1, n_runs + 1))
    design_actual_df.insert(1, "RunOrder", range(1, n_runs + 1))

    # Build model matrix
    X_model = model_builder(design_coded)
    XtX = X_model.T @ X_model

    try:
        condition_number = np.linalg.cond(XtX)
    except:
        condition_number = np.inf

    # Compute D-efficiency
    det_achieved = np.linalg.det(XtX)
    det_benchmark, benchmark_name = compute_benchmark_criterion(
        factors, model_type, model_builder, criterion_type="D"
    )
    d_efficiency = compute_d_efficiency_vs_benchmark(
        det_achieved, n_runs, n_params, det_benchmark
    )

    # Compute I-criterion and efficiency if I-optimal
    i_crit = None
    i_eff = None
    if criterion == "I":
        # Get model terms for I-criterion computation
        from src.core.analysis import generate_model_terms
        model_terms = generate_model_terms(factors, model_type, include_intercept=True)
        i_crit = compute_i_criterion(
            design_actual_df[[f.name for f in factors]], 
            factors, 
            model_terms, 
            prediction_grid_config
        )

        # Compute I-efficiency
        i_eff = compute_i_efficiency(
            design_coded=design_coded,
            factors=factors,
            model_type=model_type,
            model_builder=model_builder,
            prediction_grid_config=prediction_grid_config,
        )

    # Warnings
    if criterion == "D":
        if model_type == "linear" and d_efficiency < 90:
            warnings.warn(
                f"D-efficiency: {d_efficiency:.2f}%. Expected >90% for linear models."
            )
        elif model_type in ("interaction", "quadratic") and d_efficiency < 100:
            warnings.warn(
                f"D-efficiency: {d_efficiency:.2f}%. Expected ≥100% for quadratic."
            )

    if condition_number > 100:
        warnings.warn(f"High condition number ({condition_number:.1f}).")

    return OptimizationResult(
        design_coded=design_coded,
        design_actual=design_actual_df,
        model_matrix=X_model,
        criterion_type=criterion_name,
        final_objective=best_objective,
        n_iterations=n_iter,
        converged_by=converged_by,
        n_runs=n_runs,
        n_parameters=n_params,
        condition_number=condition_number,
        d_efficiency_vs_benchmark=d_efficiency,
        benchmark_design_name=benchmark_name,
        i_criterion=i_crit,
        i_efficiency_vs_benchmark=i_eff,
    )


def generate_d_optimal_design(
    factors: List[Factor],
    model_type: Literal["linear", "interaction", "quadratic"],
    n_runs: int,
    constraints: Optional[List[LinearConstraint]] = None,
    candidate_config: Optional[CandidatePoolConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    seed: Optional[int] = None,
) -> OptimizationResult:
    """
    Generate D-optimal experimental design.

    This is a convenience wrapper around generate_optimal_design() with criterion='D'.
    D-optimal designs maximize det(X'X), minimizing the generalized variance of
    parameter estimates. Ideal for parameter estimation tasks.

    Parameters
    ----------
    factors : List[Factor]
        Factor definitions (must be continuous)
    model_type : {'linear', 'interaction', 'quadratic'}
        Type of polynomial model to fit
    n_runs : int
        Number of experimental runs
    constraints : List[LinearConstraint], optional
        Linear constraints on factor combinations
    candidate_config : CandidatePoolConfig, optional
        Configuration for candidate pool generation
    optimizer_config : OptimizerConfig, optional
        Configuration for optimization algorithm
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    OptimizationResult
        Complete optimization results with design, diagnostics, and metadata

    See Also
    --------
    generate_optimal_design : Unified API with selectable criterion

    Examples
    --------
    >>> result = generate_d_optimal_design(
    ...     factors=factors,
    ...     model_type='quadratic',
    ...     n_runs=20,
    ...     seed=42
    ... )
    """
    return generate_optimal_design(
        factors=factors,
        model_type=model_type,
        n_runs=n_runs,
        criterion="D",
        constraints=constraints,
        candidate_config=candidate_config,
        optimizer_config=optimizer_config,
        seed=seed,
    )


def compute_i_efficiency(
    design_coded: np.ndarray,
    factors: List[Factor],
    model_type: Literal["linear", "interaction", "quadratic"],
    model_builder,
    prediction_grid_config: Optional[Dict] = None,
) -> float:
    """
    Calculate I-efficiency relative to benchmark design.

    I-efficiency measures how close a design's average prediction variance
    is to a benchmark design (typically orthogonal design like full factorial
    or CCD).

    Parameters
    ----------
    design_coded : np.ndarray, shape (n_runs, k)
        Design in coded space
    factors : List[Factor]
        Factor definitions
    model_type : {'linear', 'interaction', 'quadratic'}
        Type of model
    model_builder : callable
        Function to build model matrix
    prediction_grid_config : dict, optional
        Configuration for prediction grid

    Returns
    -------
    float
        I-efficiency as percentage (0-200%)
        Capped at 200% to handle numerical edge cases

    Notes
    -----
    I-efficiency formula:
        I_eff = (I_benchmark / I_design) * 100%

    where I is the average scaled prediction variance.

    Lower I values are better (less prediction variance), so the
    benchmark is in the numerator.

    Examples
    --------
    >>> i_eff = compute_i_efficiency(
    ...     design_coded=design,
    ...     factors=factors,
    ...     model_type='quadratic',
    ...     model_builder=builder,
    ...     prediction_grid_config={'n_points_per_dim': 5}
    ... )
    >>> print(f"I-efficiency: {i_eff:.2f}%")
    """
    # Generate model terms
    from src.core.analysis import generate_model_terms
    model_terms = generate_model_terms(factors, model_type, include_intercept=True)
    
    # Convert design to DataFrame
    design_df = pd.DataFrame(design_coded, columns=[f.name for f in factors])
    
    # Compute I-criterion for this design
    i_design = compute_i_criterion(design_df, factors, model_terms, prediction_grid_config)

    # Compute benchmark I-criterion
    _, benchmark_name = compute_benchmark_criterion(
        factors, model_type, model_builder, criterion_type="I",
        prediction_grid_config=prediction_grid_config
    )
    
    # Get benchmark design
    k = len(factors)
    if model_type == "linear":
        from itertools import product
        if k > 10:
            return 100.0  # Can't compute benchmark
        benchmark_coded = np.array(list(product([-1, 1], repeat=k)))
    else:
        from src.core.response_surface import CentralCompositeDesign
        from src.core.factors import FactorType, ChangeabilityLevel
        temp_factors = [
            Factor(f"X{i}", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, 
                   levels=[-1, 1])
            for i in range(k)
        ]
        ccd = CentralCompositeDesign(
            factors=temp_factors, alpha="face",
            center_points=6 if k <= 4 else 5
        )
        ccd_design = ccd.generate(randomize=False)
        factor_cols = [f.name for f in temp_factors]
        benchmark_coded = ccd_design[factor_cols].values

    benchmark_df = pd.DataFrame(benchmark_coded, columns=[f.name for f in factors])
    i_benchmark = compute_i_criterion(benchmark_df, factors, model_terms, prediction_grid_config)

    # Calculate efficiency (lower I is better, so benchmark in numerator)
    if i_design <= 0:
        return 0.0

    efficiency = (i_benchmark / i_design) * 100
    return round(min(efficiency, 200.0), 2)