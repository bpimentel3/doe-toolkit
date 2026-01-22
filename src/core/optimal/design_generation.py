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
    Generate D-optimal experimental design (main API)

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
from typing import List, Literal, Optional

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
from src.core.optimal.criteria import DOptimalityCriterion, create_polynomial_builder
from src.core.optimal.optimizer import OptimizerConfig, cexch_optimize
from src.core.optimal.utils import (
    compute_benchmark_determinant,
    compute_d_efficiency_vs_benchmark,
    decode_point,
)


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


# ============================================================
# MAIN API FUNCTION
# ============================================================


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

    D-optimal designs maximize the determinant of X'X, which minimizes
    the generalized variance of parameter estimates. This makes them
    ideal for parameter estimation tasks.

    The algorithm uses Meyer & Nachtsheim's coordinate exchange (CEXCH)
    with Sherman-Morrison determinant updates for computational efficiency.

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
        Default: CandidatePoolConfig(lhs_multiplier=5)
    optimizer_config : OptimizerConfig, optional
        Configuration for optimization algorithm
        Default: OptimizerConfig() with sensible defaults
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
        - If constraints are too restrictive (insufficient feasible candidates)

    Warnings
    --------
    - Low D-efficiency (<90% for linear, <100% for quadratic)
    - High condition number (>100)
    - Insufficient candidate density for constrained designs

    Notes
    -----
    Number of parameters for each model type:
    - Linear: p = 1 + k
    - Interaction: p = 1 + k + k(k-1)/2
    - Quadratic: p = 1 + 2k + k(k-1)/2

    Benchmark designs for efficiency comparison:
    - Linear models: Full Factorial 2^k
    - Quadratic models: Face-Centered CCD

    Examples
    --------
    >>> # Simple 3-factor D-optimal design
    >>> from src.core.factors import Factor, FactorType, ChangeabilityLevel
    >>> factors = [
    ...     Factor("Temp", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
    ...            levels=[100, 200]),
    ...     Factor("Press", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
    ...            levels=[50, 100]),
    ...     Factor("Time", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
    ...            levels=[10, 30])
    ... ]
    >>> result = generate_d_optimal_design(
    ...     factors=factors,
    ...     model_type='quadratic',
    ...     n_runs=20,
    ...     seed=42
    ... )
    >>> result.design_actual
       StdOrder  RunOrder   Temp  Press  Time
    0         1         1  150.0   75.0  20.0
    ...

    >>> # With linear constraints
    >>> from src.core.optimal.constraints import LinearConstraint
    >>> constraints = [
    ...     LinearConstraint(
    ...         coefficients={'Temp': 1, 'Press': 2},
    ...         bound=400,
    ...         constraint_type='le'
    ...     )
    ... ]
    >>> result = generate_d_optimal_design(
    ...     factors=factors,
    ...     model_type='linear',
    ...     n_runs=12,
    ...     constraints=constraints,
    ...     seed=42
    ... )

    References
    ----------
    .. [1] Meyer, R. K., & Nachtsheim, C. J. (1995). The coordinate-exchange
           algorithm for constructing exact optimal experimental designs.
           Technometrics, 37(1), 60-69.
    """
    # Validate factors
    for factor in factors:
        if factor.factor_type != FactorType.CONTINUOUS:
            raise ValueError(
                f"Factor '{factor.name}' must be continuous. "
                "Categorical/discrete not yet supported."
            )

    k = len(factors)

    # Build model matrix builder
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

    # Apply feasibility filter and handle constrained candidate pools
    if constraints:
        is_feasible = create_linear_constraint_predicate(factors, constraints)
        candidates_initial = filter_feasible_candidates(candidates, is_feasible)

        # Calculate density requirements
        min_candidates_needed = max(n_runs * 5, n_params * 10)

        # Decision tree based on candidate availability
        if len(candidates_initial) < n_runs:
            # CRITICAL: Not even enough for minimum design
            # Try augmentation as last resort BEFORE failing
            warnings.warn(
                f"Only {len(candidates_initial)} feasible candidates found, "
                f"need {n_runs} runs. Attempting augmentation via rejection sampling..."
            )

            candidates_augmented = augment_constrained_candidates(
                factors=factors,
                existing_candidates=candidates_initial,
                is_feasible=is_feasible,
                target_size=max(n_runs * 2, min_candidates_needed),
                seed=seed,
            )

            # Check if augmentation saved us
            if len(candidates_augmented) < n_runs:
                raise ValueError(
                    f"Even after augmentation, only {len(candidates_augmented)} "
                    f"feasible candidates found (need {n_runs} runs). "
                    f"Constraints are too restrictive or infeasible. Consider:\n"
                    f"  - Relaxing constraints\n"
                    f"  - Reducing n_runs\n"
                    f"  - Verifying constraints are not contradictory"
                )

            # Success! Use augmented pool
            candidates = candidates_augmented
            warnings.warn(
                f"Augmentation successful: {len(candidates)} feasible candidates "
                f"found after rejection sampling."
            )

        elif len(candidates_initial) < min_candidates_needed:
            # Have minimum for design, but density is low
            warnings.warn(
                f"Low candidate density: {len(candidates_initial)} feasible candidates "
                f"for {n_runs} runs (recommended: ≥{min_candidates_needed}). "
                f"Attempting to improve density via rejection sampling..."
            )

            # Try to improve density
            candidates_augmented = augment_constrained_candidates(
                factors=factors,
                existing_candidates=candidates_initial,
                is_feasible=is_feasible,
                target_size=min_candidates_needed,
                seed=seed,
            )

            if len(candidates_augmented) >= min_candidates_needed * 0.7:
                # Augmentation helped significantly
                candidates = candidates_augmented
                warnings.warn(
                    f"Augmented candidate pool to {len(candidates)} points "
                    f"({len(candidates) - len(candidates_initial)} new feasible points)."
                )
            else:
                # Augmentation didn't help much
                candidates = candidates_initial
                warnings.warn(
                    f"Augmentation only reached {len(candidates_augmented)} candidates "
                    f"(target: {min_candidates_needed}). Proceeding with limited pool. "
                    f"Design quality may be suboptimal. Consider:\n"
                    f"  - Relaxing constraints\n"
                    f"  - Reducing n_runs\n"
                    f"  - Increasing candidate pool size"
                )
        else:
            # Sufficient candidate density - no augmentation needed
            candidates = candidates_initial
    else:
        # No constraints - use full candidate pool
        pass

    # Create D-optimal criterion
    criterion = DOptimalityCriterion()

    # Optimize
    if optimizer_config is None:
        optimizer_config = OptimizerConfig()

    indices, best_logdet, n_iter, converged_by = cexch_optimize(
        candidates=candidates,
        n_runs=n_runs,
        model_builder=model_builder,
        criterion=criterion,
        config=optimizer_config,
        seed=seed,
    )

    # Build final design
    design_coded = candidates[indices]

    design_actual_array = np.array(
        [decode_point(design_coded[i], factors) for i in range(n_runs)]
    )

    design_actual_df = pd.DataFrame(design_actual_array, columns=[f.name for f in factors])
    design_actual_df.insert(0, "StdOrder", range(1, n_runs + 1))
    design_actual_df.insert(1, "RunOrder", range(1, n_runs + 1))

    # Build model matrix
    X_model = model_builder(design_coded)

    # Compute diagnostics
    XtX = X_model.T @ X_model
    try:
        condition_number = np.linalg.cond(XtX)
    except:
        condition_number = np.inf

    # Compute efficiency vs benchmark design
    det_achieved = np.exp(best_logdet)
    det_benchmark, benchmark_name = compute_benchmark_determinant(
        factors, model_type, model_builder
    )

    d_efficiency = compute_d_efficiency_vs_benchmark(
        det_achieved, n_runs, n_params, det_benchmark
    )

    # Warnings
    if model_type == "linear" and d_efficiency < 90:
        warnings.warn(
            f"D-efficiency vs {benchmark_name}: {d_efficiency:.2f}%. "
            f"Expected >90% for linear models. Consider more runs or fewer constraints."
        )
    elif model_type in ("interaction", "quadratic") and d_efficiency < 100:
        warnings.warn(
            f"D-efficiency vs {benchmark_name}: {d_efficiency:.2f}%. "
            f"D-optimal should match or exceed CCD for quadratic models."
        )

    if condition_number > 100:
        warnings.warn(
            f"High condition number ({condition_number:.1f}). Design may be ill-conditioned."
        )

    return OptimizationResult(
        design_coded=design_coded,
        design_actual=design_actual_df,
        model_matrix=X_model,
        criterion_type="D-optimal",
        final_objective=best_logdet,
        n_iterations=n_iter,
        converged_by=converged_by,
        n_runs=n_runs,
        n_parameters=n_params,
        condition_number=condition_number,
        d_efficiency_vs_benchmark=d_efficiency,
        benchmark_design_name=benchmark_name,
    )
