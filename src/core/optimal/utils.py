"""
Utility Functions for Optimal Design.

This module provides helper functions for:
- Coding/decoding between actual and coded factor spaces
- Computing design efficiency metrics
- Generating benchmark designs for comparison

Functions
---------
code_point
    Convert actual values to coded [-1, 1] scale
decode_point
    Convert coded values to actual scale
compute_d_efficiency_vs_benchmark
    Calculate D-efficiency relative to benchmark
compute_benchmark_criterion
    Generate benchmark design for comparison (supports D and I criteria)
"""

import warnings
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from src.core.factors import Factor
from src.core.optimal.criteria import ModelMatrixBuilder


# ============================================================
# CODING/DECODING
# ============================================================


def code_point(x_actual: np.ndarray, factors: List[Factor]) -> np.ndarray:
    """
    Convert actual values to coded [-1, 1] scale.

    Parameters
    ----------
    x_actual : np.ndarray, shape (k,)
        Actual factor values
    factors : List[Factor]
        Factor definitions with min/max values

    Returns
    -------
    np.ndarray, shape (k,)
        Coded values in [-1, 1] range

    Notes
    -----
    Coding formula:
        x_coded = (x_actual - center) / half_range
        where center = (max + min) / 2
              half_range = (max - min) / 2

    Examples
    --------
    >>> # Temperature: [100, 200] -> actual value 150
    >>> x_coded = code_point(np.array([150]), factors)
    >>> x_coded
    array([0.0])  # Center point
    """
    x_coded = np.zeros_like(x_actual)
    for i, factor in enumerate(factors):
        center = (factor.max_value + factor.min_value) / 2
        half_range = (factor.max_value - factor.min_value) / 2
        x_coded[i] = (x_actual[i] - center) / half_range
    return x_coded


def decode_point(x_coded: np.ndarray, factors: List[Factor]) -> np.ndarray:
    """
    Convert coded values to actual scale.

    Parameters
    ----------
    x_coded : np.ndarray, shape (k,)
        Coded factor values in [-1, 1] range
    factors : List[Factor]
        Factor definitions with min/max values

    Returns
    -------
    np.ndarray, shape (k,)
        Actual factor values

    Notes
    -----
    Decoding formula:
        x_actual = center + x_coded * half_range
        where center = (max + min) / 2
              half_range = (max - min) / 2

    Examples
    --------
    >>> # Temperature: [100, 200] -> coded value 0.0
    >>> x_actual = decode_point(np.array([0.0]), factors)
    >>> x_actual
    array([150.0])  # Center point in actual space
    """
    x_actual = np.zeros_like(x_coded)
    for i, factor in enumerate(factors):
        center = (factor.max_value + factor.min_value) / 2
        half_range = (factor.max_value - factor.min_value) / 2
        x_actual[i] = center + x_coded[i] * half_range
    return x_actual


# ============================================================
# EFFICIENCY METRICS
# ============================================================


def compute_d_efficiency_vs_benchmark(
    det: float, n_runs: int, n_params: int, benchmark_det: float
) -> float:
    """
    Compute D-efficiency relative to benchmark design.

    D-efficiency measures how close a design's determinant is to
    a benchmark design (typically orthogonal design like full factorial
    or CCD).

    Parameters
    ----------
    det : float
        Determinant of X'X for the design
    n_runs : int
        Number of runs in the design
    n_params : int
        Number of parameters (p)
    benchmark_det : float
        Determinant of benchmark design's X'X

    Returns
    -------
    float
        D-efficiency as percentage (0-200%)
        Capped at 200% to handle numerical edge cases

    Notes
    -----
    D-efficiency formula:
        D_eff = (det / benchmark_det)^(1/p) * 100%

    Interpretation:
    - 100% = Equal to benchmark
    - >100% = Better than benchmark
    - <100% = Worse than benchmark

    For D-optimal designs:
    - Linear models: Should achieve ≥90% of full factorial
    - Quadratic models: Should match or exceed CCD (≥100%)

    Examples
    --------
    >>> compute_d_efficiency_vs_benchmark(
    ...     det=1000, n_runs=20, n_params=11, benchmark_det=950
    ... )
    100.47  # Slightly better than benchmark
    """
    if det <= 0 or benchmark_det <= 0:
        return 0.0

    efficiency = (det / benchmark_det) ** (1 / n_params)
    efficiency_pct = efficiency * 100

    return round(min(efficiency_pct, 200.0), 2)  # Cap at 200%


# ============================================================
# BENCHMARK DESIGNS
# ============================================================


def compute_benchmark_criterion(
    factors: List[Factor],
    model_type: Literal["linear", "interaction", "quadratic"],
    model_builder: ModelMatrixBuilder,
    criterion_type: Literal["D", "I"] = "D",
    prediction_grid_config: Optional[Dict] = None,
) -> Tuple[float, str]:
    """
    Compute criterion value of benchmark design for comparison.

    Uses designs constrained to [-1, 1]^k space for fair comparison:
    - Linear models: Full Factorial 2^k
    - Quadratic models: Face-Centered CCD (alpha='face', stays in bounds)

    Supports both D-optimality (determinant) and I-optimality (avg prediction variance).

    Parameters
    ----------
    factors : List[Factor]
        Factor definitions
    model_type : {'linear', 'interaction', 'quadratic'}
        Type of model
    model_builder : ModelMatrixBuilder
        Function to build model matrix
    criterion_type : {'D', 'I'}, default='D'
        Which criterion to compute:
        - 'D': Determinant of X'X
        - 'I': Average scaled prediction variance
    prediction_grid_config : dict, optional
        Configuration for prediction grid (only used for I-optimality)

    Returns
    -------
    criterion_value : float
        Criterion value for benchmark design:
        - For D: determinant of X'X
        - For I: average scaled prediction variance
    design_name : str
        Name of benchmark design used

    Notes
    -----
    For k > 10 factors, full factorial computation is skipped
    (2^k becomes too large) and returns 1.0 as placeholder.

    Face-centered CCD is used for quadratic models because:
    - Stays within [-1, 1]^k bounds (fair comparison)
    - Standard benchmark for response surface designs
    - alpha='face' places axial points on cube faces (alpha=1)

    Examples
    --------
    >>> # D-optimality benchmark
    >>> det, name = compute_benchmark_criterion(
    ...     factors, 'quadratic', model_builder, criterion_type='D'
    ... )
    >>> print(f"Benchmark: {name}, det={det:.2e}")
    
    >>> # I-optimality benchmark
    >>> i_crit, name = compute_benchmark_criterion(
    ...     factors, 'quadratic', model_builder, criterion_type='I',
    ...     prediction_grid_config={'n_points_per_dim': 5}
    ... )
    >>> print(f"Benchmark: {name}, I={i_crit:.4f}")
    """
    k = len(factors)

    # Generate benchmark design
    if model_type == "linear":
        # Benchmark: Full Factorial 2^k
        if k > 10:
            warnings.warn(
                f"Benchmark computation skipped: 2^{k} = {2**k} runs is too large. "
                f"Efficiency will be reported as 100%."
            )
            return 1.0, f"Full Factorial 2^{k} (not computed)"

        from itertools import product

        benchmark_coded = np.array(list(product([-1, 1], repeat=k)))
        design_name = f"Full Factorial 2^{k}"

    elif model_type in ("interaction", "quadratic"):
        # Benchmark: Face-Centered CCD (alpha='face')
        from src.core.response_surface import CentralCompositeDesign
        from src.core.factors import FactorType, ChangeabilityLevel

        temp_factors = [
            Factor(
                f"X{i}",
                FactorType.CONTINUOUS,
                ChangeabilityLevel.EASY,
                levels=[-1, 1],
            )
            for i in range(k)
        ]

        ccd = CentralCompositeDesign(
            factors=temp_factors,
            alpha="face",
            center_points=6 if k <= 4 else 5,
        )
        ccd_design = ccd.generate(randomize=False)
        factor_cols = [f.name for f in temp_factors]
        benchmark_coded = ccd_design[factor_cols].values
        design_name = f"Face-Centered CCD (k={k})"

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Compute requested criterion
    X_benchmark = model_builder(benchmark_coded)

    if criterion_type == "D":
        # Compute determinant
        det_benchmark = np.linalg.det(X_benchmark.T @ X_benchmark)
        return det_benchmark, design_name

    elif criterion_type == "I":
        # Compute I-criterion (average scaled prediction variance)
        from src.core.diagnostics.variance import compute_i_criterion
        from src.core.analysis import generate_model_terms
        import pandas as pd
        
        # Generate model terms
        model_terms = generate_model_terms(factors, model_type, include_intercept=True)
        
        # Convert benchmark design to DataFrame
        benchmark_df = pd.DataFrame(benchmark_coded, columns=[f.name for f in factors])
        
        # Compute I-criterion
        i_benchmark = compute_i_criterion(
            benchmark_df, factors, model_terms, prediction_grid_config
        )
        return i_benchmark, design_name

    else:
        raise ValueError(f"Unknown criterion_type: {criterion_type}")