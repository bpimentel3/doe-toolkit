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
compute_benchmark_determinant
    Generate benchmark design for comparison
"""

import warnings
from typing import List, Literal, Tuple

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


def compute_benchmark_determinant(
    factors: List[Factor],
    model_type: Literal["linear", "interaction", "quadratic"],
    model_builder: ModelMatrixBuilder,
) -> Tuple[float, str]:
    """
    Compute determinant of benchmark design for comparison.

    Uses designs constrained to [-1, 1]^k space for fair comparison:
    - Linear models: Full Factorial 2^k
    - Quadratic models: Face-Centered CCD (alpha='face', stays in bounds)

    Parameters
    ----------
    factors : List[Factor]
        Factor definitions
    model_type : {'linear', 'interaction', 'quadratic'}
        Type of model
    model_builder : ModelMatrixBuilder
        Function to build model matrix

    Returns
    -------
    det : float
        Determinant of X'X for benchmark design
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
    >>> det, name = compute_benchmark_determinant(
    ...     factors, 'quadratic', model_builder
    ... )
    >>> print(f"Benchmark: {name}, det={det:.2e}")
    Benchmark: Face-Centered CCD (k=4), det=1.23e+10
    """
    k = len(factors)

    if model_type == "linear":
        # Benchmark: Full Factorial 2^k
        if k > 10:
            warnings.warn(
                f"Benchmark computation skipped: 2^{k} = {2**k} runs is too large. "
                "D-efficiency will be reported as 100%."
            )
            return 1.0, f"Full Factorial 2^{k} (not computed)"

        from itertools import product

        ff_coded = np.array(list(product([-1, 1], repeat=k)))
        X_ff = model_builder(ff_coded)
        det_ff = np.linalg.det(X_ff.T @ X_ff)
        return det_ff, f"Full Factorial 2^{k}"

    elif model_type in ("interaction", "quadratic"):
        # Benchmark: Face-Centered CCD (alpha='face')
        # This stays within [-1, 1]^k bounds for fair comparison with D-optimal
        from src.core.response_surface import CentralCompositeDesign
        from src.core.factors import FactorType, ChangeabilityLevel

        # Create temporary CCD
        temp_factors = [
            Factor(
                f"X{i}",
                FactorType.CONTINUOUS,
                ChangeabilityLevel.EASY,
                levels=[-1, 1],
            )
            for i in range(k)
        ]

        # Use face-centered design to stay in bounds
        ccd = CentralCompositeDesign(
            factors=temp_factors,
            alpha="face",  # Face-centered: axial points on cube faces (alpha=1)
            center_points=6 if k <= 4 else 5,
        )
        ccd_design = ccd.generate(randomize=False)

        # Extract coded design (skip StdOrder, RunOrder, PointType)
        factor_cols = [f.name for f in temp_factors]
        ccd_coded = ccd_design[factor_cols].values

        X_ccd = model_builder(ccd_coded)
        det_ccd = np.linalg.det(X_ccd.T @ X_ccd)
        return det_ccd, f"Face-Centered CCD (k={k})"

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
