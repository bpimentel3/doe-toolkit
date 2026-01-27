"""
Optimality Criteria for Experimental Design.

This module defines optimality criteria (D-optimal, I-optimal) and model
matrix builders used during design optimization.

Classes
-------
OptimalityCriterion : ABC
    Abstract base class for optimality criteria
DOptimalityCriterion : OptimalityCriterion
    D-optimality (maximize determinant)
IOptimalityCriterion : OptimalityCriterion
    I-optimality (minimize average prediction variance)

Functions
---------
create_polynomial_builder
    Create model matrix builder for polynomial models
generate_prediction_grid
    Generate prediction points for I-optimality
create_optimality_criterion
    Factory function to create criterion objects

References
----------
.. [1] Atkinson, A. C., Donev, A. N., & Tobias, R. D. (2007).
       Optimum experimental designs, with SAS. Oxford University Press.
.. [2] Jones, B., & Goos, P. (2012). I-optimal versus D-optimal
       split-plot response surface designs. Journal of Quality Technology.
"""

from abc import ABC, abstractmethod
from typing import List, Literal, Optional, Protocol

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube

from src.core.factors import Factor


# ============================================================
# MODEL MATRIX BUILDER PROTOCOL
# ============================================================


class ModelMatrixBuilder(Protocol):
    """Protocol for building model matrix from factor settings."""

    def __call__(self, X_points: np.ndarray) -> np.ndarray:
        """
        Build model matrix from factor settings.

        Parameters
        ----------
        X_points : np.ndarray, shape (n, k)
            Factor settings in coded space

        Returns
        -------
        np.ndarray, shape (n, p)
            Model matrix with p parameters
        """
        ...


def create_polynomial_builder(
    factors: List[Factor], model_type: Literal["linear", "interaction", "quadratic"]
) -> ModelMatrixBuilder:
    """
    Create a polynomial model matrix builder.

    Parameters
    ----------
    factors : List[Factor]
        Factor definitions
    model_type : {'linear', 'interaction', 'quadratic'}
        Type of polynomial model

    Returns
    -------
    ModelMatrixBuilder
        Function that builds model matrix from coded factor settings

    Examples
    --------
    >>> builder = create_polynomial_builder(factors, 'quadratic')
    >>> X_model = builder(coded_points)
    >>> X_model.shape
    (20, 11)  # 20 runs, 11 parameters for 4 factors quadratic
    """
    from src.core.analysis import generate_model_terms
    from src.core.diagnostics.variance import build_model_matrix

    # Generate standard terms for this model type
    model_terms = generate_model_terms(factors, model_type, include_intercept=True)

    def builder(X_points: np.ndarray) -> np.ndarray:
        """Build model matrix from factor settings."""
        # Convert points array to DataFrame
        design_df = pd.DataFrame(X_points, columns=[f.name for f in factors])

        # Use centralized model matrix builder
        return build_model_matrix(design_df, factors, model_terms)

    return builder


# ============================================================
# OPTIMALITY CRITERION BASE CLASS
# ============================================================


class OptimalityCriterion(ABC):
    """
    Abstract base class for optimality criteria.

    All optimality criteria must implement the objective() method
    which returns a value to MAXIMIZE during optimization.
    """

    @abstractmethod
    def objective(self, X_model: np.ndarray) -> float:
        """
        Compute objective value to MAXIMIZE.

        Parameters
        ----------
        X_model : np.ndarray, shape (n, p)
            Model matrix for current design

        Returns
        -------
        float
            Objective value (higher is better)
            Returns large negative value if design is singular
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return criterion name."""
        pass


# ============================================================
# D-OPTIMALITY
# ============================================================


class DOptimalityCriterion(OptimalityCriterion):
    """
    D-optimality criterion: Maximize determinant of X'X.

    D-optimal designs minimize the generalized variance of parameter
    estimates, making them ideal for parameter estimation tasks.

    Parameters
    ----------
    ridge : float, default=1e-10
        Ridge regularization for numerical stability

    Notes
    -----
    The objective function returns log(det(X'X)) rather than det(X'X)
    to avoid numerical overflow and enable additive updates.

    For a design to be valid, the model matrix must have full rank.
    Singular designs return a large negative value (-1e10).
    """

    def __init__(self, ridge: float = 1e-10):
        self.ridge = ridge

    def objective(self, X_model: np.ndarray) -> float:
        """
        Compute log-determinant of X'X.

        Parameters
        ----------
        X_model : np.ndarray, shape (n, p)
            Model matrix

        Returns
        -------
        float
            log(det(X'X + ridge*I))
            Returns -1e10 if design is singular
        """
        try:
            XtX = X_model.T @ X_model
            rank = np.linalg.matrix_rank(XtX)
            if rank < XtX.shape[0]:
                return -1e10

            XtX_ridge = XtX + self.ridge * np.eye(XtX.shape[0])
            sign, logdet = np.linalg.slogdet(XtX_ridge)

            if sign <= 0:
                return -1e10

            return logdet

        except (np.linalg.LinAlgError, ValueError):
            return -1e10

    @property
    def name(self) -> str:
        return "D-optimal"


# ============================================================
# I-OPTIMALITY
# ============================================================


class IOptimalityCriterion(OptimalityCriterion):
    """
    I-optimality: Minimize average prediction variance.

    The I-optimality criterion minimizes the average prediction variance
    across the design region, making it ideal for creating prediction
    equations that work well across the entire experimental space.

    Mathematical Definition
    -----------------------
    I-criterion = trace((X'X)^(-1) * M)

    where M is the moment matrix over the prediction region:
    M = ∫∫ x(s) x(s)' ds  (continuous)
    M ≈ (1/N) Σ x_i x_i'  (discretized over N prediction points)

    Since we want to MAXIMIZE the objective in the optimization framework,
    we return -I (negative I-criterion).

    Parameters
    ----------
    prediction_points : np.ndarray, shape (N_pred, k)
        Points where prediction variance is evaluated (coded space)
    model_builder : ModelMatrixBuilder
        Function to build model matrix from factor settings
    ridge : float, default=1e-10
        Ridge regularization parameter for numerical stability

    Attributes
    ----------
    M : np.ndarray, shape (p, p)
        Precomputed moment matrix

    Notes
    -----
    The prediction points should cover the design region of interest.
    Common choices:
    - Factorial grid: n^k points for n points per dimension
    - Latin Hypercube: space-filling sample for large k

    References
    ----------
    .. [1] Atkinson, A. C., Donev, A. N., & Tobias, R. D. (2007).
           Optimum experimental designs, with SAS. Chapter 11.
    .. [2] Jones, B., & Goos, P. (2012). I-optimal versus D-optimal
           split-plot response surface designs. Journal of Quality
           Technology, 44(2), 85-101.

    Examples
    --------
    >>> prediction_grid = generate_prediction_grid(factors)
    >>> criterion = IOptimalityCriterion(prediction_grid, model_builder)
    >>> objective = criterion.objective(X_model)  # Returns -I for maximization
    """

    def __init__(
        self,
        prediction_points: np.ndarray,
        model_builder: ModelMatrixBuilder,
        ridge: float = 1e-10,
    ):
        """
        Initialize I-optimality criterion.

        Parameters
        ----------
        prediction_points : np.ndarray, shape (N_pred, k)
            Points where to evaluate prediction variance (coded space)
        model_builder : ModelMatrixBuilder
            Function to build model matrix from factor settings
        ridge : float, default=1e-10
            Ridge parameter for numerical stability
        """
        self.prediction_points = prediction_points
        self.model_builder = model_builder
        self.ridge = ridge

        # Precompute moment matrix M = (1/N) Σ x_i x_i'
        # This only needs to be done once
        X_pred = model_builder(prediction_points)
        self.M = (X_pred.T @ X_pred) / len(prediction_points)

    def objective(self, X_model: np.ndarray) -> float:
        """
        Compute negative I-criterion to MAXIMIZE.

        The I-criterion is:
        I = trace((X'X)^(-1) * M)

        Lower I means better prediction quality.
        We return -I so that maximizing the objective minimizes I.

        Parameters
        ----------
        X_model : np.ndarray, shape (n, p)
            Model matrix for current design

        Returns
        -------
        float
            Negative I-criterion (for maximization framework)
            Returns -1e10 if design is singular or invalid

        Notes
        -----
        The trace computation is efficient because M is precomputed.
        Complexity: O(p^2) for the matrix multiply + O(p) for trace.
        """
        try:
            XtX = X_model.T @ X_model
            rank = np.linalg.matrix_rank(XtX)
            if rank < XtX.shape[0]:
                return -1e10

            # Add ridge for numerical stability
            XtX_ridge = XtX + self.ridge * np.eye(XtX.shape[0])
            XtX_inv = np.linalg.inv(XtX_ridge)

            # I-criterion = trace((X'X)^(-1) * M)
            i_criterion = np.trace(XtX_inv @ self.M)

            # Return negative for maximization
            # (Lower I is better, so maximizing -I minimizes I)
            return -i_criterion

        except (np.linalg.LinAlgError, ValueError):
            return -1e10

    @property
    def name(self) -> str:
        return "I-optimal"


# ============================================================
# PREDICTION GRID GENERATION
# ============================================================


def generate_prediction_grid(
    factors: List[Factor], config: Optional[dict] = None
) -> np.ndarray:
    """
    Generate prediction grid for I-optimality moment matrix.

    The prediction grid defines the region over which average prediction
    variance is computed for I-optimality. The grid should cover the
    experimental region of interest.

    Parameters
    ----------
    factors : List[Factor]
        Factor definitions
    config : dict, optional
        Configuration with keys:
        - 'n_points_per_dim': int (default: 5)
          Number of points per dimension for grid
        - 'include_vertices': bool (default: True)
          Include factorial vertices (corners)
        - 'include_center': bool (default: True)
          Include center point
        - 'grid_type': {'factorial', 'lhs'} (default: auto-select)
          Grid generation strategy

    Returns
    -------
    np.ndarray, shape (N_pred, k)
        Prediction points in coded [-1, 1]^k space

    Notes
    -----
    Grid type selection:
    - k ≤ 4: Factorial grid (5^4 = 625 points is manageable)
    - k > 4: LHS grid (avoids exponential growth)

    For factorial grid with n points per dimension:
    Total points = n^k (grows exponentially)

    For LHS grid:
    Total points = n^2 (linear in n, independent of k)

    Examples
    --------
    >>> # 3 factors, default settings (5^3 = 125 points)
    >>> grid = generate_prediction_grid(factors)
    >>> grid.shape
    (125, 3)

    >>> # 6 factors, LHS to avoid 5^6 = 15625 points
    >>> grid = generate_prediction_grid(factors_6, {'grid_type': 'lhs'})
    >>> grid.shape
    (25, 6)
    """
    if config is None:
        config = {}

    n_points_per_dim = config.get("n_points_per_dim", 5)
    include_vertices = config.get("include_vertices", True)
    include_center = config.get("include_center", True)

    k = len(factors)

    # Auto-select grid type based on k
    if "grid_type" in config:
        grid_type = config["grid_type"]
    else:
        grid_type = "factorial" if k <= 4 else "lhs"

    points_list = []

    # Strategy 1: Factorial grid (good for k ≤ 4)
    if grid_type == "factorial":
        # Create regular grid
        grid_1d = np.linspace(-1, 1, n_points_per_dim)

        from itertools import product

        grid_points = np.array(list(product(grid_1d, repeat=k)))
        points_list.append(grid_points)

    # Strategy 2: LHS (better for k > 4)
    elif grid_type == "lhs":
        # Use n^2 points to get good coverage without exponential growth
        n_points_total = n_points_per_dim**2
        sampler = LatinHypercube(d=k, seed=42)
        lhs_01 = sampler.random(n=n_points_total)
        lhs_coded = 2 * lhs_01 - 1
        points_list.append(lhs_coded)

        # Add vertices for boundary coverage
        if include_vertices:
            from src.core.optimal.candidates import generate_vertices

            vertices = generate_vertices(k)
            points_list.append(vertices)

    else:
        raise ValueError(
            f"Unknown grid_type: {grid_type}. " f"Must be 'factorial' or 'lhs'."
        )

    # Optional: Add center point
    if include_center:
        center = np.zeros((1, k))
        points_list.append(center)

    # Combine and deduplicate
    prediction_points = np.vstack(points_list)
    prediction_points = np.unique(np.round(prediction_points, decimals=6), axis=0)

    return prediction_points


# ============================================================
# FACTORY FUNCTION
# ============================================================


def create_optimality_criterion(
    criterion_type: Literal["D", "I"],
    model_builder: ModelMatrixBuilder,
    factors: List[Factor],
    prediction_grid_config: Optional[dict] = None,
    ridge: float = 1e-10,
) -> OptimalityCriterion:
    """
    Factory function to create optimality criterion object.

    This function provides a unified interface for creating either
    D-optimal or I-optimal criterion objects, abstracting away the
    differences in their construction.

    Parameters
    ----------
    criterion_type : {'D', 'I'}
        Optimality criterion type:
        - 'D': D-optimal (maximize det(X'X))
        - 'I': I-optimal (minimize average prediction variance)
    model_builder : ModelMatrixBuilder
        Function that builds model matrix from factor settings
    factors : List[Factor]
        Factor definitions
    prediction_grid_config : dict, optional
        Configuration for I-optimal prediction grid.
        Ignored for D-optimal.
        See generate_prediction_grid() for options.
    ridge : float, default=1e-10
        Ridge regularization parameter for numerical stability

    Returns
    -------
    OptimalityCriterion
        Criterion object implementing objective() method

    Raises
    ------
    ValueError
        If criterion_type is not 'D' or 'I'

    Examples
    --------
    >>> # D-optimal criterion
    >>> criterion_d = create_optimality_criterion(
    ...     'D', model_builder, factors
    ... )
    >>> objective_d = criterion_d.objective(X_model)

    >>> # I-optimal criterion with custom prediction grid
    >>> criterion_i = create_optimality_criterion(
    ...     'I', model_builder, factors,
    ...     prediction_grid_config={'n_points_per_dim': 7}
    ... )
    >>> objective_i = criterion_i.objective(X_model)
    """
    if criterion_type == "D":
        return DOptimalityCriterion(ridge=ridge)

    elif criterion_type == "I":
        # Generate prediction grid
        prediction_points = generate_prediction_grid(
            factors=factors, config=prediction_grid_config or {}
        )

        return IOptimalityCriterion(
            prediction_points=prediction_points,
            model_builder=model_builder,
            ridge=ridge,
        )

    else:
        raise ValueError(
            f"Unknown criterion_type: '{criterion_type}'. " f"Must be 'D' or 'I'."
        )
