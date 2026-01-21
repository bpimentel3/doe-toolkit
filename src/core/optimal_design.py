"""
D-Optimal Design Generation for Design of Experiments.

OPTIMIZED VERSION with:
- Sherman-Morrison reuse (no duplicate computation)
- Efficiency benchmarking against known designs (FF, CCD)
- Simplified naive LHS candidate generation

Fast implementation using:
- Meyer & Nachtsheim's CEXCH (full row exchange)
- Sherman-Morrison determinant updates with intermediate reuse
- Restricted candidate subsets
- Naive LHS candidate pools (no stratification)

References:
    [1] Meyer, R. K., & Nachtsheim, C. J. (1995). The coordinate-exchange 
        algorithm for constructing exact optimal experimental designs.
        Technometrics, 37(1), 60-69.
    [2] Atkinson, A. C., Donev, A. N., & Tobias, R. D. (2007).
        Optimum experimental designs, with SAS. Oxford University Press.
"""

import numpy as np
import pandas as pd
from typing import List, Literal, Optional, Tuple, Callable, Protocol
from dataclasses import dataclass
from scipy.stats.qmc import LatinHypercube
import warnings
from abc import ABC, abstractmethod

from src.core.factors import Factor, FactorType


# ============================================================
# SECTION 1: MODEL MATRIX BUILDER
# ============================================================

class ModelMatrixBuilder(Protocol):
    """Protocol for building model matrix from factor settings."""
    
    def __call__(self, X_points: np.ndarray) -> np.ndarray:
        """Build model matrix from factor settings."""
        ...


def create_polynomial_builder(
    factors: List[Factor],
    model_type: Literal['linear', 'interaction', 'quadratic'] = 'linear'
) -> ModelMatrixBuilder:
    """Create a polynomial model matrix builder."""
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
# SECTION 2: OPTIMALITY CRITERIA
# ============================================================

class OptimalityCriterion(ABC):
    """Abstract base class for optimality criteria."""
    
    @abstractmethod
    def objective(self, X_model: np.ndarray) -> float:
        """Compute objective value to MAXIMIZE."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return criterion name."""
        pass


class DOptimalityCriterion(OptimalityCriterion):
    """D-optimality: Maximize determinant of X'X."""
    
    def __init__(self, ridge: float = 1e-10):
        self.ridge = ridge
    
    def objective(self, X_model: np.ndarray) -> float:
        """Compute log-determinant of X'X."""
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
    
    Attributes
    ----------
    prediction_points : np.ndarray, shape (N_pred, k)
        Points where prediction variance is evaluated (coded space)
    model_builder : ModelMatrixBuilder
        Function to build model matrix from factor settings
    M : np.ndarray, shape (p, p)
        Precomputed moment matrix
    ridge : float
        Ridge regularization parameter for numerical stability
    
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
        ridge: float = 1e-10
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


def generate_prediction_grid(
    factors: List[Factor],
    config: Optional[dict] = None
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
    
    n_points_per_dim = config.get('n_points_per_dim', 5)
    include_vertices = config.get('include_vertices', True)
    include_center = config.get('include_center', True)
    
    k = len(factors)
    
    # Auto-select grid type based on k
    if 'grid_type' in config:
        grid_type = config['grid_type']
    else:
        grid_type = 'factorial' if k <= 4 else 'lhs'
    
    points_list = []
    
    # Strategy 1: Factorial grid (good for k ≤ 4)
    if grid_type == 'factorial':
        # Create regular grid
        grid_1d = np.linspace(-1, 1, n_points_per_dim)
        
        from itertools import product
        grid_points = np.array(list(product(grid_1d, repeat=k)))
        points_list.append(grid_points)
    
    # Strategy 2: LHS (better for k > 4)
    elif grid_type == 'lhs':
        # Use n^2 points to get good coverage without exponential growth
        n_points_total = n_points_per_dim ** 2
        sampler = LatinHypercube(d=k, seed=42)
        lhs_01 = sampler.random(n=n_points_total)
        lhs_coded = 2 * lhs_01 - 1
        points_list.append(lhs_coded)
        
        # Add vertices for boundary coverage
        if include_vertices:
            vertices = generate_vertices(k)
            points_list.append(vertices)
    
    else:
        raise ValueError(
            f"Unknown grid_type: {grid_type}. "
            f"Must be 'factorial' or 'lhs'."
        )
    
    # Optional: Add center point
    if include_center:
        center = np.zeros((1, k))
        points_list.append(center)
    
    # Combine and deduplicate
    prediction_points = np.vstack(points_list)
    prediction_points = np.unique(np.round(prediction_points, decimals=6), axis=0)
    
    return prediction_points


def create_optimality_criterion(
    criterion_type: Literal['D', 'I'],
    model_builder: ModelMatrixBuilder,
    factors: List[Factor],
    prediction_grid_config: Optional[dict] = None,
    ridge: float = 1e-10
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
    if criterion_type == 'D':
        return DOptimalityCriterion(ridge=ridge)
    
    elif criterion_type == 'I':
        # Generate prediction grid
        prediction_points = generate_prediction_grid(
            factors=factors,
            config=prediction_grid_config or {}
        )
        
        return IOptimalityCriterion(
            prediction_points=prediction_points,
            model_builder=model_builder,
            ridge=ridge
        )
    
    else:
        raise ValueError(
            f"Unknown criterion_type: '{criterion_type}'. "
            f"Must be 'D' or 'I'."
        )


# ============================================================
# SECTION 3: CANDIDATE POOL GENERATION (SIMPLIFIED)
# ============================================================

@dataclass
class CandidatePoolConfig:
    """Configuration for candidate pool generation."""
    
    lhs_multiplier: int = 5  # Generate n_runs * lhs_multiplier LHS points
    include_vertices: bool = True
    include_axial: bool = True
    include_center: bool = True
    alpha_axial: float = 1.0


def generate_vertices(k: int) -> np.ndarray:
    """Generate all 2^k vertices of [-1, 1]^k hypercube."""
    from itertools import product
    return np.array(list(product([-1, 1], repeat=k)))


def generate_axial_points(k: int, alpha: float = 1.0) -> np.ndarray:
    """Generate 2k axial (star) points."""
    points = []
    for j in range(k):
        point_pos = np.zeros(k)
        point_pos[j] = alpha
        points.append(point_pos)
        
        point_neg = np.zeros(k)
        point_neg[j] = -alpha
        points.append(point_neg)
    
    return np.array(points)


def generate_candidate_pool(
    factors: List[Factor],
    n_runs: int,
    config: CandidatePoolConfig,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate candidate pool with structured points + naive LHS.
    
    Strategy:
    - Include factorial vertices (2^k points)
    - Include axial/star points (2k points)
    - Include center point
    - Add n_runs * lhs_multiplier points from naive LHS (no stratification)
    
    This gives optimizer flexibility to choose interior or boundary points
    based on criterion, rather than biasing the candidate pool.
    """
    k = len(factors)
    candidates_list = []
    
    if config.include_vertices:
        vertices = generate_vertices(k)
        candidates_list.append(vertices)
    
    if config.include_axial:
        axial = generate_axial_points(k, config.alpha_axial)
        candidates_list.append(axial)
    
    if config.include_center:
        center = np.zeros((1, k))
        candidates_list.append(center)
    
    # Naive LHS - no stratification by boundary/interior
    # Just generate n_runs * lhs_multiplier points uniformly
    n_lhs = n_runs * config.lhs_multiplier
    sampler = LatinHypercube(d=k, seed=seed)
    lhs_01 = sampler.random(n=n_lhs)
    lhs_coded = 2 * lhs_01 - 1  # Scale to [-1, 1]
    candidates_list.append(lhs_coded)
    
    candidates = np.vstack(candidates_list)
    candidates = np.unique(np.round(candidates, decimals=6), axis=0)
    
    return candidates


# ============================================================
# SECTION 4: FEASIBILITY FILTERING
# ============================================================

class FeasibilityPredicate(Protocol):
    """Protocol for feasibility checking."""
    
    def __call__(self, x: np.ndarray) -> bool:
        """Check if point is feasible."""
        ...


@dataclass
class LinearConstraint:
    """Linear constraint in actual (decoded) factor space."""
    coefficients: dict
    bound: float
    constraint_type: Literal['le', 'ge', 'eq'] = 'le'
    
    def evaluate(self, point: dict) -> bool:
        """Check if point satisfies constraint."""
        lhs = sum(self.coefficients.get(k, 0) * v for k, v in point.items())
        
        if self.constraint_type == 'le':
            return lhs <= self.bound + 1e-10
        elif self.constraint_type == 'ge':
            return lhs >= self.bound - 1e-10
        elif self.constraint_type == 'eq':
            return abs(lhs - self.bound) < 1e-10
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")


def create_linear_constraint_predicate(
    factors: List[Factor],
    constraints: List[LinearConstraint]
) -> FeasibilityPredicate:
    """Create feasibility predicate from linear constraints."""
    def is_feasible(x_coded: np.ndarray) -> bool:
        x_actual = decode_point(x_coded, factors)
        point_dict = {f.name: x_actual[i] for i, f in enumerate(factors)}
        
        for constraint in constraints:
            if not constraint.evaluate(point_dict):
                return False
        
        return True
    
    return is_feasible


def filter_feasible_candidates(
    candidates: np.ndarray,
    is_feasible: FeasibilityPredicate
) -> np.ndarray:
    """Filter candidate pool to only feasible points."""
    mask = np.array([is_feasible(candidates[i]) for i in range(len(candidates))])
    return candidates[mask]


def augment_constrained_candidates(
    factors: List[Factor],
    existing_candidates: np.ndarray,
    is_feasible: FeasibilityPredicate,
    target_size: int,
    seed: Optional[int] = None,
    max_attempts: int = 10000
) -> np.ndarray:
    """
    Augment candidate pool using rejection sampling in feasible region.
    
    When constraints carve out a small feasible region, the standard candidate
    pool may have insufficient density. This uses rejection sampling to find
    more feasible points.
    
    Parameters
    ----------
    factors : List[Factor]
        Factor definitions
    existing_candidates : np.ndarray
        Current feasible candidates
    is_feasible : FeasibilityPredicate
        Feasibility checker
    target_size : int
        Desired number of feasible candidates
    seed : int, optional
        Random seed
    max_attempts : int
        Maximum rejection sampling attempts
    
    Returns
    -------
    np.ndarray
        Augmented candidate pool (existing + new feasible points)
    """
    rng = np.random.default_rng(seed)
    k = len(factors)
    
    n_needed = target_size - len(existing_candidates)
    if n_needed <= 0:
        return existing_candidates
    
    # Strategy: Sample uniformly in [-1, 1]^k and accept if feasible
    new_candidates = []
    attempts = 0
    accept_count = 0
    
    while accept_count < n_needed and attempts < max_attempts:
        # Generate batch of random points
        batch_size = min(100, n_needed - accept_count)
        random_points = rng.uniform(-1, 1, size=(batch_size, k))
        
        for point in random_points:
            attempts += 1
            if attempts >= max_attempts:
                break
            
            if is_feasible(point):
                new_candidates.append(point)
                accept_count += 1
                
                if accept_count >= n_needed:
                    break
    
    # Report acceptance rate
    acceptance_rate = accept_count / attempts if attempts > 0 else 0
    
    if acceptance_rate < 0.01:
        warnings.warn(
            f"Very low feasible region: {acceptance_rate:.2%} acceptance rate. "
            f"Constraints may be too restrictive. Only found {accept_count} "
            f"additional feasible points after {attempts} attempts."
        )
    
    if len(new_candidates) == 0:
        return existing_candidates
    
    # Combine and deduplicate
    augmented = np.vstack([existing_candidates, np.array(new_candidates)])
    augmented = np.unique(np.round(augmented, decimals=6), axis=0)
    
    return augmented


# ============================================================
# SECTION 5: OPTIMIZED SHERMAN-MORRISON
# ============================================================

@dataclass
class ShermanMorrisonResult:
    """Result from Sherman-Morrison update computation."""
    det_ratio: float
    XtX_inv_updated: np.ndarray
    is_valid: bool


def sherman_morrison_swap(
    XtX_inv: np.ndarray,
    x_old: np.ndarray,
    x_new: np.ndarray
) -> ShermanMorrisonResult:
    """
    Compute det ratio AND updated inverse for swapping x_old -> x_new.
    
    Parameters
    ----------
    XtX_inv : np.ndarray, shape (p, p)
        Current inverse of X'X
    x_old : np.ndarray, shape (p,)
        Row being removed (model matrix row)
    x_new : np.ndarray, shape (p,)
        Row being added (model matrix row)
    
    Returns
    -------
    ShermanMorrisonResult
        Contains: det_ratio, updated inverse, validity flag
    """
    # Validate inputs
    if XtX_inv.ndim != 2 or XtX_inv.shape[0] != XtX_inv.shape[1]:
        raise ValueError("XtX_inv must be square matrix")
    
    p = XtX_inv.shape[0]
    
    if x_old.shape != (p,) or x_new.shape != (p,):
        raise ValueError(
            f"Vectors must have shape ({p},), got x_old: {x_old.shape}, x_new: {x_new.shape}"
        )
    
    # Step 1: Remove x_old contribution
    v_old = XtX_inv @ x_old
    denom_old = 1 - x_old @ v_old
    
    # ... rest of implementation
    
    if abs(denom_old) < 1e-14:
        return ShermanMorrisonResult(
            det_ratio=0.0,
            XtX_inv_updated=XtX_inv,
            is_valid=False
        )
    
    # Update inverse after removal
    XtX_inv_1 = XtX_inv + np.outer(v_old, v_old) / denom_old
    
    # Step 2: Add x_new contribution
    v_new = XtX_inv_1 @ x_new
    denom_new = 1 + x_new @ v_new
    
    if denom_new <= 1e-14:
        return ShermanMorrisonResult(
            det_ratio=0.0,
            XtX_inv_updated=XtX_inv,
            is_valid=False
        )
    
    # Update inverse after addition
    XtX_inv_2 = XtX_inv_1 - np.outer(v_new, v_new) / denom_new
    
    # Determinant ratio
    det_ratio = denom_old * denom_new
    
    return ShermanMorrisonResult(
        det_ratio=det_ratio,
        XtX_inv_updated=XtX_inv_2,
        is_valid=True
    )


# ============================================================
# SECTION 6: FAST CEXCH OPTIMIZER WITH REUSE
# ============================================================

@dataclass
class OptimizerConfig:
    """Configuration for CEXCH optimizer."""
    
    max_iterations: int = 200
    relative_improvement_tolerance: float = 1e-4
    stability_window: int = 15
    n_random_starts: int = 3
    max_candidates_per_row: int = 50
    use_sherman_morrison: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.relative_improvement_tolerance <= 0:
            raise ValueError("relative_improvement_tolerance must be > 0")
        if self.stability_window < 1:
            raise ValueError("stability_window must be >= 1")
        if self.n_random_starts < 1:
            raise ValueError("n_random_starts must be >= 1")
        if self.max_candidates_per_row < 1:
            raise ValueError("max_candidates_per_row must be >= 1")


def cexch_optimize(
    candidates: np.ndarray,
    n_runs: int,
    model_builder: ModelMatrixBuilder,
    config: OptimizerConfig,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float, int, str]:
    """
    Fast CEXCH optimizer with Sherman-Morrison intermediate reuse.
    
    Parameters
    ----------
    candidates : np.ndarray, shape (N_cand, k)
        Feasible candidate points
    n_runs : int
        Number of runs to select
    model_builder : ModelMatrixBuilder
        Model matrix builder
    config : OptimizerConfig
        Optimization configuration
    seed : int, optional
        Random seed
    
    Returns
    -------
    best_indices : np.ndarray
        Indices into candidates
    best_logdet : float
        Final log-determinant
    n_iterations : int
        Iterations performed
    converged_by : str
        Convergence reason
    """
    rng = np.random.default_rng(seed)
    N_cand = len(candidates)
    
    if n_runs > N_cand:
        raise ValueError(f"Not enough candidates ({N_cand}) for {n_runs} runs")
    
    # Precompute model matrix for all candidates
    X_model_candidates = model_builder(candidates)
    p = X_model_candidates.shape[1]
    
    best_logdet = -np.inf
    best_indices = None
    best_n_iter = 0
    best_converged_by = ""
    
    # Multiple random starts
    for start in range(config.n_random_starts):
        # Initialize
        indices = rng.choice(N_cand, size=n_runs, replace=False)
        X_current = X_model_candidates[indices]
        
        # Compute initial X'X and its inverse
        XtX = X_current.T @ X_current + 1e-10 * np.eye(p)
        
        try:
            XtX_inv = np.linalg.inv(XtX)
            sign, logdet = np.linalg.slogdet(XtX)
        except np.linalg.LinAlgError:
            continue
        
        if sign <= 0:
            continue
        
        logdet_history = [logdet]
        no_improvement_count = 0
        converged_by = "max_iterations"
        
        for iteration in range(config.max_iterations):
            improved_this_iter = False
            
            # Try improving each row
            for i in range(n_runs):
                x_old = X_current[i]
                best_logdet_for_i = logdet
                best_idx_for_i = indices[i]
                best_sm_result = None  # Store SM result for best candidate
                
                # Select candidate subset for this row
                structured_mask = np.max(np.abs(candidates), axis=1) >= 0.99
                structured_indices = np.where(structured_mask)[0]
                
                non_structured = np.where(~structured_mask)[0]
                n_sample = min(
                    config.max_candidates_per_row - len(structured_indices),
                    len(non_structured)
                )
                
                if n_sample > 0:
                    sampled_indices = rng.choice(non_structured, size=n_sample, replace=False)
                    candidate_subset = np.concatenate([structured_indices, sampled_indices])
                else:
                    candidate_subset = structured_indices
                
                # Try candidates in subset
                for cand_idx in candidate_subset:
                    if cand_idx == indices[i]:
                        continue
                    
                    if cand_idx in np.delete(indices, i):
                        continue
                    
                    x_new = X_model_candidates[cand_idx]
                    
                    if config.use_sherman_morrison:
                        # Compute SM update ONCE (returns both ratio and inverse)
                        sm_result = sherman_morrison_swap(XtX_inv, x_old, x_new)
                        
                        if not sm_result.is_valid or sm_result.det_ratio <= 0:
                            continue
                        
                        logdet_trial = logdet + np.log(sm_result.det_ratio)
                    else:
                        # Fallback: explicit computation
                        X_trial = X_current.copy()
                        X_trial[i] = x_new
                        XtX_trial = X_trial.T @ X_trial + 1e-10 * np.eye(p)
                        
                        try:
                            sign_trial, logdet_trial = np.linalg.slogdet(XtX_trial)
                            if sign_trial <= 0:
                                continue
                            sm_result = None
                        except:
                            continue
                    
                    # Accept if better
                    if logdet_trial > best_logdet_for_i + 1e-10:
                        best_logdet_for_i = logdet_trial
                        best_idx_for_i = cand_idx
                        best_sm_result = sm_result  # Store for reuse
                
                # Apply best swap for this row
                if best_idx_for_i != indices[i]:
                    x_new = X_model_candidates[best_idx_for_i]
                    
                    # REUSE the already-computed inverse from SM
                    if config.use_sherman_morrison and best_sm_result is not None:
                        XtX_inv = best_sm_result.XtX_inv_updated
                    else:
                        # Fallback: recompute (only if not using SM)
                        X_current[i] = x_new
                        XtX = X_current.T @ X_current + 1e-10 * np.eye(p)
                        try:
                            XtX_inv = np.linalg.inv(XtX)
                        except:
                            continue
                    
                    X_current[i] = x_new
                    indices[i] = best_idx_for_i
                    logdet = best_logdet_for_i
                    improved_this_iter = True
            
            logdet_history.append(logdet)
            
            # Check stopping
            if improved_this_iter:
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            if no_improvement_count >= config.stability_window:
                converged_by = "stability"
                break
            
            # Relative improvement check
            if len(logdet_history) >= config.stability_window:
                old_val = logdet_history[-config.stability_window]
                new_val = logdet_history[-1]
                
                if abs(old_val) > 1e-10:
                    rel_improvement = (new_val - old_val) / abs(old_val)
                    
                    if rel_improvement < config.relative_improvement_tolerance:
                        converged_by = "stability"
                        break
        
        n_iter = len(logdet_history) - 1
        
        # Keep best across starts
        if logdet > best_logdet:
            best_logdet = logdet
            best_indices = indices.copy()
            best_n_iter = n_iter
            best_converged_by = converged_by
    
    return best_indices, best_logdet, best_n_iter, best_converged_by


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def code_point(x_actual: np.ndarray, factors: List[Factor]) -> np.ndarray:
    """Convert actual values to coded [-1, 1] scale."""
    x_coded = np.zeros_like(x_actual)
    for i, factor in enumerate(factors):
        center = (factor.max_value + factor.min_value) / 2
        half_range = (factor.max_value - factor.min_value) / 2
        x_coded[i] = (x_actual[i] - center) / half_range
    return x_coded


def decode_point(x_coded: np.ndarray, factors: List[Factor]) -> np.ndarray:
    """Convert coded values to actual scale."""
    x_actual = np.zeros_like(x_coded)
    for i, factor in enumerate(factors):
        center = (factor.max_value + factor.min_value) / 2
        half_range = (factor.max_value - factor.min_value) / 2
        x_actual[i] = center + x_coded[i] * half_range
    return x_actual


def compute_d_efficiency_vs_benchmark(
    det: float,
    n_runs: int,
    n_params: int,
    benchmark_det: float
) -> float:
    """
    Compute D-efficiency relative to benchmark design.
    
    Returns percentage: (det / benchmark_det)^(1/p) * 100
    """
    if det <= 0 or benchmark_det <= 0:
        return 0.0
    
    efficiency = (det / benchmark_det) ** (1 / n_params)
    efficiency_pct = efficiency * 100
    
    return round(min(efficiency_pct, 200.0), 2)  # Cap at 200%


def compute_benchmark_determinant(
    factors: List[Factor],
    model_type: Literal['linear', 'interaction', 'quadratic'],
    model_builder: ModelMatrixBuilder
) -> Tuple[float, str]:
    """
    Compute determinant of benchmark design for comparison.
    
    Uses designs constrained to [-1, 1]^k space for fair comparison:
    - Linear models: Full Factorial 2^k
    - Quadratic models: Face-Centered CCD (alpha='face_centered', stays in bounds)
    
    Returns
    -------
    det : float
        Determinant of X'X for benchmark
    design_name : str
        Name of benchmark design used
    """
    k = len(factors)
    
    if model_type == 'linear':
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
    
    elif model_type in ('interaction', 'quadratic'):
        # Benchmark: Face-Centered CCD (alpha='face_centered')
        # This stays within [-1, 1]^k bounds for fair comparison with D-optimal
        from src.core.response_surface import CentralCompositeDesign
        
        # Create temporary CCD
        temp_factors = [
            Factor(f"X{i}", FactorType.CONTINUOUS, factors[0].changeability, 
                   levels=[-1, 1])
            for i in range(k)
        ]
        
        # Use face-centered design to stay in bounds
        ccd = CentralCompositeDesign(
            factors=temp_factors,
            alpha='face',  # Face-centered: axial points on cube faces (alpha=1)
            center_points=6 if k <= 4 else 5
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


# ============================================================
# SECTION 7: HIGH-LEVEL API
# ============================================================

@dataclass
class OptimizationResult:
    """Result from optimization."""
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


def generate_d_optimal_design(
    factors: List[Factor],
    model_type: Literal['linear', 'interaction', 'quadratic'],
    n_runs: int,
    constraints: Optional[List[LinearConstraint]] = None,
    candidate_config: Optional[CandidatePoolConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    seed: Optional[int] = None
) -> OptimizationResult:
    """
    Generate D-optimal experimental design (OPTIMIZED VERSION).
    
    Uses Meyer & Nachtsheim's CEXCH with Sherman-Morrison reuse.
    Benchmarks efficiency against Full Factorial (linear) or CCD (quadratic).
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
    if model_type == 'linear':
        n_params = 1 + k
    elif model_type == 'interaction':
        n_params = 1 + k + k*(k-1)//2
    elif model_type == 'quadratic':
        n_params = 1 + k + k*(k-1)//2 + k
    
    if n_runs < n_params:
        raise ValueError(
            f"n_runs ({n_runs}) must be >= n_parameters ({n_params})"
        )
    
    # Generate candidate pool (simplified - no stratification)
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
                seed=seed
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
                seed=seed
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
    
    # Optimize
    if optimizer_config is None:
        optimizer_config = OptimizerConfig()
    
    indices, best_logdet, n_iter, converged_by = cexch_optimize(
        candidates=candidates,
        n_runs=n_runs,
        model_builder=model_builder,
        config=optimizer_config,
        seed=seed
    )
    
    # Build final design
    design_coded = candidates[indices]
    
    design_actual_array = np.array([
        decode_point(design_coded[i], factors)
        for i in range(n_runs)
    ])
    
    design_actual_df = pd.DataFrame(
        design_actual_array,
        columns=[f.name for f in factors]
    )
    design_actual_df.insert(0, 'StdOrder', range(1, n_runs + 1))
    design_actual_df.insert(1, 'RunOrder', range(1, n_runs + 1))
    
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
        det_achieved,
        n_runs,
        n_params,
        det_benchmark
    )
    
    # Warnings
    if model_type == 'linear' and d_efficiency < 90:
        warnings.warn(
            f"D-efficiency vs {benchmark_name}: {d_efficiency:.2f}%. "
            f"Expected >90% for linear models. Consider more runs or fewer constraints."
        )
    elif model_type in ('interaction', 'quadratic') and d_efficiency < 100:
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
        criterion_type='D-optimal',
        final_objective=best_logdet,
        n_iterations=n_iter,
        converged_by=converged_by,
        n_runs=n_runs,
        n_parameters=n_params,
        condition_number=condition_number,
        d_efficiency_vs_benchmark=d_efficiency,
        benchmark_design_name=benchmark_name
    )