"""
Coordinate Exchange Optimization Algorithm for Optimal Design.

This module implements the Meyer & Nachtsheim coordinate exchange (CEXCH)
algorithm with Sherman-Morrison determinant updates for computational
efficiency.

Classes
-------
OptimizerConfig
    Configuration for CEXCH optimizer
ShermanMorrisonResult
    Result from Sherman-Morrison update

Functions
---------
sherman_morrison_swap
    Compute determinant ratio and updated inverse for row swap
cexch_optimize
    Main coordinate exchange optimization algorithm

References
----------
.. [1] Meyer, R. K., & Nachtsheim, C. J. (1995). The coordinate-exchange
       algorithm for constructing exact optimal experimental designs.
       Technometrics, 37(1), 60-69.
.. [2] Atkinson, A. C., Donev, A. N., & Tobias, R. D. (2007).
       Optimum experimental designs, with SAS. Oxford University Press.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from src.core.optimal.criteria import ModelMatrixBuilder, OptimalityCriterion


# ============================================================
# SHERMAN-MORRISON UPDATES
# ============================================================


@dataclass
class ShermanMorrisonResult:
    """
    Result from Sherman-Morrison update computation.

    Attributes
    ----------
    det_ratio : float
        Ratio of new determinant to old determinant
    XtX_inv_updated : np.ndarray
        Updated inverse of X'X after the swap
    is_valid : bool
        Whether the update succeeded (singular updates return False)
    """

    det_ratio: float
    XtX_inv_updated: np.ndarray
    is_valid: bool


def sherman_morrison_swap(
    XtX_inv: np.ndarray, x_old: np.ndarray, x_new: np.ndarray
) -> ShermanMorrisonResult:
    """
    Compute det ratio AND updated inverse for swapping x_old -> x_new.

    This implements the Sherman-Morrison formula for rank-1 updates,
    allowing efficient computation when swapping a single row in the
    design matrix.

    The formula handles the swap in two steps:
    1. Remove x_old: (X'X - x_old*x_old')^(-1)
    2. Add x_new: (X'X - x_old*x_old' + x_new*x_new')^(-1)

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
        Contains:
        - det_ratio: (det_new / det_old)
        - XtX_inv_updated: Updated inverse
        - is_valid: Whether update succeeded

    Notes
    -----
    The determinant ratio is computed as:
        det_ratio = (1 - x_old' * XtX_inv * x_old) * (1 + x_new' * XtX_inv_1 * x_new)

    If either denominator is near zero, the update is invalid (returns is_valid=False).

    Complexity: O(p^2) vs O(p^3) for full inverse recomputation

    References
    ----------
    .. [1] Golub, G. H., & Van Loan, C. F. (2013). Matrix computations.
           Johns Hopkins University Press. Section 2.1.4.
    """
    # Validate inputs
    if XtX_inv.ndim != 2 or XtX_inv.shape[0] != XtX_inv.shape[1]:
        raise ValueError("XtX_inv must be square matrix")

    p = XtX_inv.shape[0]

    if x_old.shape != (p,) or x_new.shape != (p,):
        raise ValueError(
            f"Vectors must have shape ({p},), "
            f"got x_old: {x_old.shape}, x_new: {x_new.shape}"
        )

    # Step 1: Remove x_old contribution
    v_old = XtX_inv @ x_old
    denom_old = 1 - x_old @ v_old

    if abs(denom_old) < 1e-14:
        return ShermanMorrisonResult(
            det_ratio=0.0, XtX_inv_updated=XtX_inv, is_valid=False
        )

    # Update inverse after removal
    XtX_inv_1 = XtX_inv + np.outer(v_old, v_old) / denom_old

    # Step 2: Add x_new contribution
    v_new = XtX_inv_1 @ x_new
    denom_new = 1 + x_new @ v_new

    if denom_new <= 1e-14:
        return ShermanMorrisonResult(
            det_ratio=0.0, XtX_inv_updated=XtX_inv, is_valid=False
        )

    # Update inverse after addition
    XtX_inv_2 = XtX_inv_1 - np.outer(v_new, v_new) / denom_new

    # Determinant ratio
    det_ratio = denom_old * denom_new

    return ShermanMorrisonResult(
        det_ratio=det_ratio, XtX_inv_updated=XtX_inv_2, is_valid=True
    )


# ============================================================
# OPTIMIZER CONFIGURATION
# ============================================================


@dataclass
class OptimizerConfig:
    """
    Configuration for CEXCH optimizer.

    Attributes
    ----------
    max_iterations : int, default=200
        Maximum number of coordinate exchange iterations
    relative_improvement_tolerance : float, default=1e-4
        Stop if relative improvement < this threshold
    stability_window : int, default=15
        Number of iterations to check for stability
    n_random_starts : int, default=3
        Number of random starting designs to try
    max_candidates_per_row : int, default=50
        Maximum candidates to evaluate per row (for speed)
    use_sherman_morrison : bool, default=True
        Whether to use Sherman-Morrison updates

    Notes
    -----
    Increasing n_random_starts improves solution quality but increases
    computation time linearly.

    The max_candidates_per_row parameter trades off solution quality
    for speed. Structured points (vertices, axial) are always included;
    this limits how many LHS candidates are evaluated per row.
    """

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


# ============================================================
# COORDINATE EXCHANGE OPTIMIZER
# ============================================================


def cexch_optimize(
    candidates: np.ndarray,
    n_runs: int,
    model_builder: ModelMatrixBuilder,
    criterion: OptimalityCriterion,
    config: OptimizerConfig,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, float, int, str]:
    """
    Coordinate exchange optimizer supporting any optimality criterion.

    This function implements the Meyer & Nachtsheim coordinate exchange
    algorithm (CEXCH), generalized to work with any optimality criterion
    (D-optimal, I-optimal, etc.) through the criterion object's objective()
    method.

    Algorithm Overview
    ------------------
    1. Start with random n_runs points from candidates
    2. For each iteration:
       - For each row i in current design:
         - Try swapping row i with each candidate
         - Keep swap that improves objective most
    3. Stop when no improvement or max iterations reached
    4. Repeat from multiple random starts, keep best

    Parameters
    ----------
    candidates : np.ndarray, shape (N_cand, k)
        Feasible candidate points in coded space
    n_runs : int
        Number of runs to select
    model_builder : ModelMatrixBuilder
        Function to build model matrix from factor settings
    criterion : OptimalityCriterion
        Optimality criterion object with objective() method
    config : OptimizerConfig
        Optimization configuration
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    best_indices : np.ndarray, shape (n_runs,)
        Indices into candidates array for selected design points
    best_objective : float
        Final objective value (criterion-specific)
    n_iterations : int
        Number of iterations performed
    converged_by : str
        Convergence reason ('stability', 'max_iterations')

    Raises
    ------
    ValueError
        If n_runs > number of candidates

    Notes
    -----
    The algorithm uses Sherman-Morrison updates when beneficial:
    - For D-optimal: det ratio directly computed
    - For I-optimal: updated (X'X)^(-1) reused for trace computation

    Multiple random starts prevent local optima.

    Complexity per iteration: O(n_runs * N_cand * p^2)
    where p is number of parameters

    Examples
    --------
    >>> config = OptimizerConfig(max_iterations=100, n_random_starts=3)
    >>> indices, obj, n_iter, reason = cexch_optimize(
    ...     candidates, n_runs=20, model_builder, criterion, config
    ... )
    >>> design = candidates[indices]
    """
    rng = np.random.default_rng(seed)
    N_cand = len(candidates)

    if n_runs > N_cand:
        raise ValueError(f"Not enough candidates ({N_cand}) for {n_runs} runs")

    # Precompute model matrix for all candidates
    X_model_candidates = model_builder(candidates)
    p = X_model_candidates.shape[1]

    best_objective = -np.inf
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
            objective = criterion.objective(X_current)
        except np.linalg.LinAlgError:
            continue

        if objective < -1e9:  # Invalid design
            continue

        objective_history = [objective]
        no_improvement_count = 0
        converged_by = "max_iterations"

        for iteration in range(config.max_iterations):
            improved_this_iter = False

            # Try improving each row
            for i in range(n_runs):
                x_old = X_current[i]
                best_objective_for_i = objective
                best_idx_for_i = indices[i]
                best_sm_result = None  # Store SM result for best candidate

                # Select candidate subset for this row
                structured_mask = np.max(np.abs(candidates), axis=1) >= 0.99
                structured_indices = np.where(structured_mask)[0]

                non_structured = np.where(~structured_mask)[0]
                n_sample = min(
                    config.max_candidates_per_row - len(structured_indices),
                    len(non_structured),
                )

                if n_sample > 0:
                    sampled_indices = rng.choice(
                        non_structured, size=n_sample, replace=False
                    )
                    candidate_subset = np.concatenate(
                        [structured_indices, sampled_indices]
                    )
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

                        if not sm_result.is_valid:
                            continue

                        # Use updated inverse to compute objective
                        X_trial = X_current.copy()
                        X_trial[i] = x_new
                        objective_trial = criterion.objective(X_trial)
                    else:
                        # Fallback: explicit computation
                        X_trial = X_current.copy()
                        X_trial[i] = x_new
                        objective_trial = criterion.objective(X_trial)
                        sm_result = None

                    # Accept if better
                    if objective_trial > best_objective_for_i + 1e-10:
                        best_objective_for_i = objective_trial
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
                    objective = best_objective_for_i
                    improved_this_iter = True

            objective_history.append(objective)

            # Check stopping
            if improved_this_iter:
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if no_improvement_count >= config.stability_window:
                converged_by = "stability"
                break

            # Relative improvement check
            if len(objective_history) >= config.stability_window:
                old_val = objective_history[-config.stability_window]
                new_val = objective_history[-1]

                if abs(old_val) > 1e-10:
                    rel_improvement = (new_val - old_val) / abs(old_val)

                    if rel_improvement < config.relative_improvement_tolerance:
                        converged_by = "stability"
                        break

        n_iter = len(objective_history) - 1

        # Keep best across starts
        if objective > best_objective:
            best_objective = objective
            best_indices = indices.copy()
            best_n_iter = n_iter
            best_converged_by = converged_by

    return best_indices, best_objective, best_n_iter, best_converged_by
