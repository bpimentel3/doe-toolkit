"""
Constraint Handling and Feasibility Filtering for Optimal Design.

This module provides tools for defining constraints, checking feasibility,
and filtering/augmenting candidate pools based on constraints.

Classes
-------
LinearConstraint
    Linear constraint in actual factor space
FeasibilityPredicate
    Protocol for feasibility checking functions

Functions
---------
create_linear_constraint_predicate
    Create feasibility checker from constraints
filter_feasible_candidates
    Filter candidates to only feasible points
augment_constrained_candidates
    Add more feasible points via rejection sampling

Notes
-----
Constraints are defined in actual (decoded) factor space for user convenience,
then evaluated on coded points by decoding them first.

Linear constraints have the form:
    c1*x1 + c2*x2 + ... ≤ bound  (or ≥, or =)
"""

import warnings
from dataclasses import dataclass
from typing import List, Literal, Optional, Protocol

import numpy as np

from src.core.factors import Factor


# ============================================================
# FEASIBILITY PROTOCOL
# ============================================================


class FeasibilityPredicate(Protocol):
    """Protocol for feasibility checking."""

    def __call__(self, x: np.ndarray) -> bool:
        """
        Check if point is feasible.

        Parameters
        ----------
        x : np.ndarray, shape (k,)
            Point in coded space

        Returns
        -------
        bool
            True if point satisfies all constraints
        """
        ...


# ============================================================
# LINEAR CONSTRAINTS
# ============================================================


@dataclass
class LinearConstraint:
    """
    Linear constraint in actual (decoded) factor space.

    Represents constraints of the form:
        c1*x1 + c2*x2 + ... + cn*xn ≤ bound  (if constraint_type='le')
        c1*x1 + c2*x2 + ... + cn*xn ≥ bound  (if constraint_type='ge')
        c1*x1 + c2*x2 + ... + cn*xn = bound  (if constraint_type='eq')

    Attributes
    ----------
    coefficients : dict
        Mapping from factor names to coefficients
    bound : float
        Right-hand side value
    constraint_type : {'le', 'ge', 'eq'}, default='le'
        Constraint type (≤, ≥, or =)

    Examples
    --------
    >>> # Constraint: Temperature + 2*Pressure ≤ 500
    >>> constraint = LinearConstraint(
    ...     coefficients={'Temperature': 1.0, 'Pressure': 2.0},
    ...     bound=500.0,
    ...     constraint_type='le'
    ... )

    >>> # Check if point satisfies constraint
    >>> point = {'Temperature': 200, 'Pressure': 100}
    >>> constraint.evaluate(point)
    True  # 200 + 2*100 = 400 ≤ 500
    """

    coefficients: dict
    bound: float
    constraint_type: Literal["le", "ge", "eq"] = "le"

    def evaluate(self, point: dict) -> bool:
        """
        Check if point satisfies constraint.

        Parameters
        ----------
        point : dict
            Mapping from factor names to actual values

        Returns
        -------
        bool
            True if constraint is satisfied

        Notes
        -----
        Uses tolerance of 1e-10 for numerical comparisons.
        """
        lhs = sum(self.coefficients.get(k, 0) * v for k, v in point.items())

        if self.constraint_type == "le":
            return lhs <= self.bound + 1e-10
        elif self.constraint_type == "ge":
            return lhs >= self.bound - 1e-10
        elif self.constraint_type == "eq":
            return abs(lhs - self.bound) < 1e-10
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")


# ============================================================
# CONSTRAINT PREDICATE CREATION
# ============================================================


def create_linear_constraint_predicate(
    factors: List[Factor], constraints: List[LinearConstraint]
) -> FeasibilityPredicate:
    """
    Create feasibility predicate from linear constraints.

    The predicate decodes points from coded space to actual space,
    then evaluates all constraints.

    Parameters
    ----------
    factors : List[Factor]
        Factor definitions
    constraints : List[LinearConstraint]
        Linear constraints to enforce

    Returns
    -------
    FeasibilityPredicate
        Function that checks if coded point is feasible

    Examples
    --------
    >>> constraints = [
    ...     LinearConstraint({'Temp': 1, 'Pressure': 2}, 500, 'le')
    ... ]
    >>> is_feasible = create_linear_constraint_predicate(factors, constraints)
    >>> is_feasible(coded_point)
    True
    """
    from src.core.optimal.utils import decode_point

    def is_feasible(x_coded: np.ndarray) -> bool:
        x_actual = decode_point(x_coded, factors)
        point_dict = {f.name: x_actual[i] for i, f in enumerate(factors)}

        for constraint in constraints:
            if not constraint.evaluate(point_dict):
                return False

        return True

    return is_feasible


# ============================================================
# CANDIDATE FILTERING
# ============================================================


def filter_feasible_candidates(
    candidates: np.ndarray, is_feasible: FeasibilityPredicate
) -> np.ndarray:
    """
    Filter candidate pool to only feasible points.

    Parameters
    ----------
    candidates : np.ndarray, shape (N, k)
        Candidate points in coded space
    is_feasible : FeasibilityPredicate
        Feasibility checker function

    Returns
    -------
    np.ndarray, shape (M, k)
        Feasible candidates (M ≤ N)

    Examples
    --------
    >>> feasible = filter_feasible_candidates(candidates, is_feasible)
    >>> len(feasible) <= len(candidates)
    True
    """
    mask = np.array([is_feasible(candidates[i]) for i in range(len(candidates))])
    return candidates[mask]


# ============================================================
# CANDIDATE AUGMENTATION
# ============================================================


def augment_constrained_candidates(
    factors: List[Factor],
    existing_candidates: np.ndarray,
    is_feasible: FeasibilityPredicate,
    target_size: int,
    seed: Optional[int] = None,
    max_attempts: int = 10000,
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
    existing_candidates : np.ndarray, shape (M, k)
        Current feasible candidates
    is_feasible : FeasibilityPredicate
        Feasibility checker
    target_size : int
        Desired number of feasible candidates
    seed : int, optional
        Random seed for reproducibility
    max_attempts : int, default=10000
        Maximum rejection sampling attempts

    Returns
    -------
    np.ndarray, shape (N, k)
        Augmented candidate pool (existing + new feasible points)

    Warnings
    --------
    Issues warning if acceptance rate < 1%, indicating very restrictive
    constraints.

    Notes
    -----
    Strategy: Sample uniformly in [-1, 1]^k and accept if feasible.

    The function will stop early if either:
    - Target size is reached
    - Maximum attempts is reached

    Examples
    --------
    >>> augmented = augment_constrained_candidates(
    ...     factors=factors,
    ...     existing_candidates=feasible_pool,
    ...     is_feasible=is_feasible,
    ...     target_size=100,
    ...     seed=42
    ... )
    >>> len(augmented) >= len(feasible_pool)
    True
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
