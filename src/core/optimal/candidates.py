"""
Candidate Pool Generation for Optimal Design.

This module provides functions to generate candidate point pools for
coordinate exchange optimization. Candidate pools combine structured
points (vertices, axial) with space-filling designs (LHS).

Classes
-------
CandidatePoolConfig
    Configuration for candidate pool generation

Functions
---------
generate_vertices
    Generate factorial vertices of hypercube
generate_axial_points
    Generate axial (star) points
generate_candidate_pool
    Generate complete candidate pool

Notes
-----
The candidate pool strategy uses:
- Factorial vertices (2^k points) for boundary coverage
- Axial points (2k points) for factor screening
- Latin Hypercube Sampling for interior fill
- No stratification by boundary/interior (lets optimizer decide)
"""

from dataclasses import dataclass
from itertools import product
from typing import List, Optional

import numpy as np
from scipy.stats.qmc import LatinHypercube

from src.core.factors import Factor


@dataclass
class CandidatePoolConfig:
    """
    Configuration for candidate pool generation.

    Attributes
    ----------
    lhs_multiplier : int, default=5
        Generate n_runs * lhs_multiplier LHS points
    include_vertices : bool, default=True
        Include factorial vertices (2^k points)
    include_axial : bool, default=True
        Include axial/star points (2k points)
    include_center : bool, default=True
        Include center point
    alpha_axial : float, default=1.0
        Distance for axial points (1.0 = on cube faces)
    """

    lhs_multiplier: int = 5
    include_vertices: bool = True
    include_axial: bool = True
    include_center: bool = True
    alpha_axial: float = 1.0


def generate_vertices(k: int) -> np.ndarray:
    """
    Generate all 2^k vertices of [-1, 1]^k hypercube.

    Parameters
    ----------
    k : int
        Number of factors (dimensions)

    Returns
    -------
    np.ndarray, shape (2^k, k)
        Factorial vertices with all combinations of -1 and +1

    Examples
    --------
    >>> vertices = generate_vertices(3)
    >>> vertices.shape
    (8, 3)
    >>> vertices
    array([[-1, -1, -1],
           [-1, -1,  1],
           ...
           [ 1,  1,  1]])
    """
    return np.array(list(product([-1, 1], repeat=k)))


def generate_axial_points(k: int, alpha: float = 1.0) -> np.ndarray:
    """
    Generate 2k axial (star) points.

    Axial points lie along factor axes, providing coverage
    for individual factor effects.

    Parameters
    ----------
    k : int
        Number of factors
    alpha : float, default=1.0
        Distance from center (1.0 = on cube faces, >1.0 = outside cube)

    Returns
    -------
    np.ndarray, shape (2k, k)
        Axial points along each axis

    Examples
    --------
    >>> axial = generate_axial_points(3, alpha=1.0)
    >>> axial.shape
    (6, 3)
    >>> axial
    array([[ 1.,  0.,  0.],
           [-1.,  0.,  0.],
           [ 0.,  1.,  0.],
           [ 0., -1.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  0., -1.]])
    """
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
    seed: Optional[int] = None,
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

    Parameters
    ----------
    factors : List[Factor]
        Factor definitions
    n_runs : int
        Target number of runs (used to scale LHS points)
    config : CandidatePoolConfig
        Configuration for pool generation
    seed : int, optional
        Random seed for LHS generation

    Returns
    -------
    np.ndarray, shape (N_cand, k)
        Candidate points in coded [-1, 1]^k space

    Notes
    -----
    Total candidates ≈ 2^k + 2k + 1 + (n_runs * lhs_multiplier)

    For k=4, n_runs=20, lhs_multiplier=5:
    Total ≈ 16 + 8 + 1 + 100 = 125 candidates

    Examples
    --------
    >>> config = CandidatePoolConfig(lhs_multiplier=5)
    >>> candidates = generate_candidate_pool(factors, n_runs=20, config=config)
    >>> candidates.shape
    (125, 4)  # Approximate, after deduplication
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
