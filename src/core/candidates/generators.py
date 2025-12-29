"""
Candidate Pool Generation for Optimal Designs and Augmentation.

This module provides unified candidate generation used by both initial optimal
design generation and design augmentation strategies.
"""

from typing import List, Optional, Literal
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube
import itertools

from src.core.factors import Factor


@dataclass
class CandidatePoolConfig:
    """
    Configuration for candidate pool generation.
    
    Attributes
    ----------
    include_vertices : bool, default=True
        Include 2^k factorial corner points
    include_axial : bool, default=True
        Include 2k axial/star points
    include_center : bool, default=True
        Include center point
    alpha_axial : float, default=1.0
        Distance of axial points from center
    lhs_multiplier : int, default=5
        Generate n_runs * lhs_multiplier LHS points
    exclude_existing_runs : bool, default=True
        Exclude points from existing design (for augmentation)
    min_distance : float, default=0.01
        Minimum distance between points (for deduplication)
    """
    include_vertices: bool = True
    include_axial: bool = True
    include_center: bool = True
    alpha_axial: float = 1.0
    lhs_multiplier: int = 5
    exclude_existing_runs: bool = True
    min_distance: float = 0.01


def generate_vertices(k: int) -> np.ndarray:
    """
    Generate all 2^k vertices of [-1, 1]^k hypercube.
    
    Parameters
    ----------
    k : int
        Number of factors
    
    Returns
    -------
    np.ndarray, shape (2^k, k)
        Factorial corner points
    
    Examples
    --------
    >>> vertices = generate_vertices(3)
    >>> print(vertices.shape)
    (8, 3)
    >>> print(vertices[0])  # First corner
    [-1 -1 -1]
    """
    return np.array(list(itertools.product([-1, 1], repeat=k)))


def generate_axial_points(k: int, alpha: float = 1.0) -> np.ndarray:
    """
    Generate 2k axial (star) points.
    
    Parameters
    ----------
    k : int
        Number of factors
    alpha : float, default=1.0
        Distance from center (alpha=1 for face-centered)
    
    Returns
    -------
    np.ndarray, shape (2k, k)
        Axial points
    
    Examples
    --------
    >>> axial = generate_axial_points(3, alpha=1.5)
    >>> print(axial.shape)
    (6, 3)
    >>> print(axial[0])  # Positive axial in first dimension
    [1.5 0.0 0.0]
    """
    points = []
    for j in range(k):
        # Positive axial point
        point_pos = np.zeros(k)
        point_pos[j] = alpha
        points.append(point_pos)
        
        # Negative axial point
        point_neg = np.zeros(k)
        point_neg[j] = -alpha
        points.append(point_neg)
    
    return np.array(points)


def generate_candidate_pool(
    factors: List[Factor],
    n_runs: int,
    config: CandidatePoolConfig,
    existing_design: Optional[pd.DataFrame] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate candidate pool for optimal design or augmentation.
    
    Strategy:
    - Include structured points (vertices, axial, center)
    - Add space-filling points via Latin Hypercube Sampling
    - Exclude existing design points if augmenting
    - Deduplicate and return unique candidates
    
    Parameters
    ----------
    factors : List[Factor]
        Factor definitions
    n_runs : int
        Target number of runs (used to scale LHS points)
    config : CandidatePoolConfig
        Configuration for candidate generation
    existing_design : pd.DataFrame, optional
        Existing design to exclude (for augmentation)
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    np.ndarray, shape (N_candidates, k)
        Candidate points in coded [-1, 1] space
    
    Examples
    --------
    >>> from src.core.factors import Factor, FactorType, ChangeabilityLevel
    >>> 
    >>> factors = [
    ...     Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1]),
    ...     Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, [-1, 1])
    ... ]
    >>> 
    >>> config = CandidatePoolConfig(lhs_multiplier=5)
    >>> candidates = generate_candidate_pool(factors, n_runs=10, config=config)
    >>> print(f"Generated {len(candidates)} candidates")
    
    Notes
    -----
    For augmentation, pass existing_design to avoid regenerating points
    already in the design.
    """
    k = len(factors)
    candidates_list = []
    
    # 1. Structured points
    if config.include_vertices:
        vertices = generate_vertices(k)
        candidates_list.append(vertices)
    
    if config.include_axial:
        axial = generate_axial_points(k, config.alpha_axial)
        candidates_list.append(axial)
    
    if config.include_center:
        center = np.zeros((1, k))
        candidates_list.append(center)
    
    # 2. Space-filling points via LHS
    n_lhs = n_runs * config.lhs_multiplier
    sampler = LatinHypercube(d=k, seed=seed)
    lhs_01 = sampler.random(n=n_lhs)
    lhs_coded = 2 * lhs_01 - 1  # Scale to [-1, 1]
    candidates_list.append(lhs_coded)
    
    # 3. Combine all candidates
    candidates = np.vstack(candidates_list)
    
    # 4. Deduplicate (round to avoid floating point issues)
    candidates = np.unique(np.round(candidates, decimals=6), axis=0)
    
    # 5. Exclude existing design points if augmenting
    if existing_design is not None and config.exclude_existing_runs:
        candidates = _exclude_existing_points(
            candidates, existing_design, factors, config.min_distance
        )
    
    return candidates


def _exclude_existing_points(
    candidates: np.ndarray,
    existing_design: pd.DataFrame,
    factors: List[Factor],
    min_distance: float
) -> np.ndarray:
    """
    Exclude candidate points that are too close to existing design points.
    
    Parameters
    ----------
    candidates : np.ndarray
        Candidate points in coded space
    existing_design : pd.DataFrame
        Existing design matrix
    factors : List[Factor]
        Factor definitions
    min_distance : float
        Minimum Euclidean distance threshold
    
    Returns
    -------
    np.ndarray
        Candidates excluding near-duplicates of existing points
    """
    # Extract existing points in coded space
    factor_names = [f.name for f in factors]
    existing_points = existing_design[factor_names].values
    
    # Round for consistency
    existing_points = np.round(existing_points, decimals=6)
    
    # Find candidates that are far enough from all existing points
    keep_mask = np.ones(len(candidates), dtype=bool)
    
    for i, candidate in enumerate(candidates):
        # Compute distances to all existing points
        distances = np.linalg.norm(existing_points - candidate, axis=1)
        
        # If any existing point is too close, exclude this candidate
        if np.any(distances < min_distance):
            keep_mask[i] = False
    
    return candidates[keep_mask]


def generate_augmentation_candidates(
    factors: List[Factor],
    original_design: pd.DataFrame,
    n_candidates: int,
    focus_regions: Optional[List[Literal['boundary', 'center', 'unexplored']]] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate candidates specifically for augmentation.
    
    This is a convenience wrapper that:
    - Automatically excludes existing design points
    - Can bias candidates toward specific regions
    
    Parameters
    ----------
    factors : List[Factor]
        Factor definitions
    original_design : pd.DataFrame
        Existing design
    n_candidates : int
        Target number of candidates
    focus_regions : List[str], optional
        Regions to emphasize:
        - 'boundary': Edge of design space
        - 'center': Interior regions
        - 'unexplored': Regions with no nearby runs
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Candidate points for augmentation
    
    Examples
    --------
    >>> # Generate candidates avoiding existing design
    >>> candidates = generate_augmentation_candidates(
    ...     factors, original_design, n_candidates=100
    ... )
    >>> 
    >>> # Focus on boundary regions
    >>> boundary_candidates = generate_augmentation_candidates(
    ...     factors, original_design, n_candidates=100,
    ...     focus_regions=['boundary']
    ... )
    
    Notes
    -----
    Focus regions are implemented via biased sampling. Future versions
    could use prediction variance maps to focus on high-variance regions.
    """
    k = len(factors)
    n_runs_estimate = len(original_design)
    
    # Create config with augmentation defaults
    config = CandidatePoolConfig(
        include_vertices=True,
        include_axial=True,
        include_center=True,
        alpha_axial=1.0,
        lhs_multiplier=max(5, n_candidates // n_runs_estimate),
        exclude_existing_runs=True,
        min_distance=0.01
    )
    
    # Generate base candidates
    candidates = generate_candidate_pool(
        factors=factors,
        n_runs=n_runs_estimate,
        config=config,
        existing_design=original_design,
        seed=seed
    )
    
    # Apply focus region biasing if requested
    if focus_regions:
        candidates = _bias_candidates_to_regions(
            candidates, focus_regions, k, seed
        )
    
    # If we have too many, subsample
    if len(candidates) > n_candidates:
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(candidates), size=n_candidates, replace=False)
        candidates = candidates[indices]
    
    return candidates


def _bias_candidates_to_regions(
    candidates: np.ndarray,
    focus_regions: List[str],
    k: int,
    seed: Optional[int]
) -> np.ndarray:
    """
    Bias candidate pool toward specific regions.
    
    Parameters
    ----------
    candidates : np.ndarray
        Candidate points
    focus_regions : List[str]
        Regions to emphasize
    k : int
        Number of factors
    seed : int, optional
        Random seed
    
    Returns
    -------
    np.ndarray
        Biased candidate pool
    """
    if 'boundary' in focus_regions:
        # Add more points near boundaries (|x_i| close to 1 for some i)
        rng = np.random.default_rng(seed)
        n_boundary = len(candidates) // 2
        
        # Generate boundary-focused points
        boundary_points = []
        for _ in range(n_boundary):
            point = rng.uniform(-1, 1, size=k)
            # Push one random dimension to boundary
            dim = rng.integers(0, k)
            point[dim] = rng.choice([-1, 1])
            boundary_points.append(point)
        
        candidates = np.vstack([candidates, np.array(boundary_points)])
        candidates = np.unique(np.round(candidates, decimals=6), axis=0)
    
    if 'center' in focus_regions:
        # Add more points near center (all |x_i| small)
        rng = np.random.default_rng(seed)
        n_center = len(candidates) // 3
        
        # Generate center-focused points (within [-0.5, 0.5]^k)
        center_points = rng.uniform(-0.5, 0.5, size=(n_center, k))
        
        candidates = np.vstack([candidates, center_points])
        candidates = np.unique(np.round(candidates, decimals=6), axis=0)
    
    # 'unexplored' would require existing design analysis
    # Defer to future enhancement
    
    return candidates


def evaluate_candidate_pool_quality(
    candidates: np.ndarray,
    n_runs: int
) -> dict:
    """
    Evaluate quality of candidate pool.
    
    Parameters
    ----------
    candidates : np.ndarray
        Candidate points
    n_runs : int
        Target number of runs
    
    Returns
    -------
    dict
        Quality metrics:
        - 'n_candidates': Number of candidates
        - 'density': Candidates per run
        - 'coverage': Fraction of design space covered
        - 'min_distance': Minimum distance between candidates
    
    Examples
    --------
    >>> metrics = evaluate_candidate_pool_quality(candidates, n_runs=20)
    >>> print(f"Density: {metrics['density']:.1f} candidates per run")
    """
    n_candidates, k = candidates.shape
    
    # Density
    density = n_candidates / n_runs if n_runs > 0 else 0
    
    # Coverage (crude estimate: fraction of hypercube volume covered)
    # Use convex hull volume (simplified: just range in each dimension)
    ranges = np.ptp(candidates, axis=0)
    coverage = np.mean(ranges / 2.0)  # Normalize by full range [-1, 1]
    
    # Minimum distance between candidates
    if len(candidates) > 1:
        from scipy.spatial.distance import pdist
        distances = pdist(candidates)
        min_distance = np.min(distances) if len(distances) > 0 else 0.0
    else:
        min_distance = np.inf
    
    return {
        'n_candidates': n_candidates,
        'density': density,
        'coverage': coverage,
        'min_distance': min_distance
    }