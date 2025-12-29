"""
Candidate Pool Generation Module.

Unified candidate generation for optimal designs and design augmentation.
"""

from src.core.candidates.generators import (
    CandidatePoolConfig,
    generate_vertices,
    generate_axial_points,
    generate_candidate_pool,
    generate_augmentation_candidates,
    evaluate_candidate_pool_quality
)

__all__ = [
    'CandidatePoolConfig',
    'generate_vertices',
    'generate_axial_points',
    'generate_candidate_pool',
    'generate_augmentation_candidates',
    'evaluate_candidate_pool_quality',
]