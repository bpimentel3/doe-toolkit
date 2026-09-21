"""Response-surface config helpers (Streamlit-free).

Shared between Step 3 (Choose Design) and Step 4 (Preview/Generate) so the
Box-Behnken / CCD variant and the CCD alpha selection are decided from the
same data instead of drifting apart (the old Step 4 code routed every
"Response Surface" choice to a CCD via an ``rsd_variant`` session key that
was never written).
"""

from typing import Optional


def alpha_for_label(label: Optional[str]) -> str:
    """Map the CCD axial-distance dropdown label to the core's semantic value."""
    if not label:
        return 'rotatable'
    if 'Face-centered' in label:
        return 'face'
    if 'Orthogonal' in label:
        return 'orthogonal'
    return 'rotatable'


def resolve_rsm_variant(design_type: str) -> str:
    """Resolve which response-surface generator the design type needs.

    Returns ``'box_behnken'`` or ``'ccd'`` based on the Step 3 design-type
    label (``"Response Surface (Box-Behnken)"`` / ``"Response Surface (CCD)"``).
    """
    return 'box_behnken' if 'Box-Behnken' in design_type else 'ccd'