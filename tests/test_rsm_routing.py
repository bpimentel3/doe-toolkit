"""
Regression tests for the Step 3 -> Step 4 response-surface routing fix.

Streamlit's Step 4 previously selected the design generator from an
``rsd_variant`` session key that nothing ever wrote, so every "Response
Surface" choice (CCD *and* Box-Behnken) silently generated a Central
Composite Design. The fix routes on the Step 3 design-type label instead.
"""

import pytest

from src.core.factors import ChangeabilityLevel, Factor, FactorType
from src.core.response_surface import BoxBehnkenDesign, CentralCompositeDesign
from src.ui.utils.rsm_config import alpha_for_label, resolve_rsm_variant


def _factors(n: int):
    return [
        Factor(f'F{i}', FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
        for i in range(1, n + 1)
    ]


class TestResolveRsmVariant:
    def test_box_behnken_label_routes_to_box_behnken(self):
        assert resolve_rsm_variant('Response Surface (Box-Behnken)') == 'box_behnken'

    def test_ccd_label_routes_to_ccd(self):
        assert resolve_rsm_variant('Response Surface (CCD)') == 'ccd'


class TestAlphaForLabel:
    def test_rotatable_default(self):
        assert alpha_for_label('Rotatable') == 'rotatable'
        assert alpha_for_label(None) == 'rotatable'

    def test_face_centered(self):
        assert alpha_for_label('Face-centered (α=1)') == 'face'

    def test_orthogonal(self):
        assert alpha_for_label('Orthogonal') == 'orthogonal'


class TestStep4Routing:
    """The actual generator chosen by the Step 4 dispatcher must match the
    Step 3 design-type label — the scenario the old rsd_variant bug broke."""

    def _routed_variant(self, design_type):
        # Mirrors 4_preview_design.py: route on the resolved variant label.
        if resolve_rsm_variant(design_type) == 'box_behnken':
            return BoxBehnkenDesign
        return CentralCompositeDesign

    def test_box_behnken_selection_generates_box_behnken(self):
        design_type = 'Response Surface (Box-Behnken)'
        generator = self._routed_variant(design_type)
        design = generator(_factors(3), center_points=3).generate()
        # BBD: 2k(k-1) + center points = 15; no axial (±alpha) points.
        assert len(design) == 15
        factor_cols = [c for c in design.columns
                       if c not in ('StdOrder', 'RunOrder', 'PointType')]
        for col in factor_cols:
            assert (design[col].abs() != 1.0).any()  # every factor is centered somewhere

    def test_ccd_selection_generates_ccd(self):
        design_type = 'Response Surface (CCD)'
        generator = self._routed_variant(design_type)
        design = generator(_factors(3), alpha='rotatable', center_points=6).generate()
        assert len(design) == 8 + 6 + 6  # factorial + axial + center
        # CCD includes axial points outside the ±1 cube.
        assert (design['PointType'] == 'Axial').sum() == 6

    def test_box_behnken_is_not_a_ccd_with_axial_points(self):
        """A real BBD must not contain point rows that look like CCD axial
        points (one factor at ±alpha while every other factor is at center)."""
        design = BoxBehnkenDesign(_factors(4), center_points=3).generate()
        factor_cols = [c for c in design.columns
                       if c not in ('StdOrder', 'RunOrder', 'PointType')]
        alpha_like = 0
        for _, row in design[factor_cols].iterrows():
            non_zero = row[row != 0]
            if len(non_zero) == 1 and abs(non_zero.iloc[0]) > 1.0:
                alpha_like += 1
        assert alpha_like == 0