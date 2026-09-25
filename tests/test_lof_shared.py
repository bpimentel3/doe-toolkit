"""
Regression tests for the shared lack-of-fit (LOF) computation.

Covers the numerical-tolerance guard that suppresses the LOF test when
replicate responses are effectively identical (pure-error variation at
floating-point noise).  Mirrors the CCR bug where bit-identical center
points produced a machine-epsilon pure-error denominator and a garbage
``F ~ 1e26`` / ``p < 0.0001`` warning.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.analysis import ANOVAAnalysis
from src.core.diagnostics.lof import (
    compute_lof_components,
    LOF_OK,
    LOF_NOT_MEANINGFUL,
    LOF_NO_REPLICATES,
)
from src.core.diagnostics.summary import _compute_lof_p_value
from src.core.factors import Factor, FactorType

#: 3-factor CCD (natural units) — the doe_design_results.csv scenario.
#: All five center runs (row 7, 8, 10, 13, 17) have response 3.92.
CCD_DATA = [
    # A,   B,        f_C,       Length
    (130.0, 15.0,     3.0,      -151.36),
    (85.0, 0.0,       5.045378491522287, 4.02),
    (85.0, -25.226892457611434, 0.0, -530.9),
    (130.0, -15.0,    -3.0,     -197.91),
    (160.6806773728343, 0.0,    0.0, 6.34),
    (85.0, 25.226892457611434, 0.0, -479.46),
    (85.0, 0.0,       0.0,      3.92),
    (85.0, 0.0,       0.0,      3.92),
    (40.0, -15.0,     -3.0,     -184.97),
    (85.0, 0.0,       0.0,      3.92),
    (130.0, -15.0,    3.0,      -198.166),
    (40.0, -15.0,     3.0,      -184.468),
    (85.0, 0.0,       0.0,      3.92),
    (130.0, 15.0,     -3.0,     -151.114),
    (85.0, 0.0,       -5.045378491522287, 3.81),
    (40.0, 15.0,      -3.0,     -170.57),
    (85.0, 0.0,       0.0,      3.92),
    (40.0, 15.0,      3.0,      -170.068),
    (9.31932262716569, 0.0,     0.0, 1.498),
]

FULL_QUADRATIC = [
    "1", "A", "B", "f_C", "I(A**2)", "I(B**2)", "I(f_C**2)",
    "A*B", "A*f_C", "B*f_C",
]


def _ccd(natural_data):
    data = np.array(natural_data, dtype=float)
    design = pd.DataFrame(data[:, :3], columns=["A", "B", "f_C"])
    response = data[:, 3]
    factors = [
        Factor(name="A", factor_type=FactorType.CONTINUOUS, levels=[40.0, 130.0]),
        Factor(name="B", factor_type=FactorType.CONTINUOUS, levels=[-15.0, 15.0]),
        Factor(name="f_C", factor_type=FactorType.CONTINUOUS, levels=[-3.0, 3.0]),
    ]
    return design, response, factors


def _fit(design, response, factors):
    analysis = ANOVAAnalysis(
        design, response, factors, response_name="Length"
    )
    return analysis.fit(FULL_QUADRATIC)


def test_degenerate_replicates_reported_not_meaningful():
    """Bit-identical center points must NOT produce a garbage F/p-value."""
    design, response, factors = _ccd(CCD_DATA)
    fit = _fit(design, response, factors)

    result = compute_lof_components(
        design, fit.residuals, factors, fit.model_terms,
        response=fit.used_response,
    )

    assert result.status == LOF_NOT_MEANINGFUL
    assert result.f_statistic is None
    assert result.p_value is None

    # Pure error must be genuinely below the noise floor even though it is
    # not bit-zero (float-arithmetic round-off across identical values).
    assert result.ss_pe is not None
    assert result.ss_pe > 0.0
    assert result.df_pe == 4

    # Old code path produced F ~ 1.6e26 / p < 1e-4 here; the p-value must be
    # absent through the diagnostics wrapper too.
    p_value = _compute_lof_p_value(
        design, fit.residuals, factors, fit.model_terms,
        response=fit.used_response,
    )
    assert p_value is None


def test_degenerate_replicates_guard_applies_in_residual_fallback():
    """Guard fires identically when pure error is built from residuals."""
    design, response, factors = _ccd(CCD_DATA)
    fit = _fit(design, response, factors)

    result = compute_lof_components(
        design, fit.residuals, factors, fit.model_terms,
        response=None,
    )

    assert result.status == LOF_NOT_MEANINGFUL
    assert result.p_value is None

    p_value = _compute_lof_p_value(
        design, fit.residuals, factors, fit.model_terms
    )
    assert p_value is None


def test_genuine_replicate_scatter_still_reports_lof():
    """Real replicate variation must preserve a meaningful F-test."""
    design, response, factors = _ccd(CCD_DATA)
    # Give one center run real scatter so pure error is well above noise.
    response = response.copy()
    response[6] = 6.92  # row 7 is a Center run
    fit = _fit(design, response, factors)

    result = compute_lof_components(
        design, fit.residuals, factors, fit.model_terms,
        response=fit.used_response,
    )

    assert result.status == LOF_OK
    assert result.f_statistic is not None and np.isfinite(result.f_statistic)
    assert result.p_value is not None
    assert 0.0 <= result.p_value <= 1.0

    p_value = _compute_lof_p_value(
        design, fit.residuals, factors, fit.model_terms,
        response=fit.used_response,
    )
    assert p_value is not None and 0.0 < p_value < 1.0


def test_no_replicates_status_when_settings_never_repeat():
    """No repeated factor settings -> no_replicates, no computation."""
    design, response, factors = _ccd(CCD_DATA)
    # Perturb every center run so no two rows share all three settings.
    design = design.copy()
    design.loc[design.index[6], "A"] = 85.1
    design.loc[design.index[7], "A"] = 85.2
    design.loc[design.index[9], "A"] = 85.3
    design.loc[design.index[12], "A"] = 85.4
    design.loc[design.index[16], "A"] = 85.5

    fit = _fit(design, response, factors)
    result = compute_lof_components(
        design, fit.residuals, factors, fit.model_terms,
        response=response,
    )

    assert result.status == LOF_NO_REPLICATES
    assert result.p_value is None