"""
Shared Lack-of-Fit (LOF) computation and numerical guards.

Single implementation used by both the Analyze-page display
(``src.ui.components.lof_testing``) and the diagnostics engine
(``src.core.diagnostics.summary``) so they stay consistent.

The classical LOF F-test partitions the residual sum of squares:

- **Pure Error (PE)** — variation among true replicates (runs with
  identical factor settings).  This is model-independent and estimated
  from the raw *response* values.
- **Lack of Fit (LOF)** — residual SS beyond pure error.

``F = MS_LOF / MS_PE`` under ``H0`` (model adequate).  When replicate
responses are effectively identical (pure-error variance at or below
floating-point noise), the denominator carries no information and the
ratio is meaningless, so the computation is guarded and returns a
``not_meaningful`` status instead of a garbage F/p.
"""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.core.factors import Factor

#: Status constants for :class:`LOFResult`.
LOF_OK = 'ok'
LOF_NO_REPLICATES = 'no_replicates'
LOF_INSUFFICIENT_DF = 'insufficient_df'
LOF_NOT_MEANINGFUL = 'not_meaningful'

#: Tolerance factor used to detect "effectively zero" pure error.  Pure error
#: is deemed meaningless when its mean square is at or below machine-epsilon
#: times the response variance scale (i.e. the replicate differences are at
#: the rounding floor of the recorded values).
_TOL_SCALE = np.finfo(float).eps


@dataclass
class LOFResult:
    """
    Result of a lack-of-fit test.

    ``status`` is one of:
    - ``'ok'`` — a meaningful F-test was computed (``f_statistic`` /
      ``p_value`` populated).
    - ``'no_replicates'`` — no replicate settings found.
    - ``'insufficient_df'`` — not enough residual/LOF degrees of freedom.
    - ``'not_meaningful'`` — pure error is effectively zero; no F or p is
      reported (``f_statistic``/``p_value`` are ``None``).
    """

    status: str
    ss_residual: Optional[float] = None
    df_residual: Optional[int] = None
    ss_pe: Optional[float] = None
    df_pe: Optional[int] = None
    ss_lof: Optional[float] = None
    df_lof: Optional[int] = None
    ms_lof: Optional[float] = None
    ms_pe: Optional[float] = None
    f_statistic: Optional[float] = None
    p_value: Optional[float] = None


def _is_meaningful_pure_error(ss_pe: float, df_pe: int, response_scale_sq: float) -> bool:
    """
    True when the pure-error estimate carries real information.

    ``ss_pe`` is meaningful only if it exceeds the floating-point noise floor
    of the measured response: ``ms_pe = ss_pe / df_pe`` must be greater than
    ``eps * response_scale_sq``.  Below that, replicate differences are merely
    rounding of the recorded values (e.g. identical center points), so any
    F-ratio built on them is numerical noise, not a test statistic.
    """
    if ss_pe <= 0.0 or df_pe <= 0:
        return False
    if response_scale_sq <= 0.0:
        return False
    ms_pe = ss_pe / df_pe
    return ms_pe > _TOL_SCALE * response_scale_sq


def compute_lof_components(
    design: pd.DataFrame,
    residuals: np.ndarray,
    factors: List[Factor],
    model_terms: List[str],
    response: Optional[np.ndarray] = None,
) -> LOFResult:
    """
    Compute the lack-of-fit test components with numerical guards.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix (natural units) containing the factor columns used to
        group replicate runs.
    residuals : np.ndarray
        Residuals of the fitted model (length ``n``).
    factors : List[Factor]
        Factor definitions used to identify the factor columns in *design*.
    model_terms : List[str]
        Fitted model terms.  ``'1'`` counts as one parameter, so
        ``n_params = len(model_terms)``.
    response : np.ndarray, optional
        Observed response values (length ``n``).  Pure error is estimated
        from the raw response variation *within* replicate groups (the
        model-independent definition).  When omitted, the residuals are used
        instead (legacy fallback); the numerical guards apply identically.

    Returns
    -------
    LOFResult
        Components plus a status flag.  ``f_statistic``/``p_value`` are only
        populated for ``status == 'ok'``.
    """
    factor_names = [f.name for f in factors]
    available_cols = [c for c in factor_names if c in design.columns]
    missing = LOFResult(status=LOF_NO_REPLICATES)

    if not available_cols:
        return missing

    residuals_arr = np.asarray(residuals, dtype=float)
    n = len(residuals_arr)
    if n == 0 or len(design) != n:
        return missing

    # Pure-error basis: raw response when available, else residuals (fallback).
    base = None
    if response is not None:
        base = np.asarray(response, dtype=float)
        if len(base) != n:
            base = None
    if base is None:
        base = residuals_arr

    factor_data = design[available_cols].copy()
    has_nan = factor_data.isnull().any().any() or bool(np.isnan(base).any())
    if has_nan:
        return missing

    # Round continuous column values to suppress floating-point noise in
    # nominally identical replicate settings.
    for f in factors:
        if f.is_continuous() and f.name in factor_data.columns:
            factor_data[f.name] = factor_data[f.name].round(6)

    group_labels = factor_data.apply(
        lambda row: tuple(row[c] for c in available_cols), axis=1
    )

    ss_resid = float(np.sum(residuals_arr ** 2))

    # Pure error: within-group SS of the base (response/residual) around the
    # group mean.
    ss_pe = 0.0
    df_pe = 0
    for _key, group_idx in group_labels.groupby(group_labels).groups.items():
        group_base = base[group_idx]
        if len(group_base) < 2:
            continue
        group_mean = float(np.mean(group_base))
        ss_pe += float(np.sum((group_base - group_mean) ** 2))
        df_pe += len(group_base) - 1

    if df_pe == 0:
        return missing

    n_params = len(model_terms)
    df_resid = n - n_params
    if df_resid <= 0:
        return LOFResult(status=LOF_INSUFFICIENT_DF)

    df_lof = df_resid - df_pe
    if df_lof <= 0:
        return LOFResult(status=LOF_INSUFFICIENT_DF)

    ss_lof = float(max(ss_resid - ss_pe, 0.0))

    # Variance scale of the measured quantity, used as the numerical floor.
    response_scale_sq = float(np.mean(base ** 2))

    if not _is_meaningful_pure_error(ss_pe, df_pe, response_scale_sq):
        return LOFResult(
            status=LOF_NOT_MEANINGFUL,
            ss_residual=ss_resid,
            df_residual=df_resid,
            ss_pe=ss_pe,
            df_pe=df_pe,
            ss_lof=ss_lof,
            df_lof=df_lof,
        )

    ms_lof = ss_lof / df_lof
    ms_pe = ss_pe / df_pe
    if ms_pe <= 0.0:
        return LOFResult(
            status=LOF_NOT_MEANINGFUL,
            ss_residual=ss_resid,
            df_residual=df_resid,
            ss_pe=ss_pe,
            df_pe=df_pe,
            ss_lof=ss_lof,
            df_lof=df_lof,
        )

    f_stat = ms_lof / ms_pe
    p_value = float(stats.f.sf(f_stat, df_lof, df_pe))

    return LOFResult(
        status=LOF_OK,
        ss_residual=ss_resid,
        df_residual=df_resid,
        ss_pe=ss_pe,
        df_pe=df_pe,
        ss_lof=ss_lof,
        df_lof=df_lof,
        ms_lof=ms_lof,
        ms_pe=ms_pe,
        f_statistic=f_stat,
        p_value=p_value,
    )