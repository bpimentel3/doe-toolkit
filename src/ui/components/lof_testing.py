"""
Lack-of-Fit testing component for ANOVA analysis.

This module handles the computation and display of lack-of-fit tests,
which determine if the model adequately describes the data by comparing
model error to pure experimental error from replicate runs.
"""

import numpy as np
import pandas as pd
import streamlit as st

from src.core.diagnostics.lof import (
    compute_lof_components,
    LOF_OK,
    LOF_NO_REPLICATES,
    LOF_INSUFFICIENT_DF,
    LOF_NOT_MEANINGFUL,
)
from src.core.formatting import ANOVA_TABLE_FORMATTERS, format_p, format_stat_table


def display_lack_of_fit_test(
    design_filtered: pd.DataFrame,
    response_filtered: np.ndarray,
    results,
    factors,
    current_terms: list,
) -> None:
    """
    Display lack-of-fit test results.

    Parameters
    ----------
    design_filtered : pd.DataFrame
        Filtered design matrix (after row exclusions)
    response_filtered : np.ndarray
        Filtered response data
    results : ANOVAResults
        Fitted model results
    factors : list
        List of Factor objects
    current_terms : list
        Model terms included in the fit

    Notes
    -----
    The underlying computation is shared with the diagnostics engine
    (``src.core.diagnostics.lof.compute_lof_components``) so both report the
    same result and the same numerical guards.  When pure-error variation is
    effectively zero (replicate responses identical to floating-point
    precision), no F statistic or p-value is reported — the F-test is
    meaningless with a degenerate denominator — and an informational message
    is shown instead.

    Examples
    --------
    >>> display_lack_of_fit_test(design, response, results, factors, terms)
    [Displays LOF results in Streamlit]
    """
    st.divider()
    st.markdown("**Lack-of-Fit Test**")

    result = compute_lof_components(
        design_filtered,
        results.residuals,
        factors,
        current_terms,
        response=response_filtered,
    )

    if result.status == LOF_NO_REPLICATES:
        st.info(
            "Lack-of-fit test requires replicate runs (identical factor settings)"
        )
        return

    if result.status == LOF_INSUFFICIENT_DF:
        st.info("Insufficient degrees of freedom for lack-of-fit test")
        return

    if result.status == LOF_NOT_MEANINGFUL:
        st.info(
            "Lack-of-fit test unavailable: replicate variation is effectively "
            "zero (replicate responses are identical to floating-point "
            "precision), so pure error cannot be estimated. "
            "Assess model adequacy with R², RMSE, and residual plots instead."
        )
        return

    if result.status != LOF_OK:
        st.info("Lack-of-fit test not available")
        return

    # Display results table
    _display_lof_table(
        result.ss_lof, result.ss_pe, result.ss_residual,
        result.df_lof, result.df_pe, result.df_residual,
        result.ms_lof, result.ms_pe,
        result.f_statistic, result.p_value
    )

    # Display interpretation
    _display_lof_interpretation(result.p_value)


def _display_lof_table(
    ss_lof: float,
    ss_pure_error: float,
    ss_residual: float,
    df_lof: int,
    df_pure_error: int,
    df_residual: int,
    ms_lof: float,
    ms_pure_error: float,
    f_lof: float,
    p_lof: float,
) -> None:
    """Display lack-of-fit ANOVA table."""
    lof_table = pd.DataFrame(
        {
            "Source": ["Lack-of-Fit", "Pure Error", "Total Error"],
            "DF": [df_lof, df_pure_error, df_residual],
            "SS": [ss_lof, ss_pure_error, ss_residual],
            "MS": [ms_lof, ms_pure_error, ss_residual / df_residual],
            "F": [f_lof, np.nan, np.nan],
            "P": [p_lof, np.nan, np.nan],
        }
    )

    st.dataframe(
        format_stat_table(lof_table, ANOVA_TABLE_FORMATTERS),
        width='stretch',
        hide_index=True,
    )


def _display_lof_interpretation(p_lof: float) -> None:
    """Display interpretation of lack-of-fit test results."""
    if p_lof < 0.05:
        st.warning(
            f"⚠️ Lack-of-fit is significant (p = {format_p(p_lof)}). "
            "Model may be inadequate."
        )
    else:
        st.success(f"✓ No significant lack-of-fit (p = {format_p(p_lof)})")
