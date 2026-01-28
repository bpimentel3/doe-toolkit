"""
Lack-of-Fit testing component for ANOVA analysis.

This module handles the computation and display of lack-of-fit tests,
which determine if the model adequately describes the data by comparing
model error to pure experimental error from replicate runs.
"""

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats


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
    Lack-of-fit test requires:
    - Replicate runs (identical factor settings with different responses)
    - At least 1 degree of freedom for lack-of-fit
    - At least 1 degree of freedom for pure error

    The test compares:
    - H0: Model is adequate (lack-of-fit is not significant)
    - H1: Model is inadequate (lack-of-fit is significant)

    Examples
    --------
    >>> display_lack_of_fit_test(design, response, results, factors, terms)
    [Displays LOF table and interpretation in Streamlit]
    """
    st.divider()
    st.markdown("**Lack-of-Fit Test**")

    # Calculate degrees of freedom
    n_runs = len(design_filtered)
    n_params = len([t for t in current_terms if t != "1"]) + 1
    df_residual = n_runs - n_params

    # Check for pure replicates
    factor_cols = [f.name for f in factors]
    duplicates = design_filtered[factor_cols].duplicated(keep=False)
    has_replicates = duplicates.any()

    if not has_replicates or df_residual <= 0:
        st.info(
            "Lack-of-fit test requires replicate runs (identical factor settings)"
        )
        return

    # Calculate pure error from replicates
    lof_results = _calculate_lof_components(
        design_filtered, response_filtered, results, factor_cols
    )

    if lof_results is None:
        st.info("No pure replicates available for lack-of-fit test")
        return

    ss_pure_error, df_pure_error = lof_results

    if df_pure_error == 0:
        st.info("No pure replicates available for lack-of-fit test")
        return

    # Calculate lack-of-fit
    ss_residual = np.sum(results.residuals**2)
    ss_lof = ss_residual - ss_pure_error
    df_lof = df_residual - df_pure_error

    if df_lof <= 0:
        st.info("Insufficient degrees of freedom for lack-of-fit test")
        return

    # Compute F-test
    ms_lof = ss_lof / df_lof
    ms_pure_error = ss_pure_error / df_pure_error
    f_lof = ms_lof / ms_pure_error
    p_lof = 1 - stats.f.cdf(f_lof, df_lof, df_pure_error)

    # Display results table
    _display_lof_table(
        ss_lof, ss_pure_error, ss_residual,
        df_lof, df_pure_error, df_residual,
        ms_lof, ms_pure_error,
        f_lof, p_lof
    )

    # Display interpretation
    _display_lof_interpretation(p_lof)


def _calculate_lof_components(
    design_filtered: pd.DataFrame,
    response_filtered: np.ndarray,
    results,
    factor_cols: list,
) -> tuple:
    """
    Calculate pure error components from replicates.

    Parameters
    ----------
    design_filtered : pd.DataFrame
        Filtered design matrix
    response_filtered : np.ndarray
        Filtered response data
    results : ANOVAResults
        Fitted model results
    factor_cols : list
        List of factor column names

    Returns
    -------
    tuple or None
        (ss_pure_error, df_pure_error) or None if no replicates
    """
    unique_settings = design_filtered[factor_cols].drop_duplicates()

    ss_pure_error = 0
    df_pure_error = 0

    for idx, row in unique_settings.iterrows():
        # Find all runs with this setting
        mask = (design_filtered[factor_cols] == row).all(axis=1)
        replicate_responses = response_filtered[mask]

        if len(replicate_responses) > 1:
            # Pure error from replicates
            ss_pure_error += np.sum(
                (replicate_responses - replicate_responses.mean()) ** 2
            )
            df_pure_error += len(replicate_responses) - 1

    if df_pure_error == 0:
        return None

    return ss_pure_error, df_pure_error


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

    st.dataframe(lof_table, use_container_width=True, hide_index=True)


def _display_lof_interpretation(p_lof: float) -> None:
    """Display interpretation of lack-of-fit test results."""
    if p_lof < 0.05:
        st.warning(
            f"⚠️ Lack-of-fit is significant (p = {p_lof:.4f}). "
            "Model may be inadequate."
        )
    else:
        st.success(f"✓ No significant lack-of-fit (p = {p_lof:.4f})")
