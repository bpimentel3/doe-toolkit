"""
Two-Strata Split-Plot ANOVA Module.

Implements the correct two-error-term ANOVA for split-plot designs.
Whole-plot factors are tested against the whole-plot error (MS_WP);
subplot factors and interactions are tested against the subplot error (MS_SP).

This is the approach used by JMP and Design-Expert, and is statistically
correct for designs where some factors are hard-to-change.

References
----------
.. [1] Montgomery, D.C. (2017). Design and Analysis of Experiments, 9th ed.
       Chapter 14: Nested and Split-Plot Designs.
.. [2] Goos, P. & Jones, B. (2011). Optimal Design of Experiments: A Case
       Study Approach. Chapter 11: Split-Plot Designs.
.. [3] Milliken, G.A. & Johnson, D.E. (2009). Analysis of Messy Data,
       Volume 1, 2nd ed. Chapter 5: Split-Plot Designs.
"""

import warnings
from typing import List, Dict, Optional, Tuple, Set
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

from src.core.factors import Factor, ChangeabilityLevel
from src.core.analysis_base import (
    ANOVAResults,
    build_anova_effect_summary,
    build_coefficient_significance,
    compute_actual_coefficients,
    find_effect_estimate_key,
    parse_model_term,
)


# ---------------------------------------------------------------------------
# Term classification helpers
# ---------------------------------------------------------------------------

def _classify_terms(
    model_terms: List[str],
    whole_plot_factors: Set[str],
    sub_plot_factors: Set[str],
) -> Tuple[List[str], List[str]]:
    """
    Partition model terms into whole-plot and subplot strata.

    A term belongs to the whole-plot stratum if ALL of its constituent factors
    are whole-plot (hard/very-hard) factors.  Any term that involves at least
    one easy factor belongs to the subplot stratum.

    Parameters
    ----------
    model_terms : List[str]
        Terms in patsy notation, e.g. ['Temperature', 'Time', 'Temperature*Time'].
    whole_plot_factors : Set[str]
        Factor names with HARD or VERY_HARD changeability.
    sub_plot_factors : Set[str]
        Factor names with EASY changeability.

    Returns
    -------
    wp_terms : List[str]
        Terms tested against whole-plot error.
    sp_terms : List[str]
        Terms tested against subplot error.

    Examples
    --------
    >>> wp, sp = _classify_terms(
    ...     ['Temperature', 'Pressure', 'Time', 'Catalyst', 'Temperature*Time'],
    ...     {'Temperature', 'Pressure'}, {'Time', 'Catalyst'}
    ... )
    >>> wp
    ['Temperature', 'Pressure']
    >>> sp
    ['Time', 'Catalyst', 'Temperature*Time']
    """
    wp_terms: List[str] = []
    sp_terms: List[str] = []

    for term in model_terms:
        if term == '1':
            continue
        factor_list, _ = parse_model_term(term)
        if all(f in whole_plot_factors for f in factor_list):
            wp_terms.append(term)
        else:
            sp_terms.append(term)

    return wp_terms, sp_terms


def _build_patsy_formula(
    response_name: str,
    terms: List[str],
    extra_terms: Optional[List[str]] = None,
) -> str:
    """
    Build a patsy formula string.

    Parameters
    ----------
    response_name : str
        Name of the response column.
    terms : List[str]
        Model terms to include.
    extra_terms : List[str], optional
        Additional terms to append (e.g. blocking variables).

    Returns
    -------
    str
        Patsy formula, e.g. 'Yield ~ Temperature + Pressure + C(WholePlot)'.
    """
    all_terms = [t for t in terms if t != '1']
    if extra_terms:
        all_terms += extra_terms
    rhs = ' + '.join(all_terms) if all_terms else '1'
    return f"{response_name} ~ {rhs}"


# ---------------------------------------------------------------------------
# SS extraction helpers
# ---------------------------------------------------------------------------

def _type2_ss(
    fitted_ols,
    terms: List[str],
) -> Dict[str, Tuple[float, int]]:
    """
    Extract Type II SS and df for a list of terms from a fitted OLS model.

    Parameters
    ----------
    fitted_ols : statsmodels OLS result
        A fitted OLS results object.
    terms : List[str]
        Model terms to extract (must be present in the model).

    Returns
    -------
    Dict mapping term -> (SS, df)
    """
    try:
        anova_tbl = sm.stats.anova_lm(fitted_ols, typ=2)
    except Exception as exc:
        raise RuntimeError(f"Could not compute Type II SS: {exc}") from exc

    result: Dict[str, Tuple[float, int]] = {}

    for term in terms:
        # patsy encodes interactions with ':' not '*'
        patsy_term = term.replace('*', ':')

        matched_row = None
        for row_label in anova_tbl.index:
            if row_label == patsy_term or row_label == term:
                matched_row = row_label
                break
            # Categorical encoding produces labels like 'C(Catalyst)[T.B]'
            # We want to aggregate all levels for a single factor
            if patsy_term in row_label or term in row_label:
                matched_row = row_label
                break

        if matched_row is not None and matched_row in anova_tbl.index:
            ss = float(anova_tbl.loc[matched_row, 'sum_sq'])
            df = int(anova_tbl.loc[matched_row, 'df'])
            result[term] = (ss, df)

    return result


def _aggregate_categorical_ss(
    fitted_ols,
    terms: List[str],
    data: pd.DataFrame,
) -> Dict[str, Tuple[float, int]]:
    """
    Aggregate SS across dummy-coded levels of categorical terms.

    When patsy encodes a categorical factor C(X) it creates multiple rows in
    the ANOVA table (one per non-reference level).  This function sums them so
    each logical model term has a single SS entry.

    Parameters
    ----------
    fitted_ols : statsmodels OLS result
    terms : List[str]
        Original term names (before patsy encoding).
    data : pd.DataFrame
        Analysis data (used to compute df for categorical terms).

    Returns
    -------
    Dict mapping original term -> (SS, df)
    """
    try:
        anova_tbl = sm.stats.anova_lm(fitted_ols, typ=2)
    except Exception as exc:
        raise RuntimeError(f"Could not compute SS: {exc}") from exc

    result: Dict[str, Tuple[float, int]] = {}

    for term in terms:
        patsy_term = term.replace('*', ':')
        # Collect all matching rows (handles multi-level categorical dummies)
        matching = [
            row for row in anova_tbl.index
            if row == patsy_term
            or row == term
            or (patsy_term and row.startswith(patsy_term))
            or (term and row.startswith(term))
        ]
        if matching:
            ss = float(anova_tbl.loc[matching, 'sum_sq'].sum())
            df = int(anova_tbl.loc[matching, 'df'].sum())
            result[term] = (ss, df)

    return result


# ---------------------------------------------------------------------------
# Whole-plot mean aggregation
# ---------------------------------------------------------------------------

def _build_whole_plot_means(
    data: pd.DataFrame,
    whole_plot_col: str,
    whole_plot_factors: List[str],
    response_name: str,
) -> pd.DataFrame:
    """
    Collapse the full run-level data to one row per whole-plot.

    The response for each whole-plot is the mean of all subplot runs within it.
    Whole-plot factor values are constant within each group so we take the first
    value.

    Parameters
    ----------
    data : pd.DataFrame
        Full run-level analysis data.
    whole_plot_col : str
        Name of the column identifying whole-plots (e.g. 'WholePlot').
    whole_plot_factors : List[str]
        Names of hard/very-hard factors.
    response_name : str
        Name of the response column.

    Returns
    -------
    pd.DataFrame
        One row per whole-plot with columns for whole-plot factors and the
        mean response.
    """
    agg_dict: Dict = {response_name: 'mean'}
    for f in whole_plot_factors:
        agg_dict[f] = 'first'

    # Also preserve blocking if present
    if 'Block' in data.columns:
        agg_dict['Block'] = 'first'

    wp_means = (
        data.groupby(whole_plot_col)
        .agg(agg_dict)
        .reset_index()
    )
    return wp_means


# ---------------------------------------------------------------------------
# ANOVA table assembly
# ---------------------------------------------------------------------------

def _assemble_split_plot_anova_table(
    wp_ss_dict: Dict[str, Tuple[float, int]],
    sp_ss_dict: Dict[str, Tuple[float, int]],
    ss_wp_error: float,
    df_wp_error: int,
    ss_sp_error: float,
    df_sp_error: int,
) -> pd.DataFrame:
    """
    Assemble the two-strata ANOVA table.

    Whole-plot terms are tested against MS_WP_error.
    Subplot terms (and cross-strata interactions) are tested against
    MS_SP_error.

    Parameters
    ----------
    wp_ss_dict : Dict[str, (SS, df)]
        Whole-plot term SS and df.
    sp_ss_dict : Dict[str, (SS, df)]
        Subplot term SS and df.
    ss_wp_error : float
        SS for whole-plot error.
    df_wp_error : int
        df for whole-plot error.
    ss_sp_error : float
        SS for subplot error.
    df_sp_error : int
        df for subplot error.

    Returns
    -------
    pd.DataFrame
        ANOVA table with columns: Source, SS, df, MS, F, P, Stratum.
    """
    rows = []

    ms_wp_error = ss_wp_error / df_wp_error if df_wp_error > 0 else np.nan
    ms_sp_error = ss_sp_error / df_sp_error if df_sp_error > 0 else np.nan

    # --- Whole-plot stratum terms ---
    for term, (ss, df) in wp_ss_dict.items():
        ms = ss / df if df > 0 else np.nan
        f_val = ms / ms_wp_error if ms_wp_error and ms_wp_error > 0 else np.nan
        p_val = (
            float(stats.f.sf(f_val, df, df_wp_error))
            if (not np.isnan(f_val) and df_wp_error > 0)
            else np.nan
        )
        rows.append({
            'Source': term,
            'SS': ss,
            'df': df,
            'MS': ms,
            'F': f_val,
            'P': p_val,
            'Stratum': 'Whole-Plot',
        })

    # --- Whole-plot error ---
    rows.append({
        'Source': 'WholePlot Error',
        'SS': ss_wp_error,
        'df': df_wp_error,
        'MS': ms_wp_error,
        'F': np.nan,
        'P': np.nan,
        'Stratum': 'Whole-Plot',
    })

    # --- Subplot stratum terms ---
    for term, (ss, df) in sp_ss_dict.items():
        ms = ss / df if df > 0 else np.nan
        f_val = ms / ms_sp_error if ms_sp_error and ms_sp_error > 0 else np.nan
        p_val = (
            float(stats.f.sf(f_val, df, df_sp_error))
            if (not np.isnan(f_val) and df_sp_error > 0)
            else np.nan
        )
        rows.append({
            'Source': term,
            'SS': ss,
            'df': df,
            'MS': ms,
            'F': f_val,
            'P': p_val,
            'Stratum': 'Sub-Plot',
        })

    # --- Subplot error ---
    rows.append({
        'Source': 'SubPlot Error',
        'SS': ss_sp_error,
        'df': df_sp_error,
        'MS': ms_sp_error,
        'F': np.nan,
        'P': np.nan,
        'Stratum': 'Sub-Plot',
    })

    anova_table = pd.DataFrame(rows).set_index('Source')
    return anova_table


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fit_split_plot_anova(
    data: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    response_name: str,
    whole_plot_col: str,
    design_structure: Dict,
) -> ANOVAResults:
    """
    Fit a two-strata split-plot ANOVA model.

    Uses the Yates / expected-mean-squares approach:

    1. Collapse run data to whole-plot means.
    2. Fit an OLS model on those means for whole-plot terms only.
       The residual from this model is the whole-plot error.
    3. Fit an OLS model on the full run data including all terms plus
       whole-plot (as a fixed block-like factor) to absorb whole-plot
       variation.  The residual from this model is the subplot error.
    4. Build the ANOVA table using correct F-ratios:
       - WP terms  → MS_WP_term  / MS_WP_error
       - SP terms  → MS_SP_term  / MS_SP_error

    Parameters
    ----------
    data : pd.DataFrame
        Run-level analysis data (factor columns + response + WholePlot).
    factors : List[Factor]
        All experimental factors.
    model_terms : List[str]
        Model terms in patsy notation (without intercept '1').
    response_name : str
        Name of the response column.
    whole_plot_col : str
        Column in *data* that identifies whole-plots.
    design_structure : Dict
        Output of detect_split_plot_structure().

    Returns
    -------
    ANOVAResults
        Results object with two-strata ANOVA table, coefficient estimates,
        residuals (subplot level), and diagnostic information.

    Raises
    ------
    ValueError
        If the whole-plot column is missing or no whole-plot factors exist.

    Notes
    -----
    The coefficient estimates (effect_estimates) come from the full subplot
    OLS model which absorbs whole-plot variation via a fixed WholePlot factor.
    Fitted values and residuals are subplot-level.

    For the ANOVA table the 'P' column contains p-values; significance at
    alpha=0.05 corresponds to LogWorth > ~1.3.
    """
    wp_factor_names: List[str] = design_structure['whole_plot_factors']
    sp_factor_names: List[str] = design_structure['sub_plot_factors']

    if not wp_factor_names:
        raise ValueError(
            "fit_split_plot_anova requires at least one whole-plot factor "
            "(HARD or VERY_HARD changeability)."
        )

    wp_factor_set = set(wp_factor_names)
    sp_factor_set = set(sp_factor_names)

    # Terms excluding the intercept token
    effective_terms = [t for t in model_terms if t != '1']

    wp_terms, sp_terms = _classify_terms(effective_terms, wp_factor_set, sp_factor_set)

    # ------------------------------------------------------------------
    # STRATUM 1: Whole-plot model fitted on per-whole-plot means
    # ------------------------------------------------------------------
    wp_means = _build_whole_plot_means(
        data, whole_plot_col, wp_factor_names, response_name
    )

    n_whole_plots = len(wp_means)

    if wp_terms:
        wp_formula = _build_patsy_formula(response_name, wp_terms)
        try:
            wp_fitted = ols(wp_formula, data=wp_means).fit()
            wp_ss_dict = _aggregate_categorical_ss(wp_fitted, wp_terms, wp_means)
            ss_wp_error = float(wp_fitted.ssr)
            df_wp_error = int(wp_fitted.df_resid)
        except Exception as exc:
            warnings.warn(
                f"Whole-plot OLS failed ({exc}); whole-plot error will be estimated "
                "from total whole-plot variation."
            )
            wp_ss_dict = {}
            # Fall back: total SS among whole-plot means minus grand mean
            grand_mean = wp_means[response_name].mean()
            ss_wp_error = float(
                np.sum((wp_means[response_name] - grand_mean) ** 2)
            )
            df_wp_error = n_whole_plots - 1
    else:
        wp_ss_dict = {}
        grand_mean = wp_means[response_name].mean()
        ss_wp_error = float(np.sum((wp_means[response_name] - grand_mean) ** 2))
        df_wp_error = n_whole_plots - 1

    # Guard: cannot have zero df for WP error if we want any WP tests
    if df_wp_error <= 0 and wp_terms:
        warnings.warn(
            f"Whole-plot error df = {df_wp_error}: cannot test whole-plot terms. "
            "Consider adding replicates or reducing the whole-plot model."
        )

    # ------------------------------------------------------------------
    # STRATUM 2: Subplot model on full data, absorbing whole-plot variation
    # via fixed WholePlot factor
    # ------------------------------------------------------------------
    # We absorb whole-plot variation by treating WholePlot as a fixed effect.
    # This gives us clean subplot residuals uncontaminated by WP differences.
    sp_formula = _build_patsy_formula(
        response_name,
        effective_terms,
        extra_terms=[f"C({whole_plot_col})"],
    )

    try:
        sp_fitted = ols(sp_formula, data=data).fit()
    except Exception as exc:
        raise RuntimeError(
            f"Subplot OLS model failed to fit: {exc}\n"
            f"Formula: {sp_formula}"
        ) from exc

    # Get subplot SS for sp_terms only (not the WholePlot absorption term)
    sp_ss_dict = _aggregate_categorical_ss(sp_fitted, sp_terms, data)

    ss_sp_error = float(sp_fitted.ssr)
    df_sp_error = int(sp_fitted.df_resid)

    if df_sp_error <= 0:
        warnings.warn(
            f"Subplot error df = {df_sp_error}: no degrees of freedom for error. "
            "Consider reducing model complexity or adding more runs."
        )

    # ------------------------------------------------------------------
    # Assemble two-strata ANOVA table
    # ------------------------------------------------------------------
    anova_table = _assemble_split_plot_anova_table(
        wp_ss_dict,
        sp_ss_dict,
        ss_wp_error,
        df_wp_error,
        ss_sp_error,
        df_sp_error,
    )

    # ------------------------------------------------------------------
    # Prediction model: same terms but WITHOUT C(WholePlot).
    # sp_fitted absorbs whole-plot variation for correct SS/error estimation,
    # but it cannot be used for new-data prediction because callers won't
    # have a WholePlot column.  We fit a simpler OLS here purely for
    # prediction and CI purposes (coefficients are marginal / population-
    # average estimates).
    # ------------------------------------------------------------------
    pred_formula = _build_patsy_formula(response_name, effective_terms)
    try:
        prediction_model = ols(pred_formula, data=data).fit()
    except Exception as exc:
        warnings.warn(
            f"Could not fit prediction model ({exc}). "
            "Profiler predictions will fall back to sp_fitted and may fail."
        )
        prediction_model = sp_fitted

    # ------------------------------------------------------------------
    # Coefficient estimates and fit statistics from the subplot OLS
    # (which has the WholePlot absorbed)
    # ------------------------------------------------------------------
    # Extract only the non-WholePlot params for the effect estimates table
    params = sp_fitted.params
    bse = sp_fitted.bse
    tvalues = sp_fitted.tvalues
    pvalues = sp_fitted.pvalues

    wp_col_prefix = f"C({whole_plot_col})"
    keep_mask = ~params.index.str.startswith(wp_col_prefix)

    effect_estimates = pd.DataFrame({
        'Coefficient': params[keep_mask],
        'Std_Error': bse[keep_mask],
        't_value': tvalues[keep_mask],
        'p_value': pvalues[keep_mask],
    })
    effect_estimates = compute_actual_coefficients(effect_estimates, factors)

    residuals = np.array(sp_fitted.resid)
    fitted_values = np.array(sp_fitted.fittedvalues)

    # R² computed from the subplot model with WholePlot absorbed; adjust
    # to reflect only the terms of interest (not the WP absorption columns)
    ss_total = float(
        np.sum((data[response_name] - data[response_name].mean()) ** 2)
    )
    r_squared = 1.0 - ss_sp_error / ss_total if ss_total > 0 else np.nan
    n_obs = len(data)
    n_params_interest = len(effective_terms) + 1  # +1 for intercept
    adj_r_squared = (
        1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - n_params_interest - 1)
        if n_obs > n_params_interest + 1
        else np.nan
    )
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    diagnostics: Dict = {}
    if len(residuals) <= 5000:
        try:
            stat, pval = stats.shapiro(residuals)
            diagnostics['shapiro_wilk'] = {'statistic': float(stat), 'p_value': float(pval)}
        except Exception:
            pass

    # ------------------------------------------------------------------
    # LogWorth (from ANOVA table p-values, not coefficient p-values)
    # ------------------------------------------------------------------
    logworth_rows = []
    for source, row in anova_table.iterrows():
        if source in ('WholePlot Error', 'SubPlot Error'):
            continue
        p = row['P']
        if pd.isna(p) or p <= 0:
            lw = np.nan
        elif p < 1e-16:
            lw = 16.0
        else:
            lw = -np.log10(p)
        key = find_effect_estimate_key(source, effect_estimates.index)
        coefficient = (
            effect_estimates.loc[key, 'Coefficient']
            if key is not None else np.nan
        )
        logworth_rows.append({
            'Source': source,
            'Coefficient': coefficient,
            'p_value': p,
            'LogWorth': lw,
            'Stratum': row['Stratum'],
        })

    logworth_df = pd.DataFrame(logworth_rows).set_index('Source')

    # Canonical effect tables (kept distinct: coefficient-level vs ANOVA
    # term-level).  Used by the two effect charts in the UI.
    block_names = ('Block',) if design_structure.get('has_blocking') else ()
    coefficient_significance = build_coefficient_significance(
        effect_estimates, anova_table, block_factor_names=block_names
    )
    anova_effect_summary = build_anova_effect_summary(
        anova_table, effect_estimates, block_factor_names=block_names
    )

    return ANOVAResults(
        anova_table=anova_table,
        effect_estimates=effect_estimates,
        logworth=logworth_df,
        residuals=residuals,
        fitted_values=fitted_values,
        fitted_model=prediction_model,
        diagnostics=diagnostics,
        model_terms=model_terms,
        is_split_plot=True,
        r_squared=r_squared,
        adj_r_squared=adj_r_squared,
        rmse=rmse,
        coefficient_significance=coefficient_significance,
        anova_effect_summary=anova_effect_summary,
    )
