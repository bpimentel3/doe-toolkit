"""
Interaction Plot component for ANOVA analysis.

Stat-Ease / Design-Expert style interaction plots.  For a selected pair of
factors ``A`` and ``B`` it draws one line per level of the grouping factor
across the levels of the x-axis factor, using per-combination observed means.
Parallel lines indicate little to no interaction; crossing / non-parallel
lines indicate an interaction.

Two renderings are produced for every pair (``A`` on the x-axis with ``B``
as the grouping factor, and the swapped orientation) so the user can inspect
the interaction from both points of view.

The heavy lifting is done by the pure, testable functions
:func:`interaction_stats` and :func:`create_interaction_plot` in
``src/ui/utils/plotting.py``; this module only wires up Streamlit controls
and renders the resulting figures.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.ui.utils.plotting import (
    interaction_stats,
    create_interaction_plot,
)

# Error-bar display modes offered to the user.  Keys match the values the
# plot builder understands.
_ERROR_MODES: List[str] = ["Mean only", "Mean ± SD", "Mean ± CI"]
_ERROR_MODE_MAP: Dict[str, str] = {
    "Mean only": "none",
    "Mean ± SD": "sd",
    "Mean ± CI": "ci",
}


def _interaction_pvalue(results, f1_name: str, f2_name: str):
    """
    Look up the ANOVA interaction-term p-value for a factor pair.

    Returns ``(p_value, present)``.  ``present`` is ``False`` when the
    interaction term is not part of the fitted model, in which case the caller
    should show "not in model" rather than an error.  Handles both standard
    statsmodels tables (``PR(>F)`` column) and split-plot tables (``P``).
    """
    if results is None or results.anova_table is None or results.anova_table.empty:
        return None, False

    key = f"{f1_name}:{f2_name}"
    reversed_key = f"{f2_name}:{f1_name}"
    index_names = [str(i) for i in results.anova_table.index]
    if key not in index_names and reversed_key not in index_names:
        return None, False

    search = key if key in index_names else reversed_key
    row_idx = results.anova_table.index[index_names.index(search)]
    col = "PR(>F)" if "PR(>F)" in results.anova_table.columns else "P"
    val = results.anova_table.loc[row_idx, col]
    if pd.isna(val) or not np.isfinite(float(val)):
        return None, True
    return float(val), True


def _factor_units_map(factors) -> Dict[str, Optional[str]]:
    return {f.name: getattr(f, "units", None) for f in factors}


def display_interaction_plot_tab(
    selected_response: str,
    design: pd.DataFrame,
    response: np.ndarray,
    factors: List,
    results=None,
    response_units: Optional[str] = None,
) -> None:
    """
    Display the Interaction Plots tab content.

    Parameters
    ----------
    selected_response : str
        Name of the currently selected response.
    design : pd.DataFrame
        Filtered design data (natural units) aligned with ``response``.
    response : np.ndarray
        Response values aligned with ``design`` rows.
    factors : List[Factor]
        Factor definitions.
    results : ANOVAResults, optional
        Fitted model results used to source the interaction p-value overlay.
    response_units : str, optional
        Units for the response (e.g. "kg"). If None, no units shown.

    Returns
    -------
    None
        Displays content directly in Streamlit.
    """
    st.subheader("📊 Interaction Plots")
    st.caption(
        "Mean response at each factor combination.  Parallel lines indicate "
        "little interaction; crossing / non-parallel lines indicate interaction."
    )

    if len(factors) < 2:
        st.info("At least two factors are required for interaction plots.")
        st.stop()

    if len(response) == 0 or np.all(pd.isna(response)):
        st.warning("No response data available to plot.")
        st.stop()

    units_map = _factor_units_map(factors)
    factor_names = [f.name for f in factors]

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        f1_name = st.selectbox(
            "Factor A (x-axis)", factor_names, key="ix_f1",
            help="Levels of this factor appear on the horizontal axis.",
        )
    with col2:
        f2_options = [n for n in factor_names if n != f1_name]
        f2_name = st.selectbox(
            "Factor B (lines)", f2_options, key="ix_f2",
            help="One line is drawn for each level of this factor.",
        )
    with col3:
        error_mode_label = st.radio(
            "Error bars",
            _ERROR_MODES,
            index=0,
            key="ix_error_mode",
            horizontal=True,
            help="Mean only, or mean ± SD / 95% CI for replicated runs.",
        )

    if f1_name is None or f2_name is None:
        st.info("Select a pair of factors to build the interaction plot.")
        st.stop()

    f1 = next(f for f in factors if f.name == f1_name)
    f2 = next(f for f in factors if f.name == f2_name)
    error_mode = _ERROR_MODE_MAP[error_mode_label]

    p_value, present = _interaction_pvalue(results, f1_name, f2_name)

    # Guard against factors whose names are missing from the design.
    if f1_name not in design.columns or f2_name not in design.columns:
        st.error(f"Factors {f1_name!r} / {f2_name!r} not found in design data.")
        st.stop()

    stats = interaction_stats(f1_name, f2_name, design, response)

    if stats.empty:
        st.warning("No complete data for this factor pair.")
        st.stop()

    plot_params = {
        "response_name": selected_response,
        "response_units": response_units,
        "f1_units": units_map.get(f1_name),
        "f2_units": units_map.get(f2_name),
        "error_mode": error_mode,
        "p_value": p_value,
        "interaction_present": present,
    }

    # Orientation 1: A on the x-axis, B as line colour/group.
    fig1 = create_interaction_plot(
        stats,
        f1_name,
        f2_name,
        f1.is_categorical(),
        f2.is_categorical(),
        **plot_params,
    )

    # Orientation 2: B on the x-axis, A as line colour/group.
    fig2 = create_interaction_plot(
        stats,
        f2_name,
        f1_name,
        f2.is_categorical(),
        f1.is_categorical(),
        **plot_params,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**{f1_name} × {f2_name}**")
        st.caption(f"{f1_name} on the x-axis, lines = {f2_name}.")
        st.plotly_chart(fig1, width="stretch", theme=None)
    with c2:
        st.markdown(f"**{f2_name} × {f1_name}**")
        st.caption(f"{f2_name} on the x-axis, lines = {f1_name}.")
        st.plotly_chart(fig2, width="stretch", theme=None)
