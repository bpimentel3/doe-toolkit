"""
Box Plot component for ANOVA analysis.

Draws a Tukey (IQR) box plot of the selected response grouped by the levels
of a user-chosen factor, so the distribution of the measured response can be
compared across factor levels at a glance.  Boxes are built from the observed
response values (not adjusted means); only factor levels present in the
filtered data are shown.

The heavy lifting is done by the pure, testable functions
:func:`box_plot_stats` and :func:`create_box_plot` in ``src/ui/utils/plotting.py``;
this module only wires up Streamlit controls and renders the resulting figure.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.ui.utils.plotting import (
    box_plot_stats,
    create_box_plot,
)


def _response_values_by_level(
    factor_name: str,
    design: pd.DataFrame,
    response: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Map each factor level string to its raw, NaN-free response values."""
    values: Dict[str, np.ndarray] = {}
    for _, row in design.iterrows():
        level = str(row[factor_name])
        y = response[row.name]
        if pd.isna(y):
            continue
        values.setdefault(level, []).append(float(y))
    return {k: np.asarray(v, dtype=float) for k, v in values.items()}


def _factor_units_map(factors) -> Dict[str, Optional[str]]:
    return {f.name: getattr(f, "units", None) for f in factors}


def display_box_plot_tab(
    selected_response: str,
    design: pd.DataFrame,
    response: np.ndarray,
    factors: List,
    factor_names_order: Optional[List[str]] = None,
    response_units: Optional[str] = None,
) -> None:
    """
    Display the Box Plots tab content.

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
    factor_names_order : list of str, optional
        Preferred order for the x-axis factor choices (defaults to factor
        definition order).
    response_units : str, optional
        Units for the response (e.g. "kg"). If None, no units shown.

    Returns
    -------
    None
        Displays content directly in Streamlit.
    """
    st.subheader("📦 Box Plots")
    st.caption(
        "Distribution of the selected response across the levels of a chosen "
        "factor.  Each box shows min, Q1, median, Q3 and max; points beyond "
        "the Tukey whiskers (1.5×IQR) are drawn as outliers."
    )

    if not factors:
        st.info("No factors defined.")
        st.stop()

    if len(response) == 0 or np.all(pd.isna(response)):
        st.warning("No response data available to plot.")
        st.stop()

    factor_names = [f.name for f in factors]
    if factor_names_order:
        ordered = [n for n in factor_names_order if n in factor_names]
        remaining = [n for n in factor_names if n not in factor_names_order]
        factor_names = ordered + remaining

    units_map = _factor_units_map(factors)

    factor_name = st.selectbox(
        "Factor (x-axis)",
        factor_names,
        key="bx_factor",
        help="Distributions of the response are shown for each level of this factor.",
    )

    if factor_name is None:
        st.info("Select a factor to build the box plot.")
        st.stop()

    if factor_name not in design.columns:
        st.error(f"Factor {factor_name!r} not found in design data.")
        st.stop()

    factor = next(f for f in factors if f.name == factor_name)

    stats = box_plot_stats(
        factor_name, design, response, is_categorical=factor.is_categorical()
    )

    if stats.empty:
        st.warning("No complete data for this factor.")
        st.stop()

    values_map = _response_values_by_level(factor_name, design, response)
    values_map = {k: v for k, v in values_map.items() if k in set(stats['level'])}

    fig = create_box_plot(
        stats,
        factor_name,
        factor_name,
        selected_response,
        values_map,
        response_units=response_units,
        factor_units=units_map.get(factor_name),
    )

    # Faint dashed gridlines on the value (y) axis only, no grid on the
    # category (x) axis, so the black box outlines read cleanly.
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#dcdcdc",
        gridwidth=0.5,
        griddash="dash",
    )
    fig.update_xaxes(showgrid=False)

    st.plotly_chart(fig, width="stretch", theme=None)