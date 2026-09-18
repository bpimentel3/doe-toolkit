"""
Prediction Profiler component for ANOVA analysis.

This module provides interactive prediction profiling capabilities:
- Response trace plots for continuous factors
- Bar charts for categorical/discrete factors
- Contour plots for factor pairs
- 3D surface plots
- Real-time prediction updates as factors change

Designed to mimic JMP-style prediction profiler functionality.
"""

from typing import Dict

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from src.ui.utils.plotting import (
    PLOT_COLORS,
    apply_plot_style,
    build_surface_mesh,
    _label_with_units,
)
from src.core.coding import encode_settings_dict


def display_profiler_tab(
    selected_response: str,
    results,
    factors,
    format_term_for_display,
    response_units: str = None,
) -> None:
    """
    Display the Prediction Profiler tab content.

    Parameters
    ----------
    selected_response : str
        Name of the currently selected response
    results : ANOVAResults
        Fitted model results
    factors : list
        List of Factor objects
    format_term_for_display : callable
        Function to format term names for display
    response_units : str, optional
        Units for the response (e.g., "kg"). If None, no units shown.

    Returns
    -------
    None
        Displays content directly in Streamlit
    """
    st.subheader("📊 Prediction Profiler")
    st.caption(
        "Interactive prediction profiler - adjust factor settings and see how the response changes."
    )

    if not factors:
        st.info("No factors available for profiling.")
        st.stop()

    # Initialize factor settings
    _initialize_profiler_settings(factors)

    # Get current factor settings
    factor_settings = {
        factor.name: st.session_state["profiler_settings"][factor.name]
        for factor in factors
    }

    # Compute current prediction
    current_prediction = _compute_prediction(results, factor_settings, factors)

    # Display current prediction
    pred_label = _label_with_units(selected_response, response_units)
    st.markdown(f"### Predicted {pred_label}: **{current_prediction:.4f}**")
    st.divider()

    # Display profiler plots grid
    _display_profiler_grid(
        factors, factor_settings, results, selected_response, current_prediction,
        response_units=response_units,
    )

    st.divider()

    # Contour and 3D plots
    continuous_factors = [f for f in factors if f.is_continuous()]
    if len(continuous_factors) >= 2:
        _display_contour_plots(
            factors, continuous_factors, results, selected_response,
            response_units=response_units,
        )
        st.divider()
        _display_3d_surface(
            factors, continuous_factors, results, selected_response,
            response_units=response_units,
        )
    else:
        st.info("Contour and 3D plots require at least 2 continuous factors.")


def _initialize_profiler_settings(factors) -> None:
    """Initialize profiler settings in session state."""
    if "profiler_settings" not in st.session_state:
        st.session_state["profiler_settings"] = {}

    for factor in factors:
        if factor.name not in st.session_state["profiler_settings"]:
            if factor.is_continuous():
                min_val, max_val = factor.levels
                st.session_state["profiler_settings"][factor.name] = (
                    min_val + max_val
                ) / 2
            else:
                mid_idx = len(factor.levels) // 2
                st.session_state["profiler_settings"][factor.name] = factor.levels[
                    mid_idx
                ]


def _compute_prediction(results, factor_settings: Dict, factors) -> float:
    """Compute prediction for given factor settings."""
    try:
        encoded_settings = encode_settings_dict(factor_settings, factors)
        return results.predict_from_settings(encoded_settings)
    except Exception as e:
        st.error(f"Could not compute prediction: {e}")
        return 0.0


def _display_profiler_grid(
    factors, factor_settings, results, selected_response, current_prediction,
    response_units=None,
) -> None:
    """Display grid of profiler plots."""
    n_factors = len(factors)
    n_cols = min(3, n_factors)  # Max 3 columns
    n_rows = (n_factors + n_cols - 1) // n_cols

    for row_idx in range(n_rows):
        cols = st.columns(n_cols)

        for col_idx in range(n_cols):
            factor_idx = row_idx * n_cols + col_idx

            if factor_idx >= n_factors:
                break

            factor = factors[factor_idx]

            with cols[col_idx]:
                factor_header = _label_with_units(factor.name, factor.units)
                st.markdown(f"**{factor_header}**")

                if factor.is_continuous():
                    _display_continuous_factor(
                        factor,
                        factor_settings,
                        results,
                        selected_response,
                        current_prediction,
                        col_idx,
                        factors,
                        response_units=response_units,
                    )
                else:
                    _display_categorical_factor(
                        factor,
                        factor_settings,
                        results,
                        selected_response,
                        col_idx,
                        factors,
                        response_units=response_units,
                    )


def _display_continuous_factor(
    factor, factor_settings, results, selected_response, current_prediction, col_idx, factors,
    response_units=None,
) -> None:
    """Display response trace plot for continuous factor."""
    min_val, max_val = factor.levels

    # Generate response trace
    trace_points = 100
    factor_range = np.linspace(min_val, max_val, trace_points)

    trace_predictions = []
    for val in factor_range:
        point = factor_settings.copy()
        point[factor.name] = val
        encoded_point = encode_settings_dict(point, factors)
        trace_predictions.append(results.predict_from_settings(encoded_point))

    # CI not available without patsy model; skip silently
    ci_lower = None
    ci_upper = None

    # Create plot
    fig = go.Figure()

    # 95% CI band
    if ci_lower is not None and ci_upper is not None:
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([factor_range, factor_range[::-1]]),
                y=np.concatenate([ci_upper, ci_lower[::-1]]),
                fill="toself",
                fillcolor="rgba(128, 128, 128, 0.2)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Response trace line
    fig.add_trace(
        go.Scatter(
            x=factor_range,
            y=trace_predictions,
            mode="lines",
            line=dict(color=PLOT_COLORS["primary"], width=2),
            hovertemplate=f"{factor.name}: %{{x:.3f}}<br>{selected_response}: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        )
    )

    # Current setting - vertical line
    current_val = factor_settings[factor.name]
    fig.add_vline(
        x=current_val,
        line=dict(color="red", dash="dash", width=2),
        annotation=dict(
            text=f"{current_val:.2f}",
            yref="paper",
            y=1.05,
            showarrow=False,
            font=dict(size=10, color="red"),
        ),
    )

    # Current prediction point
    fig.add_trace(
        go.Scatter(
            x=[current_val],
            y=[current_prediction],
            mode="markers",
            marker=dict(size=10, color="red", symbol="circle"),
            showlegend=False,
            hovertemplate=f"{factor.name}: {current_val:.3f}<br>{selected_response}: {current_prediction:.3f}<extra></extra>",
        )
    )

    response_label = _label_with_units(selected_response, response_units)
    fig.update_layout(
        height=200,
        margin=dict(l=40, r=10, t=20, b=40),
        xaxis_title=None,
        yaxis_title=response_label if col_idx == 0 else None,
        showlegend=False,
    )

    fig.update_xaxes(
        range=[
            min_val - 0.05 * (max_val - min_val),
            max_val + 0.05 * (max_val - min_val),
        ]
    )

    fig = apply_plot_style(fig)
    st.plotly_chart(fig, width='stretch', key=f"plot_{factor.name}")

    # Slider below plot
    new_val = st.slider(
        f"{factor.name} setting",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(current_val),
        format="%.3f",
        key=f"profiler_slider_{factor.name}",
        label_visibility="collapsed",
    )

    # Update if changed
    if new_val != st.session_state["profiler_settings"][factor.name]:
        st.session_state["profiler_settings"][factor.name] = new_val
        st.rerun()


def _display_categorical_factor(
    factor, factor_settings, results, selected_response, col_idx, factors,
    response_units=None,
) -> None:
    """Display bar chart for categorical/discrete factor."""
    # Generate predictions for each level
    level_predictions = []
    for level in factor.levels:
        point = factor_settings.copy()
        point[factor.name] = level
        encoded_point = encode_settings_dict(point, factors)
        level_predictions.append(results.predict_from_settings(encoded_point))

    # Determine which bar is current
    current_level = factor_settings[factor.name]
    current_idx = factor.levels.index(current_level)

    # Color bars (current one red, others blue)
    colors = [
        PLOT_COLORS["danger"] if i == current_idx else PLOT_COLORS["primary"]
        for i in range(len(factor.levels))
    ]

    # Create bar chart
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[str(level) for level in factor.levels],
            y=level_predictions,
            marker=dict(color=colors, line=dict(color="#000000", width=1)),
            hovertemplate=f"{factor.name}: %{{x}}<br>{selected_response}: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        )
    )

    response_label = _label_with_units(selected_response, response_units)
    fig.update_layout(
        height=200,
        margin=dict(l=40, r=10, t=20, b=40),
        xaxis_title=None,
        yaxis_title=response_label if col_idx == 0 else None,
        showlegend=False,
    )

    fig = apply_plot_style(fig)
    st.plotly_chart(fig, width='stretch', key=f"plot_{factor.name}")

    # Selectbox below plot
    new_val = st.selectbox(
        f"{factor.name} setting",
        options=factor.levels,
        index=current_idx,
        key=f"profiler_select_{factor.name}",
        label_visibility="collapsed",
    )

    # Update if changed
    if new_val != st.session_state["profiler_settings"][factor.name]:
        st.session_state["profiler_settings"][factor.name] = new_val
        st.rerun()


def _display_contour_plots(
    factors, continuous_factors, results, selected_response, response_units=None,
) -> None:
    """Display 2D contour plots."""
    st.markdown("### 🗺️ Contour Plots")
    st.caption("2D contour maps showing response surface for pairs of factors.")

    # Initialize contour settings
    if "contour_settings" not in st.session_state:
        st.session_state["contour_settings"] = {}
        for factor in factors:
            if factor.is_continuous():
                min_val, max_val = factor.levels
                st.session_state["contour_settings"][factor.name] = (
                    min_val + max_val
                ) / 2
            else:
                mid_idx = len(factor.levels) // 2
                st.session_state["contour_settings"][factor.name] = factor.levels[
                    mid_idx
                ]

    # Factor pair selector
    col1, col2 = st.columns(2)

    with col1:
        x_factor = st.selectbox(
            "X-axis factor",
            options=[f.name for f in continuous_factors],
            index=0,
            key="contour_x",
        )

    with col2:
        y_factor = st.selectbox(
            "Y-axis factor",
            options=[f.name for f in continuous_factors if f.name != x_factor],
            index=0,
            key="contour_y",
        )

    # Sliders for other factors
    other_factors = [f for f in factors if f.name not in [x_factor, y_factor]]

    if other_factors:
        st.markdown("**Hold constant at:**")
        _display_other_factor_controls(other_factors)

    # Generate and display contour plot
    if x_factor and y_factor:
        Z_mesh, x_grid, y_grid = _generate_contour_mesh(
            factors, x_factor, y_factor, results
        )

        if Z_mesh is not None:
            # Build factor unit lookups for contour axis labels
            _factor_map = {f.name: f for f in factors}
            x_factor_units = _factor_map[x_factor].units if x_factor in _factor_map else None
            y_factor_units = _factor_map[y_factor].units if y_factor in _factor_map else None
            _plot_contour(
                x_grid, y_grid, Z_mesh, x_factor, y_factor, selected_response,
                x_factor_units=x_factor_units, y_factor_units=y_factor_units,
                response_units=response_units,
            )

            # Store mesh for 3D plot
            st.session_state["_contour_mesh"] = (Z_mesh, x_grid, y_grid)


def _display_other_factor_controls(other_factors) -> None:
    """Display controls for factors held constant in contour plots."""
    n_other = len(other_factors)
    n_cols = min(3, n_other)
    n_rows = (n_other + n_cols - 1) // n_cols

    for row_idx in range(n_rows):
        cols = st.columns(n_cols)

        for col_idx in range(n_cols):
            factor_idx = row_idx * n_cols + col_idx

            if factor_idx >= n_other:
                break

            factor = other_factors[factor_idx]

            with cols[col_idx]:
                if factor.is_continuous():
                    min_val, max_val = factor.levels
                    current_val = st.session_state["contour_settings"][factor.name]

                    new_val = st.slider(
                        factor.name,
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(current_val),
                        format="%.3f",
                        key=f"contour_slider_{factor.name}",
                    )

                    if new_val != st.session_state["contour_settings"][factor.name]:
                        st.session_state["contour_settings"][factor.name] = new_val
                        st.rerun()
                else:
                    current_val = st.session_state["contour_settings"][factor.name]
                    current_idx = factor.levels.index(current_val)

                    new_val = st.selectbox(
                        factor.name,
                        options=factor.levels,
                        index=current_idx,
                        key=f"contour_select_{factor.name}",
                    )

                    if new_val != st.session_state["contour_settings"][factor.name]:
                        st.session_state["contour_settings"][factor.name] = new_val
                        st.rerun()


def _generate_contour_mesh(factors, x_factor, y_factor, results):
    """Generate mesh for contour and 3D plots."""
    try:
        # Get factor objects
        x_factor_obj = next(f for f in factors if f.name == x_factor)
        y_factor_obj = next(f for f in factors if f.name == y_factor)

        # Create grid (in actual values)
        x_min, x_max = x_factor_obj.levels
        y_min, y_max = y_factor_obj.levels

        x_grid = np.linspace(x_min, x_max, 50)
        y_grid = np.linspace(y_min, y_max, 50)

        base_settings = st.session_state["contour_settings"]

        def _predict(point):
            return results.predict_from_settings(
                encode_settings_dict(point, factors)
            )

        Z_mesh = build_surface_mesh(
            x_grid, y_grid, x_factor, y_factor, base_settings, _predict
        )

        return Z_mesh, x_grid, y_grid

    except Exception as e:
        st.error(f"Could not create contour plot: {e}")
        return None, None, None


def _plot_contour(
    x_grid, y_grid, Z_mesh, x_factor, y_factor, selected_response,
    x_factor_units=None, y_factor_units=None, response_units=None,
):
    """Plot 2D contour."""
    x_label = _label_with_units(x_factor, x_factor_units)
    y_label = _label_with_units(y_factor, y_factor_units)
    response_label = _label_with_units(selected_response, response_units)

    fig = go.Figure()

    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=Z_mesh,
            colorscale="RdYlGn",
            colorbar=dict(title=response_label),
            contours=dict(
                coloring="heatmap",
                showlabels=True,
                labelfont=dict(size=10, color="white"),
            ),
            hovertemplate=(
                f"{x_label}: %{{x:.2f}}<br>"
                f"{y_label}: %{{y:.2f}}<br>"
                f"{response_label}: %{{z:.2f}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        xaxis_title=x_label, yaxis_title=y_label, height=500, showlegend=True
    )

    fig = apply_plot_style(fig)
    st.plotly_chart(fig, width='stretch')


def _display_3d_surface(
    factors, continuous_factors, results, selected_response, response_units=None,
) -> None:
    """Display 3D response surface plot."""
    st.markdown("### 🏔️ 3D Response Surface")
    st.caption("Interactive 3D visualization of the response surface.")

    # Retrieve mesh from contour plot
    if "_contour_mesh" not in st.session_state:
        st.info("Generate a contour plot first to see the 3D surface.")
        return

    Z_mesh, x_grid, y_grid = st.session_state["_contour_mesh"]

    # Get factor names from session state
    x_factor = st.session_state.get("contour_x")
    y_factor = st.session_state.get("contour_y")

    if x_factor and y_factor and Z_mesh is not None:
        try:
            _factor_map = {f.name: f for f in factors}
            x_label = _label_with_units(
                x_factor, _factor_map[x_factor].units if x_factor in _factor_map else None
            )
            y_label = _label_with_units(
                y_factor, _factor_map[y_factor].units if y_factor in _factor_map else None
            )
            response_label = _label_with_units(selected_response, response_units)

            fig_3d = go.Figure()

            fig_3d.add_trace(
                go.Surface(
                    x=x_grid,
                    y=y_grid,
                    z=Z_mesh,
                    colorscale="RdYlGn",
                    colorbar=dict(title=response_label),
                    hovertemplate=(
                        f"{x_label}: %{{x:.2f}}<br>"
                        f"{y_label}: %{{y:.2f}}<br>"
                        f"{response_label}: %{{z:.2f}}<extra></extra>"
                    ),
                )
            )

            fig_3d.update_layout(
                scene=dict(
                    xaxis_title=x_label,
                    yaxis_title=y_label,
                    zaxis_title=response_label,
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                ),
                height=600,
            )

            fig_3d = apply_plot_style(fig_3d)
            st.plotly_chart(fig_3d, width='stretch')

        except Exception as e:
            st.error(f"Could not create 3D surface plot: {e}")