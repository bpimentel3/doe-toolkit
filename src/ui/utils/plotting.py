"""
Plotting utilities for ANOVA analysis and diagnostics.

This module provides consistent, publication-quality plot generation for:
- Model fit assessment (parity plots, residual plots)
- Effect significance (LogWorth plots, half-normal plots)
- Diagnostic plots (Q-Q plots, residuals vs factors)
- Response surfaces (contour plots, 3D surfaces)

All plots use consistent ACS-style formatting with cohesive color schemes.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression

# ==================== PLOT STYLING ====================

PLOT_COLORS: Dict[str, str] = {
    "primary": "#1f77b4",
    "secondary": "#ff7f0e",
    "success": "#2ca02c",
    "danger": "#d62728",
    "neutral": "#7f7f7f",
    "purple": "#9467bd",
    "brown": "#8c564b",
    "pink": "#e377c2",
    "sigma1": "#90EE90",  # Light green for 1σ
    "sigma2": "#FFD700",  # Gold for 2σ
    "sigma3": "#FF6347",  # Tomato red for 3σ
}


def apply_plot_style(fig: go.Figure) -> go.Figure:
    """
    Apply consistent ACS-style formatting to plotly figures.

    Parameters
    ----------
    fig : go.Figure
        Plotly figure to style

    Returns
    -------
    go.Figure
        Styled figure with white background, black text, and grid lines

    Notes
    -----
    Applies:
    - White plot and paper backgrounds
    - Arial font, 11pt, full black
    - Tight margins (~5%)
    - Grid lines with subtle gray
    - Black axis lines with outside ticks
    - Mirrored axes

    Examples
    --------
    >>> fig = go.Figure()
    >>> fig = apply_plot_style(fig)
    """
    # Update layout (background, font, margins)
    fig.update_layout(
        plot_bgcolor="white",  # White plot background
        paper_bgcolor="white",  # White paper background
        font=dict(family="Arial, sans-serif", size=11, color="#000000"),
        margin=dict(l=50, r=30, t=30, b=50, pad=5),  # Tighter margins (~5%)
    )

    # Update axes separately to preserve titles
    fig.update_xaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor="#e0e0e0",
        linecolor="#000000",
        linewidth=1.5,
        mirror=True,
        ticks="outside",
        tickwidth=1,
        tickcolor="#000000",
        showline=True,
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
    )

    fig.update_yaxes(
        showgrid=True,
        gridwidth=0.5,
        gridcolor="#e0e0e0",
        linecolor="#000000",
        linewidth=1.5,
        mirror=True,
        ticks="outside",
        tickwidth=1,
        tickcolor="#000000",
        showline=True,
        tickfont=dict(color="#000000"),
        title_font=dict(color="#000000"),
    )

    return fig


# ==================== MODEL FIT PLOTS ====================


def create_parity_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    r_squared: float,
    adj_r_squared: float,
    rmse: float,
    p_value: float,
) -> go.Figure:
    """
    Create actual vs predicted parity plot with 95% CI of the fit.

    Parameters
    ----------
    actual : np.ndarray
        Actual response values
    predicted : np.ndarray
        Predicted response values from model
    r_squared : float
        R-squared value
    adj_r_squared : float
        Adjusted R-squared value
    rmse : float
        Root mean squared error
    p_value : float
        Model p-value

    Returns
    -------
    go.Figure
        Parity plot with 1:1 reference line, 95% CI band, and statistics

    Notes
    -----
    The 95% confidence interval represents uncertainty in the fit line,
    not prediction intervals for individual points. Points should scatter
    around the 1:1 line if the model is unbiased.

    Examples
    --------
    >>> actual = np.array([1.0, 2.0, 3.0])
    >>> predicted = np.array([1.1, 1.9, 3.2])
    >>> fig = create_parity_plot(actual, predicted, 0.95, 0.94, 0.15, 0.001)
    """
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    margin = (max_val - min_val) * 0.1
    plot_min = min_val - margin
    plot_max = max_val + margin

    # Calculate 95% CI of the fit (not prediction interval)
    n = len(actual)
    residuals = actual - predicted
    mse = np.mean(residuals**2)

    # For parity plot, CI is tighter near the mean
    mean_actual = np.mean(actual)
    x_line = np.linspace(plot_min, plot_max, 100)

    # Standard error of the fit
    se_fit = np.sqrt(
        mse * (1 / n + (x_line - mean_actual) ** 2 / np.sum((actual - mean_actual) ** 2))
    )
    t_crit = stats.t.ppf(0.975, n - 2)  # 95% CI
    ci_width = t_crit * se_fit

    fig = go.Figure()

    # 95% CI band (around 1:1 line)
    y_upper = x_line + ci_width
    y_lower = x_line - ci_width

    # Add shaded CI region
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x_line, x_line[::-1]]),
            y=np.concatenate([y_upper, y_lower[::-1]]),
            fill="toself",
            fillcolor="rgba(128, 128, 128, 0.25)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    hover_text = [
        f"Run {i+1}<br>Actual: {a:.3f}<br>Predicted: {p:.3f}"
        for i, (a, p) in enumerate(zip(actual, predicted))
    ]

    fig.add_trace(
        go.Scatter(
            x=actual,
            y=predicted,
            mode="markers",
            marker=dict(
                size=8,
                color=PLOT_COLORS["primary"],
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            name="Data",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[plot_min, plot_max],
            y=[plot_min, plot_max],
            mode="lines",
            line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2),
            name="1:1 Line",
            hoverinfo="skip",
            showlegend=False,
        )
    )

    stats_text = (
        f"R² = {r_squared:.4f}<br>Adj R² = {adj_r_squared:.4f}<br>"
        f"RMSE = {rmse:.4f}<br>p = {p_value:.4e}"
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.05,
        y=0.95,
        text=stats_text,
        showarrow=False,
        font=dict(size=10, color="#000000"),
        bgcolor="rgba(255, 255, 255, 0.95)",
        bordercolor="#000000",
        borderwidth=1,
        align="left",
    )

    fig.update_layout(
        xaxis_title="Actual", yaxis_title="Predicted", height=400, showlegend=False
    )

    fig.update_xaxes(scaleanchor="y", scaleratio=1, range=[plot_min, plot_max])
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[plot_min, plot_max])

    return apply_plot_style(fig)


def create_residual_plot(fitted: np.ndarray, residuals: np.ndarray) -> go.Figure:
    """
    Create studentized residuals vs fitted with color-coded thresholds.

    Parameters
    ----------
    fitted : np.ndarray
        Fitted values from model
    residuals : np.ndarray
        Raw residuals (actual - predicted)

    Returns
    -------
    go.Figure
        Residual plot with reference lines at ±1σ, ±2σ, ±3σ

    Notes
    -----
    Studentized residuals should:
    - Scatter randomly around zero (no pattern)
    - Most points within ±2σ (95%)
    - Very few beyond ±3σ

    Patterns indicate:
    - Funnel shape: Non-constant variance
    - Curvature: Missing model terms
    - Outliers: Data quality issues or influential points

    Examples
    --------
    >>> fitted = np.array([1.0, 2.0, 3.0])
    >>> residuals = np.array([0.1, -0.2, 0.1])
    >>> fig = create_residual_plot(fitted, residuals)
    """
    # Calculate studentized residuals
    std_resid = np.std(residuals)
    studentized = residuals / std_resid

    fig = go.Figure()

    x_range = [fitted.min(), fitted.max()]

    # Zero line
    fig.add_trace(
        go.Scatter(
            x=x_range,
            y=[0, 0],
            mode="lines",
            line=dict(color="#000000", dash="solid", width=1.5),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    # Sigma reference lines with increased opacity
    for sigma, color in [
        (1, PLOT_COLORS["sigma1"]),
        (2, PLOT_COLORS["sigma2"]),
        (3, PLOT_COLORS["sigma3"]),
    ]:
        for sign in [1, -1]:
            y_val = sign * sigma
            # Convert hex color to rgba with opacity
            if color == PLOT_COLORS["sigma1"]:
                rgba_color = "rgba(144, 238, 144, 0.8)"  # Light green
            elif color == PLOT_COLORS["sigma2"]:
                rgba_color = "rgba(255, 215, 0, 0.8)"  # Gold
            else:
                rgba_color = "rgba(255, 99, 71, 0.8)"  # Tomato red

            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=[y_val, y_val],
                    mode="lines",
                    line=dict(color=rgba_color, dash="dash", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # All data points same color (no color coding)
    hover_text = [
        f"Run {i+1}<br>Fitted: {f:.3f}<br>Studentized: {s:.3f}"
        for i, (f, s) in enumerate(zip(fitted, studentized))
    ]

    fig.add_trace(
        go.Scatter(
            x=fitted,
            y=studentized,
            mode="markers",
            marker=dict(
                size=8,
                color=PLOT_COLORS["primary"],
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            name="Residuals",
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        xaxis_title="Fitted Values",
        yaxis_title="Studentized Residuals",
        height=400,
        showlegend=False,
    )

    y_max = max(abs(studentized.min()), abs(studentized.max()))
    y_max = max(y_max, 3.5) * 1.1  # At least show ±3σ range
    fig.update_yaxes(range=[-y_max, y_max])

    x_range_val = fitted.max() - fitted.min()
    fig.update_xaxes(
        range=[fitted.min() - 0.1 * x_range_val, fitted.max() + 0.1 * x_range_val]
    )

    return apply_plot_style(fig)


# ==================== EFFECT SIGNIFICANCE PLOTS ====================


def create_logworth_plot(
    logworth_df: pd.DataFrame, p_values: Dict[str, float]
) -> go.Figure:
    """
    Create LogWorth bar plot sorted Pareto-style with p-values on bars.

    Parameters
    ----------
    logworth_df : pd.DataFrame
        DataFrame with 'LogWorth' column and term names as index
    p_values : Dict[str, float]
        Dictionary mapping term names to p-values

    Returns
    -------
    go.Figure
        Horizontal bar chart sorted by significance with α=0.05 threshold

    Notes
    -----
    LogWorth = -log₁₀(p-value)
    - LogWorth > 1.301 indicates p < 0.05 (significant)
    - Bars sorted Pareto-style (most significant at top)
    - P-values displayed on bars for reference

    Examples
    --------
    >>> logworth_df = pd.DataFrame({'LogWorth': [2.5, 1.0]}, index=['A', 'B'])
    >>> p_values = {'A': 0.003, 'B': 0.10}
    >>> fig = create_logworth_plot(logworth_df, p_values)
    """
    logworth_sorted = logworth_df.sort_values("LogWorth", ascending=True)
    p_values_sorted = [p_values[term] for term in logworth_sorted.index]

    p_text = [
        f"p={p:.4f}" if p >= 0.0001 else f"p={p:.2e}" for p in p_values_sorted
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=logworth_sorted["LogWorth"],
            y=logworth_sorted.index,
            orientation="h",
            marker=dict(
                color=PLOT_COLORS["primary"], line=dict(color="#000000", width=0.5)
            ),
            text=p_text,
            textposition="outside",
            textfont=dict(size=10),
            hovertemplate="%{y}<br>LogWorth: %{x:.2f}<br>%{text}<extra></extra>",
        )
    )

    threshold = -np.log10(0.05)
    fig.add_vline(
        x=threshold,
        line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2),
        annotation=dict(
            text="α=0.05", textangle=0, yref="paper", y=0.95, font=dict(size=10)
        ),
    )

    fig.update_layout(
        xaxis_title="LogWorth (-log₁₀(p))",
        yaxis_title="",
        height=max(250, len(logworth_sorted) * 25),
        showlegend=False,
        margin=dict(l=150, r=100),
    )

    return apply_plot_style(fig)


def create_half_normal_plot(
    effects: np.ndarray, effect_names: List[str]
) -> go.Figure:
    """
    Create half-normal probability plot for effects.

    Parameters
    ----------
    effects : np.ndarray
        Array of effect estimates (coefficients)
    effect_names : List[str]
        Names corresponding to each effect

    Returns
    -------
    go.Figure
        Half-normal plot with reference line and labeled points

    Notes
    -----
    Half-normal plots help identify significant effects in screening designs:
    - Negligible effects fall along reference line
    - Significant effects deviate from line
    - Reference line fit to lower 1/3 of effects (assumed negligible)

    Used primarily for fractional factorial screening where many effects
    are assumed to be negligible.

    Examples
    --------
    >>> effects = np.array([0.1, 0.5, 2.0])
    >>> names = ['A', 'B', 'A*B']
    >>> fig = create_half_normal_plot(effects, names)
    """
    abs_effects = np.abs(effects)
    sorted_indices = np.argsort(abs_effects)
    sorted_effects = abs_effects[sorted_indices]
    sorted_names = [effect_names[i] for i in sorted_indices]

    n = len(sorted_effects)
    quantiles = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    half_normal_quantiles = np.abs(quantiles)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=half_normal_quantiles,
            y=sorted_effects,
            mode="markers+text",
            marker=dict(
                size=8,
                color=PLOT_COLORS["primary"],
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            text=sorted_names,
            textposition="top center",
            textfont=dict(size=9),
            hovertemplate="%{text}<br>|Effect|: %{y:.3f}<extra></extra>",
        )
    )

    if len(sorted_effects) > 2:
        n_baseline = max(3, len(sorted_effects) // 3)
        lr = LinearRegression()
        lr.fit(
            half_normal_quantiles[:n_baseline].reshape(-1, 1),
            sorted_effects[:n_baseline],
        )

        x_line = np.array([0, half_normal_quantiles.max()])
        y_line = lr.predict(x_line.reshape(-1, 1))

        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=y_line,
                mode="lines",
                line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    fig.update_layout(
        xaxis_title="Half-Normal Quantiles",
        yaxis_title="|Effect|",
        height=400,
        showlegend=False,
    )

    return apply_plot_style(fig)


# ==================== DIAGNOSTIC PLOTS ====================


def create_qq_plot(residuals: np.ndarray) -> go.Figure:
    """
    Create Q-Q normal probability plot.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals

    Returns
    -------
    go.Figure
        Q-Q plot with reference line

    Notes
    -----
    Q-Q plot assesses normality of residuals:
    - Points should follow diagonal line if normally distributed
    - Deviations indicate:
        - S-curve: Heavy or light tails
        - Bow: Skewness
        - Isolated points: Outliers

    Uses proper plotting positions: (i - 0.5) / n

    Examples
    --------
    >>> residuals = np.random.normal(0, 1, 100)
    >>> fig = create_qq_plot(residuals)
    """
    # Properly calculate theoretical quantiles using plotting positions
    n = len(residuals)
    # Use plotting position: (i - 0.5) / n
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    theoretical = stats.norm.ppf(probabilities)
    sample = np.sort(residuals)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=theoretical,
            y=sample,
            mode="markers",
            marker=dict(
                size=8,
                color=PLOT_COLORS["primary"],
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            name="Data",
            hovertemplate="Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>",
        )
    )

    min_val = min(theoretical.min(), sample.min())
    max_val = max(theoretical.max(), sample.max())

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        height=350,
        showlegend=False,
    )

    return apply_plot_style(fig)


def create_residual_vs_run_order_plot(residuals: np.ndarray) -> go.Figure:
    """
    Create residuals vs run order plot.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals

    Returns
    -------
    go.Figure
        Time series plot of residuals

    Notes
    -----
    Detects time-dependent patterns:
    - Trends: Drift in process over time
    - Cycles: Periodic effects
    - Clusters: Batch effects or systematic changes

    Random scatter around zero indicates no time-dependent issues.

    Examples
    --------
    >>> residuals = np.random.normal(0, 1, 50)
    >>> fig = create_residual_vs_run_order_plot(residuals)
    """
    run_order = np.arange(1, len(residuals) + 1)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=run_order,
            y=residuals,
            mode="markers+lines",
            marker=dict(
                size=8,
                color=PLOT_COLORS["primary"],
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            line=dict(color=PLOT_COLORS["primary"], width=1, dash="dot"),
            hovertemplate="Run %{x}<br>Residual: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0, line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2)
    )
    fig.update_layout(
        xaxis_title="Run Order", yaxis_title="Residuals", height=350
    )

    return apply_plot_style(fig)


def create_residual_vs_factor_plot(
    factor_values: np.ndarray, residuals: np.ndarray, factor_name: str
) -> go.Figure:
    """
    Create residuals vs factor plot.

    Parameters
    ----------
    factor_values : np.ndarray
        Values of the factor
    residuals : np.ndarray
        Model residuals
    factor_name : str
        Name of the factor for axis label

    Returns
    -------
    go.Figure
        Scatter plot of residuals vs factor levels

    Notes
    -----
    Checks for:
    - Patterns suggesting missing terms (curvature, trends)
    - Non-constant variance (funnel shape)
    - Outliers at specific factor levels

    Should show random scatter around zero.

    Examples
    --------
    >>> factor_values = np.array([1, 2, 3, 1, 2, 3])
    >>> residuals = np.array([0.1, -0.2, 0.1, -0.1, 0.2, 0.0])
    >>> fig = create_residual_vs_factor_plot(factor_values, residuals, 'Temperature')
    """
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=factor_values,
            y=residuals,
            mode="markers",
            marker=dict(
                size=8,
                color=PLOT_COLORS["primary"],
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            hovertemplate=f"{factor_name}: %{{x}}<br>Residual: %{{y:.3f}}<extra></extra>",
        )
    )
    fig.add_hline(
        y=0, line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2)
    )
    fig.update_layout(
        xaxis_title=factor_name, yaxis_title="Residuals", height=250
    )

    return apply_plot_style(fig)


# ==================== PROFILER PLOTS ====================


def create_response_trace_plot(
    factor_range: np.ndarray,
    predictions: np.ndarray,
    current_value: float,
    current_prediction: float,
    factor_name: str,
    response_name: str,
    ci_lower: Optional[np.ndarray] = None,
    ci_upper: Optional[np.ndarray] = None,
) -> go.Figure:
    """
    Create response trace plot for prediction profiler.

    Parameters
    ----------
    factor_range : np.ndarray
        Range of factor values to plot
    predictions : np.ndarray
        Predicted response at each factor value
    current_value : float
        Current setting of the factor (for vertical line)
    current_prediction : float
        Current predicted response (for marker)
    factor_name : str
        Name of the factor
    response_name : str
        Name of the response
    ci_lower : Optional[np.ndarray]
        Lower 95% CI boundary (if available)
    ci_upper : Optional[np.ndarray]
        Upper 95% CI boundary (if available)

    Returns
    -------
    go.Figure
        Response trace with current setting marked

    Notes
    -----
    Shows how response changes as one factor varies while others are held constant.
    Current setting marked with vertical line and point.

    Examples
    --------
    >>> factor_range = np.linspace(10, 30, 50)
    >>> predictions = 2.0 + 0.5 * factor_range
    >>> fig = create_response_trace_plot(
    ...     factor_range, predictions, 20.0, 12.0, 'Temp', 'Yield'
    ... )
    """
    fig = go.Figure()

    # 95% CI band (if available)
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
            y=predictions,
            mode="lines",
            line=dict(color=PLOT_COLORS["primary"], width=2),
            hovertemplate=f"{factor_name}: %{{x:.3f}}<br>{response_name}: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        )
    )

    # Current setting - vertical line
    fig.add_vline(
        x=current_value,
        line=dict(color="red", dash="dash", width=2),
        annotation=dict(
            text=f"{current_value:.2f}",
            yref="paper",
            y=1.05,
            showarrow=False,
            font=dict(size=10, color="red"),
        ),
    )

    # Current prediction point
    fig.add_trace(
        go.Scatter(
            x=[current_value],
            y=[current_prediction],
            mode="markers",
            marker=dict(size=10, color="red", symbol="circle"),
            showlegend=False,
            hovertemplate=f"{factor_name}: {current_value:.3f}<br>{response_name}: {current_prediction:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=200,
        margin=dict(l=40, r=10, t=20, b=40),
        xaxis_title=None,
        yaxis_title=response_name,
        showlegend=False,
    )

    min_val = factor_range.min()
    max_val = factor_range.max()
    fig.update_xaxes(
        range=[min_val - 0.05 * (max_val - min_val), max_val + 0.05 * (max_val - min_val)]
    )

    return apply_plot_style(fig)


def create_categorical_response_plot(
    levels: List,
    predictions: np.ndarray,
    current_level,
    factor_name: str,
    response_name: str,
) -> go.Figure:
    """
    Create bar chart for categorical factor in profiler.

    Parameters
    ----------
    levels : List
        Categorical levels of the factor
    predictions : np.ndarray
        Predicted response at each level
    current_level
        Currently selected level (highlighted in red)
    factor_name : str
        Name of the factor
    response_name : str
        Name of the response

    Returns
    -------
    go.Figure
        Bar chart with current level highlighted

    Examples
    --------
    >>> levels = ['Low', 'Medium', 'High']
    >>> predictions = np.array([10.0, 15.0, 12.0])
    >>> fig = create_categorical_response_plot(
    ...     levels, predictions, 'Medium', 'Material', 'Strength'
    ... )
    """
    # Determine which bar is current
    current_idx = levels.index(current_level)

    # Color bars (current one red, others blue)
    colors = [
        PLOT_COLORS["danger"] if i == current_idx else PLOT_COLORS["primary"]
        for i in range(len(levels))
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=[str(level) for level in levels],
            y=predictions,
            marker=dict(color=colors, line=dict(color="#000000", width=1)),
            hovertemplate=f"{factor_name}: %{{x}}<br>{response_name}: %{{y:.3f}}<extra></extra>",
            showlegend=False,
        )
    )

    fig.update_layout(
        height=200,
        margin=dict(l=40, r=10, t=20, b=40),
        xaxis_title=None,
        yaxis_title=response_name,
        showlegend=False,
    )

    return apply_plot_style(fig)


def create_contour_plot(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_mesh: np.ndarray,
    x_factor_name: str,
    y_factor_name: str,
    response_name: str,
) -> go.Figure:
    """
    Create 2D contour plot of response surface.

    Parameters
    ----------
    x_grid : np.ndarray
        X-axis grid values
    y_grid : np.ndarray
        Y-axis grid values
    z_mesh : np.ndarray
        Response values on mesh (2D array)
    x_factor_name : str
        Name of X-axis factor
    y_factor_name : str
        Name of Y-axis factor
    response_name : str
        Name of response

    Returns
    -------
    go.Figure
        Contour plot with color scale and labeled contours

    Notes
    -----
    Visualizes response surface for two factors while holding others constant.
    Useful for finding optimal regions and understanding interactions.

    Examples
    --------
    >>> x_grid = np.linspace(0, 10, 50)
    >>> y_grid = np.linspace(0, 10, 50)
    >>> X, Y = np.meshgrid(x_grid, y_grid)
    >>> Z = X**2 + Y**2
    >>> fig = create_contour_plot(x_grid, y_grid, Z, 'A', 'B', 'Response')
    """
    fig = go.Figure()

    # Add contour
    fig.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=z_mesh,
            colorscale="RdYlGn",
            colorbar=dict(title=response_name),
            contours=dict(
                coloring="heatmap",
                showlabels=True,
                labelfont=dict(size=10, color="white"),
            ),
            hovertemplate=(
                f"{x_factor_name}: %{{x:.2f}}<br>"
                f"{y_factor_name}: %{{y:.2f}}<br>"
                f"{response_name}: %{{z:.2f}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        xaxis_title=x_factor_name,
        yaxis_title=y_factor_name,
        height=500,
        showlegend=True,
    )

    return apply_plot_style(fig)


def create_3d_surface_plot(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_mesh: np.ndarray,
    x_factor_name: str,
    y_factor_name: str,
    response_name: str,
) -> go.Figure:
    """
    Create 3D surface plot of response surface.

    Parameters
    ----------
    x_grid : np.ndarray
        X-axis grid values
    y_grid : np.ndarray
        Y-axis grid values
    z_mesh : np.ndarray
        Response values on mesh (2D array)
    x_factor_name : str
        Name of X-axis factor
    y_factor_name : str
        Name of Y-axis factor
    response_name : str
        Name of response

    Returns
    -------
    go.Figure
        3D surface plot with color scale

    Notes
    -----
    Interactive 3D visualization of response surface.
    Users can rotate to see features from different angles.

    Examples
    --------
    >>> x_grid = np.linspace(0, 10, 50)
    >>> y_grid = np.linspace(0, 10, 50)
    >>> X, Y = np.meshgrid(x_grid, y_grid)
    >>> Z = X**2 + Y**2
    >>> fig = create_3d_surface_plot(x_grid, y_grid, Z, 'A', 'B', 'Response')
    """
    fig = go.Figure()

    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_mesh,
            colorscale="RdYlGn",
            colorbar=dict(title=response_name),
            hovertemplate=(
                f"{x_factor_name}: %{{x:.2f}}<br>"
                f"{y_factor_name}: %{{y:.2f}}<br>"
                f"{response_name}: %{{z:.2f}}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        scene=dict(
            xaxis_title=x_factor_name,
            yaxis_title=y_factor_name,
            zaxis_title=response_name,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        height=600,
    )

    return apply_plot_style(fig)