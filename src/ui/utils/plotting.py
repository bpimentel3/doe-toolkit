"""
Plotting utilities for ANOVA analysis and diagnostics.

This module provides consistent, publication-quality plot generation for:
- Model fit assessment (parity plots, residual plots)
- Effect significance (LogWorth plots, half-normal plots)
- Diagnostic plots (Q-Q plots, residuals vs factors)
- Response surfaces (contour plots, 3D surfaces)

All plots use consistent ACS-style formatting with cohesive color schemes.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression

from src.core.analysis_base import attach_critical_limits

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

# Qualitative palette for categorical factor levels (one colour per line).
# Mirrors the Plotly default 10-colour cycle so it stays consistent with the
# rest of PLOT_COLORS.
QUALITATIVE_COLORS: List[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]

# Sequential palette for numeric factor levels, mapped by rank so the
# gradient order always matches the numeric ordering of the levels.
SEQUENTIAL_COLORS: List[str] = [
    "#1f77b4", "#3182bd", "#6baed6", "#9ecae1",
    "#fd8d3c", "#e6550d", "#d62728",
]


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
        legend=dict(
            font=dict(color="#000000"),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="#000000",
            borderwidth=1,
        ),
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


# ==================== UNIT LABEL HELPER ====================


def _label_with_units(name: str, units: Optional[str]) -> str:
    """
    Append units to an axis label if units are provided.

    Parameters
    ----------
    name : str
        Base axis label (e.g., "Temperature").
    units : Optional[str]
        Units string (e.g., "°C"). If None or empty, returns name unchanged.

    Returns
    -------
    str
        Label with units appended in parentheses, e.g., "Temperature (°C)".

    Examples
    --------
    >>> _label_with_units("Yield", "%")
    'Yield (%)'
    >>> _label_with_units("Temperature", None)
    'Temperature'
    """
    if units:
        return f"{name} ({units})"
    return name


# ==================== MODEL FIT PLOTS ====================


def create_parity_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    response_units: Optional[str] = None,
) -> go.Figure:
    """
    Create actual vs predicted parity plot with 95% CI of the fit.

    Parameters
    ----------
    actual : np.ndarray
        Actual response values
    predicted : np.ndarray
        Predicted response values from model
    response_units : Optional[str]
        Units for the response axis labels (e.g., "kg"). If None, no units shown.

    Returns
    -------
    go.Figure
        Parity plot with 1:1 reference line and 95% CI band

    Notes
    -----
    The 95% confidence interval represents uncertainty in the fit line,
    not prediction intervals for individual points. Points should scatter
    around the 1:1 line if the model is unbiased.
    Model fit statistics (R², RMSE, p) are rendered outside the figure
    as native UI elements to avoid dark-mode contrast issues.

    Examples
    --------
    >>> actual = np.array([1.0, 2.0, 3.0])
    >>> predicted = np.array([1.1, 1.9, 3.2])
    >>> fig = create_parity_plot(actual, predicted)
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

    x_label = _label_with_units("Actual", response_units)
    y_label = _label_with_units("Predicted", response_units)
    fig.update_layout(
        xaxis_title=x_label, yaxis_title=y_label, height=400, showlegend=False
    )

    fig.update_xaxes(scaleanchor="y", scaleratio=1, range=[plot_min, plot_max])
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[plot_min, plot_max])

    return apply_plot_style(fig)


def create_residual_plot(
    fitted: np.ndarray,
    residuals: np.ndarray,
    response_units: Optional[str] = None,
) -> go.Figure:
    """
    Create studentized residuals vs fitted with color-coded thresholds.

    Parameters
    ----------
    fitted : np.ndarray
        Fitted values from model
    residuals : np.ndarray
        Raw residuals (actual - predicted)
    response_units : Optional[str]
        Units for the fitted values axis (e.g., "kg"). If None, no units shown.

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

    fitted_label = _label_with_units("Fitted Values", response_units)
    fig.update_layout(
        xaxis_title=fitted_label,
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


def _pformat(p: float) -> str:
    """Format a p-value for hover text."""
    if not np.isfinite(p):
        return "n/a"
    if p >= 0.0001:
        return f"{p:.4f}"
    return f"{p:.3g}"


def _effect_plot_height(n_terms: int) -> int:
    """Adaptive height proportional to the number of displayed terms."""
    return max(280, n_terms * 26)


def create_coefficient_significance_plot(
    coefficient_significance_df: Optional[pd.DataFrame],
    alpha: float = 0.05,
    show_block: bool = True,
) -> go.Figure:
    """
    Coefficient-level LogWorth Pareto (kept as-is from fitted-model tests).

    Bars show ``-log10(coefficient p-value)`` for each fitted coefficient.
    Intercept is excluded.  Block/design terms are styled distinctly and can
    be filtered at display time only (no refit).  Significant rows are
    colored; insignificant rows are desaturated so the α line reads clearly.

    Parameters
    ----------
    coefficient_significance_df : Optional[pd.DataFrame]
        Canonical coefficient_significance table.
    alpha : float
        Significance level; reference line at ``-log10(alpha)``.
    show_block : bool
        Whether to display block/design terms.

    Returns
    -------
    go.Figure
    """
    if coefficient_significance_df is None or coefficient_significance_df.empty:
        return _empty_effects_figure("Coefficient Significance (LogWorth)")
    df = coefficient_significance_df.copy()
    if not show_block:
        df = df.loc[~df["is_block"].fillna(False)].copy()
    if df.empty:
        return _empty_effects_figure("Coefficient Significance (LogWorth)", hidden=True)

    df["_is_block"] = df["is_block"].fillna(False).astype(bool)
    df = df.sort_values("logworth", ascending=True)

    colors = [
        PLOT_COLORS["secondary"] if blk else PLOT_COLORS["primary"]
        for blk in df["_is_block"]
    ]
    p_text = [_pformat(p) for p in df["p_value"]]
    hover = [
        (
            f"<b>{name}</b><br>"
            f"Parent ANOVA term: {parent or '—-'}"
            f"{f' (DF={int(parent_df)})' if pd.notna(parent_df) else ''}<br>"
            f"estimate={est:.4g} &nbsp;SE={se:.4g} &nbsp;t={t:.3f}<br>"
            f"coefficient p={_pformat(p)} &nbsp;LogWorth={lw:.2f}<br>"
            f"{'<b>Block/design term</b><br>' if blk else ''}"
            f"Source: {src}"
        )
        for name, parent, parent_df, est, se, t, p, lw, blk, src in zip(
            df["coefficient_name"],
            df["parent_anova_term"],
            df["parent_anova_df"],
            df["coefficient_estimate"],
            df["standard_error"],
            df["t_value"],
            df["p_value"],
            df["logworth"],
            df["_is_block"],
            df["source"],
        )
    ]

    fig = go.Figure()

    colormap = dict(zip(df.index, colors))
    cmap_p = dict(zip(df.index, p_text))
    cmap_h = dict(zip(df.index, hover))

    def _add_bars(sub, pattern):
        if sub.empty:
            return
        fig.add_trace(
            go.Bar(
                x=sub["logworth"],
                y=sub["coefficient_name"],
                orientation="h",
                marker=dict(
                    color=[colormap[i] for i in sub.index],
                    line=dict(color="#000000", width=0.5),
                    pattern=dict(shape=pattern) if pattern else None,
                ),
                text=[cmap_p[i] for i in sub.index],
                textposition="outside",
                textfont=dict(size=10),
                customdata=[cmap_h[i] for i in sub.index],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

    _add_bars(df.loc[~df["_is_block"]], pattern="")
    _add_bars(df.loc[df["_is_block"]], pattern="/")

    threshold = -np.log10(alpha)
    fig.add_vline(
        x=threshold,
        line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2),
        annotation=dict(
            text=f"α={alpha}", textangle=0, yref="paper", y=0.95, font=dict(size=10)
        ),
    )
    fig.update_layout(
        title=dict(
            text="Coefficient Significance (LogWorth)",
            font=dict(size=15),
        ),
        xaxis_title="LogWorth (-log₁₀ of fitted-model coefficient p)",
        yaxis_title="",
        height=_effect_plot_height(len(df)),
        showlegend=False,
        margin=dict(l=190, r=110),
    )
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    return apply_plot_style(fig)


def create_standardized_effects_plot(
    anova_effect_summary_df: Optional[pd.DataFrame],
    alpha: float = 0.05,
    show_block: bool = True,
) -> go.Figure:
    """
    DOE Pareto of Standardized Effects, sourced from the ANOVA table.

    One-Df terms use ``|t| = sqrt(F)`` (signed by the coefficient estimate).
    Multi-Df terms (e.g. categorical main effects) use an omnibus
    ``sqrt(F)`` score and are shown in a neutral color; they are not
    presented as one-Df effects.  Critical limits (t and Bonferroni) are
    drawn only when every displayed term shares a single residual DF.
    """
    if anova_effect_summary_df is None or anova_effect_summary_df.empty:
        return _empty_effects_figure("DOE Pareto of Standardized Effects")
    df = anova_effect_summary_df.copy()
    if not show_block:
        df = df.loc[~df["is_block"].fillna(False)].copy()
    if df.empty:
        return _empty_effects_figure(
            "DOE Pareto of Standardized Effects", hidden=True
        )

    df["_is_block"] = df["is_block"].fillna(False).astype(bool)
    df["_abs"] = df["standardized_statistic"].astype(float).abs()
    df = df.loc[df["_abs"].notna()].sort_values("_abs", ascending=True)

    colors = []
    for _, row in df.iterrows():
        if row["_is_block"] or row["standardized_statistic_type"] == "omnibus sqrt(F)":
            colors.append(PLOT_COLORS["neutral"])
        elif float(row["standardized_statistic"]) < 0:
            colors.append(PLOT_COLORS["danger"])
        else:
            colors.append(PLOT_COLORS["primary"])

    labels = [
        (
            f"<b>{term}</b>{' <i>(block)</i>' if blk else ''}<br>"
            f"{stat_type} = {stat:.3f}<br>"
            f"DF={int(dof) if pd.notna(dof) else '?'} &nbsp;"
            f"F={fstat:.3f} &nbsp;p={_pformat(p)}<br>"
            f"residual DF={int(rd) if pd.notna(rd) else '?'}<br>"
            f"{'t-critical=%.3f  Bonferroni=%.3f' % (tc, bc) if pd.notna(tc) else ''}<br>"
            f"<i>{'Multi-Df term: shown as omnibus sqrt(F), not a one-Df t' if dof is not None and dof > 1 else ''}</i>"
            f"effect estimate={est:.4g} &nbsp;sign={'+' if sgn > 0 else ('-' if sgn < 0 else '0')}<br>"
            f"Source: {src}"
        )
        for term, blk, stat_type, stat, dof, fstat, p, rd, tc, bc, est, sgn, src in zip(
            df["term"],
            df["_is_block"],
            df["standardized_statistic_type"],
            df["standardized_statistic"],
            df["df"],
            df["F"],
            df["p_value"],
            df["residual_df"],
            df["t_critical"],
            df["bonferroni_limit"],
            df["effect_estimate"],
            df["effect_sign"],
            df["source"],
        )
    ]
    # Short single-line axis labels; the verbose block lives in hover only.
    ylabels = [
        f"{term}{' (block)' if blk else ''}"
        for term, blk in zip(df["term"], df["_is_block"])
    ]

    fig = go.Figure()

    colormap = dict(zip(df.index, colors))
    cmap_l = dict(zip(df.index, labels))
    cmap_y = dict(zip(df.index, ylabels))

    def _add_std_bars(sub, pattern):
        if sub.empty:
            return
        fig.add_trace(
            go.Bar(
                x=sub["_abs"],
                y=[cmap_y[i] for i in sub.index],
                orientation="h",
                marker=dict(
                    color=[colormap[i] for i in sub.index],
                    line=dict(color="#000000", width=0.5),
                    pattern=dict(shape=pattern) if pattern else None,
                ),
                text=[f"{s:.2f}".replace("+", "") for s in sub["standardized_statistic"]],
                textposition="outside",
                textfont=dict(size=10),
                customdata=[cmap_l[i] for i in sub.index],
                hovertemplate="%{customdata}<extra></extra>",
            )
        )

    _add_std_bars(df.loc[~df["_is_block"]], pattern="")
    _add_std_bars(df.loc[df["_is_block"]], pattern="/")

    # Critical limits: only when every displayed term shares one residual DF.
    residuals = df["residual_df"].dropna().unique()
    if len(residuals) == 1 and np.isfinite(residuals[0]):
        tc, bc = df["t_critical"].iloc[0], df["bonferroni_limit"].iloc[0]
        if np.isfinite(tc):
            fig.add_vline(x=tc, line=dict(color=PLOT_COLORS["danger"], dash="dash"))
        if np.isfinite(bc):
            fig.add_vline(x=bc, line=dict(color=PLOT_COLORS["sigma2"], dash="dot"))
    else:
        fig.add_annotation(
            text="No universal t/Bonferroni limit: residual DF vary across strata",
            xref="paper", yref="paper", x=1.0, y=-0.12, showarrow=False,
            font=dict(size=9, color=PLOT_COLORS["neutral"]),
            xanchor="right",
        )

    fig.update_layout(
        title=dict(text="DOE Pareto of Standardized Effects", font=dict(size=15)),
        xaxis_title="Standardized ANOVA Statistic (|t| = sqrt(F) for one-Df terms)",
        yaxis_title="",
        height=_effect_plot_height(len(df)),
        showlegend=False,
        margin=dict(l=190, r=130),
    )
    fig.update_yaxes(automargin=True, tickfont=dict(size=11))
    return apply_plot_style(fig)


def _empty_effects_figure(title: str, hidden: bool = False) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text="No effects available"
        if not hidden
        else "No effects to show (block/design terms hidden)",
        xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
    )
    fig.update_layout(title=dict(text=title, font=dict(size=15)), height=280)
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


def create_residual_vs_run_order_plot(
    residuals: np.ndarray,
    response_units: Optional[str] = None,
) -> go.Figure:
    """
    Create residuals vs run order plot.

    Parameters
    ----------
    residuals : np.ndarray
        Model residuals
    response_units : Optional[str]
        Units for the residuals axis (e.g., "kg"). If None, no units shown.

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
    residuals_label = _label_with_units("Residuals", response_units)
    fig.update_layout(
        xaxis_title="Run Order", yaxis_title=residuals_label, height=350
    )

    return apply_plot_style(fig)


def create_residual_vs_factor_plot(
    factor_values: np.ndarray,
    residuals: np.ndarray,
    factor_name: str,
    factor_units: Optional[str] = None,
    response_units: Optional[str] = None,
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
    factor_units : Optional[str]
        Units for the factor axis (e.g., "°C"). If None, no units shown.
    response_units : Optional[str]
        Units for the residuals axis (e.g., "kg"). If None, no units shown.

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
    x_label = _label_with_units(factor_name, factor_units)
    residuals_label = _label_with_units("Residuals", response_units)
    fig.update_layout(
        xaxis_title=x_label, yaxis_title=residuals_label, height=250
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
    factor_units: Optional[str] = None,
    response_units: Optional[str] = None,
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
    factor_units : Optional[str]
        Units for the factor x-axis (e.g., "°C"). If None, no units shown.
    response_units : Optional[str]
        Units for the response y-axis (e.g., "kg"). If None, no units shown.

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

    response_label = _label_with_units(response_name, response_units)
    fig.update_layout(
        height=200,
        margin=dict(l=40, r=10, t=20, b=40),
        xaxis_title=None,
        yaxis_title=response_label,
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
    response_units: Optional[str] = None,
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
    response_units : Optional[str]
        Units for the response y-axis (e.g., "kg"). If None, no units shown.

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

    response_label = _label_with_units(response_name, response_units)
    fig.update_layout(
        height=200,
        margin=dict(l=40, r=10, t=20, b=40),
        xaxis_title=None,
        yaxis_title=response_label,
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
    x_factor_units: Optional[str] = None,
    y_factor_units: Optional[str] = None,
    response_units: Optional[str] = None,
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
    x_factor_units : Optional[str]
        Units for the X-axis factor (e.g., "°C"). If None, no units shown.
    y_factor_units : Optional[str]
        Units for the Y-axis factor (e.g., "psi"). If None, no units shown.
    response_units : Optional[str]
        Units for the colorbar label (e.g., "kg"). If None, no units shown.

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

    x_label = _label_with_units(x_factor_name, x_factor_units)
    y_label = _label_with_units(y_factor_name, y_factor_units)
    response_label = _label_with_units(response_name, response_units)
    fig.data[0].colorbar.title = response_label
    fig.update_layout(
        xaxis_title=x_label,
        yaxis_title=y_label,
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
    x_factor_units: Optional[str] = None,
    y_factor_units: Optional[str] = None,
    response_units: Optional[str] = None,
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
    x_factor_units : Optional[str]
        Units for the X-axis factor (e.g., "°C"). If None, no units shown.
    y_factor_units : Optional[str]
        Units for the Y-axis factor (e.g., "psi"). If None, no units shown.
    response_units : Optional[str]
        Units for the Z-axis and colorbar label (e.g., "kg"). If None, no units shown.

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

    x_label = _label_with_units(x_factor_name, x_factor_units)
    y_label = _label_with_units(y_factor_name, y_factor_units)
    response_label = _label_with_units(response_name, response_units)
    fig.update_layout(
        scene=dict(
            xaxis_title=x_label,
            yaxis_title=y_label,
            zaxis_title=response_label,
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
        ),
        height=600,
    )

    return apply_plot_style(fig)

def interaction_stats(
    f1_name: str,
    f2_name: str,
    design: pd.DataFrame,
    response: np.ndarray,
) -> pd.DataFrame:
    """
    Aggregate per-combination response statistics for an interaction plot.

    For every observed combination of factor ``f1`` x ``f2`` this returns one
    row with the mean response, sample standard deviation, replicate count,
    standard error of the mean, and a 95% confidence interval for the mean.

    Combinations with a missing (NaN) response are dropped; a combination with
    a single replicate yields ``std = sem = 0`` and a CI collapsed to the mean.

    Parameters
    ----------
    f1_name : str
        Name of the x-axis factor column.
    f2_name : str
        Name of the line-encoding factor column.
    design : pd.DataFrame
        Design data (natural units) containing at least ``f1_name`` and
        ``f2_name``.
    response : np.ndarray
        Response values aligned with ``design`` rows.

    Returns
    -------
    pd.DataFrame
        Columns ``[f1_name, f2_name, mean, std, n, sem, ci_lower, ci_upper]``.

    Examples
    --------
    >>> stats = interaction_stats('A', 'B', design, response)
    """
    df = pd.DataFrame({
        f1_name: np.asarray(design[f1_name].values),
        f2_name: np.asarray(design[f2_name].values),
        'response': np.asarray(response, dtype=float),
    })
    # Missing responses are excluded from the means.
    df = df.dropna(subset=['response'])

    grouped = df.groupby([f1_name, f2_name], sort=False)['response']
    mean = grouped.mean().rename('mean').reset_index()
    counts = grouped.size().rename('n').reset_index()
    std = grouped.std(ddof=1).rename('std').reset_index()
    sem = grouped.sem().rename('sem').reset_index()

    result = mean.merge(counts, on=[f1_name, f2_name])
    result = result.merge(std, on=[f1_name, f2_name])
    result = result.merge(sem, on=[f1_name, f2_name])

    result['mean'] = result['mean'].astype(float)
    result['std'] = result['std'].fillna(0.0)
    result['sem'] = result['sem'].fillna(0.0)
    result['n'] = result['n'].astype(int)

    # 95% CI for the mean using a t-distribution; with a single replicate
    # there is no spread so the interval collapses onto the mean.
    tcrit = np.where(
        result['n'] > 1,
        stats.t.ppf(0.975, df=np.clip(result['n'] - 1, 1, None)),
        0.0,
    )
    result['ci_lower'] = result['mean'] - tcrit * result['sem']
    result['ci_upper'] = result['mean'] + tcrit * result['sem']

    return result


def _sorted_levels(values, is_categorical: bool):
    """Return the ordered set of level strings for a factor column.

    Categorical levels are sorted lexicographically to give a stable order;
    numeric levels are sorted numerically so the axis/colour gradient follows
    the actual values, not their string representation.  Returns the original
    string representations (not floats) so downstream ``str(level)`` joins
    match the stringified stats DataFrame exactly.
    """
    unique = list(dict.fromkeys(str(v) for v in values))
    if is_categorical:
        return sorted(unique)
    pairs = []
    for s in unique:
        try:
            pairs.append((float(s), s))
        except (TypeError, ValueError):
            pairs.append((float('inf'), s))
    pairs.sort(key=lambda p: (np.isnan(p[0]), p[0]))
    return [p[1] for p in pairs]


def create_interaction_plot(
    stats: pd.DataFrame,
    f1_name: str,
    f2_name: str,
    f1_is_categorical: bool,
    f2_is_categorical: bool,
    response_name: str,
    response_units: Optional[str] = None,
    f1_units: Optional[str] = None,
    f2_units: Optional[str] = None,
    error_mode: str = "none",
    p_value: Optional[float] = None,
    interaction_present: bool = True,
) -> go.Figure:
    """
    Create an interaction plot (Stat-Ease / Design-Expert style).

    One line is drawn for each level of ``f2`` spanning the levels of ``f1``
    on the x-axis.  Parallel lines indicate little interaction while
    crossing / non-parallel lines indicate an interaction.

    Parameters
    ----------
    stats : pd.DataFrame
        Output of :func:`interaction_stats`.
    f1_name : str
        Name of the x-axis factor.
    f2_name : str
        Name of the line-encoding (grouping) factor.
    f1_is_categorical / f2_is_categorical : bool
        Whether each factor is categorical (affects axis type, colouring and
        level ordering).  Categorical levels are never coerced to numbers.
    response_name : str
        Display name of the response for the y-axis/tooltip.
    response_units / f1_units / f2_units : str, optional
        Units appended to axis labels.
    error_mode : str
        One of ``"none"`` (mean only), ``"sd"`` (mean +/- SD), ``"ci"``
        (mean +/- 95% CI).  Error bars are omitted for single-replicate
        combinations.
    p_value : float, optional
        Interaction term p-value from the fitted ANOVA, if available.
    interaction_present : bool
        Whether the ``f1:f2`` interaction term is present in the fitted model.
        When ``False`` the significance subtitle reports "not in model".

    Returns
    -------
    go.Figure
        The interaction plot.

    Examples
    --------
    >>> stats = interaction_stats('A', 'B', design, response)
    >>> fig = create_interaction_plot(stats, 'A', 'B', True, False, 'Yield')
    """
    # Ordered level lists, matching the order used for grouping/colouring.
    f1_levels = _sorted_levels(stats[f1_name].values, f1_is_categorical)
    f2_levels = _sorted_levels(stats[f2_name].values, f2_is_categorical)

    stats = stats.copy()
    stats[f1_name] = stats[f1_name].map(lambda v: str(v))
    stats[f2_name] = stats[f2_name].map(lambda v: str(v))

    # Colour assignment per grouping level.
    if f2_is_categorical:
        line_colors = {
            str(level): QUALITATIVE_COLORS[i % len(QUALITATIVE_COLORS)]
            for i, level in enumerate(f2_levels)
        }
    else:
        line_colors = {}
        n = max(len(f2_levels), 1)
        for i, level in enumerate(f2_levels):
            color = SEQUENTIAL_COLORS[
                int(round(i * (len(SEQUENTIAL_COLORS) - 1) / (n - 1)))
            ] if n > 1 else SEQUENTIAL_COLORS[0]
            line_colors[str(level)] = color

    # X-axis: category axis for categorical factor, linear for numeric.
    x_is_categorical = f1_is_categorical
    x_values = list(f1_levels)
    if x_is_categorical:
        x_values = [str(v) for v in x_values]
    else:
        x_values = [float(v) for v in x_values]

    f1_label = _label_with_units(f1_name, f1_units)
    f2_label = _label_with_units(f2_name, f2_units)
    response_label = _label_with_units(response_name, response_units)

    fig = go.Figure()

    for level in f2_levels:
        lvl_str = str(level)
        subset = stats[stats[f2_name] == lvl_str]
        # Align with the x-level order (fill missing combos with None).
        y = []
        sd_vals = []
        n_vals = []
        for x_lvl in f1_levels:
            row = subset[subset[f1_name] == str(x_lvl)]
            if row.empty:
                y.append(None)
                sd_vals.append(None)
                n_vals.append(None)
                continue
            r = row.iloc[0]
            y.append(None if pd.isna(r['mean']) else float(r['mean']))
            sd_vals.append(
                None if pd.isna(r['std']) else float(r['std'])
            )
            n_vals.append(int(r['n']))

        error_y = None
        if error_mode in ("sd", "ci") and len(f2_levels) > 0:
            arrays = []
            for idx, x_lvl in enumerate(f1_levels):
                row = subset[subset[f1_name] == str(x_lvl)]
                if row.empty or int(row.iloc[0]['n']) < 2:
                    arrays.append(0.0)
                elif error_mode == "sd":
                    arrays.append(float(row.iloc[0]['std']))
                else:
                    arrays.append(float(
                        row.iloc[0]['ci_upper'] - row.iloc[0]['mean']
                    ))
            error_y = dict(
                type="data",
                symmetric=True,
                array=arrays,
                thickness=1,
                width=4,
            )

        # customdata carries (level, SD, n) so the tooltip can show them.
        customdata = [
            [lvl_str, sd_vals[i], n_vals[i]]
            for i in range(len(f1_levels))
        ]
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=y,
                mode="lines+markers",
                name=lvl_str,
                line=dict(color=line_colors[lvl_str], width=2.5),
                marker=dict(size=8, line=dict(width=1, color="white")),
                error_y=error_y,
                customdata=customdata,
                hovertemplate=(
                    f"{f1_name}: %{{x}}<br>"
                    f"{f2_name}: %{{customdata[0]}}<br>"
                    f"{response_name}: %{{y:.3f}}<br>"
                    f"SD: %{{customdata[1]}}<br>"
                    f"n: %{{customdata[2]}}<extra></extra>"
                ),
            )
        )

    # Interaction significance subtitle.
    subtitle = None
    if interaction_present and p_value is not None:
        if p_value < 0.05:
            verdict = "Significant interaction"
        elif p_value < 0.10:
            verdict = "Marginal interaction"
        else:
            verdict = "No significant interaction"
        subtitle = f"{verdict} (interaction p-value = {p_value:g})"
    elif not interaction_present:
        subtitle = "Interaction not in model (N/A)"
    else:
        subtitle = None

    layout_kwargs: Dict[str, object] = dict(
        title=dict(
            text=f"{response_label}  |  {f1_name} × {f2_name}",
            font=dict(size=14),
        ),
        xaxis_title=f1_label,
        yaxis_title=response_label,
        xaxis_type="category" if x_is_categorical else "linear",
        showlegend=True,
        height=480,
        legend=dict(title=dict(text=f2_label)),
        hovermode="closest",
    )
    if subtitle:
        layout_kwargs["title"]["subtitle"] = dict(text=subtitle)

    fig.update_layout(**layout_kwargs)
    fig.update_traces(connectgaps=False)

    return apply_plot_style(fig)


def box_plot_stats(
    factor_name: str,
    design: pd.DataFrame,
    response: np.ndarray,
    is_categorical: bool,
) -> pd.DataFrame:
    """
    Aggregate per-level response statistics for a box plot.

    One row is returned per observed level of ``factor_name``.  Boxes are
    drawn from the raw response values, so the Quartile/Min/Q1/Median/Q3/Max
    summary is derived from the observed data rather than adjusted means.

    Rows with a missing (NaN) response are dropped; factor levels that are
    absent from the data produce no row at all (present levels only).

    Parameters
    ----------
    factor_name : str
        Name of the x-axis factor column.
    design : pd.DataFrame
        Design data (natural units) containing ``factor_name``.
    response : np.ndarray
        Response values aligned with ``design`` rows.
    is_categorical : bool
        Whether the factor is categorical.  Controls level ordering (see
        :func:`_sorted_levels`); categorical levels are never coerced to
        numbers.

    Returns
    -------
    pd.DataFrame
        Columns ``[factor_name, mean, std, n]`` ordered by level, plus
        ``level`` holding the stable sorted level string used for plotting.
    """
    df = pd.DataFrame({
        factor_name: np.asarray(design[factor_name].values),
        'response': np.asarray(response, dtype=float),
    })
    # Missing responses are excluded.
    df = df.dropna(subset=['response'])

    ordered = _sorted_levels(df[factor_name].values, is_categorical)

    rows = []
    for level in ordered:
        vals = df[df[factor_name].map(lambda v: str(v)) == str(level)]['response']
        if vals.empty:
            continue
        vals = vals.astype(float).values
        rows.append({
            factor_name: level,
            'level': str(level),
            'n': int(len(vals)),
            'mean': float(vals.mean()),
            'std': float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
        })

    result = pd.DataFrame(rows, columns=[
        factor_name, 'level', 'n', 'mean', 'std'
    ])
    return result


def create_box_plot(
    stats: pd.DataFrame,
    factor_name: str,
    factor_label: str,
    response_name: str,
    response_values_by_level: Dict[str, np.ndarray],
    response_units: Optional[str] = None,
    factor_units: Optional[str] = None,
) -> go.Figure:
    """
    Create an IQR/Tukey box plot of the response for each factor level.

    One ``go.Box`` trace is drawn per factor level and the raw response values
    are used directly, so Plotly computes the whisker/outlier range (Q1-1.5
    IQR, Q3+1.5 IQR) from the observed responses.  Boxes are drawn unfilled
    (transparent body) with all-black outlines, whiskers, median line and
    outlier markers, matching the standard industry convention for a
    single-factor box plot.  The x-axis is a category axis keyed on the
    factor level labels.

    Parameters
    ----------
    stats : pd.DataFrame
        Output of :func:`box_plot_stats`.
    factor_name : str
        Name of the factor column (used for tooltips).
    factor_label : str
        Display label for the factor (the natural name).
    response_name : str
        Display name of the response for the y-axis/tooltip.
    response_values_by_level : Dict[str, np.ndarray]
        Mapping from each level string to its raw (NaN-free) response values.
        Keys must be the same level strings as ``stats['level']``.
    response_units : str, optional
        Units appended to the response axis label.
    factor_units : str, optional
        Units appended to the factor axis label.

    Returns
    -------
    go.Figure
        The box plot.
    """
    factor_axis_label = _label_with_units(factor_label, factor_units)
    response_label = _label_with_units(response_name, response_units)

    fig = go.Figure()

    for level in stats['level']:
        values = np.asarray(response_values_by_level[level], dtype=float)
        n = stats.loc[stats['level'] == level, 'n'].iloc[0]
        fig.add_trace(
            go.Box(
                y=values,
                name=level,
                boxpoints="outliers",
                fillcolor="rgba(0,0,0,0)",
                line=dict(color="#000000", width=1),
                marker=dict(color="#000000", size=5),
                hoveron="boxes",
                hovertemplate=(
                    f"{factor_name}: {level}<br>"
                    f"{response_name}: %{{y:.3f}}<br>"
                    f"n: {n}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=dict(
            text=f"{response_label}  |  {factor_label}",
            font=dict(size=14),
        ),
        xaxis_title=factor_axis_label,
        yaxis_title=response_label,
        xaxis_type="category",
        showlegend=False,
        height=480,
        hovermode="closest",
    )
    return apply_plot_style(fig)