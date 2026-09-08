"""
Export utilities for DOE Toolkit.

Provides functions for exporting projects and generating comprehensive
HTML reports with all analysis results, plots, and tables.

The HTML report embeds Plotly figures as interactive JSON (via Plotly.js
CDN) — no static image renderer (kaleido) required.
"""

import json
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_html_report() -> str:
    """
    Generate a self-contained HTML report of the entire DOE project.

    The report structure is:
    1. Header / metadata
    2. Data table (collapsible)
    3. For every fitted response:
       a. Model Fit tab content (parity, residuals vs fitted, LogWorth, ANOVA)
       b. Effects & Residuals tab content (coefficients, diagnostic plots)
       c. Design Diagnostics (if computed)
       d. Profiler (if profiler settings exist)
       e. Optimization results (if available)

    Returns
    -------
    str
        Complete, self-contained HTML document string.

    Examples
    --------
    >>> html = generate_html_report()
    >>> assert html.startswith("<!DOCTYPE html>")
    """
    parts: List[str] = []
    parts.append(_html_head())
    parts.append('<body><div class="container">')
    parts.append(_section_header())
    parts.append(_section_data())

    fitted_models: Dict[str, Any] = st.session_state.get("fitted_models", {})
    responses: Dict[str, Any] = st.session_state.get("responses", {})
    factors: List[Any] = st.session_state.get("factors", [])
    design: Optional[pd.DataFrame] = st.session_state.get("design")

    if not fitted_models:
        parts.append(
            '<p class="muted">No fitted models found — run analysis first.</p>'
        )
    else:
        for response_name, results in fitted_models.items():
            response_data = responses.get(response_name)
            excluded = st.session_state.get("excluded_rows", [])
            if response_data is not None and excluded and design is not None:
                mask = np.ones(len(design), dtype=bool)
                mask[excluded] = False
                response_data = response_data[mask]

            design_filtered = design
            if design is not None and excluded:
                mask = np.ones(len(design), dtype=bool)
                mask[excluded] = False
                design_filtered = design[mask].reset_index(drop=True)

            parts.append(
                f'<section class="response-section">'
                f'<h2 class="response-title">Response: {response_name}</h2>'
            )
            parts.append(
                _build_model_fit_section(
                    response_name, results, response_data, factors
                )
            )
            parts.append(
                _build_effects_residuals_section(
                    response_name, results, response_data, design_filtered, factors
                )
            )
            parts.append(_build_diagnostics_section(response_name))
            parts.append(_build_profiler_section(response_name, results, factors))
            parts.append(_build_optimization_section(response_name))
            parts.append("</section>")

    parts.append(_html_footer())
    parts.append("</div></body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _section_header() -> str:
    """Return title block and project metadata."""
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    design_type = (st.session_state.get("design_type") or "Unknown").replace(
        "_", " "
    ).title()
    factors: List[Any] = st.session_state.get("factors", [])
    responses: Dict = st.session_state.get("responses", {})
    design: Optional[pd.DataFrame] = st.session_state.get("design")
    n_runs = len(design) if design is not None else 0

    return f"""
<h1>DOE Toolkit — Analysis Report</h1>
<div class="meta-grid">
  <div class="meta-card"><span class="meta-label">Generated</span>{timestamp}</div>
  <div class="meta-card"><span class="meta-label">Design Type</span>{design_type}</div>
  <div class="meta-card"><span class="meta-label">Factors</span>{len(factors)}</div>
  <div class="meta-card"><span class="meta-label">Runs</span>{n_runs}</div>
  <div class="meta-card"><span class="meta-label">Responses</span>{len(responses)}</div>
</div>
"""


def _section_data() -> str:
    """Return collapsible data table (design + response columns)."""
    design: Optional[pd.DataFrame] = st.session_state.get("design")
    responses: Dict[str, Any] = st.session_state.get("responses", {})

    if design is None:
        return ""

    combined = design.copy().reset_index(drop=True)
    combined.index = combined.index + 1
    combined.index.name = "Run"
    combined = combined.reset_index()

    for resp_name, resp_data in responses.items():
        combined[resp_name] = list(resp_data)

    table_html = _df_to_html(combined)

    return f"""
<details class="collapsible" open>
  <summary class="collapsible-header">&#9660; Data ({len(combined)} runs)</summary>
  <div class="collapsible-body">
    {table_html}
  </div>
</details>
"""


def _build_model_fit_section(
    response_name: str,
    results: Any,
    response_data: Optional[Any],
    factors: List[Any],
) -> str:
    """
    Build Tab 1 — Model Fit content.

    Parameters
    ----------
    response_name : str
        Name of the response variable.
    results : ANOVAResults
        Fitted model results object.
    response_data : Optional[array-like]
        Observed response values (filtered for exclusions).
    factors : list
        List of Factor objects.

    Returns
    -------
    str
        HTML string for this section.
    """
    from src.ui.utils.plotting import (
        create_coefficient_significance_plot,
        create_logworth_plot,
        create_standardized_effects_plot,
        create_parity_plot,
        create_residual_plot,
    )

    parts: List[str] = ['<div class="tab-section"><h3>&#128202; Model Fit</h3>']

    # --- Fit metrics ---
    model_p = _extract_model_p(results)
    p_display = (
        "N/A"
        if np.isnan(model_p)
        else (f"{model_p:.4e}" if model_p < 0.0001 else f"{model_p:.4f}")
    )
    parts.append(f"""
<div class="metric-row">
  <div class="metric-card"><span class="metric-label">R²</span>{results.r_squared:.4f}</div>
  <div class="metric-card"><span class="metric-label">Adj R²</span>{results.adj_r_squared:.4f}</div>
  <div class="metric-card"><span class="metric-label">RMSE</span>{results.rmse:.4f}</div>
  <div class="metric-card"><span class="metric-label">Model p</span>{p_display}</div>
</div>
""")

    # --- Model formula ---
    terms_display = [t for t in (results.model_terms or []) if t != "1"]
    formula = (
        f"{response_name} ~ "
        + (" + ".join(terms_display) if terms_display else "1")
    )
    parts.append(f'<div class="formula">{formula}</div>')

    # --- Parity + residuals vs fitted ---
    parts.append('<div class="plot-row">')
    if response_data is not None:
        try:
            fig = create_parity_plot(response_data, results.fitted_values)
            parts.append(
                _plotly_div(fig, f"parity_{response_name}", "Actual vs Predicted")
            )
        except Exception as exc:
            parts.append(f'<div class="plot-error">Parity plot failed: {exc}</div>')

        try:
            fig = create_residual_plot(results.fitted_values, results.residuals)
            parts.append(
                _plotly_div(
                    fig, f"resvfit_{response_name}", "Residuals vs Fitted"
                )
            )
        except Exception as exc:
            parts.append(
                f'<div class="plot-error">Residuals vs Fitted failed: {exc}</div>'
            )
    parts.append("</div>")

    # --- Coefficient LogWorth Pareto + ANOVA standardized effects ---
    if results.coefficient_significance is not None and not results.coefficient_significance.empty:
        try:
            fig = create_coefficient_significance_plot(
                results.coefficient_significance,
                alpha=0.05,
                show_block=True,
            )
            parts.append(
                _plotly_div(
                    fig,
                    f"logworth_{response_name}",
                    "Coefficient Significance (LogWorth)",
                )
            )
        except Exception as exc:
            parts.append(
                f'<div class="plot-error">LogWorth plot failed: {exc}</div>'
            )
    elif results.logworth is not None and not results.logworth.empty:
        try:
            p_values = {
                term: 10 ** (-results.logworth.loc[term, "LogWorth"])
                for term in results.logworth.index
            }
            fig = create_logworth_plot(results.logworth, p_values)
            parts.append(
                _plotly_div(
                    fig,
                    f"logworth_{response_name}",
                    "Coefficient Significance (LogWorth)",
                )
            )
        except Exception as exc:
            parts.append(
                f'<div class="plot-error">LogWorth plot failed: {exc}</div>'
            )

    if results.anova_effect_summary is not None and not results.anova_effect_summary.empty:
        try:
            fig = create_standardized_effects_plot(
                results.anova_effect_summary,
                alpha=0.05,
                show_block=True,
            )
            parts.append(
                _plotly_div(
                    fig,
                    f"effects_{response_name}",
                    "DOE Pareto of Standardized Effects",
                )
            )
            parts.append(
                "<p class='muted'>Bar color — blue: positive effect · red: "
                "negative effect · gray: multi-df or block/design term. "
                "Dashed vertical line — t-critical at α=0.05; dotted vertical "
                "line — Bonferroni limit (α/m).</p>"
            )
        except Exception as exc:
            parts.append(
                f'<div class="plot-error">Standardized effects plot failed: {exc}</div>'
            )

    # --- ANOVA table ---
    parts.append("<h4>ANOVA Table</h4>")
    if results.anova_table is not None and not results.anova_table.empty:
        anova_display = results.anova_table.reset_index().rename(
            columns={"index": "Term"}
        )
        p_col = "PR(>F)" if "PR(>F)" in anova_display.columns else "P"
        parts.append(_df_to_html(anova_display, highlight_col=p_col))
    else:
        parts.append("<p class='muted'>ANOVA table not available.</p>")

    parts.append("</div>")  # tab-section
    return "\n".join(parts)


def _build_effects_residuals_section(
    response_name: str,
    results: Any,
    response_data: Optional[Any],
    design_filtered: Optional[pd.DataFrame],
    factors: List[Any],
) -> str:
    """
    Build Tab 2 — Effects & Residuals content.

    Parameters
    ----------
    response_name : str
        Name of the response variable.
    results : ANOVAResults
        Fitted model results object.
    response_data : Optional[array-like]
        Observed response values.
    design_filtered : Optional[pd.DataFrame]
        Design matrix with exclusions applied.
    factors : list
        List of Factor objects.

    Returns
    -------
    str
        HTML string for this section.
    """
    import plotly.graph_objects as go

    from src.ui.utils.plotting import (
        PLOT_COLORS,
        apply_plot_style,
        create_half_normal_plot,
        create_qq_plot,
    )

    parts: List[str] = [
        '<div class="tab-section"><h3>&#128201; Effects &amp; Residuals</h3>'
    ]

    # --- Coefficient table ---
    parts.append("<h4>Coefficient Table</h4>")
    if results.effect_estimates is not None and not results.effect_estimates.empty:
        effects_display = results.effect_estimates.reset_index().rename(
            columns={"index": "Term"}
        )
        parts.append(_df_to_html(effects_display, highlight_col="p_value"))
    else:
        parts.append("<p class='muted'>Coefficient table not available.</p>")

    # --- Residuals vs Run Order + Q-Q ---
    parts.append('<div class="plot-row">')

    try:
        run_order = np.arange(1, len(results.residuals) + 1)
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=run_order,
                y=results.residuals,
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
            y=0,
            line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2),
        )
        fig.update_layout(
            xaxis_title="Run Order",
            yaxis_title="Residuals",
            height=350,
            title="Residuals vs Run Order",
        )
        fig = apply_plot_style(fig)
        parts.append(
            _plotly_div(fig, f"resrun_{response_name}", "Residuals vs Run Order")
        )
    except Exception as exc:
        parts.append(
            f'<div class="plot-error">Residuals vs Run Order failed: {exc}</div>'
        )

    try:
        fig = create_qq_plot(results.residuals)
        parts.append(_plotly_div(fig, f"qq_{response_name}", "Normal Q-Q Plot"))
    except Exception as exc:
        parts.append(f'<div class="plot-error">Q-Q plot failed: {exc}</div>')

    parts.append("</div>")  # plot-row

    # --- Residuals vs Factors ---
    if design_filtered is not None and len(factors) > 0:
        parts.append("<h4>Residuals vs Factors</h4>")
        parts.append('<div class="plot-row">')
        for factor in factors:
            if factor.name not in design_filtered.columns:
                continue
            try:
                factor_vals = design_filtered[factor.name].values
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=factor_vals,
                        y=results.residuals,
                        mode="markers",
                        marker=dict(
                            size=8,
                            color=PLOT_COLORS["primary"],
                            opacity=0.7,
                            line=dict(width=0.5, color="white"),
                        ),
                        hovertemplate=(
                            f"{factor.name}: %{{x}}<br>"
                            "Residual: %{y:.3f}<extra></extra>"
                        ),
                    )
                )
                fig.add_hline(
                    y=0,
                    line=dict(color=PLOT_COLORS["danger"], dash="dash", width=2),
                )
                fig.update_layout(
                    xaxis_title=factor.name,
                    yaxis_title="Residuals",
                    height=280,
                    title=f"Residuals vs {factor.name}",
                )
                fig = apply_plot_style(fig)
                parts.append(
                    _plotly_div(
                        fig,
                        f"resfac_{response_name}_{factor.name}",
                        f"Residuals vs {factor.name}",
                    )
                )
            except Exception as exc:
                parts.append(
                    f'<div class="plot-error">'
                    f"Residuals vs {factor.name} failed: {exc}"
                    f"</div>"
                )
        parts.append("</div>")  # plot-row

    # --- Half-Normal plot ---
    if results.effect_estimates is not None and not results.effect_estimates.empty:
        try:
            effects_data = results.effect_estimates[
                results.effect_estimates.index != "Intercept"
            ]
            if not effects_data.empty:
                if "Estimate" in effects_data.columns:
                    effects = effects_data["Estimate"].values
                elif "Coefficient" in effects_data.columns:
                    effects = effects_data["Coefficient"].values
                else:
                    effects = None

                if effects is not None:
                    from src.ui.components.model_builder import format_term_for_display

                    effect_names = [
                        format_term_for_display(t) for t in effects_data.index
                    ]
                    fig = create_half_normal_plot(effects, effect_names)
                    parts.append(
                        _plotly_div(
                            fig,
                            f"halfnormal_{response_name}",
                            "Half-Normal Plot",
                        )
                    )
        except Exception as exc:
            parts.append(
                f'<div class="plot-error">Half-Normal plot failed: {exc}</div>'
            )

    # Note about leverage plots
    parts.append(
        '<p class="muted" style="font-size:0.85em;margin-top:8px;">'
        "&#9432; Leverage plots are not included in the report — "
        "they require interactive factor selection and cannot be "
        "pre-rendered at report generation time."
        "</p>"
    )

    parts.append("</div>")  # tab-section
    return "\n".join(parts)


def _build_diagnostics_section(response_name: str) -> str:
    """
    Build Tab 3 — Design Diagnostics content from stored session state.

    Parameters
    ----------
    response_name : str
        Name of the response variable.

    Returns
    -------
    str
        HTML string for this section.
    """
    summary = st.session_state.get("diagnostics_summary")
    report = st.session_state.get("quality_report")

    parts: List[str] = [
        '<div class="tab-section"><h3>&#128269; Design Diagnostics</h3>'
    ]

    if summary is None or report is None:
        parts.append(
            "<p class='muted'>Diagnostics were not generated — "
            "click 'Generate Diagnostics' on the Analysis page first.</p>"
        )
        parts.append("</div>")
        return "\n".join(parts)

    # --- VIF table ---
    try:
        vif_data = getattr(summary, "vif", None)
        if vif_data is not None and not vif_data.empty:
            parts.append("<h4>Variance Inflation Factors (VIF)</h4>")
            parts.append(_df_to_html(vif_data.reset_index()))
    except Exception:
        pass

    # --- Alias structure ---
    try:
        alias_data = getattr(summary, "alias_structure", None)
        if alias_data is not None and not alias_data.empty:
            parts.append("<h4>Alias Structure</h4>")
            parts.append(_df_to_html(alias_data.reset_index()))
    except Exception:
        pass

    # --- Prediction variance ---
    try:
        pv_data = getattr(summary, "prediction_variance", None)
        if pv_data is not None:
            parts.append("<h4>Prediction Variance Statistics</h4>")
            if isinstance(pv_data, pd.DataFrame):
                parts.append(_df_to_html(pv_data))
            else:
                parts.append(f"<p>{pv_data}</p>")
    except Exception:
        pass

    # --- Quality report ---
    try:
        response_quality = getattr(report, "response_quality", {})
        if response_name in response_quality:
            assessment = response_quality[response_name]
            grade = getattr(assessment, "overall_grade", "N/A")
            parts.append(f"<h4>Quality Assessment — {response_name}: {grade}</h4>")

            issues = getattr(assessment, "issues", [])
            if issues:
                parts.append("<ul>")
                for issue in issues:
                    parts.append(
                        f"<li><strong>{issue.category}:</strong> "
                        f"{issue.description}</li>"
                    )
                parts.append("</ul>")

        # Global critical issues / warnings
        critical = getattr(report, "critical_issues", [])
        warnings = getattr(report, "warnings", [])
        satisfactory = getattr(report, "satisfactory_aspects", [])

        if critical:
            parts.append(
                '<div class="callout callout-warn">'
                "<strong>Critical Issues</strong><ul>"
                + "".join(f"<li>{i}</li>" for i in critical)
                + "</ul></div>"
            )
        if warnings:
            parts.append(
                '<div class="callout callout-info">'
                "<strong>Warnings</strong><ul>"
                + "".join(f"<li>{w}</li>" for w in warnings)
                + "</ul></div>"
            )
        if satisfactory:
            parts.append(
                '<div class="callout callout-ok">'
                "<strong>Satisfactory</strong><ul>"
                + "".join(f"<li>{s}</li>" for s in satisfactory)
                + "</ul></div>"
            )
    except Exception:
        pass

    parts.append("</div>")
    return "\n".join(parts)


def _build_profiler_section(
    response_name: str, results: Any, factors: List[Any]
) -> str:
    """
    Build Tab 4 — Profiler trace plots at current profiler settings.

    Parameters
    ----------
    response_name : str
        Name of the response variable.
    results : ANOVAResults
        Fitted model results object.
    factors : list
        List of Factor objects.

    Returns
    -------
    str
        HTML string for this section.
    """
    import plotly.graph_objects as go

    from src.core.coding import encode_settings_dict
    from src.ui.utils.plotting import PLOT_COLORS, apply_plot_style

    profiler_settings: Optional[Dict[str, Any]] = st.session_state.get(
        "profiler_settings"
    )

    parts: List[str] = [
        '<div class="tab-section"><h3>&#128200; Prediction Profiler</h3>'
    ]

    if profiler_settings is None or results is None:
        parts.append(
            "<p class='muted'>Profiler not yet used — visit the Profiler tab "
            "on the Analysis page to generate plots, then re-export.</p>"
        )
        parts.append("</div>")
        return "\n".join(parts)

    try:
        current_settings: Dict[str, Any] = {
            f.name: profiler_settings.get(f.name, 0) for f in factors
        }
        encoded_base = encode_settings_dict(current_settings, factors)
        import pandas as _pd

        base_row = _pd.DataFrame([encoded_base])
        try:
            current_pred = float(results.fitted_model.predict(base_row).iloc[0])
        except Exception:
            current_pred = float("nan")

        parts.append(
            f"<p><strong>Prediction at displayed settings: "
            f"{current_pred:.4f}</strong></p>"
        )

        # One trace plot per factor
        parts.append('<div class="plot-row">')
        for factor in factors:
            try:
                if factor.is_continuous():
                    x_vals = np.linspace(factor.min_value, factor.max_value, 60)
                else:
                    x_vals = np.array(factor.levels, dtype=float)

                y_preds: List[float] = []
                for x in x_vals:
                    settings_i = dict(current_settings)
                    settings_i[factor.name] = x
                    encoded_i = encode_settings_dict(settings_i, factors)
                    row_i = _pd.DataFrame([encoded_i])
                    try:
                        y_preds.append(float(results.fitted_model.predict(row_i).iloc[0]))
                    except Exception:
                        y_preds.append(float("nan"))

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=y_preds,
                        mode="lines+markers" if not factor.is_continuous() else "lines",
                        line=dict(color=PLOT_COLORS["primary"], width=2),
                        marker=dict(size=8, color=PLOT_COLORS["primary"]),
                    )
                )
                # Mark current setting
                fig.add_vline(
                    x=float(current_settings[factor.name]),
                    line=dict(color=PLOT_COLORS["danger"], dash="dash", width=1.5),
                )
                fig.update_layout(
                    xaxis_title=factor.name,
                    yaxis_title=response_name,
                    height=280,
                    title=f"Trace: {factor.name}",
                    showlegend=False,
                )
                fig = apply_plot_style(fig)
                parts.append(
                    _plotly_div(
                        fig,
                        f"profiler_{response_name}_{factor.name}",
                        f"Trace: {factor.name}",
                    )
                )
            except Exception as exc:
                parts.append(
                    f'<div class="plot-error">Profiler trace for '
                    f"{factor.name} failed: {exc}</div>"
                )

        parts.append("</div>")  # plot-row

    except Exception as exc:
        parts.append(
            f'<div class="plot-error">Profiler section failed: {exc}</div>'
        )

    parts.append("</div>")
    return "\n".join(parts)


def _build_optimization_section(response_name: str) -> str:
    """
    Build optimization results section for a single response.

    Covers both single-response results (stored per-response in
    ``st.session_state['opt_results']``) and the shared multi-response
    desirability result.

    Parameters
    ----------
    response_name : str
        Name of the response variable.

    Returns
    -------
    str
        HTML string for this section.
    """
    import plotly.graph_objects as go

    parts: List[str] = [
        '<div class="tab-section"><h3>&#127919; Optimization</h3>'
    ]

    added_anything = False

    # --- Single-response results ---
    opt_results: Dict[str, Any] = st.session_state.get("opt_results", {})
    single_result = opt_results.get(response_name)

    if single_result is not None:
        added_anything = True
        obj = single_result.get("objective", "Unknown")
        parts.append(f"<h4>Single-Response Optimization ({obj})</h4>")

        settings = single_result.get("optimal_settings", {})
        pred = single_result.get("predicted_response")
        ci = single_result.get("confidence_interval")
        pi = single_result.get("prediction_interval")

        if settings:
            rows = [
                {
                    "Factor": k,
                    "Optimal Setting": v if isinstance(v, str) else f"{v:.4f}",
                }
                for k, v in settings.items()
            ]
            parts.append(_df_to_html(pd.DataFrame(rows)))

        if pred is not None:
            parts.append(
                f'<div class="metric-row">'
                f'<div class="metric-card">'
                f'<span class="metric-label">Predicted {response_name}</span>'
                f"{pred:.4f}</div>"
            )
            if ci:
                parts.append(
                    f'<div class="metric-card">'
                    f'<span class="metric-label">95% CI</span>'
                    f"[{ci[0]:.4f}, {ci[1]:.4f}]</div>"
                )
            if pi:
                parts.append(
                    f'<div class="metric-card">'
                    f'<span class="metric-label">95% PI</span>'
                    f"[{pi[0]:.4f}, {pi[1]:.4f}]</div>"
                )
            parts.append("</div>")  # metric-row

        # Surface / contour (if stored)
        stored_figs: Dict[str, Any] = single_result.get("figures", {})
        if stored_figs.get("surface"):
            try:
                parts.append(
                    _plotly_div(
                        stored_figs["surface"],
                        f"surface_{response_name}",
                        "Response Surface",
                    )
                )
            except Exception:
                pass
        if stored_figs.get("contour"):
            try:
                parts.append(
                    _plotly_div(
                        stored_figs["contour"],
                        f"contour_{response_name}",
                        "Contour Plot",
                    )
                )
            except Exception:
                pass

    # --- Multi-response desirability ---
    d_result = st.session_state.get("desirability_result")
    d_config: Dict[str, Any] = st.session_state.get("desirability_config", {})

    if d_result is not None and response_name in (
        d_result.predicted_responses or {}
    ):
        added_anything = True
        parts.append("<h4>Multi-Response Desirability Optimization</h4>")

        # Optimal settings table
        settings = d_result.optimal_settings or {}
        if settings:
            rows = [
                {
                    "Factor": k,
                    "Optimal Setting": v if isinstance(v, str) else f"{v:.4f}",
                }
                for k, v in settings.items()
            ]
            parts.append(_df_to_html(pd.DataFrame(rows)))

        # Per-response table
        resp_names = list(d_result.predicted_responses.keys())
        summary_rows = []
        for rn in resp_names:
            cfg = d_config.get(rn, {})
            summary_rows.append(
                {
                    "Response": rn,
                    "Goal": cfg.get("goal", "—"),
                    "Predicted": round(d_result.predicted_responses[rn], 4),
                    "dᵢ": round(d_result.individual_desirabilities[rn], 4),
                }
            )
        parts.append(_df_to_html(pd.DataFrame(summary_rows)))

        # Overall desirability metric
        parts.append(
            f'<div class="metric-row">'
            f'<div class="metric-card">'
            f'<span class="metric-label">Overall Desirability (D)</span>'
            f"{d_result.overall_desirability:.4f}</div></div>"
        )

        # Desirability bar chart
        try:
            d_values = [
                d_result.individual_desirabilities[r] for r in resp_names
            ]
            bar_colors = [
                "#2ecc71" if v >= 0.8 else "#f39c12" if v >= 0.5 else "#e74c3c"
                for v in d_values
            ]
            fig = go.Figure(
                go.Bar(
                    x=resp_names,
                    y=d_values,
                    marker_color=bar_colors,
                    text=[f"{v:.3f}" for v in d_values],
                    textposition="outside",
                )
            )
            fig.update_layout(
                title="Individual Desirabilities",
                yaxis=dict(range=[0, 1.15], title="Desirability"),
                xaxis_title="Response",
                showlegend=False,
                height=350,
            )
            fig.add_hline(
                y=d_result.overall_desirability,
                line_dash="dash",
                line_color="navy",
                annotation_text=f"Overall D = {d_result.overall_desirability:.3f}",
                annotation_position="top right",
            )
            parts.append(
                _plotly_div(
                    fig,
                    f"desirability_bar_{response_name}",
                    "Individual Desirabilities",
                )
            )
        except Exception as exc:
            parts.append(
                f'<div class="plot-error">Desirability bar chart failed: {exc}</div>'
            )

    if not added_anything:
        parts.append(
            "<p class='muted'>No optimization results found — run optimization "
            "on the Optimize page first, then re-export.</p>"
        )

    parts.append("</div>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------


class _NumpyEncoder(json.JSONEncoder):
    """
    JSON encoder that converts numpy scalars and arrays to plain Python types.

    Plotly>=5.14 serializes numpy arrays as base64-encoded typed-array objects
    (``{"bdata": "...", "dtype": "f8"}``) by default.  When embedded verbatim
    in an HTML ``<script>`` block, these objects are not decoded by all
    Plotly.js CDN versions, so data traces render empty.  Converting every
    numpy value to a plain Python float/int/list before JSON serialization
    sidesteps this entirely.
    """

    def default(self, obj: Any) -> Any:  # noqa: ANN401
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return None if np.isnan(obj) or np.isinf(obj) else float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return super().default(obj)


def _unwrap_bdata(obj: Any) -> Any:
    """
    Recursively unwrap Plotly typed-array dicts into plain Python lists.

    Plotly>=5.14 ``to_dict()`` may still embed numpy arrays as
    ``{"bdata": <bytes>, "dtype": "f8"}`` dicts inside the figure dict.
    This function walks the dict tree and converts any such structure to
    a plain Python list so downstream JSON serialization produces
    standard JSON arrays.

    Parameters
    ----------
    obj : Any
        Node in the figure dict tree.

    Returns
    -------
    Any
        Cleaned object with no typed-array dicts.
    """
    if isinstance(obj, dict):
        # Plotly typed-array format: {"bdata": bytes_or_str, "dtype": "f8", ...}
        if "bdata" in obj and "dtype" in obj:
            import base64
            import struct

            dtype_map = {
                "f4": ("f", 4),
                "f8": ("d", 8),
                "i4": ("i", 4),
                "i8": ("q", 8),
                "u1": ("B", 1),
                "u2": ("H", 2),
                "u4": ("I", 4),
            }
            bdata = obj["bdata"]
            dtype_str = obj["dtype"]
            try:
                if isinstance(bdata, str):
                    raw = base64.b64decode(bdata)
                else:
                    raw = bytes(bdata)
                fmt_char, item_size = dtype_map.get(dtype_str, ("d", 8))
                n_items = len(raw) // item_size
                values = list(struct.unpack(f"<{n_items}{fmt_char}", raw))
                # Replace nan/inf with None for valid JSON
                import math
                return [
                    None if isinstance(v, float) and (math.isnan(v) or math.isinf(v))
                    else v
                    for v in values
                ]
            except Exception:
                return []  # Fallback: empty list rather than broken bdata
        return {k: _unwrap_bdata(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_unwrap_bdata(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        import math
        return None if math.isnan(float(obj)) or math.isinf(float(obj)) else float(obj)
    return obj


def _fig_to_json_safe(fig: Any) -> str:
    """
    Serialize a Plotly figure to a JSON string using plain Python types.

    Calls ``fig.to_dict()``, recursively unwraps any typed-array
    (``bdata``) structures that Plotly>=5.14 may embed, then serializes
    with ``_NumpyEncoder``.  The result is a plain JSON string containing
    only standard JSON arrays — safe for direct embedding in HTML
    ``<script>`` blocks and readable by all Plotly.js CDN versions.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure to serialize.

    Returns
    -------
    str
        JSON string safe for direct embedding in an HTML script block.
    """
    fig_dict = fig.to_dict()
    fig_dict = _unwrap_bdata(fig_dict)
    return json.dumps(fig_dict, cls=_NumpyEncoder, allow_nan=False)


def _plotly_div(fig: Any, div_id: str, title: str) -> str:
    """
    Serialize a Plotly figure into a self-contained HTML div.

    Uses a custom numpy-aware JSON encoder so that all array data is
    serialized as plain JSON arrays (not Plotly's base64 typed-array
    format introduced in plotly>=5.14).  This ensures Plotly.js can
    decode the embedded spec without any special binary decoder.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        The figure to embed.
    div_id : str
        Unique DOM id for this chart.
    title : str
        Human-readable title shown above the chart.

    Returns
    -------
    str
        HTML string containing the div and inline Plotly.newPlot call.
    """
    fig_json = _fig_to_json_safe(fig)
    return f"""
<div class="plot-wrapper">
  <p class="plot-title">{title}</p>
  <div id="{div_id}" class="plotly-chart"></div>
  <script>
    (function() {{
      var spec = {fig_json};
      Plotly.newPlot("{div_id}", spec.data, spec.layout || {{}}, {{responsive: true, displaylogo: false}});
    }})();
  </script>
</div>
"""


def _df_to_html(
    df: pd.DataFrame,
    highlight_col: Optional[str] = None,
) -> str:
    """
    Convert a DataFrame to a styled HTML table.

    Rows where the value in ``highlight_col`` is less than 0.05 are
    highlighted in yellow to indicate statistical significance.

    Parameters
    ----------
    df : pd.DataFrame
        Data to render.
    highlight_col : Optional[str]
        Column name to use for significance highlighting (p < 0.05).

    Returns
    -------
    str
        HTML table string.
    """
    html = '<div class="table-wrapper"><table>'
    html += "<thead><tr>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr></thead><tbody>"

    for _, row in df.iterrows():
        sig = False
        if highlight_col and highlight_col in df.columns:
            try:
                sig = float(row[highlight_col]) < 0.05
            except (ValueError, TypeError):
                pass

        row_cls = ' class="sig-row"' if sig else ""
        html += f"<tr{row_cls}>"
        for col in df.columns:
            val = row[col]
            if isinstance(val, (int, float, np.number)):
                if pd.isna(val):
                    fmt = "—"
                elif abs(val) < 0.001 and val != 0:
                    fmt = f"{val:.2e}"
                else:
                    fmt = f"{val:.4f}" if isinstance(val, float) else str(int(val))
                html += f'<td class="num">{fmt}</td>'
            else:
                html += f"<td>{val}</td>"
        html += "</tr>"

    html += "</tbody></table></div>"
    return html


def _extract_model_p(results: Any) -> float:
    """Extract the overall model F-test p-value from a results object."""
    try:
        if (
            results.anova_table is not None
            and not results.anova_table.empty
            and "P" in results.anova_table.columns
        ):
            return float(results.anova_table.loc["Model", "P"])
    except (KeyError, ValueError, TypeError):
        pass
    try:
        return float(results.fitted_model.f_pvalue)
    except (AttributeError, ValueError, TypeError):
        pass
    return float("nan")


# ---------------------------------------------------------------------------
# CSS / HTML boilerplate
# ---------------------------------------------------------------------------


def _html_head() -> str:
    """Return the HTML document head with all CSS."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DOE Toolkit — Analysis Report</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      font-size: 14px;
      line-height: 1.6;
      color: #1a1a2e;
      background: #f0f2f5;
      padding: 24px;
    }

    .container {
      max-width: 1300px;
      margin: 0 auto;
      background: #ffffff;
      border-radius: 10px;
      padding: 40px 48px;
      box-shadow: 0 4px 24px rgba(0,0,0,0.08);
    }

    h1 {
      font-size: 1.9em;
      color: #0f3460;
      border-bottom: 3px solid #3498db;
      padding-bottom: 10px;
      margin-bottom: 20px;
    }

    h2.response-title {
      font-size: 1.5em;
      color: #fff;
      background: linear-gradient(90deg, #1a3a6b, #2980b9);
      padding: 10px 18px;
      border-radius: 6px;
      margin: 32px 0 16px;
    }

    h3 {
      font-size: 1.15em;
      color: #0f3460;
      margin: 24px 0 12px;
      padding-bottom: 4px;
      border-bottom: 1px solid #dce3eb;
    }

    h4 {
      font-size: 1em;
      color: #34495e;
      margin: 18px 0 8px;
    }

    .response-section {
      border: 1px solid #dce3eb;
      border-radius: 8px;
      padding: 20px 24px;
      margin-bottom: 32px;
    }

    .tab-section {
      border-left: 3px solid #3498db;
      padding: 16px 20px;
      margin: 20px 0;
      background: #f8fafd;
      border-radius: 0 6px 6px 0;
    }

    /* Metadata grid */
    .meta-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 28px;
    }
    .meta-card {
      background: #eef2f7;
      border-radius: 6px;
      padding: 10px 16px;
      font-size: 0.9em;
      flex: 1 1 140px;
    }
    .meta-label {
      display: block;
      font-weight: 600;
      color: #555;
      font-size: 0.78em;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 2px;
    }

    /* Metric cards */
    .metric-row {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin: 12px 0;
    }
    .metric-card {
      background: #f0f7ff;
      border-left: 3px solid #3498db;
      border-radius: 4px;
      padding: 10px 16px;
      flex: 1 1 140px;
      font-size: 1.1em;
      font-weight: 600;
      color: #0f3460;
    }
    .metric-card .metric-label {
      display: block;
      font-weight: 400;
      font-size: 0.75em;
      text-transform: uppercase;
      color: #777;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
    }

    /* Formula block */
    .formula {
      background: #1e1e2e;
      color: #a6e22e;
      font-family: "Courier New", monospace;
      padding: 10px 14px;
      border-radius: 5px;
      overflow-x: auto;
      margin: 10px 0;
      font-size: 0.9em;
    }

    /* Collapsible data section */
    .collapsible {
      border: 1px solid #dce3eb;
      border-radius: 8px;
      margin-bottom: 28px;
      overflow: hidden;
    }
    .collapsible-header {
      background: #eef2f7;
      padding: 12px 18px;
      cursor: pointer;
      font-weight: 600;
      color: #0f3460;
      user-select: none;
      font-size: 1em;
      list-style: none;
    }
    .collapsible-header::-webkit-details-marker { display: none; }
    .collapsible-body {
      padding: 16px;
      overflow-x: auto;
    }

    /* Tables */
    .table-wrapper { overflow-x: auto; margin: 12px 0; }
    table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
    thead th {
      background: #1a3a6b;
      color: #fff;
      padding: 9px 12px;
      text-align: left;
      font-weight: 600;
    }
    tbody td { padding: 8px 12px; border-bottom: 1px solid #eaecf0; }
    tbody tr:nth-child(even) { background: #f7f9fc; }
    tbody tr:hover { background: #eef5fb; }
    td.num { text-align: right; font-family: "Courier New", monospace; }
    tr.sig-row { background: #fffbcc !important; font-weight: 600; }

    /* Plots */
    .plot-row {
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin: 16px 0;
    }
    .plot-wrapper {
      flex: 1 1 420px;
      background: #fff;
      border: 1px solid #dce3eb;
      border-radius: 6px;
      padding: 12px;
    }
    .plot-title {
      font-weight: 600;
      font-size: 0.88em;
      color: #555;
      margin-bottom: 6px;
    }
    .plotly-chart { width: 100%; min-height: 320px; }
    .plot-error {
      color: #c0392b;
      background: #fdf0ef;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 0.85em;
      margin: 8px 0;
    }

    /* Callout boxes */
    .callout { padding: 12px 16px; border-radius: 5px; margin: 12px 0; }
    .callout ul { padding-left: 20px; margin-top: 6px; }
    .callout-warn  { background: #fff3cd; border-left: 4px solid #f39c12; }
    .callout-info  { background: #d1ecf1; border-left: 4px solid #17a2b8; }
    .callout-ok    { background: #d4edda; border-left: 4px solid #28a745; }

    .muted { color: #888; font-style: italic; font-size: 0.9em; margin: 8px 0; }

    footer {
      margin-top: 48px;
      padding-top: 16px;
      border-top: 1px solid #dce3eb;
      text-align: center;
      color: #aaa;
      font-size: 0.82em;
    }

    @media print {
      body { background: #fff; padding: 0; }
      .container { box-shadow: none; }
    }
  </style>
</head>"""


def _html_footer() -> str:
    """Return the HTML footer."""
    return """
<footer>
  <p>Generated by DOE Toolkit &mdash; Free, open-source Design of Experiments software</p>
</footer>"""


# ---------------------------------------------------------------------------
# Remaining public helpers (used by other modules)
# ---------------------------------------------------------------------------


def export_design_with_metadata(
    design: pd.DataFrame,
    factors: List[Any],
    response_names: List[str],
    metadata: Dict[str, Any],
) -> str:
    """
    Export design matrix to CSV with metadata header comments.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix to export.
    factors : list
        List of Factor objects.
    response_names : list of str
        Names of response columns (blank placeholders added).
    metadata : dict
        Design metadata dict (design_type, n_runs, etc.).

    Returns
    -------
    str
        CSV content string.

    Examples
    --------
    >>> csv = export_design_with_metadata(df, factors, ["Y"], {})
    >>> assert csv.startswith("#")
    """
    lines: List[str] = []
    lines.append(f"# DOE Toolkit Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    for key, value in metadata.items():
        lines.append(f"# {key}: {value}")
    lines.append("#")

    export_df = design.copy()
    for resp in response_names:
        if resp not in export_df.columns:
            export_df[resp] = ""

    lines.append(export_df.to_csv(index=False))
    return "\n".join(lines)
