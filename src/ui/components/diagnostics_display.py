"""
Diagnostics display component for ANOVA analysis.

This module handles the display of design diagnostics including:
- Variance Inflation Factors (VIF)
- Alias structure for fractional designs
- Prediction variance statistics
- Effect significance summaries
- Detected issues and recommendations
- High leverage points

All components are designed to work within Streamlit's UI framework.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from src.core.diagnostics.summary import DesignDiagnosticSummary, QualityReport


def display_diagnostics_tab(
    selected_response: str,
    summary: Optional[DesignDiagnosticSummary],
    report: Optional[QualityReport],
    factors,
    design,
    responses: Dict,
    fitted_models: Dict,
    model_terms_per_response: Dict,
    format_term_for_display,
) -> None:
    """
    Display the Design Diagnostics tab content.

    Parameters
    ----------
    selected_response : str
        Name of the currently selected response
    summary : Optional[DesignDiagnosticSummary]
        Computed diagnostic summary, if available
    report : Optional[QualityReport]
        Quality report, if available
    factors : list
        List of Factor objects
    design : pd.DataFrame
        Design matrix
    responses : Dict
        Dictionary of response data
    fitted_models : Dict
        Dictionary of fitted models
    model_terms_per_response : Dict
        Dictionary of model terms per response
    format_term_for_display : callable
        Function to format term names for display

    Returns
    -------
    None
        Displays content directly in Streamlit
    """
    st.subheader("📋 Design Diagnostics")

    # Get diagnostics from session state if available
    if summary and report:
        _display_computed_diagnostics(
            selected_response, summary, report, format_term_for_display
        )
    else:
        _display_diagnostics_prompt(
            design, responses, fitted_models, factors, model_terms_per_response
        )


def _display_computed_diagnostics(
    selected_response: str,
    summary: DesignDiagnosticSummary,
    report: QualityReport,
    format_term_for_display,
) -> None:
    """Display diagnostics when they have been computed."""
    # Get diagnostics for current response
    if selected_response in summary.response_diagnostics:
        diag = summary.response_diagnostics[selected_response]

        # Metrics overview
        _display_metrics_overview(diag, report, selected_response)

        st.divider()

        # Variance Inflation Factors
        _display_vif_table(diag, format_term_for_display)

        st.divider()

        # Alias Structure (for fractional designs)
        _display_alias_structure(diag)

        st.divider()

        # Prediction Variance
        _display_prediction_variance(diag)

        st.divider()

        # Effect Significance Summary
        _display_effect_significance(diag, format_term_for_display)

        st.divider()

        # Issues Summary
        _display_issues_summary(diag)

        # High Leverage Points
        _display_high_leverage_points(diag)

    else:
        st.warning(f"No diagnostics available for {selected_response}")


def _display_metrics_overview(diag, report, selected_response: str) -> None:
    """Display metrics overview cards."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("R²", f"{diag.r_squared:.3f}")

    with col2:
        st.metric("Adj R²", f"{diag.adj_r_squared:.3f}")

    with col3:
        st.metric("RMSE", f"{diag.rmse:.2f}")

    with col4:
        grade = report.response_quality[selected_response].overall_grade
        grade_icon = {
            "Excellent": "🌟",
            "Good": "✅",
            "Fair": "⚡",
            "Poor": "⚠️",
            "Inadequate": "❌",
        }.get(grade, "❓")
        st.metric("Grade", f"{grade_icon} {grade}")


def _display_vif_table(diag, format_term_for_display) -> None:
    """Display Variance Inflation Factors table."""
    st.markdown("### Variance Inflation Factors (VIF)")
    st.caption(
        "VIF measures multicollinearity. VIF > 10 indicates problematic collinearity."
    )

    if diag.vif_values:
        vif_data = []
        for term, vif in sorted(
            diag.vif_values.items(),
            key=lambda x: (
                x[1] if not pd.isna(x[1]) and not np.isinf(x[1]) else -1
            ),
            reverse=True,
        ):
            if pd.isna(vif) or np.isinf(vif):
                vif_str = "∞"
                status = "⚠️ Singular"
            elif vif > 10:
                vif_str = f"{vif:.2f}"
                status = "❌ High"
            elif vif > 5:
                vif_str = f"{vif:.2f}"
                status = "⚡ Moderate"
            else:
                vif_str = f"{vif:.2f}"
                status = "✅ Good"

            vif_data.append(
                {
                    "Term": format_term_for_display(term),
                    "VIF": vif_str,
                    "Status": status,
                }
            )

        if vif_data:
            # Display as standard dataframe
            vif_df = pd.DataFrame(vif_data)
            st.dataframe(vif_df, use_container_width=True, hide_index=True)

            # Add interpretation
            high_vif = [
                row["Term"]
                for row in vif_data
                if "❌" in row["Status"] or "⚠️" in row["Status"]
            ]
            if high_vif:
                st.warning(
                    f"**High collinearity detected in:** {', '.join(high_vif)}\n\n"
                    "These terms are highly correlated with other predictors. "
                    "Consider removing redundant terms or adding orthogonalizing runs."
                )
        else:
            st.info("No VIF values computed (intercept-only model)")
    else:
        st.info("VIF not available (may be saturated design)")


def _display_alias_structure(diag) -> None:
    """Display alias structure for fractional designs."""
    if diag.resolution is not None:
        st.markdown("### Alias Structure")
        st.caption(f"Design Resolution: **{diag.resolution}**")

        if diag.aliased_effects:
            st.markdown("**Confounding Patterns:**")

            # Group by aliasing severity
            critical_aliases = []
            other_aliases = []

            for effect, aliases in diag.aliased_effects.items():
                if aliases:
                    alias_str = f"**{effect}** = {' = '.join(aliases)}"

                    # Check if main effect aliased with 2FI (critical)
                    if len(effect) == 1 and any(len(a) == 2 for a in aliases):
                        critical_aliases.append(alias_str)
                    else:
                        other_aliases.append(alias_str)

            if critical_aliases:
                st.error("**Critical Confounding (Main effects with 2FI):**")
                for alias in critical_aliases:
                    st.markdown(f"- {alias}")

            if other_aliases:
                with st.expander("View Other Confounding Patterns"):
                    for alias in other_aliases:
                        st.markdown(f"- {alias}")

            if diag.resolution <= 3:
                st.warning(
                    f"⚠️ Resolution {diag.resolution} design: Main effects are aliased with "
                    "2-factor interactions. Consider foldover to increase resolution."
                )
        else:
            st.success("✅ No critical aliasing detected")


def _display_prediction_variance(diag) -> None:
    """Display prediction variance statistics."""
    if diag.prediction_variance_stats:
        st.markdown("### Prediction Variance")
        st.caption(
            "Lower variance = more precise predictions. High max/mean ratio indicates non-uniform precision."
        )

        stats_dict = diag.prediction_variance_stats

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Min", f"{stats_dict.get('min', 0):.3f}")
        with col2:
            st.metric("Mean", f"{stats_dict.get('mean', 0):.3f}")
        with col3:
            st.metric("Max", f"{stats_dict.get('max', 0):.3f}")
        with col4:
            max_val = stats_dict.get("max", 0)
            mean_val = stats_dict.get("mean", 1)
            ratio = max_val / mean_val if mean_val > 0 else 0
            st.metric("Max/Mean", f"{ratio:.2f}")

        if ratio > 3:
            st.warning(
                f"⚠️ High variance ratio ({ratio:.1f}x) indicates non-uniform precision. "
                "Some regions of the design space have much higher prediction variance."
            )
        else:
            st.success("✅ Prediction variance is reasonably uniform")


def _display_effect_significance(diag, format_term_for_display) -> None:
    """Display effect significance summary."""
    st.markdown("### Effect Significance")

    col1, col2 = st.columns(2)

    with col1:
        if diag.significant_effects:
            st.success(
                f"**Significant (p < 0.05):** {len(diag.significant_effects)}"
            )
            for effect in diag.significant_effects:
                st.markdown(f"- {format_term_for_display(effect)}")
        else:
            st.info("No significant effects detected")

    with col2:
        if diag.marginally_significant:
            st.warning(
                f"**Marginal (0.05 < p < 0.10):** {len(diag.marginally_significant)}"
            )
            for effect in diag.marginally_significant:
                st.markdown(f"- {format_term_for_display(effect)}")
        else:
            st.info("No marginally significant effects")


def _display_issues_summary(diag) -> None:
    """Display detected issues and recommendations."""
    if diag.issues:
        st.markdown("### Issues Detected")

        for issue in diag.issues:
            if issue.severity == "critical":
                st.error(f"**✗ {issue.description}**")
                st.markdown(f"*Recommendation:* {issue.recommended_action}")
                if issue.affected_terms:
                    st.markdown(
                        f"*Affected terms:* {', '.join(issue.affected_terms)}"
                    )
            elif issue.severity == "warning":
                st.warning(f"**⚡ {issue.description}**")
                st.markdown(f"*Recommendation:* {issue.recommended_action}")
                if issue.affected_terms:
                    st.markdown(
                        f"*Affected terms:* {', '.join(issue.affected_terms)}"
                    )
            else:
                st.info(f"ℹ️ {issue.description}")
                st.markdown(f"*Recommendation:* {issue.recommended_action}")
    else:
        st.success("✅ No critical issues detected for this response")


def _display_high_leverage_points(diag) -> None:
    """Display high leverage points warning."""
    if diag.high_leverage_points:
        st.divider()
        st.markdown("### High Leverage Points")
        st.caption(
            f"Found {len(diag.high_leverage_points)} observation(s) with high leverage"
        )

        st.warning(
            f"**Runs with high leverage:** {', '.join(map(str, [p+1 for p in diag.high_leverage_points]))}\n\n"
            "High leverage points have unusual factor combinations and can disproportionately "
            "influence the model. Verify these runs for accuracy."
        )


def _display_diagnostics_prompt(
    design, responses, fitted_models, factors, model_terms_per_response
) -> None:
    """Display prompt to generate diagnostics."""
    st.info(
        "💡 Click to generate comprehensive design diagnostics including VIF, alias structure, prediction variance, and quality assessment."
    )

    if st.button("Generate Diagnostics", type="primary"):
        with st.spinner("Computing diagnostics..."):
            try:
                from src.core.diagnostics.summary import (
                    compute_design_diagnostic_summary,
                    generate_quality_report,
                )

                # Prepare metadata
                design_metadata = {
                    "design_type": st.session_state.get("design_type", "unknown"),
                    "generators": st.session_state.get("design_metadata", {}).get(
                        "generators"
                    ),
                    "is_split_plot": st.session_state.get("design_metadata", {}).get(
                        "is_split_plot", False
                    ),
                    "has_blocking": "Block" in design.columns,
                    "has_center_points": st.session_state.get(
                        "design_metadata", {}
                    ).get("has_center_points", False),
                }

                # Compute diagnostics
                summary = compute_design_diagnostic_summary(
                    design=design,
                    responses=responses,
                    fitted_models={
                        k: v.fitted_model for k, v in fitted_models.items()
                    },
                    factors=factors,
                    model_terms_per_response=model_terms_per_response,
                    design_metadata=design_metadata,
                )

                # Generate quality report
                report = generate_quality_report(summary)

                # Save to session state
                st.session_state["diagnostics_summary"] = summary
                st.session_state["quality_report"] = report

                st.success("✅ Diagnostics computed successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Failed to compute diagnostics: {e}")
                st.exception(e)