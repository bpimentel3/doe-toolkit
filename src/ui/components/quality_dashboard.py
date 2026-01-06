"""
Design Quality Dashboard for DOE Toolkit.

Displays quality assessment after ANOVA analysis, showing per-response
diagnostics and augmentation recommendations.
"""

from typing import Dict, Optional
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from src.core.diagnostics import (
    DesignDiagnosticSummary,
    DesignQualityReport,
    ResponseQualityAssessment
)


def display_quality_dashboard(
    report: DesignQualityReport,
    show_augmentation_button: bool = True
) -> None:
    """
    Display complete design quality dashboard.
    
    Parameters
    ----------
    report : DesignQualityReport
        Quality report from diagnostics
    show_augmentation_button : bool
        Whether to show "View Recommendations" button
    """
    st.header("📊 Design Quality Report")
    
    # Overall status
    _display_overall_status(report)
    
    st.divider()
    
    # Per-response quality cards
    st.subheader("Response-Level Quality Assessment")
    
    for response_name, assessment in report.response_quality.items():
        _display_response_card(response_name, assessment)
    
    st.divider()
    
    # Augmentation recommendation
    if report.summary.needs_any_augmentation() and show_augmentation_button:
        _display_augmentation_prompt(report)


def _display_overall_status(report: DesignQualityReport) -> None:
    """Display overall design health status."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Design Type",
            report.summary.design_type.replace('_', ' ').title()
        )
    
    with col2:
        st.metric("Total Runs", report.summary.n_runs)
    
    with col3:
        st.metric("Factors", report.summary.n_factors)
    
    with col4:
        st.metric("Responses", len(report.summary.response_diagnostics))
    
    # Status banner
    if report.critical_issues:
        st.error(
            f"⚠️ **{len(report.critical_issues)} Critical Issue(s) Detected**\n\n"
            "These issues should be addressed before proceeding to optimization."
        )
        
        with st.expander("View Critical Issues"):
            for issue in report.critical_issues:
                st.markdown(f"- {issue}")
    
    elif report.warnings:
        st.warning(
            f"⚡ **{len(report.warnings)} Warning(s)**\n\n"
            "Design quality could be improved."
        )
        
        with st.expander("View Warnings"):
            for warning in report.warnings:
                st.markdown(f"- {warning}")
    
    else:
        st.success(
            "✅ **Design Quality Satisfactory**\n\n"
            "All responses show adequate model fit and precision."
        )
        
        if report.satisfactory_aspects:
            with st.expander("Details"):
                for aspect in report.satisfactory_aspects:
                    st.markdown(f"- {aspect}")


def _display_response_card(
    response_name: str,
    assessment: ResponseQualityAssessment
) -> None:
    """Display quality card for one response."""
    
    # Grade color mapping
    grade_colors = {
        'Excellent': 'green',
        'Good': 'blue',
        'Fair': 'orange',
        'Poor': 'red',
        'Inadequate': 'red'
    }
    
    grade_icons = {
        'Excellent': '🌟',
        'Good': '✅',
        'Fair': '⚡',
        'Poor': '⚠️',
        'Inadequate': '❌'
    }
    
    color = grade_colors.get(assessment.overall_grade, 'gray')
    icon = grade_icons.get(assessment.overall_grade, '❓')
    
    with st.expander(
        f"{icon} **{response_name}** — {assessment.overall_grade}",
        expanded=(assessment.overall_grade in ['Poor', 'Inadequate'])
    ):
        
        # Metrics row
        diag = assessment.diagnostics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R²", f"{diag.r_squared:.3f}")
        
        with col2:
            st.metric("Adj R²", f"{diag.adj_r_squared:.3f}")
        
        with col3:
            st.metric("RMSE", f"{diag.rmse:.2f}")
        
        # Issues
        if assessment.issues:
            st.markdown("**Issues Detected:**")
            
            for issue in assessment.issues:
                if issue.severity == 'critical':
                    st.error(f"**✗ {issue.description}**")
                elif issue.severity == 'warning':
                    st.warning(f"**⚡ {issue.description}**")
                else:
                    st.info(f"ℹ️ {issue.description}")
                
                st.markdown(f"*Recommendation:* {issue.recommended_action}")
                st.markdown("")
        
        else:
            st.success("✓ No issues detected for this response")
        
        # Additional diagnostics (collapsible)
        with st.expander("View Detailed Diagnostics"):
            _display_detailed_diagnostics(diag)


def _display_detailed_diagnostics(diag) -> None:
    """Display detailed diagnostic information."""
    
    # Prediction variance
    if diag.prediction_variance_stats:
        st.markdown("**Prediction Variance:**")
        stats = diag.prediction_variance_stats
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min", f"{stats['min']:.3f}")
        with col2:
            st.metric("Mean", f"{stats['mean']:.3f}")
        with col3:
            st.metric("Max", f"{stats['max']:.3f}")
        
        if stats.get('max_ratio'):
            ratio = stats['max_ratio']
            if ratio > 3:
                st.warning(f"Max/Min Ratio: {ratio:.1f} (>3 indicates non-uniformity)")
            else:
                st.info(f"Max/Min Ratio: {ratio:.1f} (acceptable)")
    
    # VIF values
    if diag.vif_values:
        st.markdown("**Variance Inflation Factors:**")
        
        # Sort by VIF descending
        sorted_vif = sorted(
            diag.vif_values.items(),
            key=lambda x: x[1] if not pd.isna(x[1]) else -1,
            reverse=True
        )
        
        vif_data = []
        for term, vif in sorted_vif:
            if pd.isna(vif) or pd.isinf(vif):
                vif_str = "∞"
                status = "Singular"
            elif vif > 10:
                vif_str = f"{vif:.1f}"
                status = "High (>10)"
            elif vif > 5:
                vif_str = f"{vif:.1f}"
                status = "Moderate"
            else:
                vif_str = f"{vif:.1f}"
                status = "Good"
            
            vif_data.append({'Term': term, 'VIF': vif_str, 'Status': status})
        
        st.dataframe(pd.DataFrame(vif_data), use_container_width=True)
    
    # Significant effects
    if diag.significant_effects or diag.marginally_significant:
        st.markdown("**Effect Significance:**")
        
        if diag.significant_effects:
            st.success(f"**Significant (p < 0.05):** {', '.join(diag.significant_effects)}")
        
        if diag.marginally_significant:
            st.info(f"**Marginal (0.05 < p < 0.10):** {', '.join(diag.marginally_significant)}")
    
    # Aliasing (for fractional designs)
    if diag.resolution:
        st.markdown(f"**Design Resolution:** {diag.resolution}")
        
        if diag.confounded_interactions:
            st.warning("**Critical Confounding Detected:**")
            for main_effect, interaction in diag.confounded_interactions:
                st.markdown(f"- {main_effect} aliased with {interaction}")


def _display_augmentation_prompt(report: DesignQualityReport) -> None:
    """Display augmentation recommendation prompt."""
    
    st.subheader("💡 Augmentation Recommendations Available")
    
    # Summary of what augmentation could help
    st.info(
        f"**Recommended Action:** {report.unified_strategy}\n\n"
        "Design augmentation can help address the issues identified above by "
        "strategically adding experimental runs."
    )
    
    # Show which responses would benefit
    responses_needing_help = [
        name for name, diag in report.summary.response_diagnostics.items()
        if diag.needs_augmentation
    ]
    
    if responses_needing_help:
        st.markdown(
            f"**Responses that would benefit:** "
            f"{', '.join(responses_needing_help)}"
        )
    
    # Conflict warning
    if report.conflict_resolution:
        st.warning(
            "**Note:** Multiple responses have different needs. "
            f"{report.conflict_resolution}"
        )
    
    # Button to view recommendations
    if st.button("🔬 View Augmentation Plans", type="primary", use_container_width=True):
        st.session_state['show_augmentation'] = True
        st.session_state['current_step'] = 6
        st.rerun()


def display_quality_comparison(
    original_report: DesignQualityReport,
    augmented_report: DesignQualityReport
) -> None:
    """
    Display before/after quality comparison.
    
    Used after augmentation to show improvements.
    
    Parameters
    ----------
    original_report : DesignQualityReport
        Report from original design
    augmented_report : DesignQualityReport
        Report after augmentation
    """
    st.header("📈 Quality Improvement Summary")
    
    # Overall metrics comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Design")
        st.metric("Runs", original_report.summary.n_runs)
        st.metric("Critical Issues", len(original_report.critical_issues))
        st.metric("Warnings", len(original_report.warnings))
    
    with col2:
        st.subheader("Augmented Design")
        st.metric(
            "Runs",
            augmented_report.summary.n_runs,
            delta=augmented_report.summary.n_runs - original_report.summary.n_runs
        )
        st.metric(
            "Critical Issues",
            len(augmented_report.critical_issues),
            delta=len(augmented_report.critical_issues) - len(original_report.critical_issues),
            delta_color="inverse"
        )
        st.metric(
            "Warnings",
            len(augmented_report.warnings),
            delta=len(augmented_report.warnings) - len(original_report.warnings),
            delta_color="inverse"
        )
    
    st.divider()
    
    # Per-response comparison
    st.subheader("Response-Level Improvements")
    
    for response_name in original_report.response_quality.keys():
        original_assessment = original_report.response_quality[response_name]
        augmented_assessment = augmented_report.response_quality.get(response_name)
        
        if augmented_assessment:
            _display_response_comparison(
                response_name,
                original_assessment,
                augmented_assessment
            )


def _display_response_comparison(
    response_name: str,
    original: ResponseQualityAssessment,
    augmented: ResponseQualityAssessment
) -> None:
    """Display before/after comparison for one response."""
    
    with st.expander(f"**{response_name}** — {original.overall_grade} → {augmented.overall_grade}"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Original**")
            diag_orig = original.diagnostics
            st.metric("R²", f"{diag_orig.r_squared:.3f}")
            st.metric("RMSE", f"{diag_orig.rmse:.2f}")
            st.metric("Issues", len(original.issues))
        
        with col2:
            st.markdown("**Augmented**")
            diag_aug = augmented.diagnostics
            st.metric(
                "R²",
                f"{diag_aug.r_squared:.3f}",
                delta=f"{diag_aug.r_squared - diag_orig.r_squared:+.3f}"
            )
            st.metric(
                "RMSE",
                f"{diag_aug.rmse:.2f}",
                delta=f"{diag_aug.rmse - diag_orig.rmse:+.2f}",
                delta_color="inverse"
            )
            st.metric(
                "Issues",
                len(augmented.issues),
                delta=len(augmented.issues) - len(original.issues),
                delta_color="inverse"
            )
        
        # Show resolved issues
        resolved_issues = set(i.description for i in original.issues) - set(
            i.description for i in augmented.issues
        )
        
        if resolved_issues:
            st.success("**Resolved Issues:**")
            for issue in resolved_issues:
                st.markdown(f"- {issue}")
        
        # Show remaining issues
        remaining_issues = [i for i in augmented.issues if i.severity in ('critical', 'warning')]
        
        if remaining_issues:
            st.warning(f"**Remaining Issues:** {len(remaining_issues)}")