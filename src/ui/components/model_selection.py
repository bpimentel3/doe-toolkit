"""
Automatic Model Selection UI component for the Step 6 model area.

Renders the Design-Expert style selector (Manual / Forward / Backward /
Stepwise), runs the selection engine, and displays the selection report,
per-candidate table, model-quality summary and predictive-quality warning.
"""

from typing import Optional

import pandas as pd
import streamlit as st

from src.core.selection import (
    LOF_HELP,
    LOF_UNAVAILABLE_TEXT,
    ModelSelectionResults,
    candidate_model_pool,
    factor_display_map,
    format_selection_summary,
    map_term_display,
    near_perfect_fit_message,
    predicted_r2_warning,
    quadratic_omission_notes,
    run_model_selection,
)
from src.core.formatting import format_coefficient, format_p
from src.ui.components.model_builder import (
    format_full_equation,
    format_term_for_display,
)

METHODS = ["Manual", "Backward", "Forward", "Stepwise", "Stepwise (BIC)"]
METHOD_KEY = {
    "Backward": "backward",
    "Forward": "forward",
    "Stepwise": "stepwise",
    "Stepwise (BIC)": "stepwise_bic",
}
REASON_HELP = (
    "**Reason** for each candidate's disposition:\n"
    "- *Significant* — included and p ≤ alpha in the full candidate model\n"
    "- *Kept for hierarchy* — included despite p > alpha because a higher-order "
    "term depends on it (parent protection)\n"
    "- *Added during forward/stepwise selection* — included with p > alpha, "
    "not required by heredity (rare)\n"
    "- *Removed by backward elimination* — dropped because p > alpha\n"
    "- *Not significant* — never met the entry threshold\n"
    "- *Excluded (model saturated)* — pruned because the design lacks degrees "
    "of freedom for the full pool\n"
    "- *Added during stepwise (BIC) selection* — included because BIC improved "
    "by at least the threshold\n"
    "- *Removed during stepwise (BIC) selection* — dropped because BIC "
    "improved by at least the threshold\n"
    "- *Selected / Not selected by BIC stepwise* — BIC-based disposition with "
    "no explicit add/remove event"
)


def _format_coefficient(value) -> str:
    return format_coefficient(value)


def _format_p(value) -> str:
    return format_p(value)


def _render_selection_results(
    results: ModelSelectionResults,
    response_name: str,
    key_prefix: str,
    display_map: Optional[dict] = None,
    candidate_pool: Optional[list] = None,
) -> None:
    """Render the full report, candidate table, and quality summary."""
    with st.expander("📋 Automatic Model Selection Results", expanded=False):
        st.markdown(format_selection_summary(
            results, response_name,
            factor_display_map=display_map,
            candidate_pool=candidate_pool,
        ))

        if results.warnings:
            for warning in results.warnings:
                st.warning(warning)
        if results.oversaturated:
            st.warning(
                "⚠️ The full candidate model could not be fitted (degrees of freedom "
                "exhausted). The candidate table below was pruned to the fit-able "
                "subset."
            )

        st.markdown("**Selection Table (all candidates):**")
        table = results.term_table.copy()
        display = pd.DataFrame({
            "Term": [
                format_term_for_display(map_term_display(t, display_map))
                for t in table["Term"]
            ],
            "Coefficient": [_format_coefficient(c) for c in table["Coefficient"]],
            "p-value": [_format_p(p) for p in table["p-value"]],
            "Included": table["Included"],
            "Reason": table["Reason"],
        })
        st.dataframe(display, width='stretch', hide_index=True)
        st.caption(REASON_HELP)

        quality = results.model_quality or {}

        lack_of_fit = quality.get("lack_of_fit_p_value")

        quality_rows = [
            ("Model p-value", _format_p(quality.get("model_p_value"))),
            ("Lack-of-Fit p-value", _format_p(lack_of_fit) if lack_of_fit is not None else LOF_UNAVAILABLE_TEXT),
            ("R²", f"{quality['r_squared']:.4f}" if quality.get("r_squared") is not None else "N/A"),
            ("Adjusted R²", f"{quality['adj_r_squared']:.4f}" if quality.get("adj_r_squared") is not None else "N/A"),
            ("Predicted R²", f"{quality['predicted_r_squared']:.4f}" if quality.get("predicted_r_squared") is not None else "N/A"),
            ("PRESS", f"{quality['press']:.4g}" if quality.get("press") is not None else "N/A"),
        ]
        st.markdown("**Model Quality Summary:**")

        column_config = None
        if lack_of_fit is None:
            column_config = {"Value": st.column_config.TextColumn(help=LOF_HELP)}
        st.dataframe(
            pd.DataFrame(quality_rows, columns=["Metric", "Value"]),
            width='stretch',
            hide_index=True,
            column_config=column_config,
        )
        st.caption(
            "Desired: significant model p-value, non-significant lack-of-fit "
            "(requires replicate runs), and good agreement between Adjusted R² "
            "and Predicted R²."
        )

        warning = predicted_r2_warning(
            quality.get("adj_r_squared"), quality.get("predicted_r_squared")
        )
        if warning:
            st.warning(warning)

        with st.expander("🔬 Developer view — raw model terms"):
            st.caption("Raw patsy term strings used for fitting (internal names).")
            st.code(" + ".join(results.final_terms))

    near_perfect = near_perfect_fit_message(
        quality.get("r_squared"),
        quality.get("adj_r_squared"),
        quality.get("predicted_r_squared"),
    )
    if near_perfect:
        st.info(near_perfect)

    curr_terms = st.session_state.get("model_terms_per_response", {}).get(response_name, ["1"])
    display_terms = [map_term_display(t, display_map) for t in curr_terms]
    st.markdown(f"**Selected model (now in use):** {format_full_equation(display_terms, response_name)}")
    st.info("✅ Model terms updated — ANOVA, coefficients, diagnostics, "
            "prediction equations, contour and surface plots now use the "
            "selected model. Switch to **Manual** mode any time to tweak "
            "the terms by hand.")


def display_model_selection(
    factors,
    anova_analysis,
    selected_response: str,
    key_prefix: str = "",
    method: str = "Backward",
) -> Optional[ModelSelectionResults]:
    """
    Display the automatic model-selection controls and report.

    Parameters
    ----------
    factors : List[Factor]
        Design factors.
    anova_analysis : ANOVAAnalysis
        Fitted analysis object (selection refits internally).
    selected_response : str
        Response currently being modeled.
    key_prefix : str
        Prefix for Streamlit widget keys.
    method : str
        One of ``'Backward'``, ``'Forward'``, ``'Stepwise'``.

    Returns
    -------
    ModelSelectionResults, optional
        Fresh results when the user just ran a selection, else None.
    """
    state_key = "model_selection_results"
    if state_key not in st.session_state:
        st.session_state[state_key] = {}

    candidate_pool, design_type = candidate_model_pool(factors, anova_analysis)

    display_map = factor_display_map(
        factors, getattr(anova_analysis, "rename_map", None) or {}
    )

    st.markdown(f"**Detected Design Type:** {design_type}")
    omitted_notes = quadratic_omission_notes(factors, anova_analysis)
    if omitted_notes:
        with st.expander(f"ℹ️ Quadratic terms omitted ({len(omitted_notes)})", expanded=False):
            for note in omitted_notes:
                st.caption(note)
    st.markdown("**Candidate Pool** (all factors, hierarchical):")
    st.caption(
        ", ".join(
            format_term_for_display(map_term_display(t, display_map))
            for t in candidate_pool
        ) or "(none)"
    )

    is_bic = method == "Stepwise (BIC)"
    if is_bic:
        bic_threshold = st.slider(
            "BIC improvement threshold",
            min_value=0.1,
            max_value=10.0,
            value=2.0,
            step=0.1,
            key=f"{key_prefix}_bic_threshold",
            help=(
                "A term is added/removed only when it improves the model BIC "
                "by at least this many units. Values around 2 roughly match "
                "0.05 significance via the Schwarz approximation."
            ),
        )
        alpha = 0.10
    else:
        alpha = st.number_input(
            "Alpha (entry/exit significance)",
            min_value=0.01,
            max_value=0.50,
            value=0.10,
            step=0.01,
            key=f"{key_prefix}_alpha",
            help=(
                "A term is selected when its p-value in the full candidate model "
                "is <= alpha. Hierarchy always overrides significance: a parent "
                "of a selected higher-order term is kept even if its own p-value "
                "exceeds alpha."
            ),
        )

    if st.button(
        f"🚀 Run {method} Selection",
        type="primary",
        width='stretch',
        key=f"{key_prefix}_run_button",
        help=(
            "Automatically select a statistically valid hierarchical DOE model. "
            "DOE Toolkit defaults to Backward Elimination: the richest "
            "hierarchical model supported by the detected design is fitted "
            "first and then simplified while preserving hierarchy and strong "
            "heredity rules. Significance is judged in the full candidate "
            "model, so selection stays consistent with the table below even "
            "when curvature/interactions dominate the response."
        ),
    ):
        method_key = METHOD_KEY.get(method)
        if method_key is None:
            st.error("Manual mode does not run automatic selection.")
            return None
        method_label = method
        progress_bar = st.progress(0, text=f"Initializing {method_key.replace('_', ' ').lower()} selection...")

        def update_progress(current: int, total: int):
            progress_bar.progress(
                min(current / total, 1.0),
                text=f"{method_label} selection: step {current}/{total}...",
            )

        with st.spinner(f"Running {method_label.lower()} selection..."):
            try:
                progress_cb = update_progress if not is_bic else None
                results = run_model_selection(
                    anova_analysis,
                    method=method_key,
                    alpha=alpha,
                    bic_threshold=bic_threshold if is_bic else 2.0,
                    progress_callback=progress_cb,
                )
                progress_bar.empty()
                st.session_state[state_key][selected_response] = results
                if results.selection_path:
                    _render_selection_results(
                        results, selected_response, key_prefix,
                        display_map=display_map, candidate_pool=candidate_pool,
                    )
                st.success(f"✓ Model selection completed in {results.n_iterations} step(s).")
                return results
            except Exception as e:
                progress_bar.empty()
                st.error(f"Automatic model selection failed: {e}")
                st.exception(e)
                return None

    persisted = st.session_state[state_key].get(selected_response)
    if persisted is not None:
        st.divider()
        _render_selection_results(
            persisted, selected_response, key_prefix,
            display_map=display_map, candidate_pool=candidate_pool,
        )
        if st.button("🗑️ Clear results", key=f"{key_prefix}_clear_results"):
            st.session_state[state_key].pop(selected_response, None)
            st.rerun()

    return None