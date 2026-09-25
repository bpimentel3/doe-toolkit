"""
Step 5: Analyze Experimental Results

Comprehensive ANOVA analysis with multiple diagnostic views.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression

from src.ui.utils.state_management import (
    initialize_session_state,
    can_access_step,
    get_active_design,
    invalidate_downstream_state
)
from src.ui.utils.plotting import (
    PLOT_COLORS,
    apply_plot_style,
    create_parity_plot,
    create_residual_plot,
    create_logworth_plot,
    create_coefficient_significance_plot,
    create_standardized_effects_plot,
    create_qq_plot,
    create_half_normal_plot,
    _label_with_units,
)
from src.ui.components.model_builder import (
    display_model_builder,
    format_full_equation,
    format_term_for_display,
)
from src.ui.components.model_selection import display_model_selection
from src.ui.components.diagnostics_display import display_diagnostics_tab
from src.ui.components.lof_testing import display_lack_of_fit_test
from src.ui.components.profiler_display import display_profiler_tab
from src.ui.components.interaction_display import display_interaction_plot_tab
from src.ui.components.box_plot_display import display_box_plot_tab
from src.core.analysis import ANOVAAnalysis, generate_model_terms  # noqa: E402
from src.core.formatting import (  # noqa: E402
    ANOVA_TABLE_FORMATTERS,
    COEFFICIENT_TABLE_FORMATTERS,
    format_p,
    format_stat_table,
)


# ==================== HELPER FUNCTIONS ====================

def _display_anova_table(anova_table: pd.DataFrame) -> None:
    """
    Render the ANOVA table with split-plot-aware formatting.

    For split-plot designs (detected by the presence of a 'Stratum' column),
    the table is split into two clearly labelled sections — Whole-Plot stratum
    and Sub-Plot stratum — with a variance component ratio and diagnostic
    warning between them.  For non-split-plot designs the table is shown as a
    plain dataframe.

    Numeric cells are formatted for presentation only (sum_sq/SS/MS at 2
    decimal places, df as integers, F at 2 decimals, p-values per the shared
    p-value rule); the underlying values are never modified.
    """
    if 'Stratum' not in anova_table.columns:
        st.dataframe(format_stat_table(anova_table, ANOVA_TABLE_FORMATTERS), width='stretch')
        return

    display_cols = [c for c in anova_table.columns if c != 'Stratum']

    wp_rows = anova_table[anova_table['Stratum'] == 'Whole-Plot']
    sp_rows = anova_table[anova_table['Stratum'] == 'Sub-Plot']

    # --- Variance component ratio ---
    ms_wp_error = np.nan
    ms_sp_error = np.nan
    if 'WholePlot Error' in wp_rows.index and 'MS' in wp_rows.columns:
        ms_wp_error = float(wp_rows.loc['WholePlot Error', 'MS'])
    if 'SubPlot Error' in sp_rows.index and 'MS' in sp_rows.columns:
        ms_sp_error = float(sp_rows.loc['SubPlot Error', 'MS'])

    # --- Whole-Plot stratum ---
    st.markdown(
        '<div style="background:#1e3a5f;color:#a8c8f0;padding:4px 10px;'
        'border-radius:4px;font-weight:600;font-size:0.85rem;margin-bottom:2px;">'
        '▶ Whole-Plot Stratum (Hard Factors — tested vs Whole-Plot Error)'
        '</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(format_stat_table(wp_rows[display_cols], ANOVA_TABLE_FORMATTERS), width='stretch')

    # --- Variance component diagnostic ---
    if not np.isnan(ms_wp_error) and not np.isnan(ms_sp_error) and ms_sp_error > 0:
        ratio = ms_wp_error / ms_sp_error
        ratio_pct = ms_wp_error / (ms_wp_error + ms_sp_error) * 100

        if ratio > 10:
            icon, colour = '🔴', '#5c1a1a'
            severity = f'**High** ({ratio:.1f}×) — whole-plot noise dominates. Hard-factor tests have low power; consider adding whole-plot replicates.'
        elif ratio > 3:
            icon, colour = '🟡', '#4a3800'
            severity = f'**Moderate** ({ratio:.1f}×) — noticeable whole-plot variation. Interpret hard-factor p-values with caution.'
        else:
            icon, colour = '🟢', '#0d3320'
            severity = f'**Low** ({ratio:.1f}×) — whole-plot and sub-plot error are comparable.'

        st.markdown(
            f'<div style="background:{colour};padding:8px 12px;border-radius:4px;'
            f'margin:4px 0;font-size:0.85rem;">'
            f'{icon} <b>Variance component ratio</b> — '
            f'MS(WP Error) / MS(SP Error) = {ratio:.2f} '
            f'({ratio_pct:.0f}% of variance is whole-plot noise). {severity}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#1a1a2e;padding:6px 12px;border-radius:4px;'
            'margin:4px 0;font-size:0.82rem;color:#9090b0;">'
            'ℹ️ Variance component ratio not available (check degrees of freedom).'
            '</div>',
            unsafe_allow_html=True,
        )

    # --- Sub-Plot stratum ---
    st.markdown(
        '<div style="background:#1a3d1a;color:#a0d4a0;padding:4px 10px;'
        'border-radius:4px;font-weight:600;font-size:0.85rem;margin-bottom:2px;">'
        '▶ Sub-Plot Stratum (Easy Factors & Cross-Strata Interactions — tested vs Sub-Plot Error)'
        '</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(format_stat_table(sp_rows[display_cols], ANOVA_TABLE_FORMATTERS), width='stretch')


# ==================== MAIN APP ====================

initialize_session_state()

# Add standard sidebar
from src.ui.components.sidebar import build_standard_sidebar  # noqa: E402
build_standard_sidebar()

if not can_access_step(5):
    st.warning("⚠️ Please complete Steps 1-4 first")
    st.stop()

st.title("Step 5: Analyze Experimental Results")

design = get_active_design()
factors = st.session_state['factors']
responses = st.session_state.get('responses', {})

# Get response names - fallback to keys if response_names not set
response_names = st.session_state.get('response_names', list(responses.keys()) if responses else [])

# Update response_names in session state if it was missing
if responses and not st.session_state.get('response_names'):
    st.session_state['response_names'] = list(responses.keys())

# Check if we have any data at all
if not responses or not response_names:
    # If we have a design but no responses, allow viewing design information
    if design is not None and len(design) > 0:
        st.warning("⚠️ No response data loaded yet.")
        st.info(
            "You can view the design structure below, or return to Step 5 to import response data.\n\n"
            "**What you can do without response data:**\n"
            "- View design matrix\n"
            "- Check factor settings\n"
            "- Export design for data collection\n\n"
            "**To analyze results:**\n"
            "Return to Step 5 and upload a CSV with filled response columns."
        )
        
        # Show design table
        st.subheader("📋 Design Matrix")
        st.dataframe(design, width='stretch')
        
        # Show factor summary
        st.subheader("📊 Factor Summary")
        factor_summary = []
        for factor in factors:
            if factor.is_continuous():
                levels_display = f"[{factor.min_value}, {factor.max_value}]"
            elif factor.is_discrete_numeric():
                levels_display = ", ".join(str(v) for v in factor.levels)
            else:
                levels_display = ", ".join(str(v) for v in factor.levels)
            
            factor_summary.append({
                'Factor': factor.name,
                'Type': factor.factor_type.value,
                'Levels/Range': levels_display,
                'Units': factor.units or ''
            })
        
        st.dataframe(pd.DataFrame(factor_summary), width='stretch')
        
        st.divider()
        
        # Navigation
        col1, col2 = st.columns(2)
        with col1:
            if st.button("← Back to Import", width='stretch', type="primary"):
                st.switch_page("pages/5_import_results.py")
        with col2:
            if st.button("Export Design Template", width='stretch'):
                from src.ui.utils.export import export_design_with_metadata
                csv_content = export_design_with_metadata(
                    design, factors, [], 
                    st.session_state.get('design_metadata', {})
                )
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_content,
                    file_name="design_template.csv",
                    mime="text/csv",
                    width='stretch'
                )
        
        st.stop()
    else:
        # No design and no responses - shouldn't be able to access this page
        st.error("No data available. Please complete Steps 1-4 to create a design, then import data in Step 5.")
        st.stop()

st.subheader("📊 Response Selection")

selected_response = st.selectbox(
    "Select Response to Analyze", response_names, key='analysis_response_selector'
)

# Guard against None selection
if selected_response is None:
    st.error("⚠️ No response selected. Please select a response from the dropdown.")
    st.stop()

st.divider()

if 'model_terms_per_response' not in st.session_state:
    st.session_state['model_terms_per_response'] = {}

if selected_response not in st.session_state['model_terms_per_response']:
    # Pre-populate from Step 2 if available, otherwise default to linear
    if 'model_terms' in st.session_state and st.session_state['model_terms']:
        default_terms = list(st.session_state['model_terms'])
        st.info(
            f"🎯 Using model selected in Step 2:\n\n"
            f"**{format_full_equation(default_terms, selected_response)}**\n\n"
            f"You can modify it below if needed."
        )
    else:
        default_terms = generate_model_terms(factors, 'linear', include_intercept=True)
        st.info(
            f"ℹ️ No model was pre-selected. Defaulting to linear model:\n\n"
            f"**{format_full_equation(default_terms, selected_response)}**\n\n"
            f"You can modify it below."
        )
    st.session_state['model_terms_per_response'][selected_response] = default_terms

current_terms = list(st.session_state['model_terms_per_response'][selected_response])

# Build the analysis object once (before the model builder / selection panel so
# both the manual builder and the automatic selector can use it).
try:
    response_data = responses[selected_response]

    if st.session_state['excluded_rows']:
        mask = np.ones(len(design), dtype=bool)
        mask[st.session_state['excluded_rows']] = False
        design_filtered = design[mask].reset_index(drop=True)
        response_filtered = response_data[mask]
    else:
        design_filtered = design
        response_filtered = response_data

    analysis = ANOVAAnalysis(
        design=design_filtered, response=response_filtered,
        factors=factors, response_name=selected_response
    )
except Exception as e:
    st.error(f"Failed to prepare analysis data: {e}")
    st.exception(e)
    st.stop()

# ==== MODEL SELECTION ====
model_selection_method = st.radio(
    "Model Selection",
    options=["Manual", "Backward", "Forward", "Stepwise", "Stepwise (BIC)"],
    index=1,  # Default: Backward (single default for every DOE family)
    key=f"model_selection_method_{selected_response}",
    horizontal=True,
    help=(
        "Manual: keep the current model-builder workflow. Backward / Forward / "
        "Stepwise / Stepwise (BIC): automatically select a statistically "
        "justified hierarchical DOE model. DOE Toolkit defaults to Backward "
        "Elimination: the richest hierarchical model supported by the detected "
        "design is fitted first and then simplified while preserving hierarchy "
        "and strong heredity rules. Candidate terms depend on the detected "
        "design type — response-surface designs include quadratic effects, "
        "while categorical factorial designs use main effects and "
        "interactions only."
    ),
)

if model_selection_method == "Manual":
    updated_terms = display_model_builder(
        factors=factors, current_terms=current_terms.copy(), response_name=selected_response,
        key_prefix=f"model_builder_{selected_response}"
    )

    # Force update if terms changed. The builder mutates the list it is given in
    # place (operator buttons) or may return a whole new list (presets), so it
    # gets a copy and changes are detected against that untouched snapshot.
    if updated_terms != current_terms:
        st.session_state['model_terms_per_response'][selected_response] = updated_terms
        invalidate_downstream_state(from_step=5)
        st.rerun()
else:
    selection_results = display_model_selection(
        factors=factors,
        anova_analysis=analysis,
        selected_response=selected_response,
        key_prefix=f"model_selection_{selected_response}",
        method=model_selection_method,
    )

    if selection_results is not None:
        st.session_state['model_terms_per_response'][selected_response] = list(
            selection_results.final_terms
        )
        invalidate_downstream_state(from_step=5)
        st.rerun()

st.divider()

st.sidebar.header("Advanced Options")
enforce_hierarchy = st.sidebar.checkbox("Enforce Hierarchy", value=True)
block_in_prediction = st.sidebar.checkbox(
    "Include block adjustment in predictions",
    value=False,
    key="include_block_in_prediction",
    help=(
        "By default the fitted equation is averaged over blocks, so block "
        "effects do not influence predictions. Enable to keep the reference "
        "block's adjustment in predicted values. ANOVA reporting is unchanged "
        "either way."
    ),
)

st.sidebar.subheader("Data Exclusion")
if 'excluded_rows' not in st.session_state:
    st.session_state['excluded_rows'] = []

exclude_mode = st.sidebar.checkbox("Enable Row Exclusion")
if exclude_mode:
    exclude_indices = st.sidebar.multiselect(
        "Exclude Runs", options=list(range(len(design))),
        default=st.session_state['excluded_rows'],
        format_func=lambda x: f"Run {x+1}", key='exclude_runs'
    )
    if exclude_indices != st.session_state['excluded_rows']:
        st.session_state['excluded_rows'] = exclude_indices
        invalidate_downstream_state(from_step=5)

with st.spinner(f"Fitting model for {selected_response}..."):
    try:
        analysis.include_block_in_prediction = block_in_prediction
        results = analysis.fit(
            model_terms=current_terms, enforce_hierarchy_flag=enforce_hierarchy
        )

        if getattr(analysis, "rename_map", {}):
            renamed = ", ".join([f"{old} → {new}" for old, new in analysis.rename_map.items()])
            st.warning(f"Factor names renamed: {renamed}")

        if analysis.excluded_term_reasons:
            details = "  \n".join(
                f"• **{term}** — {reason}"
                for term, reason in analysis.excluded_term_reasons
            )
            st.warning(
                "Some candidate terms could not be estimated and were excluded "
                "from the fitted model:\n\n" + details
            )

        if not getattr(results, "blocks_in_predictions", True) and getattr(
            results, "block_mean_shift", 0.0
        ) != 0.0:
            st.info(
                f"Block is analyzed in the ANOVA but averaged out of the "
                f"fitted prediction equation (intercept shifted by "
                f"{results.block_mean_shift:+.4f}). Predictions do not depend "
                f"on any specific block."
            )

        if 'fitted_models' not in st.session_state:
            st.session_state['fitted_models'] = {}
        st.session_state['fitted_models'][selected_response] = results

        # Store analysis object for automatic model selection
        st.session_state[f'analysis_{selected_response}'] = analysis

    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        st.exception(e)
        st.stop()

st.subheader("📈 Analysis Results")

# Build unit lookup dicts for current responses and factors
_response_defs = st.session_state.get('response_definitions', [])
_response_units_map = {
    r['name']: r.get('units') for r in _response_defs
}
_factor_units_map = {f.name: f.units for f in factors}

# --- Observation usage: which runs were actually fitted (missing response
# rows are auto-dropped by the fit).  Alias the used subset so every plot and
# table below aligns its arrays with residuals/fitted_values.
_used_indices = getattr(results, 'used_row_indices', None)
_used_actual = getattr(results, 'used_response', None)
if _used_actual is None and _used_indices is not None:
    _used_actual = np.asarray(response_filtered)[_used_indices]
_used_design = None
if _used_indices is not None:
    _used_design = design_filtered.iloc[_used_indices].reset_index(drop=True)

if getattr(results, 'n_obs_excluded', 0) > 0:
    st.warning(
        f"⚠️ **Missing response values detected**\n\n"
        f"- **Response:** {selected_response}\n"
        f"- **Total runs:** {results.n_obs_total}\n"
        f"- **Used in analysis:** {results.n_obs_used}\n"
        f"- **Excluded:** {results.n_obs_excluded}\n\n"
        "Rows with missing response values were automatically removed before "
        "model fitting. All ANOVA results, diagnostics, predictions, profiler "
        "outputs, and optimization results are based only on the analyzed "
        "observations."
    )
    if getattr(results, 'excluded_obs_labels', None):
        with st.expander("🔍 Which runs were excluded?"):
            st.markdown(
                f"**{results.n_obs_excluded} run(s)** removed from the fit due to "
                "missing response values:"
            )
            st.caption(" · ".join(results.excluded_obs_labels))
            st.caption(
                f"Observation counts in the Model Fit summary (n = "
                f"{results.n_obs_used} / {results.n_obs_total}) and all "
                "diagnostic plots reflect only the analyzed observations."
            )

current_response_units = _response_units_map.get(selected_response)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Model Fit", "📉 Effects & Residuals",
    "🔍 Design Diagnostics", "📈 Profiler", "🕸️ Interaction Plots",
    "📦 Box Plots"
])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Actual vs Predicted**")
        # Split-plot tables use a 'P' column with a 'Model' row.
        # Standard statsmodels anova_lm tables use 'PR(>F)' with no 'Model' row;
        # the overall model p-value lives on the fitted model's f_pvalue.
        if not results.anova_table.empty and 'P' in results.anova_table.columns:
            # Split-plot path
            model_p = results.anova_table.loc['Model', 'P'] if 'Model' in results.anova_table.index else np.nan
        elif hasattr(results.fitted_model, 'f_pvalue'):
            # Standard OLS path
            model_p = float(results.fitted_model.f_pvalue)
        else:
            model_p = np.nan
        
        fig = create_parity_plot(
            _used_actual if _used_actual is not None else np.asarray(response_filtered),
            results.fitted_values,
            response_units=current_response_units,
        )
        st.plotly_chart(fig, width='stretch', theme=None)
        if np.isnan(model_p):
            p_display = "N/A"
        else:
            p_display = format_p(model_p)
        _fit_obs = f"n = {results.n_obs_used} / {results.n_obs_total}"
        if getattr(results, 'n_obs_excluded', 0) > 0:
            _fit_obs += f" ({results.n_obs_excluded} excluded)"
        st.caption(
            f"R² = {results.r_squared:.4f}   |   "
            f"Adj R² = {results.adj_r_squared:.4f}   |   "
            f"RMSE = {results.rmse:.4f}   |   "
            f"p = {p_display}   |   "
            f"{_fit_obs}"
        )
    
    with col2:
        st.markdown("**Residuals vs Fitted**")
        fig = create_residual_plot(
            results.fitted_values, results.residuals,
            response_units=current_response_units,
        )
        st.plotly_chart(fig, width='stretch')
    
    st.divider()

    # --- Effect significance: coefficient-level LogWorth vs ANOVA effects ---
    _has_coef = (
        results.coefficient_significance is not None
        and not results.coefficient_significance.empty
    )
    _has_anova_effects = (
        results.anova_effect_summary is not None
        and not results.anova_effect_summary.empty
    )

    if _has_coef or _has_anova_effects or not results.logworth.empty:
        st.markdown("## Effect Significance")
        view = st.segmented_control(
            "Effect chart view",
            options=["**Side-by-Side**", "**Coefficient LogWorth**", "**DOE Standardized Effects**"],
            default="**Side-by-Side**",
            selection_mode="single",
            label_visibility="collapsed",
        )
        show_block = st.checkbox(
            "Show block/design terms",
            value=True,
            help="Controls display only; hiding block terms does not refit the model.",
        )
        st.caption(
            "Coefficient LogWorth summarizes individual fitted coefficients. "
            "DOE standardized effects summarize term-level ANOVA tests. "
            "The results can differ for categorical, blocked, split-plot, "
            "hierarchical, and interaction models."
        )

        def _coef_fig():
            if _has_coef:
                return create_coefficient_significance_plot(
                    results.coefficient_significance, alpha=0.05, show_block=show_block
                )
            if not results.logworth.empty:
                p_values = {
                    term: 10 ** (-results.logworth.loc[term, 'LogWorth'])
                    for term in results.logworth.index
                }
                return create_logworth_plot(results.logworth, p_values)
            return None

        def _anova_fig():
            if _has_anova_effects:
                return create_standardized_effects_plot(
                    results.anova_effect_summary, alpha=0.05, show_block=show_block
                )
            return None

        _effects_caption = (
            "Bar color — blue: positive effect · red: negative effect · "
            "gray: multi-df or block/design term. "
            "Lines (shown when a single error stratum applies) — "
            "dashed: t-critical at α=0.05 · dotted: Bonferroni limit (α/m)."
        )

        def _shared_height():
            n = 0
            if _has_coef:
                n = max(n, len(results.coefficient_significance))
            if _has_anova_effects:
                n = max(n, len(results.anova_effect_summary))
            return max(320, n * 26)

        if view == "**Side-by-Side**":
            c1, c2 = st.columns(2)
            with c1:
                fig = _coef_fig()
                if fig is not None:
                    st.plotly_chart(
                        fig, width='stretch', height=_shared_height(), theme=None
                    )
            with c2:
                fig = _anova_fig()
                if fig is not None:
                    st.plotly_chart(
                        fig, width='stretch', height=_shared_height(), theme=None
                    )
                    st.caption(_effects_caption)
                else:
                    st.info(
                        "ANOVA-based standardized effects are only available "
                        "for models with a term-level ANOVA table."
                    )
        elif view == "**Coefficient LogWorth**":
            fig = _coef_fig()
            if fig is not None:
                st.plotly_chart(fig, width='stretch', theme=None)
        else:
            fig = _anova_fig()
            if fig is not None:
                st.plotly_chart(fig, width='stretch', theme=None)
                st.caption(_effects_caption)
            else:
                st.info(
                    "ANOVA-based standardized effects are only available "
                    "for models with a term-level ANOVA table."
                )

    st.divider()
    
    st.markdown("**ANOVA Table**")
    if not results.anova_table.empty:
        _display_anova_table(results.anova_table)
    
    # Lack-of-Fit test (pure-error replicates grouped over the fitted subset)
    display_lack_of_fit_test(
        _used_design if _used_design is not None else design_filtered,
        _used_actual if _used_actual is not None else np.asarray(response_filtered),
        results, factors, current_terms
    )
with tab2:
    st.markdown("**Coefficient Table**")
    st.dataframe(
        format_stat_table(results.effect_estimates, COEFFICIENT_TABLE_FORMATTERS),
        width='stretch',
    )
    
    st.divider()
    
    with st.expander("🔍 Leverage Plots", expanded=False):
        model_terms = [t for t in current_terms if t != '1']
        
        if model_terms:
            # Get model matrix from fitted model
            try:
                if hasattr(results.fitted_model, 'model'):
                    X_df = pd.DataFrame(
                        results.fitted_model.model.exog,
                        columns=results.fitted_model.model.exog_names
                    )
                else:
                    st.error("Cannot access model matrix from fitted model")
                    X_df = None
            except Exception as e:
                st.error(f"Error accessing model matrix: {e}")
                X_df = None
            
            if X_df is not None:
                for term in model_terms:
                    st.markdown(f"**{format_term_for_display(term)}**")
                    
                    try:
                        # Find corresponding column in model matrix
                        # Try multiple matching strategies
                        term_col = None
                        
                        # Strategy 1: Exact match
                        if term in X_df.columns:
                            term_col = term
                        
                        # Strategy 2: Replace * with : for interactions
                        if term_col is None and '*' in term:
                            term_colon = term.replace('*', ':')
                            if term_colon in X_df.columns:
                                term_col = term_colon
                            # Try reverse order
                            else:
                                parts = term.split('*')
                                if len(parts) == 2:
                                    reverse_term = f"{parts[1]}:{parts[0]}"
                                    if reverse_term in X_df.columns:
                                        term_col = reverse_term
                        
                        # Strategy 3: Partial match (for categorical terms with C())
                        if term_col is None:
                            for col in X_df.columns:
                                # Remove categorical encoding syntax
                                clean_col = col.replace('C(', '').replace(')', '').replace('[T.', '').replace(']', '')
                                if term == clean_col or term.replace('*', ':') == clean_col:
                                    term_col = col
                                    break
                        
                        if term_col and term_col in X_df.columns:
                            x_vals = X_df[term_col].values
                            other_cols = [c for c in X_df.columns if c != term_col and c != 'Intercept']
                            
                            if other_cols:
                                X_other = X_df[['Intercept'] + other_cols] if 'Intercept' in X_df.columns else X_df[other_cols]
                                _lev_y = _used_actual if _used_actual is not None else np.asarray(response_filtered)
                                lr_other = LinearRegression(fit_intercept=False)
                                lr_other.fit(X_other, _lev_y)
                                y_other = lr_other.predict(X_other)
                                y_adj = _lev_y - y_other + _lev_y.mean()
                            else:
                                y_adj = _used_actual if _used_actual is not None else np.asarray(response_filtered)
                        
                            fig = go.Figure()
                            
                            # Calculate 95% CI of the fit
                            lr_term = LinearRegression()
                            lr_term.fit(x_vals.reshape(-1, 1), y_adj)
                            
                            # Generate line for plotting
                            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                            y_line = lr_term.predict(x_line.reshape(-1, 1))
                            
                            # Calculate CI
                            n = len(x_vals)
                            residuals_leverage = y_adj - lr_term.predict(x_vals.reshape(-1, 1))
                            mse = np.mean(residuals_leverage**2)
                            mean_x = np.mean(x_vals)
                            se_fit = np.sqrt(mse * (1/n + (x_line - mean_x)**2 / np.sum((x_vals - mean_x)**2)))
                            t_crit = stats.t.ppf(0.975, n-2)
                            ci_width = t_crit * se_fit
                            
                            # Add 95% CI band
                            y_upper = y_line + ci_width
                            y_lower = y_line - ci_width
                            
                            fig.add_trace(go.Scatter(
                                x=np.concatenate([x_line, x_line[::-1]]),
                                y=np.concatenate([y_upper, y_lower[::-1]]),
                                fill='toself', fillcolor='rgba(128, 128, 128, 0.25)',
                                line=dict(width=0), showlegend=False, hoverinfo='skip'
                            ))
                            
                            # Add data points
                            fig.add_trace(go.Scatter(
                                x=x_vals, y=y_adj, mode='markers',
                                marker=dict(size=8, color=PLOT_COLORS['primary'], opacity=0.7,
                                           line=dict(width=0.5, color='white')),
                                name='Data', showlegend=False
                            ))
                            
                            # Add fit line
                            fig.add_trace(go.Scatter(
                                x=x_line, y=y_line, mode='lines',
                                line=dict(color=PLOT_COLORS['danger'], width=2),
                                name='Effect', showlegend=False
                            ))
                            
                            response_adj_label = _label_with_units(
                                f"{selected_response} (adjusted)",
                                current_response_units,
                            )
                            fig.update_layout(
                                xaxis_title=format_term_for_display(term),
                                yaxis_title=response_adj_label,
                                height=300, showlegend=False
                            )
                            fig = apply_plot_style(fig)
                            st.plotly_chart(fig, width='stretch')
                        else:
                            st.warning(f"Could not find term '{term}' in model matrix")
                    
                    except Exception as e:
                        st.error(f"Could not create leverage plot: {e}")
        else:
            st.info("No terms in model (intercept only)")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Residuals vs Run Order**")
        if _used_indices is not None and 'RunOrder' in design_filtered.columns:
            run_order = pd.to_numeric(
                design_filtered['RunOrder'], errors='coerce'
            ).to_numpy()[_used_indices]
            run_order = np.nan_to_num(run_order, nan=0).astype(int)
        else:
            run_order = np.arange(1, len(results.residuals) + 1)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=run_order, y=results.residuals, mode='markers+lines',
            marker=dict(size=8, color=PLOT_COLORS['primary'], opacity=0.7,
                       line=dict(width=0.5, color='white')),
            line=dict(color=PLOT_COLORS['primary'], width=1, dash='dot'),
            hovertemplate='Run %{x}<br>Residual: %{y:.3f}<extra></extra>'
        ))
        fig.add_hline(y=0, line=dict(color=PLOT_COLORS['danger'], dash='dash', width=2))
        residuals_label = _label_with_units('Residuals', current_response_units)
        fig.update_layout(xaxis_title='Run Order', yaxis_title=residuals_label, height=350)
        fig = apply_plot_style(fig)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.markdown("**Normal Q-Q Plot**")
        fig = create_qq_plot(results.residuals)
        st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    st.markdown("**Residuals vs Factors**")
    factor_cols = st.columns(min(3, len(factors)))
    
    for idx, factor in enumerate(factors):
        with factor_cols[idx % len(factor_cols)]:
            st.markdown(f"*{factor.name}*")
            if _used_indices is not None:
                factor_vals = design_filtered[factor.name].to_numpy()[_used_indices]
            else:
                factor_vals = design_filtered[factor.name].values
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=factor_vals, y=results.residuals, mode='markers',
                marker=dict(size=8, color=PLOT_COLORS['primary'], opacity=0.7,
                           line=dict(width=0.5, color='white')),
                hovertemplate=f'{factor.name}: %{{x}}<br>Residual: %{{y:.3f}}<extra></extra>'
            ))
            fig.add_hline(y=0, line=dict(color=PLOT_COLORS['danger'], dash='dash', width=2))
            factor_x_label = _label_with_units(factor.name, _factor_units_map.get(factor.name))
            residuals_label = _label_with_units('Residuals', current_response_units)
            fig.update_layout(xaxis_title=factor_x_label, yaxis_title=residuals_label, height=250)
            fig = apply_plot_style(fig)
            st.plotly_chart(fig, width='stretch')
    
    st.divider()
    
    st.markdown("**Half-Normal Plot**")
    effects_data = results.effect_estimates[results.effect_estimates.index != 'Intercept']
    if not effects_data.empty:
        # Try 'Estimate' first, then 'Coefficient'
        if 'Estimate' in effects_data.columns:
            effects = effects_data['Estimate'].values
        elif 'Coefficient' in effects_data.columns:
            effects = effects_data['Coefficient'].values
        else:
            st.info("No coefficient column found in effect estimates")
            effects = None
        
        if effects is not None:
            effect_names = [format_term_for_display(term) for term in effects_data.index]
            fig = create_half_normal_plot(effects, effect_names)
            st.plotly_chart(fig, width='stretch')
    else:
        st.info("No effects to plot (intercept-only model)")

with tab3:
    # Use diagnostics display component
    display_diagnostics_tab(
        selected_response=selected_response,
        summary=st.session_state.get('diagnostics_summary'),
        report=st.session_state.get('quality_report'),
        factors=factors,
        design=design,
        responses=responses,
        fitted_models=st.session_state.get('fitted_models', {}),
        model_terms_per_response=st.session_state.get('model_terms_per_response', {}),
        format_term_for_display=format_term_for_display,
    )
with tab4:
    # Check model availability
    if selected_response not in st.session_state['fitted_models']:
        st.warning("Please fit a model first")
        st.stop()
    
    results = st.session_state['fitted_models'][selected_response]
    
    # Use profiler display component
    display_profiler_tab(
        selected_response=selected_response,
        results=results,
        factors=factors,
        format_term_for_display=format_term_for_display,
        response_units=current_response_units,
    )
with tab5:
    # Interaction plots require a fitted model for the significance overlay.
    if selected_response not in st.session_state['fitted_models']:
        st.warning("Please fit a model first")
        st.stop()

    interaction_results = st.session_state['fitted_models'][selected_response]

    display_interaction_plot_tab(
        selected_response=selected_response,
        design=_used_design if _used_design is not None else design_filtered,
        response=_used_actual if _used_actual is not None else response_filtered,
        factors=factors,
        results=interaction_results,
        response_units=current_response_units,
    )
with tab6:
    display_box_plot_tab(
        selected_response=selected_response,
        design=_used_design if _used_design is not None else design_filtered,
        response=_used_actual if _used_actual is not None else response_filtered,
        factors=factors,
        response_units=current_response_units,
    )
st.divider()

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("← Back to Import", width='stretch'):
        st.session_state['current_step'] = 5
        st.switch_page("pages/5_import_results.py")

with col2:
    if st.session_state.get('quality_report'):
        if st.session_state['quality_report'].summary.needs_any_augmentation():
            if st.button("🔬 Augmentation", type="primary", width='stretch'):
                st.session_state['current_step'] = 7
                st.session_state['show_augmentation'] = True
                st.switch_page("pages/7_augmentation.py")

with col3:
    if st.button("Optimize →", width='stretch'):
        st.session_state['current_step'] = 8
        st.switch_page("pages/8_optimize.py")