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
    create_qq_plot,
    create_half_normal_plot
)
from src.ui.components.model_builder import display_model_builder, format_term_for_display, display_stepwise_button
from src.ui.components.diagnostics_display import display_diagnostics_tab
from src.ui.components.lof_testing import display_lack_of_fit_test
from src.ui.components.profiler_display import display_profiler_tab
from src.core.analysis import ANOVAAnalysis, generate_model_terms


# ==================== MAIN APP ====================

initialize_session_state()

# Add standard sidebar
from src.ui.components.sidebar import build_standard_sidebar
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

if not responses or not response_names:
    st.error("No response data available. Please import data in Step 5 (Import Results).")
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
        default_terms = st.session_state['model_terms']
        st.info("🎯 Using model selected in Step 2. You can modify it below if needed.")
    else:
        default_terms = generate_model_terms(factors, 'linear', include_intercept=True)
        st.info("ℹ️ No model was pre-selected. Defaulting to linear model. You can modify it below.")
    st.session_state['model_terms_per_response'][selected_response] = default_terms

current_terms = st.session_state['model_terms_per_response'][selected_response]

updated_terms = display_model_builder(
    factors=factors, current_terms=current_terms, response_name=selected_response,
    key_prefix=f"model_builder_{selected_response}"
)

# Force update if terms changed
if updated_terms != current_terms:
    st.session_state['model_terms_per_response'][selected_response] = updated_terms
    invalidate_downstream_state(from_step=5)
    st.rerun()


st.divider()

st.sidebar.header("Advanced Options")
enforce_hierarchy = st.sidebar.checkbox("Enforce Hierarchy", value=True)

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
        
        results = analysis.fit(
            model_terms=current_terms, enforce_hierarchy_flag=enforce_hierarchy
        )
        
        if getattr(analysis, "rename_map", {}):
            renamed = ", ".join([f"{old} → {new}" for old, new in analysis.rename_map.items()])
            st.warning(f"Factor names renamed: {renamed}")

        if 'fitted_models' not in st.session_state:
            st.session_state['fitted_models'] = {}
        st.session_state['fitted_models'][selected_response] = results
        
        # Store analysis object for stepwise regression
        st.session_state[f'analysis_{selected_response}'] = analysis
        
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        st.exception(e)
        st.stop()

# ==== STEPWISE REGRESSION (BIC-based Automatic Model Selection) ====
# Display AFTER model fitting so we have the analysis object
st.divider()
stepwise_results = display_stepwise_button(
    factors=factors,
    anova_analysis=analysis,
    key_prefix=f"stepwise_{selected_response}"
)

# If stepwise returned results, update model terms
if stepwise_results is not None:
    st.session_state['model_terms_per_response'][selected_response] = stepwise_results.final_terms
    invalidate_downstream_state(from_step=5)
    st.rerun()

st.divider()

st.subheader("📈 Analysis Results")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Model Fit", "📉 Effects & Residuals",
    "🔍 Design Diagnostics", "📈 Profiler"
])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Actual vs Predicted**")
        if not results.anova_table.empty and 'P' in results.anova_table.columns:
            model_p = results.anova_table.loc['Model', 'P'] if 'Model' in results.anova_table.index else 0.0
        else:
            model_p = 0.0
        
        fig = create_parity_plot(
            response_filtered, results.fitted_values,
            results.r_squared, results.adj_r_squared, results.rmse, model_p
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Residuals vs Fitted**")
        fig = create_residual_plot(results.fitted_values, results.residuals)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.markdown("**Effect Significance (Pareto)**")
    if not results.logworth.empty:
        p_values = {}
        for term in results.logworth.index:
            p_val = 10 ** (-results.logworth.loc[term, 'LogWorth'])
            p_values[term] = p_val
        
        fig = create_logworth_plot(results.logworth, p_values)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.markdown("**ANOVA Table**")
    if not results.anova_table.empty:
        st.dataframe(results.anova_table, use_container_width=True)
    
    # Lack-of-Fit test
    display_lack_of_fit_test(
        design_filtered, response_filtered, results, factors, current_terms
    )
with tab2:
    st.markdown("**Coefficient Table**")
    st.dataframe(results.effect_estimates, use_container_width=True)
    
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
                                lr_other = LinearRegression(fit_intercept=False)
                                lr_other.fit(X_other, response_filtered)
                                y_other = lr_other.predict(X_other)
                                y_adj = response_filtered - y_other + response_filtered.mean()
                            else:
                                y_adj = response_filtered
                        
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
                            
                            fig.update_layout(
                                xaxis_title=format_term_for_display(term),
                                yaxis_title=f"{selected_response} (adjusted)",
                                height=300, showlegend=False
                            )
                            fig = apply_plot_style(fig)
                            st.plotly_chart(fig, use_container_width=True)
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
        fig.update_layout(xaxis_title='Run Order', yaxis_title='Residuals', height=350)
        fig = apply_plot_style(fig)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Normal Q-Q Plot**")
        fig = create_qq_plot(results.residuals)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    st.markdown("**Residuals vs Factors**")
    factor_cols = st.columns(min(3, len(factors)))
    
    for idx, factor in enumerate(factors):
        with factor_cols[idx % len(factor_cols)]:
            st.markdown(f"*{factor.name}*")
            factor_vals = design_filtered[factor.name].values
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=factor_vals, y=results.residuals, mode='markers',
                marker=dict(size=8, color=PLOT_COLORS['primary'], opacity=0.7,
                           line=dict(width=0.5, color='white')),
                hovertemplate=f'{factor.name}: %{{x}}<br>Residual: %{{y:.3f}}<extra></extra>'
            ))
            fig.add_hline(y=0, line=dict(color=PLOT_COLORS['danger'], dash='dash', width=2))
            fig.update_layout(xaxis_title=factor.name, yaxis_title='Residuals', height=250)
            fig = apply_plot_style(fig)
            st.plotly_chart(fig, use_container_width=True)
    
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
            st.plotly_chart(fig, use_container_width=True)
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
    )
st.divider()

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("← Back to Import", use_container_width=True):
        st.session_state['current_step'] = 5
        st.switch_page("pages/5_import_results.py")

with col2:
    if st.session_state.get('quality_report'):
        if st.session_state['quality_report'].summary.needs_any_augmentation():
            if st.button("🔬 Augmentation", type="primary", use_container_width=True):
                st.session_state['current_step'] = 7
                st.session_state['show_augmentation'] = True
                st.switch_page("pages/7_augmentation.py")

with col3:
    if st.button("Optimize →", use_container_width=True):
        st.session_state['current_step'] = 8
        st.switch_page("pages/8_optimize.py")