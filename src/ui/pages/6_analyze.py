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
import plotly.express as px
from scipy import stats
from sklearn.linear_model import LinearRegression

from src.ui.utils.state_management import (
    initialize_session_state,
    is_step_complete,
    can_access_step,
    get_active_design,
    invalidate_downstream_state
)
from src.ui.components.quality_dashboard import display_quality_dashboard
from src.ui.components.model_builder import display_model_builder, format_term_for_display, display_stepwise_button
from src.ui.components.diagnostics_display import display_diagnostics_tab
from src.ui.components.profiler_display import display_profiler_tab
from src.core.analysis import ANOVAAnalysis, generate_model_terms
from src.core.diagnostics.summary import (
    compute_design_diagnostic_summary,
    generate_quality_report
)


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
    
    # Add Lack-of-Fit table
    st.divider()
    st.markdown("**Lack-of-Fit Test**")
    
    # Check if we have replicates (pure error)
    n_runs = len(design_filtered)
    n_params = len([t for t in current_terms if t != '1']) + 1
    df_model = n_params - 1
    df_residual = n_runs - n_params
    
    # Check for pure replicates (identical factor settings)
    factor_cols = [f.name for f in factors]
    duplicates = design_filtered[factor_cols].duplicated(keep=False)
    has_replicates = duplicates.any()
    
    if has_replicates and df_residual > 0:
        # Calculate pure error from replicates
        unique_settings = design_filtered[factor_cols].drop_duplicates()
        n_unique = len(unique_settings)
        
        ss_pure_error = 0
        df_pure_error = 0
        
        for idx, row in unique_settings.iterrows():
            # Find all runs with this setting
            mask = (design_filtered[factor_cols] == row).all(axis=1)
            replicate_responses = response_filtered[mask]
            
            if len(replicate_responses) > 1:
                # Pure error from replicates
                ss_pure_error += np.sum((replicate_responses - replicate_responses.mean())**2)
                df_pure_error += len(replicate_responses) - 1
        
        if df_pure_error > 0:
            # Calculate lack-of-fit
            ss_residual = np.sum(results.residuals**2)
            ss_lof = ss_residual - ss_pure_error
            df_lof = df_residual - df_pure_error
            
            if df_lof > 0:
                ms_lof = ss_lof / df_lof
                ms_pure_error = ss_pure_error / df_pure_error
                f_lof = ms_lof / ms_pure_error
                p_lof = 1 - stats.f.cdf(f_lof, df_lof, df_pure_error)
                
                lof_table = pd.DataFrame({
                    'Source': ['Lack-of-Fit', 'Pure Error', 'Total Error'],
                    'DF': [df_lof, df_pure_error, df_residual],
                    'SS': [ss_lof, ss_pure_error, ss_residual],
                    'MS': [ms_lof, ms_pure_error, ss_residual/df_residual],
                    'F': [f_lof, np.nan, np.nan],
                    'P': [p_lof, np.nan, np.nan]
                })
                
                st.dataframe(lof_table, use_container_width=True, hide_index=True)
                
                if p_lof < 0.05:
                    st.warning(f"⚠️ Lack-of-fit is significant (p = {p_lof:.4f}). Model may be inadequate.")
                else:
                    st.success(f"✓ No significant lack-of-fit (p = {p_lof:.4f})")
            else:
                st.info("Insufficient degrees of freedom for lack-of-fit test")
        else:
            st.info("No pure replicates available for lack-of-fit test")
    else:
        st.info("Lack-of-fit test requires replicate runs (identical factor settings)")

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
    st.subheader("📊 Prediction Profiler")
    st.caption("Interactive prediction profiler - adjust factor settings and see how the response changes.")
    
    # Get current model and results
    if selected_response not in st.session_state['fitted_models']:
        st.warning("Please fit a model first")
        st.stop()
    
    results = st.session_state['fitted_models'][selected_response]
    model_terms = st.session_state['model_terms_per_response'][selected_response]
    
    # Check if we have factors for profiling
    if not factors:
        st.info("No factors available for profiling.")
        st.stop()
    
    # === JMP-Style Prediction Profiler ===
    
    # Initialize factor settings if not in session state
    if 'profiler_settings' not in st.session_state:
        st.session_state['profiler_settings'] = {}
    
    # Set default values (center for continuous, middle option for categorical)
    factor_settings = {}
    for factor in factors:
        if factor.name not in st.session_state['profiler_settings']:
            if factor.is_continuous():
                min_val, max_val = factor.levels
                st.session_state['profiler_settings'][factor.name] = (min_val + max_val) / 2
            else:
                st.session_state['profiler_settings'][factor.name] = factor.levels[len(factor.levels) // 2]
        factor_settings[factor.name] = st.session_state['profiler_settings'][factor.name]
    
    # Compute current prediction
    try:
        pred_df = pd.DataFrame([factor_settings])
        current_prediction = results.fitted_model.predict(pred_df)[0]
    except Exception as e:
        st.error(f"Could not compute prediction: {e}")
        current_prediction = 0.0
    
    # Display current prediction prominently at top
    st.markdown(f"### Predicted {selected_response}: **{current_prediction:.4f}**")
    
    st.divider()
    
    # Create profiler plots in a grid
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
                st.markdown(f"**{factor.name}**")
                
                if factor.is_continuous():
                    # === Continuous Factor: Response Trace Plot ===
                    min_val, max_val = factor.levels
                    
                    # Generate response trace (holding other factors constant)
                    trace_points = 100
                    factor_range = np.linspace(min_val, max_val, trace_points)
                    
                    trace_predictions = []
                    trace_data = []
                    for val in factor_range:
                        point = factor_settings.copy()
                        point[factor.name] = val
                        trace_data.append(point)
                        point_df = pd.DataFrame([point])
                        pred = results.fitted_model.predict(point_df)[0]
                        trace_predictions.append(pred)
                    
                    # Calculate 95% CI of the fit
                    trace_df = pd.DataFrame(trace_data)
                    
                    # Get prediction with standard errors
                    # Note: statsmodels get_prediction() provides standard errors
                    try:
                        pred_obj = results.fitted_model.get_prediction(trace_df)
                        pred_summary = pred_obj.summary_frame(alpha=0.05)
                        ci_lower = pred_summary['mean_ci_lower'].values
                        ci_upper = pred_summary['mean_ci_upper'].values
                    except Exception:
                        # Fallback: no CI if get_prediction fails
                        ci_lower = None
                        ci_upper = None
                    
                    # Create trace plot
                    fig = go.Figure()
                    
                    # 95% CI band (if available)
                    if ci_lower is not None and ci_upper is not None:
                        fig.add_trace(go.Scatter(
                            x=np.concatenate([factor_range, factor_range[::-1]]),
                            y=np.concatenate([ci_upper, ci_lower[::-1]]),
                            fill='toself',
                            fillcolor='rgba(128, 128, 128, 0.2)',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                    
                    # Response trace line
                    fig.add_trace(go.Scatter(
                        x=factor_range,
                        y=trace_predictions,
                        mode='lines',
                        line=dict(color=PLOT_COLORS['primary'], width=2),
                        hovertemplate=f"{factor.name}: %{{x:.3f}}<br>{selected_response}: %{{y:.3f}}<extra></extra>",
                        showlegend=False
                    ))
                    
                    # Current setting - vertical line
                    current_val = factor_settings[factor.name]
                    fig.add_vline(
                        x=current_val,
                        line=dict(color='red', dash='dash', width=2),
                        annotation=dict(
                            text=f"{current_val:.2f}",
                            yref='paper',
                            y=1.05,
                            showarrow=False,
                            font=dict(size=10, color='red')
                        )
                    )
                    
                    # Current prediction point
                    fig.add_trace(go.Scatter(
                        x=[current_val],
                        y=[current_prediction],
                        mode='markers',
                        marker=dict(size=10, color='red', symbol='circle'),
                        showlegend=False,
                        hovertemplate=f"{factor.name}: {current_val:.3f}<br>{selected_response}: {current_prediction:.3f}<extra></extra>"
                    ))
                    
                    fig.update_layout(
                        height=200,
                        margin=dict(l=40, r=10, t=20, b=40),
                        xaxis_title=None,
                        yaxis_title=selected_response if col_idx == 0 else None,
                        showlegend=False
                    )
                    
                    fig.update_xaxes(range=[min_val - 0.05*(max_val-min_val), 
                                           max_val + 0.05*(max_val-min_val)])
                    
                    fig = apply_plot_style(fig)
                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{factor.name}")
                    
                    # Slider below plot
                    new_val = st.slider(
                        f"{factor.name} setting",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(current_val),
                        format="%.3f",
                        key=f"profiler_slider_{factor.name}",
                        label_visibility="collapsed"
                    )
                    
                    # Update session state if changed
                    if new_val != st.session_state['profiler_settings'][factor.name]:
                        st.session_state['profiler_settings'][factor.name] = new_val
                        st.rerun()
                
                else:
                    # === Categorical/Discrete Factor: Bar Chart ===
                    
                    # Generate predictions for each level
                    level_predictions = []
                    for level in factor.levels:
                        point = factor_settings.copy()
                        point[factor.name] = level
                        point_df = pd.DataFrame([point])
                        pred = results.fitted_model.predict(point_df)[0]
                        level_predictions.append(pred)
                    
                    # Create bar chart
                    fig = go.Figure()
                    
                    # Determine which bar is current
                    current_level = factor_settings[factor.name]
                    current_idx = factor.levels.index(current_level)
                    
                    # Color bars (current one red, others blue)
                    colors = [PLOT_COLORS['danger'] if i == current_idx else PLOT_COLORS['primary'] 
                             for i in range(len(factor.levels))]
                    
                    fig.add_trace(go.Bar(
                        x=[str(level) for level in factor.levels],
                        y=level_predictions,
                        marker=dict(color=colors, line=dict(color='#000000', width=1)),
                        hovertemplate=f"{factor.name}: %{{x}}<br>{selected_response}: %{{y:.3f}}<extra></extra>",
                        showlegend=False
                    ))
                    
                    fig.update_layout(
                        height=200,
                        margin=dict(l=40, r=10, t=20, b=40),
                        xaxis_title=None,
                        yaxis_title=selected_response if col_idx == 0 else None,
                        showlegend=False
                    )
                    
                    fig = apply_plot_style(fig)
                    st.plotly_chart(fig, use_container_width=True, key=f"plot_{factor.name}")
                    
                    # Selectbox below plot
                    new_val = st.selectbox(
                        f"{factor.name} setting",
                        options=factor.levels,
                        index=current_idx,
                        key=f"profiler_select_{factor.name}",
                        label_visibility="collapsed"
                    )
                    
                    # Update session state if changed
                    if new_val != st.session_state['profiler_settings'][factor.name]:
                        st.session_state['profiler_settings'][factor.name] = new_val
                        st.rerun()
    
    st.divider()
    
    # Define continuous factors for contour/3D plots
    continuous_factors = [f for f in factors if f.is_continuous()]
    
    # === Contour Plots ===
    if len(continuous_factors) >= 2:
        st.markdown("### 🗺️ Contour Plots")
        st.caption("2D contour maps showing response surface for pairs of factors.")
        
        # Initialize contour settings if not in session state
        if 'contour_settings' not in st.session_state:
            st.session_state['contour_settings'] = {}
            for factor in factors:
                if factor.is_continuous():
                    min_val, max_val = factor.levels
                    st.session_state['contour_settings'][factor.name] = (min_val + max_val) / 2
                else:
                    st.session_state['contour_settings'][factor.name] = factor.levels[len(factor.levels) // 2]
        
        # Factor pair selector
        col1, col2 = st.columns(2)
        
        with col1:
            x_factor = st.selectbox(
                "X-axis factor",
                options=[f.name for f in continuous_factors],
                index=0,
                key="contour_x"
            )
        
        with col2:
            y_factor = st.selectbox(
                "Y-axis factor",
                options=[f.name for f in continuous_factors if f.name != x_factor],
                index=0,
                key="contour_y"
            )
        
        # Sliders for other factors (held constant)
        other_factors = [f for f in factors if f.name not in [x_factor, y_factor]]
        
        if other_factors:
            st.markdown("**Hold constant at:**")
            
            # Create sliders/selects for other factors
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
                            current_val = st.session_state['contour_settings'][factor.name]
                            
                            new_val = st.slider(
                                factor.name,
                                min_value=float(min_val),
                                max_value=float(max_val),
                                value=float(current_val),
                                format="%.3f",
                                key=f"contour_slider_{factor.name}"
                            )
                            
                            if new_val != st.session_state['contour_settings'][factor.name]:
                                st.session_state['contour_settings'][factor.name] = new_val
                                st.rerun()
                        else:
                            current_val = st.session_state['contour_settings'][factor.name]
                            current_idx = factor.levels.index(current_val)
                            
                            new_val = st.selectbox(
                                factor.name,
                                options=factor.levels,
                                index=current_idx,
                                key=f"contour_select_{factor.name}"
                            )
                            
                            if new_val != st.session_state['contour_settings'][factor.name]:
                                st.session_state['contour_settings'][factor.name] = new_val
                                st.rerun()
        
        # Store mesh data at outer scope for both contour and 3D plots
        Z_mesh = None
        x_grid = None
        y_grid = None
        
        if x_factor and y_factor:
            try:
                # Get factor objects
                x_factor_obj = next(f for f in factors if f.name == x_factor)
                y_factor_obj = next(f for f in factors if f.name == y_factor)
                
                # Create grid
                x_min, x_max = x_factor_obj.levels
                y_min, y_max = y_factor_obj.levels
                
                x_grid = np.linspace(x_min, x_max, 50)
                y_grid = np.linspace(y_min, y_max, 50)
                X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
                
                # Prepare prediction grid
                # Hold other factors at contour settings
                grid_points = []
                for i in range(len(x_grid)):
                    for j in range(len(y_grid)):
                        point = st.session_state['contour_settings'].copy()
                        point[x_factor] = X_mesh[j, i]
                        point[y_factor] = Y_mesh[j, i]
                        grid_points.append(point)
                
                grid_df = pd.DataFrame(grid_points)
                
                # Predict on grid
                Z_pred = results.fitted_model.predict(grid_df)
                # Convert Series to numpy array before reshaping
                Z_mesh = np.array(Z_pred).reshape(X_mesh.shape)
                
                # Create contour plot
                fig = go.Figure()
                
                # Add contour
                fig.add_trace(go.Contour(
                    x=x_grid,
                    y=y_grid,
                    z=Z_mesh,
                    colorscale='RdYlGn',
                    colorbar=dict(title=selected_response),
                    contours=dict(
                        coloring='heatmap',
                        showlabels=True,
                        labelfont=dict(size=10, color='white')
                    ),
                    hovertemplate=(
                        f"{x_factor}: %{{x:.2f}}<br>"
                        f"{y_factor}: %{{y:.2f}}<br>"
                        f"{selected_response}: %{{z:.2f}}<extra></extra>"
                    )
                ))
                
                fig.update_layout(
                    xaxis_title=x_factor,
                    yaxis_title=y_factor,
                    height=500,
                    showlegend=True
                )
                
                fig = apply_plot_style(fig)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Could not create contour plot: {e}")
                st.exception(e)
    
    else:
        st.info("Contour plots require at least 2 continuous factors.")
    
    st.divider()
    
    # === 3D Surface Plot ===
    if len(continuous_factors) >= 2:
        st.markdown("### 🏔️ 3D Response Surface")
        st.caption("Interactive 3D visualization of the response surface.")
        
        # Use same factors as contour plot
        if x_factor and y_factor and Z_mesh is not None:
            try:
                # Reuse the mesh from contour plot
                fig_3d = go.Figure()
                
                fig_3d.add_trace(go.Surface(
                    x=x_grid,
                    y=y_grid,
                    z=Z_mesh,
                    colorscale='RdYlGn',
                    colorbar=dict(title=selected_response),
                    hovertemplate=(
                        f"{x_factor}: %{{x:.2f}}<br>"
                        f"{y_factor}: %{{y:.2f}}<br>"
                        f"{selected_response}: %{{z:.2f}}<extra></extra>"
                    )
                ))
                
                fig_3d.update_layout(
                    scene=dict(
                        xaxis_title=x_factor,
                        yaxis_title=y_factor,
                        zaxis_title=selected_response,
                        camera=dict(
                            eye=dict(x=1.5, y=1.5, z=1.3)
                        )
                    ),
                    height=600
                )
                
                fig_3d = apply_plot_style(fig_3d)
                st.plotly_chart(fig_3d, use_container_width=True)
                
            except Exception as e:
                st.error(f"Could not create 3D surface plot: {e}")
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