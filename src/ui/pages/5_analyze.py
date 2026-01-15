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
from src.ui.components.model_builder import display_model_builder, format_term_for_display
from src.core.analysis import ANOVAAnalysis, generate_model_terms
from src.core.diagnostics.summary import (
    compute_design_diagnostic_summary,
    generate_quality_report
)

# ==================== COHESIVE PLOT STYLING ====================

PLOT_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'neutral': '#7f7f7f',
    'purple': '#9467bd',
    'brown': '#8c564b',
    'pink': '#e377c2',
    'sigma1': '#90EE90',  # Light green for 1σ
    'sigma2': '#FFD700',  # Gold for 2σ
    'sigma3': '#FF6347'   # Tomato red for 3σ
}

def apply_plot_style(fig):
    """Apply consistent ACS-style formatting to plotly figures."""
    # Update layout (background, font, margins)
    fig.update_layout(
        plot_bgcolor='#f0f0f0',  # Light gray background to match dark theme
        paper_bgcolor='#f0f0f0',  # Light gray paper to match dark theme
        font=dict(family='Arial, sans-serif', size=11, color='#000000'),  # Full black text
        margin=dict(l=50, r=30, t=30, b=50, pad=5)  # Tighter margins (~5%)
    )
    
    # Update axes separately to preserve titles
    fig.update_xaxes(
        showgrid=True, gridwidth=0.5, gridcolor='#d0d0d0',
        linecolor='#000000', linewidth=1.5, mirror=True,
        ticks='outside', tickwidth=1, tickcolor='#000000', showline=True,
        tickfont=dict(color='#000000'),  # Ensure tick labels are black
        title_font=dict(color='#000000')  # Ensure axis titles are black
    )
    
    fig.update_yaxes(
        showgrid=True, gridwidth=0.5, gridcolor='#d0d0d0',
        linecolor='#000000', linewidth=1.5, mirror=True,
        ticks='outside', tickwidth=1, tickcolor='#000000', showline=True,
        tickfont=dict(color='#000000'),  # Ensure tick labels are black
        title_font=dict(color='#000000')  # Ensure axis titles are black
    )
    
    return fig

def create_parity_plot(actual, predicted, r_squared, adj_r_squared, rmse, p_value):
    """Create actual vs predicted parity plot with 95% CI of the fit."""
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
    se_fit = np.sqrt(mse * (1/n + (x_line - mean_actual)**2 / np.sum((actual - mean_actual)**2)))
    t_crit = stats.t.ppf(0.975, n-2)  # 95% CI
    ci_width = t_crit * se_fit
    
    fig = go.Figure()
    
    # 95% CI band (around 1:1 line)
    y_upper = x_line + ci_width
    y_lower = x_line - ci_width
    
    # Add shaded CI region
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_line, x_line[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself', fillcolor='rgba(128, 128, 128, 0.25)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    
    hover_text = [f"Run {i+1}<br>Actual: {a:.3f}<br>Predicted: {p:.3f}" 
                  for i, (a, p) in enumerate(zip(actual, predicted))]
    
    fig.add_trace(go.Scatter(
        x=actual, y=predicted, mode='markers',
        marker=dict(size=8, color=PLOT_COLORS['primary'], opacity=0.7,
                   line=dict(width=0.5, color='white')),
        name='Data', text=hover_text, hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=[plot_min, plot_max], y=[plot_min, plot_max], mode='lines',
        line=dict(color=PLOT_COLORS['danger'], dash='dash', width=2),
        name='1:1 Line', hoverinfo='skip', showlegend=False
    ))
    
    stats_text = (f"R² = {r_squared:.4f}<br>Adj R² = {adj_r_squared:.4f}<br>"
                  f"RMSE = {rmse:.4f}<br>p = {p_value:.4e}")
    
    fig.add_annotation(
        xref='paper', yref='paper', x=0.05, y=0.95, text=stats_text,
        showarrow=False, font=dict(size=10, color='#000000'), 
        bgcolor='rgba(255, 255, 255, 0.95)',  # White box with high opacity for readability
        bordercolor='#000000', borderwidth=1, align='left'
    )
    
    fig.update_layout(
        xaxis_title='Actual', yaxis_title='Predicted', height=400,
        showlegend=False
    )
    
    fig.update_xaxes(scaleanchor='y', scaleratio=1, range=[plot_min, plot_max])
    fig.update_yaxes(scaleanchor='x', scaleratio=1, range=[plot_min, plot_max])
    
    return apply_plot_style(fig)

def create_residual_plot(fitted, residuals):
    """Create studentized residuals vs fitted with color-coded thresholds."""
    # Calculate studentized residuals
    std_resid = np.std(residuals)
    studentized = residuals / std_resid
    
    fig = go.Figure()
    
    x_range = [fitted.min(), fitted.max()]
    
    # Zero line
    fig.add_trace(go.Scatter(
        x=x_range, y=[0, 0], mode='lines',
        line=dict(color='#000000', dash='solid', width=1.5),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Sigma reference lines with increased opacity
    for sigma, color in [(1, PLOT_COLORS['sigma1']), 
                          (2, PLOT_COLORS['sigma2']), 
                          (3, PLOT_COLORS['sigma3'])]:
        for sign in [1, -1]:
            y_val = sign * sigma
            # Convert hex color to rgba with opacity
            if color == PLOT_COLORS['sigma1']:
                rgba_color = 'rgba(144, 238, 144, 0.8)'  # Light green
            elif color == PLOT_COLORS['sigma2']:
                rgba_color = 'rgba(255, 215, 0, 0.8)'    # Gold
            else:
                rgba_color = 'rgba(255, 99, 71, 0.8)'   # Tomato red
            
            fig.add_trace(go.Scatter(
                x=x_range, y=[y_val, y_val], mode='lines',
                line=dict(color=rgba_color, dash='dash', width=2),
                showlegend=False, hoverinfo='skip'
            ))
    
    # All data points same color (no color coding)
    hover_text = [f"Run {i+1}<br>Fitted: {f:.3f}<br>Studentized: {s:.3f}" 
                  for i, (f, s) in enumerate(zip(fitted, studentized))]
    
    fig.add_trace(go.Scatter(
        x=fitted, y=studentized, mode='markers',
        marker=dict(size=8, color=PLOT_COLORS['primary'], opacity=0.7,
                   line=dict(width=0.5, color='white')),
        name='Residuals', text=hover_text, hovertemplate='%{text}<extra></extra>',
        showlegend=False
    ))
    
    fig.update_layout(
        xaxis_title='Fitted Values', yaxis_title='Studentized Residuals',
        height=400, showlegend=False
    )
    
    y_max = max(abs(studentized.min()), abs(studentized.max()))
    y_max = max(y_max, 3.5) * 1.1  # At least show ±3σ range
    fig.update_yaxes(range=[-y_max, y_max])
    
    x_range_val = fitted.max() - fitted.min()
    fig.update_xaxes(range=[fitted.min() - 0.1*x_range_val, 
                           fitted.max() + 0.1*x_range_val])
    
    return apply_plot_style(fig)

def create_logworth_plot(logworth_df, p_values):
    """Create LogWorth bar plot sorted Pareto-style with p-values on bars."""
    logworth_sorted = logworth_df.sort_values('LogWorth', ascending=True)
    p_values_sorted = [p_values[term] for term in logworth_sorted.index]
    
    p_text = [f"p={p:.4f}" if p >= 0.0001 else f"p={p:.2e}" 
              for p in p_values_sorted]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=logworth_sorted['LogWorth'], y=logworth_sorted.index, orientation='h',
        marker=dict(color=PLOT_COLORS['primary'], 
                   line=dict(color='#000000', width=0.5)),
        text=p_text, textposition='outside', textfont=dict(size=10),
        hovertemplate='%{y}<br>LogWorth: %{x:.2f}<br>%{text}<extra></extra>'
    ))
    
    threshold = -np.log10(0.05)
    fig.add_vline(
        x=threshold, line=dict(color=PLOT_COLORS['danger'], dash='dash', width=2),
        annotation=dict(text='α=0.05', textangle=0, yref='paper', y=0.95, font=dict(size=10))
    )
    
    fig.update_layout(
        xaxis_title='LogWorth (-log₁₀(p))', yaxis_title='',
        height=max(250, len(logworth_sorted) * 25),
        showlegend=False, margin=dict(l=150, r=100)
    )
    
    return apply_plot_style(fig)

def create_qq_plot(residuals):
    """Create Q-Q normal probability plot."""
    # Properly calculate theoretical quantiles using plotting positions
    n = len(residuals)
    # Use plotting position: (i - 0.5) / n
    probabilities = (np.arange(1, n+1) - 0.5) / n
    theoretical = stats.norm.ppf(probabilities)
    sample = np.sort(residuals)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=theoretical, y=sample, mode='markers',
        marker=dict(size=8, color=PLOT_COLORS['primary'], opacity=0.7,
                   line=dict(width=0.5, color='white')),
        name='Data', hovertemplate='Theoretical: %{x:.3f}<br>Sample: %{y:.3f}<extra></extra>'
    ))
    
    min_val = min(theoretical.min(), sample.min())
    max_val = max(theoretical.max(), sample.max())
    
    fig.add_trace(go.Scatter(
        x=[min_val, max_val], y=[min_val, max_val], mode='lines',
        line=dict(color=PLOT_COLORS['danger'], dash='dash', width=2),
        showlegend=False, hoverinfo='skip'
    ))
    
    fig.update_layout(
        xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles',
        height=350, showlegend=False
    )
    
    return apply_plot_style(fig)

def create_half_normal_plot(effects, effect_names):
    """Create half-normal probability plot for effects."""
    abs_effects = np.abs(effects)
    sorted_indices = np.argsort(abs_effects)
    sorted_effects = abs_effects[sorted_indices]
    sorted_names = [effect_names[i] for i in sorted_indices]
    
    n = len(sorted_effects)
    quantiles = stats.norm.ppf((np.arange(1, n+1) - 0.5) / n)
    half_normal_quantiles = np.abs(quantiles)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=half_normal_quantiles, y=sorted_effects, mode='markers+text',
        marker=dict(size=8, color=PLOT_COLORS['primary'], opacity=0.7,
                   line=dict(width=0.5, color='white')),
        text=sorted_names, textposition='top center', textfont=dict(size=9),
        hovertemplate='%{text}<br>|Effect|: %{y:.3f}<extra></extra>'
    ))
    
    if len(sorted_effects) > 2:
        n_baseline = max(3, len(sorted_effects) // 3)
        lr = LinearRegression()
        lr.fit(half_normal_quantiles[:n_baseline].reshape(-1, 1), 
               sorted_effects[:n_baseline])
        
        x_line = np.array([0, half_normal_quantiles.max()])
        y_line = lr.predict(x_line.reshape(-1, 1))
        
        fig.add_trace(go.Scatter(
            x=x_line, y=y_line, mode='lines',
            line=dict(color=PLOT_COLORS['danger'], dash='dash', width=2),
            showlegend=False, hoverinfo='skip'
        ))
    
    fig.update_layout(
        xaxis_title='Half-Normal Quantiles', yaxis_title='|Effect|',
        height=400, showlegend=False
    )
    
    return apply_plot_style(fig)

# ==================== MAIN APP ====================

initialize_session_state()

if not can_access_step(5):
    st.warning("⚠️ Please complete Steps 1-4 first")
    st.stop()

st.title("Step 5: Analyze Experimental Results")

design = get_active_design()
factors = st.session_state['factors']
responses = st.session_state['responses']
response_names = st.session_state['response_names']

if not responses:
    st.error("No response data available. Please import data in Step 4.")
    st.stop()

st.subheader("📊 Response Selection")

selected_response = st.selectbox(
    "Select Response to Analyze", response_names, key='analysis_response_selector'
)

st.divider()

if 'model_terms_per_response' not in st.session_state:
    st.session_state['model_terms_per_response'] = {}

if selected_response not in st.session_state['model_terms_per_response']:
    default_terms = generate_model_terms(factors, 'linear', include_intercept=True)
    st.session_state['model_terms_per_response'][selected_response] = default_terms

current_terms = st.session_state['model_terms_per_response'][selected_response]

updated_terms = display_model_builder(
    factors=factors, current_terms=current_terms, response_name=selected_response,
    key_prefix=f"model_builder_{selected_response}"
)

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
        
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        st.exception(e)
        st.stop()

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
    st.subheader("📋 Design Diagnostics")
    
    # Get diagnostics from session state if available
    if st.session_state.get('diagnostics_summary') and st.session_state.get('quality_report'):
        summary = st.session_state['diagnostics_summary']
        report = st.session_state['quality_report']
        
        # Get diagnostics for current response
        if selected_response in summary.response_diagnostics:
            diag = summary.response_diagnostics[selected_response]
            
            # Metrics overview
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("R²", f"{diag.r_squared:.3f}")
            with col2:
                st.metric("Adj R²", f"{diag.adj_r_squared:.3f}")
            with col3:
                st.metric("RMSE", f"{diag.rmse:.2f}")
            with col4:
                grade = report.response_quality[selected_response].overall_grade
                grade_icon = {'Excellent': '🌟', 'Good': '✅', 'Fair': '⚡', 
                             'Poor': '⚠️', 'Inadequate': '❌'}.get(grade, '❓')
                st.metric("Grade", f"{grade_icon} {grade}")
            
            st.divider()
            
            # Variance Inflation Factors
            st.markdown("### Variance Inflation Factors (VIF)")
            st.caption("VIF measures multicollinearity. VIF > 10 indicates problematic collinearity.")
            
            if diag.vif_values:
                vif_data = []
                for term, vif in sorted(diag.vif_values.items(), 
                                       key=lambda x: x[1] if not pd.isna(x[1]) and not np.isinf(x[1]) else -1, 
                                       reverse=True):
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
                    
                    vif_data.append({
                        'Term': format_term_for_display(term),
                        'VIF': vif_str,
                        'Status': status
                    })
                
                if vif_data:
                    # Display as standard dataframe
                    vif_df = pd.DataFrame(vif_data)
                    st.dataframe(
                        vif_df,
                        use_container_width=True,
                        hide_index=True
                    )
                    
                    # Add interpretation
                    high_vif = [row['Term'] for row in vif_data if '❌' in row['Status'] or '⚠️' in row['Status']]
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
            
            st.divider()
            
            # Alias Structure (for fractional designs)
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
            
            st.divider()
            
            # Prediction Variance
            if diag.prediction_variance_stats:
                st.markdown("### Prediction Variance")
                st.caption("Lower variance = more precise predictions. High max/mean ratio indicates non-uniform precision.")
                
                stats_dict = diag.prediction_variance_stats
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Min", f"{stats_dict.get('min', 0):.3f}")
                with col2:
                    st.metric("Mean", f"{stats_dict.get('mean', 0):.3f}")
                with col3:
                    st.metric("Max", f"{stats_dict.get('max', 0):.3f}")
                with col4:
                    max_val = stats_dict.get('max', 0)
                    mean_val = stats_dict.get('mean', 1)
                    ratio = max_val / mean_val if mean_val > 0 else 0
                    st.metric("Max/Mean", f"{ratio:.2f}")
                
                if ratio > 3:
                    st.warning(
                        f"⚠️ High variance ratio ({ratio:.1f}x) indicates non-uniform precision. "
                        "Some regions of the design space have much higher prediction variance."
                    )
                else:
                    st.success("✅ Prediction variance is reasonably uniform")
            
            st.divider()
            
            # Effect Significance Summary
            st.markdown("### Effect Significance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if diag.significant_effects:
                    st.success(f"**Significant (p < 0.05):** {len(diag.significant_effects)}")
                    for effect in diag.significant_effects:
                        st.markdown(f"- {format_term_for_display(effect)}")
                else:
                    st.info("No significant effects detected")
            
            with col2:
                if diag.marginally_significant:
                    st.warning(f"**Marginal (0.05 < p < 0.10):** {len(diag.marginally_significant)}")
                    for effect in diag.marginally_significant:
                        st.markdown(f"- {format_term_for_display(effect)}")
                else:
                    st.info("No marginally significant effects")
            
            st.divider()
            
            # Issues Summary
            if diag.issues:
                st.markdown("### Issues Detected")
                
                for issue in diag.issues:
                    if issue.severity == 'critical':
                        st.error(f"**✗ {issue.description}**")
                        st.markdown(f"*Recommendation:* {issue.recommended_action}")
                        if issue.affected_terms:
                            st.markdown(f"*Affected terms:* {', '.join(issue.affected_terms)}")
                    elif issue.severity == 'warning':
                        st.warning(f"**⚡ {issue.description}**")
                        st.markdown(f"*Recommendation:* {issue.recommended_action}")
                        if issue.affected_terms:
                            st.markdown(f"*Affected terms:* {', '.join(issue.affected_terms)}")
                    else:
                        st.info(f"ℹ️ {issue.description}")
                        st.markdown(f"*Recommendation:* {issue.recommended_action}")
            else:
                st.success("✅ No critical issues detected for this response")
            
            # High Leverage Points
            if diag.high_leverage_points:
                st.divider()
                st.markdown("### High Leverage Points")
                st.caption(f"Found {len(diag.high_leverage_points)} observation(s) with high leverage")
                
                st.warning(
                    f"**Runs with high leverage:** {', '.join(map(str, [p+1 for p in diag.high_leverage_points]))}\n\n"
                    "High leverage points have unusual factor combinations and can disproportionately "
                    "influence the model. Verify these runs for accuracy."
                )
            
        else:
            st.warning(f"No diagnostics available for {selected_response}")
    
    else:
        # Compute diagnostics on demand
        st.info("💡 Click to generate comprehensive design diagnostics including VIF, alias structure, prediction variance, and quality assessment.")
        
        if st.button("Generate Diagnostics", type="primary"):
            with st.spinner("Computing diagnostics..."):
                try:
                    # Prepare metadata
                    design = get_active_design()
                    design_metadata = {
                        'design_type': st.session_state.get('design_type', 'unknown'),
                        'generators': st.session_state.get('design_metadata', {}).get('generators'),
                        'is_split_plot': st.session_state.get('design_metadata', {}).get('is_split_plot', False),
                        'has_blocking': 'Block' in design.columns,
                        'has_center_points': st.session_state.get('design_metadata', {}).get('has_center_points', False)
                    }
                    
                    # Compute diagnostics
                    summary = compute_design_diagnostic_summary(
                        design=design,
                        responses=st.session_state['responses'],
                        fitted_models={k: v.fitted_model for k, v in st.session_state['fitted_models'].items()},
                        factors=factors,
                        model_terms_per_response=st.session_state['model_terms_per_response'],
                        design_metadata=design_metadata
                    )
                    
                    # Generate quality report
                    report = generate_quality_report(summary)
                    
                    # Save to session state
                    st.session_state['diagnostics_summary'] = summary
                    st.session_state['quality_report'] = report
                    
                    st.success("✅ Diagnostics computed successfully!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Failed to compute diagnostics: {e}")
                    st.exception(e)

with tab4:
    st.subheader("📊 Response Profiler")
    
    # Get current model and results
    if selected_response not in st.session_state['fitted_models']:
        st.warning("Please fit a model first")
        st.stop()
    
    results = st.session_state['fitted_models'][selected_response]
    model_terms = st.session_state['model_terms_per_response'][selected_response]
    
    # Check if we have continuous factors for profiling
    continuous_factors = [f for f in factors if f.is_continuous()]
    
    if not continuous_factors:
        st.info(
            "Response profiling requires at least one continuous factor. "
            "Your design contains only categorical/discrete factors."
        )
        st.stop()
    
    # === Prediction Profiler ===
    st.markdown("### 🎚️ Prediction Profiler")
    st.caption("Use sliders to explore how factor settings affect the predicted response.")
    
    # Create sliders for each factor
    factor_settings = {}
    
    slider_cols = st.columns(min(3, len(factors)))
    
    for idx, factor in enumerate(factors):
        with slider_cols[idx % len(slider_cols)]:
            if factor.is_continuous():
                # Continuous factor - slider
                min_val, max_val = factor.levels
                center = (min_val + max_val) / 2
                
                factor_settings[factor.name] = st.slider(
                    factor.name,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(center),
                    format="%.2f",
                    key=f"profiler_{factor.name}"
                )
            else:
                # Categorical/discrete - selectbox
                factor_settings[factor.name] = st.selectbox(
                    factor.name,
                    options=factor.levels,
                    index=len(factor.levels) // 2,
                    key=f"profiler_{factor.name}"
                )
    
    # Build prediction from current settings
    try:
        # Create a single-row dataframe with current settings
        pred_df = pd.DataFrame([factor_settings])
        
        # Make prediction using fitted model
        prediction = results.fitted_model.predict(pred_df)[0]
        
        # Display prediction
        st.metric(
            f"Predicted {selected_response}",
            f"{prediction:.3f}",
            help="Prediction at current factor settings"
        )
        
    except Exception as e:
        st.error(f"Could not compute prediction: {e}")
    
    st.divider()
    
    # === Contour Plots ===
    if len(continuous_factors) >= 2:
        st.markdown("### 🗺️ Contour Plots")
        st.caption("2D contour maps showing response surface for pairs of factors.")
        
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
                # Hold other factors at current settings
                grid_points = []
                for i in range(len(x_grid)):
                    for j in range(len(y_grid)):
                        point = factor_settings.copy()
                        point[x_factor] = X_mesh[j, i]
                        point[y_factor] = Y_mesh[j, i]
                        grid_points.append(point)
                
                grid_df = pd.DataFrame(grid_points)
                
                # Predict on grid
                Z_pred = results.fitted_model.predict(grid_df)
                Z_mesh = Z_pred.reshape(X_mesh.shape)
                
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
                
                # Add current point
                fig.add_trace(go.Scatter(
                    x=[factor_settings[x_factor]],
                    y=[factor_settings[y_factor]],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='x', line=dict(width=2, color='white')),
                    name='Current Setting',
                    hovertemplate=f"{x_factor}: %{{x:.2f}}<br>{y_factor}: %{{y:.2f}}<extra></extra>"
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
        if x_factor and y_factor:
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
    
    # === Individual Factor Response Curves ===
    st.markdown("### 📈 Individual Factor Effects")
    st.caption("Response vs each factor (holding others at current settings).")
    
    # Show response curves for continuous factors
    factor_cols = st.columns(min(3, len(continuous_factors)))
    
    for idx, factor in enumerate(continuous_factors):
        with factor_cols[idx % len(factor_cols)]:
            st.markdown(f"**{factor.name}**")
            
            try:
                # Create range for this factor
                f_min, f_max = factor.levels
                f_range = np.linspace(f_min, f_max, 100)
                
                # Create prediction points
                points = []
                for val in f_range:
                    point = factor_settings.copy()
                    point[factor.name] = val
                    points.append(point)
                
                points_df = pd.DataFrame(points)
                predictions = results.fitted_model.predict(points_df)
                
                # Create line plot
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=f_range,
                    y=predictions,
                    mode='lines',
                    line=dict(color=PLOT_COLORS['primary'], width=3),
                    hovertemplate=f"{factor.name}: %{{x:.2f}}<br>{selected_response}: %{{y:.2f}}<extra></extra>"
                ))
                
                # Mark current setting
                current_val = factor_settings[factor.name]
                current_pred = results.fitted_model.predict(pd.DataFrame([factor_settings]))[0]
                
                fig.add_trace(go.Scatter(
                    x=[current_val],
                    y=[current_pred],
                    mode='markers',
                    marker=dict(size=12, color='red', symbol='diamond'),
                    name='Current',
                    hovertemplate=f"{factor.name}: {current_val:.2f}<br>{selected_response}: {current_pred:.2f}<extra></extra>"
                ))
                
                fig.update_layout(
                    xaxis_title=factor.name,
                    yaxis_title=selected_response,
                    height=300,
                    showlegend=False
                )
                
                fig = apply_plot_style(fig)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error plotting {factor.name}: {e}")

st.divider()

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("← Back to Import", use_container_width=True):
        st.session_state['current_step'] = 4
        st.switch_page("pages/4_import_results.py")

with col2:
    if st.session_state.get('quality_report'):
        if st.session_state['quality_report'].summary.needs_any_augmentation():
            if st.button("🔬 Augmentation", type="primary", use_container_width=True):
                st.session_state['current_step'] = 6
                st.session_state['show_augmentation'] = True
                st.switch_page("pages/6_augmentation.py")

with col3:
    if st.button("Optimize →", use_container_width=True):
        st.session_state['current_step'] = 7
        st.switch_page("pages/7_optimize.py")
