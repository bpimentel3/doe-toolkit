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
    'pink': '#e377c2'
}

def apply_plot_style(fig):
    """Apply consistent ACS-style formatting to plotly figures."""
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial, sans-serif', size=11, color='#2d2d2d'),
        xaxis=dict(
            showgrid=True, gridwidth=0.5, gridcolor='#e0e0e0',
            linecolor='#000000', linewidth=1.5, mirror=True,
            ticks='outside', tickwidth=1, tickcolor='#000000', showline=True
        ),
        yaxis=dict(
            showgrid=True, gridwidth=0.5, gridcolor='#e0e0e0',
            linecolor='#000000', linewidth=1.5, mirror=True,
            ticks='outside', tickwidth=1, tickcolor='#000000', showline=True
        )
    )
    return fig

def create_parity_plot(actual, predicted, r_squared, adj_r_squared, rmse, p_value):
    """Create actual vs predicted parity plot with 95% CI."""
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    margin = (max_val - min_val) * 0.1
    plot_min = min_val - margin
    plot_max = max_val + margin
    
    residuals = actual - predicted
    std_resid = np.std(residuals)
    pred_interval = 1.96 * std_resid
    
    fig = go.Figure()
    
    # 95% CI shaded region
    x_fill = np.array([plot_min, plot_max, plot_max, plot_min])
    y_upper = x_fill + pred_interval
    y_lower = x_fill - pred_interval
    
    fig.add_trace(go.Scatter(
        x=np.concatenate([x_fill, x_fill[::-1]]),
        y=np.concatenate([y_upper, y_lower[::-1]]),
        fill='toself', fillcolor='rgba(128, 128, 128, 0.2)',
        line=dict(width=0), showlegend=True, name='95% PI', hoverinfo='skip'
    ))
    
    hover_text = [f"Run {i+1}<br>Actual: {a:.3f}<br>Predicted: {p:.3f}" 
                  for i, (a, p) in enumerate(zip(actual, predicted))]
    
    fig.add_trace(go.Scatter(
        x=actual, y=predicted, mode='markers',
        marker=dict(size=8, color=PLOT_COLORS['primary'], opacity=0.7,
                   line=dict(width=0.5, color='white')),
        name='Data', text=hover_text, hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=[plot_min, plot_max], y=[plot_min, plot_max], mode='lines',
        line=dict(color=PLOT_COLORS['danger'], dash='dash', width=2),
        name='1:1 Line', hoverinfo='skip'
    ))
    
    stats_text = (f"R² = {r_squared:.4f}<br>Adj R² = {adj_r_squared:.4f}<br>"
                  f"RMSE = {rmse:.4f}<br>p = {p_value:.4e}")
    
    fig.add_annotation(
        xref='paper', yref='paper', x=0.05, y=0.95, text=stats_text,
        showarrow=False, font=dict(size=10), bgcolor='rgba(255, 255, 255, 0.9)',
        bordercolor='#000000', borderwidth=1, align='left'
    )
    
    fig.update_layout(
        xaxis_title='Actual', yaxis_title='Predicted', height=400,
        showlegend=True, legend=dict(x=0.02, y=0.02, bgcolor='rgba(255,255,255,0.8)')
    )
    
    fig.update_xaxes(scaleanchor='y', scaleratio=1, range=[plot_min, plot_max])
    fig.update_yaxes(scaleanchor='x', scaleratio=1, range=[plot_min, plot_max])
    
    return apply_plot_style(fig)

def create_residual_plot(fitted, residuals):
    """Create residuals vs fitted with studentized reference lines."""
    std_resid = np.std(residuals)
    fig = go.Figure()
    
    x_range = [fitted.min(), fitted.max()]
    
    fig.add_trace(go.Scatter(
        x=x_range, y=[0, 0], mode='lines',
        line=dict(color=PLOT_COLORS['danger'], dash='dash', width=2),
        showlegend=False, hoverinfo='skip'
    ))
    
    for n_std in [1, 2, 3]:
        for sign in [1, -1]:
            y_val = sign * n_std * std_resid
            fig.add_trace(go.Scatter(
                x=x_range, y=[y_val, y_val], mode='lines',
                line=dict(color=PLOT_COLORS['neutral'], dash='dash', width=1),
                showlegend=False, hoverinfo='skip'
            ))
    
    hover_text = [f"Run {i+1}<br>Fitted: {f:.3f}<br>Residual: {r:.3f}" 
                  for i, (f, r) in enumerate(zip(fitted, residuals))]
    
    fig.add_trace(go.Scatter(
        x=fitted, y=residuals, mode='markers',
        marker=dict(size=8, color=PLOT_COLORS['primary'], opacity=0.7,
                   line=dict(width=0.5, color='white')),
        name='Residuals', text=hover_text, hovertemplate='%{text}<extra></extra>'
    ))
    
    fig.update_layout(
        xaxis_title='Fitted Values', yaxis_title='Residuals',
        height=400, showlegend=False
    )
    
    y_max = max(abs(residuals.min()), abs(residuals.max()))
    y_max = max(y_max, 3 * std_resid) * 1.1
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
    theoretical = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
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

with tab2:
    st.markdown("**Coefficient Table**")
    st.dataframe(results.effect_estimates, use_container_width=True)
    
    st.divider()
    
    with st.expander("🔍 Leverage Plots", expanded=False):
        model_terms = [t for t in current_terms if t != '1']
        
        if model_terms:
            for term in model_terms:
                st.markdown(f"**{format_term_for_display(term)}**")
                
                try:
                    X = results.model_matrix
                    if term in X.columns:
                        x_vals = X[term].values
                        other_terms = [t for t in X.columns if t != term and t != 'Intercept']
                        
                        if other_terms:
                            X_other = X[['Intercept'] + other_terms]
                            lr_other = LinearRegression(fit_intercept=False)
                            lr_other.fit(X_other, response_filtered)
                            y_other = lr_other.predict(X_other)
                            y_adj = response_filtered - y_other + response_filtered.mean()
                        else:
                            y_adj = response_filtered
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=x_vals, y=y_adj, mode='markers',
                            marker=dict(size=8, color=PLOT_COLORS['primary'], opacity=0.7,
                                       line=dict(width=0.5, color='white')),
                            name='Data'
                        ))
                        
                        lr_term = LinearRegression()
                        lr_term.fit(x_vals.reshape(-1, 1), y_adj)
                        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_line = lr_term.predict(x_line.reshape(-1, 1))
                        
                        fig.add_trace(go.Scatter(
                            x=x_line, y=y_line, mode='lines',
                            line=dict(color=PLOT_COLORS['danger'], width=2),
                            name='Effect'
                        ))
                        
                        fig.update_layout(
                            xaxis_title=format_term_for_display(term),
                            yaxis_title=f"{selected_response} (adjusted)",
                            height=300, showlegend=False
                        )
                        fig = apply_plot_style(fig)
                        st.plotly_chart(fig, use_container_width=True)
                
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
    if not effects_data.empty and 'Estimate' in effects_data.columns:
        effects = effects_data['Estimate'].values
        effect_names = [format_term_for_display(term) for term in effects_data.index]
        
        fig = create_half_normal_plot(effects, effect_names)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No effects to plot (intercept-only model)")

with tab3:
    st.info("📋 **Design Diagnostics** - Will be implemented in Session 11")
    
    st.markdown("**Planned Features:**")
    st.markdown("""
    - **Alias Structure**: Shows confounding patterns in fractional designs
    - **VIF (Variance Inflation Factors)**: Detects multicollinearity (VIF > 10 is concerning)
    - **Prediction Variance Map**: Visualizes prediction precision across design space
    - **Design Efficiencies**: D-efficiency, A-efficiency, G-efficiency metrics
    - **Variance vs Design Space**: Shows how prediction variance changes across factor space
    """)
    
    st.markdown("**Variance Inflation Factors (VIF)**")
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X = results.model_matrix
        if X is not None and len(X.columns) > 1:
            vif_data = pd.DataFrame()
            vif_cols = [col for col in X.columns if col != 'Intercept']
            if vif_cols:
                vif_data["Term"] = vif_cols
                vif_vals = []
                for i, col in enumerate(X.columns):
                    if col != 'Intercept':
                        try:
                            vif = variance_inflation_factor(X.values, i)
                            vif_vals.append(vif)
                        except:
                            vif_vals.append(np.nan)
                vif_data["VIF"] = vif_vals
                
                def highlight_vif(row):
                    if row['VIF'] > 10:
                        return ['background-color: #ffcccc'] * len(row)
                    return [''] * len(row)
                
                st.dataframe(vif_data.style.apply(highlight_vif, axis=1), hide_index=True)
                st.caption("⚠️ VIF > 10 indicates potential multicollinearity")
    except Exception as e:
        st.info(f"VIF calculation not available: {e}")

with tab4:
    st.info("📊 **Response Profiler** - Will be implemented in Session 11")
    
    st.markdown("**Planned Features:**")
    st.markdown("""
    - **Contour Plots**: 2D contour maps for pairs of factors
    - **3D Surface Plots**: Interactive 3D visualization of response surface
    - **Prediction Profiler**: Interactive sliders to explore factor effects
    - **Desirability Functions**: Multi-objective optimization interface
    - **Sweet Spot Plot**: Overlay feasible regions for multiple responses
    """)

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
