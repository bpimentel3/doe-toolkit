"""
Step 5: Analyze Experimental Results

Fit ANOVA models, assess quality, and identify augmentation needs.
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

from src.ui.utils.state_management import (
    initialize_session_state,
    is_step_complete,
    can_access_step,
    get_active_design,
    invalidate_downstream_state
)
from src.ui.components.quality_dashboard import display_quality_dashboard
from src.ui.components.model_builder import display_model_builder, format_full_equation
from src.core.analysis import ANOVAAnalysis, generate_model_terms
from src.core.diagnostics.summary import (
    compute_design_diagnostic_summary,
    generate_quality_report
)

# Initialize state
initialize_session_state()

# Check access
if not can_access_step(5):
    st.warning("⚠️ Please complete Steps 1-4 first")
    st.stop()

st.title("Step 5: Analyze Experimental Results")

# Get active design and responses
design = get_active_design()
factors = st.session_state['factors']
responses = st.session_state['responses']
response_names = st.session_state['response_names']

if not responses:
    st.error("No response data available. Please import data in Step 4.")
    st.stop()

# Response Selection (moved to main page)
st.subheader("📊 Response Selection")

selected_response = st.selectbox(
    "Select Response to Analyze",
    response_names,
    key='analysis_response_selector'
)

st.divider()

# Get or initialize model terms for this response
if 'model_terms_per_response' not in st.session_state:
    st.session_state['model_terms_per_response'] = {}

if selected_response not in st.session_state['model_terms_per_response']:
    # Default: linear model
    default_terms = generate_model_terms(factors, 'linear', include_intercept=True)
    st.session_state['model_terms_per_response'][selected_response] = default_terms

current_terms = st.session_state['model_terms_per_response'][selected_response]

# Model Builder Interface
updated_terms = display_model_builder(
    factors=factors,
    current_terms=current_terms,
    response_name=selected_response,
    key_prefix=f"model_builder_{selected_response}"
)

# Check if terms changed
if updated_terms != current_terms:
    st.session_state['model_terms_per_response'][selected_response] = updated_terms
    invalidate_downstream_state(from_step=5)
    st.rerun()

st.divider()

# Advanced Options in Sidebar
st.sidebar.header("Advanced Options")

# Hierarchy enforcement
enforce_hierarchy = st.sidebar.checkbox(
    "Enforce Hierarchy",
    value=True,
    help="Automatically include lower-order terms when fitting model"
)

# Data exclusion
st.sidebar.subheader("Data Exclusion")

if 'excluded_rows' not in st.session_state:
    st.session_state['excluded_rows'] = []

exclude_mode = st.sidebar.checkbox("Enable Row Exclusion")

if exclude_mode:
    exclude_indices = st.sidebar.multiselect(
        "Exclude Runs",
        options=list(range(len(design))),
        default=st.session_state['excluded_rows'],
        format_func=lambda x: f"Run {x+1}",
        key='exclude_runs'
    )
    
    if exclude_indices != st.session_state['excluded_rows']:
        st.session_state['excluded_rows'] = exclude_indices
        invalidate_downstream_state(from_step=5)

# Results Area
st.subheader("📈 Analysis Results")

tab1, tab2, tab3 = st.tabs(["📊 Model Fit", "🔍 Diagnostics", "📈 Quality Report"])

# Fit model for current response
with st.spinner(f"Fitting model for {selected_response}..."):
    try:
        # Get response data
        response_data = responses[selected_response]
        
        # Apply exclusions
        if st.session_state['excluded_rows']:
            mask = np.ones(len(design), dtype=bool)
            mask[st.session_state['excluded_rows']] = False
            design_filtered = design[mask].reset_index(drop=True)
            response_filtered = response_data[mask]
        else:
            design_filtered = design
            response_filtered = response_data
        
        # Fit ANOVA
        analysis = ANOVAAnalysis(
            design=design_filtered,
            response=response_filtered,
            factors=factors,
            response_name=selected_response
        )
        
        results = analysis.fit(
            model_terms=current_terms,
            enforce_hierarchy_flag=enforce_hierarchy
        )
        
        # Warn if factor names were sanitized
        if getattr(analysis, "rename_map", {}):
            renamed = ", ".join([f"{old} → {new}" for old, new in analysis.rename_map.items()])
            st.warning(
                f"Some factor names were automatically renamed for compatibility: {renamed}. "
                "This prevents errors in the statistical model formula."
            )

        # Save to state
        if 'fitted_models' not in st.session_state:
            st.session_state['fitted_models'] = {}
        
        st.session_state['fitted_models'][selected_response] = results
        
    except Exception as e:
        st.error(f"Model fitting failed: {e}")
        st.exception(e)
        st.stop()

# Tab 1: Model Fit
with tab1:
    st.subheader(f"ANOVA Results: {selected_response}")
    
    # Model summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R²", f"{results.r_squared:.4f}")
    with col2:
        st.metric("Adj R²", f"{results.adj_r_squared:.4f}")
    with col3:
        st.metric("RMSE", f"{results.rmse:.3f}")
    with col4:
        n_runs = len(design_filtered)
        n_params = len([t for t in current_terms if t != '1']) + 1
        st.metric("DF Error", n_runs - n_params)
    
    st.divider()
    
    # ANOVA Table
    if not results.anova_table.empty:
        st.subheader("ANOVA Table")
        st.dataframe(results.anova_table, use_container_width=True)
    
    # Effect Estimates
    st.subheader("Effect Estimates")
    st.dataframe(results.effect_estimates, use_container_width=True)
    
    # LogWorth plot with p-values - SMALLER SIZE
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Effect Significance")
        
        if not results.logworth.empty:
            # Create figure with reduced height
            fig = px.bar(
                results.logworth,
                x='LogWorth',
                y=results.logworth.index,
                orientation='h',
                title='LogWorth (-log10(p-value))'
            )
            
            # Add significance threshold line
            fig.add_vline(
                x=-np.log10(0.05),
                line_dash="dash",
                line_color="red",
                annotation_text="p=0.05"
            )
            
            # Reduce plot height
            fig.update_layout(height=300)
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("p-values")
        
        if not results.logworth.empty:
            # Calculate p-values from LogWorth
            p_values = pd.DataFrame({
                'Term': results.logworth.index,
                'p-value': 10 ** (-results.logworth['LogWorth'])
            })
            
            # Format p-values
            p_values['p-value'] = p_values['p-value'].apply(
                lambda x: f"{x:.4f}" if x >= 0.0001 else f"{x:.2e}"
            )
            
            st.dataframe(p_values, hide_index=True, use_container_width=True)

# Tab 2: Diagnostics
with tab2:
    st.subheader("Model Diagnostics")
    
    # Row 1: Actual vs Predicted (parity plot)
    st.markdown("**Actual vs Predicted**")
    
    # Create parity plot
    actual = response_filtered
    predicted = results.fitted_values
    
    # Calculate 1:1 line limits
    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    margin = (max_val - min_val) * 0.1
    plot_min = min_val - margin
    plot_max = max_val + margin
    
    fig = go.Figure()
    
    # Scatter points
    fig.add_trace(go.Scatter(
        x=actual,
        y=predicted,
        mode='markers',
        marker=dict(size=10, opacity=0.6, color='#1f77b4'),
        name='Data'
    ))
    
    # 1:1 line
    fig.add_trace(go.Scatter(
        x=[plot_min, plot_max],
        y=[plot_min, plot_max],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Fit (1:1)'
    ))
    
    # Fitted line (linear regression through scatter)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(actual.reshape(-1, 1), predicted)
    fitted_line_x = np.array([plot_min, plot_max])
    fitted_line_y = lr.predict(fitted_line_x.reshape(-1, 1))
    
    fig.add_trace(go.Scatter(
        x=fitted_line_x,
        y=fitted_line_y,
        mode='lines',
        line=dict(color='green', width=2),
        name=f'Fitted (R²={results.r_squared:.3f})'
    ))
    
    fig.update_layout(
        xaxis_title="Actual",
        yaxis_title="Predicted",
        height=400,
        showlegend=True
    )
    
    # Make axes equal scale
    fig.update_xaxes(scaleanchor="y", scaleratio=1)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Row 2: Residual plots
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Residuals vs Fitted**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=results.fitted_values,
            y=results.residuals,
            mode='markers',
            marker=dict(size=8, opacity=0.6)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        fig.update_layout(
            xaxis_title="Fitted Values",
            yaxis_title="Residuals",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**Normal Q-Q Plot**")
        
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(results.residuals))
        )
        sample_quantiles = np.sort(results.residuals)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=theoretical_quantiles,
            y=sample_quantiles,
            mode='markers',
            marker=dict(size=8, opacity=0.6)
        ))
        
        # Add reference line
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            showlegend=False
        ))
        
        fig.update_layout(
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Sample Quantiles",
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Normality test
    if 'shapiro_wilk' in results.diagnostics:
        sw = results.diagnostics['shapiro_wilk']
        st.info(
            f"**Shapiro-Wilk Test:** W = {sw['statistic']:.4f}, "
            f"p-value = {sw['p_value']:.4f}"
        )
        if sw['p_value'] < 0.05:
            st.warning("Residuals may not be normally distributed (p < 0.05)")
    
    st.divider()
    
    # Leverage plots (hidden in expander)
    with st.expander("🔍 Leverage Plots (Effect Plots)", expanded=False):
        st.markdown("""
        **Leverage plots** show the effect of each term on the response while accounting 
        for all other terms in the model. The slope indicates the magnitude of the effect.
        """)
        
        # Get model terms (excluding intercept)
        model_terms = [t for t in current_terms if t != '1']
        
        if not model_terms:
            st.info("No terms to plot (model is intercept-only)")
        else:
            # Create leverage plots for each term
            for term in model_terms:
                st.markdown(f"**{term}**")
                
                try:
                    # Get the model matrix column for this term
                    X = results.model_matrix
                    
                    # Find column index for this term
                    if term in X.columns:
                        # Extract predictor values
                        x_vals = X[term].values
                        
                        # Compute leverage residuals:
                        # Response adjusted for all other terms
                        other_terms = [t for t in X.columns if t != term and t != 'Intercept']
                        
                        if other_terms:
                            # Fit model without this term
                            from sklearn.linear_model import LinearRegression
                            X_other = X[['Intercept'] + other_terms]
                            lr_other = LinearRegression(fit_intercept=False)
                            lr_other.fit(X_other, response_filtered)
                            y_other = lr_other.predict(X_other)
                            
                            # Adjusted response (what's left after removing other terms)
                            y_adj = response_filtered - y_other + response_filtered.mean()
                        else:
                            # If no other terms, just use raw response
                            y_adj = response_filtered
                        
                        # Create leverage plot
                        fig = go.Figure()
                        
                        # Scatter
                        fig.add_trace(go.Scatter(
                            x=x_vals,
                            y=y_adj,
                            mode='markers',
                            marker=dict(size=8, opacity=0.6),
                            name='Data'
                        ))
                        
                        # Fitted line for this term
                        lr_term = LinearRegression()
                        lr_term.fit(x_vals.reshape(-1, 1), y_adj)
                        x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
                        y_line = lr_term.predict(x_line.reshape(-1, 1))
                        
                        fig.add_trace(go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode='lines',
                            line=dict(color='red', width=2),
                            name='Effect'
                        ))
                        
                        fig.update_layout(
                            xaxis_title=term,
                            yaxis_title=f"{selected_response} (adjusted)",
                            height=300,
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    else:
                        st.warning(f"Term '{term}' not found in model matrix")
                
                except Exception as e:
                    st.error(f"Could not create leverage plot for '{term}': {e}")

# Tab 3: Quality Report
with tab3:
    st.subheader("Design Quality Assessment")
    
    # Check if we've computed diagnostics for ALL responses
    compute_diagnostics = st.button(
        "🔍 Compute Quality Report for All Responses",
        type="primary",
        use_container_width=True
    )
    
    if compute_diagnostics or st.session_state.get('quality_report'):
        
        # Ensure all responses have fitted models
        missing_models = [r for r in response_names 
                         if r not in st.session_state['fitted_models']]
        
        if missing_models:
            st.warning(
                f"Some responses not yet analyzed: {', '.join(missing_models)}. "
                "Fit models for all responses first."
            )
        else:
            with st.spinner("Computing diagnostics across all responses..."):
                try:
                    # Compute diagnostic summary
                    summary = compute_design_diagnostic_summary(
                        design=design,
                        responses=responses,
                        fitted_models=st.session_state['fitted_models'],
                        factors=factors,
                        model_terms_per_response=st.session_state['model_terms_per_response'],
                        design_metadata=st.session_state['design_metadata']
                    )
                    
                    # Generate quality report
                    report = generate_quality_report(summary)
                    
                    # Save to state
                    st.session_state['diagnostics_summary'] = summary
                    st.session_state['quality_report'] = report
                    
                    # Display dashboard
                    display_quality_dashboard(report, show_augmentation_button=True)
                    
                except Exception as e:
                    st.error(f"Quality assessment failed: {e}")
                    st.exception(e)
            
            # Show raw report in expander
            if st.session_state.get('quality_report'):
                with st.expander("📄 View Full Report (Markdown)"):
                    st.markdown(st.session_state['quality_report'].to_markdown())

# Navigation
st.divider()

col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    if st.button("← Back to Import", use_container_width=True):
        st.session_state['current_step'] = 4
        st.switch_page("pages/4_import_results.py")

with col2:
    # Show augmentation if quality report suggests it
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
