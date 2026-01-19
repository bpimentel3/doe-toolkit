"""
Step 7: Response Optimization

Find optimal factor settings to maximize/minimize responses using
desirability functions and prediction models.
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
from typing import Dict, List, Literal

from src.ui.utils.state_management import (
    initialize_session_state,
    is_step_complete,
    can_access_step,
    get_active_design,
    is_using_augmented_design
)

# Initialize state
initialize_session_state()

# Check access
if not can_access_step(7):
    st.warning("⚠️ Please complete Steps 1-6 first")
    st.stop()

st.title("Step 7: Response Optimization")

# Get active design and fitted models
design = get_active_design()
factors = st.session_state['factors']
fitted_models = st.session_state.get('fitted_models', {})

if not fitted_models:
    st.error("No fitted models available. Please complete analysis in Step 5.")
    st.stop()

# Show design status
if is_using_augmented_design():
    augmented = st.session_state['augmented_design']
    st.info(
        f"🔬 **Using augmented design** with {augmented.n_runs_added} additional runs "
        f"({augmented.n_runs_total} total)"
    )
    
    # Show phase distribution
    if 'Phase' in design.columns:
        phase_counts = design['Phase'].value_counts().sort_index()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Runs", phase_counts.get(1, 0))
        with col2:
            st.metric("Augmented Runs", phase_counts.get(2, 0))

st.divider()

# Sidebar: Optimization settings
st.sidebar.header("Optimization Settings")

response_names = list(fitted_models.keys())

if len(response_names) == 1:
    st.sidebar.markdown("**Single Response Optimization**")
    optimization_mode = 'single'
    primary_response = response_names[0]
else:
    st.sidebar.markdown("**Multi-Response Optimization**")
    optimization_mode = st.sidebar.radio(
        "Mode",
        ["Single Response", "Desirability Function"],
        key='opt_mode'
    )
    
    if optimization_mode == "Single Response":
        optimization_mode = 'single'
        primary_response = st.sidebar.selectbox(
            "Response to Optimize",
            response_names
        )
    else:
        optimization_mode = 'desirability'
        primary_response = None

# Main content tabs
if optimization_mode == 'single':
    tab1, tab2, tab3 = st.tabs(["🎯 Find Optimum", "📈 Response Surface", "📊 Contours"])
else:
    tab1, tab2 = st.tabs(["🎯 Multi-Response Optimization", "📈 Desirability Profile"])

# Tab 1: Find Optimum (Single Response)
if optimization_mode == 'single':
    with tab1:
        st.subheader(f"Optimize {primary_response}")
        
        # Optimization objective
        objective = st.radio(
            "Objective",
            ["Maximize", "Minimize", "Target"],
            horizontal=True,
            key=f'objective_{primary_response}'
        )
        
        if objective == "Target":
            target_value = st.number_input(
                "Target Value",
                value=0.0,
                key=f'target_{primary_response}'
            )
        
        # Constraints on factors
        st.markdown("**Factor Constraints**")
        
        factor_constraints = {}
        for factor in factors:
            if factor.is_continuous():
                col1, col2 = st.columns(2)
                
                with col1:
                    min_val = st.number_input(
                        f"{factor.name} Min",
                        value=float(factor.min_value),
                        key=f'min_{factor.name}'
                    )
                
                with col2:
                    max_val = st.number_input(
                        f"{factor.name} Max",
                        value=float(factor.max_value),
                        key=f'max_{factor.name}'
                    )
                
                factor_constraints[factor.name] = (min_val, max_val)
        
        # Optimize button
        if st.button("🔍 Find Optimal Settings", type="primary", use_container_width=True):
            with st.spinner("Optimizing..."):
                try:
                    from src.core.optimization import optimize_response
                    
                    # Get fitted model
                    anova_results = fitted_models[primary_response]
                    
                    # Prepare objective and target
                    obj_map = {"Maximize": "maximize", "Minimize": "minimize", "Target": "target"}
                    opt_objective = obj_map[objective]
                    
                    target_val = target_value if objective == "Target" else None
                    
                    # Run optimization
                    opt_result = optimize_response(
                        anova_results=anova_results,
                        factors=factors,
                        objective=opt_objective,
                        target_value=target_val,
                        bounds=factor_constraints if factor_constraints else None,
                        seed=42
                    )
                    
                    if opt_result.success:
                        # Display results
                        st.success(f"✅ Optimal settings found!")
                        
                        st.subheader("Optimal Factor Settings")
                        for fname, value in opt_result.optimal_settings.items():
                            factor = next(f for f in factors if f.name == fname)
                            if factor.units:
                                st.metric(fname, f"{value:.3f} {factor.units}")
                            else:
                                st.metric(fname, f"{value:.3f}")
                        
                        st.metric(f"Predicted {primary_response}", f"{opt_result.predicted_response:.3f}")
                        
                        # Confidence and prediction intervals
                        ci_lower, ci_upper = opt_result.confidence_interval
                        pi_lower, pi_upper = opt_result.prediction_interval
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.caption(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
                            st.caption("(for the mean response)")
                        with col2:
                            st.caption(f"95% Prediction Interval: [{pi_lower:.3f}, {pi_upper:.3f}]")
                            st.caption("(for a single observation)")
                        
                        # Show optimization details
                        with st.expander("🔍 Optimization Details"):
                            st.write(f"**Iterations:** {opt_result.n_iterations}")
                            st.write(f"**Objective Value:** {opt_result.objective_value:.6f}")
                            st.write(f"**Status:** {opt_result.message}")
                    else:
                        st.error(f"Optimization failed: {opt_result.message}")
                
                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    st.exception(e)
    
    # Tab 2: Response Surface
    with tab2:
        st.subheader(f"Response Surface: {primary_response}")
        
        # Select 2 factors for surface plot
        continuous_factors = [f for f in factors if f.is_continuous()]
        
        if len(continuous_factors) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_factor = st.selectbox("X-Axis Factor", [f.name for f in continuous_factors])
            
            with col2:
                y_factors = [f.name for f in continuous_factors if f.name != x_factor]
                y_factor = st.selectbox("Y-Axis Factor", y_factors)
            
            # Generate surface
            if st.button("Generate Surface", use_container_width=True):
                with st.spinner("Generating response surface..."):
                    try:
                        # Get factor ranges
                        x_fac = next(f for f in factors if f.name == x_factor)
                        y_fac = next(f for f in factors if f.name == y_factor)
                        
                        x_vals = np.linspace(x_fac.min_value, x_fac.max_value, 30)
                        y_vals = np.linspace(y_fac.min_value, y_fac.max_value, 30)
                        
                        X, Y = np.meshgrid(x_vals, y_vals)
                        
                        # Create prediction grid
                        grid_design = pd.DataFrame({
                            x_factor: X.ravel(),
                            y_factor: Y.ravel()
                        })
                        
                        # Add other factors at center
                        for factor in factors:
                            if factor.name not in [x_factor, y_factor]:
                                if factor.is_continuous():
                                    grid_design[factor.name] = (factor.min_value + factor.max_value) / 2
                                else:
                                    grid_design[factor.name] = factor.levels[0]
                        
                        # Predict
                        results = fitted_models[primary_response]
                        Z_pred = results.fitted_model.predict(grid_design).reshape(X.shape)
                        
                        # Plot
                        fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z_pred)])
                        
                        fig.update_layout(
                            title=f"Response Surface: {primary_response}",
                            scene=dict(
                                xaxis_title=x_factor,
                                yaxis_title=y_factor,
                                zaxis_title=primary_response
                            ),
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Surface generation failed: {e}")
        else:
            st.warning("Need at least 2 continuous factors for surface plots")
    
    # Tab 3: Contours
    with tab3:
        st.subheader(f"Contour Plot: {primary_response}")
        
        continuous_factors = [f for f in factors if f.is_continuous()]
        
        if len(continuous_factors) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                x_factor = st.selectbox("X-Axis", [f.name for f in continuous_factors], key='contour_x')
            
            with col2:
                y_factors = [f.name for f in continuous_factors if f.name != x_factor]
                y_factor = st.selectbox("Y-Axis", y_factors, key='contour_y')
            
            if st.button("Generate Contours", use_container_width=True):
                with st.spinner("Generating contours..."):
                    try:
                        # Similar to surface, but contour plot
                        x_fac = next(f for f in factors if f.name == x_factor)
                        y_fac = next(f for f in factors if f.name == y_factor)
                        
                        x_vals = np.linspace(x_fac.min_value, x_fac.max_value, 30)
                        y_vals = np.linspace(y_fac.min_value, y_fac.max_value, 30)
                        
                        X, Y = np.meshgrid(x_vals, y_vals)
                        
                        grid_design = pd.DataFrame({
                            x_factor: X.ravel(),
                            y_factor: Y.ravel()
                        })
                        
                        for factor in factors:
                            if factor.name not in [x_factor, y_factor]:
                                if factor.is_continuous():
                                    grid_design[factor.name] = (factor.min_value + factor.max_value) / 2
                        
                        results = fitted_models[primary_response]
                        Z_pred = results.fitted_model.predict(grid_design).reshape(X.shape)
                        
                        fig = go.Figure(data=go.Contour(x=x_vals, y=y_vals, z=Z_pred))
                        
                        fig.update_layout(
                            title=f"Contour Plot: {primary_response}",
                            xaxis_title=x_factor,
                            yaxis_title=y_factor,
                            height=600
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    
                    except Exception as e:
                        st.error(f"Contour generation failed: {e}")
        else:
            st.warning("Need at least 2 continuous factors for contour plots")

# Multi-response optimization
elif optimization_mode == 'desirability':
    with tab1:
        st.subheader("Multi-Response Optimization via Desirability")
        
        st.info(
            "Define desirability functions for each response. "
            "The optimizer will find factor settings that maximize overall desirability."
        )
        
        # Desirability configuration for each response
        desirability_config = {}
        
        for response_name in response_names:
            with st.expander(f"⚙️ Configure {response_name}"):
                goal = st.selectbox(
                    "Goal",
                    ["Maximize", "Minimize", "Target", "In Range"],
                    key=f'goal_{response_name}'
                )
                
                importance = st.slider(
                    "Importance",
                    min_value=1,
                    max_value=5,
                    value=3,
                    key=f'importance_{response_name}',
                    help="Relative importance (1=low, 5=critical)"
                )
                
                desirability_config[response_name] = {
                    'goal': goal,
                    'importance': importance
                }
        
        st.markdown("**Note:** Multi-response optimization implementation coming soon!")
        st.markdown("For now, optimize each response individually in single-response mode.")

# Navigation
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("← Back to Augmentation", use_container_width=True):
        st.session_state['current_step'] = 6
        st.switch_page("pages/6_augmentation.py")

with col2:
    st.markdown("*Workflow complete! Download results or start new project.*")