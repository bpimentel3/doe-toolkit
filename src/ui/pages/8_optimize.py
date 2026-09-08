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
from typing import Dict, List

from src.ui.utils.state_management import (
    initialize_session_state,
    is_step_complete,
    can_access_step,
    get_active_design,
    is_using_augmented_design
)
from src.core.coding import DesignSpace
from src.ui.utils.plotting import (
    create_3d_surface_plot,
    create_contour_plot
)

# Initialize state
initialize_session_state()

# Add standard sidebar
from src.ui.components.sidebar import build_standard_sidebar
build_standard_sidebar()

# Check access
if not can_access_step(7):
    st.warning("⚠️ Please complete Steps 1-6 first")
    st.stop()

st.title("Step 7: Response Optimization")

# Get active design and fitted models
design = get_active_design()
factors = st.session_state['factors']
fitted_models = st.session_state.get('fitted_models', {})

# Models are always fit in coded space (ANOVAAnalysis encodes on construction).
# Build the design space so optimizers can encode/decode factor settings.
_design_space = DesignSpace.from_factors(factors)
_model_is_coded = True  # invariant: ANOVAAnalysis always fits in coded space

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
if optimization_mode != 'single':
    tab1, tab2 = st.tabs(["🎯 Multi-Response Optimization", "📈 Desirability Profile"])

# Single Response: render content directly
if optimization_mode == 'single':
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
    pinned_levels = {}
    for factor in factors:
        if factor.is_continuous() or factor.is_discrete_numeric():
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

        elif factor.is_categorical():
            # The optimizer can pick the best level (default), or the user can
            # pin the level by selecting one explicitly.
            level_choice = st.selectbox(
                f"{factor.name} level",
                ["(Optimize — choose best level)"] + list(factor.levels),
                key=f'cat_{factor.name}',
                help="Leave as '(Optimize)' to let the optimizer select the "
                     "level that maximizes/minimizes the response."
            )
            if level_choice != "(Optimize — choose best level)":
                pinned_levels[factor.name] = level_choice

    # Optimize button
    if st.button("🔍 Find Optimal Settings", type="primary", width='stretch'):
        with st.spinner("Optimizing..."):
            try:
                from src.core.optimization import optimize_response

                anova_results = fitted_models[primary_response]
                obj_map = {"Maximize": "maximize", "Minimize": "minimize", "Target": "target"}
                opt_objective = obj_map[objective]
                target_val = target_value if objective == "Target" else None

                opt_result = optimize_response(
                    anova_results=anova_results,
                    factors=factors,
                    objective=opt_objective,
                    target_value=target_val,
                    bounds=factor_constraints if factor_constraints else None,
                    seed=42,
                    model_is_coded=_model_is_coded,
                    pinned_levels=pinned_levels if pinned_levels else None,
                )

                if opt_result.success:
                    st.success("✅ Optimal settings found!")

                    # Persist to session state for HTML report
                    if 'opt_results' not in st.session_state:
                        st.session_state['opt_results'] = {}
                    st.session_state['opt_results'][primary_response] = {
                        'objective': objective,
                        'optimal_settings': opt_result.optimal_settings,
                        'predicted_response': opt_result.predicted_response,
                        'confidence_interval': opt_result.confidence_interval,
                        'prediction_interval': opt_result.prediction_interval,
                        'figures': {},
                    }

                    # Silently generate surface/contour figures for the HTML
                    # report. Uses the first two continuous factors by default.
                    _cont_factors = [f for f in factors if f.is_continuous()]
                    if len(_cont_factors) >= 2:
                        try:
                            from src.core.coding import encode_design
                            _xf, _yf = _cont_factors[0], _cont_factors[1]
                            _xv = np.linspace(_xf.min_value, _xf.max_value, 30)
                            _yv = np.linspace(_yf.min_value, _yf.max_value, 30)
                            _X, _Y = np.meshgrid(_xv, _yv)
                            _grid = pd.DataFrame({
                                _xf.name: _X.ravel(),
                                _yf.name: _Y.ravel(),
                            })
                            for _f in factors:
                                if _f.name not in [_xf.name, _yf.name]:
                                    _grid[_f.name] = (
                                        (_f.min_value + _f.max_value) / 2
                                        if _f.is_continuous()
                                        else _f.levels[0]
                                    )
                            _grid_enc = encode_design(_grid, factors)
                            _preds = fitted_models[primary_response].fitted_model.predict(_grid_enc)
                            _Z = np.array(_preds).reshape(_X.shape)
                            _surface_fig = create_3d_surface_plot(
                                x_grid=_xv, y_grid=_yv, z_mesh=_Z,
                                x_factor_name=_xf.name, y_factor_name=_yf.name,
                                response_name=primary_response
                            )
                            _contour_fig = create_contour_plot(
                                x_grid=_xv, y_grid=_yv, z_mesh=_Z,
                                x_factor_name=_xf.name, y_factor_name=_yf.name,
                                response_name=primary_response
                            )
                            st.session_state['opt_results'][primary_response]['figures'] = {
                                'surface': _surface_fig,
                                'contour': _contour_fig,
                            }
                        except Exception:
                            pass  # Non-critical; report omits plots gracefully

                    st.subheader("Optimal Factor Settings")
                    for fname, value in opt_result.optimal_settings.items():
                        factor = next(f for f in factors if f.name == fname)
                        if factor.is_categorical():
                            st.metric(fname, str(value))
                        elif factor.units:
                            st.metric(fname, f"{value:.3f} {factor.units}")
                        else:
                            st.metric(fname, f"{value:.3f}")

                    st.metric(f"Predicted {primary_response}", f"{opt_result.predicted_response:.3f}")

                    ci_lower, ci_upper = opt_result.confidence_interval
                    pi_lower, pi_upper = opt_result.prediction_interval
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"95% Confidence Interval: [{ci_lower:.3f}, {ci_upper:.3f}]")
                        st.caption("(for the mean response)")
                    with col2:
                        st.caption(f"95% Prediction Interval: [{pi_lower:.3f}, {pi_upper:.3f}]")
                        st.caption("(for a single observation)")

                    with st.expander("🔍 Optimization Details"):
                        st.write(f"**Iterations:** {opt_result.n_iterations}")
                        st.write(f"**Objective Value:** {opt_result.objective_value:.6f}")
                        st.write(f"**Status:** {opt_result.message}")
                else:
                    st.error(f"Optimization failed: {opt_result.message}")

            except Exception as e:
                st.error(f"Optimization failed: {e}")
                st.exception(e)

# Multi-response optimization
elif optimization_mode == 'desirability':
    with tab1:
        st.subheader("Multi-Response Optimization via Desirability")

        st.info(
            "Configure a desirability goal for each response. "
            "The optimizer maximizes overall desirability — the geometric mean of "
            "individual desirabilities — to find factor settings that satisfy all "
            "response objectives simultaneously."
        )

        # --- Per-response desirability configuration ---
        desirability_config: Dict[str, Dict] = {}

        for response_name in response_names:
            with st.expander(f"⚙️ Configure: {response_name}", expanded=True):
                col_goal, col_imp = st.columns([2, 1])

                with col_goal:
                    goal = st.selectbox(
                        "Goal",
                        ["Maximize", "Minimize", "Target"],
                        key=f'goal_{response_name}'
                    )

                with col_imp:
                    importance = st.slider(
                        "Importance",
                        min_value=1,
                        max_value=5,
                        value=3,
                        key=f'importance_{response_name}',
                        help="Relative importance in geometric mean (1=low, 5=critical)"
                    )

                col_lo, col_hi = st.columns(2)

                with col_lo:
                    low_val = st.number_input(
                        "Low (d=0)" if goal == "Maximize" else
                        "Low (d=1)" if goal == "Minimize" else
                        "Low (d=0)",
                        value=0.0,
                        key=f'low_{response_name}'
                    )

                with col_hi:
                    high_val = st.number_input(
                        "High (d=1)" if goal == "Maximize" else
                        "High (d=0)" if goal == "Minimize" else
                        "High (d=0)",
                        value=1.0,
                        key=f'high_{response_name}'
                    )

                target_val: float = 0.0
                if goal == "Target":
                    target_val = st.number_input(
                        "Target (d=1)",
                        value=(low_val + high_val) / 2,
                        key=f'target_{response_name}'
                    )

                weight_val = st.slider(
                    "Weight (shape)",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    key=f'weight_{response_name}',
                    help="1=linear ramp, >1=emphasize target, <1=more tolerant"
                )

                desirability_config[response_name] = {
                    'goal': goal,
                    'low': low_val,
                    'high': high_val,
                    'target': target_val,
                    'weight': weight_val,
                    'importance': float(importance)
                }

        st.divider()

        # --- Factor bounds ---
        pinned_levels_d: Dict[str, object] = {}
        with st.expander("🔧 Factor Bounds (optional)"):
            factor_bounds_d: Dict[str, tuple] = {}
            for factor in factors:
                if factor.is_continuous() or factor.is_discrete_numeric():
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        bmin = st.number_input(
                            f"{factor.name} Min",
                            value=float(factor.min_value),
                            key=f'dmin_{factor.name}'
                        )
                    with bc2:
                        bmax = st.number_input(
                            f"{factor.name} Max",
                            value=float(factor.max_value),
                            key=f'dmax_{factor.name}'
                        )
                    factor_bounds_d[factor.name] = (bmin, bmax)

                elif factor.is_categorical():
                    level_choice = st.selectbox(
                        f"{factor.name} level",
                        ["(Optimize — choose best level)"] + list(factor.levels),
                        key=f'dcat_{factor.name}',
                        help="Leave as '(Optimize)' to let the optimizer select "
                             "the best level across all responses."
                    )
                    if level_choice != "(Optimize — choose best level)":
                        pinned_levels_d[factor.name] = level_choice

        # --- Validate config before allowing run ---
        config_errors: List[str] = []
        for rn, cfg in desirability_config.items():
            if cfg['low'] >= cfg['high']:
                config_errors.append(
                    f"{rn}: Low must be less than High."
                )
            if cfg['goal'] == 'Target':
                if not (cfg['low'] < cfg['target'] < cfg['high']):
                    config_errors.append(
                        f"{rn}: Target must be strictly between Low and High."
                    )

        if config_errors:
            for err in config_errors:
                st.error(err)

        run_disabled = bool(config_errors)

        if st.button(
            "🔍 Find Optimal Settings",
            type="primary",
            disabled=run_disabled,
            key='run_desirability'
        ):
            with st.spinner("Optimizing across all responses..."):
                try:
                    from src.core.optimization import (
                        DesirabilityFunction,
                        optimize_desirability
                    )

                    # Build DesirabilityFunction
                    d_func = DesirabilityFunction(response_names)

                    for rn, cfg in desirability_config.items():
                        goal_map = {
                            'Maximize': 'maximize',
                            'Minimize': 'minimize',
                            'Target': 'target'
                        }
                        d_func.add_response(
                            response_name=rn,
                            objective=goal_map[cfg['goal']],
                            low=cfg['low'],
                            high=cfg['high'],
                            target=cfg['target'] if cfg['goal'] == 'Target' else None,
                            weight=cfg['weight'],
                            importance=cfg['importance']
                        )

                    # Run optimizer
                    d_result = optimize_desirability(
                        anova_results_dict=fitted_models,
                        factors=factors,
                        desirability_func=d_func,
                        bounds=factor_bounds_d if factor_bounds_d else None,
                        seed=42,
                        model_is_coded=_model_is_coded,
                        pinned_levels=pinned_levels_d if pinned_levels_d else None,
                    )

                    # Store result for profile tab
                    st.session_state['desirability_result'] = d_result
                    st.session_state['desirability_config'] = desirability_config

                    if d_result.success:
                        st.success("✅ Optimal settings found!")
                    else:
                        st.warning(
                            f"⚠️ Optimizer did not fully converge: {d_result.message}. "
                            "Results may still be useful."
                        )

                    # --- Optimal factor settings ---
                    st.subheader("Optimal Factor Settings")
                    settings_cols = st.columns(min(len(factors), 4))
                    for idx, factor in enumerate(factors):
                        val = d_result.optimal_settings.get(factor.name)
                        if val is not None:
                            label = f"{factor.name} ({factor.units})" if factor.units else factor.name
                            if factor.is_categorical():
                                settings_cols[idx % len(settings_cols)].metric(
                                    label, str(val)
                                )
                            else:
                                settings_cols[idx % len(settings_cols)].metric(
                                    label, f"{val:.3f}"
                                )

                    st.divider()

                    # --- Response predictions and desirabilities ---
                    st.subheader("Predicted Responses & Desirabilities")

                    results_rows = []
                    for rn in response_names:
                        results_rows.append({
                            'Response': rn,
                            'Predicted': round(d_result.predicted_responses[rn], 4),
                            'Desirability (dᵢ)': round(
                                d_result.individual_desirabilities[rn], 4
                            ),
                            'Goal': desirability_config[rn]['goal'],
                            'Importance': int(desirability_config[rn]['importance'])
                        })

                    results_df = pd.DataFrame(results_rows)
                    st.dataframe(results_df, hide_index=True, width="stretch")

                    # Overall desirability — prominent display
                    st.metric(
                        "Overall Desirability (D)",
                        f"{d_result.overall_desirability:.4f}",
                        help="Geometric mean of individual desirabilities. "
                             "D=1 is ideal; D=0 means at least one response is unacceptable."
                    )

                    with st.expander("🔍 Optimization Details"):
                        st.write(f"**Iterations:** {d_result.n_iterations}")
                        st.write(f"**Status:** {d_result.message}")

                except Exception as e:
                    st.error(f"Optimization failed: {e}")
                    st.exception(e)

    with tab2:
        st.subheader("Desirability Profile")

        d_result = st.session_state.get('desirability_result')
        d_config = st.session_state.get('desirability_config', {})

        if d_result is None:
            st.info("Run the optimization first to see the desirability profile.")
        else:
            import plotly.graph_objects as go

            # Individual desirability bar chart
            resp_labels = list(d_result.individual_desirabilities.keys())
            d_values = [d_result.individual_desirabilities[r] for r in resp_labels]

            bar_colors = [
                '#2ecc71' if v >= 0.8 else '#f39c12' if v >= 0.5 else '#e74c3c'
                for v in d_values
            ]

            fig_bar = go.Figure(go.Bar(
                x=resp_labels,
                y=d_values,
                marker_color=bar_colors,
                text=[f"{v:.3f}" for v in d_values],
                textposition='outside'
            ))
            fig_bar.update_layout(
                title="Individual Desirabilities",
                yaxis=dict(range=[0, 1.1], title="Desirability"),
                xaxis_title="Response",
                showlegend=False,
                height=350
            )
            fig_bar.add_hline(
                y=d_result.overall_desirability,
                line_dash='dash',
                line_color='navy',
                annotation_text=f"Overall D = {d_result.overall_desirability:.3f}",
                annotation_position='top right'
            )
            st.plotly_chart(fig_bar, width="stretch")

            # Desirability summary table
            st.subheader("Summary at Optimal Settings")
            summary_rows = []
            for rn in resp_labels:
                cfg = d_config.get(rn, {})
                summary_rows.append({
                    'Response': rn,
                    'Goal': cfg.get('goal', '—'),
                    'Low': cfg.get('low', '—'),
                    'High': cfg.get('high', '—'),
                    'Target': cfg.get('target', '—') if cfg.get('goal') == 'Target' else '—',
                    'Predicted': round(d_result.predicted_responses[rn], 4),
                    'dᵢ': round(d_result.individual_desirabilities[rn], 4)
                })
            st.dataframe(
                pd.DataFrame(summary_rows),
                hide_index=True,
                width="stretch"
            )

            st.metric(
                "Overall Desirability (D)",
                f"{d_result.overall_desirability:.4f}"
            )

# Navigation
st.divider()

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("← Back to Augmentation", width='stretch'):
        st.session_state['current_step'] = 6
        st.switch_page("pages/6_augmentation.py")

with col2:
    st.markdown("*Workflow complete! Download results or start new project.*")