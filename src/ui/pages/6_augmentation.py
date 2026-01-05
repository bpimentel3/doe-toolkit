"""
Step 6: Design Augmentation (Optional)

Displays augmentation recommendations and allows user to add strategic runs
to improve design quality.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

from src.ui.utils.state_management import (
    initialize_session_state,
    is_step_complete,
    can_access_step,
    get_active_design,
    reset_augmentation
)
from src.ui.components.augmentation_wizard import (
    display_augmentation_plans,
    display_no_augmentation_needed,
    display_augmentation_placeholder
)

# Initialize state
initialize_session_state()

# Check access
if not can_access_step(6):
    st.warning("⚠️ Please complete Steps 1-5 first")
    
    # Show which step is incomplete
    for i in range(1, 6):
        if not is_step_complete(i):
            st.error(f"Step {i} is not complete")
    
    st.stop()

st.title("Step 6: Design Augmentation (Optional)")

# Introduction
st.markdown("""
Design augmentation intelligently adds experimental runs to:
- **Resolve aliasing** in fractional factorial designs
- **Add model terms** when lack of fit is detected  
- **Improve precision** in regions with high prediction variance

This step is *optional* - if your design quality is satisfactory, you can skip directly to optimization.
""")

st.divider()

# Check if augmentation already executed and waiting for new data
if st.session_state.get('augmented_design') is not None:
    augmented = st.session_state['augmented_design']
    
    # Check if we have data for the augmented design yet
    design = get_active_design()
    
    if len(design) == augmented.n_runs_total:
        # User has uploaded new data
        st.success(
            "✅ **Augmented design data imported!**\n\n"
            "You can now proceed to optimization with your improved design."
        )
        
        # Show summary
        st.metric("Total Runs", augmented.n_runs_total)
        st.metric("Original", augmented.n_runs_original)
        st.metric("Added", augmented.n_runs_added)
        
        # Navigation
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("← Back to Analysis", use_container_width=True):
                st.session_state['current_step'] = 5
                st.switch_page("pages/5_analyze.py")
        
        with col2:
            if st.button("Optimize →", type="primary", use_container_width=True):
                st.session_state['current_step'] = 7
                st.switch_page("pages/7_optimize.py")
    
    else:
        # Still waiting for new experimental data
        display_augmentation_placeholder()
        
        # Option to reset (start over)
        st.divider()
        
        with st.expander("⚙️ Augmentation Options"):
            st.markdown("**Start Over**")
            st.markdown(
                "If you want to select a different augmentation plan or "
                "skip augmentation, you can reset."
            )
            
            if st.button("🔄 Reset Augmentation", type="secondary"):
                reset_augmentation()
                st.rerun()
    
    st.stop()

# Check if quality report exists
if not st.session_state.get('quality_report'):
    st.warning(
        "⚠️ **No quality assessment available.**\n\n"
        "Please run quality assessment in **Step 5: Analyze** first."
    )
    
    if st.button("← Back to Analysis"):
        st.session_state['current_step'] = 5
        st.switch_page("pages/5_analyze.py")
    
    st.stop()

# Get quality report and diagnostics
quality_report = st.session_state['quality_report']
diagnostics = st.session_state['diagnostics_summary']

# Check if augmentation is needed
if not diagnostics.needs_any_augmentation():
    display_no_augmentation_needed()
    st.stop()

# Generate augmentation plans if not already generated
if 'augmentation_plans' not in st.session_state or not st.session_state['augmentation_plans']:
    
    with st.spinner("Analyzing design and generating augmentation recommendations..."):
        try:
            from src.core.augmentation.recommendations import recommend_augmentation_plans
            
            # Get fitted models
            fitted_models = st.session_state['fitted_models']
            
            # Budget constraint (optional user input)
            with st.sidebar:
                st.header("Augmentation Settings")
                
                use_budget = st.checkbox("Limit additional runs", value=False)
                
                if use_budget:
                    budget_constraint = st.number_input(
                        "Maximum additional runs",
                        min_value=1,
                        max_value=100,
                        value=16,
                        help="Maximum number of new runs to add"
                    )
                else:
                    budget_constraint = None
            
            # Generate plans
            plans = recommend_augmentation_plans(
                diagnostics=diagnostics,
                fitted_models=fitted_models,
                budget_constraint=budget_constraint
            )
            
            # Save to state
            st.session_state['augmentation_plans'] = plans
            
        except Exception as e:
            st.error(f"Failed to generate augmentation plans: {e}")
            st.exception(e)
            
            # Fallback navigation
            if st.button("← Back to Analysis"):
                st.session_state['current_step'] = 5
                st.switch_page("pages/5_analyze.py")
            
            st.stop()

# Display augmentation plans
plans = st.session_state['augmentation_plans']

if not plans:
    st.warning(
        "No augmentation plans could be generated. "
        "This might indicate an issue with the design or analysis."
    )
    
    if st.button("← Back to Analysis"):
        st.session_state['current_step'] = 5
        st.switch_page("pages/5_analyze.py")
    
    st.stop()

# Show the augmentation wizard
display_augmentation_plans(plans)

# Sidebar: Plan comparison
if len(plans) > 1:
    st.sidebar.header("Plan Comparison")
    
    comparison_data = []
    for plan in plans:
        comparison_data.append({
            'Plan': plan.plan_name,
            'Utility': f"{plan.utility_score:.0f}",
            'Runs': plan.n_runs_to_add,
            'Cost': f"${plan.experimental_cost:.0f}" if plan.experimental_cost else "N/A",
            'Benefits': ', '.join(plan.benefits_responses[:2])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.sidebar.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Help section
with st.sidebar.expander("ℹ️ Understanding Augmentation"):
    st.markdown("""
    **When to augment:**
    - Aliased effects in fractional designs
    - Lack of fit (curvature not captured)
    - High prediction variance
    - Collinearity issues
    
    **Augmentation strategies:**
    - **Foldover:** Resolves aliasing by flipping factor signs
    - **D-Optimal:** Adds runs optimized for model extension
    - **I-Optimal:** Improves prediction precision (future)
    
    **Costs vs Benefits:**
    - Additional runs require time and resources
    - But improve confidence in conclusions
    - Utility score helps prioritize
    """)