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
    display_diagnostic_summary,
    display_type_selection,
    display_augmentation_plans,
    display_plan_execution,
    display_no_augmentation_needed,
    display_augmentation_placeholder
)
from src.core.augmentation import (
    AugmentationRequest,
    recommend_augmentation
)

# Initialize state
initialize_session_state()

# Add standard sidebar
from src.ui.components.sidebar import build_standard_sidebar
build_standard_sidebar()

# Check access - only need a design structure
if not can_access_step(6):
    st.warning("⚠️ Please create or import a design first")
    
    st.info(
        "**To access augmentation:**\n\n"
        "Option 1: Create a design (Steps 1-3)\n"
        "Option 2: Import a CSV with design structure (Step 4)"
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Go to Factor Definition", width='stretch'):
            st.session_state['current_step'] = 1
            st.switch_page("pages/1_define_factors.py")
    
    with col2:
        if st.button("Go to Import Data →", width='stretch'):
            st.session_state['current_step'] = 5
            st.switch_page("pages/5_import_results.py")
    
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
    
    results_displayed = st.session_state.get('augmentation_results_displayed', False)

    if results_displayed:
        # Still on the results screen — re-render the full results view
        display_plan_execution(st.session_state['selected_augmentation_plan'])

    elif len(design) == augmented.n_runs_total:
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
            if st.button("← Back to Analysis", width='stretch'):
                st.session_state['current_step'] = 5
                st.switch_page("pages/6_analyze.py")
        
        with col2:
            if st.button("Optimize →", type="primary", width='stretch'):
                st.session_state['current_step'] = 7
                st.switch_page("pages/8_optimize.py")
    
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

# Check if selected plan exists and needs execution
if st.session_state.get('selected_augmentation_plan'):
    plan = st.session_state['selected_augmentation_plan']
    display_plan_execution(plan)
    st.stop()

# Check if quality report exists - if not, show design info and suggest analysis
if not st.session_state.get('quality_report'):
    st.info(
        "📊 **Design Structure Available**\n\n"
        "You have a design loaded, but haven't run analysis yet. "
        "Augmentation recommendations are most powerful when based on analysis results."
    )
    
    # Show basic design info
    design = get_active_design()
    factors = st.session_state.get('factors', [])
    
    st.subheader("📋 Current Design")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Factors", len(factors))
    with col2:
        st.metric("Runs", len(design))
    with col3:
        design_type = st.session_state.get('design_metadata', {}).get('design_type', 'Unknown')
        st.metric("Type", design_type.replace('_', ' ').title())
    
    # Show design preview
    with st.expander("👁️ View Design"):
        st.dataframe(design, width='stretch', hide_index=False)
    
    st.divider()
    
    # Suggest next steps
    st.markdown("### Recommended Next Steps")
    
    st.markdown(
        "**Option 1: Analyze First (Recommended)**\n"
        "- Import experimental results (Step 4)\n"
        "- Fit models and view diagnostics (Step 5)\n"
        "- Get intelligent augmentation recommendations based on your data\n\n"
        
        "**Option 2: Plan Augmentation Now**\n"
        "- You can still augment based on design structure alone\n"
        "- Useful for planning experiments before running them\n"
        "- Limited to structural augmentation strategies"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Go to Analysis", type="primary", width='stretch'):
            st.session_state['current_step'] = 5
            st.switch_page("pages/6_analyze.py")
    
    with col2:
        # Allow proceeding without analysis
        if st.button("➡️ Continue Without Analysis", width='stretch'):
            st.session_state['skip_quality_check'] = True
            st.rerun()
    
    # If user chose to skip quality check, continue
    if not st.session_state.get('skip_quality_check', False):
        st.stop()
    
    # User chose to continue - show basic augmentation options
    st.divider()
    st.subheader("🔧 Basic Augmentation Options (No Analysis)")
    
    st.markdown(
        "Without analysis results, augmentation options are limited to structural strategies:\n\n"
        "- **Foldover designs** (for fractional factorials)\n"
        "- **Add center points** (for curvature detection)\n"
        "- **Add axial points** (to extend to response surface)\n\n"
        "For intelligent, data-driven recommendations, please complete Step 5: Analysis first."
    )
    
    # TODO: Implement basic structural augmentation options
    st.warning("⚠️ Basic augmentation (without analysis) is not yet implemented.")
    st.info("Please complete Step 5: Analysis to access full augmentation capabilities.")
    
    st.stop()

# Get quality report and diagnostics
quality_report = st.session_state['quality_report']
diagnostics = st.session_state['diagnostics_summary']

# Step 1 (always): show plain-text diagnostic summary
display_diagnostic_summary(diagnostics)

st.divider()

# Step 2: flat augmentation type selection
if 'augmentation_type' not in st.session_state:
    st.subheader("🔧 Choose an Augmentation Type")
    selected_type = display_type_selection(diagnostics)

    if selected_type:
        st.session_state['augmentation_type'] = selected_type
        st.rerun()

    st.stop()

# Step 3: generate plans for the chosen type
if 'augmentation_plans' not in st.session_state or not st.session_state['augmentation_plans']:

    with st.spinner("Generating augmentation recommendations..."):
        try:
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

            request = AugmentationRequest(
                mode='select_type',
                diagnostics=diagnostics,
                selected_goal=None,
                selected_type=st.session_state['augmentation_type'],
                budget_constraint=budget_constraint
            )

            plans = recommend_augmentation(request)
            st.session_state['augmentation_plans'] = plans

        except Exception as e:
            st.error(f"Failed to generate augmentation plans: {e}")
            st.exception(e)

            if st.button("← Start Over"):
                for key in ['augmentation_type', 'augmentation_plans']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

            st.stop()

# Display augmentation plans
plans = st.session_state['augmentation_plans']

if not plans:
    st.warning(
        "No augmentation plans could be generated for the selected type. "
        "Please try a different augmentation type."
    )

    if st.button("← Choose Different Type"):
        for key in ['augmentation_type', 'augmentation_plans']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

    st.stop()

display_augmentation_plans(plans, 'select_type')

# Back / reset button
if st.button("← Choose Different Type"):
    for key in ['augmentation_type', 'augmentation_plans']:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# Sidebar: Plan comparison
if len(plans) > 1:
    st.sidebar.header("Plan Comparison")

    comparison_data = [
        {
            'Plan': plan.plan_name,
            'Runs': plan.n_runs_to_add,
            'Total After': plan.total_runs_after,
            'Benefits': ', '.join(plan.benefits_responses[:2]),
        }
        for plan in plans
    ]

    st.sidebar.dataframe(
        pd.DataFrame(comparison_data),
        width='stretch',
        hide_index=True
    )

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
    - Compare plans by runs added and expected improvements
    """)