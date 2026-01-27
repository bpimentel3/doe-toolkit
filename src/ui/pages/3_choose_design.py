"""
Step 2: Choose Design Type

Select and configure experimental design type.
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
    can_access_step,
    invalidate_downstream_state
)
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.ui.components.constraint_builder import (
    show_constraint_builder,
    show_constraint_help
)

# Initialize state
initialize_session_state()

# Add standard sidebar
from src.ui.components.sidebar import build_standard_sidebar
build_standard_sidebar()

# Check access
if not can_access_step(3):
    st.warning("⚠️ Please complete Steps 1-2 first")
    st.stop()

st.title("Step 3: Choose Design Type")

factors = st.session_state['factors']

# Check if model is selected
model_selected = 'model_terms' in st.session_state and st.session_state['model_terms']

if not model_selected:
    st.warning(
        "⚠️ **No analysis model selected yet.** "
        "Some design types (especially D-Optimal) require knowing the model upfront. "
        "We recommend completing Step 2 first."
    )
    if st.button("← Return to Step 2: Select Model", type="primary"):
        st.switch_page("pages/2_select_model.py")
else:
    # Show selected model
    from src.ui.components.model_builder import format_full_equation
    model_terms = st.session_state['model_terms']
    equation = format_full_equation(model_terms, "Y")
    
    with st.expander("📋 Selected Model (from Step 2)", expanded=False):
        st.markdown(f"**{equation}**")
        st.caption(f"{len(model_terms)} terms to estimate")
        
        # Analyze model complexity for recommendations
        has_quadratic = any(t.startswith('I(') and '**2' in t for t in model_terms)
        has_interactions = any('*' in t and not t.startswith('I(') for t in model_terms)
        
        if has_quadratic:
            st.info("💡 Your model includes quadratic terms - Response Surface designs (CCD/Box-Behnken) are recommended.")
        elif has_interactions:
            st.info("💡 Your model includes interactions - Full Factorial or D-Optimal designs work well.")
        else:
            st.info("💡 Your model is linear - Most design types will work.")

st.markdown(f"""
You have defined **{len(factors)} factors**. Now choose the design type that best suits your objectives.
""")

# Show factor summary
with st.expander("📋 View Defined Factors"):
    factor_summary = []
    for f in factors:
        if f.is_continuous():
            levels_str = f"[{f.levels[0]}, {f.levels[1]}]"
        else:
            levels_str = f"{len(f.levels)} levels"
        
        factor_summary.append({
            'Name': f.name,
            'Type': f.factor_type.value.replace('_', ' ').title(),
            'Levels': levels_str,
            'Changeability': f.changeability.value.title()
        })
    
    st.dataframe(pd.DataFrame(factor_summary), use_container_width=True, hide_index=True)

st.divider()

# Design type selection
st.subheader("Select Design Type")

# Determine available designs based on factors
all_continuous = all(f.is_continuous() for f in factors)
has_categorical = any(f.is_categorical() for f in factors)
has_hard_factors = any(f.changeability != ChangeabilityLevel.EASY for f in factors)

# Design type descriptions
design_options = {
    "Full Factorial": {
        "enabled": True,
        "description": "All possible combinations of factor levels. Best for small experiments.",
        "when_to_use": "2-4 factors, want to estimate all interactions",
        "runs": "2^k for 2-level factors (grows exponentially)"
    },
    "Fractional Factorial": {
        "enabled": len(factors) >= 4,
        "description": "Subset of full factorial. Efficient screening for many factors.",
        "when_to_use": "4+ factors, willing to sacrifice some interactions",
        "runs": "2^(k-p) where p is the fraction"
    },
    "Response Surface (CCD)": {
        "enabled": all_continuous and len(factors) >= 2,
        "description": "Central Composite Design for quadratic models and optimization.",
        "when_to_use": "Continuous factors, need to model curvature",
        "runs": "2^k + 2k + center points"
    },
    "Response Surface (Box-Behnken)": {
        "enabled": all_continuous and len(factors) >= 3,
        "description": "Efficient response surface design (no corner points).",
        "when_to_use": "3+ continuous factors, avoid extreme combinations",
        "runs": "Fewer than CCD, no axial points at ±α"
    },
    "D-Optimal": {
        "enabled": True,
        "description": "Computer-generated optimal design with constraints.",
        "when_to_use": "Constrained design space, irregular regions, mixed factors",
        "runs": "User-specified (typically p+1 to 2p where p=parameters)"
    },
    "Latin Hypercube": {
        "enabled": all_continuous,
        "description": "Space-filling design for exploration and screening.",
        "when_to_use": "Initial exploration, many factors, computer experiments",
        "runs": "User-specified (flexible)"
    },
    "Split-Plot": {
        "enabled": has_hard_factors,
        "description": "Hierarchical design for hard-to-change factors.",
        "when_to_use": "Some factors are expensive or slow to change",
        "runs": "Based on whole-plot structure"
    }
}

# Display design options
design_choice = None
current_design = st.session_state.get('design_type')

for design_name, design_info in design_options.items():
    if design_info["enabled"]:
        # Check if this is the selected design
        is_selected = (design_name == current_design)
        
        # Customize expander label based on selection
        if is_selected:
            expander_label = f"✅ {design_name} (Currently Selected)"
        else:
            expander_label = f"✓ {design_name}"
        
        with st.expander(expander_label, expanded=is_selected):
            st.markdown(f"**{design_info['description']}**")
            st.markdown(f"**When to use:** {design_info['when_to_use']}")
            st.markdown(f"**Typical runs:** {design_info['runs']}")
            
            if is_selected:
                st.info("🎯 This design is currently selected. Modify configuration below or choose a different design.")
            
            if st.button(f"Select {design_name}", key=f"select_{design_name}", disabled=is_selected, type="primary" if not is_selected else "secondary"):
                design_choice = design_name
    else:
        with st.expander(f"🔒 {design_name} (Not Available)", expanded=False):
            st.markdown(f"**{design_info['description']}**")
            
            # Explain why not available
            if design_name == "Fractional Factorial" and len(factors) < 4:
                st.warning("Requires at least 4 factors")
            elif "Response Surface" in design_name and not all_continuous:
                st.warning("Requires all continuous factors")
            elif design_name == "Response Surface (Box-Behnken)" and len(factors) < 3:
                st.warning("Requires at least 3 factors")
            elif design_name == "Split-Plot" and not has_hard_factors:
                st.warning("Requires at least one hard-to-change factor")
            elif design_name == "Latin Hypercube" and not all_continuous:
                st.warning("Requires all continuous factors")

# If user selected a design, show configuration
if design_choice:
    st.session_state['design_type'] = design_choice
    invalidate_downstream_state(from_step=2)
    st.rerun()

# If design already selected, show configuration
if st.session_state.get('design_type'):
    st.divider()
    st.subheader(f"⚙️ Configure {st.session_state['design_type']}")
    
    design_type = st.session_state['design_type']
    
    # Configuration forms for each design type
    if design_type == "Full Factorial":
        st.markdown("**Full Factorial Configuration**")
        
        # Number of levels per factor (for continuous/discrete)
        n_levels = st.radio(
            "Number of Levels",
            [2, 3],
            help="2-level: factorial corners only. 3-level: includes midpoints.",
            horizontal=True
        )
        
        n_center_points = st.number_input(
            "Number of Center Points",
            min_value=0,
            max_value=10,
            value=3,
            help="Center points estimate pure error and check curvature"
        )
        
        n_replicates = st.number_input(
            "Number of Replicates",
            min_value=1,
            max_value=10,
            value=1,
            help="Full repetitions of entire design"
        )
        
        randomize = st.checkbox("Randomize Run Order", value=True)
        
        # Store config
        st.session_state['design_config'] = {
            'n_levels': n_levels,
            'n_center_points': n_center_points,
            'n_replicates': n_replicates,
            'randomize': randomize
        }
        
        # Estimate runs
        total_runs = (n_levels ** len(factors)) * n_replicates + n_center_points
        st.info(f"**Estimated runs:** {total_runs}")
    
    elif design_type == "Fractional Factorial":
        st.markdown("**Fractional Factorial Configuration**")
        
        # Check model compatibility
        if model_selected:
            model_terms = st.session_state['model_terms']
            has_quadratic = any(t.startswith('I(') and '**2' in t for t in model_terms)
            
            if has_quadratic:
                st.error(
                    "⚠️ **Model Incompatibility:** Your selected model includes quadratic terms, "
                    "but fractional factorial designs cannot estimate quadratic effects.\n\n"
                    "**Recommendations:**\n"
                    "1. Use Response Surface design (CCD or Box-Behnken) instead, OR\n"
                    "2. Simplify model to remove quadratic terms (return to Step 2), OR\n"
                    "3. Proceed anyway and modify model during analysis (Step 6)"
                )
        
        k = len(factors)
        
        # Fraction selection
        fractions = ["1/2", "1/4", "1/8", "1/16"]
        valid_fractions = []
        
        for frac in fractions:
            p = int(frac.split('/')[1]).bit_length() - 1
            if k - p >= 3:
                valid_fractions.append(frac)
        
        fraction = st.selectbox(
            "Fraction Size",
            valid_fractions,
            help="Smaller fractions = fewer runs but more aliasing"
        )
        
        # Resolution
        resolution = st.selectbox(
            "Resolution",
            [3, 4, 5],
            index=2,
            help="Higher resolution = less aliasing. V is best, III is minimum."
        )
        
        # Generator specification
        generator_mode = st.radio(
            "Generator Specification",
            ["Standard (Recommended)", "Custom"],
            help="Standard generators from Montgomery/Box-Hunter-Hunter"
        )
        
        if generator_mode == "Custom":
            st.warning("Custom generators require knowledge of alias structure")
            
            p = int(fraction.split('/')[1]).bit_length() - 1
            generators_input = st.text_area(
                f"Generators (one per line, need {p})",
                placeholder="E=ABCD\nF=ABC",
                help="Format: NewFactor=Expression (e.g., E=ABCD)"
            )
            
            custom_generators = [g.strip() for g in generators_input.split('\n') if g.strip()]
        else:
            custom_generators = None
        
        n_center_points = st.number_input(
            "Number of Center Points",
            min_value=0,
            max_value=10,
            value=3
        )
        
        randomize = st.checkbox("Randomize Run Order", value=True)
        
        # Store config
        st.session_state['design_config'] = {
            'fraction': fraction,
            'resolution': resolution,
            'generator_mode': generator_mode,
            'custom_generators': custom_generators,
            'n_center_points': n_center_points,
            'randomize': randomize
        }
        
        # Estimate runs
        p = int(fraction.split('/')[1]).bit_length() - 1
        total_runs = 2 ** (k - p) + n_center_points
        st.info(f"**Estimated runs:** {total_runs}")
    
    elif design_type in ["Response Surface (CCD)", "Response Surface (Box-Behnken)"]:
        st.markdown(f"**{design_type} Configuration**")
        
        if "CCD" in design_type:
            alpha = st.selectbox(
                "Axial Distance (α)",
                ["Face-centered (α=1)", "Orthogonal", "Rotatable"],
                help="α determines axial point distance from center"
            )
            
            # Parse alpha
            if "Face-centered" in alpha:
                alpha_value = 1.0
            elif "Orthogonal" in alpha:
                # Calculate orthogonal alpha
                k = len(factors)
                alpha_value = (2 ** k) ** 0.25
            else:  # Rotatable
                k = len(factors)
                alpha_value = k ** 0.5
            
            st.session_state['design_config'] = {
                'alpha': alpha_value,
                'alpha_type': alpha
            }
        else:
            st.session_state['design_config'] = {}
        
        n_center_points = st.number_input(
            "Number of Center Points",
            min_value=1,
            max_value=10,
            value=5,
            help="RSM designs need multiple center points"
        )
        
        st.session_state['design_config']['n_center_points'] = n_center_points
        
        randomize = st.checkbox("Randomize Run Order", value=True)
        st.session_state['design_config']['randomize'] = randomize
        
        # Estimate runs
        k = len(factors)
        if "CCD" in design_type:
            factorial_runs = 2 ** k
            axial_runs = 2 * k
            total_runs = factorial_runs + axial_runs + n_center_points
        else:  # Box-Behnken
            # Approximate formula
            total_runs = 2 * k * (k - 1) + n_center_points
        
        st.info(f"**Estimated runs:** {total_runs}")
    
    elif design_type == "D-Optimal":
        st.markdown("**D-Optimal Design Configuration**")
        
        # Get model terms from Step 2
        if 'model_terms' not in st.session_state or not st.session_state['model_terms']:
            st.error("⚠️ No model defined. Please return to Step 2 to select analysis model.")
            if st.button("← Go to Step 2: Select Model", type="primary"):
                st.switch_page("pages/2_select_model.py")
            st.stop()
        
        model_terms = st.session_state['model_terms']
        
        # Display selected model
        from src.ui.components.model_builder import format_full_equation
        equation = format_full_equation(model_terms, "Y")
        
        st.info(
            f"🎯 **Selected Model (from Step 2):**\n\n"
            f"{equation}\n\n"
            f"This model has **{len(model_terms)} terms** to estimate. "
            f"D-optimal design will be optimized for these exact terms."
        )
        
        if st.button("✏️ Edit Model", key="edit_model_button"):
            st.switch_page("pages/2_select_model.py")
        
        st.divider()
        
        # Number of runs
        min_runs = len(model_terms)
        
        n_runs = st.number_input(
            f"Number of Runs (minimum: {min_runs})",
            min_value=min_runs,
            max_value=min_runs * 5,
            value=min_runs * 2,
            help="More runs = better precision"
        )
        
        st.divider()
        
        # Constraints (full implementation)
        show_constraint_builder(factors)
        show_constraint_help()
        
        # Store config with constraints
        constraints = st.session_state.get('constraints', [])
        
        st.session_state['design_config'] = {
            'n_runs': n_runs,
            'n_constraints': len(constraints),
            'model_terms': model_terms  # Store for preview/generation
        }
        
        if constraints:
            st.info(f"**Number of runs:** {n_runs} | **Constraints:** {len(constraints)}")
        else:
            st.info(f"**Number of runs:** {n_runs}")
    
    elif design_type == "Latin Hypercube":
        st.markdown("**Latin Hypercube Sampling Configuration**")
        
        n_runs = st.number_input(
            "Number of Runs",
            min_value=len(factors) + 1,
            max_value=1000,
            value=len(factors) * 10,
            help="Typically 5-10 times number of factors"
        )
        
        criterion = st.selectbox(
            "Optimization Criterion",
            ["None", "Maximin", "Correlation"],
            help="Criterion for optimizing space-filling"
        )
        
        st.session_state['design_config'] = {
            'n_runs': n_runs,
            'criterion': criterion
        }
        
        st.info(f"**Number of runs:** {n_runs}")
    
    elif design_type == "Split-Plot":
        st.markdown("**Split-Plot Design Configuration**")
        
        # Show which factors are hard/easy
        hard_factors = [f.name for f in factors if f.changeability == ChangeabilityLevel.HARD]
        very_hard_factors = [f.name for f in factors if f.changeability == ChangeabilityLevel.VERY_HARD]
        easy_factors = [f.name for f in factors if f.changeability == ChangeabilityLevel.EASY]
        
        st.info(
            f"**Hard-to-change factors:** {', '.join(hard_factors + very_hard_factors)}\n\n"
            f"**Easy-to-change factors:** {', '.join(easy_factors)}"
        )
        
        n_whole_plots = st.number_input(
            "Number of Whole-Plots",
            min_value=2,
            max_value=32,
            value=4,
            help="Number of settings for hard-to-change factors"
        )
        
        n_subplot_runs = st.number_input(
            "Runs per Whole-Plot",
            min_value=2,
            max_value=16,
            value=4,
            help="Combinations of easy factors at each whole-plot setting"
        )
        
        randomize_whole_plots = st.checkbox("Randomize Whole-Plot Order", value=True)
        randomize_subplots = st.checkbox("Randomize Sub-Plot Order", value=True)
        
        st.session_state['design_config'] = {
            'n_whole_plots': n_whole_plots,
            'n_subplot_runs': n_subplot_runs,
            'randomize_whole_plots': randomize_whole_plots,
            'randomize_subplots': randomize_subplots
        }
        
        total_runs = n_whole_plots * n_subplot_runs
        st.info(f"**Total runs:** {total_runs}")
    
    # Generate button
    st.divider()
    
    if st.button("Generate Design →", type="primary", use_container_width=True):
        st.session_state['current_step'] = 4
        st.switch_page("pages/4_preview_design.py")

# Navigation
st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("← Back to Model", use_container_width=True):
        st.session_state['current_step'] = 2
        st.switch_page("pages/2_select_model.py")

with col2:
    if st.session_state.get('design_type') and st.session_state.get('design_config'):
        if st.button("Preview Design →", use_container_width=True):
            st.session_state['current_step'] = 4
            st.switch_page("pages/4_preview_design.py")