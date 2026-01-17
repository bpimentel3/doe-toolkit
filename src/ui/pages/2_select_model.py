"""
Step 2: Select Analysis Model

Define the statistical model BEFORE choosing the design type.
This ensures D-optimal and other model-dependent designs know the correct structure.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.ui.utils.state_management import (
    initialize_session_state,
    is_step_complete,
    can_access_step,
    invalidate_downstream_state
)
from src.ui.components.model_builder import display_model_builder

# ==================== PAGE CONFIG ====================

st.set_page_config(
    page_title="Select Model - DOE Toolkit",
    page_icon="🔧",
    layout="wide"
)

# Initialize session state
initialize_session_state()

# ==================== ACCESS CONTROL ====================

if not can_access_step(2):
    st.error("⚠️ Complete Step 1 (Define Factors) first")
    st.stop()

# ==================== MAIN PAGE ====================

st.title("🔧 Step 2: Select Analysis Model")

st.markdown("""
### Why Define the Model First?

Different design types support different models:
- **D-optimal designs** require knowing the model upfront (they optimize for those specific terms)
- **Fractional factorial designs** may alias certain interactions
- **Full factorial designs** can fit any model up to full interactions

Define your anticipated model now to:
1. Enable design generation that supports your analysis goals
2. Get warnings if a design type can't fit your model
3. Ensure efficient experimental designs
""")

st.info("""
💡 **Don't worry if you're unsure!** You can always simplify the model during analysis or use 
augmentation to upgrade your design later.
""")

# ==================== MODEL DEFINITION ====================

# Get factors from state
factors = st.session_state.get('factors', [])

if not factors:
    st.warning("⚠️ No factors defined. Return to Step 1.")
    st.stop()

# Display current factors
with st.expander("📋 Defined Factors", expanded=False):
    factor_data = []
    for f in factors:
        factor_type = (
            "Continuous" if f.is_continuous() else
            "Discrete" if f.factor_type.name == "DISCRETE_NUMERIC" else
            "Categorical"
        )
        if f.is_continuous():
            levels_str = f"[{f.continuous_range[0]}, {f.continuous_range[1]}]"
        else:
            levels_str = f"{len(f.levels)} levels"
        
        factor_data.append({
            "Factor": f.name,
            "Type": factor_type,
            "Levels": levels_str,
            "Changeability": f.changeability.name.replace('_', ' ').title()
        })
    
    import pandas as pd
    st.dataframe(
        pd.DataFrame(factor_data),
        use_container_width=True,
        hide_index=True
    )

st.markdown("---")

# ==================== MODEL BUILDER ====================

# Initialize model terms if not present
if 'model_terms' not in st.session_state or st.session_state['model_terms'] is None:
    # Default: Linear model with intercept
    st.session_state['model_terms'] = ['1'] + [f.name for f in factors]

# Get current terms
current_terms = st.session_state.get('model_terms', ['1'])

# Display model builder
updated_terms = display_model_builder(
    factors=factors,
    current_terms=current_terms,
    response_name="Y",
    key_prefix="step2"
)

# Update if changed
if updated_terms != current_terms:
    st.session_state['model_terms'] = updated_terms
    # Invalidate downstream steps since model changed
    invalidate_downstream_state(current_step=2)
    st.rerun()

# ==================== MODEL COMPATIBILITY GUIDANCE ====================

st.markdown("---")

st.subheader("📊 Design Type Recommendations")

# Analyze model to suggest best designs
terms = st.session_state['model_terms']
has_intercept = '1' in terms
has_main_effects = any(t != '1' and '*' not in t and not t.startswith('I(') for t in terms)
has_interactions = any('*' in t and not t.startswith('I(') for t in terms)
has_quadratic = any(t.startswith('I(') and '**2' in t for t in terms)
has_cubic_or_higher = any(t.startswith('I(') and '**' in t and '**2' not in t for t in terms)

n_factors = len(factors)
continuous_factors = [f for f in factors if f.is_continuous()]
n_continuous = len(continuous_factors)

# Determine model complexity
if has_cubic_or_higher:
    model_complexity = "Very High (Cubic+ terms)"
elif has_quadratic:
    model_complexity = "High (Quadratic)"
elif has_interactions:
    model_complexity = "Medium (Interactions)"
else:
    model_complexity = "Low (Main effects only)"

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("Model Complexity", model_complexity)
    st.metric("Terms to Estimate", len(terms))
    st.metric("Minimum Runs Needed", len(terms))

with col2:
    st.markdown("**✅ Compatible Design Types:**")
    
    compatible = []
    warnings = []
    
    # Full Factorial
    if n_factors <= 5:
        compatible.append("✅ **Full Factorial** - Can fit any model")
    else:
        warnings.append("⚠️ **Full Factorial** - May need many runs (2^{})".format(n_factors))
    
    # Fractional Factorial
    if not has_quadratic and n_factors >= 4:
        compatible.append("✅ **Fractional Factorial** - Check resolution for interactions")
    elif has_quadratic:
        warnings.append("❌ **Fractional Factorial** - Cannot fit quadratic terms")
    
    # Response Surface (CCD, Box-Behnken)
    if has_quadratic and n_continuous >= 2:
        compatible.append("✅ **Response Surface (CCD, Box-Behnken)** - Designed for quadratic models")
    elif has_quadratic and n_continuous < 2:
        warnings.append("⚠️ **Response Surface** - Requires 2+ continuous factors")
    elif not has_quadratic:
        warnings.append("⚠️ **Response Surface** - Overqualified (no quadratic terms in model)")
    
    # D-Optimal
    compatible.append("✅ **D-Optimal** - Can fit this exact model efficiently")
    
    # Latin Hypercube
    if not has_interactions and not has_quadratic:
        compatible.append("✅ **Latin Hypercube** - Good for main effects screening")
    else:
        warnings.append("⚠️ **Latin Hypercube** - Better for screening, not interaction/RSM models")
    
    for item in compatible:
        st.markdown(item)
    
    if warnings:
        st.markdown("**Compatibility Notes:**")
        for item in warnings:
            st.markdown(item)

# ==================== MODEL SUMMARY ====================

st.markdown("---")

st.subheader("📝 Model Summary")

from src.ui.components.model_builder import format_full_equation

equation = format_full_equation(terms, "Y")

st.markdown(f"""
<div style='background-color: #f0f0f0; padding: 20px; border-radius: 5px; border-left: 4px solid #1f77b4;'>
<p style='font-size: 1.8em; margin: 0; text-align: center;'>
<em>{equation}</em>
</p>
</div>
""", unsafe_allow_html=True)

st.caption("""
This model will be used to:
- Guide design type selection (next step)
- Pre-populate analysis model (Step 6)
- Enable D-optimal design generation with correct structure
""")

# ==================== MODEL TERM BREAKDOWN ====================

with st.expander("🔍 Term Details", expanded=False):
    intercept_terms = [t for t in terms if t == '1']
    main_effects = [t for t in terms if t != '1' and '*' not in t and not t.startswith('I(')]
    interactions = [t for t in terms if '*' in t and not t.startswith('I(')]
    powers = [t for t in terms if t.startswith('I(') and '*' not in t]
    power_interactions = [t for t in terms if t.startswith('I(') and '*' in t]
    
    from src.ui.components.model_builder import format_term_for_display
    
    if intercept_terms:
        st.markdown("**Intercept:**")
        for t in intercept_terms:
            st.write(f"  • {format_term_for_display(t)}")
    
    if main_effects:
        st.markdown("**Main Effects:**")
        for t in main_effects:
            st.write(f"  • {format_term_for_display(t)}")
    
    if interactions:
        st.markdown("**Interactions:**")
        for t in interactions:
            st.write(f"  • {format_term_for_display(t)}")
    
    if powers:
        st.markdown("**Power Terms:**")
        for t in powers:
            st.write(f"  • {format_term_for_display(t)}")
    
    if power_interactions:
        st.markdown("**Power Interactions:**")
        for t in power_interactions:
            st.write(f"  • {format_term_for_display(t)}")

# ==================== NAVIGATION ====================

st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    if st.button("← Back to Factors", use_container_width=True):
        st.switch_page("pages/1_define_factors.py")

with col3:
    # Model selection complete if we have at least one term
    if len(terms) > 0:
        st.session_state['step_2_complete'] = True
        if st.button("Next: Choose Design →", use_container_width=True, type="primary"):
            st.switch_page("pages/3_choose_design.py")
    else:
        st.warning("Select at least one model term to continue")
