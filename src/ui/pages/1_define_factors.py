"""
Step 1: Define Experimental Factors

Define factors, their types, levels, and changeability.
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
    invalidate_downstream_state
)
from src.core.factors import Factor, FactorType, ChangeabilityLevel

# Initialize state
initialize_session_state()

st.title("Step 1: Define Experimental Factors")

st.markdown("""
Define the factors (independent variables) you'll manipulate in your experiment.
Each factor needs:
- **Name**: Descriptive identifier (e.g., "Temperature", "Material")
- **Type**: Continuous, Discrete Numeric, or Categorical
- **Levels**: Range or discrete values
- **Changeability**: How easy it is to change between runs
""")

st.divider()

# Initialize factors list
if 'factors' not in st.session_state:
    st.session_state['factors'] = []

factors = st.session_state['factors']

# Display current factors
if factors:
    st.subheader("Current Factors")
    
    # Build display table
    factor_data = []
    for i, factor in enumerate(factors):
        if factor.is_continuous():
            levels_str = f"[{factor.levels[0]}, {factor.levels[1]}]"
        else:
            levels_str = ", ".join(str(l) for l in factor.levels)
        
        factor_data.append({
            'Index': i,
            'Name': factor.name,
            'Type': factor.factor_type.value.replace('_', ' ').title(),
            'Levels': levels_str,
            'Units': factor.units or '—',
            'Changeability': factor.changeability.value.title()
        })
    
    factors_df = pd.DataFrame(factor_data)
    
    # Display as table
    st.dataframe(factors_df.drop('Index', axis=1), use_container_width=True)
    
    st.divider()
    
    # Factor management
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("➕ Add Factor", type="primary", use_container_width=True):
            st.session_state['add_factor_mode'] = True
    
    with col2:
        # Edit factor
        if len(factors) > 0:
            factor_to_edit = st.selectbox(
                "Select Factor to Edit",
                options=range(len(factors)),
                format_func=lambda i: factors[i].name,
                key='edit_select'
            )
            
            if st.button("✏️ Edit Selected", use_container_width=True):
                st.session_state['edit_factor_mode'] = True
                st.session_state['edit_factor_idx'] = factor_to_edit
    
    with col3:
        # Delete factor
        if len(factors) > 0:
            factor_to_delete = st.selectbox(
                "Select Factor to Delete",
                options=range(len(factors)),
                format_func=lambda i: factors[i].name,
                key='delete_select'
            )
            
            if st.button("🗑️ Delete Selected", use_container_width=True):
                del st.session_state['factors'][factor_to_delete]
                invalidate_downstream_state(from_step=1)
                st.rerun()

else:
    st.info("No factors defined yet. Click 'Add Factor' to get started.")
    
    if st.button("➕ Add First Factor", type="primary", use_container_width=True):
        st.session_state['add_factor_mode'] = True

st.divider()

# Add/Edit Factor Form
if st.session_state.get('add_factor_mode') or st.session_state.get('edit_factor_mode'):
    
    if st.session_state.get('edit_factor_mode'):
        st.subheader("✏️ Edit Factor")
        edit_idx = st.session_state['edit_factor_idx']
        existing_factor = factors[edit_idx]
    else:
        st.subheader("➕ Add New Factor")
        existing_factor = None
    
    with st.form("factor_form"):
        # Factor name
        factor_name = st.text_input(
            "Factor Name*",
            value=existing_factor.name if existing_factor else "",
            placeholder="e.g., Temperature, Pressure, Material",
            help="Use a descriptive, unique name"
        )
        
        # Factor type
        factor_type = st.selectbox(
            "Factor Type*",
            options=[
                FactorType.CONTINUOUS,
                FactorType.DISCRETE_NUMERIC,
                FactorType.CATEGORICAL
            ],
            format_func=lambda x: x.value.replace('_', ' ').title(),
            index=(
                [FactorType.CONTINUOUS, FactorType.DISCRETE_NUMERIC, FactorType.CATEGORICAL]
                .index(existing_factor.factor_type)
                if existing_factor else 0
            )
        )
        
        # Changeability
        changeability = st.radio(
            "Changeability*",
            options=[
                ChangeabilityLevel.EASY,
                ChangeabilityLevel.HARD,
                ChangeabilityLevel.VERY_HARD
            ],
            format_func=lambda x: {
                ChangeabilityLevel.EASY: "Easy (can change every run)",
                ChangeabilityLevel.HARD: "Hard (change infrequently - defines whole-plots)",
                ChangeabilityLevel.VERY_HARD: "Very Hard (rarely changed - defines whole-whole-plots)"
            }[x],
            index=(
                [ChangeabilityLevel.EASY, ChangeabilityLevel.HARD, ChangeabilityLevel.VERY_HARD]
                .index(existing_factor.changeability)
                if existing_factor else 0
            ),
            horizontal=True,
            help="Important for split-plot designs. Most factors are 'Easy'."
        )
        
        # Units (optional)
        units = st.text_input(
            "Units (optional)",
            value=existing_factor.units if existing_factor and existing_factor.units else "",
            placeholder="e.g., °C, psi, minutes"
        )
        
        st.markdown("---")
        
        # Levels definition (varies by type)
        if factor_type == FactorType.CONTINUOUS:
            st.markdown("**Continuous Factor - Define Range**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                min_val = st.number_input(
                    "Minimum Value*",
                    value=float(existing_factor.levels[0]) if existing_factor else 0.0,
                    format="%.4f"
                )
            
            with col2:
                max_val = st.number_input(
                    "Maximum Value*",
                    value=float(existing_factor.levels[1]) if existing_factor else 10.0,
                    format="%.4f"
                )
            
            levels = [min_val, max_val]
        
        elif factor_type == FactorType.DISCRETE_NUMERIC:
            st.markdown("**Discrete Numeric - Define Specific Values**")
            
            levels_input = st.text_input(
                "Levels (comma-separated)*",
                value=(
                    ", ".join(str(l) for l in existing_factor.levels)
                    if existing_factor else ""
                ),
                placeholder="e.g., 100, 150, 200, 250"
            )
            
            # Parse levels
            if levels_input:
                try:
                    levels = [float(x.strip()) for x in levels_input.split(',')]
                except ValueError:
                    st.error("All levels must be numeric")
                    levels = []
            else:
                levels = []
        
        else:  # CATEGORICAL
            st.markdown("**Categorical Factor - Define Categories**")
            
            levels_input = st.text_input(
                "Levels (comma-separated)*",
                value=(
                    ", ".join(str(l) for l in existing_factor.levels)
                    if existing_factor else ""
                ),
                placeholder="e.g., Material_A, Material_B, Material_C"
            )
            
            # Parse levels
            if levels_input:
                levels = [x.strip() for x in levels_input.split(',')]
            else:
                levels = []
        
        # Form submission
        col1, col2 = st.columns([1, 1])
        
        with col1:
            submitted = st.form_submit_button("✓ Save Factor", type="primary", use_container_width=True)
        
        with col2:
            cancelled = st.form_submit_button("✗ Cancel", use_container_width=True)
        
        if submitted:
            # Validate
            errors = []
            
            if not factor_name or not factor_name.strip():
                errors.append("Factor name is required")
            
            if not levels:
                errors.append("At least one level is required")
            
            if factor_type == FactorType.CONTINUOUS:
                if min_val >= max_val:
                    errors.append("Maximum must be greater than minimum")
            
            elif factor_type in [FactorType.DISCRETE_NUMERIC, FactorType.CATEGORICAL]:
                if len(levels) < 2:
                    errors.append("At least 2 levels required")
            
            # Check for duplicate name (if adding or renaming)
            if existing_factor:
                # Editing - check duplicates excluding self
                other_names = [f.name for i, f in enumerate(factors) if i != edit_idx]
            else:
                # Adding - check all names
                other_names = [f.name for f in factors]
            
            if factor_name in other_names:
                errors.append(f"Factor name '{factor_name}' already exists")
            
            if errors:
                for error in errors:
                    st.error(error)
            else:
                # Create/update factor
                try:
                    new_factor = Factor(
                        name=factor_name.strip(),
                        factor_type=factor_type,
                        changeability=changeability,
                        levels=levels,
                        units=units.strip() if units else None
                    )
                    
                    if existing_factor:
                        # Update existing
                        st.session_state['factors'][edit_idx] = new_factor
                        st.success(f"✓ Factor '{factor_name}' updated")
                    else:
                        # Add new
                        st.session_state['factors'].append(new_factor)
                        st.success(f"✓ Factor '{factor_name}' added")
                    
                    # Clear modes
                    st.session_state['add_factor_mode'] = False
                    st.session_state['edit_factor_mode'] = False
                    
                    # Invalidate downstream
                    invalidate_downstream_state(from_step=1)
                    
                    st.rerun()
                
                except ValueError as e:
                    st.error(f"Validation error: {e}")
        
        if cancelled:
            st.session_state['add_factor_mode'] = False
            st.session_state['edit_factor_mode'] = False
            st.rerun()

# Navigation and info
st.divider()

col1, col2 = st.columns([3, 1])

with col1:
    # Tips
    with st.expander("💡 Factor Definition Tips"):
        st.markdown("""
        **Continuous Factors:**
        - Use for quantitative variables with any value in a range
        - Examples: Temperature (150-200°C), pH (4-8), Concentration (10-30%)
        - Enables response surface modeling and optimization
        
        **Discrete Numeric Factors:**
        - Use for quantitative variables with specific allowed values
        - Examples: RPM (100, 150, 200), Batch Size (5, 10, 15 kg)
        - Limited to the exact values specified
        
        **Categorical Factors:**
        - Use for qualitative variables
        - Examples: Material (A, B, C), Supplier (Vendor1, Vendor2), Method (Old, New)
        - Cannot be ordered or interpolated
        
        **Changeability:**
        - **Easy**: Most factors - can be changed every run
        - **Hard**: Defines whole-plots (e.g., oven temperature if multiple batches per oven setting)
        - **Very Hard**: Rare - defines whole-whole-plots (nested structure)
        """)

with col2:
    if len(factors) >= 2:
        if st.button("Continue to Design Selection →", type="primary", use_container_width=True):
            st.session_state['current_step'] = 2
            st.switch_page("pages/2_choose_design.py")
    else:
        st.info("Define at least 2 factors to continue")