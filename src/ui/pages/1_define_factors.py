"""
Step 1: Define Experimental Factors (JMP-Style Table Editor)

NEW: In-place editable table with dropdowns, similar to JMP interface
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from typing import List, Dict, Optional

from src.ui.utils.state_management import (
    initialize_session_state,
    invalidate_downstream_state
)
from src.core.factors import (
    Factor, 
    FactorType, 
    ChangeabilityLevel,
    sanitize_factor_name,
    get_sanitization_report
)

# Initialize state
initialize_session_state()

st.title("Step 1: Define Experimental Factors")

st.markdown("""
Define your experimental factors in the table below. Edit cells directly, similar to JMP.
""")

st.divider()

# Initialize factors if empty
if 'factors' not in st.session_state or st.session_state['factors'] is None:
    st.session_state['factors'] = []

# Convert factors to DataFrame for editing
def factors_to_dataframe(factors: List[Factor]) -> pd.DataFrame:
    """Convert factor list to editable DataFrame."""
    if not factors:
        # Create empty DataFrame with correct dtypes
        return pd.DataFrame({
            'Name': pd.Series([], dtype='str'),
            'Type': pd.Series([], dtype='str'),
            'Min': pd.Series([], dtype='float64'),
            'Max': pd.Series([], dtype='float64'),
            'Levels': pd.Series([], dtype='str'),
            'Units': pd.Series([], dtype='str'),
            'Changeability': pd.Series([], dtype='str')
        })
    
    rows = []
    for factor in factors:
        if factor.is_continuous():
            min_val = factor.levels[0]
            max_val = factor.levels[1]
            levels_str = ''
        else:
            min_val = None
            max_val = None
            levels_str = ', '.join(str(l) for l in factor.levels)
        
        rows.append({
            'Name': str(factor.name),  # Ensure it's a plain string
            'Type': str(factor.factor_type.value),
            'Min': min_val,
            'Max': max_val,
            'Levels': str(levels_str),
            'Units': str(factor.units) if factor.units else '',
            'Changeability': str(factor.changeability.value)
        })
    
    df = pd.DataFrame(rows)
    # Ensure string columns are proper string dtype
    df['Name'] = df['Name'].astype(str)
    df['Type'] = df['Type'].astype(str)
    df['Levels'] = df['Levels'].astype(str)
    df['Units'] = df['Units'].astype(str)
    df['Changeability'] = df['Changeability'].astype(str)
    return df


def dataframe_to_factors(df: pd.DataFrame) -> tuple[List[Factor], List[str]]:
    """
    Convert edited DataFrame back to Factor objects.
    
    Returns
    -------
    factors : List[Factor]
        Valid factors
    errors : List[str]
        Validation errors
    """
    factors = []
    errors = []
    sanitization_warnings = []
    
    for idx, row in df.iterrows():
        try:
            # Sanitize name - handle various string representations
            name_raw = str(row['Name']).strip()
            
            # Remove brackets/quotes that might be added by pandas or string conversion
            # Common artifacts: ['Name'], ["Name"], "['Name']"
            name_raw = name_raw.strip("[]'\"")
            name_raw = name_raw.strip()  # Strip again after bracket removal
            
            if not name_raw or name_raw.lower() in ['nan', 'none', '']:
                errors.append(f"Row {idx+1}: Name cannot be empty")
                continue
            
            # Clean name for comparison
            clean_name, was_modified = sanitize_factor_name(name_raw)
            
            # Only show sanitization warning if actually modified
            # (not just from pandas string conversion artifacts)
            if was_modified:
                # Double-check: was it only artifacts being stripped?
                recheck_name = name_raw.strip("[]'\"")
                _, recheck_modified = sanitize_factor_name(recheck_name)
                
                if recheck_modified:
                    report = get_sanitization_report(name_raw)
                    sanitization_warnings.append({
                        'row': idx + 1,
                        'original': name_raw,
                        'sanitized': clean_name,
                        'changes': report['changes']
                    })
            
            # Parse type
            factor_type_str = str(row['Type']).strip()
            # Remove brackets/quotes that might be added by pandas
            factor_type_str = factor_type_str.strip("[]'\"").lower()
            
            if factor_type_str == 'continuous':
                factor_type = FactorType.CONTINUOUS
            elif factor_type_str == 'discrete_numeric':
                factor_type = FactorType.DISCRETE_NUMERIC
            elif factor_type_str == 'categorical':
                factor_type = FactorType.CATEGORICAL
            else:
                errors.append(f"Row {idx+1}: Invalid type '{row['Type']}' (cleaned: '{factor_type_str}')")
                continue
            
            # Parse levels
            if factor_type == FactorType.CONTINUOUS:
                min_val = row['Min']
                max_val = row['Max']
                
                if pd.isna(min_val) or pd.isna(max_val):
                    errors.append(f"Row {idx+1} ({clean_name}): Min and Max required for continuous")
                    continue
                
                try:
                    min_val = float(min_val)
                    max_val = float(max_val)
                except:
                    errors.append(f"Row {idx+1} ({clean_name}): Min/Max must be numeric")
                    continue
                
                if min_val >= max_val:
                    errors.append(f"Row {idx+1} ({clean_name}): Max must be > Min")
                    continue
                
                levels = [min_val, max_val]
                # Ignore Levels column for continuous factors
            
            else:
                # Discrete or categorical - ignore Min/Max columns
                levels_str = str(row['Levels']).strip()
                # Remove brackets/quotes that might be added by pandas
                levels_str = levels_str.strip("[]'\"")
                
                if not levels_str or levels_str.lower() in ['nan', 'none', '']:
                    errors.append(f"Row {idx+1} ({clean_name}): Levels required for {factor_type.value}")
                    continue
                
                # Parse comma-separated
                levels = [l.strip() for l in levels_str.split(',')]
                
                if len(levels) < 2:
                    errors.append(f"Row {idx+1} ({clean_name}): At least 2 levels required")
                    continue
                
                # Convert to numeric for discrete_numeric
                if factor_type == FactorType.DISCRETE_NUMERIC:
                    try:
                        levels = [float(l) for l in levels]
                    except:
                        errors.append(f"Row {idx+1} ({clean_name}): Levels must be numeric for discrete_numeric")
                        continue
            
            # Parse changeability
            change_str = str(row['Changeability']).strip()
            # Remove brackets/quotes that might be added by pandas
            change_str = change_str.strip("[]'\"").lower()
            
            if change_str == 'easy':
                changeability = ChangeabilityLevel.EASY
            elif change_str == 'hard':
                changeability = ChangeabilityLevel.HARD
            elif change_str == 'very_hard':
                changeability = ChangeabilityLevel.VERY_HARD
            else:
                changeability = ChangeabilityLevel.EASY  # Default
            
            # Units
            units = str(row['Units']).strip()
            # Remove brackets/quotes
            units = units.strip("[]'\"")
            units = units if units and units.lower() not in ['nan', 'none', ''] else None
            
            # Create factor
            factor = Factor(
                name=clean_name,
                factor_type=factor_type,
                changeability=changeability,
                levels=levels,
                units=units if units else None
            )
            
            factors.append(factor)
        
        except Exception as e:
            errors.append(f"Row {idx+1}: {str(e)}")
    
    # Show sanitization warnings
    if sanitization_warnings:
        with st.expander("⚠️ Factor Name Sanitization", expanded=True):
            st.warning(
                f"**{len(sanitization_warnings)} factor name(s) were modified for compatibility:**"
            )
            
            for warning in sanitization_warnings:
                st.markdown(f"**Row {warning['row']}:**")
                st.markdown(f"- Original: `{warning['original']}`")
                st.markdown(f"- Sanitized: `{warning['sanitized']}`")
                
                if warning['changes']:
                    with st.expander(f"Why was '{warning['original']}' changed?"):
                        for change in warning['changes']:
                            st.caption(f"• {change}")
    
    return factors, errors


# Get current factors as DataFrame
factors_df = factors_to_dataframe(st.session_state['factors'])

# Editable table
st.subheader("Factor Table")

st.info(
    "💡 **Quick Tips:**\n"
    "- Use the **+ button at the bottom** of the table to add new factors\n"
    "- For **continuous** factors: Set Min/Max (Levels column is ignored)\n"
    "- For **discrete/categorical** factors: Enter comma-separated values in Levels (Min/Max ignored)\n"
    "- **Changeability** defaults to 'easy' (can be changed for split-plot designs)"
)

# Column configuration for data_editor
column_config = {
    'Name': st.column_config.TextColumn(
        'Factor Name',
        help='Descriptive name (e.g., Temperature, Pressure)',
        required=True,
        max_chars=50
    ),
    'Type': st.column_config.SelectboxColumn(
        'Type',
        help='Factor type',
        options=['continuous', 'discrete_numeric', 'categorical'],
        required=True,
        default='continuous'
    ),
    'Min': st.column_config.NumberColumn(
        'Min',
        help='Minimum value (continuous only)',
        format='%.4f'
    ),
    'Max': st.column_config.NumberColumn(
        'Max',
        help='Maximum value (continuous only)',
        format='%.4f'
    ),
    'Levels': st.column_config.TextColumn(
        'Levels',
        help='Comma-separated values (discrete_numeric/categorical only). Leave blank for continuous.',
        max_chars=200
    ),
    'Units': st.column_config.TextColumn(
        'Units',
        help='Optional units (e.g., °C, psi)',
        max_chars=20
    ),
    'Changeability': st.column_config.SelectboxColumn(
        'Changeability',
        help='How easy to change (important for split-plot designs)',
        options=['easy', 'hard', 'very_hard'],
        required=True,
        default='easy'
    )
}

edited_df = st.data_editor(
    factors_df,
    column_config=column_config,
    use_container_width=True,
    num_rows='dynamic',  # Allow adding/deleting rows
    key='factor_table_editor'
)

st.divider()

# Action buttons
col1, col2 = st.columns([1, 1])

with col1:
    if st.button("💾 Save Changes", type="primary", use_container_width=True):
        # Validate and save
        with st.spinner("Validating factors..."):
            factors, errors = dataframe_to_factors(edited_df)
            
            if errors:
                st.error("**Validation Errors:**")
                for error in errors:
                    st.error(f"• {error}")
            else:
                # Check for duplicate names
                names = [f.name for f in factors]
                if len(names) != len(set(names)):
                    st.error("⚠️ Duplicate factor names detected. Each factor must have a unique name.")
                else:
                    # Save to session state
                    st.session_state['factors'] = factors
                    invalidate_downstream_state(from_step=1)
                    
                    st.success(f"✅ Saved {len(factors)} factor(s) successfully!")
                    st.rerun()

with col2:
    if len(st.session_state.get('factors', [])) > 0:
        if st.button("🗑️ Clear All Factors", use_container_width=True):
            st.session_state['factors'] = []
            invalidate_downstream_state(from_step=1)
            st.rerun()

st.divider()

# Help section
with st.expander("📚 Factor Definition Guide"):
    st.markdown("""
    ### Factor Types
    
    **Continuous:**
    - Use for numeric variables with any value in a range
    - Define Min and Max values
    - Examples: Temperature (150-200), pH (4-8)
    - Leave "Levels" column empty
    
    **Discrete Numeric:**
    - Use for numeric variables with specific allowed values
    - Enter comma-separated values in "Levels" column
    - Example: `100, 150, 200, 250` for RPM settings
    - Leave Min/Max empty
    
    **Categorical:**
    - Use for non-numeric variables
    - Enter comma-separated labels in "Levels" column
    - Example: `Material_A, Material_B, Material_C`
    - Leave Min/Max empty
    
    ### Changeability
    
    - **Easy**: Can be changed every run (most common)
    - **Hard**: Expensive/slow to change - defines whole-plots in split-plot designs
    - **Very Hard**: Rarely changed - defines whole-whole-plots
    
    ### Factor Names
    
    - Use letters, numbers, and underscores only
    - Avoid special characters: `*`, `+`, `()`, etc.
    - Names starting with numbers will be auto-fixed
    - Python keywords (for, if, class) will be auto-fixed
    """)
    
    st.markdown("### Example Factors")
    
    example_data = pd.DataFrame({
        'Name': ['Temperature', 'RPM', 'Catalyst'],
        'Type': ['continuous', 'discrete_numeric', 'categorical'],
        'Min': [150.0, None, None],
        'Max': [200.0, None, None],
        'Levels': ['', '100, 150, 200', 'Type_A, Type_B, Type_C'],
        'Units': ['°C', 'RPM', ''],
        'Changeability': ['easy', 'easy', 'hard']
    })
    
    st.dataframe(example_data, use_container_width=True, hide_index=True)

# Navigation
st.divider()

if len(st.session_state.get('factors', [])) >= 2:
    if st.button("Continue to Design Selection →", type="primary", use_container_width=True):
        st.session_state['current_step'] = 2
        st.switch_page("pages/2_choose_design.py")
else:
    st.info("💡 Define at least 2 factors to continue")