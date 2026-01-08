"""
Step 4: Import Experimental Results (Enhanced)

FIXED VERSION - Properly handles CSV numeric type conversion

NEW FEATURES:
- Accessible from home screen (no prerequisites)
- Auto-detect and create factors from CSV
- Flexible column mapping for mismatched names
- Persistent data preview on page
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

from src.ui.utils.state_management import (
    initialize_session_state,
    invalidate_downstream_state
)
from src.core.factors import (
    Factor, 
    FactorType, 
    ChangeabilityLevel,
    sanitize_factor_name
)

# Initialize state
initialize_session_state()

st.title("Step 4: Import Experimental Results")

# Show current data status if exists
if st.session_state.get('design') is not None and st.session_state.get('responses'):
    with st.expander("📊 Currently Loaded Data", expanded=False):
        design = st.session_state['design']
        responses = st.session_state['responses']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Runs", len(design))
        with col2:
            st.metric("Factors", len(st.session_state.get('factors', [])))
        with col3:
            st.metric("Responses", len(responses))
        
        st.markdown("**Preview (first 10 rows):**")
        preview_df = design.copy()
        for name, data in responses.items():
            preview_df[name] = data
        st.dataframe(preview_df.head(10), use_container_width=True)

st.divider()

# Import mode selection
st.subheader("Import Mode")

import_mode = st.radio(
    "How would you like to import data?",
    [
        "🆕 Upload new CSV (auto-detect factors)",
        "📂 Upload CSV with existing factors",
        "➕ Add responses to current design"
    ],
    help="Choose based on your workflow"
)

st.divider()

# Mode 1: Auto-detect (new workflow start)
if "auto-detect" in import_mode.lower():
    st.subheader("🆕 Upload CSV - Auto-Detect Factors")
    
    st.markdown("""
    Upload any DOE CSV file. The app will:
    1. Automatically detect factor columns
    2. Identify response columns
    3. Create factor definitions for you
    4. Allow you to adjust mappings if needed
    """)
    
    uploaded_file = st.file_uploader(
        "Upload Design + Results CSV",
        type=['csv'],
        key='auto_detect_upload'
    )
    
    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            
            st.success(f"✓ CSV loaded: {len(uploaded_df)} rows, {len(uploaded_df.columns)} columns")
            
            with st.expander("📋 Raw Data Preview", expanded=True):
                st.dataframe(uploaded_df.head(10), use_container_width=True)
            
            st.divider()
            
            # Auto-detect factor vs response columns
            st.subheader("🔍 Column Detection")
            
            # Strategy: numeric columns with few unique values = likely factors
            # Numeric columns with many unique values = likely responses
            # Non-numeric columns = categorical factors
            
            detected_factors = {}
            detected_responses = {}
            metadata_cols = ['StdOrder', 'RunOrder', 'Block', 'WholePlot', 'Phase']
            
            # ═══════════════════════════════════════════════════════════════
            # FIX #1: DETECTION STAGE - Force numeric conversion early
            # ═══════════════════════════════════════════════════════════════
            for col in uploaded_df.columns:
                if col in metadata_cols:
                    continue  # Skip metadata
                
                col_data = uploaded_df[col]
                
                # ★★★ CRITICAL FIX: Try to convert to numeric first ★★★
                # pandas may read numeric columns as strings
                is_numeric = False
                try:
                    numeric_col = pd.to_numeric(col_data, errors='coerce')
                    if not numeric_col.isna().any():
                        # Successfully converted all values without NaNs
                        col_data = numeric_col
                        is_numeric = True
                except:
                    is_numeric = False
                
                n_unique = col_data.nunique()
                n_total = len(col_data)
                
                # Heuristics
                if not is_numeric:
                    # Non-numeric = categorical factor
                    detected_factors[col] = {
                        'type': FactorType.CATEGORICAL,
                        'levels': sorted(col_data.unique().tolist()),
                        'confidence': 'high'
                    }
                elif n_unique <= 10:  # Increased threshold from 5 to 10
                    # Few unique values = likely factor
                    unique_vals = sorted(col_data.unique())
                    
                    # Check if values look like coded levels (-1, 0, 1) or (-1, 1)
                    if set(unique_vals).issubset({-1, -0.5, 0, 0.5, 1}):
                        detected_factors[col] = {
                            'type': FactorType.CONTINUOUS,
                            'levels': [float(unique_vals[0]), float(unique_vals[-1])],
                            'confidence': 'high',
                            'note': 'Detected as coded continuous (-1 to +1)'
                        }
                    # Check if values are evenly spaced (suggests continuous)
                    elif n_unique >= 3:
                        diffs = np.diff(unique_vals)
                        is_evenly_spaced = np.allclose(diffs, diffs[0], rtol=0.01)
                        
                        if is_evenly_spaced and n_unique >= 5:
                            # Evenly spaced with 5+ levels → likely continuous
                            detected_factors[col] = {
                                'type': FactorType.CONTINUOUS,
                                'levels': [float(unique_vals[0]), float(unique_vals[-1])],
                                'confidence': 'medium',
                                'note': f'Evenly spaced {n_unique} levels - assumed continuous'
                            }
                        else:
                            # Not evenly spaced or fewer levels → discrete numeric
                            detected_factors[col] = {
                                'type': FactorType.DISCRETE_NUMERIC,
                                'levels': [float(v) for v in unique_vals],
                                'confidence': 'medium',
                                'note': f'{n_unique} discrete levels'
                            }
                    else:
                        # 2 levels only - could be either, default to discrete
                        detected_factors[col] = {
                            'type': FactorType.DISCRETE_NUMERIC,
                            'levels': [float(v) for v in unique_vals],
                            'confidence': 'low',
                            'note': 'Only 2 levels - review if should be continuous'
                        }
                else:
                    # Many unique values = likely response
                    detected_responses[col] = {
                        'mean': float(col_data.mean()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max())
                    }
            
            # Display detection results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🔧 Detected Factors")
                if detected_factors:
                    factor_summary = []
                    for name, info in detected_factors.items():
                        if info['type'] == FactorType.CONTINUOUS:
                            levels_str = f"[{info['levels'][0]:.2f}, {info['levels'][1]:.2f}]"
                        else:
                            levels_str = f"{len(info['levels'])} levels"
                        
                        factor_summary.append({
                            'Column': name,
                            'Type': info['type'].value.replace('_', ' ').title(),
                            'Levels': levels_str,
                            'Confidence': info['confidence'].title()
                        })
                    
                    st.dataframe(pd.DataFrame(factor_summary), use_container_width=True, hide_index=True)
                else:
                    st.warning("No factors detected. All columns appear to be responses.")
            
            with col2:
                st.markdown("### 📊 Detected Responses")
                if detected_responses:
                    response_summary = []
                    for name, stats in detected_responses.items():
                        response_summary.append({
                            'Column': name,
                            'Mean': f"{stats['mean']:.2f}",
                            'Std Dev': f"{stats['std']:.2f}",
                            'Range': f"[{stats['min']:.2f}, {stats['max']:.2f}]"
                        })
                    
                    st.dataframe(pd.DataFrame(response_summary), use_container_width=True, hide_index=True)
                else:
                    st.warning("No responses detected. All columns appear to be factors.")
            
            st.divider()
            
            # Allow user to adjust mappings
            st.subheader("✏️ Adjust Mappings (Optional)")
            
            with st.form("adjust_mappings"):
                st.markdown("**Reassign columns if detection is incorrect:**")
                
                all_cols = [c for c in uploaded_df.columns if c not in metadata_cols]
                
                # Multi-select for factors
                factor_cols = st.multiselect(
                    "Factor Columns",
                    all_cols,
                    default=list(detected_factors.keys()),
                    help="Columns that represent experimental factors"
                )
                
                # Multi-select for responses
                response_cols = st.multiselect(
                    "Response Columns",
                    all_cols,
                    default=list(detected_responses.keys()),
                    help="Columns that represent measured outcomes"
                )
                
                # Validation
                overlap = set(factor_cols) & set(response_cols)
                if overlap:
                    st.error(f"⚠️ Columns cannot be both factor and response: {overlap}")
                
                submitted = st.form_submit_button("✓ Confirm Mappings", type="primary")
                
                # ═══════════════════════════════════════════════════════════════
                # FIX #2: FACTOR CREATION STAGE - Ensure numeric types
                # ═══════════════════════════════════════════════════════════════
                if submitted and not overlap:
                    # Create factors from detected/adjusted columns
                    factors = []
                    
                    for col in factor_cols:
                        # Sanitize name
                        clean_name, was_modified = sanitize_factor_name(col)
                        
                        # Determine type and levels
                        if col in detected_factors:
                            info = detected_factors[col]
                            factor_type = info['type']
                            levels = info['levels']
                            
                            # ★★★ CRITICAL FIX: Ensure numeric types are properly converted ★★★
                            # Even though we converted earlier, double-check here
                            if factor_type in [FactorType.CONTINUOUS, FactorType.DISCRETE_NUMERIC]:
                                try:
                                    levels = [float(x) for x in levels]
                                except (ValueError, TypeError) as e:
                                    st.error(f"Factor '{col}': Cannot convert levels to numeric: {e}")
                                    st.stop()
                        
                        else:
                            # User added this column manually - infer type
                            col_data = uploaded_df[col]
                            
                            # ★★★ CRITICAL FIX: Try to convert to numeric first ★★★
                            try:
                                numeric_data = pd.to_numeric(col_data, errors='coerce')
                                if not numeric_data.isna().any():
                                    # Successfully numeric - all values converted
                                    unique_vals = sorted(numeric_data.unique())
                                    if len(unique_vals) <= 5:
                                        factor_type = FactorType.DISCRETE_NUMERIC
                                        levels = [float(x) for x in unique_vals]
                                    else:
                                        factor_type = FactorType.CONTINUOUS
                                        levels = [float(numeric_data.min()), float(numeric_data.max())]
                                else:
                                    # Has NaNs after conversion - treat as categorical
                                    factor_type = FactorType.CATEGORICAL
                                    levels = sorted(col_data.unique().tolist())
                            except:
                                # Conversion failed - categorical
                                factor_type = FactorType.CATEGORICAL
                                levels = sorted(col_data.unique().tolist())
                        
                        try:
                            factor = Factor(
                                name=clean_name,
                                factor_type=factor_type,
                                changeability=ChangeabilityLevel.EASY,
                                levels=levels
                            )
                            factors.append(factor)
                            
                            if was_modified:
                                st.info(f"Factor '{col}' renamed to '{clean_name}' for compatibility")
                        
                        except ValueError as e:
                            st.error(f"Failed to create factor '{col}': {e}")
                            st.stop()
                    
                    # Extract design and responses
                    design_df = uploaded_df[[f.name for f in factors]].copy()
                    
                    # Add metadata columns if present
                    for meta_col in metadata_cols:
                        if meta_col in uploaded_df.columns:
                            design_df[meta_col] = uploaded_df[meta_col]
                    
                    # Add StdOrder/RunOrder if missing
                    if 'StdOrder' not in design_df.columns:
                        design_df.insert(0, 'StdOrder', range(1, len(design_df) + 1))
                    if 'RunOrder' not in design_df.columns:
                        design_df.insert(1, 'RunOrder', range(1, len(design_df) + 1))
                    
                    # Extract responses
                    responses = {}
                    for col in response_cols:
                        try:
                            responses[col] = pd.to_numeric(uploaded_df[col], errors='coerce').values
                            
                            n_missing = np.isnan(responses[col]).sum()
                            if n_missing > 0:
                                st.warning(f"Response '{col}' has {n_missing} missing value(s)")
                        except:
                            st.error(f"Response '{col}' could not be converted to numeric")
                            st.stop()
                    
                    # Save to session state
                    st.session_state['factors'] = factors
                    st.session_state['design'] = design_df
                    st.session_state['responses'] = responses
                    st.session_state['response_names'] = list(responses.keys())
                    st.session_state['design_metadata'] = {
                        'design_type': 'imported',
                        'import_mode': 'auto_detect',
                        'is_split_plot': 'WholePlot' in design_df.columns,
                        'has_blocking': 'Block' in design_df.columns
                    }
                    
                    st.success(
                        f"✅ Imported successfully!\n\n"
                        f"- {len(factors)} factor(s) created\n"
                        f"- {len(design_df)} run(s)\n"
                        f"- {len(responses)} response(s)"
                    )
                    
                    invalidate_downstream_state(from_step=4)
                    st.rerun()
        
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            st.exception(e)

# Mode 2: Upload with existing factors (flexible mapping)
elif "existing factors" in import_mode.lower():
    st.subheader("📂 Upload CSV - Map to Existing Factors")
    
    # Check if factors exist
    if not st.session_state.get('factors'):
        st.warning(
            "⚠️ No factors defined yet. Either:\n"
            "1. Define factors in Step 1 first, OR\n"
            "2. Use 'Auto-detect' mode above"
        )
        
        if st.button("→ Go to Step 1: Define Factors"):
            st.session_state['current_step'] = 1
            st.switch_page("pages/1_define_factors.py")
        
        st.stop()
    
    factors = st.session_state['factors']
    
    st.markdown(f"""
    You have **{len(factors)} factors** defined. Upload a CSV and we'll help you map columns.
    
    **Defined factors:** {', '.join([f.name for f in factors])}
    """)
    
    uploaded_file = st.file_uploader(
        "Upload CSV with Design + Results",
        type=['csv'],
        key='flexible_upload'
    )
    
    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            
            st.success(f"✓ CSV loaded: {len(uploaded_df)} rows, {len(uploaded_df.columns)} columns")
            
            with st.expander("📋 CSV Preview"):
                st.dataframe(uploaded_df.head(10), use_container_width=True)
            
            st.divider()
            
            # Column mapping interface
            st.subheader("🔗 Map CSV Columns to Factors")
            
            csv_cols = uploaded_df.columns.tolist()
            metadata_cols = ['StdOrder', 'RunOrder', 'Block', 'WholePlot', 'Phase']
            
            # Auto-suggest mappings based on name similarity
            def suggest_mapping(factor_name: str, csv_columns: List[str]) -> Optional[str]:
                """Suggest best matching CSV column for a factor."""
                factor_lower = factor_name.lower()
                
                # Exact match
                if factor_name in csv_columns:
                    return factor_name
                
                # Case-insensitive match
                for col in csv_columns:
                    if col.lower() == factor_lower:
                        return col
                
                # Partial match
                for col in csv_columns:
                    if factor_lower in col.lower() or col.lower() in factor_lower:
                        return col
                
                return None
            
            # Build mapping form
            with st.form("column_mapping"):
                st.markdown("**Map each factor to a CSV column:**")
                
                column_map = {}
                
                for factor in factors:
                    suggested = suggest_mapping(factor.name, csv_cols)
                    
                    col1, col2, col3 = st.columns([2, 1, 2])
                    
                    with col1:
                        st.markdown(f"**{factor.name}**")
                        st.caption(f"{factor.factor_type.value.replace('_', ' ').title()}")
                    
                    with col2:
                        st.markdown("→")
                    
                    with col3:
                        mapped_col = st.selectbox(
                            f"CSV Column",
                            options=['(unmapped)'] + csv_cols,
                            index=csv_cols.index(suggested) + 1 if suggested else 0,
                            key=f'map_{factor.name}',
                            label_visibility='collapsed'
                        )
                        
                        if mapped_col != '(unmapped)':
                            column_map[factor.name] = mapped_col
                
                st.divider()
                
                st.markdown("**Response columns** (all others will be treated as responses):")
                
                # Calculate which columns are unmapped
                mapped_cols = set(column_map.values())
                potential_response_cols = [c for c in csv_cols 
                                          if c not in mapped_cols 
                                          and c not in metadata_cols]
                
                response_cols = st.multiselect(
                    "Select Response Columns",
                    potential_response_cols,
                    default=potential_response_cols,
                    help="Measured outcomes from experiments"
                )
                
                submitted = st.form_submit_button("✓ Import with Mapping", type="primary")
                
                if submitted:
                    # Validate all factors mapped
                    unmapped_factors = [f.name for f in factors if f.name not in column_map]
                    
                    if unmapped_factors:
                        st.error(f"⚠️ Unmapped factors: {', '.join(unmapped_factors)}")
                        st.stop()
                    
                    if not response_cols:
                        st.error("⚠️ At least one response column required")
                        st.stop()
                    
                    # Build design DataFrame with correct factor names
                    design_df = pd.DataFrame()
                    
                    for factor in factors:
                        csv_col = column_map[factor.name]
                        design_df[factor.name] = uploaded_df[csv_col]
                    
                    # Add metadata columns if present
                    for meta_col in metadata_cols:
                        if meta_col in uploaded_df.columns:
                            design_df[meta_col] = uploaded_df[meta_col]
                    
                    # Add StdOrder/RunOrder if missing
                    if 'StdOrder' not in design_df.columns:
                        design_df.insert(0, 'StdOrder', range(1, len(design_df) + 1))
                    if 'RunOrder' not in design_df.columns:
                        design_df.insert(1, 'RunOrder', range(1, len(design_df) + 1))
                    
                    # Extract responses
                    responses = {}
                    for col in response_cols:
                        try:
                            responses[col] = pd.to_numeric(uploaded_df[col], errors='coerce').values
                            
                            n_missing = np.isnan(responses[col]).sum()
                            if n_missing > 0:
                                st.warning(f"Response '{col}' has {n_missing} missing value(s)")
                        except:
                            st.error(f"Response '{col}' could not be converted to numeric")
                            st.stop()
                    
                    # Validate factor values against definitions
                    validation_warnings = []
                    for factor in factors:
                        col_data = design_df[factor.name]
                        
                        if factor.is_continuous():
                            if col_data.min() < factor.min_value or col_data.max() > factor.max_value:
                                validation_warnings.append(
                                    f"⚠️ '{factor.name}': values outside range [{factor.min_value}, {factor.max_value}]"
                                )
                        elif factor.is_categorical():
                            invalid_vals = set(col_data.unique()) - set(factor.levels)
                            if invalid_vals:
                                validation_warnings.append(
                                    f"⚠️ '{factor.name}': unexpected values: {invalid_vals}"
                                )
                    
                    if validation_warnings:
                        st.warning("**Validation Warnings:**")
                        for warning in validation_warnings:
                            st.warning(warning)
                        st.info("Import will proceed, but check your data")
                    
                    # Save to session state
                    st.session_state['design'] = design_df
                    st.session_state['responses'] = responses
                    st.session_state['response_names'] = list(responses.keys())
                    st.session_state['design_metadata'] = {
                        'design_type': 'imported',
                        'import_mode': 'mapped',
                        'column_mapping': column_map,
                        'is_split_plot': 'WholePlot' in design_df.columns,
                        'has_blocking': 'Block' in design_df.columns
                    }
                    
                    st.success(
                        f"✅ Imported successfully!\n\n"
                        f"- {len(design_df)} run(s)\n"
                        f"- {len(responses)} response(s)\n"
                        f"- Column mapping applied"
                    )
                    
                    invalidate_downstream_state(from_step=4)
                    st.rerun()
        
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
            st.exception(e)

# Mode 3: Add responses to existing design (original workflow)
else:
    st.subheader("➕ Add Responses to Current Design")
    
    if not st.session_state.get('design') is not None:
        st.warning("⚠️ No design loaded. Please generate a design first or use one of the import modes above.")
        st.stop()
    
    design = st.session_state['design']
    factors = st.session_state['factors']
    
    st.markdown(f"Current design: **{len(design)} runs**, **{len(factors)} factors**")
    
    with st.expander("📋 Current Design Preview"):
        st.dataframe(design.head(10), use_container_width=True)
    
    # Download template
    st.download_button(
        "📥 Download Results Template",
        data=design.to_csv(index=False),
        file_name="results_template.csv",
        mime="text/csv",
        help="Pre-filled with design, just add response column(s)"
    )
    
    st.divider()
    
    uploaded_file = st.file_uploader(
        "Upload CSV with Results",
        type=['csv'],
        help="Can be: (1) full design+results, or (2) responses only"
    )
    
    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            
            st.dataframe(uploaded_df.head(10), use_container_width=True)
            
            # Detect upload type
            factor_names = [f.name for f in factors]
            has_factors = all(fname in uploaded_df.columns for fname in factor_names)
            
            if has_factors:
                st.info("✓ Detected full design + results CSV")
                
                # Validate row count
                if len(uploaded_df) != len(design):
                    st.error(f"Row count mismatch: Expected {len(design)}, got {len(uploaded_df)}")
                    st.stop()
                
                # Identify response columns
                design_cols = set(design.columns)
                uploaded_cols = set(uploaded_df.columns)
                response_cols = list(uploaded_cols - design_cols)
                
                if not response_cols:
                    st.error("No new response columns found")
                    st.stop()
                
                st.success(f"Found {len(response_cols)} response(s): {', '.join(response_cols)}")
            
            else:
                st.info("Detected responses-only CSV")
                
                if len(uploaded_df) != len(design):
                    st.error(f"Row count mismatch: Expected {len(design)}, got {len(uploaded_df)}")
                    st.stop()
                
                response_cols = list(uploaded_df.columns)
                st.success(f"Found {len(response_cols)} response(s): {', '.join(response_cols)}")
            
            # Extract responses
            responses = {}
            for col in response_cols:
                try:
                    responses[col] = pd.to_numeric(uploaded_df[col], errors='coerce').values
                    
                    n_missing = np.isnan(responses[col]).sum()
                    if n_missing > 0:
                        st.warning(f"'{col}' has {n_missing} missing value(s)")
                except:
                    st.error(f"'{col}' could not be converted to numeric")
                    st.stop()
            
            # Show statistics
            stats_data = []
            for name, data in responses.items():
                stats_data.append({
                    'Response': name,
                    'Count': len(data),
                    'Missing': np.isnan(data).sum(),
                    'Mean': np.nanmean(data),
                    'Std Dev': np.nanstd(data),
                    'Min': np.nanmin(data),
                    'Max': np.nanmax(data)
                })
            
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True)
            
            if st.button("✓ Import Responses", type="primary", use_container_width=True):
                st.session_state['responses'] = responses
                st.session_state['response_names'] = list(responses.keys())
                
                st.success(f"✅ Imported {len(responses)} response(s)!")
                
                invalidate_downstream_state(from_step=4)
                st.rerun()
        
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.exception(e)

# Navigation
st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("← Back to Home", use_container_width=True):
        st.switch_page("app.py")

with col2:
    if st.session_state.get('responses'):
        if st.button("Analyze →", type="primary", use_container_width=True):
            st.session_state['current_step'] = 5
            st.switch_page("pages/5_analyze.py")