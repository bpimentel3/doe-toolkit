"""
Step 4: Import Experimental Results

Import CSV with design + response data, or continue from in-memory design.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
import io

from src.ui.utils.state_management import (
    initialize_session_state,
    is_step_complete,
    can_access_step,
    get_active_design,
    invalidate_downstream_state
)

# Initialize state
initialize_session_state()

# Check access
if not can_access_step(4):
    st.warning("⚠️ Please complete Steps 1-3 first")
    st.stop()

st.title("Step 4: Import Experimental Results")

factors = st.session_state['factors']
design_in_memory = st.session_state.get('design') is not None

# Determine import mode
if design_in_memory:
    st.success("✓ Design available from current session")
    
    import_mode = st.radio(
        "Import Mode",
        [
            "Continue with current design (add responses only)",
            "Upload new design + results CSV"
        ],
        help="Choose based on whether you're continuing immediately or resuming later"
    )
    
    continue_mode = "continue" in import_mode.lower()
else:
    st.info("📂 Upload your design + results CSV to begin analysis")
    continue_mode = False

st.divider()

# Mode 1: Continue from memory
if continue_mode:
    st.subheader("Add Response Data to Current Design")
    
    design = st.session_state['design']
    
    st.markdown(f"""
    Your design has **{len(design)} runs**. Upload a CSV with response columns, 
    or upload the full design+results CSV.
    """)
    
    # Show design preview
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
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload CSV with Results",
        type=['csv'],
        help="Can be: (1) full design+results, or (2) responses only"
    )
    
    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(uploaded_df.head(10), use_container_width=True)
            
            # Detect upload type
            factor_names = [f.name for f in factors]
            has_factors = all(fname in uploaded_df.columns for fname in factor_names)
            
            if has_factors:
                st.info("✓ Detected full design + results CSV")
                
                # Validate row count
                if len(uploaded_df) != len(design):
                    st.error(
                        f"Row count mismatch: Expected {len(design)} rows, "
                        f"got {len(uploaded_df)} rows"
                    )
                    st.stop()
                
                # Identify response columns
                design_cols = set(design.columns)
                uploaded_cols = set(uploaded_df.columns)
                response_cols = list(uploaded_cols - design_cols)
                
                if not response_cols:
                    st.error("No new response columns found. Expected at least one response.")
                    st.stop()
                
                st.success(f"Found {len(response_cols)} response column(s): {', '.join(response_cols)}")
                
                # Extract responses
                responses = {}
                for col in response_cols:
                    # Validate numeric
                    try:
                        responses[col] = pd.to_numeric(uploaded_df[col], errors='coerce').values
                        
                        # Check for NaNs
                        n_missing = np.isnan(responses[col]).sum()
                        if n_missing > 0:
                            st.warning(f"Response '{col}' has {n_missing} missing value(s)")
                    except:
                        st.error(f"Response '{col}' could not be converted to numeric")
                        st.stop()
            
            else:
                st.info("Detected responses-only CSV (no factor columns)")
                
                # Assume all columns are responses
                response_cols = list(uploaded_df.columns)
                
                # Validate row count
                if len(uploaded_df) != len(design):
                    st.error(
                        f"Row count mismatch: Expected {len(design)} rows, "
                        f"got {len(uploaded_df)} rows"
                    )
                    st.stop()
                
                st.success(f"Found {len(response_cols)} response column(s): {', '.join(response_cols)}")
                
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
            
            # Show response statistics
            st.subheader("Response Statistics")
            
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
            
            # Confirm import
            if st.button("✓ Import Responses", type="primary", use_container_width=True):
                st.session_state['responses'] = responses
                st.session_state['response_names'] = list(responses.keys())
                
                st.success(f"✓ Imported {len(responses)} response(s) successfully!")
                
                # Navigate to analysis
                if st.button("Continue to Analysis →", type="primary"):
                    st.session_state['current_step'] = 5
                    st.switch_page("pages/5_analyze.py")
        
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
            st.exception(e)

# Mode 2: Upload full CSV
else:
    st.subheader("Upload Design + Results CSV")
    
    st.markdown("""
    Upload a CSV containing both:
    - **Design columns** (factor settings)
    - **Response columns** (measured outcomes)
    
    The CSV should match the design you generated, with additional response column(s).
    """)
    
    # Show expected format
    with st.expander("📋 Expected CSV Format"):
        st.markdown("**Required columns:**")
        st.code(", ".join([f.name for f in factors]))
        
        st.markdown("**Plus at least one response column:**")
        st.code("Response1, Response2, ... (your measured outcomes)")
        
        st.markdown("**Optional columns:**")
        st.code("StdOrder, RunOrder, Block, WholePlot (if present in design)")
        
        # Show example
        st.markdown("**Example:**")
        example_data = {
            factors[0].name: [-1, 1, -1, 1],
            factors[1].name: [-1, -1, 1, 1],
            'Yield': [45.2, 52.1, 48.7, 55.3],
            'Purity': [92.1, 94.5, 91.8, 95.2]
        }
        st.dataframe(pd.DataFrame(example_data))
    
    st.divider()
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Design + Results CSV*",
        type=['csv'],
        help="CSV with factor columns + response columns"
    )
    
    if uploaded_file:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            
            st.subheader("Uploaded Data Preview")
            st.dataframe(uploaded_df.head(10), use_container_width=True)
            
            st.divider()
            
            # Validate factor columns
            st.subheader("Validation")
            
            factor_names = [f.name for f in factors]
            missing_factors = [fname for fname in factor_names if fname not in uploaded_df.columns]
            
            if missing_factors:
                st.error(f"Missing factor columns: {', '.join(missing_factors)}")
                st.stop()
            
            st.success(f"✓ All {len(factor_names)} factor columns found")
            
            # Validate factor values (ensure they're in defined ranges/levels)
            validation_issues = []
            
            for factor in factors:
                col_data = uploaded_df[factor.name]
                
                if factor.is_continuous():
                    # Check range
                    if col_data.min() < factor.min_value or col_data.max() > factor.max_value:
                        validation_issues.append(
                            f"⚠️ Factor '{factor.name}': values outside range "
                            f"[{factor.min_value}, {factor.max_value}]"
                        )
                
                elif factor.is_categorical():
                    # Check categories
                    invalid_vals = set(col_data.unique()) - set(factor.levels)
                    if invalid_vals:
                        validation_issues.append(
                            f"⚠️ Factor '{factor.name}': unexpected values: {invalid_vals}"
                        )
            
            if validation_issues:
                st.warning("**Validation Warnings:**")
                for issue in validation_issues:
                    st.warning(issue)
            else:
                st.success("✓ Factor values validated")
            
            # Identify response columns
            uploaded_cols = set(uploaded_df.columns)
            design_cols = set(factor_names) | {'StdOrder', 'RunOrder', 'Block', 'WholePlot'}
            response_cols = list(uploaded_cols - design_cols)
            
            if not response_cols:
                st.error(
                    "No response columns found. Expected at least one column besides factors."
                )
                st.stop()
            
            st.success(f"✓ Found {len(response_cols)} response column(s): {', '.join(response_cols)}")
            
            # Extract design and responses
            design_df = uploaded_df[factor_names].copy()
            
            # Add standard columns if present
            for col in ['StdOrder', 'RunOrder', 'Block', 'WholePlot']:
                if col in uploaded_df.columns:
                    design_df[col] = uploaded_df[col]
            
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
            
            # Show response statistics
            st.subheader("Response Statistics")
            
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
            
            # Confirm import
            st.divider()
            
            if st.button("✓ Import Design + Results", type="primary", use_container_width=True):
                # Save to session state
                st.session_state['design'] = design_df
                st.session_state['responses'] = responses
                st.session_state['response_names'] = list(responses.keys())
                
                # Infer design metadata if not present
                if not st.session_state.get('design_metadata'):
                    st.session_state['design_metadata'] = {
                        'design_type': 'imported',
                        'is_split_plot': 'WholePlot' in design_df.columns,
                        'has_blocking': 'Block' in design_df.columns
                    }
                
                st.success(
                    f"✓ Imported design ({len(design_df)} runs) and "
                    f"{len(responses)} response(s) successfully!"
                )
                
                invalidate_downstream_state(from_step=4)
                
                # Navigate to analysis
                if st.button("Continue to Analysis →", type="primary"):
                    st.session_state['current_step'] = 5
                    st.switch_page("pages/5_analyze.py")
        
        except Exception as e:
            st.error(f"Failed to process CSV: {e}")
            st.exception(e)

# Navigation
st.divider()

col1, col2 = st.columns(2)

with col1:
    if st.button("← Back to Design Preview", use_container_width=True):
        st.session_state['current_step'] = 3
        st.switch_page("pages/3_preview_design.py")

with col2:
    if st.session_state.get('responses'):
        if st.button("Analyze →", type="primary", use_container_width=True):
            st.session_state['current_step'] = 5
            st.switch_page("pages/5_analyze.py")