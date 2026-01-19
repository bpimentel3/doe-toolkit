"""
Step 4: Import Experimental Results (with CSV Metadata Support)

NEW FEATURES:
- Parse CSV with DOE-Toolkit metadata headers
- Import response definitions from CSV
- Compare factors with session (show mismatch dialog)
- Three import paths: fresh session, active session, response-only
- Graceful fallback for plain CSVs
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
from src.ui.utils.csv_parser import (
    parse_doe_csv,
    validate_csv_structure,
    ParseResult,
    CSVParseError
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

# File upload
st.subheader("📤 Upload Results CSV")

uploaded_file = st.file_uploader(
    "Choose CSV file (with or without DOE-Toolkit metadata)",
    type=['csv'],
    key='results_upload'
)

if uploaded_file:
    # Read file content
    file_content = uploaded_file.getvalue().decode('utf-8')
    
    # Try to parse as DOE-Toolkit format
    parse_result = parse_doe_csv(file_content)
    
    if parse_result.is_valid:
        st.success("✓ CSV with DOE-Toolkit metadata detected!")
        
        # Determine import path
        has_session_design = st.session_state.get('design') is not None
        has_session_factors = len(st.session_state.get('factors', [])) > 0
        
        # PATH 1: Fresh session (no design yet)
        if not has_session_design:
            st.info("📌 **Fresh Session Import** - Loading factors and design from CSV")
            
            # Show factor summary
            with st.expander("📋 Factors to Import", expanded=True):
                st.write(f"**{len(parse_result.factors)} factors found:**")
                for factor in parse_result.factors:
                    if factor.factor_type == FactorType.CONTINUOUS:
                        st.caption(f"• {factor.name}: continuous [{factor.min_value}, {factor.max_value}] {factor.units or ''}")
                    elif factor.factor_type == FactorType.DISCRETE_NUMERIC:
                        st.caption(f"• {factor.name}: discrete {factor.discrete_values} {factor.units or ''}")
                    else:  # CATEGORICAL
                        st.caption(f"• {factor.name}: categorical {factor.categorical_levels} {factor.units or ''}")
            
            # Show response summary
            if parse_result.response_definitions:
                with st.expander("📊 Responses to Import", expanded=True):
                    st.write(f"**{len(parse_result.response_definitions)} responses found:**")
                    for resp in parse_result.response_definitions:
                        units_str = f" ({resp['units']})" if resp['units'] else ""
                        st.caption(f"• {resp['name']}{units_str}")
            
            # Show design preview
            with st.expander("🔍 Design Data Preview", expanded=True):
                st.dataframe(parse_result.design_data.head(10), use_container_width=True)
            
            # Import button
            if st.button("✅ Import Design + Factors + Responses", type="primary", use_container_width=True):
                # Load into session
                st.session_state['factors'] = parse_result.factors
                st.session_state['design'] = parse_result.design_data
                st.session_state['response_definitions'] = parse_result.response_definitions
                st.session_state['design_metadata'] = parse_result.metadata
                
                # Extract responses from design data (columns not in factors or metadata)
                factor_names = {f.name for f in parse_result.factors}
                meta_cols = {'StdOrder', 'RunOrder', 'Block', 'WholePlot', 'Phase'}
                response_names = {r['name'] for r in parse_result.response_definitions}
                
                responses = {}
                for col in parse_result.design_data.columns:
                    if col not in factor_names and col not in meta_cols and col in response_names:
                        # Extract response data (skip empty values)
                        col_data = parse_result.design_data[col]
                        # Only add if has non-empty values
                        if col_data.notna().any():
                            responses[col] = col_data.values
                
                if responses:
                    st.session_state['responses'] = responses
                    st.session_state['response_names'] = list(responses.keys())
                    st.success(f"✓ Imported {len(parse_result.factors)} factors, {len(responses)} responses")
                else:
                    st.warning("⚠️ No response data found in CSV (empty columns)")
                    st.info("You can add response data in Step 5 (Analyze)")
                
                st.rerun()
        
        # PATH 2: Active session with factors
        elif has_session_factors:
            st.info("📌 **Factor Comparison** - Checking CSV factors against session factors")
            
            # Validate factor compatibility
            is_valid, errors = validate_csv_structure(parse_result, st.session_state.get('factors'))
            
            if is_valid:
                st.success("✓ Factors match! Proceeding with import...")
                
                # Show design preview
                with st.expander("🔍 Design + Response Data Preview", expanded=True):
                    st.dataframe(parse_result.design_data.head(10), use_container_width=True)
                
                # Check for response mismatch
                session_responses = set(st.session_state.get('responses', {}).keys())
                csv_responses = {r['name'] for r in parse_result.response_definitions}
                
                if csv_responses and session_responses and csv_responses != session_responses:
                    st.warning(f"⚠️ Response names differ:")
                    st.caption(f"Session has: {session_responses}")
                    st.caption(f"CSV has: {csv_responses}")
                
                # Import button
                if st.button("✅ Import Results (Keep Factors)", type="primary", use_container_width=True):
                    # Extract responses
                    factor_names = {f.name for f in st.session_state['factors']}
                    meta_cols = {'StdOrder', 'RunOrder', 'Block', 'WholePlot', 'Phase'}
                    
                    responses = {}
                    for col in parse_result.design_data.columns:
                        if col not in factor_names and col not in meta_cols:
                            col_data = parse_result.design_data[col]
                            if col_data.notna().any():
                                responses[col] = col_data.values
                    
                        if responses:
                            st.session_state['responses'] = responses
                            st.session_state['response_names'] = list(responses.keys())
                            st.session_state['response_definitions'] = parse_result.response_definitions
                            st.success(f"✓ Imported {len(responses)} response(s)")
                        else:
                            st.warning("No response data in CSV")
                    
                    st.rerun()
            
            else:
                # MISMATCH DIALOG
                st.error("❌ Factor mismatch detected!")
                
                with st.expander("🔍 Comparison Details", expanded=True):
                    for error in errors:
                        st.error(f"• {error}")
                
                # Show factor comparison table
                st.markdown("**Factor Comparison:**")
                comparison_data = []
                
                for i, csv_factor in enumerate(parse_result.factors):
                    if i < len(st.session_state['factors']):
                        session_factor = st.session_state['factors'][i]
                        comparison_data.append({
                            'CSV Factor': csv_factor.name,
                            'Session Factor': session_factor.name,
                            'Match': '✓' if csv_factor.name == session_factor.name else '✗'
                        })
                    else:
                        comparison_data.append({
                            'CSV Factor': csv_factor.name,
                            'Session Factor': '—',
                            'Match': '✗'
                        })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)
                
                st.divider()
                
                # User chooses resolution
                st.subheader("How would you like to proceed?")
                
                resolution = st.radio(
                    "Resolution strategy:",
                    [
                        "Use CSV factors (replace session)",
                        "Use session factors (skip CSV factors)",
                        "Cancel import"
                    ],
                    label_visibility="collapsed"
                )
                
                if resolution == "Use CSV factors (replace session)":
                    if st.button("✅ Replace Factors & Import", type="primary", use_container_width=True):
                        st.session_state['factors'] = parse_result.factors
                        st.session_state['design'] = parse_result.design_data
                        
                        # Extract responses
                        factor_names = {f.name for f in parse_result.factors}
                        meta_cols = {'StdOrder', 'RunOrder', 'Block', 'WholePlot', 'Phase'}
                        
                        responses = {}
                        for col in parse_result.design_data.columns:
                            if col not in factor_names and col not in meta_cols:
                                col_data = parse_result.design_data[col]
                                if col_data.notna().any():
                                    responses[col] = col_data.values
                        
                        if responses:
                            st.session_state['responses'] = responses
                            st.session_state['response_names'] = list(responses.keys())
                            st.session_state['response_definitions'] = parse_result.response_definitions
                            st.success(f"✓ Replaced factors and imported {len(responses)} response(s)")
                        
                        st.rerun()
                
                elif resolution == "Use session factors (skip CSV factors)":
                    if st.button("✅ Import Responses Only", type="primary", use_container_width=True):
                        # Extract responses only
                        factor_names = {f.name for f in st.session_state['factors']}
                        meta_cols = {'StdOrder', 'RunOrder', 'Block', 'WholePlot', 'Phase'}
                        
                        responses = {}
                        for col in parse_result.design_data.columns:
                            if col not in factor_names and col not in meta_cols:
                                col_data = parse_result.design_data[col]
                                if col_data.notna().any():
                                    responses[col] = col_data.values
                        
                        if responses:
                            st.session_state['responses'] = responses
                            st.session_state['response_names'] = list(responses.keys())
                            st.session_state['response_definitions'] = parse_result.response_definitions
                            st.success(f"✓ Imported {len(responses)} response(s) (factors unchanged)")
                        else:
                            st.warning("No response data to import")
                        
                        st.rerun()
    
    else:
        # FALLBACK: Plain CSV without metadata
        st.warning("⚠️ CSV format not recognized as DOE-Toolkit metadata format")
        
        # Show parse error for debugging
        if parse_result.error:
            with st.expander("🔍 Parse Error Details", expanded=False):
                st.error(f"**Error:** {parse_result.error}")
                st.caption("If you believe this is a valid DOE-Toolkit format file, please report this issue.")
        
        st.info("Attempting to parse as plain CSV...")
        
        try:
            # Try basic parsing
            csv_df = pd.read_csv(uploaded_file)
            
            st.success(f"✓ Plain CSV loaded: {len(csv_df)} rows, {len(csv_df.columns)} columns")
            
            with st.expander("📋 Data Preview", expanded=True):
                st.dataframe(csv_df.head(10), use_container_width=True)
            
            # For plain CSV, guide user through mapping
            st.subheader("Map Columns to Responses")
            
            factor_cols = set(st.session_state.get('factor_names', []))
            meta_cols = {'StdOrder', 'RunOrder', 'Block', 'WholePlot', 'Phase'}
            potential_response_cols = [c for c in csv_df.columns if c not in factor_cols and c not in meta_cols]
            
            if potential_response_cols:
                st.write(f"**{len(potential_response_cols)} potential response column(s) detected:**")
                
                responses = {}
                for col in potential_response_cols:
                    # Check if numeric
                    if pd.api.types.is_numeric_dtype(csv_df[col]):
                        st.caption(f"✓ {col} (numeric)")
                        responses[col] = csv_df[col].values
                    else:
                        st.caption(f"⚠️ {col} (non-numeric, skipping)")
                
                if responses and st.button("✅ Import as Responses", type="primary", use_container_width=True):
                    st.session_state['responses'] = responses
                    st.session_state['response_names'] = list(responses.keys())
                    st.success(f"✓ Imported {len(responses)} response(s)")
                    st.rerun()
            
            else:
                st.warning("No identifiable response columns found")
        
        except Exception as e:
            st.error(f"Failed to parse CSV: {e}")

# Navigation
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("← Back to Design", use_container_width=True):
        st.switch_page("pages/4_preview_design.py")

with col3:
    if st.session_state.get('responses'):
        if st.button("Analyze Results →", type="primary", use_container_width=True):
            st.session_state['current_step'] = 6
            st.switch_page("pages/6_analyze.py")
