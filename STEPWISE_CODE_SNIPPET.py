"""
PASTE THIS CODE INTO 6_analyze.py

Location: Replace the line "st.divider()" that appears after the model builder
          and before "st.sidebar.header("Advanced Options")"

Approximately line 487 in the current file.
"""

# === STEPWISE REGRESSION BUTTON ===
# Create analysis object early for stepwise (only if data available)
stepwise_results = None
if responses.get(selected_response) is not None:
    # Create temporary analysis object for stepwise
    response_data = responses[selected_response]
    
    # Apply row exclusions if any
    if st.session_state.get('excluded_rows'):
        mask = np.ones(len(design), dtype=bool)
        mask[st.session_state['excluded_rows']] = False
        design_for_stepwise = design[mask].reset_index(drop=True)
        response_for_stepwise = response_data[mask]
    else:
        design_for_stepwise = design
        response_for_stepwise = response_data
    
    temp_analysis = ANOVAAnalysis(
        design=design_for_stepwise,
        response=response_for_stepwise,
        factors=factors,
        response_name=selected_response
    )
    
    # Display stepwise button
    stepwise_results = display_stepwise_button(
        factors=factors,
        anova_analysis=temp_analysis,
        key_prefix=f"stepwise_{selected_response}"
    )
    
    # If stepwise completed, update model terms
    if stepwise_results is not None:
        st.session_state['model_terms_per_response'][selected_response] = stepwise_results.final_terms
        st.info(f"✓ Model updated with {len(stepwise_results.final_terms)} terms from stepwise selection")
        st.rerun()

st.divider()
