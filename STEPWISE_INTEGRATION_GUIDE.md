# Stepwise Regression Integration Guide

## Files Modified

1. `src/core/stepwise.py` - NEW FILE (core stepwise algorithm)
2. `src/ui/components/model_builder.py` - UPDATED (added stepwise button function)
3. `src/ui/pages/6_analyze.py` - NEEDS INTEGRATION (add stepwise button call)
4. `tests/test_stepwise.py` - NEW FILE (comprehensive tests)

## Integration Instructions for 6_analyze.py

### Step 1: Import already added
The import has been updated:
```python
from src.ui.components.model_builder import display_model_builder, format_term_for_display, display_stepwise_button
```

### Step 2: Add stepwise button call

Find this section in `6_analyze.py` (around line 475-485):

```python
    st.session_state['model_terms_per_response'][selected_response] = updated_terms
    invalidate_downstream_state(from_step=5)
    st.rerun()

st.divider()  # <-- FIND THIS LINE

st.sidebar.header("Advanced Options")
enforce_hierarchy = st.sidebar.checkbox("Enforce Hierarchy", value=True)
```

**REPLACE** the `st.divider()` line with:

```python
# === STEPWISE REGRESSION BUTTON ===
# Note: Create analysis object early for stepwise
# Only if data is available
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
```

## Testing

### 1. Run unit tests:
```bash
conda activate doe-toolkit
pytest tests/test_stepwise.py -v
```

Expected: All tests pass (10+ tests)

### 2. Test in Streamlit UI:
```bash
streamlit run src/ui/app.py
```

**Test workflow:**
1. Define 3 continuous factors (A, B, C)
2. Choose Response Surface Design (Box-Behnken)
3. Generate design
4. Import mock CSV data (or use real data)
5. Go to Analyze page
6. Select a response
7. Expand "⚙️ Stepwise Regression Settings"
8. Click "🔍 Stepwise Regression (BIC)" button
9. Watch progress bar update
10. See summary table with steps
11. Model terms should update automatically
12. ANOVA should refit with new terms

## Expected Behavior

### Stepwise Button Location
- Appears **after** the model builder component
- Appears **before** the "Advanced Options" sidebar
- Only shown if response data is available

### When Clicked
1. Progress bar shows: "Step X/Y: Evaluating candidates..."
2. Algorithm runs in background (no page freezes)
3. Upon completion:
   - Success message: "✓ Stepwise regression completed in N iterations"
   - Summary table shows:
     - Convergence reason
     - Initial BIC → Final BIC
     - Improvement
     - Step-by-step decisions (add/remove which terms)
     - Final model terms
4. Model terms automatically update
5. Page reruns with new model
6. ANOVA refits with selected terms

### Settings Expander
- Max iterations: 10-200 (default 50)
- BIC threshold: 0.1-10.0 (default 2.0)
- Caption explains candidate pool

### Candidate Pool
Algorithm considers:
- All main effects
- All 2-way interactions
- Quadratic terms (continuous factors only)
- Respects hierarchy at every step

## Design Decisions

1. **Bidirectional stepwise**: Can add AND remove terms
2. **Hierarchy enforced**: Never violates hierarchy rules
3. **BIC criterion**: Lower is better, standard threshold = 2.0
4. **Starting model**: Intercept only
5. **Mandatory terms**: Intercept cannot be removed
6. **Progress tracking**: Live progress bar, no intermediate output
7. **Automatic update**: Model terms update on completion

## Error Handling

If stepwise fails:
- Progress bar clears
- Error message displayed
- Exception traceback shown
- Model terms unchanged
- User can adjust settings and retry

## Git Commit Message

After testing successfully:

```
feat: implement bidirectional stepwise regression with BIC

- Add core stepwise algorithm in src/core/stepwise.py
  - Bidirectional selection (add and remove terms)
  - BIC criterion with configurable threshold
  - Hierarchy enforcement at every step
  - Progress callback for UI integration

- Add stepwise button to model builder component
  - Expandable settings (max iterations, BIC threshold)
  - Progress bar during execution
  - Summary table with step-by-step decisions
  - Automatic model term update

- Integrate into analyze page
  - Button appears after model builder
  - Uses live ANOVA analysis object
  - Updates session state on completion

- Comprehensive test coverage (10+ tests)
  - BIC computation validation
  - Candidate generation tests
  - Full stepwise algorithm tests
  - Hierarchy enforcement tests
  - Summary formatting tests

Closes #XX (if applicable)
```

## Known Limitations

1. **Candidate pool**: Currently limited to:
   - Main effects
   - 2-way interactions only
   - Quadratic terms (continuous factors)
   - Does NOT include 3-way interactions or cubic terms

2. **Computational cost**: For large candidate pools (10+ factors):
   - May take 10-30 seconds
   - Progress bar provides feedback

3. **Optimal solution**: Stepwise is greedy (not globally optimal)
   - May miss best model in complex scenarios
   - Standard limitation of all stepwise methods

## Future Enhancements

Potential improvements for later:
1. Allow user to specify custom candidate pool
2. Support for 3-way interactions
3. Support for cubic/higher-order terms
4. Alternative criteria (AIC, AICc, Cp)
5. Cross-validation based selection
6. Save/load stepwise history
