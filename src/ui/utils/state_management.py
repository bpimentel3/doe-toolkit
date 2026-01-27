"""
Session State Management for DOE Toolkit Streamlit UI.

This module provides centralized state management functions and state
initialization for the multi-step DOE workflow.
"""

from typing import Any, Optional, List, Dict
import streamlit as st
import pandas as pd
import numpy as np

from src.core.factors import Factor
from src.core.augmentation.plan import AugmentationPlan, AugmentedDesign
from src.core.diagnostics import DesignDiagnosticSummary, DesignQualityReport


def initialize_session_state() -> None:
    """
    Initialize all session state variables.
    
    Called at app startup to ensure all required state exists.
    """
    # Step 1: Factor Definition
    if 'factors' not in st.session_state:
        st.session_state['factors'] = []
    
    # Step 2: Design Generation
    if 'design_type' not in st.session_state:
        st.session_state['design_type'] = None
    
    if 'design' not in st.session_state:
        st.session_state['design'] = None
    
    if 'design_metadata' not in st.session_state:
        st.session_state['design_metadata'] = {}
    
    # Step 3: Preview (no additional state needed)
    
    # Step 4: Import Results
    if 'responses' not in st.session_state:
        st.session_state['responses'] = {}
    
    if 'response_names' not in st.session_state:
        st.session_state['response_names'] = []
    
    # Step 5: Analysis
    if 'fitted_models' not in st.session_state:
        st.session_state['fitted_models'] = {}
    
    if 'model_terms_per_response' not in st.session_state:
        st.session_state['model_terms_per_response'] = {}
    
    if 'excluded_rows' not in st.session_state:
        st.session_state['excluded_rows'] = []
    
    # Step 5b: Quality Assessment (NEW)
    if 'diagnostics_summary' not in st.session_state:
        st.session_state['diagnostics_summary'] = None
    
    if 'quality_report' not in st.session_state:
        st.session_state['quality_report'] = None
    
    # Step 6: Augmentation (NEW)
    if 'show_augmentation' not in st.session_state:
        st.session_state['show_augmentation'] = False
    
    if 'augmentation_plans' not in st.session_state:
        st.session_state['augmentation_plans'] = []
    
    if 'selected_plan' not in st.session_state:
        st.session_state['selected_plan'] = None
    
    if 'augmented_design' not in st.session_state:
        st.session_state['augmented_design'] = None
    
    # Step 7: Optimization
    if 'optimization_results' not in st.session_state:
        st.session_state['optimization_results'] = None
    
    # Navigation
    if 'current_step' not in st.session_state:
        st.session_state['current_step'] = 1
    
    # Export controls
    if 'show_save_project' not in st.session_state:
        st.session_state['show_save_project'] = False
    
    if 'show_generate_report' not in st.session_state:
        st.session_state['show_generate_report'] = False


def get_current_step() -> int:
    """Get current workflow step (1-7)."""
    return st.session_state.get('current_step', 1)


def set_current_step(step: int) -> None:
    """Set current workflow step."""
    if 1 <= step <= 8:
        st.session_state['current_step'] = step


def is_step_complete(step: int) -> bool:
    """
    Check if a workflow step is complete and valid.
    
    Parameters
    ----------
    step : int
        Step number (1-8)
    
    Returns
    -------
    bool
        Whether step is complete
    """
    if step == 1:  # Define Factors
        return len(st.session_state.get('factors', [])) > 0
    
    elif step == 2:  # Select Model
        # Model selection is optional - consider complete if user proceeded
        # OR if model_terms are defined
        return (
            st.session_state.get('model_terms') is not None or
            st.session_state.get('design_type') is not None
        )
    
    elif step == 3:  # Choose Design
        return st.session_state.get('design_type') is not None
    
    elif step == 4:  # Preview Design
        return st.session_state.get('design') is not None
    
    elif step == 5:  # Import Results
        # FIXED: Check both responses dict AND design exists
        # Allow step 5 to be complete if user imported CSV with auto-detect
        # which creates both design and responses simultaneously
        responses = st.session_state.get('responses', {})
        design = st.session_state.get('design')
        
        # Step 5 is complete if we have responses AND design
        return (len(responses) > 0) and (design is not None)
    
    elif step == 6:  # Analysis
        return len(st.session_state.get('fitted_models', {})) > 0
    
    elif step == 7:  # Augmentation (optional)
        # Augmentation is optional - considered complete if:
        # - User explicitly skipped, OR
        # - Plan was executed
        return (
            st.session_state.get('augmented_design') is not None or
            not st.session_state.get('show_augmentation', False)
        )
    
    elif step == 8:  # Optimization
        return st.session_state.get('optimization_results') is not None
    
    return False


def can_access_step(step: int) -> bool:
    """
    Check if user can access a step (all prerequisites complete).
    
    Parameters
    ----------
    step : int
        Step number to check
    
    Returns
    -------
    bool
        Whether step is accessible
    """
    # Can always access step 1
    if step == 1:
        return True
    
    # Step 5 can be accessed from home (auto-detect mode)
    # This allows users to start by importing CSV
    if step == 5:
        return True
    
    # For step 6 (Analyze), check if step 5 is complete
    # This is the critical fix - we need design AND responses
    if step == 6:
        return is_step_complete(5)
    
    # For step 7 (Augmentation), only requires design structure
    # Analysis is helpful but not required - user can view design and decide to augment
    if step == 7:
        return st.session_state.get('design') is not None
    
    # For step 8 (Optimization), requires analysis to be complete
    # Augmentation (step 7) is optional, so skip it
    if step == 8:
        return is_step_complete(6)
    
    # For other steps, all previous steps must be complete
    for i in range(1, step):
        if not is_step_complete(i):
            return False
    
    return True


def invalidate_downstream_state(from_step: int) -> None:
    """
    Invalidate state for steps after the given step.
    
    Called when user modifies an earlier step (e.g., changes factors).
    
    Parameters
    ----------
    from_step : int
        Step that was modified
    """
    if from_step <= 1:
        # Factors changed - clear everything
        st.session_state['design_type'] = None
        st.session_state['design'] = None
        st.session_state['design_metadata'] = {}
        st.session_state['responses'] = {}
        st.session_state['response_names'] = []
        st.session_state['fitted_models'] = {}
        st.session_state['model_terms_per_response'] = {}
        st.session_state['diagnostics_summary'] = None
        st.session_state['quality_report'] = None
        st.session_state['augmentation_plans'] = []
        st.session_state['selected_plan'] = None
        st.session_state['augmented_design'] = None
        st.session_state['optimization_results'] = None
    
    elif from_step <= 2:
        # Design type changed - clear design and downstream
        st.session_state['design'] = None
        st.session_state['design_metadata'] = {}
        st.session_state['responses'] = {}
        st.session_state['response_names'] = []
        st.session_state['fitted_models'] = {}
        st.session_state['model_terms_per_response'] = {}
        st.session_state['diagnostics_summary'] = None
        st.session_state['quality_report'] = None
        st.session_state['augmentation_plans'] = []
        st.session_state['selected_plan'] = None
        st.session_state['augmented_design'] = None
        st.session_state['optimization_results'] = None
    
    elif from_step <= 3:
        # Design regenerated - clear results and downstream
        st.session_state['responses'] = {}
        st.session_state['response_names'] = []
        st.session_state['fitted_models'] = {}
        st.session_state['model_terms_per_response'] = {}
        st.session_state['diagnostics_summary'] = None
        st.session_state['quality_report'] = None
        st.session_state['augmentation_plans'] = []
        st.session_state['selected_plan'] = None
        st.session_state['augmented_design'] = None
        st.session_state['optimization_results'] = None
    
    elif from_step <= 4:
        # New data imported - clear analysis and downstream
        st.session_state['fitted_models'] = {}
        st.session_state['model_terms_per_response'] = {}
        st.session_state['diagnostics_summary'] = None
        st.session_state['quality_report'] = None
        st.session_state['augmentation_plans'] = []
        st.session_state['selected_plan'] = None
        st.session_state['augmented_design'] = None
        st.session_state['optimization_results'] = None
    
    elif from_step <= 5:
        # Model refit - clear diagnostics and downstream
        st.session_state['diagnostics_summary'] = None
        st.session_state['quality_report'] = None
        st.session_state['augmentation_plans'] = []
        st.session_state['selected_plan'] = None
        st.session_state['augmented_design'] = None
        st.session_state['optimization_results'] = None
    
    elif from_step <= 6:
        # Augmentation changed - clear optimization
        st.session_state['optimization_results'] = None


def get_active_design() -> Optional[pd.DataFrame]:
    """
    Get the currently active design (augmented if available, else original).
    
    Returns
    -------
    pd.DataFrame or None
        Active design matrix
    """
    if st.session_state.get('augmented_design') is not None:
        return st.session_state['augmented_design'].combined_design
    
    return st.session_state.get('design')


def get_active_responses() -> Dict[str, np.ndarray]:
    """
    Get active response data (matching active design).
    
    If design was augmented, responses should include new measurements.
    If not augmented, returns original responses.
    
    Returns
    -------
    Dict[str, np.ndarray]
        Response name -> measurements
    """
    return st.session_state.get('responses', {})


def is_using_augmented_design() -> bool:
    """Check if currently using augmented design."""
    return st.session_state.get('augmented_design') is not None


def reset_augmentation() -> None:
    """Reset augmentation state (allow user to start over)."""
    st.session_state['show_augmentation'] = False
    st.session_state['augmentation_plans'] = []
    st.session_state['selected_plan'] = None
    st.session_state['augmented_design'] = None
    st.session_state['optimization_results'] = None


def get_workflow_progress() -> Dict[str, Any]:
    """
    Get workflow progress summary.
    
    Returns
    -------
    dict
        Progress information for display
    """
    steps = [
        "Define Factors",
        "Select Model",
        "Choose Design",
        "Preview Design",
        "Import Results",
        "Analyze",
        "Augmentation (Optional)",
        "Optimize"
    ]
    
    progress = {
        'total_steps': 8,
        'current_step': get_current_step(),
        'steps': steps,
        'completed': [is_step_complete(i) for i in range(1, 9)],
        'accessible': [can_access_step(i) for i in range(1, 9)]
    }
    
    return progress


def display_workflow_progress() -> None:
    """Display workflow progress in sidebar."""
    progress = get_workflow_progress()
    
    st.sidebar.markdown("### Workflow Progress")
    
    for i, step_name in enumerate(progress['steps'], start=1):
        if progress['completed'][i-1]:
            icon = "✅"
        elif progress['accessible'][i-1]:
            icon = "⭕"
        else:
            icon = "🔒"
        
        if i == progress['current_step']:
            st.sidebar.markdown(f"**{icon} {i}. {step_name}**")
        else:
            st.sidebar.markdown(f"{icon} {i}. {step_name}")


def save_state_to_file(filepath: str) -> None:
    """
    Save session state to file for reproducibility.
    
    Parameters
    ----------
    filepath : str
        Path to save state
    """
    import pickle
    
    # Select serializable state
    state_to_save = {
        'factors': st.session_state.get('factors'),
        'design_type': st.session_state.get('design_type'),
        'design': st.session_state.get('design'),
        'design_metadata': st.session_state.get('design_metadata'),
        'responses': st.session_state.get('responses'),
        'response_names': st.session_state.get('response_names'),
        'model_terms_per_response': st.session_state.get('model_terms_per_response'),
        'excluded_rows': st.session_state.get('excluded_rows'),
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(state_to_save, f)


def load_state_from_file(filepath: str) -> None:
    """
    Load session state from file.
    
    Parameters
    ----------
    filepath : str
        Path to saved state
    """
    import pickle
    
    with open(filepath, 'rb') as f:
        saved_state = pickle.load(f)
    
    # Restore state
    for key, value in saved_state.items():
        st.session_state[key] = value
    
    # Invalidate fitted models and downstream (need to refit)
    invalidate_downstream_state(from_step=4)


def create_project_file() -> str:
    """
    Create project file (JSON) from current session state.
    
    Captures complete project state including:
    - Factors, design, and design metadata
    - Responses and model terms
    - Augmentation plans and augmented designs
    - Optimization results
    - Diagnostics and quality reports
    
    Returns
    -------
    str
        JSON string of project data
    """
    import json
    from datetime import datetime
    
    # Serialize factors
    factors_data = []
    for factor in st.session_state.get('factors', []):
        factors_data.append({
            'name': factor.name,
            'type': factor.factor_type.value,
            'changeability': factor.changeability.value,
            'levels': factor.levels,
            'units': factor.units
        })
    
    # Build project data
    project = {
        'version': '0.1.0',
        'created': datetime.now().isoformat(),
        'factors': factors_data,
        'design_type': st.session_state.get('design_type'),
        'design_config': st.session_state.get('design_config', {}),
        'design_metadata': st.session_state.get('design_metadata', {})
    }
    
    # Add design if exists
    if st.session_state.get('design') is not None:
        project['design'] = st.session_state['design'].to_dict('records')
    
    # Add responses if imported
    if st.session_state.get('responses'):
        responses_data = {}
        for name, data in st.session_state['responses'].items():
            responses_data[name] = data.tolist() if hasattr(data, 'tolist') else list(data)
        project['responses'] = responses_data
        project['response_names'] = st.session_state.get('response_names', [])
    
    # Add model terms if fitted
    if st.session_state.get('model_terms_per_response'):
        project['model_terms_per_response'] = st.session_state['model_terms_per_response']
    
    # Add excluded rows if any
    if st.session_state.get('excluded_rows'):
        project['excluded_rows'] = st.session_state['excluded_rows']
    
    # Add augmentation data if exists
    if st.session_state.get('augmented_design') is not None:
        augmented = st.session_state['augmented_design']
        project['augmented_design'] = {
            'combined_design': augmented.combined_design.to_dict('records'),
            'new_runs_only': augmented.new_runs_only.to_dict('records'),
            'block_column': augmented.block_column,
            'n_runs_original': augmented.n_runs_original,
            'n_runs_added': augmented.n_runs_added,
            'n_runs_total': augmented.n_runs_total,
            'achieved_improvements': augmented.achieved_improvements,
            'resolution': augmented.resolution,
            'd_efficiency': augmented.d_efficiency,
            'condition_number': augmented.condition_number
        }
    
    # Add selected augmentation plan metadata if exists
    if st.session_state.get('selected_plan') is not None:
        plan = st.session_state['selected_plan']
        project['selected_augmentation_plan'] = {
            'plan_id': plan.plan_id,
            'plan_name': plan.plan_name,
            'strategy': plan.strategy,
            'n_runs_to_add': plan.n_runs_to_add,
            'expected_improvements': plan.expected_improvements
        }
    
    # Add optimization results if exists
    if st.session_state.get('optimization_results') is not None:
        opt_result = st.session_state['optimization_results']
        project['optimization_results'] = {
            'optimal_settings': opt_result.optimal_settings,
            'predicted_response': opt_result.predicted_response,
            'confidence_interval': opt_result.confidence_interval,
            'prediction_interval': opt_result.prediction_interval,
            'objective_value': opt_result.objective_value,
            'n_iterations': opt_result.n_iterations,
            'success': opt_result.success,
            'message': opt_result.message
        }
    
    return json.dumps(project, indent=2)


def load_project_file(file_content: str) -> None:
    """
    Load project from JSON file content with factor name sanitization.
    
    This function handles legacy projects that may have factor names with
    special characters that are no longer allowed.
    
    Parameters
    ----------
    file_content : str
        JSON string from uploaded file
    """
    import json
    import pandas as pd
    import numpy as np
    from src.core.factors import (
        Factor, 
        FactorType, 
        ChangeabilityLevel,
        sanitize_factor_name
    )
    
    try:
        project = json.loads(file_content)
    except json.JSONDecodeError as e:
        st.error(f"Invalid project file: {e}")
        return
    
    # Validate version (simple check)
    if 'version' not in project:
        st.warning("Project file missing version. Attempting to load anyway...")
    
    # Restore factors with sanitization
    factors = []
    renamed_factors = []
    
    for f_data in project.get('factors', []):
        original_name = f_data['name']
        
        # Sanitize the name
        clean_name, was_modified = sanitize_factor_name(original_name)
        
        # Track renames for user notification
        if was_modified:
            renamed_factors.append({
                'original': original_name,
                'sanitized': clean_name
            })
        
        try:
            factor = Factor(
                name=clean_name,  # Use sanitized name
                factor_type=FactorType(f_data['type']),
                changeability=ChangeabilityLevel(f_data['changeability']),
                levels=f_data['levels'],
                units=f_data.get('units')
            )
            factors.append(factor)
        except Exception as e:
            st.error(f"Failed to load factor '{original_name}': {e}")
            continue
    
    if not factors:
        st.error("No valid factors found in project file")
        return
    
    st.session_state['factors'] = factors
    
    # Show sanitization warnings if any
    if renamed_factors:
        st.warning(
            "⚠️ **Factor names were updated for compatibility**\n\n"
            "The following factor names contained special characters and were sanitized:"
        )
        
        rename_df = pd.DataFrame(renamed_factors)
        rename_df.columns = ['Original Name', 'Sanitized Name']
        st.dataframe(rename_df, use_container_width=True)
        
        st.info(
            "💡 **Why?** Special characters like `*`, `+`, `()` are not allowed in "
            "factor names because they have special meaning in statistical formulas. "
            "Your project has been updated to use safe names."
        )
    
    # Restore design settings
    st.session_state['design_type'] = project.get('design_type')
    st.session_state['design_config'] = project.get('design_config', {})
    st.session_state['design_metadata'] = project.get('design_metadata', {})
    
    # Restore design if exists
    # Note: Design column names should match sanitized factor names
    if 'design' in project:
        design_df = pd.DataFrame(project['design'])
        
        # Check if design columns need renaming to match sanitized factor names
        if renamed_factors:
            rename_map = {r['original']: r['sanitized'] for r in renamed_factors}
            
            # Rename columns if they exist in the design
            cols_to_rename = {
                old: new for old, new in rename_map.items() 
                if old in design_df.columns
            }
            
            if cols_to_rename:
                design_df = design_df.rename(columns=cols_to_rename)
                st.info(
                    f"✓ Updated {len(cols_to_rename)} column(s) in design to match "
                    "sanitized factor names"
                )
        
        st.session_state['design'] = design_df
    
    # Restore responses if exist
    if 'responses' in project:
        responses = {}
        for name, data in project['responses'].items():
            responses[name] = np.array(data)
        st.session_state['responses'] = responses
        st.session_state['response_names'] = project.get('response_names', [])
    
    # Restore model terms (need to update if factor names changed)
    if 'model_terms_per_response' in project:
        model_terms = project['model_terms_per_response']
        
        # Update model terms if factor names changed
        if renamed_factors:
            rename_map = {r['original']: r['sanitized'] for r in renamed_factors}
            
            updated_model_terms = {}
            for response_name, terms in model_terms.items():
                updated_terms = []
                for term in terms:
                    # Replace factor names in terms
                    updated_term = term
                    for old_name, new_name in rename_map.items():
                        # Handle different term formats
                        # Main effect: "OldName" -> "NewName"
                        if updated_term == old_name:
                            updated_term = new_name
                        # Interaction: "A*OldName" -> "A*NewName"
                        updated_term = updated_term.replace(f"*{old_name}", f"*{new_name}")
                        updated_term = updated_term.replace(f"{old_name}*", f"{new_name}*")
                        # Quadratic: "I(OldName**2)" -> "I(NewName**2)"
                        updated_term = updated_term.replace(f"I({old_name}**", f"I({new_name}**")
                    
                    updated_terms.append(updated_term)
                
                updated_model_terms[response_name] = updated_terms
            
            st.session_state['model_terms_per_response'] = updated_model_terms
            
            st.info(
                "✓ Updated model terms to use sanitized factor names"
            )
        else:
            st.session_state['model_terms_per_response'] = model_terms
    
    # Set current step based on what's loaded
    if st.session_state.get('responses'):
        st.session_state['current_step'] = 5  # Go to analysis
        st.success(
            f"✓ Project loaded successfully!\n\n"
            f"- {len(factors)} factor(s)\n"
            f"- {len(st.session_state.get('responses', {}))} response(s)\n"
            f"- Navigating to analysis..."
        )
    elif st.session_state.get('design') is not None:
        st.session_state['current_step'] = 4  # Go to import
        st.success(
            f"✓ Project loaded successfully!\n\n"
            f"- {len(factors)} factor(s)\n"
            f"- Design with {len(st.session_state['design'])} runs\n"
            f"- Navigating to data import..."
        )
    elif st.session_state.get('design_type'):
        st.session_state['current_step'] = 3  # Go to preview
        st.success(
            f"✓ Project loaded successfully!\n\n"
            f"- {len(factors)} factor(s)\n"
            f"- Design type: {st.session_state['design_type']}\n"
            f"- Navigating to design preview..."
        )
    elif factors:
        st.session_state['current_step'] = 2  # Go to design selection
        st.success(
            f"✓ Project loaded successfully!\n\n"
            f"- {len(factors)} factor(s) defined\n"
            f"- Navigating to design selection..."
        )
    else:
        st.session_state['current_step'] = 1  # Start at factors
        st.success("✓ Project loaded successfully!")


def sanitize_design_columns(design: pd.DataFrame, factors: list) -> pd.DataFrame:
    """
    Sanitize column names in imported design to match factor names.
    
    Use this when importing CSV files that might have unsanitized column names.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design with potentially unsafe column names
    factors : List[Factor]
        Factors with sanitized names
    
    Returns
    -------
    pd.DataFrame
        Design with sanitized column names
    """
    from src.core.factors import sanitize_factor_name
    
    # Build mapping from potential unsafe names to safe names
    factor_map = {}
    for factor in factors:
        safe_name = factor.name
        
        # If safe_name not in design, try to find column that sanitizes to it
        if safe_name not in design.columns:
            for col in design.columns:
                sanitized_col, _ = sanitize_factor_name(col)
                if sanitized_col == safe_name:
                    factor_map[col] = safe_name
                    break
    
    if factor_map:
        design = design.rename(columns=factor_map)
        st.info(
            f"✓ Sanitized {len(factor_map)} column name(s) to match factor definitions"
        )
    
    return design


def display_data_status() -> None:
    """
    Display current data status in sidebar.
    
    Shows summary of loaded design and responses.
    """
    design = st.session_state.get('design')
    responses = st.session_state.get('responses', {})
    factors = st.session_state.get('factors', [])
    
    if design is None and not responses:
        return  # No data loaded
    
    st.sidebar.markdown("### 📊 Current Data")
    
    if design is not None:
        design_type = st.session_state.get('design_metadata', {}).get('design_type', 'Unknown')
        
        st.sidebar.markdown(f"**Design:** {design_type.replace('_', ' ').title()}")
        st.sidebar.markdown(f"- {len(design)} runs")
        st.sidebar.markdown(f"- {len(factors)} factors")
        
        # Show block/phase info if present
        if 'Block' in design.columns:
            n_blocks = design['Block'].nunique()
            st.sidebar.markdown(f"- {n_blocks} blocks")
        
        if 'Phase' in design.columns:
            phases = design['Phase'].value_counts().sort_index()
            if len(phases) > 1:
                st.sidebar.markdown(
                    f"- Phase 1: {phases.get(1, 0)} runs"
                )
                st.sidebar.markdown(
                    f"- Phase 2: {phases.get(2, 0)} runs (augmented)"
                )
    
    if responses:
        st.sidebar.markdown(f"**Responses:** {len(responses)}")
        for name in list(responses.keys())[:3]:  # Show first 3
            st.sidebar.markdown(f"- {name}")
        
        if len(responses) > 3:
            st.sidebar.markdown(f"- ... and {len(responses) - 3} more")
    
    st.sidebar.markdown("---")


def update_workflow_progress_display() -> None:
    """
    Enhanced workflow progress display.
    
    This replaces the display_workflow_progress function
    to include data status.
    """
    progress = get_workflow_progress()
    
    st.sidebar.markdown("### 📋 Workflow Progress")
    
    for i, step_name in enumerate(progress['steps'], start=1):
        if progress['completed'][i-1]:
            icon = "✅"
        elif progress['accessible'][i-1]:
            icon = "⭕"
        else:
            icon = "🔒"
        
        if i == progress['current_step']:
            st.sidebar.markdown(f"**{icon} {i}. {step_name}**")
        else:
            st.sidebar.markdown(f"{icon} {i}. {step_name}")
    
    st.sidebar.markdown("---")
    
    # Add data status below workflow progress
    display_data_status()
