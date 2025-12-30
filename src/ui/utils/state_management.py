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


def get_current_step() -> int:
    """Get current workflow step (1-7)."""
    return st.session_state.get('current_step', 1)


def set_current_step(step: int) -> None:
    """Set current workflow step."""
    if 1 <= step <= 7:
        st.session_state['current_step'] = step


def is_step_complete(step: int) -> bool:
    """
    Check if a workflow step is complete and valid.
    
    Parameters
    ----------
    step : int
        Step number (1-7)
    
    Returns
    -------
    bool
        Whether step is complete
    """
    if step == 1:  # Define Factors
        return len(st.session_state.get('factors', [])) > 0
    
    elif step == 2:  # Choose Design
        return st.session_state.get('design_type') is not None
    
    elif step == 3:  # Preview Design
        return st.session_state.get('design') is not None
    
    elif step == 4:  # Import Results
        return len(st.session_state.get('responses', {})) > 0
    
    elif step == 5:  # Analysis
        return len(st.session_state.get('fitted_models', {})) > 0
    
    elif step == 6:  # Augmentation (optional)
        # Augmentation is optional - considered complete if:
        # - User explicitly skipped, OR
        # - Plan was executed
        return (
            st.session_state.get('augmented_design') is not None or
            not st.session_state.get('show_augmentation', False)
        )
    
    elif step == 7:  # Optimization
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
        "Choose Design",
        "Preview Design",
        "Import Results",
        "Analyze",
        "Augmentation (Optional)",
        "Optimize"
    ]
    
    progress = {
        'total_steps': 7,
        'current_step': get_current_step(),
        'steps': steps,
        'completed': [is_step_complete(i) for i in range(1, 8)],
        'accessible': [can_access_step(i) for i in range(1, 8)]
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
    
    return json.dumps(project, indent=2)


def load_project_file(file_content: str) -> None:
    """
    Load project from JSON file content.
    
    Parameters
    ----------
    file_content : str
        JSON string from uploaded file
    """
    import json
    import pandas as pd
    import numpy as np
    from src.core.factors import Factor, FactorType, ChangeabilityLevel
    
    project = json.loads(file_content)
    
    # Validate version (simple check)
    if 'version' not in project:
        raise ValueError("Invalid project file: missing version")
    
    # Restore factors
    factors = []
    for f_data in project.get('factors', []):
        factor = Factor(
            name=f_data['name'],
            factor_type=FactorType(f_data['type']),
            changeability=ChangeabilityLevel(f_data['changeability']),
            levels=f_data['levels'],
            units=f_data.get('units')
        )
        factors.append(factor)
    
    st.session_state['factors'] = factors
    
    # Restore design settings
    st.session_state['design_type'] = project.get('design_type')
    st.session_state['design_config'] = project.get('design_config', {})
    st.session_state['design_metadata'] = project.get('design_metadata', {})
    
    # Restore design if exists
    if 'design' in project:
        st.session_state['design'] = pd.DataFrame(project['design'])
    
    # Restore responses if exist
    if 'responses' in project:
        responses = {}
        for name, data in project['responses'].items():
            responses[name] = np.array(data)
        st.session_state['responses'] = responses
        st.session_state['response_names'] = project.get('response_names', [])
    
    # Restore model terms
    if 'model_terms_per_response' in project:
        st.session_state['model_terms_per_response'] = project['model_terms_per_response']
    
    # Set current step based on what's loaded
    if st.session_state.get('responses'):
        st.session_state['current_step'] = 5  # Go to analysis
    elif st.session_state.get('design') is not None:
        st.session_state['current_step'] = 4  # Go to import
    elif st.session_state.get('design_type'):
        st.session_state['current_step'] = 3  # Go to preview
    elif factors:
        st.session_state['current_step'] = 2  # Go to design selection
    else:
        st.session_state['current_step'] = 1  # Start at factors