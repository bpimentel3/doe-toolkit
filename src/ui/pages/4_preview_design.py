"""
Step 3: Preview and Generate Design

Generate experimental design and preview before running experiments.
"""
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re

from src.ui.utils.state_management import (
    initialize_session_state,
    is_step_complete,
    can_access_step,
    invalidate_downstream_state
)
from src.core.factors import Factor
from src.ui.utils.csv_parser import generate_doe_csv
from src.ui.components.constraint_builder import (
    format_constraint_preview,
    validate_constraints
)


def _validate_response_name(name: str, existing_responses: list) -> bool:
    """
    Validate response name.
    
    Rules:
    - Must be alphanumeric + underscore
    - Must not be duplicate (case-insensitive)
    - Must not be reserved word
    """
    # Check format
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        return False
    
    # Check for duplicates (case-insensitive)
    existing_names = [r['name'].lower() for r in existing_responses]
    if name.lower() in existing_names:
        return False
    
    # Check for reserved words
    reserved = {'I', 'C', 'Q', 'T'}  # patsy/pandas reserved
    if name in reserved:
        return False
    
    return True


# Initialize state
initialize_session_state()

# Check access
if not can_access_step(3):
    st.warning("⚠️ Please complete Steps 1-2 first")
    st.stop()

st.title("Step 3: Preview Design")

factors = st.session_state['factors']
design_type = st.session_state['design_type']
design_config = st.session_state.get('design_config', {})

st.markdown(f"""
**Design Type:** {design_type}  
**Factors:** {len(factors)}
""")

st.divider()

# Response Definition Section (appears BEFORE design preview)
if st.session_state.get('design') is None:
    st.subheader("📊 Define Responses to Measure")
    st.caption("Declare what you'll measure so the CSV template includes these columns")
    
    # Initialize response definitions if not present
    if 'response_definitions' not in st.session_state:
        st.session_state['response_definitions'] = []
    
    response_defs = st.session_state['response_definitions']
    
    # Response definition form
    with st.form("response_form", border=True):
        # Container for response rows
        response_rows = []
        
        # Show existing responses with option to edit/remove
        for i, response in enumerate(response_defs):
            col1, col2, col3 = st.columns([2, 1.5, 0.5])
            
            with col1:
                response_rows.append({
                    'name': st.text_input(
                        "Response Name",
                        value=response.get('name', ''),
                        key=f"response_name_{i}",
                        label_visibility="collapsed"
                    ),
                    'index': i
                })
            
            with col2:
                units = st.text_input(
                    "Units",
                    value=response.get('units', '') if response.get('units') else '',
                    key=f"response_units_{i}",
                    placeholder="e.g., %, mg/mL",
                    label_visibility="collapsed"
                )
                response_rows[i]['units'] = units if units else None
            
            with col3:
                if st.form_submit_button(
                    "❌",
                    key=f"remove_response_{i}",
                    help="Remove this response"
                ):
                    st.session_state['response_definitions'].pop(i)
                    st.rerun()
        
        # Add new empty row for adding response
        col1, col2, col3 = st.columns([2, 1.5, 0.5])
        
        with col1:
            new_name = st.text_input(
                "Response Name",
                value='',
                key="response_name_new",
                label_visibility="collapsed"
            )
        
        with col2:
            new_units = st.text_input(
                "Units",
                value='',
                key="response_units_new",
                placeholder="e.g., %, mg/mL",
                label_visibility="collapsed"
            )
        
        col_left, col_right = st.columns([2.5, 1])
        
        with col_right:
            if st.form_submit_button("+ Add Response", use_container_width=True):
                if new_name.strip():
                    # Validate response name
                    if not _validate_response_name(new_name, st.session_state['response_definitions']):
                        st.error("Response name invalid or already exists")
                    else:
                        st.session_state['response_definitions'].append({
                            'name': new_name.strip(),
                            'units': new_units.strip() if new_units.strip() else None
                        })
                        st.rerun()
                else:
                    st.warning("Response name cannot be empty")
        
        # Update existing responses from form input
        for i, row in enumerate(response_rows):
            if i < len(st.session_state['response_definitions']):
                st.session_state['response_definitions'][i]['name'] = row['name']
                st.session_state['response_definitions'][i]['units'] = row['units']
    
    # Show summary
    if st.session_state['response_definitions']:
        st.write(f"**Defined responses ({len(st.session_state['response_definitions'])}):**")
        for resp in st.session_state['response_definitions']:
            units_str = f" ({resp['units']})" if resp['units'] else ""
            st.caption(f"• {resp['name']}{units_str}")
    else:
        st.info("ℹ️ No responses defined yet. You can still generate a design, but won't have response columns in the CSV.")
    
    st.divider()

# Generate design button
if st.session_state.get('design') is None:
    
    st.subheader("Generate Design")
    
    # Show configuration summary
    with st.expander("📋 Configuration Summary"):
        st.write(f"**Design Type:** {design_type}")
        st.write(f"**Number of Factors:** {len(factors)}")
        
        if design_config:
            st.write("**Configuration:**")
            for key, value in design_config.items():
                st.write(f"- {key}: {value}")
    
    # Seed for reproducibility
    col1, col2 = st.columns([3, 1])
    
    with col1:
        use_seed = st.checkbox("Use Random Seed (for reproducibility)", value=True)
    
    with col2:
        if use_seed:
            seed = st.number_input("Seed", min_value=0, value=42, step=1)
        else:
            seed = None
    
    # Validate and display constraints if D-Optimal
    if design_type == "D-Optimal":
        constraints = st.session_state.get('constraints', [])
        
        if constraints:
            st.info(f"ℹ️ Design will respect {len(constraints)} constraint(s)")
            
            with st.expander("View Constraints"):
                for i, constraint in enumerate(constraints):
                    constraint_str = format_constraint_preview(
                        constraint.coefficients,
                        constraint.bound,
                        constraint.constraint_type
                    )
                    st.code(f"{i+1}. {constraint_str}")
            
            # Validate constraints
            is_valid, warnings = validate_constraints(constraints, factors)
            
            if not is_valid:
                st.error("❌ **Invalid Constraints**")
                for warning in warnings:
                    st.error(warning)
                st.stop()
            
            if warnings:
                for warning in warnings:
                    st.warning(warning)
    
    # Generate button
    if st.button("🔬 Generate Design", type="primary", use_container_width=True):
        
        with st.spinner("Generating design..."):
            try:
                # Import appropriate design generator
                if design_type == "Full Factorial":
                    from src.core.full_factorial import full_factorial
                    
                    design = full_factorial(
                        factors=factors,
                        n_center_points = design_config.get('n_center_points', 0),
                        randomize=st.session_state.get('randomize', True),
                        random_seed=st.session_state.get('random_seed'),
                        n_blocks=st.session_state.get('n_blocks')
                    )
                    
                    metadata = {
                        'design_type': 'full_factorial',
                        'n_levels': design_config.get('n_levels', 2),
                        'is_split_plot': False
                    }
                
                elif design_type == "Fractional Factorial":
                    from src.core.fractional_factorial import FractionalFactorial
                    
                    fraction = st.session_state.get('fraction', '1/2')
                    resolution = st.session_state.get('resolution')
                    generators = st.session_state.get('custom_generators')
                    
                    # Create FractionalFactorial object
                    ff = FractionalFactorial(
                        factors=factors,
                        fraction=fraction,
                        resolution=resolution,
                        generators=generators
                    )
                    
                    # Generate design
                    design = ff.generate(
                        randomize=st.session_state.get('randomize', True),
                        random_seed=st.session_state.get('random_seed'),
                        n_blocks=st.session_state.get('n_blocks')
                    )
                    
                    # Store metadata
                    st.session_state['design_metadata'] = {
                        'fraction': fraction,
                        'resolution': ff.resolution,
                        'generators': ff.generators_algebraic,
                        'defining_relation': ff.defining_relation,
                        'alias_structure': ff.alias_structure
                    }
                
                elif "Response Surface" in design_type:
                    from src.core.response_surface import (
                        CentralCompositeDesign,  
                        BoxBehnkenDesign          
                    )
                    
                    rsd_variant = st.session_state.get('rsd_variant', 'CCD')
    
                    if rsd_variant == 'CCD':
                        alpha = st.session_state.get('ccd_alpha', 'rotatable')
                        center_points = design_config.get('n_center_points', 6)
                        fraction = st.session_state.get('ccd_fraction')  # For fractional CCD
                        
                        # Create CCD object
                        ccd = CentralCompositeDesign(
                            factors=factors,
                            alpha=alpha,
                            center_points=center_points,
                            fraction=fraction
                        )
                        
                        # Generate design
                        design = ccd.generate(
                            randomize=st.session_state.get('randomize', True),
                            random_seed=st.session_state.get('random_seed')
                        )
                        
                        # Store metadata
                        st.session_state['design_metadata'] = {
                            'variant': 'CCD',
                            'alpha': ccd.alpha,
                            'design_type': ccd.design_type,
                            'n_factorial': ccd.n_factorial,
                            'n_axial': ccd.n_axial,
                            'n_center': ccd.n_center
                        }
                        
                    elif rsd_variant == 'Box-Behnken':
                        center_points = design_config.get('n_center_points', 3)
                        
                        # Create Box-Behnken object
                        bbd = BoxBehnkenDesign(
                            factors=factors,
                            center_points=center_points
                        )
                        
                        # Generate design
                        design = bbd.generate(
                            randomize=st.session_state.get('randomize', True),
                            random_seed=st.session_state.get('random_seed')
                        )
                        
                        # Store metadata
                        st.session_state['design_metadata'] = {
                            'variant': 'Box-Behnken',
                            'n_factorial': bbd.n_factorial,
                            'n_center': bbd.n_center
                        }
                
                elif "D-Optimal" in design_type:
                    from src.core.optimal_design import generate_d_optimal_design
                    from src.core.optimal_design import LinearConstraint  # For constraints
                    
                    model_type = st.session_state.get('model_type', 'linear')
                    n_runs = st.session_state.get('n_runs', 20)
                    constraints = st.session_state.get('constraints', [])  # List[LinearConstraint]
                    
                    result = generate_d_optimal_design(
                        factors=factors,
                        model_type=model_type,
                        n_runs=n_runs,
                        constraints=constraints,
                        seed=st.session_state.get('random_seed')
                    )
                    
                    design = result.design_actual
                    
                    # Store metadata
                    st.session_state['design_metadata'] = {
                        'd_efficiency': result.d_efficiency_vs_benchmark,
                        'condition_number': result.condition_number,
                        'n_parameters': result.n_parameters,
                        'converged_by': result.converged_by
                    }
                
                elif design_type == "Latin Hypercube":
                    from src.core.latin_hypercube import generate_latin_hypercube
    
                    n_runs = st.session_state.get('n_runs', 20)
                    criterion = st.session_state.get('lhs_criterion', 'maximin')
                    n_candidates = st.session_state.get('n_candidates', 10)
                    
                    result = generate_latin_hypercube(
                        factors=factors,
                        n_runs=n_runs,
                        criterion=criterion,
                        n_candidates=n_candidates,
                        seed=st.session_state.get('random_seed')
                    )
                    
                    design = result.design
                    
                    # Store metadata
                    st.session_state['design_metadata'] = {
                        'criterion': result.criterion,
                        'criterion_value': result.criterion_value,
                        'n_runs': result.n_runs
                    }
                
                elif "Split-Plot" in design_type:
                    from src.core.split_plot import generate_split_plot_design
                    
                    n_replicates = st.session_state.get('n_replicates', 1)
                    n_center_points = design_config.get('n_center_points', 0)
                    n_blocks = st.session_state.get('n_blocks', 1)
                    
                    result = generate_split_plot_design(
                        factors=factors,
                        n_replicates=n_replicates,
                        n_center_points=n_center_points,
                        n_blocks=n_blocks,
                        randomize_whole_plots=st.session_state.get('randomize', True),
                        randomize_sub_plots=st.session_state.get('randomize', True),
                        seed=st.session_state.get('random_seed')
                    )
                    
                    design = result.design
                    
                    # Store metadata
                    st.session_state['design_metadata'] = {
                        'n_whole_plots': result.n_whole_plots,
                        'n_sub_plots_per_whole_plot': result.n_sub_plots_per_whole_plot,
                        'whole_plot_factors': result.whole_plot_factors,
                        'sub_plot_factors': result.sub_plot_factors,
                        'has_very_hard_factors': result.has_very_hard_factors
                    }
                
                else:
                    st.error(f"Design type '{design_type}' not implemented yet")
                    st.stop()
                
                # Handle design result (might be dict with 'design' key)
                if isinstance(design, dict) and 'design' in design:
                    # Extract generators if present
                    if 'generators' in design:
                        metadata['generators'] = design['generators']
                    design = design['design']
                
                # Save to session state
                st.session_state['design'] = design
                st.session_state['design_metadata'] = metadata
                st.session_state['design_metadata']['design_type'] = design_type
                
                st.success(f"✓ Design generated successfully! ({len(design)} runs)")
                st.rerun()
            
            except Exception as e:
                st.error(f"Design generation failed: {e}")
                st.exception(e)

# If design exists, show preview
else:
    design = st.session_state['design']
    metadata = st.session_state.get('design_metadata', {})
    
    st.success(f"✓ Design generated ({len(design)} runs)")
    
    # Design metrics
    st.subheader("Design Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Runs", len(design))
    
    with col2:
        if metadata.get('is_split_plot'):
            n_wp = design['WholePlot'].nunique() if 'WholePlot' in design.columns else 0
            st.metric("Whole-Plots", n_wp)
        else:
            factor_cols = [f.name for f in factors if f.name in design.columns]
            st.metric("Factors", len(factor_cols))
    
    with col3:
        if 'Block' in design.columns:
            st.metric("Blocks", design['Block'].nunique())
        elif metadata.get('resolution'):
            st.metric("Resolution", metadata['resolution'])
        else:
            st.metric("Design Points", len(design))
    
    with col4:
        # Compute D-efficiency if possible
        try:
            from src.core.optimal_design import evaluate_design
            from src.core.analysis import generate_model_terms
            
            model_terms = metadata.get('model_terms')
            if not model_terms:
                model_terms = generate_model_terms(factors, 'linear', include_intercept=True)
            
            metrics = evaluate_design(design, factors, model_terms)
            st.metric("D-Efficiency", f"{metrics['d_efficiency']:.1f}%")
        except:
            st.metric("D-Efficiency", "—")
    
    st.divider()
    
    # Design preview table
    st.subheader("Design Matrix Preview")
    
    # Show first/last rows toggle
    show_mode = st.radio(
        "Display",
        ["First 10 rows", "Last 10 rows", "Random 10 rows", "Full design"],
        horizontal=True
    )
    
    if show_mode == "First 10 rows":
        preview_df = design.head(10)
    elif show_mode == "Last 10 rows":
        preview_df = design.tail(10)
    elif show_mode == "Random 10 rows":
        preview_df = design.sample(n=min(10, len(design)), random_state=42)
    else:
        preview_df = design
    
    st.dataframe(preview_df, use_container_width=True)
    
    # Additional info
    if metadata.get('generators'):
        with st.expander("🔍 Design Details"):
            st.markdown("**Generators:**")
            for gen in metadata['generators']:
                if isinstance(gen, tuple):
                    st.code(f"{gen[0]} = {gen[1]}")
                else:
                    st.code(gen)
    
    # Show constraints for D-Optimal designs
    if design_type == "D-Optimal" and st.session_state.get('constraints'):
        with st.expander("📌 Applied Constraints"):
            constraints = st.session_state['constraints']
            st.markdown(f"**{len(constraints)} constraint(s) were applied:**")
            for i, constraint in enumerate(constraints):
                constraint_str = format_constraint_preview(
                    constraint.coefficients,
                    constraint.bound,
                    constraint.constraint_type
                )
                st.code(f"{i+1}. {constraint_str}")
            st.info("✅ All design points satisfy these constraints")
    
    if metadata.get('is_split_plot'):
        with st.expander("📊 Split-Plot Structure"):
            if 'WholePlot' in design.columns:
                wp_counts = design['WholePlot'].value_counts().sort_index()
                st.write("**Runs per Whole-Plot:**")
                st.dataframe(wp_counts.to_frame('Runs'))
    
    st.divider()
    
    # Download options
    st.subheader("📥 Export Design")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Design CSV** (for experiments)")
        st.caption("Download this, run your experiments, then add response data")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate CSV with metadata
        csv_content = generate_doe_csv(
            design=design,
            factors=factors,
            response_definitions=st.session_state.get('response_definitions'),
            design_type=design_type,
            design_metadata=metadata
        )
        
        st.download_button(
            label="📥 Download Design CSV",
            data=csv_content,
            file_name=f"doe_design_{timestamp}.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        st.markdown("**Project File** (save session)")
        st.caption("Save your entire project to resume later")
        
        try:
            from src.ui.utils.state_management import create_project_file
            
            project_json = create_project_file()
            
            st.download_button(
                label="💾 Download Project",
                data=project_json,
                file_name=f"doe_project_{timestamp}.doeproject",
                mime="application/json",
                use_container_width=True
            )
        except Exception as e:
            st.error(f"Project export failed: {e}")
    
    st.divider()
    
    # Next steps guidance
    response_count = len(st.session_state.get('response_definitions', []))
    
    if response_count > 0:
        st.success(f"✓ {response_count} response(s) defined. Empty columns in CSV ready for data.")
        resp_names = ", ".join([f"{r['name']}" for r in st.session_state['response_definitions']])
        st.info(f"""
        ### 📋 Next Steps
        
        1. **Download the Design CSV** above
        2. **Run your experiments** using the factor settings in the CSV
        3. **Measure your response(s)** and fill in: {resp_names}
        4. **Return to this app** and proceed to Step 4: Import Results
        
        💡 **Tip:** CSV includes metadata header so you can always see factor definitions.
        """)
    else:
        st.warning("⚠️ No responses defined - CSV won't have response columns")
        st.info("""
        ### 📋 Next Steps
        
        1. **Download the Design CSV** above
        2. **Run your experiments** using the factor settings in the CSV
        3. **Manually add response columns** to the CSV in Excel/spreadsheet
        4. **Return to this app** and proceed to Step 4: Import Results
        """)
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Generate New Design", use_container_width=True):
            st.session_state['design'] = None
            st.session_state['design_metadata'] = {}
            invalidate_downstream_state(from_step=3)
            st.rerun()
    
    with col2:
        if st.button("← Back to Configuration", use_container_width=True):
            st.session_state['current_step'] = 3
            st.switch_page("pages/3_choose_design.py")
    
    with col3:
        if st.button("Import Results →", type="primary", use_container_width=True):
            st.session_state['current_step'] = 5
            st.switch_page("pages/5_import_results.py")

# Navigation
if st.session_state.get('design') is None:
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Configuration", use_container_width=True):
            st.session_state['current_step'] = 3
            st.switch_page("pages/3_choose_design.py")
