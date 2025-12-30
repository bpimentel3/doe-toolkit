"""
Step 3: Preview and Generate Design

Generate experimental design and preview before running experiments.
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

from src.ui.utils.state_management import (
    initialize_session_state,
    is_step_complete,
    can_access_step,
    invalidate_downstream_state
)
from src.core.factors import Factor

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
    
    # Generate button
    if st.button("🔬 Generate Design", type="primary", use_container_width=True):
        
        with st.spinner("Generating design..."):
            try:
                # Import appropriate design generator
                if design_type == "Full Factorial":
                    from src.core.full_factorial import generate_full_factorial
                    
                    design = generate_full_factorial(
                        factors=factors,
                        n_levels=design_config.get('n_levels', 2),
                        n_center_points=design_config.get('n_center_points', 0),
                        n_replicates=design_config.get('n_replicates', 1),
                        randomize=design_config.get('randomize', True),
                        seed=seed
                    )
                    
                    metadata = {
                        'design_type': 'full_factorial',
                        'n_levels': design_config.get('n_levels', 2),
                        'is_split_plot': False
                    }
                
                elif design_type == "Fractional Factorial":
                    from src.core.fractional_factorial import generate_fractional_factorial
                    
                    design = generate_fractional_factorial(
                        factors=factors,
                        fraction=design_config.get('fraction', '1/2'),
                        resolution=design_config.get('resolution', 5),
                        n_center_points=design_config.get('n_center_points', 0),
                        generators=design_config.get('custom_generators'),
                        randomize=design_config.get('randomize', True),
                        seed=seed
                    )
                    
                    metadata = {
                        'design_type': 'fractional_factorial',
                        'fraction': design_config.get('fraction'),
                        'resolution': design_config.get('resolution'),
                        'generators': design.get('generators'),  # From design result
                        'is_split_plot': False
                    }
                
                elif "Response Surface" in design_type:
                    from src.core.response_surface import (
                        generate_central_composite,
                        generate_box_behnken
                    )
                    
                    if "CCD" in design_type:
                        design = generate_central_composite(
                            factors=factors,
                            alpha=design_config.get('alpha', 1.0),
                            n_center_points=design_config.get('n_center_points', 5),
                            randomize=design_config.get('randomize', True),
                            seed=seed
                        )
                    else:  # Box-Behnken
                        design = generate_box_behnken(
                            factors=factors,
                            n_center_points=design_config.get('n_center_points', 5),
                            randomize=design_config.get('randomize', True),
                            seed=seed
                        )
                    
                    metadata = {
                        'design_type': 'response_surface',
                        'is_split_plot': False
                    }
                
                elif design_type == "D-Optimal":
                    from src.core.optimal_design import generate_optimal_design
                    from src.core.analysis import generate_model_terms
                    
                    model_type = design_config.get('model_type', 'linear').lower()
                    model_terms = generate_model_terms(factors, model_type, include_intercept=True)
                    
                    design = generate_optimal_design(
                        factors=factors,
                        n_runs=design_config.get('n_runs', len(model_terms) * 2),
                        model_terms=model_terms,
                        criterion='D',
                        seed=seed
                    )
                    
                    metadata = {
                        'design_type': 'd_optimal',
                        'model_terms': model_terms,
                        'is_split_plot': False
                    }
                
                elif design_type == "Latin Hypercube":
                    from src.core.latin_hypercube import generate_latin_hypercube
                    
                    design = generate_latin_hypercube(
                        factors=factors,
                        n_runs=design_config.get('n_runs', len(factors) * 10),
                        criterion=design_config.get('criterion', 'None'),
                        seed=seed
                    )
                    
                    metadata = {
                        'design_type': 'latin_hypercube',
                        'is_split_plot': False
                    }
                
                elif design_type == "Split-Plot":
                    from src.core.split_plot import generate_split_plot_design
                    
                    design = generate_split_plot_design(
                        factors=factors,
                        n_whole_plots=design_config.get('n_whole_plots', 4),
                        subplot_design_type='full_factorial',
                        randomize_whole_plots=design_config.get('randomize_whole_plots', True),
                        randomize_subplots=design_config.get('randomize_subplots', True),
                        seed=seed
                    )
                    
                    metadata = {
                        'design_type': 'split_plot',
                        'is_split_plot': True,
                        'has_blocking': 'Block' in design.columns if isinstance(design, pd.DataFrame) else False
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
        st.caption("Download this, run your experiments, then add response columns")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        st.download_button(
            label="📥 Download Design CSV",
            data=design.to_csv(index=False),
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
    st.info("""
    ### 📋 Next Steps
    
    1. **Download the Design CSV** above
    2. **Run your experiments** using the factor settings in the CSV
    3. **Measure your response(s)** (e.g., Yield, Purity, etc.)
    4. **Add response columns** to the CSV in Excel/spreadsheet
    5. **Return to this app** and proceed to Step 4: Import Results
    
    💡 **Tip:** You can close this app now. When you return later, upload your project file 
    or the completed design+results CSV to continue.
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
            st.session_state['current_step'] = 2
            st.switch_page("pages/2_choose_design.py")
    
    with col3:
        if st.button("Import Results →", type="primary", use_container_width=True):
            st.session_state['current_step'] = 4
            st.switch_page("pages/4_import_results.py")

# Navigation
if st.session_state.get('design') is None:
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("← Back to Configuration", use_container_width=True):
            st.session_state['current_step'] = 2
            st.switch_page("pages/2_choose_design.py")