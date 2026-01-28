"""
DOE Toolkit - Main Streamlit Application (Enhanced)

NEW: Multiple workflow entry points including "Start by Importing Data"
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.ui.utils.state_management import (
    initialize_session_state,
    get_workflow_progress
)

# Page configuration
st.set_page_config(
    page_title="DOE Toolkit",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
initialize_session_state()

# Sidebar: Workflow progress and navigation
with st.sidebar:
    st.image("https://via.placeholder.com/200x80/4A90E2/ffffff?text=DOE+Toolkit", use_container_width=True)
    
    # Restart session button at top of sidebar
    if st.button("🔄 Restart Session", use_container_width=True, type="secondary"):
        # Clear all session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    st.divider()
    
    # Use standard sidebar
    from src.ui.components.sidebar import build_standard_sidebar
    build_standard_sidebar()
    
    with st.expander("ℹ️ About DOE Toolkit"):
        st.markdown("""
        **Free, open-source Design of Experiments software**
        
        Features:
        - Full/Fractional Factorial Designs
        - Response Surface Methodology
        - D-Optimal Designs
        - Split-Plot Designs
        - ANOVA Analysis
        - Design Augmentation
        - Response Optimization
        
        **License:** MIT  
        **Version:** 0.1.0 (MVP)
        """)
    
    with st.expander("🆘 Get Help"):
        st.markdown("""
        **Documentation:** [GitHub Wiki](https://github.com/bpimentel3/doe-toolkit)
        
        **Report Issues:** [GitHub Issues](https://github.com/bpimentel3/doe-toolkit/issues)
        
        **Quick Tips:**
        - You can start by defining factors OR importing existing data
        - All data stays on your machine (local-first)
        - Download designs as CSV at any step
        - Use hierarchy enforcement for stable models
        """)

# Main content area
st.title("🔬 DOE Toolkit")
st.markdown("### Professional Design of Experiments for Everyone")

st.markdown("""
Welcome to DOE Toolkit! This free, open-source software helps you design experiments, 
analyze results, and find optimal settings—no programming required.

**Choose how you want to start:**
""")

st.divider()

# Workflow entry points
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🆕 Start Fresh")
    
    st.markdown("""
    **Traditional Workflow**
    
    1. Define your experimental factors
    2. Select model terms
    3. Generate an optimal design
    4. Preview and export design
    5. Run experiments
    6. Import results
    7. Analyze and optimize
    """)
    
    if st.button("1️⃣ Define Factors →", type="primary", use_container_width=True):
        st.session_state['current_step'] = 1
        st.switch_page("pages/1_define_factors.py")
    
    st.caption("Best for: New experiments, need design guidance")

with col2:
    st.markdown("### 📂 Import Existing Data")
    
    st.markdown("""
    **Data-First Workflow**
    
    1. Upload your design + results CSV
    2. Auto-detect factors and responses
    3. Analyze immediately
    4. Augment if needed
    5. Optimize
    """)
    
    if st.button("5️⃣ Import Data →", type="primary", use_container_width=True):
        st.session_state['current_step'] = 5
        st.switch_page("pages/5_import_results.py")
    
    st.caption("Best for: Existing data, completed experiments")

st.divider()

# Continue existing workflow if in progress
progress = get_workflow_progress()

if any(progress['completed']):
    st.markdown("### 📋 Continue Where You Left Off")
    
    # Find first incomplete step
    next_step = None
    for i in range(1, 8):
        if not progress['completed'][i-1] and progress['accessible'][i-1]:
            next_step = i
            break
    
    if next_step:
        step_names = [
            "Define Factors",
            "Select Model",
            "Choose Design",
            "Preview Design", 
            "Import Results",
            "Analyze",
            "Augmentation",
            "Optimize"
        ]
        
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.markdown(f"**Next step:** {step_names[next_step-1]}")
            
            # Show progress
            completed_count = sum(progress['completed'])
            st.progress(completed_count / 8, text=f"{completed_count}/8 steps complete")
        
        with col2:
            st.markdown("")  # Spacer
        
        with col3:
            pages = [
                "pages/1_define_factors.py",
                "pages/2_select_model.py",
                "pages/3_choose_design.py",
                "pages/4_preview_design.py",
                "pages/5_import_results.py",
                "pages/6_analyze.py",
                "pages/7_augmentation.py",
                "pages/8_optimize.py"
            ]
            
            if st.button(f"Continue to Step {next_step} →", type="primary", use_container_width=True):
                st.session_state['current_step'] = next_step
                st.switch_page(pages[next_step-1])

st.divider()

# Feature highlights
st.markdown("### ✨ What Makes DOE Toolkit Different")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("#### 🆓 Free Forever")
    st.markdown("No subscriptions, no licensing fees. MIT open-source license.")

with col2:
    st.markdown("#### 🔒 Privacy First")
    st.markdown("All computation happens locally. Your data never leaves your computer.")

with col3:
    st.markdown("#### 🎓 Professional Quality")
    st.markdown("Match JMP/Design-Expert statistical rigor without the $8,400/year cost.")

with col4:
    st.markdown("#### 🔬 Advanced Features")
    st.markdown("Split-plot ANOVA, D-optimal designs, intelligent augmentation.")

# Example workflows
with st.expander("📚 Example Workflows"):
    st.markdown("""
    **Manufacturing Process Optimization:**
    1. Define factors: Temperature (150-200°C), Pressure (50-100 psi), Time (10-20 min)
    2. Generate Response Surface Design (CCD with 20 runs)
    3. Run experiments and measure Yield
    4. Analyze with quadratic model
    5. Find optimal settings (maximize Yield)
    
    **Import Existing Experiment:**
    1. Upload CSV with completed experiment data
    2. App auto-detects 7 factors and 2 responses
    3. Map any mismatched column names
    4. Analyze immediately with ANOVA
    5. Augment design if quality issues detected
    6. Optimize for best settings
    
    **Pharmaceutical Formulation Screening:**
    1. Define 7 factors (5 continuous, 2 categorical)
    2. Generate Fractional Factorial (Resolution V, 32 runs)
    3. Measure Dissolution Rate and Stability
    4. Analyze split-plot design (mixing = hard to change)
    5. Identify significant factors
    6. Augment with foldover if needed
    """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    DOE Toolkit v0.1.0 | MIT License | 
    <a href='https://github.com/bpimentel3/doe-toolkit'>GitHub</a> | 
    Built with ❤️ for engineers who can't afford expensive software
</div>
""", unsafe_allow_html=True)