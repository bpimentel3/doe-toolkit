"""
DOE Toolkit - Main Streamlit Application

A free, open-source Design of Experiments toolkit for engineers.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from src.ui.utils.state_management import (
    initialize_session_state,
    display_workflow_progress,
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
    
    st.markdown("---")
    
    # Display workflow progress
    display_workflow_progress()
    
    st.markdown("---")
    
    # Quick navigation
    st.markdown("### Quick Navigation")
    
    progress = get_workflow_progress()
    
    # Page navigation using st.page_link
    # Note: page_link automatically disables inaccessible pages
    st.page_link("app.py", label="🏠 Home", icon="🏠")
    
    st.page_link(
        "pages/1_define_factors.py",
        label="1. Define Factors",
        disabled=not progress['accessible'][0]
    )
    
    st.page_link(
        "pages/2_choose_design.py",
        label="2. Choose Design",
        disabled=not progress['accessible'][1]
    )
    
    st.page_link(
        "pages/3_preview_design.py",
        label="3. Preview Design",
        disabled=not progress['accessible'][2]
    )
    
    st.page_link(
        "pages/4_import_results.py",
        label="4. Import Results",
        disabled=not progress['accessible'][3]
    )
    
    st.page_link(
        "pages/5_analyze.py",
        label="5. Analyze",
        disabled=not progress['accessible'][4]
    )
    
    st.page_link(
        "pages/6_augmentation.py",
        label="6. Augmentation",
        disabled=not progress['accessible'][5]
    )
    
    st.page_link(
        "pages/7_optimize.py",
        label="7. Optimize",
        disabled=not progress['accessible'][6]
    )
    
    st.markdown("---")
    
    # Project file management
    st.markdown("### 💾 Project")
    
    # Save project
    if st.sidebar.button("💾 Save Project", use_container_width=True):
        st.session_state['show_save_project'] = True
    
    if st.session_state.get('show_save_project'):
        try:
            from src.ui.utils.state_management import create_project_file
            from datetime import datetime
            
            project_json = create_project_file()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.sidebar.download_button(
                "📥 Download .doeproject",
                data=project_json,
                file_name=f"doe_project_{timestamp}.doeproject",
                mime="application/json",
                use_container_width=True
            )
            st.sidebar.success("✓ Ready to download!")
        except Exception as e:
            st.sidebar.error(f"Save failed: {e}")
    
    # Load project
    uploaded_project = st.sidebar.file_uploader(
        "📂 Load Project",
        type=['doeproject', 'json'],
        help="Resume from saved project"
    )
    
    if uploaded_project:
        try:
            from src.ui.utils.state_management import load_project_file
            
            project_content = uploaded_project.read().decode('utf-8')
            load_project_file(project_content)
            
            st.sidebar.success("✓ Project loaded!")
            st.rerun()
        except Exception as e:
            st.sidebar.error(f"Load failed: {e}")
    
    st.markdown("---")
    
    # Help & Info
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
        - Start by defining your experimental factors
        - All data stays on your machine (local-first)
        - Download designs as CSV at any step
        - Use hierarchy enforcement for stable models
        """)

# Main content area
st.title("🔬 DOE Toolkit")
st.markdown("### Professional Design of Experiments for Everyone")

# Welcome message
st.markdown("""
Welcome to DOE Toolkit! This free, open-source software helps you design experiments, 
analyze results, and find optimal settings—no programming required.

**Get started by defining your experimental factors** using the sidebar navigation or the button below.
""")

st.divider()

# Quick start section
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 🎯 Step 1: Define Factors")
    st.markdown("""
    Start by defining your experimental factors:
    - Continuous (temperature, pressure)
    - Discrete numeric (RPM settings)
    - Categorical (materials, methods)
    """)
    
    if st.button("Define Factors →", type="primary", use_container_width=True):
        st.session_state['current_step'] = 1
        st.switch_page("pages/1_define_factors.py")

with col2:
    st.markdown("#### 📊 Step 2-5: Design & Analyze")
    st.markdown("""
    - Choose design type
    - Preview runs before experimenting
    - Import your data
    - Fit ANOVA models
    """)
    
    if progress['completed'][0]:
        if st.button("Continue Workflow →", use_container_width=True):
            # Navigate to first incomplete step
            for i in range(1, 8):
                if not progress['completed'][i-1]:
                    st.session_state['current_step'] = i
                    st.switch_page(pages[i-1][0])
                    break

with col3:
    st.markdown("#### 🎯 Step 6-7: Augment & Optimize")
    st.markdown("""
    - Add strategic runs if needed
    - Find optimal factor settings
    - Maximize/minimize responses
    """)
    
    if progress['completed'][4]:
        if st.button("Go to Optimization →", use_container_width=True):
            st.session_state['current_step'] = 7
            st.switch_page("pages/7_optimize.py")

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
    
    **Pharmaceutical Formulation Screening:**
    1. Define 7 factors (5 continuous, 2 categorical)
    2. Generate Fractional Factorial (Resolution V, 32 runs)
    3. Measure Dissolution Rate and Stability
    4. Analyze split-plot design (mixing = hard to change)
    5. Identify significant factors
    6. Augment with foldover if needed
    
    **Chemical Process Development:**
    1. Define factors: Catalyst (A/B/C), pH (4-8), Concentration (10-30%)
    2. Generate D-Optimal Design (20 runs with constraints)
    3. Measure Conversion and Selectivity
    4. Fit multi-response model
    5. Optimize desirability function
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