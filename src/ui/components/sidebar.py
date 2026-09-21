"""
Reusable sidebar components for DOE Toolkit.

Provides consistent export buttons and other sidebar elements
that should appear across all pages.
"""

import streamlit as st
from datetime import datetime
from typing import Optional

import requests

from src.__version__ import __version__

_GITHUB_RELEASES_URL = (
    "https://api.github.com/repos/bpimentel3/doe-toolkit/releases/latest"
)


def _fetch_latest_version() -> Optional[str]:
    """
    Fetch the latest release tag from GitHub.

    Returns
    -------
    Optional[str]
        The latest version string (e.g. '0.2.0'), or None if unreachable.

    Notes
    -----
    Fails silently — a network error or timeout will return None so the
    app continues to load normally.
    """
    try:
        response = requests.get(_GITHUB_RELEASES_URL, timeout=3)
        response.raise_for_status()
        tag = response.json().get("tag_name", "")
        return tag.lstrip("v") if tag else None
    except Exception:
        return None


def _is_newer(remote: str, local: str) -> bool:
    """
    Compare two semantic version strings.

    Parameters
    ----------
    remote : str
        Version string from GitHub (e.g. '0.2.0').
    local : str
        Current installed version string (e.g. '0.1.0').

    Returns
    -------
    bool
        True if remote is strictly newer than local.
    """
    try:
        remote_parts = tuple(int(x) for x in remote.split("."))
        local_parts = tuple(int(x) for x in local.split("."))
        return remote_parts > local_parts
    except ValueError:
        return False


def add_version_footer() -> None:
    """
    Add version display and update notification to the sidebar footer.

    Displays the current version and, if a newer GitHub release exists,
    shows a banner with a link to the releases page.

    Returns
    -------
    None
    """
    st.sidebar.markdown("---")

    latest = _fetch_latest_version()
    if latest and _is_newer(latest, __version__):
        st.sidebar.warning(
            f"🆕 **v{latest} available!** "
            f"[Download on GitHub](https://github.com/bpimentel3/doe-toolkit/releases)"
        )

    st.sidebar.caption(f"DOE Toolkit v{__version__}")


def add_export_section():
    """
    Add export buttons to sidebar.
    
    This function should be called in the sidebar of every page
    to provide consistent export functionality throughout the app.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📤 Export")
    
    # Save Project button
    if st.sidebar.button("💾 Save Project File", width='stretch', key="save_project_btn"):
        st.session_state['show_save_project'] = True
    
    if st.session_state.get('show_save_project'):
        try:
            from src.ui.utils.state_management import create_project_file
            
            project_json = create_project_file()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            st.sidebar.download_button(
                "📥 Download .doeproject",
                data=project_json,
                file_name=f"doe_project_{timestamp}.doeproject",
                mime="application/json",
                width='stretch',
                key="download_project"
            )
            st.sidebar.success("✓ Ready to download!")
        except Exception as e:
            st.sidebar.error(f"Save failed: {e}")
    
    # Generate Report button
    if st.sidebar.button("📄 Generate HTML Report", width='stretch', key="generate_report_btn"):
        st.session_state['show_generate_report'] = True
    
    if st.session_state.get('show_generate_report'):
        try:
            from src.ui.utils.export import generate_html_report
            
            # Check if there's enough data to generate report
            if not st.session_state.get('factors'):
                st.sidebar.warning("⚠️ No data to export. Define factors first.")
            else:
                with st.spinner("Generating report..."):
                    html_report = generate_html_report()
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    st.sidebar.download_button(
                        "📥 Download Report.html",
                        data=html_report,
                        file_name=f"doe_report_{timestamp}.html",
                        mime="text/html",
                        width='stretch',
                        key="download_report"
                    )
                    st.sidebar.success("✓ Report ready!")
        except Exception as e:
            st.sidebar.error(f"Report generation failed: {e}")


def add_quick_navigation():
    """
    Add quick navigation links to sidebar.
    
    Provides page_link buttons for all workflow steps.
    """
    from src.ui.utils.state_management import get_workflow_progress
    
    st.sidebar.markdown("### Quick Navigation")
    
    progress = get_workflow_progress()
    
    st.sidebar.page_link("app.py", label="🏠 Home")
    
    st.sidebar.page_link(
        "pages/1_define_factors.py",
        label="1. Define Factors",
        disabled=not progress['accessible'][0]
    )
    
    st.sidebar.page_link(
        "pages/2_select_model.py",
        label="2. Select Model",
        disabled=not progress['accessible'][1]
    )
    
    st.sidebar.page_link(
        "pages/3_choose_design.py",
        label="3. Choose Design",
        disabled=not progress['accessible'][2]
    )
    
    st.sidebar.page_link(
        "pages/4_preview_design.py",
        label="4. Preview Design",
        disabled=not progress['accessible'][3]
    )
    
    st.sidebar.page_link(
        "pages/5_import_results.py",
        label="5. Import Results"
        # NO disabled - always accessible
    )
    
    st.sidebar.page_link(
        "pages/6_analyze.py",
        label="6. Analyze",
        disabled=not progress['accessible'][5]
    )
    
    st.sidebar.page_link(
        "pages/7_augmentation.py",
        label="7. Augmentation",
        disabled=not progress['accessible'][6]
    )
    
    st.sidebar.page_link(
        "pages/8_optimize.py",
        label="8. Optimize",
        disabled=not progress['accessible'][7]
    )

    st.sidebar.page_link(
        "pages/0_help.py",
        label="📖 Help & Docs"
    )


def add_project_load():
    """
    Add project load section to sidebar.
    
    Should be called near the top, before workflow progress.
    """
    uploaded_project = st.sidebar.file_uploader(
        "📂 Load Project",
        type=['doeproject', 'json'],
        help="Resume from saved project",
        key="project_uploader"
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
    
    st.sidebar.markdown("---")


def build_standard_sidebar():
    """
    Build the complete standard sidebar for all pages.
    
    Call this function at the start of every page to get
    consistent sidebar layout:
    - Project load
    - Quick navigation (includes progress indicators)
    - Export buttons
    """
    # Hide default Streamlit page selector
    st.markdown("""
        <style>
            [data-testid="stSidebarNav"] {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Project load at top
    add_project_load()
    
    # Quick navigation (with progress indicators built-in)
    add_quick_navigation()
    
    # Export buttons at bottom
    add_export_section()

    # Version display and update check at very bottom
    add_version_footer()
