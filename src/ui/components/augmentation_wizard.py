"""
Augmentation wizard components for Streamlit UI.

Implements dual-mode augmentation workflow:
- Mode A: Fix Issues (diagnostics-driven, automatic)
- Mode B: Enhance Design (user-intent driven, goal-based)
"""
import streamlit as st
import pandas as pd
from typing import List, Optional, Dict

from src.core.augmentation import (
    AugmentationPlan,
    AugmentedDesign,
    AugmentationRequest,
    AugmentationGoal,
    recommend_augmentation,
    get_mode_availability,
    get_mode_recommendations,
    get_available_enhancement_goals,
    create_plan_comparison_table
)
from src.core.diagnostics import DesignDiagnosticSummary


def display_mode_selection(
    diagnostics: DesignDiagnosticSummary
) -> Optional[str]:
    """
    Display mode selection interface.
    
    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics
    
    Returns
    -------
    str or None
        Selected mode ('fix_issues' or 'enhance_design'), or None if no selection
    """
    
    st.header("🔬 Design Augmentation")
    
    # Get mode availability and recommendations
    availability = get_mode_availability(diagnostics)
    recommendations = get_mode_recommendations(diagnostics)
    
    st.markdown("""
    Choose how you want to augment your design:
    
    - **Mode A: Fix Issues** — Automatic recommendations to address detected problems
    - **Mode B: Enhance Design** — Select a goal and get strategies to achieve it
    """)
    
    st.divider()
    
    # Mode A: Fix Issues
    with st.expander(
        "🔧 **Mode A: Fix Issues** (Diagnostics-Driven)",
        expanded=availability['fix_issues']
    ):
        st.markdown(recommendations['fix_issues'])
        
        if availability['fix_issues']:
            st.markdown("""
            **This mode will:**
            - Analyze diagnostic results for your design
            - Identify critical problems (aliasing, rank deficiency, lack of fit, etc.)
            - Recommend targeted augmentations to fix each issue
            - Prioritize fixes by severity and impact
            
            **Best for:** Addressing specific statistical problems before proceeding
            """)
            
            if st.button(
                "🔧 Fix Detected Issues",
                type="primary" if availability['fix_issues'] else "secondary",
                use_container_width=True,
                key="mode_a_button"
            ):
                return 'fix_issues'
        else:
            st.info(
                "✅ No critical issues detected. Your design appears adequate. "
                "Consider Mode B if you want to enhance capabilities."
            )
    
    # Mode B: Enhance Design
    with st.expander(
        "🎯 **Mode B: Enhance Design** (Goal-Driven)",
        expanded=not availability['fix_issues']
    ):
        st.markdown(recommendations['enhance_design'])
        
        st.markdown("""
        **This mode will:**
        - Let you select a high-level engineering goal
        - Recommend strategies to accomplish that goal
        - Use diagnostics to inform (not dictate) the augmentation
        - Allow you to adjust parameters before executing
        
        **Best for:** Proactively improving design capabilities
        """)
        
        if st.button(
            "🎯 Select Enhancement Goal",
            type="primary" if not availability['fix_issues'] else "secondary",
            use_container_width=True,
            key="mode_b_button"
        ):
            return 'enhance_design'
    
    return None


def display_goal_selection(
    diagnostics: DesignDiagnosticSummary
) -> Optional[AugmentationGoal]:
    """
    Display goal selection interface for Mode B.
    
    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics
    
    Returns
    -------
    AugmentationGoal or None
        Selected goal, or None if no selection
    """
    
    st.subheader("🎯 Select Your Enhancement Goal")
    
    st.markdown("""
    Choose what you want to accomplish with this augmentation. 
    Select the goal that best matches your engineering intent.
    """)
    
    # Get available goals
    available_goals = get_available_enhancement_goals(diagnostics)
    
    if not available_goals:
        st.warning("No enhancement goals available for this design type.")
        return None
    
    # Display goals as cards
    selected_goal = None
    
    for goal_info in available_goals:
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"### {goal_info['title']}")
                
                # Show diagnostic alignment if present
                if goal_info['diagnostic_alignment']:
                    st.markdown(goal_info['diagnostic_alignment'])
                
                st.markdown(f"**Description:** {goal_info['description']}")
                
                with st.expander("ℹ️ More Details"):
                    st.markdown(f"**When to use:** {goal_info['when_appropriate']}")
                    st.markdown(f"**Example:** {goal_info['example_scenario']}")
                    st.markdown(f"**Typical strategies:** {goal_info['typical_strategies']}")
            
            with col2:
                if st.button(
                    "Select",
                    key=f"select_goal_{goal_info['goal']}",
                    use_container_width=True
                ):
                    selected_goal = AugmentationGoal(goal_info['goal'])
            
            st.divider()
    
    return selected_goal


def display_augmentation_plans(
    plans: List[AugmentationPlan],
    mode: str
) -> None:
    """
    Display ranked augmentation plans for user selection.
    
    Parameters
    ----------
    plans : List[AugmentationPlan]
        Ranked augmentation plans
    mode : str
        Mode that generated these plans ('fix_issues' or 'enhance_design')
    """
    
    if mode == 'fix_issues':
        st.header("🔧 Recommended Fixes")
        intro = (
            "Based on diagnostic analysis, here are targeted augmentations "
            "to address detected issues, ranked by priority:"
        )
    else:
        st.header("🎯 Recommended Strategies")
        intro = (
            "Here are strategies to accomplish your selected goal, "
            "ranked by effectiveness:"
        )
    
    st.markdown(intro)
    
    if not plans:
        st.info("No augmentation plans generated.")
        return
    
    # Show comparison table
    with st.expander("📊 Plan Comparison Table"):
        comparison = create_plan_comparison_table(plans)
        st.dataframe(
            pd.DataFrame(comparison),
            use_container_width=True,
            hide_index=True
        )
    
    st.divider()
    
    # Display each plan
    for i, plan in enumerate(plans, 1):
        _display_single_plan(plan, i, mode)


def _display_single_plan(
    plan: AugmentationPlan,
    rank: int,
    mode: str
) -> None:
    """Display a single augmentation plan."""
    
    # Extract metadata
    is_primary = plan.metadata.get('is_primary_strategy', rank == 1)
    diagnostic_warnings = plan.metadata.get('diagnostic_warnings', [])
    diagnostic_suggestions = plan.metadata.get('diagnostic_suggestions', [])
    strategy_rationale = plan.metadata.get('strategy_rationale', '')
    
    # Expander title
    title = f"**Plan {rank}: {plan.plan_name}**"
    if is_primary and mode == 'enhance_design':
        title += " (Recommended)"
    
    with st.expander(
        f"{title} — Utility: {plan.utility_score:.0f}/100, +{plan.n_runs_to_add} runs",
        expanded=(rank == 1)
    ):
        # Strategy overview
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"**Strategy:** {plan.strategy.replace('_', ' ').title()}")
            
            if strategy_rationale:
                st.markdown(f"**Why this works:** {strategy_rationale}")
            
            st.markdown(f"**Runs to add:** {plan.n_runs_to_add} experiments")
            st.markdown(f"**Total runs after:** {plan.total_runs_after}")
        
        with col2:
            st.metric("Utility Score", f"{plan.utility_score:.0f}/100")
            st.metric("Experimental Cost", f"{plan.experimental_cost:.0f} runs")
        
        # Expected improvements
        if plan.expected_improvements:
            st.markdown("**Expected Improvements:**")
            for metric, improvement in plan.expected_improvements.items():
                st.markdown(f"- **{metric}:** {improvement}")
        
        # Benefits
        if plan.benefits_responses:
            st.markdown(f"**Benefits:** {', '.join(plan.benefits_responses)}")
            if plan.primary_beneficiary != 'All':
                st.markdown(f"**Primary beneficiary:** {plan.primary_beneficiary}")
        
        # Diagnostic warnings (for Mode B)
        if diagnostic_warnings:
            with st.expander("⚠️ Diagnostic Notes", expanded=False):
                for warning in diagnostic_warnings:
                    st.warning(warning)
                
                for suggestion in diagnostic_suggestions:
                    st.info(suggestion)
        
        # Parameter adjustment (future feature)
        with st.expander("⚙️ Adjust Parameters (Advanced)", expanded=False):
            st.markdown("**Customize this plan:**")
            
            # Run count adjustment
            adjusted_runs = st.number_input(
                "Number of runs to add",
                min_value=1,
                max_value=plan.n_runs_to_add * 3,
                value=plan.n_runs_to_add,
                key=f"adjust_runs_{plan.plan_id}"
            )
            
            if adjusted_runs != plan.n_runs_to_add:
                st.info(f"Adjusted to {adjusted_runs} runs (original: {plan.n_runs_to_add})")
                # TODO: Update plan with new run count
            
            # Strategy-specific adjustments
            if plan.strategy == 'foldover':
                config = plan.strategy_config
                if config.foldover_type == 'single_factor':
                    st.markdown(f"**Foldover factor:** {config.factor_to_fold}")
                    # TODO: Allow changing the factor
        
        # Selection button
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button(
                f"Select Plan {rank}",
                key=f"select_plan_{plan.plan_id}",
                type="primary" if rank == 1 else "secondary",
                use_container_width=True
            ):
                st.session_state['selected_augmentation_plan'] = plan
                st.rerun()


def display_plan_execution(plan: AugmentationPlan) -> None:
    """
    Execute selected augmentation plan and display results.
    
    Parameters
    ----------
    plan : AugmentationPlan
        The plan to execute
    """
    
    st.header(f"Executing: {plan.plan_name}")
    
    # Show plan details
    st.markdown(f"**Strategy:** {plan.strategy.replace('_', ' ').title()}")
    st.markdown(f"**Runs to add:** {plan.n_runs_to_add}")
    
    with st.spinner("Generating augmented design..."):
        try:
            # Execute plan
            augmented = plan.execute()
            
            # Validate
            validation = augmented.validate()
            
            if not validation.is_valid:
                st.error("❌ Augmentation failed validation:")
                for error in validation.errors:
                    st.error(f"  • {error}")
                
                if validation.warnings:
                    st.warning("Warnings:")
                    for warning in validation.warnings:
                        st.warning(f"  • {warning}")
                return
            
            # Success - store in session state
            st.session_state['augmented_design'] = augmented
            st.success(f"✅ Successfully added {augmented.n_runs_added} runs")
            
            # Display results
            _display_augmented_design(augmented)
            
            # Show validation warnings if any
            if validation.warnings:
                with st.expander("⚠️ Validation Warnings"):
                    for warning in validation.warnings:
                        st.warning(f"  • {warning}")
            
        except Exception as e:
            st.error(f"❌ Error executing augmentation plan: {str(e)}")
            st.exception(e)


def _display_augmented_design(augmented: AugmentedDesign) -> None:
    """Display augmented design details and download options."""
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Original Runs", augmented.n_runs_original)
    with col2:
        st.metric("Runs Added", augmented.n_runs_added)
    with col3:
        st.metric("Total Runs", augmented.n_runs_total)
    
    # Quality improvements
    if augmented.achieved_improvements:
        st.subheader("Achieved Improvements")
        for metric, value in augmented.achieved_improvements.items():
            st.write(f"• **{metric}:** {value}")
    
    # Display design tables
    tab1, tab2, tab3 = st.tabs([
        "📋 Complete Design",
        "🆕 New Runs Only",
        "📊 Design Metrics"
    ])
    
    with tab1:
        st.write(
            f"Combined design with {augmented.n_runs_total} total runs. "
            f"The '{augmented.block_column}' column indicates original (1) vs augmented (2) runs."
        )
        st.dataframe(
            augmented.combined_design,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = augmented.combined_design.to_csv(index=False)
        st.download_button(
            label="📥 Download Complete Design (CSV)",
            data=csv,
            file_name="augmented_design_complete.csv",
            mime="text/csv",
            key="download_complete"
        )
    
    with tab2:
        st.write(
            f"Only the {augmented.n_runs_added} new runs to conduct. "
            "Use this to plan your additional experiments."
        )
        st.dataframe(
            augmented.new_runs_only,
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv = augmented.new_runs_only.to_csv(index=False)
        st.download_button(
            label="📥 Download New Runs (CSV)",
            data=csv,
            file_name="augmented_design_new_runs.csv",
            mime="text/csv",
            key="download_new"
        )
    
    with tab3:
        st.write("**Design Quality Metrics:**")
        
        metrics_data = {
            "Metric": [],
            "Value": []
        }
        
        if augmented.resolution is not None:
            metrics_data["Metric"].append("Resolution")
            metrics_data["Value"].append(f"Resolution {augmented.resolution}")
        
        if augmented.d_efficiency is not None:
            metrics_data["Metric"].append("D-Efficiency")
            metrics_data["Value"].append(f"{augmented.d_efficiency:.1f}%")
        
        metrics_data["Metric"].append("Condition Number")
        metrics_data["Value"].append(f"{augmented.condition_number:.2f}")
        
        if metrics_data["Metric"]:
            st.table(pd.DataFrame(metrics_data))
        
        # Alias structure if available
        if augmented.updated_alias_structure:
            with st.expander("🔗 Updated Alias Structure"):
                for effect, aliases in augmented.updated_alias_structure.items():
                    if aliases:
                        st.write(f"**{effect}** = {' = '.join(aliases)}")
    
    # Next steps
    st.info(
        "**Next Steps:**\n\n"
        "1. 📥 Download the new runs CSV\n"
        "2. 🔬 Conduct the additional experiments\n"
        "3. 📊 Combine new results with original data\n"
        "4. ⬆️ Return to Step 4: Import Results and upload combined data\n"
        "5. 🔄 Re-run Step 5: Analysis with the augmented design"
    )


def display_no_augmentation_needed() -> None:
    """Display message when design quality is satisfactory."""
    
    st.success("✅ Design Quality Satisfactory")
    st.write(
        "Your current design appears adequate for the responses analyzed. "
        "No critical issues detected."
    )
    
    st.markdown("""
    **You have three options:**
    
    1. **Proceed to Optimization** — Find optimal factor settings with your current design
    2. **Enhance Capabilities** — Use Mode B to add features (curvature, robustness, etc.)
    3. **Return to Analysis** — Explore diagnostics further
    """)


def display_augmentation_placeholder() -> None:
    """Display placeholder when augmentation hasn't been computed yet."""
    
    st.info("ℹ️ Augmentation Analysis Pending")
    st.write(
        "Augmentation recommendations are generated after analyzing your experimental results."
    )
    st.write(
        "**To access augmentation recommendations:**\n"
        "1. Complete Step 4: Import Results\n"
        "2. Complete Step 5: Analysis (fit ANOVA models)\n"
        "3. Return here to view recommendations"
    )
