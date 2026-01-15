"""
Augmentation wizard components for Streamlit UI.

Displays augmentation recommendations and manages plan selection/execution.
"""
import streamlit as st
import pandas as pd
from typing import List, Optional

from src.core.augmentation.plan import AugmentationPlan, AugmentedDesign


def display_augmentation_plans(plans: List[AugmentationPlan]) -> None:
    """
    Display ranked augmentation plans for user selection.
    
    Parameters
    ----------
    plans : List[AugmentationPlan]
        Ranked augmentation plans from recommendation engine.
    """
    st.header("🔬 Recommended Augmentation Plans")
    
    if not plans:
        st.info("No augmentation plans available.")
        return
    
    # Display each plan
    for i, plan in enumerate(plans, 1):
        with st.expander(
            f"**Plan {i}: {plan.plan_name}** "
            f"(Utility: {plan.utility_score:.0f}/100, +{plan.n_runs_to_add} runs)",
            expanded=(i == 1)  # Expand first plan by default
        ):
            # Strategy overview
            st.write(f"**Strategy:** {plan.strategy}")
            st.write(f"**Runs to add:** {plan.n_runs_to_add} experiments")
            st.write(f"**Total runs after:** {plan.total_runs_after}")
            
            # Benefits
            st.write(f"**Benefits responses:** {', '.join(plan.benefits_responses)}")
            if plan.primary_beneficiary:
                st.write(f"**Primary beneficiary:** {plan.primary_beneficiary}")
            
            # Expected improvements
            if plan.expected_improvements:
                st.write("**Expected improvements:**")
                for metric, improvement in plan.expected_improvements.items():
                    st.write(f"  • {metric}: {improvement}")
            
            # Selection button
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button(
                    f"Select Plan {i}",
                    key=f"select_plan_{i}",
                    type="primary" if i == 1 else "secondary"
                ):
                    st.session_state['selected_augmentation_plan'] = plan
                    st.rerun()


def display_no_augmentation_needed() -> None:
    """
    Display message when design quality is satisfactory.
    """
    st.success("✅ Design Quality Satisfactory")
    st.write(
        "Your current design appears adequate for the responses analyzed. "
        "No augmentation is recommended at this time."
    )
    st.write(
        "**You can proceed to:**\n"
        "- Step 7: Optimization to find optimal factor settings\n"
        "- Or return to Step 5: Analysis to explore diagnostics further"
    )


def display_augmentation_placeholder() -> None:
    """
    Display placeholder when augmentation hasn't been computed yet.
    """
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


def display_plan_execution(plan: AugmentationPlan) -> None:
    """
    Execute selected augmentation plan and display results.
    
    Parameters
    ----------
    plan : AugmentationPlan
        The plan to execute.
    """
    st.header(f"Executing: {plan.plan_name}")
    
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
    """
    Display augmented design details and download options.
    
    Parameters
    ----------
    augmented : AugmentedDesign
        The augmented design to display.
    """
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
