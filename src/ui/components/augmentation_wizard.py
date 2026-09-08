"""
Augmentation wizard components for Streamlit UI.

Workflow:
1. Display plain-text diagnostic summary (issues + recommendations).
2. Flat augmentation type selection menu.
3. Plan display and execution.
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
    get_available_augmentation_types,
    create_plan_comparison_table
)
from src.core.diagnostics import DesignDiagnosticSummary


def display_diagnostic_summary(
    diagnostics: DesignDiagnosticSummary
) -> None:
    """
    Display a plain-text summary of current design issues and recommendations.

    Replaces the former Mode A interactive workflow.  Issues are surfaced
    as inline callouts so the user understands the context before choosing
    an augmentation type below.

    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics.
    """
    st.subheader("📋 Design Diagnostic Summary")

    any_issues = (
        diagnostics.has_aliasing
        or diagnostics.has_high_vif
        or diagnostics.has_lack_of_fit
        or diagnostics.has_rank_deficiency
        or diagnostics.has_insufficient_replication
    )

    if not any_issues:
        st.success(
            "✅ **No critical issues detected.** Your design appears statistically sound. "
            "You may still augment below to improve precision or add curvature detection."
        )
        return

    st.markdown(
        "The following issues were detected in your current design. "
        "Select an augmentation type below to address them."
    )

    if diagnostics.has_rank_deficiency:
        st.error(
            "❌ **Rank deficiency** — The design matrix is singular and the model "
            "cannot be estimated. Add runs or remove collinear factors. "
            "**Recommended:** D-Optimal augmentation."
        )

    if diagnostics.has_aliasing:
        st.warning(
            "⚠️ **Aliasing** — Some effects are confounded and cannot be separated "
            "from one another. **Recommended:** Foldover augmentation to de-alias "
            "main effects, or D-Optimal augmentation to break specific aliases."
        )

    if diagnostics.has_high_vif:
        st.warning(
            "⚠️ **High multicollinearity (VIF)** — Correlated factors reduce "
            "statistical power and inflate coefficient uncertainty. "
            "**Recommended:** D-Optimal augmentation to improve orthogonality."
        )

    if diagnostics.has_lack_of_fit:
        st.warning(
            "⚠️ **Lack of fit** — The current model does not adequately explain "
            "the response surface. Curvature may be present. "
            "**Recommended:** Add axial points (CCD) or use I-Optimal augmentation "
            "to extend to a response surface model."
        )

    if diagnostics.has_insufficient_replication:
        st.warning(
            "⚠️ **Insufficient replication** — Pure error cannot be reliably "
            "estimated without replicated runs. "
            "**Recommended:** Add replicate runs at existing design points."
        )


def display_type_selection(
    diagnostics: DesignDiagnosticSummary,
) -> Optional[str]:
    """
    Display a flat list of augmentation types for Mode B direct selection.

    Eligible types are shown with a select button.  Ineligible types are
    shown greyed-out with a short explanation of why they are locked.

    Parameters
    ----------
    diagnostics : DesignDiagnosticSummary
        Current design diagnostics used to determine eligibility.

    Returns
    -------
    str or None
        The selected augmentation type key (e.g. 'foldover'), or None if
        the user has not yet made a selection.
    """
    st.subheader("🔬 Choose an Augmentation Type")
    st.markdown(
        "Select the type of augmentation you want to add. "
        "Options that are not applicable to your current design are shown greyed out."
    )
    st.divider()

    aug_types = get_available_augmentation_types(diagnostics)
    selected_type: Optional[str] = None

    for entry in aug_types:
        eligible: bool = entry['eligible']
        lock_reason: Optional[str] = entry['lock_reason']

        with st.container():
            col_text, col_btn = st.columns([5, 1])

            with col_text:
                if eligible:
                    st.markdown(f"### {entry['label']}")
                else:
                    # Visual de-emphasis for locked options
                    st.markdown(
                        f"<span style='color:#888; font-size:1.1rem;'>"
                        f"<strong>🔒 {entry['label']}</strong></span>",
                        unsafe_allow_html=True,
                    )

                st.markdown(entry['description'])

                with st.expander("ℹ️ Details", expanded=False):
                    st.markdown(f"**When to use:** {entry['when_to_use']}")
                    st.markdown(f"**Typical runs added:** {entry['typical_runs']}")
                    if lock_reason:
                        st.info(f"🔒 **Not available:** {lock_reason}")

            with col_btn:
                if eligible:
                    if st.button(
                        "Select",
                        key=f"select_type_{entry['type']}",
                        type="primary",
                        width="stretch",
                    ):
                        selected_type = entry['type']
                else:
                    st.button(
                        "Locked",
                        key=f"locked_type_{entry['type']}",
                        disabled=True,
                        width="stretch",
                        help=lock_reason or "Not available for this design.",
                    )

            st.divider()

    return selected_type


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
    
    st.header("🔧 Augmentation Plans")
    intro = (
        "Here are plans for the selected augmentation type, "
        "ordered by number of runs added:"
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
            width='stretch',
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
    if is_primary:
        title += " (Recommended)"
    
    with st.expander(
        f"{title} — +{plan.n_runs_to_add} runs",
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
            st.metric("Runs Added", plan.n_runs_to_add)
            st.metric("Total After", plan.total_runs_after)
        
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
        
        # Parameter adjustment
        with st.expander("⚙️ Adjust Parameters (Advanced)", expanded=False):
            st.markdown("**Customize this plan:**")

            # Foldover run count is fixed (always doubles the design)
            if plan.strategy == 'foldover':
                st.markdown(f"**Runs to add:** {plan.n_runs_to_add} (fixed for foldover)")
                config = plan.strategy_config
                if config.foldover_type == 'single_factor':
                    st.markdown(f"**Foldover factor:** {config.factor_to_fold}")
            else:
                # Run count adjustment — writes back into the plan immediately
                adjusted_runs = st.number_input(
                    "Number of runs to add",
                    min_value=1,
                    max_value=plan.n_runs_to_add * 5,
                    value=plan.n_runs_to_add,
                    key=f"adjust_runs_{plan.plan_id}"
                )

                if adjusted_runs != plan.n_runs_to_add:
                    # Mutate plan in place so the Select button picks up the change
                    plan.n_runs_to_add = adjusted_runs
                    plan.total_runs_after = len(plan.original_design) + adjusted_runs
                    plan.experimental_cost = float(adjusted_runs)
                    # Keep config in sync for strategies that store it there
                    if hasattr(plan.strategy_config, 'n_runs_to_add'):
                        plan.strategy_config.n_runs_to_add = adjusted_runs
                    st.info(f"Will add {adjusted_runs} runs (click Select to confirm)")
        
        # Selection button
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button(
                f"Select Plan {rank}",
                key=f"select_plan_{plan.plan_id}",
                type="primary" if rank == 1 else "secondary",
                width='stretch'
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
            st.session_state['augmentation_results_displayed'] = True
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
            width='stretch',
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
            width='stretch',
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
