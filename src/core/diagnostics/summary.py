"""
Diagnostic Summary Generation for Design Quality Assessment.

This module ties together all diagnostic components to create comprehensive
design quality reports consumed by augmentation recommendation engines.
"""

from typing import Dict, List, Optional, Tuple, Literal
import numpy as np
import pandas as pd

from src.core.factors import Factor
from src.core.coding import DesignSpace
from src.core.diagnostics import (
    ResponseDiagnostics,
    DesignDiagnosticSummary,
    DesignQualityReport,
    ResponseQualityAssessment,
    Issue
)
from src.core.diagnostics.variance import prediction_variance_stats
from src.core.diagnostics.estimability import (
    compute_vif,
    check_collinearity,
    compute_condition_number,
    identify_high_leverage_points
)


def compute_response_diagnostics(
    design: pd.DataFrame,
    response: np.ndarray,
    response_name: str,
    factors: List[Factor],
    model_terms: List[str],
    fitted_model: object,
    design_metadata: Dict
) -> ResponseDiagnostics:
    """
    Compute complete diagnostics for one response.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with factor columns
    response : np.ndarray
        Response measurements
    response_name : str
        Name of response variable
    factors : List[Factor]
        Factor definitions
    model_terms : List[str]
        Fitted model terms
    fitted_model : object
        Fitted model object (from ANOVAAnalysis)
    design_metadata : dict
        Design metadata with keys:
        - 'design_type': str
        - 'generators': Optional[List[Tuple[str, str]]]
        - 'is_split_plot': bool
    
    Returns
    -------
    ResponseDiagnostics
        Complete diagnostic summary for this response
    
    Notes
    -----
    This function integrates:
    - Model fit quality (R², RMSE, LOF)
    - Aliasing (for fractional designs)
    - Prediction variance
    - VIF and collinearity
    - Effect significance
    
    Examples
    --------
    >>> from src.core.analysis import ANOVAAnalysis
    >>> 
    >>> analysis = ANOVAAnalysis(design, response, factors)
    >>> results = analysis.fit(['A', 'B', 'A*B'])
    >>> 
    >>> metadata = {
    ...     'design_type': 'fractional',
    ...     'generators': [('E', 'ABCD')],
    ...     'is_split_plot': False
    ... }
    >>> 
    >>> diag = compute_response_diagnostics(
    ...     design, response, 'Yield', factors, ['A', 'B', 'A*B'],
    ...     results.fitted_model, metadata
    ... )
    """
    diag = ResponseDiagnostics(response_name=response_name)
    diag.model_terms = list(model_terms)

    # Encode design to coded space for all matrix-based computations.
    # The incoming ``design`` is in natural units (canonical session-state
    # format after Phase 2).  Diagnostics that build model matrices require
    # continuous factors in coded [-1, +1] space.
    design_space = DesignSpace.from_factors(factors)
    design_coded = design_space.encode_dataframe(design)

    # Extract model fit statistics
    if hasattr(fitted_model, 'rsquared'):
        diag.r_squared = float(fitted_model.rsquared)
        diag.adj_r_squared = float(fitted_model.rsquared_adj)

    # RMSE from residuals
    residuals = fitted_model.resid
    diag.rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # Lack of fit — replicate detection uses natural values for grouping
    # (rounded to suppress floating-point noise), so pass natural design.
    diag.lack_of_fit_p_value = _compute_lof_p_value(
        design, residuals, factors, model_terms, response=response
    )

    # Aliasing diagnostics (for fractional designs)
    if design_metadata['design_type'] == 'fractional':
        _add_aliasing_diagnostics(diag, design_metadata, factors)

    # Prediction variance — requires coded design
    sigma_squared = diag.rmse ** 2
    diag.prediction_variance_stats = prediction_variance_stats(
        design_coded, factors, model_terms, sigma_squared
    )

    # VIF and collinearity — requires coded design
    diag.vif_values = compute_vif(design_coded, factors, model_terms)

    # High leverage points — requires coded design
    diag.high_leverage_points = identify_high_leverage_points(
        design_coded, factors, model_terms
    )
    
    # Effect significance (from fitted model)
    if hasattr(fitted_model, 'pvalues'):
        for term, pval in fitted_model.pvalues.items():
            if term == 'Intercept':
                continue
            
            if pval < 0.05:
                diag.significant_effects.append(term)
            elif pval < 0.10:
                diag.marginally_significant.append(term)
    
    # Identify issues and recommend augmentation
    _identify_issues(diag)
    
    return diag


def _compute_lof_p_value(
    design: pd.DataFrame,
    residuals: np.ndarray,
    factors: List[Factor],
    model_terms: List[str],
    response: Optional[np.ndarray] = None,
) -> Optional[float]:
    """
    Compute the lack-of-fit (LOF) F-test p-value.

    Delegates to the shared :func:`src.core.diagnostics.lof.compute_lof_components`
    so the diagnostics engine and the Analyze-page display apply the same
    formula and the same numerical guards.

    Returns ``None`` when the LOF test cannot produce a meaningful result:
    the design has no true replicates, the degrees of freedom are
    insufficient, or pure error is effectively zero (replicate responses
    identical to floating-point precision).  This keeps degenerate
    ``F = MS_LOF / MS_PE`` ratios (e.g. ``F ~ 1e26`` from a machine-epsilon
    denominator) out of downstream reports.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix (natural units) containing factor columns.
    residuals : np.ndarray
        Residuals from the fitted model (length n).
    factors : List[Factor]
        Factor definitions (used to identify factor columns in *design*).
    model_terms : List[str]
        Fitted model terms; ``len(model_terms)`` counts the intercept.
    response : np.ndarray, optional
        Observed response values (length n).  Pure error is estimated from
        within-replicate response variation when provided; otherwise the
        residuals are used (legacy fallback).

    Returns
    -------
    float or None
        p-value from the LOF F-test, or ``None`` when the test is not
        applicable / not meaningful.
    """
    from src.core.diagnostics.lof import compute_lof_components, LOF_OK

    result = compute_lof_components(
        design, residuals, factors, model_terms, response=response
    )
    if result.status != LOF_OK:
        return None
    return result.p_value


def _add_aliasing_diagnostics(
    diag: ResponseDiagnostics,
    metadata: Dict,
    factors: List[Factor]
) -> None:
    """
        Add aliasing diagnostics for fractional factorial designs.
        
        This function modifies the ResponseDiagnostics object in place,
        adding resolution, aliased_effects, and confounded_interactions.
        
        Parameters
        ----------
        diag : ResponseDiagnostics
            Diagnostic object to modify (mutated in place)
        metadata : Dict
            Design metadata containing 'generators'
        factors : List[Factor]
            Factor definitions
        """
    # Import aliasing module (stays in its current location)
    from src.core.aliasing import AliasingEngine
    
    generators = metadata.get('generators')
    if not generators:
        return
    
    k = len(factors)
    engine = AliasingEngine(k, generators)
    
    diag.resolution = engine.resolution
    diag.aliased_effects = engine.alias_structure
    
    # Identify critical confoundings (main effects aliased with 2FI)
    for effect, aliases in engine.alias_structure.items():
        # Main effect (single letter)
        if len(effect) == 1:
            # Check if aliased with 2FI (two letters)
            two_fi_aliases = [a for a in aliases if len(a) == 2]
            if two_fi_aliases:
                for alias in two_fi_aliases:
                    diag.confounded_interactions.append((effect, alias))
    
    # Add issue if resolution is low
    if diag.resolution <= 3:
        diag.add_issue(
            severity='critical' if diag.significant_effects else 'warning',
            category='aliasing',
            description=f"Design has Resolution {diag.resolution} - main effects aliased with 2-factor interactions",
            affected_terms=list(diag.aliased_effects.keys()),
            recommended_action="Consider foldover to increase resolution"
        )


def _identify_issues(diag: ResponseDiagnostics) -> None:
    """Identify issues and populate the issues list."""
    
    # Issue 1: Lack of fit
    if diag.lack_of_fit_p_value is not None and diag.lack_of_fit_p_value < 0.05:
        diag.add_issue(
            severity='critical',
            category='lack_of_fit',
            description=f"Lack of fit detected (p = {diag.lack_of_fit_p_value:.3f})",
            affected_terms=[],
            recommended_action="Add quadratic terms or higher-order interactions to model"
        )
    
    # Issue 2: Low R-squared
    if diag.r_squared < 0.7 and not diag.issues:
        # Only flag if no other critical issues
        diag.add_issue(
            severity='warning',
            category='lack_of_fit',
            description=f"Low R² ({diag.r_squared:.2f}) - model may be inadequate",
            affected_terms=[],
            recommended_action="Consider adding terms or transforming response"
        )
    
    # Issue 3: High prediction variance
    if diag.prediction_variance_stats:
        max_var = diag.prediction_variance_stats.get('max', 0)
        mean_var = diag.prediction_variance_stats.get('mean', 1)
        
        if max_var > 3 * mean_var:
            diag.add_issue(
                severity='warning',
                category='precision',
                description=f"High prediction variance in some regions (max/mean = {max_var/mean_var:.1f})",
                affected_terms=[],
                recommended_action="Add runs in high-variance regions"
            )
    
    # Issue 4: Collinearity
    problematic_vif = [
        term for term, vif in diag.vif_values.items()
        if vif > 10 and not np.isinf(vif)
    ]
    
    if problematic_vif:
        diag.add_issue(
            severity='warning',
            category='estimability',
            description=f"Collinearity detected (VIF > 10): {', '.join(problematic_vif)}",
            affected_terms=problematic_vif,
            recommended_action="Add orthogonalizing runs or remove redundant terms"
        )
    
    # Issue 5: Significant aliased effects (critical for fractional)
    if diag.resolution and diag.resolution <= 3:
        aliased_significant = [
            effect for effect in diag.significant_effects
            if effect in diag.aliased_effects
        ]
        
        if aliased_significant:
            diag.add_issue(
                severity='critical',
                category='aliasing',
                description=f"Significant effects are aliased: {', '.join(aliased_significant)}",
                affected_terms=aliased_significant,
                recommended_action=f"Single-factor foldover on {aliased_significant[0]} (or full foldover)"
            )


def compute_design_diagnostic_summary(
    design: pd.DataFrame,
    responses: Dict[str, np.ndarray],
    fitted_models: Dict[str, object],
    factors: List[Factor],
    model_terms_per_response: Dict[str, List[str]],
    design_metadata: Dict
) -> DesignDiagnosticSummary:
    """
    Compute diagnostics across all responses.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix
    responses : Dict[str, np.ndarray]
        Response measurements (name -> array)
    fitted_models : Dict[str, object]
        Fitted models (name -> model object)
    factors : List[Factor]
        Factor definitions
    model_terms_per_response : Dict[str, List[str]]
        Model terms for each response
    design_metadata : dict
        Design metadata:
        - 'design_type': str
        - 'generators': Optional[List[Tuple[str, str]]]
        - 'is_split_plot': bool
        - 'has_blocking': bool
        - 'has_center_points': bool
    
    Returns
    -------
    DesignDiagnosticSummary
        Complete diagnostic summary
    
    Examples
    --------
    >>> responses = {'Yield': yield_data, 'Purity': purity_data}
    >>> models = {'Yield': yield_model, 'Purity': purity_model}
    >>> terms = {'Yield': ['A', 'B'], 'Purity': ['A', 'B', 'C']}
    >>> 
    >>> summary = compute_design_diagnostic_summary(
    ...     design, responses, models, factors, terms, metadata
    ... )
    """
    # Compute per-response diagnostics
    response_diagnostics = {}
    
    for response_name in responses:
        diag = compute_response_diagnostics(
            design=design,
            response=responses[response_name],
            response_name=response_name,
            factors=factors,
            model_terms=model_terms_per_response[response_name],
            fitted_model=fitted_models[response_name],
            design_metadata=design_metadata
        )
        response_diagnostics[response_name] = diag
    
    # Create summary
    summary = DesignDiagnosticSummary(
        design_type=design_metadata['design_type'],
        n_runs=len(design),
        n_factors=len(factors),
        response_diagnostics=response_diagnostics,
        is_split_plot=design_metadata.get('is_split_plot', False),
        has_blocking=design_metadata.get('has_blocking', False),
        has_center_points=design_metadata.get('has_center_points', False),
        generators=design_metadata.get('generators'),
        original_design=design,
        factors=factors
    )
    
    # Identify consistent issues
    _identify_consistent_issues(summary)
    
    # Check for conflicting recommendations
    _check_conflicting_recommendations(summary)
    
    return summary


def _identify_consistent_issues(summary: DesignDiagnosticSummary) -> None:
    """Identify issues present across multiple responses."""
    issue_categories = {}
    
    for diag in summary.response_diagnostics.values():
        for issue in diag.issues:
            key = (issue.category, issue.severity)
            if key not in issue_categories:
                issue_categories[key] = []
            issue_categories[key].append(issue.description)
    
    # Issues present in >1 response are "consistent"
    for (category, severity), descriptions in issue_categories.items():
        if len(descriptions) > 1:
            summary.consistent_issues.append(
                f"{category.replace('_', ' ').title()} ({len(descriptions)} responses)"
            )


def _check_conflicting_recommendations(summary: DesignDiagnosticSummary) -> None:
    """Check if responses need different augmentation strategies."""
    strategies = set()
    
    for diag in summary.response_diagnostics.values():
        for issue in diag.issues:
            if issue.severity in ('critical', 'warning'):
                strategies.add(issue.category)
    
    # If more than one primary strategy needed, we have conflict
    summary.conflicting_recommendations = len(strategies) > 1


def generate_quality_report(
    summary: DesignDiagnosticSummary,
    alpha: float = 0.05
) -> DesignQualityReport:
    """
    Generate user-facing design quality report.
    
    Parameters
    ----------
    summary : DesignDiagnosticSummary
        Diagnostic summary
    alpha : float, default=0.05
        Significance level for grading
    
    Returns
    -------
    DesignQualityReport
        Complete quality report with recommendations
    
    Examples
    --------
    >>> report = generate_quality_report(summary)
    >>> print(report.to_markdown())
    >>> 
    >>> for response_name, assessment in report.response_quality.items():
    ...     print(f"{response_name}: {assessment.overall_grade}")
    """
    # Grade each response
    response_quality = {}
    
    for response_name, diag in summary.response_diagnostics.items():
        grade = _grade_response(diag)
        
        assessment = ResponseQualityAssessment(
            response_name=response_name,
            overall_grade=grade,
            issues=diag.issues,
            recommendations=[issue.recommended_action for issue in diag.issues],
            diagnostics=diag
        )
        
        response_quality[response_name] = assessment
    
    # Create report
    report = DesignQualityReport(
        summary=summary,
        response_quality=response_quality
    )
    
    # Collect critical issues and warnings
    for response_name, assessment in response_quality.items():
        for issue in assessment.issues:
            if issue.severity == 'critical':
                report.critical_issues.append(f"{response_name}: {issue.description}")
            elif issue.severity == 'warning':
                report.warnings.append(f"{response_name}: {issue.description}")
    
    # Collect satisfactory aspects
    if not report.critical_issues and not report.warnings:
        report.satisfactory_aspects.append("All responses show good model fit")
        report.satisfactory_aspects.append("No critical aliasing or collinearity detected")
    
    # Unified strategy
    report.unified_strategy = summary.get_unified_recommendation()
    
    # Conflict resolution
    if summary.conflicting_recommendations:
        report.conflict_resolution = _generate_conflict_resolution(summary)
    
    return report


def _grade_response(diag: ResponseDiagnostics) -> Literal['Excellent', 'Good', 'Fair', 'Poor', 'Inadequate']:
    """
    Grade response quality based on diagnostics.
    
    Grading criteria:
    - Excellent: R² > 0.95, no issues
    - Good: R² > 0.85, no critical issues
    - Fair: R² > 0.70, minor issues only
    - Poor: R² > 0.50, critical issues
    - Inadequate: R² ≤ 0.50 or severe issues
    """
    has_critical = any(i.severity == 'critical' for i in diag.issues)
    has_warnings = any(i.severity == 'warning' for i in diag.issues)
    
    if has_critical:
        if diag.r_squared > 0.70:
            return 'Poor'
        else:
            return 'Inadequate'
    
    if has_warnings:
        if diag.r_squared > 0.85:
            return 'Good'
        else:
            return 'Fair'
    
    # No issues
    if diag.r_squared > 0.95:
        return 'Excellent'
    elif diag.r_squared > 0.85:
        return 'Good'
    elif diag.r_squared > 0.70:
        return 'Fair'
    else:
        return 'Poor'


def _generate_conflict_resolution(summary: DesignDiagnosticSummary) -> str:
    """Generate recommendation for conflicting response needs."""
    priority_response = summary.get_priority_response()
    
    return (
        f"Multiple responses have different needs. "
        f"Recommended approach: Address {priority_response} first "
        f"(highest severity issues), then re-evaluate other responses."
    )