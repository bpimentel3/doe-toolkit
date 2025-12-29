"""
Diagnostic Infrastructure for Design Quality Assessment.

This module provides diagnostic data structures consumed by augmentation
recommendation engines and quality reporting systems.
"""

from typing import Dict, List, Optional, Tuple, Literal
from dataclasses import dataclass, field
import pandas as pd

from src.core.factors import Factor


@dataclass
class Issue:
    """
    A specific design quality issue.
    
    Attributes
    ----------
    severity : {'critical', 'warning', 'info'}
        Issue severity level
    category : {'aliasing', 'lack_of_fit', 'precision', 'estimability', 'other'}
        Issue category
    description : str
        Human-readable description
    affected_terms : List[str]
        Model terms or factors affected
    recommended_action : str
        What to do about it
    
    Examples
    --------
    >>> issue = Issue(
    ...     severity='critical',
    ...     category='aliasing',
    ...     description='Temperature aliased with Pressure*Time',
    ...     affected_terms=['Temperature', 'Pressure*Time'],
    ...     recommended_action='Single-factor foldover on Temperature'
    ... )
    """
    severity: Literal['critical', 'warning', 'info']
    category: Literal['aliasing', 'lack_of_fit', 'precision', 'estimability', 'other']
    description: str
    affected_terms: List[str]
    recommended_action: str


@dataclass
class ResponseDiagnostics:
    """
    Diagnostics for a single response variable.
    
    This is consumed by augmentation recommendation engines to determine
    what type of augmentation (if any) would benefit this response.
    
    Attributes
    ----------
    response_name : str
        Name of response variable
    lack_of_fit_p_value : float, optional
        Lack-of-fit test p-value (if pure error available)
    r_squared : float
        Model R-squared value
    adj_r_squared : float
        Adjusted R-squared value
    rmse : float
        Root mean squared error
    aliased_effects : Dict[str, List[str]]
        Mapping of effects to what they're aliased with
    resolution : int, optional
        Design resolution (for fractional factorials)
    confounded_interactions : List[Tuple[str, str]]
        Critical aliased pairs (main effect aliased with 2FI)
    prediction_variance_stats : Dict[str, float]
        Statistics: min, max, mean, std of prediction variance
    vif_values : Dict[str, float]
        Variance Inflation Factors for each term
    high_leverage_points : List[int]
        Indices of high-leverage observations
    significant_effects : List[str]
        Effects with p < 0.05
    marginally_significant : List[str]
        Effects with 0.05 < p < 0.10
    needs_augmentation : bool
        Whether augmentation is recommended
    augmentation_reasons : List[str]
        Reasons why augmentation recommended
    issues : List[Issue]
        Structured list of issues detected
    
    Examples
    --------
    >>> diag = ResponseDiagnostics(
    ...     response_name='Yield',
    ...     lack_of_fit_p_value=0.008,
    ...     r_squared=0.85,
    ...     adj_r_squared=0.82,
    ...     rmse=2.3,
    ...     aliased_effects={},
    ...     resolution=None,
    ...     confounded_interactions=[],
    ...     prediction_variance_stats={'min': 0.5, 'max': 4.2, 'mean': 1.8, 'std': 1.2},
    ...     vif_values={'A': 1.2, 'B': 1.3, 'A*B': 1.5},
    ...     high_leverage_points=[],
    ...     significant_effects=['A', 'B'],
    ...     marginally_significant=['A*B'],
    ...     needs_augmentation=True,
    ...     augmentation_reasons=['Lack of fit detected', 'High prediction variance'],
    ...     issues=[Issue(...)]
    ... )
    """
    response_name: str
    
    # Model adequacy
    lack_of_fit_p_value: Optional[float] = None
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    rmse: float = 0.0
    
    # Aliasing (for fractional designs)
    aliased_effects: Dict[str, List[str]] = field(default_factory=dict)
    resolution: Optional[int] = None
    confounded_interactions: List[Tuple[str, str]] = field(default_factory=list)
    
    # Precision & estimability
    prediction_variance_stats: Dict[str, float] = field(default_factory=dict)
    vif_values: Dict[str, float] = field(default_factory=dict)
    high_leverage_points: List[int] = field(default_factory=list)
    
    # Effect significance
    significant_effects: List[str] = field(default_factory=list)
    marginally_significant: List[str] = field(default_factory=list)
    
    # Recommendations
    needs_augmentation: bool = False
    augmentation_reasons: List[str] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    
    def add_issue(
        self,
        severity: Literal['critical', 'warning', 'info'],
        category: Literal['aliasing', 'lack_of_fit', 'precision', 'estimability', 'other'],
        description: str,
        affected_terms: List[str],
        recommended_action: str
    ) -> None:
        """Add an issue to the diagnostics."""
        issue = Issue(
            severity=severity,
            category=category,
            description=description,
            affected_terms=affected_terms,
            recommended_action=recommended_action
        )
        self.issues.append(issue)
        
        if severity == 'critical':
            self.needs_augmentation = True
            self.augmentation_reasons.append(description)
    
    def get_primary_issue(self) -> Optional[Issue]:
        """Get most critical issue."""
        critical = [i for i in self.issues if i.severity == 'critical']
        if critical:
            return critical[0]
        
        warnings = [i for i in self.issues if i.severity == 'warning']
        if warnings:
            return warnings[0]
        
        return None


@dataclass
class DesignDiagnosticSummary:
    """
    Complete diagnostic summary across all responses.
    
    This is the central object that augmentation methods consume to make
    recommendations. It aggregates diagnostics from all responses and provides
    cross-response insights.
    
    Attributes
    ----------
    design_type : str
        Type of design ('fractional', 'response_surface', 'd_optimal', etc.)
    n_runs : int
        Number of experimental runs
    n_factors : int
        Number of factors
    response_diagnostics : Dict[str, ResponseDiagnostics]
        Per-response diagnostics
    is_split_plot : bool
        Whether design has split-plot structure
    has_blocking : bool
        Whether design includes blocking
    has_center_points : bool
        Whether design includes center points
    generators : List[Tuple[str, str]], optional
        Generators for fractional designs
    original_design : pd.DataFrame
        Original design matrix
    factors : List[Factor]
        Factor definitions
    consistent_issues : List[str]
        Issues present across multiple responses
    conflicting_recommendations : bool
        Whether responses need different augmentation strategies
    
    Examples
    --------
    >>> summary = DesignDiagnosticSummary(
    ...     design_type='fractional',
    ...     n_runs=16,
    ...     n_factors=5,
    ...     response_diagnostics={'Yield': diag1, 'Purity': diag2},
    ...     is_split_plot=False,
    ...     has_blocking=False,
    ...     has_center_points=False,
    ...     generators=[('E', 'ABCD')],
    ...     original_design=design_df,
    ...     factors=factor_list,
    ...     consistent_issues=['Temperature aliased'],
    ...     conflicting_recommendations=True
    ... )
    """
    design_type: str
    n_runs: int
    n_factors: int
    
    # Per-response diagnostics
    response_diagnostics: Dict[str, ResponseDiagnostics]
    
    # Design-level properties
    is_split_plot: bool = False
    has_blocking: bool = False
    has_center_points: bool = False
    
    # Original design metadata (needed for augmentation)
    generators: Optional[List[Tuple[str, str]]] = None
    original_design: Optional[pd.DataFrame] = None
    factors: Optional[List[Factor]] = None
    
    # Cross-response insights
    consistent_issues: List[str] = field(default_factory=list)
    conflicting_recommendations: bool = False
    
    def get_priority_response(self) -> str:
        """
        Identify which response has most critical issues.
        
        Returns
        -------
        str
            Name of response with highest priority issues
        """
        if not self.response_diagnostics:
            return ""
        
        # Score responses by issue severity
        scores = {}
        for response_name, diag in self.response_diagnostics.items():
            score = 0
            for issue in diag.issues:
                if issue.severity == 'critical':
                    score += 100
                elif issue.severity == 'warning':
                    score += 10
                else:
                    score += 1
            scores[response_name] = score
        
        # Return response with highest score
        if scores:
            return max(scores, key=scores.get)
        
        return list(self.response_diagnostics.keys())[0]
    
    def needs_any_augmentation(self) -> bool:
        """
        Check if any response needs augmentation.
        
        Returns
        -------
        bool
            True if at least one response recommends augmentation
        """
        return any(
            diag.needs_augmentation 
            for diag in self.response_diagnostics.values()
        )
    
    def get_all_critical_issues(self) -> List[Tuple[str, Issue]]:
        """
        Get all critical issues across all responses.
        
        Returns
        -------
        List[Tuple[str, Issue]]
            List of (response_name, issue) pairs for critical issues
        """
        critical = []
        for response_name, diag in self.response_diagnostics.items():
            for issue in diag.issues:
                if issue.severity == 'critical':
                    critical.append((response_name, issue))
        return critical
    
    def get_unified_recommendation(self) -> str:
        """
        Get single recommendation that addresses all responses.
        
        Returns
        -------
        str
            Unified augmentation strategy recommendation
        """
        if not self.needs_any_augmentation():
            return "No augmentation needed - design quality satisfactory"
        
        # Collect all recommended actions
        actions = set()
        for diag in self.response_diagnostics.values():
            for issue in diag.issues:
                if issue.severity in ('critical', 'warning'):
                    actions.add(issue.category)
        
        # Priority order for recommendations
        if 'aliasing' in actions:
            return "Resolve aliasing via foldover (highest priority)"
        elif 'lack_of_fit' in actions:
            return "Add terms to model (lack of fit detected)"
        elif 'precision' in actions:
            return "Improve precision via additional runs"
        elif 'estimability' in actions:
            return "Add runs to improve estimability"
        else:
            return "Consider design augmentation to improve model quality"


@dataclass
class ResponseQualityAssessment:
    """
    Quality assessment for one response.
    
    Used in quality reports displayed to users.
    
    Attributes
    ----------
    response_name : str
        Response variable name
    overall_grade : {'Excellent', 'Good', 'Fair', 'Poor', 'Inadequate'}
        Overall quality grade
    issues : List[Issue]
        All issues for this response
    recommendations : List[str]
        Recommended actions
    diagnostics : ResponseDiagnostics
        Full diagnostic data
    """
    response_name: str
    overall_grade: Literal['Excellent', 'Good', 'Fair', 'Poor', 'Inadequate']
    issues: List[Issue]
    recommendations: List[str]
    diagnostics: ResponseDiagnostics
    
    def primary_issue(self) -> Optional[Issue]:
        """Get most critical issue."""
        return self.diagnostics.get_primary_issue()


@dataclass
class DesignQualityReport:
    """
    Multi-response design quality report with prioritized recommendations.
    
    This is what the UI displays to users after analysis.
    
    Attributes
    ----------
    summary : DesignDiagnosticSummary
        Complete diagnostic summary
    response_quality : Dict[str, ResponseQualityAssessment]
        Quality assessment per response
    unified_strategy : str, optional
        Single strategy to help all responses
    conflict_resolution : str, optional
        How to handle conflicting needs
    critical_issues : List[str]
        Must address before optimization
    warnings : List[str]
        Should address if budget allows
    satisfactory_aspects : List[str]
        What's working well
    """
    summary: DesignDiagnosticSummary
    response_quality: Dict[str, ResponseQualityAssessment]
    
    # Cross-response recommendations
    unified_strategy: Optional[str] = None
    conflict_resolution: Optional[str] = None
    
    # Prioritized action items
    critical_issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    satisfactory_aspects: List[str] = field(default_factory=list)
    
    def to_markdown(self) -> str:
        """
        Format report for display.
        
        Returns
        -------
        str
            Markdown-formatted quality report
        """
        lines = []
        lines.append("# Design Quality Report")
        lines.append("")
        lines.append(f"**Design Type:** {self.summary.design_type}")
        lines.append(f"**Runs:** {self.summary.n_runs}")
        lines.append(f"**Factors:** {self.summary.n_factors}")
        lines.append(f"**Responses Analyzed:** {len(self.summary.response_diagnostics)}")
        lines.append("")
        
        # Overall status
        if self.critical_issues:
            lines.append(f"⚠️ **Status:** {len(self.critical_issues)} critical issue(s)")
        elif self.warnings:
            lines.append(f"⚡ **Status:** {len(self.warnings)} warning(s)")
        else:
            lines.append("✓ **Status:** Design quality satisfactory")
        lines.append("")
        
        # Per-response quality
        lines.append("## Response Quality")
        lines.append("")
        for response_name, assessment in self.response_quality.items():
            lines.append(f"### {response_name} ({assessment.overall_grade})")
            lines.append("")
            
            if assessment.issues:
                for issue in assessment.issues:
                    if issue.severity == 'critical':
                        icon = "✗"
                    elif issue.severity == 'warning':
                        icon = "⚡"
                    else:
                        icon = "ℹ️"
                    
                    lines.append(f"{icon} {issue.description}")
                    lines.append(f"   *Recommendation:* {issue.recommended_action}")
                    lines.append("")
            else:
                lines.append("✓ No issues detected")
                lines.append("")
        
        # Unified recommendation
        if self.unified_strategy:
            lines.append("## Recommended Action")
            lines.append("")
            lines.append(self.unified_strategy)
            lines.append("")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.
        
        Returns
        -------
        dict
            Report as dictionary
        """
        return {
            'design_type': self.summary.design_type,
            'n_runs': self.summary.n_runs,
            'n_factors': self.summary.n_factors,
            'n_responses': len(self.summary.response_diagnostics),
            'critical_issues': self.critical_issues,
            'warnings': self.warnings,
            'satisfactory_aspects': self.satisfactory_aspects,
            'unified_strategy': self.unified_strategy,
            'response_quality': {
                name: {
                    'grade': assessment.overall_grade,
                    'n_issues': len(assessment.issues),
                    'recommendations': assessment.recommendations
                }
                for name, assessment in self.response_quality.items()
            }
        }