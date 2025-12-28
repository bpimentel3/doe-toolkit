"""
ANOVA Analysis Module for Design of Experiments.

This module provides comprehensive analysis capabilities including:
- Regular factorial ANOVA with proper error terms
- Split-plot ANOVA with multiple error terms (whole-plot, sub-plot)
- Blocked design support (Block as fixed or random effect)
- Model term selection with hierarchy enforcement
- Effect estimation with standard errors
- Diagnostic computations (no plotting - separation of concerns)

Refactored: removed plotting, improved API clarity, eliminated shadowing.
"""

import warnings
from typing import List, Dict, Optional, Union, Literal, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm

from src.core.factors import Factor, FactorType, ChangeabilityLevel


# ============================================================
# SECTION 1: MODEL TERM GENERATION
# ============================================================


def generate_model_terms(
    factors: List[Factor],
    model_type: Literal['linear', 'interaction', 'quadratic'],
    include_intercept: bool = True
) -> List[str]:
    """
    Generate standard model terms for given factors and model type.
    
    Parameters
    ----------
    factors : List[Factor]
        List of experimental factors
    model_type : {'linear', 'interaction', 'quadratic'}
        Type of model to generate
    include_intercept : bool, default=True
        Whether to include intercept term
    
    Returns
    -------
    List[str]
        List of model term strings
    
    Examples
    --------
    >>> factors = [
    ...     Factor("A", FactorType.CONTINUOUS, levels=[-1, 1]),
    ...     Factor("B", FactorType.CONTINUOUS, levels=[-1, 1])
    ... ]
    >>> generate_model_terms(factors, 'interaction')
    ['1', 'A', 'B', 'A*B']
    """
    terms = []
    
    if include_intercept:
        terms.append('1')
    
    # Main effects
    factor_names = [f.name for f in factors]
    terms.extend(factor_names)
    
    # Two-way interactions
    if model_type in ('interaction', 'quadratic'):
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                terms.append(f"{factor_names[i]}*{factor_names[j]}")
    
    # Quadratic terms (only for continuous factors)
    if model_type == 'quadratic':
        for factor in factors:
            if factor.is_continuous():
                terms.append(f"{factor.name}^2")
    
    return terms


def parse_model_term(term: str) -> Tuple[List[str], str]:
    """
    Parse a model term string into constituent factors and operator.
    
    Parameters
    ----------
    term : str
        Model term (e.g., "A*B", "A^2", "Temperature")
    
    Returns
    -------
    factor_list : List[str]
        List of factor names in term
    operator : str
        Operator ('*' for interaction, '^' for power, '' for main effect)
    
    Examples
    --------
    >>> parse_model_term("A*B")
    (['A', 'B'], '*')
    >>> parse_model_term("Temperature^2")
    (['Temperature'], '^')
    >>> parse_model_term("A")
    (['A'], '')
    """
    if '*' in term:
        factor_list = [f.strip() for f in term.split('*')]
        return factor_list, '*'
    elif '^' in term:
        parts = term.split('^')
        return [parts[0].strip()], '^'
    else:
        return [term.strip()], ''


def enforce_hierarchy(
    terms: List[str],
    factor_names: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Enforce model hierarchy: if A*B is included, ensure A and B are included.
    
    Also reorders terms to proper hierarchy: intercept, main effects, 
    interactions, quadratic terms.
    
    Parameters
    ----------
    terms : List[str]
        Requested model terms
    factor_names : List[str]
        All available factor names
    
    Returns
    -------
    complete_terms : List[str]
        Terms with hierarchy enforced and properly ordered
    added_terms : List[str]
        Terms that were added to enforce hierarchy
    """
    complete_terms = terms.copy()
    added_terms = []
    
    # Check each term
    for term in terms:
        if term == '1':  # Skip intercept
            continue
        
        factor_list, operator = parse_model_term(term)
        
        # For interactions and quadratics, ensure main effects present
        if operator in ('*', '^'):
            for factor_name in factor_list:
                if factor_name not in complete_terms and factor_name in factor_names:
                    complete_terms.append(factor_name)
                    added_terms.append(factor_name)
    
    # Remove duplicates
    seen = set()
    unique_terms = []
    for term in complete_terms:
        if term not in seen:
            seen.add(term)
            unique_terms.append(term)
    
    # Reorder terms: intercept, main effects, interactions, quadratic
    ordered_terms = []
    
    # 1. Intercept first
    if '1' in unique_terms:
        ordered_terms.append('1')
    
    # 2. Main effects (single factors)
    for term in unique_terms:
        if term != '1':
            factor_list, operator = parse_model_term(term)
            if operator == '' and len(factor_list) == 1:
                ordered_terms.append(term)
    
    # 3. Two-way interactions
    for term in unique_terms:
        if term != '1':
            factor_list, operator = parse_model_term(term)
            if operator == '*' and len(factor_list) == 2:
                ordered_terms.append(term)
    
    # 4. Quadratic terms
    for term in unique_terms:
        if term != '1':
            factor_list, operator = parse_model_term(term)
            if operator == '^':
                ordered_terms.append(term)
    
    # 5. Any remaining terms (higher-order interactions, etc.)
    for term in unique_terms:
        if term not in ordered_terms:
            ordered_terms.append(term)
    
    return ordered_terms, added_terms


# ============================================================
# SECTION 2: DESIGN STRUCTURE DETECTION
# ============================================================


def detect_split_plot_structure(
    design: pd.DataFrame,
    factors: List[Factor]
) -> Dict[str, any]:
    """
    Detect split-plot structure from factor changeability.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix
    factors : List[Factor]
        Factor definitions
    
    Returns
    -------
    dict
        Structure information:
        - is_split_plot: bool
        - whole_plot_factors: List[str]
        - sub_plot_factors: List[str]
        - has_blocking: bool
        - whole_plot_column: Optional[str]
    """
    # Separate factors by changeability
    very_hard = [f.name for f in factors if f.changeability == ChangeabilityLevel.VERY_HARD]
    hard = [f.name for f in factors if f.changeability == ChangeabilityLevel.HARD]
    easy = [f.name for f in factors if f.changeability == ChangeabilityLevel.EASY]
    
    is_split_plot = len(hard) > 0 or len(very_hard) > 0
    
    # Detect whole-plot identifier column
    whole_plot_column = None
    if 'WholePlot' in design.columns:
        whole_plot_column = 'WholePlot'
    
    # Detect blocking
    has_blocking = 'Block' in design.columns
    
    return {
        'is_split_plot': is_split_plot,
        'whole_plot_factors': very_hard + hard,
        'sub_plot_factors': easy,
        'has_blocking': has_blocking,
        'whole_plot_column': whole_plot_column
    }


# ============================================================
# SECTION 3: DATA PREPARATION
# ============================================================


def prepare_analysis_data(
    design: pd.DataFrame,
    response: Union[np.ndarray, pd.Series],
    factors: List[Factor],
    response_name: str = "Response"
) -> pd.DataFrame:
    """
    Prepare data for analysis by combining design and response.
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with factor columns
    response : array-like
        Response measurements
    factors : List[Factor]
        Factor definitions
    response_name : str
        Name for response column
    
    Returns
    -------
    pd.DataFrame
        Combined data with factor columns, response, and metadata
    """
    if len(response) != len(design):
        raise ValueError(
            f"Response length ({len(response)}) must match design length ({len(design)})"
        )
    
    # Create analysis dataframe
    factor_names = [f.name for f in factors]
    
    # Start with factor columns from design
    analysis_df = design[factor_names].copy()
    
    # Add response
    analysis_df[response_name] = response
    
    # Add metadata columns if present
    for col in ['Block', 'WholePlot', 'VeryHardPlot', 'Replicate', 'RunOrder', 'StdOrder']:
        if col in design.columns:
            analysis_df[col] = design[col]
    
    return analysis_df


def validate_model_terms(
    terms: List[str],
    factors: List[Factor],
    design: pd.DataFrame
) -> None:
    """
    Validate that model terms are compatible with design.
    
    Parameters
    ----------
    terms : List[str]
        Model terms to validate
    factors : List[Factor]
        Factor definitions
    design : pd.DataFrame
        Design matrix
    
    Raises
    ------
    ValueError
        If terms are incompatible with design
    """
    factor_dict = {f.name: f for f in factors}
    
    for term in terms:
        if term == '1':  # Skip intercept
            continue
        
        factor_list, operator = parse_model_term(term)
        
        # Check factors exist
        for fname in factor_list:
            if fname not in factor_dict:
                raise ValueError(f"Factor '{fname}' in term '{term}' not found in factor list")
        
        # Check quadratic terms only for continuous factors with >2 levels
        if operator == '^':
            factor = factor_dict[factor_list[0]]
            if not factor.is_continuous():
                raise ValueError(
                    f"Quadratic term '{term}' only valid for continuous factors, "
                    f"but '{factor.name}' is {factor.factor_type.value}"
                )
            
            # Check if design has >2 unique values for this factor
            unique_vals = design[factor.name].nunique()
            if unique_vals <= 2:
                warnings.warn(
                    f"Quadratic term '{term}' may be problematic: factor '{factor.name}' "
                    f"has only {unique_vals} unique levels in design. "
                    f"Consider center points or more levels."
                )


# ============================================================
# SECTION 4: ANOVA RESULTS CONTAINER
# ============================================================


@dataclass
class ANOVAResults:
    """
    Container for ANOVA analysis results.
    
    Attributes
    ----------
    anova_table : pd.DataFrame
        ANOVA table with SS, MS, F, p-values
    effect_estimates : pd.DataFrame
        Coefficient estimates with SE, t, p-values
    logworth : pd.DataFrame
        LogWorth values (-log10(p-value)) for effect significance
    residuals : np.ndarray
        Model residuals
    fitted_values : np.ndarray
        Fitted response values
    fitted_model : object
        statsmodels fitted model object
    diagnostics : dict
        Diagnostic test results (Shapiro-Wilk, etc.)
    model_terms : List[str]
        Terms included in fitted model
    is_split_plot : bool
        Whether split-plot analysis was used
    r_squared : float
        R-squared value
    adj_r_squared : float
        Adjusted R-squared value
    rmse : float
        Root mean squared error
    """
    anova_table: pd.DataFrame
    effect_estimates: pd.DataFrame
    logworth: pd.DataFrame
    residuals: np.ndarray
    fitted_values: np.ndarray
    fitted_model: object
    diagnostics: Dict[str, any]
    model_terms: List[str]
    is_split_plot: bool
    r_squared: float
    adj_r_squared: float
    rmse: float


# ============================================================
# SECTION 5: MAIN ANOVA ANALYSIS CLASS
# ============================================================


class ANOVAAnalysis:
    """
    ANOVA analysis for experimental designs.
    
    Supports:
    - Regular factorial ANOVA
    - Split-plot ANOVA with proper error terms
    - Blocked designs (Block as fixed or random effect)
    - Model term selection with hierarchy enforcement
    
    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with factor columns
    response : array-like
        Response measurements
    factors : List[Factor]
        Factor definitions with changeability levels
    response_name : str, default="Response"
        Name for response variable
    is_split_plot : bool, optional
        Override automatic split-plot detection. If None, auto-detect from
        factor changeability
    block_as_random : bool, default=False
        If True, treat Block as random effect (mixed model).
        If False, treat Block as fixed effect (standard for complete blocks).
    
    Examples
    --------
    >>> analysis = ANOVAAnalysis(design, response, factors)
    >>> results = analysis.fit(['A', 'B', 'A*B'])
    >>> print(results.anova_table)
    >>> print(results.logworth)  # Significance metrics
    """
    
    def __init__(
        self,
        design: pd.DataFrame,
        response: Union[np.ndarray, pd.Series],
        factors: List[Factor],
        response_name: str = "Response",
        is_split_plot: Optional[bool] = None,
        block_as_random: bool = False
    ):
        self.design = design
        self.response = np.array(response)
        self.factors = factors
        self.response_name = response_name
        self.block_as_random = block_as_random
        
        # Prepare analysis data
        self.data = prepare_analysis_data(design, response, factors, response_name)
        
        # Detect structure
        self.design_structure = detect_split_plot_structure(design, factors)
        
        # Override split-plot detection if specified
        if is_split_plot is not None:
            self.design_structure['is_split_plot'] = is_split_plot
        
        # Current model state
        self.current_model = None
        self.current_results = None
    
    def fit(
        self,
        model_terms: List[str],
        enforce_hierarchy_flag: bool = True
    ) -> ANOVAResults:
        """
        Fit ANOVA model with specified terms.
        
        Parameters
        ----------
        model_terms : List[str]
            Model terms to include (e.g., ['A', 'B', 'A*B', 'A^2'])
        enforce_hierarchy_flag : bool, default=True
            If True, automatically add main effects when interactions present
        
        Returns
        -------
        ANOVAResults
            Analysis results object with precomputed LogWorth values
        
        Raises
        ------
        ValueError
            If model has insufficient degrees of freedom for error estimation
        """
        # Validate terms
        validate_model_terms(model_terms, self.factors, self.design)
        
        # Enforce hierarchy if requested
        if enforce_hierarchy_flag:
            factor_names = [f.name for f in self.factors]
            complete_terms, added = enforce_hierarchy(model_terms, factor_names)
            
            if added:
                warnings.warn(
                    f"Added terms to enforce hierarchy: {added}. "
                    f"Interactions require their main effects."
                )
            
            model_terms = complete_terms
        
        # Store model terms
        self.current_model = model_terms
        
        # Check degrees of freedom BEFORE fitting
        self._validate_degrees_of_freedom(model_terms)
        
        # Fit appropriate model type
        if self.design_structure['is_split_plot']:
            results = self._fit_mixed_effects_model(model_terms)
        else:
            results = self._fit_fixed_effects_model(model_terms)
        
        # Store results
        self.current_results = results
        
        return results
    
    def _fit_fixed_effects_model(self, model_terms: List[str]) -> ANOVAResults:
        """Fit regular factorial ANOVA with fixed effects (including blocking)."""
        # Build formula
        formula = self._build_formula(model_terms)
        
        # Add blocking as fixed effect if present
        if self.design_structure['has_blocking'] and not self.block_as_random:
            formula = formula + " + C(Block)"
            model = ols(formula, data=self.data)
            fitted_model = model.fit()
        elif self.design_structure['has_blocking'] and self.block_as_random:
            # Use mixed model with Block as random effect
            model = mixedlm(
                formula,
                data=self.data,
                groups=self.data['Block'],
                re_formula='1'
            )
            fitted_model = model.fit(method='lbfgs')
        else:
            # Regular OLS
            model = ols(formula, data=self.data)
            fitted_model = model.fit()
        
        # Extract results
        results = self._build_results_object(fitted_model, model_terms, is_split_plot=False)
        
        return results
    
    def _fit_mixed_effects_model(self, model_terms: List[str]) -> ANOVAResults:
        """Fit split-plot ANOVA with proper error terms."""
        if self.design_structure['whole_plot_column'] is None:
            raise ValueError(
                "Split-plot analysis requires 'WholePlot' column in design. "
                "Generate design using generate_split_plot_design()."
            )
        
        # Build formula
        formula = self._build_formula(model_terms)
        
        # Fit mixed model with WholePlot as random effect
        # If blocking present, nest WholePlot within Block
        if self.design_structure['has_blocking']:
            # Nested random effects: Block and WholePlot(Block)
            # Workaround: create composite grouping variable
            self.data['Block_WholePlot'] = (
                self.data['Block'].astype(str) + '_' + 
                self.data[self.design_structure['whole_plot_column']].astype(str)
            )
            groups = self.data['Block_WholePlot']
        else:
            groups = self.data[self.design_structure['whole_plot_column']]
        
        model = mixedlm(
            formula,
            data=self.data,
            groups=groups,
            re_formula='1'
        )
        
        fitted_model = model.fit(method='lbfgs')
        
        # Extract results
        results = self._build_results_object(fitted_model, model_terms, is_split_plot=True)
        
        return results
    
    def _build_formula(self, model_terms: List[str]) -> str:
        """Build statsmodels formula from model terms."""
        # Remove intercept term if present (statsmodels adds by default)
        terms = [t for t in model_terms if t != '1']
        
        # Convert ^ to ** for statsmodels
        terms = [t.replace('^', '**') for t in terms]
        
        # Join terms
        formula_rhs = ' + '.join(terms)
        
        # Complete formula
        formula = f"{self.response_name} ~ {formula_rhs}"
        
        return formula
    
    def _build_results_object(
        self,
        fitted_model: object,
        model_terms: List[str],
        is_split_plot: bool
    ) -> ANOVAResults:
        """Build ANOVAResults object from fitted model."""
        # ANOVA table - handle saturated models gracefully
        try:
            if hasattr(fitted_model, 'anova_table'):
                anova_table = fitted_model.anova_table()
            else:
                # For OLS, compute ANOVA table
                anova_table = sm.stats.anova_lm(fitted_model, typ=2)
        except (ValueError, np.linalg.LinAlgError) as e:
            # Saturated or oversaturated model - ANOVA table cannot be computed
            warnings.warn(
                f"Could not compute ANOVA table: {str(e)}\n"
                f"This typically occurs with saturated (df=0) or oversaturated models.\n"
                f"Coefficient estimates will still be available.",
                UserWarning
            )
            # Create empty ANOVA table
            anova_table = pd.DataFrame()
        
        # Effect estimates
        effect_estimates = pd.DataFrame({
            'Coefficient': fitted_model.params,
            'Std_Error': fitted_model.bse,
            't_value': fitted_model.tvalues,
            'p_value': fitted_model.pvalues
        })
        
        # Residuals and fitted values
        residuals = fitted_model.resid
        fitted_values = fitted_model.fittedvalues
        
        # Model fit statistics
        if hasattr(fitted_model, 'rsquared'):
            r_squared = fitted_model.rsquared
            adj_r_squared = fitted_model.rsquared_adj
        else:
            # For mixed models, compute pseudo R-squared
            ss_total = np.sum((self.data[self.response_name] - self.data[self.response_name].mean())**2)
            ss_resid = np.sum(residuals**2)
            r_squared = 1 - ss_resid / ss_total
            
            n = len(self.data)
            p = len(fitted_model.params)
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        
        rmse = np.sqrt(np.mean(residuals**2))
        
        # Diagnostics
        diagnostics = self._compute_diagnostics(residuals, fitted_values)
        
        # Compute LogWorth values (no plotting - for UI layer)
        # Handle case where p-values might be NaN for saturated models
        logworth_df = effect_estimates.copy()
        logworth_df = logworth_df[logworth_df.index != 'Intercept']
        
        # Compute LogWorth, handling perfect fits (p=0 -> LogWorth=infinity)
        # and missing p-values (saturated models)
        logworth_values = []
        for p in logworth_df['p_value']:
            if pd.isna(p) or p <= 0:
                # No p-value available (saturated) or zero p-value
                logworth_values.append(np.nan)
            elif p < 1e-16:  # Near-zero p-value
                logworth_values.append(16.0)  # Cap at -log10(1e-16)
            else:
                logworth_values.append(-np.log10(p))
        
        logworth_df['LogWorth'] = logworth_values
        
        return ANOVAResults(
            anova_table=anova_table,
            effect_estimates=effect_estimates,
            logworth=logworth_df,
            residuals=residuals,
            fitted_values=fitted_values,
            fitted_model=fitted_model,
            diagnostics=diagnostics,
            model_terms=model_terms,
            is_split_plot=is_split_plot,
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            rmse=rmse
        )
    
    def _compute_diagnostics(
        self,
        residuals: np.ndarray,
        fitted_values: np.ndarray
    ) -> Dict[str, any]:
        """Compute diagnostic statistics."""
        diagnostics = {}
        
        # Normality test (Shapiro-Wilk)
        if len(residuals) <= 5000:  # Shapiro-Wilk has sample size limit
            stat, pval = stats.shapiro(residuals)
            diagnostics['shapiro_wilk'] = {'statistic': stat, 'p_value': pval}
        
        return diagnostics
    
    def _validate_degrees_of_freedom(self, model_terms: List[str]) -> None:
        """
        Validate degrees of freedom and warn about saturated/oversaturated models.
        
        Does NOT raise errors - allows fitting saturated models like JMP does.
        
        Parameters
        ----------
        model_terms : List[str]
            Model terms to be fitted
        
        Warnings
        --------
        UserWarning
            For saturated (df=0) or oversaturated (df<0) models
            For low df (df < 3) models
            For split-plot designs with insufficient whole-plots
        """
        n_runs = len(self.data)
        
        # Count parameters (including intercept if present)
        n_params = len([t for t in model_terms if t != '1']) + 1  # +1 for intercept
        
        # Calculate degrees of freedom for error
        df_error = n_runs - n_params
        
        if df_error < 0:
            warnings.warn(
                f"Oversaturated model: more parameters than observations!\n"
                f"  Runs: {n_runs}\n"
                f"  Parameters: {n_params} (including intercept)\n"
                f"  DF for error: {df_error}\n\n"
                f"Model cannot be fit - not enough data to estimate all parameters.\n"
                f"The fit will fail. To fix:\n"
                f"  1. Add more runs (recommended)\n"
                f"  2. Remove terms from model\n"
                f"  3. Use a simpler model (remove interactions/quadratics)",
                UserWarning
            )
        elif df_error == 0:
            warnings.warn(
                f"Saturated model: zero degrees of freedom for error.\n"
                f"  Runs: {n_runs}\n"
                f"  Parameters: {n_params} (including intercept)\n"
                f"  DF for error: 0\n\n"
                f"Perfect fit will be achieved, but:\n"
                f"  • No residual variance estimate\n"
                f"  • No F-tests or p-values available\n"
                f"  • Cannot assess significance of effects\n"
                f"  • ANOVA table will be incomplete\n\n"
                f"To enable statistical inference:\n"
                f"  1. Add center points or replicates\n"
                f"  2. Reduce model complexity",
                UserWarning
            )
        elif df_error < 3:
            warnings.warn(
                f"Low degrees of freedom for error: {df_error}.\n"
                f"  Runs: {n_runs}\n"
                f"  Parameters: {n_params}\n\n"
                f"Statistical inference will be unreliable with df < 3.\n"
                f"Consider adding center points or replicates.\n"
                f"Recommended: df_error >= 3-5 for reliable results.",
                UserWarning
            )
        
        # Additional check for split-plot designs
        if self.design_structure['is_split_plot']:
            n_whole_plots = self.data[self.design_structure['whole_plot_column']].nunique()
            if n_whole_plots < 3:
                warnings.warn(
                    f"Only {n_whole_plots} whole-plots in split-plot design.\n"
                    f"Recommended: >= 3 whole-plots for reliable whole-plot error estimation.",
                    UserWarning
                )
    
    def update_model(
        self,
        terms_to_add: Optional[List[str]] = None,
        terms_to_remove: Optional[List[str]] = None,
        enforce_hierarchy_flag: bool = True
    ) -> ANOVAResults:
        """
        Update current model by adding or removing terms.
        
        Parameters
        ----------
        terms_to_add : List[str], optional
            Terms to add to model
        terms_to_remove : List[str], optional
            Terms to remove from model
        enforce_hierarchy_flag : bool, default=True
            Enforce model hierarchy after changes
        
        Returns
        -------
        ANOVAResults
            Updated analysis results
        """
        if self.current_model is None:
            raise ValueError("No model fitted yet. Call fit() first.")
        
        new_terms = self.current_model.copy()
        
        if terms_to_add:
            new_terms.extend(terms_to_add)
        
        if terms_to_remove:
            new_terms = [t for t in new_terms if t not in terms_to_remove]
        
        # Remove duplicates while preserving order
        seen = set()
        new_terms = [t for t in new_terms if not (t in seen or seen.add(t))]
        
        return self.fit(new_terms, enforce_hierarchy_flag=enforce_hierarchy_flag)