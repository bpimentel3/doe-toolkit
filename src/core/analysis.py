"""
ANOVA Analysis Module for Design of Experiments.

Model Term Notation:
- Main effects: 'A', 'B', 'Temperature'
- Interactions: 'A*B', 'Temperature*Pressure'
- Quadratic: 'I(A**2)', 'I(Temperature**2)'
  (uses patsy I() identity operator for Python exponentiation)
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


def generate_model_terms(
    factors: List[Factor],
    model_type: Literal['linear', 'interaction', 'quadratic'],
    include_intercept: bool = True
) -> List[str]:
    """
    Generate standard model terms using Python/patsy notation.
    
    Examples
    --------
    >>> generate_model_terms(factors, 'quadratic')
    ['1', 'A', 'B', 'A*B', 'I(A**2)', 'I(B**2)']
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
    
    # Quadratic terms - use patsy I() notation directly
    if model_type == 'quadratic':
        for factor in factors:
            if factor.is_continuous():
                terms.append(f"I({factor.name}**2)")
    
    return terms


def parse_model_term(term: str) -> Tuple[List[str], str]:
    """Parse model term into factors and operator."""
    if '*' in term and not term.startswith('I('):
        # Interaction: A*B
        factor_list = [f.strip() for f in term.split('*')]
        return factor_list, '*'
    elif term.startswith('I(') and '**' in term:
        # Quadratic: I(A**2)
        inner = term[2:-1]
        base = inner.split('**')[0].strip()
        return [base], '**'
    else:
        # Main effect
        return [term.strip()], ''


def enforce_hierarchy(
    terms: List[str],
    factor_names: List[str]
) -> Tuple[List[str], List[str]]:
    """Enforce model hierarchy and proper term ordering."""
    complete_terms = terms.copy()
    added_terms = []
    
    for term in terms:
        if term == '1':
            continue
        
        factor_list, operator = parse_model_term(term)
        
        if operator in ('*', '**'):
            for factor_name in factor_list:
                if factor_name not in complete_terms and factor_name in factor_names:
                    complete_terms.append(factor_name)
                    added_terms.append(factor_name)
    
    # Remove duplicates
    seen = set()
    unique_terms = [t for t in complete_terms if not (t in seen or seen.add(t))]
    
    # Reorder: intercept, main, interaction, quadratic
    ordered = []
    if '1' in unique_terms:
        ordered.append('1')
    
    for term in unique_terms:
        if term != '1':
            factor_list, op = parse_model_term(term)
            if op == '' and len(factor_list) == 1:
                ordered.append(term)
    
    for term in unique_terms:
        if term != '1':
            factor_list, op = parse_model_term(term)
            if op == '*' and len(factor_list) == 2:
                ordered.append(term)
    
    for term in unique_terms:
        if term != '1':
            factor_list, op = parse_model_term(term)
            if op == '**':
                ordered.append(term)
    
    for term in unique_terms:
        if term not in ordered:
            ordered.append(term)
    
    return ordered, added_terms


def quadratic(factor_name: str) -> str:
    """Helper to generate quadratic term notation."""
    return f"I({factor_name}**2)"


def detect_split_plot_structure(design: pd.DataFrame, factors: List[Factor]) -> Dict:
    """Detect split-plot structure from factor changeability."""
    very_hard = [f.name for f in factors if f.changeability == ChangeabilityLevel.VERY_HARD]
    hard = [f.name for f in factors if f.changeability == ChangeabilityLevel.HARD]
    easy = [f.name for f in factors if f.changeability == ChangeabilityLevel.EASY]
    
    return {
        'is_split_plot': len(hard) > 0 or len(very_hard) > 0,
        'whole_plot_factors': very_hard + hard,
        'sub_plot_factors': easy,
        'has_blocking': 'Block' in design.columns,
        'whole_plot_column': 'WholePlot' if 'WholePlot' in design.columns else None
    }


def prepare_analysis_data(
    design: pd.DataFrame,
    response: Union[np.ndarray, pd.Series],
    factors: List[Factor],
    response_name: str = "Response"
) -> pd.DataFrame:
    """Prepare data for analysis."""
    if len(response) != len(design):
        raise ValueError(f"Response length mismatch: {len(response)} != {len(design)}")
    
    factor_names = [f.name for f in factors]
    analysis_df = design[factor_names].copy()
    analysis_df[response_name] = response
    
    for col in ['Block', 'WholePlot', 'VeryHardPlot', 'Replicate', 'RunOrder', 'StdOrder']:
        if col in design.columns:
            analysis_df[col] = design[col]
    
    return analysis_df


def validate_model_terms(terms: List[str], factors: List[Factor], design: pd.DataFrame) -> None:
    """Validate model terms compatibility."""
    factor_dict = {f.name: f for f in factors}
    
    for term in terms:
        if term == '1':
            continue
        
        factor_list, operator = parse_model_term(term)
        
        for fname in factor_list:
            if fname not in factor_dict:
                raise ValueError(f"Factor '{fname}' in '{term}' not found")
        
        if operator == '**':
            factor = factor_dict[factor_list[0]]
            if not factor.is_continuous():
                raise ValueError(f"Quadratic '{term}' requires continuous factor")
            
            unique_vals = design[factor.name].nunique()
            if unique_vals <= 2:
                warnings.warn(f"Quadratic '{term}': only {unique_vals} levels")


@dataclass
class ANOVAResults:
    """Container for ANOVA results."""
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


class ANOVAAnalysis:
    """ANOVA analysis for experimental designs."""
    
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
        
        self.data = prepare_analysis_data(design, response, factors, response_name)
        self.design_structure = detect_split_plot_structure(design, factors)
        
        if is_split_plot is not None:
            self.design_structure['is_split_plot'] = is_split_plot
        
        self.current_model = None
        self.current_results = None
    
    def fit(self, model_terms: List[str], enforce_hierarchy_flag: bool = True) -> ANOVAResults:
        """Fit ANOVA model. Use I(A**2) notation for quadratic terms."""
        validate_model_terms(model_terms, self.factors, self.design)
        
        if enforce_hierarchy_flag:
            factor_names = [f.name for f in self.factors]
            complete_terms, added = enforce_hierarchy(model_terms, factor_names)
            
            if added:
                warnings.warn(f"Added for hierarchy: {added}")
            
            model_terms = complete_terms
        
        self.current_model = model_terms
        self._validate_degrees_of_freedom(model_terms)
        
        if self.design_structure['is_split_plot']:
            results = self._fit_mixed_effects_model(model_terms)
        else:
            results = self._fit_fixed_effects_model(model_terms)
        
        self.current_results = results
        return results
    
    def _fit_fixed_effects_model(self, model_terms: List[str]) -> ANOVAResults:
        """Fit fixed effects ANOVA."""
        formula = self._build_formula(model_terms)
        
        if self.design_structure['has_blocking'] and not self.block_as_random:
            formula += " + C(Block)"
            fitted_model = ols(formula, data=self.data).fit()
        elif self.design_structure['has_blocking'] and self.block_as_random:
            model = mixedlm(formula, data=self.data, groups=self.data['Block'], re_formula='1')
            fitted_model = model.fit(method='lbfgs')
        else:
            fitted_model = ols(formula, data=self.data).fit()
        
        return self._build_results_object(fitted_model, model_terms, False)
    
    def _fit_mixed_effects_model(self, model_terms: List[str]) -> ANOVAResults:
        """Fit split-plot mixed model."""
        if self.design_structure['whole_plot_column'] is None:
            raise ValueError("Split-plot requires 'WholePlot' column")
        
        formula = self._build_formula(model_terms)
        
        if self.design_structure['has_blocking']:
            self.data['Block_WholePlot'] = (
                self.data['Block'].astype(str) + '_' + 
                self.data[self.design_structure['whole_plot_column']].astype(str)
            )
            groups = self.data['Block_WholePlot']
        else:
            groups = self.data[self.design_structure['whole_plot_column']]
        
        model = mixedlm(formula, data=self.data, groups=groups, re_formula='1')
        fitted_model = model.fit(method='lbfgs')
        
        return self._build_results_object(fitted_model, model_terms, True)
    
    def _build_formula(self, model_terms: List[str]) -> str:
        """Build formula - terms already in patsy notation."""
        terms = [t for t in model_terms if t != '1']
        formula_rhs = ' + '.join(terms)
        return f"{self.response_name} ~ {formula_rhs}"
    
    def _build_results_object(self, fitted_model, model_terms: List[str], is_split_plot: bool) -> ANOVAResults:
        """Build results from fitted model."""
        try:
            if hasattr(fitted_model, 'anova_table'):
                anova_table = fitted_model.anova_table()
            else:
                anova_table = sm.stats.anova_lm(fitted_model, typ=2)
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(f"Could not compute ANOVA: {e}")
            anova_table = pd.DataFrame()
        
        effect_estimates = pd.DataFrame({
            'Coefficient': fitted_model.params,
            'Std_Error': fitted_model.bse,
            't_value': fitted_model.tvalues,
            'p_value': fitted_model.pvalues
        })
        
        residuals = fitted_model.resid
        fitted_values = fitted_model.fittedvalues
        
        if hasattr(fitted_model, 'rsquared'):
            r_squared = fitted_model.rsquared
            adj_r_squared = fitted_model.rsquared_adj
        else:
            ss_total = np.sum((self.data[self.response_name] - self.data[self.response_name].mean())**2)
            ss_resid = np.sum(residuals**2)
            r_squared = 1 - ss_resid / ss_total
            n, p = len(self.data), len(fitted_model.params)
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        
        rmse = np.sqrt(np.mean(residuals**2))
        diagnostics = self._compute_diagnostics(residuals, fitted_values)
        
        # LogWorth
        logworth_df = effect_estimates[effect_estimates.index != 'Intercept'].copy()
        logworth_values = []
        for p in logworth_df['p_value']:
            if pd.isna(p) or p <= 0:
                logworth_values.append(np.nan)
            elif p < 1e-16:
                logworth_values.append(16.0)
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
    
    def _compute_diagnostics(self, residuals: np.ndarray, fitted_values: np.ndarray) -> Dict:
        """Compute diagnostics."""
        diagnostics = {}
        if len(residuals) <= 5000:
            stat, pval = stats.shapiro(residuals)
            diagnostics['shapiro_wilk'] = {'statistic': stat, 'p_value': pval}
        return diagnostics
    
    def _validate_degrees_of_freedom(self, model_terms: List[str]) -> None:
        """Validate DF and warn about saturation."""
        n_runs = len(self.data)
        n_params = len([t for t in model_terms if t != '1']) + 1
        df_error = n_runs - n_params
        
        if df_error < 0:
            warnings.warn(f"Oversaturated: {n_runs} runs, {n_params} params, df={df_error}")
        elif df_error == 0:
            warnings.warn(f"Saturated: df=0, no error estimate possible")
        elif df_error < 3:
            warnings.warn(f"Low df={df_error}, unreliable inference")
        
        if self.design_structure['is_split_plot']:
            n_wp = self.data[self.design_structure['whole_plot_column']].nunique()
            if n_wp < 3:
                warnings.warn(f"Only {n_wp} whole-plots, need ≥3")
    
    def update_model(
        self,
        terms_to_add: Optional[List[str]] = None,
        terms_to_remove: Optional[List[str]] = None,
        enforce_hierarchy_flag: bool = True
    ) -> ANOVAResults:
        """Update model by adding/removing terms."""
        if self.current_model is None:
            raise ValueError("No model fitted yet")
        
        new_terms = self.current_model.copy()
        
        if terms_to_add:
            new_terms.extend(terms_to_add)
        
        if terms_to_remove:
            new_terms = [t for t in new_terms if t not in terms_to_remove]
        
        seen = set()
        new_terms = [t for t in new_terms if not (t in seen or seen.add(t))]
        
        return self.fit(new_terms, enforce_hierarchy_flag)