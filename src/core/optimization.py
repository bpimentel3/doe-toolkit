"""
Response Optimization Module for Design of Experiments.

This module provides optimization capabilities for finding optimal factor
settings based on fitted models from ANOVA analysis. Supports:
- Single-response optimization (maximize, minimize, target)
- Multi-response optimization using desirability functions
- Linear constraints and factor bounds
- Confidence and prediction intervals

References
----------
.. [1] Myers, R. H., Montgomery, D. C., & Anderson-Cook, C. M. (2016).
       Response surface methodology: process and product optimization
       using designed experiments. John Wiley & Sons.
.. [2] Derringer, G., & Suich, R. (1980). Simultaneous optimization of
       several response variables. Journal of Quality Technology, 12(4), 214-219.
"""

import warnings
from typing import List, Dict, Optional, Union, Literal, Tuple, Callable
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint as ScipyLinearConstraint

from src.core.factors import Factor, FactorType
from src.core.analysis import ANOVAResults


# ============================================================
# SECTION 1: PREDICTION WITH UNCERTAINTY
# ============================================================


def predict_with_intervals(
    model: object,
    x_pred: np.ndarray,
    factor_names: List[str],
    alpha: float = 0.05
) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
    """
    Predict response with confidence and prediction intervals.
    
    Raises
    ------
    AttributeError
        If model doesn't support get_prediction (e.g., some mixed-effects models)

    Parameters
    ----------
    model : statsmodels fitted model
        Fitted model from ANOVAResults
    x_pred : np.ndarray
        Factor values at which to predict (actual scale)
    factor_names : List[str]
        Factor names in order
    alpha : float, default=0.05
        Significance level (0.05 gives 95% intervals)
    
    Returns
    -------
    prediction : float
        Predicted mean response
    confidence_interval : Tuple[float, float]
        Confidence interval for mean response
    prediction_interval : Tuple[float, float]
        Prediction interval for individual observation
    
    Notes
    -----
    Confidence interval estimates uncertainty in the mean response at x_pred.
    Prediction interval estimates uncertainty for a single future observation.
    
    PI is always wider than CI because it includes both parameter uncertainty
    and random error variance.
    """
    # Create prediction dataframe
    pred_df = pd.DataFrame([x_pred], columns=factor_names)
    
    # Get prediction with confidence interval
    try:
        pred_result = model.get_prediction(pred_df)
        pred_summary = pred_result.summary_frame(alpha=alpha)
    except AttributeError:
        raise AttributeError(
            "Model does not support get_prediction(). "
            "Interval prediction not available for this model type. "
            "Use model.predict() for point predictions only."
        )
    
    prediction = pred_summary['mean'].values[0]
    ci_lower = pred_summary['mean_ci_lower'].values[0]
    ci_upper = pred_summary['mean_ci_upper'].values[0]
    
    # Prediction interval
    pi_lower = pred_summary['obs_ci_lower'].values[0]
    pi_upper = pred_summary['obs_ci_upper'].values[0]
    
    return prediction, (ci_lower, ci_upper), (pi_lower, pi_upper)


# ============================================================
# SECTION 2: SINGLE-RESPONSE OPTIMIZATION
# ============================================================


@dataclass
class OptimizationResult:
    """
    Result from single-response optimization.
    
    Attributes
    ----------
    optimal_settings : Dict[str, float]
        Optimal factor values
    predicted_response : float
        Predicted response at optimum
    confidence_interval : Tuple[float, float]
        95% CI for mean response
    prediction_interval : Tuple[float, float]
        95% PI for individual observation
    objective_value : float
        Objective function value at optimum
    success : bool
        Whether optimization converged
    message : str
        Optimization status message
    n_iterations : int
        Number of iterations
    """
    optimal_settings: Dict[str, float]
    predicted_response: float
    confidence_interval: Tuple[float, float]
    prediction_interval: Tuple[float, float]
    objective_value: float
    success: bool
    message: str
    n_iterations: int


def optimize_response(
    anova_results: ANOVAResults,
    factors: List[Factor],
    objective: Literal['maximize', 'minimize', 'target'] = 'maximize',
    target_value: Optional[float] = None,
    target_tolerance: float = 0.1,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    linear_constraints: Optional[List['LinearConstraint']] = None,
    alpha: float = 0.05,
    seed: Optional[int] = None
) -> OptimizationResult:
    """
    Find optimal factor settings for single response.
    
    Parameters
    ----------
    anova_results : ANOVAResults
        Fitted model from ANOVA analysis
    factors : List[Factor]
        Factor definitions
    objective : {'maximize', 'minimize', 'target'}
        Optimization objective
    target_value : float, optional
        Target value (required if objective='target')
    target_tolerance : float, default=0.1
        Acceptable deviation from target (for target objective)
    bounds : Dict[str, Tuple[float, float]], optional
        Factor bounds. If None, uses factor min/max from definitions
    linear_constraints : List[LinearConstraint], optional
        Linear constraints on factors
    alpha : float, default=0.05
        Significance level for intervals
    seed : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    OptimizationResult
        Optimization results with optimal settings and predictions
    
    Raises
    ------
    ValueError
        If objective='target' but target_value not provided
    
    Examples
    --------
    >>> result = optimize_response(
    ...     anova_results=results,
    ...     factors=factors,
    ...     objective='maximize'
    ... )
    >>> print(result.optimal_settings)
    {'Temperature': 195.3, 'Pressure': 98.7}
    """
    if objective == 'target' and target_value is None:
        raise ValueError("target_value must be provided when objective='target'")
    
    # Extract model and factor names
    model = anova_results.fitted_model
    factor_names = [f.name for f in factors]
    
    # Build bounds
    if bounds is None:
        bounds_list = [(f.min_value, f.max_value) for f in factors]
    else:
        bounds_list = [bounds.get(f.name, (f.min_value, f.max_value)) for f in factors]
    
    # Build objective function
    def objective_func(x: np.ndarray) -> float:
        """Objective to minimize (negate for maximize)."""
        pred_df = pd.DataFrame([x], columns=factor_names)
        y_pred = model.predict(pred_df)[0]
        
        if objective == 'maximize':
            return -y_pred  # Negate for minimization
        elif objective == 'minimize':
            return y_pred
        else:  # target
            # Minimize squared deviation from target
            return (y_pred - target_value) ** 2
    
    # Convert linear constraints to scipy format if provided
    scipy_constraints = []
    if linear_constraints is not None:
        scipy_constraints = _convert_linear_constraints(
            linear_constraints, factors
        )
    
    # Starting point: center of design space
    x0 = np.array([(f.min_value + f.max_value) / 2 for f in factors])
    
    # Add random perturbation if seed provided
    if seed is not None:
        rng = np.random.default_rng(seed)
        perturbation = rng.uniform(-0.1, 0.1, size=len(factors))
        ranges = np.array([f.max_value - f.min_value for f in factors])
        x0 = x0 + perturbation * ranges
        x0 = np.clip(x0, [b[0] for b in bounds_list], [b[1] for b in bounds_list])
    
    # Optimize using SLSQP (handles bounds and linear constraints)
    result = minimize(
        objective_func,
        x0=x0,
        method='SLSQP',
        bounds=bounds_list,
        constraints=scipy_constraints,
        options={'maxiter': 500, 'ftol': 1e-9}
    )
    
    # If failed, try differential_evolution (global optimizer)
    if not result.success:
        warnings.warn(
            f"SLSQP failed: {result.message}. Trying global optimizer..."
        )
        from scipy.optimize import differential_evolution
        
        result = differential_evolution(
            objective_func,
            bounds=bounds_list,
            constraints=scipy_constraints if scipy_constraints else None,
            seed=seed,
            maxiter=300,
            atol=1e-9,
            tol=1e-9
        )
    
    # Extract optimal settings
    x_opt = result.x
    optimal_settings = {factor_names[i]: x_opt[i] for i in range(len(factors))}
    
    # Predict at optimum with intervals
    pred, ci, pi = predict_with_intervals(
        model, x_opt, factor_names, alpha
    )
    
    return OptimizationResult(
        optimal_settings=optimal_settings,
        predicted_response=pred,
        confidence_interval=ci,
        prediction_interval=pi,
        objective_value=result.fun,
        success=result.success,
        message=result.message,
        n_iterations=result.nit if hasattr(result, 'nit') else 0
    )


# ============================================================
# SECTION 3: DESIRABILITY FUNCTIONS
# ============================================================


def desirability_maximize(
    y: float,
    low: float,
    high: float,
    weight: float = 1.0
) -> float:
    """
    Desirability function for maximizing response.
    
    Parameters
    ----------
    y : float
        Response value
    low : float
        Minimum acceptable value (d=0)
    high : float
        Target value (d=1)
    weight : float, default=1.0
        Shape parameter (1=linear, >1=emphasize target, <1=more tolerant)
    
    Returns
    -------
    float
        Desirability value in [0, 1]
    
    Notes
    -----
    Formula: d = ((y - low) / (high - low))^weight for low <= y <= high
             d = 0 for y < low
             d = 1 for y > high
    """
    if y < low:
        return 0.0
    elif y > high:
        return 1.0
    else:
        return ((y - low) / (high - low)) ** weight


def desirability_minimize(
    y: float,
    low: float,
    high: float,
    weight: float = 1.0
) -> float:
    """
    Desirability function for minimizing response.
    
    Parameters
    ----------
    y : float
        Response value
    low : float
        Target value (d=1)
    high : float
        Maximum acceptable value (d=0)
    weight : float, default=1.0
        Shape parameter
    
    Returns
    -------
    float
        Desirability value in [0, 1]
    
    Notes
    -----
    Formula: d = ((high - y) / (high - low))^weight for low <= y <= high
             d = 1 for y < low
             d = 0 for y > high
    """
    if y < low:
        return 1.0
    elif y > high:
        return 0.0
    else:
        return ((high - y) / (high - low)) ** weight


def desirability_target(
    y: float,
    low: float,
    target: float,
    high: float,
    weight_low: float = 1.0,
    weight_high: float = 1.0
) -> float:
    """
    Desirability function for target response.
    
    Parameters
    ----------
    y : float
        Response value
    low : float
        Minimum acceptable value (d=0)
    target : float
        Target value (d=1)
    high : float
        Maximum acceptable value (d=0)
    weight_low : float, default=1.0
        Shape parameter for y < target
    weight_high : float, default=1.0
        Shape parameter for y > target
    
    Returns
    -------
    float
        Desirability value in [0, 1]
    
    Notes
    -----
    Two-sided desirability with separate weights below and above target.
    """
    if y < low or y > high:
        return 0.0
    elif y <= target:
        # Rising to target
        return ((y - low) / (target - low)) ** weight_low
    else:
        # Falling from target
        return ((high - y) / (high - target)) ** weight_high


class DesirabilityFunction:
    """
    Multi-response desirability function.
    
    Combines individual response desirabilities into overall desirability
    using geometric mean (Derringer & Suich, 1980).
    
    Parameters
    ----------
    response_names : List[str]
        Names of response variables
    
    Examples
    --------
    >>> df = DesirabilityFunction(['Yield', 'Purity', 'Cost'])
    >>> df.add_response('Yield', 'maximize', low=80, high=95)
    >>> df.add_response('Purity', 'target', low=98, target=99.5, high=100)
    >>> df.add_response('Cost', 'minimize', low=10, high=20)
    >>> overall_d = df.evaluate({'Yield': 90, 'Purity': 99.2, 'Cost': 12})
    """
    
    def __init__(self, response_names: List[str]):
        self.response_names = response_names
        self.response_configs: Dict[str, Dict] = {}
    
    def add_response(
        self,
        response_name: str,
        objective: Literal['maximize', 'minimize', 'target'],
        low: float,
        high: float,
        target: Optional[float] = None,
        weight: float = 1.0,
        weight_low: Optional[float] = None,
        weight_high: Optional[float] = None,
        importance: float = 1.0
    ) -> None:
        """
        Configure desirability for a response.
        
        Parameters
        ----------
        response_name : str
            Name of response variable
        objective : {'maximize', 'minimize', 'target'}
            Desirability type
        low : float
            Lower bound (meaning depends on objective)
        high : float
            Upper bound (meaning depends on objective)
        target : float, optional
            Target value (required for 'target' objective)
        weight : float, default=1.0
            Shape parameter (used for maximize/minimize)
        weight_low : float, optional
            Shape parameter below target (for 'target' objective)
        weight_high : float, optional
            Shape parameter above target (for 'target' objective)
        importance : float, default=1.0
            Relative importance (used as exponent in geometric mean)
        """
        if response_name not in self.response_names:
            raise ValueError(f"Unknown response: {response_name}")
        
        if objective == 'target' and target is None:
            raise ValueError("target value required for 'target' objective")
        
        # Set default weights for target objective
        if objective == 'target':
            if weight_low is None:
                weight_low = weight
            if weight_high is None:
                weight_high = weight
        
        self.response_configs[response_name] = {
            'objective': objective,
            'low': low,
            'high': high,
            'target': target,
            'weight': weight,
            'weight_low': weight_low,
            'weight_high': weight_high,
            'importance': importance
        }
    
    def evaluate_individual(
        self,
        response_name: str,
        value: float
    ) -> float:
        """
        Evaluate individual desirability for one response.
        
        Parameters
        ----------
        response_name : str
            Response name
        value : float
            Response value
        
        Returns
        -------
        float
            Individual desirability in [0, 1]
        """
        if response_name not in self.response_configs:
            raise ValueError(f"Response {response_name} not configured")
        
        config = self.response_configs[response_name]
        objective = config['objective']
        
        if objective == 'maximize':
            return desirability_maximize(
                value, config['low'], config['high'], config['weight']
            )
        elif objective == 'minimize':
            return desirability_minimize(
                value, config['low'], config['high'], config['weight']
            )
        else:  # target
            return desirability_target(
                value, config['low'], config['target'], config['high'],
                config['weight_low'], config['weight_high']
            )
    
    def evaluate(self, responses: Dict[str, float]) -> float:
        """
        Evaluate overall desirability (geometric mean).
        
        Parameters
        ----------
        responses : Dict[str, float]
            Response values
        
        Returns
        -------
        float
            Overall desirability in [0, 1]
        
        Notes
        -----
        **Important**: If ANY individual desirability is 0, the overall
        desirability is 0. This implements strict constraint behavior:
        all responses must be at least minimally acceptable.
        
        If you want softer tradeoffs, adjust the low/high bounds for
        each response to allow some tolerance.
        
        Overall desirability D = (d1^r1 * d2^r2 * ... * dn^rn)^(1/sum(r))
        where di is individual desirability and ri is importance weight.
        """
        individual_desirabilities = []
        importances = []
        
        for response_name in self.response_names:
            if response_name not in self.response_configs:
                raise ValueError(f"Response {response_name} not configured")
            
            if response_name not in responses:
                raise ValueError(f"Response {response_name} not provided")
            
            d_i = self.evaluate_individual(response_name, responses[response_name])
            
            # If any desirability is 0, overall is 0
            if d_i == 0:
                return 0.0
            
            importance = self.response_configs[response_name]['importance']
            individual_desirabilities.append(d_i ** importance)
            importances.append(importance)
        
        # Geometric mean with importance weights
        product = np.prod(individual_desirabilities)
        overall = product ** (1 / np.sum(importances))
        
        return overall


# ============================================================
# SECTION 4: MULTI-RESPONSE OPTIMIZATION
# ============================================================


@dataclass
class DesirabilityResult:
    """
    Result from multi-response desirability optimization.
    
    Attributes
    ----------
    optimal_settings : Dict[str, float]
        Optimal factor values
    predicted_responses : Dict[str, float]
        Predicted values for each response
    individual_desirabilities : Dict[str, float]
        Individual desirability for each response
    overall_desirability : float
        Overall desirability (geometric mean)
    success : bool
        Whether optimization converged
    message : str
        Optimization status message
    n_iterations : int
        Number of iterations
    """
    optimal_settings: Dict[str, float]
    predicted_responses: Dict[str, float]
    individual_desirabilities: Dict[str, float]
    overall_desirability: float
    success: bool
    message: str
    n_iterations: int


def optimize_desirability(
    anova_results_dict: Dict[str, ANOVAResults],
    factors: List[Factor],
    desirability_func: DesirabilityFunction,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    linear_constraints: Optional[List['LinearConstraint']] = None,
    seed: Optional[int] = None
) -> DesirabilityResult:
    """
    Optimize multiple responses using desirability functions.
    
    Parameters
    ----------
    anova_results_dict : Dict[str, ANOVAResults]
        Fitted models for each response (response_name -> ANOVAResults)
    factors : List[Factor]
        Factor definitions
    desirability_func : DesirabilityFunction
        Configured desirability function
    bounds : Dict[str, Tuple[float, float]], optional
        Factor bounds
    linear_constraints : List[LinearConstraint], optional
        Linear constraints on factors
    seed : int, optional
        Random seed
    
    Returns
    -------
    DesirabilityResult
        Optimization results with optimal settings and desirabilities
    
    Examples
    --------
    >>> df = DesirabilityFunction(['Yield', 'Purity'])
    >>> df.add_response('Yield', 'maximize', low=80, high=95)
    >>> df.add_response('Purity', 'target', low=98, target=99.5, high=100)
    >>> 
    >>> models = {
    ...     'Yield': yield_anova_results,
    ...     'Purity': purity_anova_results
    ... }
    >>> 
    >>> result = optimize_desirability(models, factors, df)
    >>> print(result.overall_desirability)
    0.87
    """
    # Validate that all responses have models
    for response_name in desirability_func.response_names:
        if response_name not in anova_results_dict:
            raise ValueError(f"No model provided for response: {response_name}")
    
    factor_names = [f.name for f in factors]
    
    # Build bounds
    if bounds is None:
        bounds_list = [(f.min_value, f.max_value) for f in factors]
    else:
        bounds_list = [bounds.get(f.name, (f.min_value, f.max_value)) for f in factors]
    
    # Build objective function (maximize overall desirability)
    def objective_func(x: np.ndarray) -> float:
        """Objective to minimize (negate desirability)."""
        pred_df = pd.DataFrame([x], columns=factor_names)
        
        # Predict all responses
        responses = {}
        for response_name, anova_results in anova_results_dict.items():
            model = anova_results.fitted_model
            y_pred = model.predict(pred_df)[0]
            responses[response_name] = y_pred
        
        # Evaluate overall desirability
        D = desirability_func.evaluate(responses)
        
        return -D  # Negate for minimization
    
    # Convert linear constraints
    scipy_constraints = []
    if linear_constraints is not None:
        scipy_constraints = _convert_linear_constraints(
            linear_constraints, factors
        )
    
    # Starting point
    x0 = np.array([(f.min_value + f.max_value) / 2 for f in factors])
    
    if seed is not None:
        rng = np.random.default_rng(seed)
        perturbation = rng.uniform(-0.1, 0.1, size=len(factors))
        ranges = np.array([f.max_value - f.min_value for f in factors])
        x0 = x0 + perturbation * ranges
        x0 = np.clip(x0, [b[0] for b in bounds_list], [b[1] for b in bounds_list])
    
    # Optimize
    result = minimize(
        objective_func,
        x0=x0,
        method='SLSQP',
        bounds=bounds_list,
        constraints=scipy_constraints,
        options={'maxiter': 500, 'ftol': 1e-9}
    )
    
    # If failed, try global optimizer
    if not result.success:
        warnings.warn(
            f"SLSQP failed: {result.message}. Trying global optimizer..."
        )
        from scipy.optimize import differential_evolution
        
        result = differential_evolution(
            objective_func,
            bounds=bounds_list,
            constraints=scipy_constraints if scipy_constraints else None,
            seed=seed,
            maxiter=300,
            atol=1e-9,
            tol=1e-9
        )
    
    # Extract optimal settings
    x_opt = result.x
    optimal_settings = {factor_names[i]: x_opt[i] for i in range(len(factors))}
    
    # Predict all responses at optimum
    pred_df = pd.DataFrame([x_opt], columns=factor_names)
    predicted_responses = {}
    individual_desirabilities = {}
    
    for response_name, anova_results in anova_results_dict.items():
        model = anova_results.fitted_model
        y_pred = model.predict(pred_df)[0]
        predicted_responses[response_name] = y_pred
        
        d_i = desirability_func.evaluate_individual(response_name, y_pred)
        individual_desirabilities[response_name] = d_i
    
    overall_D = desirability_func.evaluate(predicted_responses)
    
    return DesirabilityResult(
        optimal_settings=optimal_settings,
        predicted_responses=predicted_responses,
        individual_desirabilities=individual_desirabilities,
        overall_desirability=overall_D,
        success=result.success,
        message=result.message,
        n_iterations=result.nit if hasattr(result, 'nit') else 0
    )


# ============================================================
# SECTION 5: HELPER FUNCTIONS
# ============================================================


@dataclass
class LinearConstraint:
    """
    Linear constraint on factors (reused from optimal_design.py).
    
    Represents constraints like: sum(coefficients[i] * x[i]) <= bound
    """
    coefficients: Dict[str, float]
    bound: float
    constraint_type: Literal['le', 'ge', 'eq'] = 'le'


def _convert_linear_constraints(
    constraints: List[LinearConstraint],
    factors: List[Factor]
) -> List[ScipyLinearConstraint]:
    """
    Convert LinearConstraint objects to scipy format.
    
    Parameters
    ----------
    constraints : List[LinearConstraint]
        Constraints in custom format
    factors : List[Factor]
        Factor definitions
    
    Returns
    -------
    List[ScipyLinearConstraint]
        Constraints in scipy format
    """
    scipy_constraints = []
    factor_names = [f.name for f in factors]
    
    for constraint in constraints:
        # Build coefficient vector
        A = np.zeros(len(factors))
        for fname, coeff in constraint.coefficients.items():
            if fname not in factor_names:
                raise ValueError(f"Unknown factor in constraint: {fname}")
            idx = factor_names.index(fname)
            A[idx] = coeff
        
        # Convert to scipy format
        if constraint.constraint_type == 'le':
            # A @ x <= bound  =>  -inf <= A @ x <= bound
            scipy_constraint = ScipyLinearConstraint(
                A, -np.inf, constraint.bound
            )
        elif constraint.constraint_type == 'ge':
            # A @ x >= bound  =>  bound <= A @ x <= inf
            scipy_constraint = ScipyLinearConstraint(
                A, constraint.bound, np.inf
            )
        else:  # eq
            # A @ x == bound  =>  bound <= A @ x <= bound
            scipy_constraint = ScipyLinearConstraint(
                A, constraint.bound, constraint.bound
            )
        
        scipy_constraints.append(scipy_constraint)
    
    return scipy_constraints