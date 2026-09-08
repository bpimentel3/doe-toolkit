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
from src.core.optimal.constraints import LinearConstraint


# ============================================================
# SECTION 1: PREDICTION WITH UNCERTAINTY
# ============================================================


def predict_with_intervals(
    model: object,
    x_pred: np.ndarray,
    factor_names: List[str],
    factors: Optional[List[Factor]] = None,
    model_is_coded: bool = False,
    alpha: float = 0.05
) -> Tuple[float, Tuple[float, float], Tuple[float, float]]:
    """
    Predict response with confidence and prediction intervals.

    The optimizer always works in actual (natural) factor space so that
    results are interpretable by the user.  The fitted model, however, may
    have been trained on coded values when the stored design is in coded
    space (e.g. CCD, fractional-factorial designs).  Set ``model_is_coded``
    to ``True`` in that case; ``x_pred`` will then be encoded to coded space
    before calling ``model.predict``.

    Raises
    ------
    AttributeError
        If model doesn't support get_prediction (e.g., some mixed-effects models)

    Parameters
    ----------
    model : statsmodels fitted model
        Fitted model from ANOVAResults
    x_pred : np.ndarray
        Factor values at which to predict (actual / natural scale)
    factor_names : List[str]
        Factor names in order
    factors : List[Factor], optional
        Factor definitions, required when ``model_is_coded`` is True.
    model_is_coded : bool, default=False
        When True, encode ``x_pred`` from actual space to coded [-1, 1]
        space before passing to ``model.predict``.
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
    if isinstance(x_pred, pd.DataFrame):
        pred_df = x_pred.copy()
    else:
        pred_df = pd.DataFrame([x_pred], columns=factor_names)
    if model_is_coded and factors is not None:
        from src.core.coding import encode_design
        pred_df = encode_design(pred_df, factors)

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


# ============================================================
# SCALAR: OPTIMIZATION DIMENSIONS
# ============================================================


class _OptimizationDims:
    """Plan the optimizer search space across mixed factor types.

    Continuous and discrete-numeric factors become real dimensions bounded
    by their [min, max] values (actual units).  Categorical factors cannot be
    optimised on their raw label value, so each is represented as an integer
    "level index" dimension over ``[0, k-1]``.  The objective, initial point
    and result extraction all go through :meth:`to_prediction_frame` /
    :meth:`snap_to_settings` so categorical indices are mapped back to their
    declared level labels before the model is asked to predict.
    """

    def __init__(self, factors: List[Factor]):
        self.factors = factors
        self.names = [f.name for f in factors]
        # Bounds are always in actual space; categorical indices in [0, k-1].
        self.bounds: List[Tuple[float, float]] = []
        self.integrality: List[int] = []  # 1 == integer dim (categorical index)
        self._categorical_index: Dict[str, int] = {}
        self._has_categorical = False
        # Non-optimizable design columns the fitted model formula references
        # (e.g. ``Block``). They are injected into the prediction frame at a
        # fixed reference value so the optimizer can ignore their influence.
        self.extra_columns: Dict[str, object] = {}

        for i, f in enumerate(factors):
            if f.is_continuous() or f.is_discrete_numeric():
                self.bounds.append((f.min_value, f.max_value))
                self.integrality.append(0)
            else:  # categorical
                self._has_categorical = True
                self._categorical_index[f.name] = i
                self.bounds.append((0, len(f.levels) - 1))
                self.integrality.append(1)

    @property
    def has_categorical(self) -> bool:
        return self._has_categorical

    def has_numeric(self) -> bool:
        return any(n == 0 for n in self.integrality)

    def x0(self) -> np.ndarray:
        """Starting point: centre of each dimension (mid-index for categoricals)."""
        x0 = np.zeros(len(self.factors))
        for i, f in enumerate(self.factors):
            if f.is_continuous() or f.is_discrete_numeric():
                x0[i] = (f.min_value + f.max_value) / 2
            else:
                x0[i] = (len(f.levels) - 1) / 2
        return x0

    def to_prediction_frame(self, x: np.ndarray) -> pd.DataFrame:
        """Build the prediction frame from an optimizer vector.

        Continuous/discrete dims keep their real values; categorical dims are
        snapped to the nearest level index and replaced with the declared
        level label so the model formula can form its dummies.  Any
        non-optimizable columns required by the model (e.g. ``Block``) are
        injected at their fixed reference value via ``extra_columns``.
        """
        row = {}
        for i, f in enumerate(self.factors):
            if f.is_continuous():
                row[f.name] = float(x[i])
            elif f.is_discrete_numeric():
                row[f.name] = _nearest_level(x[i], f.levels)
            else:
                idx = int(round(float(x[i])))
                idx = int(np.clip(idx, 0, len(f.levels) - 1))
                row[f.name] = f.levels[idx]
        row.update(self.extra_columns)
        return pd.DataFrame([row], columns=self.names + list(self.extra_columns))

    def snap_to_settings(self, x: np.ndarray) -> Dict[str, object]:
        """Map an optimizer vector to human-readable factor settings."""
        settings = {}
        for i, f in enumerate(self.factors):
            if f.is_continuous():
                settings[f.name] = float(x[i])
            elif f.is_discrete_numeric():
                settings[f.name] = _nearest_level(x[i], f.levels)
            else:
                idx = int(np.clip(round(float(x[i])), 0, len(f.levels) - 1))
                settings[f.name] = f.levels[idx]
        return settings


def _nearest_level(value: float, levels) -> float:
    """Snap *value* to the nearest declared level for discrete factors."""
    return min(levels, key=lambda lv: abs(float(lv) - value))


def _pinned_categorical_defaults(factors: List[Factor]) -> Dict[str, object]:
    """Default categorical level choices (first declared level) per factor."""
    return {
        f.name: f.levels[0]
        for f in factors
        if f.is_categorical()
    }


# Non-optimizable design columns that may appear in a fitted model's formula
# but are not controlled by the optimizer. They are pinned to a reference
# level in the prediction frame so their influence is ignored.
_NUISANCE_COLUMNS = ('Block', 'WholePlot', 'VeryHardPlot')


def _nuisance_columns_for_model(fitted_model, factors: List[Factor]) -> Dict[str, object]:
    """Return extra prediction columns required by *fitted_model*.

    A blocked (or split-plot) design appends nuisance terms such as ``Block``
    to its formula. The optimizer's prediction frame must contain those
    columns for patsy to build the design matrix. Each nuisance column is set
    to its reference (first) level so its coefficient contributes zero — the
    block/whole-plot effect is effectively ignored, which is the desired
    behaviour for optimization.

    Parameters
    ----------
    fitted_model : statsmodels fitted model (OLSM/OLSResults)
        The fitted model whose formula references the nuisance columns.
    factors : List[Factor]
        Optimizable factors. Columns that name a factor are skipped.

    Returns
    -------
    Dict[str, object]
        Map of nuisance column name -> reference level (or empty if none).
    """
    factor_names = {f.name for f in factors}
    model = getattr(fitted_model, 'model', fitted_model)
    frame = getattr(getattr(model, 'data', None), 'frame', None)

    try:
        exog_names = list(getattr(model, 'exog_names', ()) or ())
    except Exception:
        exog_names = []

    needed = {}
    for base in (n.split('[')[0].strip() for n in exog_names):
        if base in factor_names or base in ('Intercept', 'const'):
            continue
        if base in _NUISANCE_COLUMNS:
            # Reference level = first distinct value in the fit frame.
            if frame is not None and base in frame.columns:
                levels = sorted(set(frame[base].dropna().tolist()))
                if levels:
                    needed[base] = levels[0]
            else:
                needed[base] = None
    return needed


def optimize_response(
    anova_results: ANOVAResults,
    factors: List[Factor],
    objective: Literal['maximize', 'minimize', 'target'] = 'maximize',
    target_value: Optional[float] = None,
    target_tolerance: float = 0.1,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    linear_constraints: Optional[List['LinearConstraint']] = None,
    alpha: float = 0.05,
    seed: Optional[int] = None,
    model_is_coded: bool = False,
    pinned_levels: Optional[Dict[str, object]] = None,
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
    pinned_levels : Dict[str, object], optional
        Categorical levels to hold fixed (name -> declared level).  Any
        categorical factor NOT listed here is treated as a free dimension and
        the optimizer selects its best level.
    
    Returns
    -------
    OptimizationResult
        Optimization results with optimal settings and predictions.  For
        free categorical factors, ``optimal_settings`` holds the chosen level
        label; for numeric factors a float value.
    
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
    
    from src.core.coding import encode_design
    model = anova_results.fitted_model

    dims = _OptimizationDims(factors)
    dims.extra_columns = _nuisance_columns_for_model(model, factors)
    factor_names = dims.names
    pinned_levels = dict(pinned_levels or {})

    # Apply user-supplied numeric bounds as hard bounds (actual space).
    bounds_list = list(dims.bounds)
    if bounds:
        idx_by_name = {f.name: i for i, f in enumerate(factors)}
        for name, (lo, hi) in bounds.items():
            i = idx_by_name.get(name)
            if i is None:
                raise ValueError(f"Unknown factor in bounds: {name}")
            if dims.integrality[i]:
                raise ValueError(
                    f"Bounds cannot be set for categorical factor '{name}'."
                )
            bounds_list[i] = (lo, hi)

    # Apply user-pinned categorical levels as hard bounds [i, i] (fixed dim).
    for name, level in pinned_levels.items():
        idx = dims._categorical_index.get(name)
        if idx is None:
            raise ValueError(
                f"Pinned level '{level}' for non-categorical factor '{name}'. "
                "Only categorical factors can be pinned to a level."
            )
        fac = factors[idx]
        if level not in fac.levels:
            raise ValueError(
                f"Level '{level}' is not a valid level for factor '{name}'. "
                f"Valid levels: {fac.levels}"
            )
        li = fac.levels.index(level)
        bounds_list[idx] = (li, li)

    # Build objective function (operates on the full optimizer vector).
    def objective_func(x: np.ndarray) -> float:
        """Objective to minimize (negate for maximize)."""
        pred_df = dims.to_prediction_frame(x)
        if model_is_coded:
            pred_df = encode_design(pred_df, factors)
        y_pred = model.predict(pred_df)[0]

        if objective == 'maximize':
            return -y_pred  # Negate for minimization
        elif objective == 'minimize':
            return y_pred
        else:  # target
            return (y_pred - target_value) ** 2

    # Convert linear constraints to scipy format if provided
    scipy_constraints = []
    if linear_constraints is not None:
        scipy_constraints = _convert_linear_constraints(
            linear_constraints, factors
        )

    # Starting point: center of each dimension (mid-index for categoricals).
    x0 = dims.x0()

    # Add random perturbation if seed provided.
    if seed is not None:
        rng = np.random.default_rng(seed)
        perturbation = rng.uniform(-0.1, 0.1, size=len(factors))
        ranges = np.array([b[1] - b[0] for b in dims.bounds])
        x0 = x0 + perturbation * ranges
        x0 = np.clip(x0, [b[0] for b in bounds_list], [b[1] for b in bounds_list])
        if dims.has_categorical:
            # Centre categorical dims on integer indices before snapping.
            for i, is_int in enumerate(dims.integrality):
                if is_int:
                    x0[i] = round(x0[i])

    # Choose the solver: SLSQP for pure-numeric designs (unchanged behaviour);
    # differential_evolution with integrality when categoricals are present.
    if not dims.has_categorical:
        result = minimize(
            objective_func,
            x0=x0,
            method='SLSQP',
            bounds=bounds_list,
            constraints=scipy_constraints,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
    else:
        result = None
        if scipy_constraints:
            warnings.warn(
                "Linear constraints are ignored for designs containing "
                "categorical factors."
            )
        try:
            from scipy.optimize import differential_evolution
            result = differential_evolution(
                objective_func,
                bounds=bounds_list,
                integrality=dims.integrality,
                maxiter=500,
                popsize=20,
                seed=seed,
                polish=False,
            )
        except Exception as exc:  # pragma: no cover - defensive
            warnings.warn(f"differential_evolution failed: {exc}")

        if result is None or not result.success:
            # Fallback: expand discrete leaves and compare directly.
            warnings.warn(
                "Global categorical optimizer did not converge; enumerating "
                "categorical level combinations instead."
            )
            result = _enumerate_categorical_best(objective_func, dims)

    # Extract optimal settings (numeric floats; categorical level labels).
    x_opt = result.x
    optimal_settings = dims.snap_to_settings(x_opt)

    # Predict at optimum with intervals.
    pred_df = dims.to_prediction_frame(x_opt)
    pred, ci, pi = predict_with_intervals(
        model, pred_df, factor_names,
        factors=factors, model_is_coded=model_is_coded, alpha=alpha
    )

    return OptimizationResult(
        optimal_settings=optimal_settings,
        predicted_response=pred,
        confidence_interval=ci,
        prediction_interval=pi,
        objective_value=result.fun,
        success=result.success,
        message=getattr(result, 'message', ''),
        n_iterations=result.nit if hasattr(result, 'nit') else 0
    )


def _enumerate_categorical_best(
    objective_func: Callable[[np.ndarray], float],
    dims: _OptimizationDims,
) -> object:
    """Brute-force the categorical dimensions.

    Used as a last-resort fallback when differential_evolution is unavailable
    or does not converge.  Only valid when every dimension is categorical
    (no free numeric dims); otherwise we return a nominal failure result so
    the caller surfaces a clear message rather than a wrong answer.
    """
    from types import SimpleNamespace

    if dims.has_numeric():
        return SimpleNamespace(
            x=np.zeros(len(dims.factors)),
            fun=float('inf'),
            success=False,
            message=(
                "Optimization failed for categorical + numeric design: "
                "global solver unavailable."
            ),
            nit=0,
        )

    cat_indices = [i for i, n in enumerate(dims.integrality) if n == 1]
    if not cat_indices:
        return SimpleNamespace(
            x=dims.x0(), fun=float('inf'), success=False,
            message="No dimensions to optimize.", nit=0,
        )

    axis_count = [len(dims.factors[i].levels) for i in cat_indices]

    best = None
    best_fun = float('inf')
    total = 1
    for c in axis_count:
        total *= c

    for flat in range(total):
        x = dims.x0()
        rem = flat
        for pos, counts in enumerate(axis_count):
            x[cat_indices[pos]] = rem % counts
            rem //= counts
        fun = objective_func(x)
        if fun < best_fun:
            best_fun = fun
            best = x.copy()

    return SimpleNamespace(
        x=best, fun=best_fun, success=True,
        message=f"Enumerated {total} categorical combination(s).",
        nit=total,
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
    seed: Optional[int] = None,
    model_is_coded: bool = False,
    pinned_levels: Optional[Dict[str, object]] = None,
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
    pinned_levels : Dict[str, object], optional
        Categorical levels to hold fixed (name -> declared level).  Any
        categorical factor NOT listed here is a free dimension and the
        optimizer selects its best level.
    
    Returns
    -------
    DesirabilityResult
        Optimization results with optimal settings and desirabilities.  For
        free categorical factors, ``optimal_settings`` holds the chosen level
        label; for numeric factors a float.
    
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
    
    from src.core.coding import encode_design
    dims = _OptimizationDims(factors)
    # Merge nuisance columns across every fitted response model. All models
    # share the same design, so reference levels agree; use the first one.
    merged: Dict[str, object] = {}
    for anova_results in anova_results_dict.values():
        merged.update(_nuisance_columns_for_model(anova_results.fitted_model, factors))
    dims.extra_columns = merged
    factor_names = dims.names
    pinned_levels = dict(pinned_levels or {})

    # Apply user-supplied numeric bounds as hard bounds (actual space).
    bounds_list = list(dims.bounds)
    if bounds:
        idx_by_name = {f.name: i for i, f in enumerate(factors)}
        for name, (lo, hi) in bounds.items():
            i = idx_by_name.get(name)
            if i is None:
                raise ValueError(f"Unknown factor in bounds: {name}")
            if dims.integrality[i]:
                raise ValueError(
                    f"Bounds cannot be set for categorical factor '{name}'."
                )
            bounds_list[i] = (lo, hi)

    # Apply user-pinned categorical levels as fixed dimensions.
    for name, level in pinned_levels.items():
        idx = dims._categorical_index.get(name)
        if idx is None:
            raise ValueError(
                f"Pinned level '{level}' for non-categorical factor '{name}'. "
                "Only categorical factors can be pinned to a level."
            )
        fac = factors[idx]
        if level not in fac.levels:
            raise ValueError(
                f"Level '{level}' is not a valid level for factor '{name}'. "
                f"Valid levels: {fac.levels}"
            )
        li = fac.levels.index(level)
        bounds_list[idx] = (li, li)

    # Build objective function (maximize overall desirability)
    def objective_func(x: np.ndarray) -> float:
        """Objective to minimize (negate desirability)."""
        pred_df = dims.to_prediction_frame(x)
        if model_is_coded:
            pred_df = encode_design(pred_df, factors)

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
    x0 = dims.x0()
    
    if seed is not None:
        rng = np.random.default_rng(seed)
        perturbation = rng.uniform(-0.1, 0.1, size=len(factors))
        ranges = np.array([b[1] - b[0] for b in dims.bounds])
        x0 = x0 + perturbation * ranges
        x0 = np.clip(x0, [b[0] for b in bounds_list], [b[1] for b in bounds_list])
        if dims.has_categorical:
            for i, is_int in enumerate(dims.integrality):
                if is_int:
                    x0[i] = round(x0[i])
    
    if not dims.has_categorical:
        result = minimize(
            objective_func,
            x0=x0,
            method='SLSQP',
            bounds=bounds_list,
            constraints=scipy_constraints,
            options={'maxiter': 500, 'ftol': 1e-9}
        )
    else:
        result = None
        if scipy_constraints:
            warnings.warn(
                "Linear constraints are ignored for designs containing "
                "categorical factors."
            )
        try:
            from scipy.optimize import differential_evolution
            result = differential_evolution(
                objective_func,
                bounds=bounds_list,
                integrality=dims.integrality,
                maxiter=500,
                popsize=20,
                seed=seed,
                polish=False,
            )
        except Exception as exc:
            warnings.warn(f"differential_evolution failed: {exc}")

        if result is None or not result.success:
            warnings.warn(
                "Global categorical optimizer did not converge; enumerating "
                "categorical level combinations instead."
            )
            result = _enumerate_categorical_best(objective_func, dims)
    
    # Extract optimal settings (numeric floats; categorical level labels).
    x_opt = result.x
    optimal_settings = dims.snap_to_settings(x_opt)
    
    # Predict all responses at optimum.
    pred_df = dims.to_prediction_frame(x_opt)
    if model_is_coded:
        pred_df = encode_design(pred_df, factors)
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
        message=getattr(result, 'message', ''),
        n_iterations=result.nit if hasattr(result, 'nit') else 0
    )


# ============================================================
# SECTION 5: HELPER FUNCTIONS
# ============================================================


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