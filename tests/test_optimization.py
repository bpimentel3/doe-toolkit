"""
Tests for response optimization module.

Test strategy:
1. Known polynomial optima (quadratic functions with known solutions)
2. Single-response optimization (maximize, minimize, target)
3. Multi-response desirability optimization
4. Constraint handling (bounds, linear constraints)
5. Prediction intervals
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock

from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.optimization import (
    optimize_response,
    optimize_desirability,
    DesirabilityFunction,
    LinearConstraint,
    desirability_maximize,
    desirability_minimize,
    desirability_target,
    predict_with_intervals
)
from src.core.analysis import ANOVAAnalysis, ANOVAResults


# ============================================================
# TEST FIXTURES
# ============================================================


@pytest.fixture
def simple_factors():
    """Two continuous factors."""
    return [
        Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("X2", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
    ]


@pytest.fixture
def three_factors():
    """Three continuous factors."""
    return [
        Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("X2", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1]),
        Factor("X3", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[-1, 1])
    ]


def generate_quadratic_data(
    factors: list,
    coefficients: dict,
    noise_std: float = 0.0,
    seed: int = 42
) -> tuple:
    """
    Generate synthetic data from known quadratic model.
    
    Parameters
    ----------
    factors : list
        Factor definitions
    coefficients : dict
        Model coefficients (e.g., {'intercept': 10, 'X1': 2, 'X1^2': -1, ...})
    noise_std : float
        Standard deviation of random noise
    seed : int
        Random seed
    
    Returns
    -------
    design : pd.DataFrame
        Design matrix
    response : np.ndarray
        Response values
    true_optimum : tuple
        True optimal settings and response (x_opt, y_opt)
    """
    from itertools import product
    
    rng = np.random.default_rng(seed)
    
    # Generate full factorial with center points
    n_factors = len(factors)
    factorial_points = list(product([-1, 1], repeat=n_factors))
    
    # Add center points
    center_points = [[0] * n_factors] * 3
    
    # Add axial points
    axial_points = []
    for i in range(n_factors):
        point_pos = [0] * n_factors
        point_pos[i] = 1
        axial_points.append(point_pos)
        
        point_neg = [0] * n_factors
        point_neg[i] = -1
        axial_points.append(point_neg)
    
    all_points = factorial_points + center_points + axial_points
    
    # Create design dataframe
    factor_names = [f.name for f in factors]
    design = pd.DataFrame(all_points, columns=factor_names)
    design.insert(0, 'StdOrder', range(1, len(design) + 1))
    design.insert(1, 'RunOrder', range(1, len(design) + 1))
    
    # Generate response from quadratic model
    response = np.zeros(len(design))
    
    for i, row in design[factor_names].iterrows():
        y = coefficients.get('intercept', 0)
        
        # Main effects
        for fname in factor_names:
            y += coefficients.get(fname, 0) * row[fname]
        
        # Interactions
        for j, fname1 in enumerate(factor_names):
            for fname2 in factor_names[j+1:]:
                key = f"{fname1}*{fname2}"
                y += coefficients.get(key, 0) * row[fname1] * row[fname2]
        
        # Quadratic terms
        for fname in factor_names:
            key = f"{fname}^2"
            y += coefficients.get(key, 0) * row[fname]**2
        
        response[i] = y
    
    # Add noise
    if noise_std > 0:
        response += rng.normal(0, noise_std, size=len(response))
    
    # Calculate true optimum analytically for simple cases
    true_optimum = _calculate_analytical_optimum(factors, coefficients)
    
    return design, response, true_optimum


def _calculate_analytical_optimum(factors, coefficients):
    """
    Calculate analytical optimum for quadratic model.
    
    For 1D: y = a + bx + cx^2, optimum at x = -b/(2c)
    For 2D with no interaction: similar for each factor
    """
    factor_names = [f.name for f in factors]
    
    if len(factors) == 1:
        # 1D quadratic
        fname = factor_names[0]
        b = coefficients.get(fname, 0)
        c = coefficients.get(f"{fname}^2", 0)
        
        if c == 0:
            return None  # Linear, no optimum
        
        x_opt = -b / (2 * c)
        
        # Clip to bounds
        x_opt = np.clip(x_opt, factors[0].min_value, factors[0].max_value)
        
        # Calculate y_opt
        y_opt = coefficients.get('intercept', 0) + b * x_opt + c * x_opt**2
        
        return (np.array([x_opt]), y_opt)
    
    elif len(factors) == 2 and f"{factor_names[0]}*{factor_names[1]}" not in coefficients:
        # 2D separable quadratic (no interaction)
        x_opts = []
        
        for fname in factor_names:
            b = coefficients.get(fname, 0)
            c = coefficients.get(f"{fname}^2", 0)
            
            if c == 0:
                return None
            
            x_opt = -b / (2 * c)
            x_opts.append(x_opt)
        
        # Clip to bounds
        x_opts = np.array(x_opts)
        for i, f in enumerate(factors):
            x_opts[i] = np.clip(x_opts[i], f.min_value, f.max_value)
        
        # Calculate y_opt
        y_opt = coefficients.get('intercept', 0)
        for i, fname in enumerate(factor_names):
            y_opt += coefficients.get(fname, 0) * x_opts[i]
            y_opt += coefficients.get(f"{fname}^2", 0) * x_opts[i]**2
        
        return (x_opts, y_opt)
    
    else:
        # Complex case - return None (use numerical optimization to verify)
        return None


# ============================================================
# SECTION 1: DESIRABILITY FUNCTION TESTS
# ============================================================


class TestDesirabilityFunctions:
    """Test individual desirability functions."""
    
    def test_desirability_maximize_basic(self):
        """Test maximize desirability at key points."""
        # Below range
        assert desirability_maximize(5, 10, 20) == 0.0
        
        # At lower bound
        assert desirability_maximize(10, 10, 20) == 0.0
        
        # Midpoint (linear weight)
        assert desirability_maximize(15, 10, 20) == pytest.approx(0.5)
        
        # At upper bound
        assert desirability_maximize(20, 10, 20) == 1.0
        
        # Above range
        assert desirability_maximize(25, 10, 20) == 1.0
    
    def test_desirability_maximize_weight(self):
        """Test maximize with different weights."""
        # Weight > 1: more emphasis on target
        d_heavy = desirability_maximize(15, 10, 20, weight=2)
        assert d_heavy == pytest.approx(0.25)  # (0.5)^2
        
        # Weight < 1: more tolerant
        d_light = desirability_maximize(15, 10, 20, weight=0.5)
        assert d_light == pytest.approx(0.707, abs=0.01)  # sqrt(0.5)
    
    def test_desirability_minimize_basic(self):
        """Test minimize desirability."""
        assert desirability_minimize(5, 10, 20) == 1.0   # Below range
        assert desirability_minimize(10, 10, 20) == 1.0  # At target
        assert desirability_minimize(15, 10, 20) == pytest.approx(0.5)
        assert desirability_minimize(20, 10, 20) == 0.0  # At max
        assert desirability_minimize(25, 10, 20) == 0.0  # Above range
    
    def test_desirability_target_basic(self):
        """Test target desirability."""
        # Below acceptable
        assert desirability_target(5, 10, 15, 20) == 0.0
        
        # At lower bound
        assert desirability_target(10, 10, 15, 20) == 0.0
        
        # Below target
        assert desirability_target(12.5, 10, 15, 20) == pytest.approx(0.5)
        
        # At target
        assert desirability_target(15, 10, 15, 20) == 1.0
        
        # Above target
        assert desirability_target(17.5, 10, 15, 20) == pytest.approx(0.5)
        
        # At upper bound
        assert desirability_target(20, 10, 15, 20) == 0.0
        
        # Above acceptable
        assert desirability_target(25, 10, 15, 20) == 0.0
    
    def test_desirability_target_asymmetric_weights(self):
        """Test target with different weights below/above."""
        # Symmetric
        d_sym = desirability_target(17.5, 10, 15, 20, weight_low=1, weight_high=1)
        assert d_sym == pytest.approx(0.5)
        
        # More tolerant above target
        d_asym = desirability_target(17.5, 10, 15, 20, weight_low=1, weight_high=0.5)
        assert d_asym > d_sym


class TestDesirabilityFunctionClass:
    """Test DesirabilityFunction class."""
    
    def test_single_response_maximize(self):
        """Test desirability with single maximize response."""
        df = DesirabilityFunction(['Yield'])
        df.add_response('Yield', 'maximize', low=80, high=95)
        
        # Test individual
        d = df.evaluate_individual('Yield', 90)
        expected = (90 - 80) / (95 - 80)
        assert d == pytest.approx(expected)
        
        # Test overall (same as individual for single response)
        D = df.evaluate({'Yield': 90})
        assert D == pytest.approx(expected)
    
    def test_two_responses_geometric_mean(self):
        """Test geometric mean of two responses."""
        df = DesirabilityFunction(['Yield', 'Purity'])
        df.add_response('Yield', 'maximize', low=80, high=95)
        df.add_response('Purity', 'minimize', low=1, high=5)
        
        # Both at midpoint
        d_yield = (87.5 - 80) / (95 - 80)  # 0.5
        d_purity = (5 - 3) / (5 - 1)       # 0.5
        
        D = df.evaluate({'Yield': 87.5, 'Purity': 3})
        
        expected = np.sqrt(d_yield * d_purity)  # Geometric mean
        assert D == pytest.approx(expected)
    
    def test_importance_weights(self):
        """Test importance weighting in geometric mean."""
        df = DesirabilityFunction(['Yield', 'Purity'])
        df.add_response('Yield', 'maximize', low=80, high=95, importance=2)
        df.add_response('Purity', 'minimize', low=1, high=5, importance=1)
        
        d_yield = (90 - 80) / (95 - 80)  # 2/3
        d_purity = (5 - 2) / (5 - 1)     # 3/4
        
        D = df.evaluate({'Yield': 90, 'Purity': 2})
        
        # D = (d_yield^2 * d_purity^1)^(1/3)
        expected = (d_yield**2 * d_purity) ** (1/3)
        assert D == pytest.approx(expected)
    
    def test_zero_desirability_makes_overall_zero(self):
        """If any individual is 0, overall is 0."""
        df = DesirabilityFunction(['Yield', 'Purity'])
        df.add_response('Yield', 'maximize', low=80, high=95)
        df.add_response('Purity', 'minimize', low=1, high=5)
        
        # Yield out of range
        D = df.evaluate({'Yield': 70, 'Purity': 3})
        assert D == 0.0
    
    def test_missing_response_raises(self):
        """Test error when response not provided."""
        df = DesirabilityFunction(['Yield', 'Purity'])
        df.add_response('Yield', 'maximize', low=80, high=95)
        df.add_response('Purity', 'minimize', low=1, high=5)
        
        with pytest.raises(ValueError, match="not provided"):
            df.evaluate({'Yield': 90})  # Missing Purity


# ============================================================
# SECTION 2: SINGLE-RESPONSE OPTIMIZATION TESTS
# ============================================================


class TestSingleResponseOptimization:
    """Test single-response optimization."""
    
    def test_maximize_simple_quadratic(self, simple_factors):
        """Test maximizing y = 10 - X1^2 - X2^2 (optimum at origin)."""
        coefficients = {
            'intercept': 10,
            'X1^2': -1,
            'X2^2': -1
        }
        
        design, response, true_opt = generate_quadratic_data(
            simple_factors, coefficients, noise_std=0.1, seed=42
        )
        
        # Fit model
        analysis = ANOVAAnalysis(design, response, simple_factors)
        results = analysis.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        # Optimize
        opt_result = optimize_response(
            results, simple_factors, objective='maximize', seed=42
        )
        
        assert opt_result.success
        
        # Should find optimum near (0, 0)
        assert opt_result.optimal_settings['X1'] == pytest.approx(0, abs=0.2)
        assert opt_result.optimal_settings['X2'] == pytest.approx(0, abs=0.2)
        
        # Response should be near 10
        assert opt_result.predicted_response == pytest.approx(10, abs=0.5)
    
    def test_minimize_simple_quadratic(self, simple_factors):
        """Test minimizing y = 5 + X1^2 + X2^2 (optimum at origin)."""
        coefficients = {
            'intercept': 5,
            'X1^2': 1,
            'X2^2': 1
        }
        
        design, response, _ = generate_quadratic_data(
            simple_factors, coefficients, noise_std=0.1, seed=42
        )
        
        analysis = ANOVAAnalysis(design, response, simple_factors)
        results = analysis.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        opt_result = optimize_response(
            results, simple_factors, objective='minimize', seed=42
        )
        
        assert opt_result.success
        assert opt_result.optimal_settings['X1'] == pytest.approx(0, abs=0.2)
        assert opt_result.optimal_settings['X2'] == pytest.approx(0, abs=0.2)
        assert opt_result.predicted_response == pytest.approx(5, abs=0.5)
    
    def test_maximize_with_linear_term(self, simple_factors):
        """Test y = 10 + 3*X1 - X1^2 - X2^2 (optimum at X1=1.5, X2=0)."""
        coefficients = {
            'intercept': 10,
            'X1': 3,
            'X1^2': -1,
            'X2^2': -1
        }
        
        design, response, _ = generate_quadratic_data(
            simple_factors, coefficients, noise_std=0.1, seed=42
        )
        
        analysis = ANOVAAnalysis(design, response, simple_factors)
        results = analysis.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        opt_result = optimize_response(
            results, simple_factors, objective='maximize', seed=42
        )
        
        assert opt_result.success
        
        # Optimum at X1 = -3/(2*-1) = 1.5, but clipped to bounds [-1, 1]
        assert opt_result.optimal_settings['X1'] == pytest.approx(1.0, abs=0.2)
        assert opt_result.optimal_settings['X2'] == pytest.approx(0, abs=0.2)
    
    def test_target_objective(self, simple_factors):
        """Test target objective."""
        coefficients = {
            'intercept': 20,
            'X1': 5,
            'X2': 3,
            'X1^2': -2,
            'X2^2': -2
        }
        
        design, response, _ = generate_quadratic_data(
            simple_factors, coefficients, noise_std=0.1, seed=42
        )
        
        analysis = ANOVAAnalysis(design, response, simple_factors)
        results = analysis.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        # Target value of 22
        opt_result = optimize_response(
            results, simple_factors,
            objective='target',
            target_value=22,
            seed=42
        )
        
        assert opt_result.success
        assert opt_result.predicted_response == pytest.approx(22, abs=1.0)
    
    def test_confidence_and_prediction_intervals(self, simple_factors):
        """Test that CI and PI are computed."""
        coefficients = {
            'intercept': 10,
            'X1^2': -1,
            'X2^2': -1
        }
        
        design, response, _ = generate_quadratic_data(
            simple_factors, coefficients, noise_std=0.5, seed=42
        )
        
        analysis = ANOVAAnalysis(design, response, simple_factors)
        results = analysis.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        opt_result = optimize_response(
            results, simple_factors, objective='maximize', seed=42
        )
        
        # Check intervals exist
        ci_lower, ci_upper = opt_result.confidence_interval
        pi_lower, pi_upper = opt_result.prediction_interval
        
        assert ci_lower < opt_result.predicted_response < ci_upper
        assert pi_lower < opt_result.predicted_response < pi_upper
        
        # PI should be wider than CI
        assert (pi_upper - pi_lower) > (ci_upper - ci_lower)
    
    def test_bounds_constraint(self, simple_factors):
        """Test custom bounds."""
        coefficients = {
            'intercept': 10,
            'X1': 5,
            'X1^2': -1,
            'X2^2': -1
        }
        
        design, response, _ = generate_quadratic_data(
            simple_factors, coefficients, noise_std=0.1, seed=42
        )
        
        analysis = ANOVAAnalysis(design, response, simple_factors)
        results = analysis.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        # Restrict X1 to [0, 1]
        bounds = {'X1': (0, 1), 'X2': (-1, 1)}
        
        opt_result = optimize_response(
            results, simple_factors,
            objective='maximize',
            bounds=bounds,
            seed=42
        )
        
        assert opt_result.success
        assert 0 <= opt_result.optimal_settings['X1'] <= 1
        assert -1 <= opt_result.optimal_settings['X2'] <= 1
    
    def test_linear_constraint_sum(self, simple_factors):
        """Test linear constraint: X1 + X2 <= 0.5."""
        coefficients = {
            'intercept': 10,
            'X1': 3,
            'X2': 2,
            'X1^2': -1,
            'X2^2': -1
        }
        
        design, response, _ = generate_quadratic_data(
            simple_factors, coefficients, noise_std=0.1, seed=42
        )
        
        analysis = ANOVAAnalysis(design, response, simple_factors)
        results = analysis.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        # Add constraint
        constraint = LinearConstraint(
            coefficients={'X1': 1, 'X2': 1},
            bound=0.5,
            constraint_type='le'
        )
        
        opt_result = optimize_response(
            results, simple_factors,
            objective='maximize',
            linear_constraints=[constraint],
            seed=42
        )
        
        assert opt_result.success
        
        # Verify constraint satisfied
        x1 = opt_result.optimal_settings['X1']
        x2 = opt_result.optimal_settings['X2']
        assert x1 + x2 <= 0.5 + 1e-6  # Small tolerance


# ============================================================
# SECTION 3: MULTI-RESPONSE OPTIMIZATION TESTS
# ============================================================


class TestMultiResponseOptimization:
    """Test multi-response desirability optimization."""
    
    def test_two_responses_maximize_both(self, simple_factors):
        """Test maximizing two responses simultaneously."""
        # Response 1: y1 = 10 - X1^2 - X2^2
        coeff1 = {
            'intercept': 10,
            'X1^2': -1,
            'X2^2': -1
        }
        
        # Response 2: y2 = 8 - 0.5*X1^2 - 0.5*X2^2
        coeff2 = {
            'intercept': 8,
            'X1^2': -0.5,
            'X2^2': -0.5
        }
        
        design1, response1, _ = generate_quadratic_data(
            simple_factors, coeff1, noise_std=0.1, seed=42
        )
        design2, response2, _ = generate_quadratic_data(
            simple_factors, coeff2, noise_std=0.1, seed=43
        )
        
        # Fit models
        analysis1 = ANOVAAnalysis(design1, response1, simple_factors)
        results1 = analysis1.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        analysis2 = ANOVAAnalysis(design2, response2, simple_factors)
        results2 = analysis2.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        # Configure desirability
        df = DesirabilityFunction(['Y1', 'Y2'])
        df.add_response('Y1', 'maximize', low=5, high=10)
        df.add_response('Y2', 'maximize', low=4, high=8)
        
        models = {'Y1': results1, 'Y2': results2}
        
        # Optimize
        opt_result = optimize_desirability(
            models, simple_factors, df, seed=42
        )
        
        assert opt_result.success
        
        # Both responses optimized at origin
        assert opt_result.optimal_settings['X1'] == pytest.approx(0, abs=0.2)
        assert opt_result.optimal_settings['X2'] == pytest.approx(0, abs=0.2)
        
        # Check individual desirabilities computed
        assert 'Y1' in opt_result.individual_desirabilities
        assert 'Y2' in opt_result.individual_desirabilities
        
        # Overall desirability should be high
        assert opt_result.overall_desirability > 0.8
    
    def test_conflicting_responses(self, simple_factors):
        """Test with conflicting responses (trade-off)."""
        # Response 1: maximized at X1=1 (y1 = 10 + 5*X1 - X1^2)
        coeff1 = {
            'intercept': 10,
            'X1': 5,
            'X1^2': -2,
            'X2^2': -1
        }
        
        # Response 2: maximized at X1=-1 (y2 = 8 - 4*X1 - X1^2)
        coeff2 = {
            'intercept': 8,
            'X1': -4,
            'X1^2': -2,
            'X2^2': -1
        }
        
        design1, response1, _ = generate_quadratic_data(
            simple_factors, coeff1, noise_std=0.1, seed=42
        )
        design2, response2, _ = generate_quadratic_data(
            simple_factors, coeff2, noise_std=0.1, seed=43
        )
        
        analysis1 = ANOVAAnalysis(design1, response1, simple_factors)
        results1 = analysis1.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        analysis2 = ANOVAAnalysis(design2, response2, simple_factors)
        results2 = analysis2.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        # Configure desirability
        df = DesirabilityFunction(['Y1', 'Y2'])
        df.add_response('Y1', 'maximize', low=8, high=15)
        df.add_response('Y2', 'maximize', low=5, high=12)
        
        models = {'Y1': results1, 'Y2': results2}
        
        opt_result = optimize_desirability(
            models, simple_factors, df, seed=42
        )
        
        assert opt_result.success
        
        # Should find compromise between X1=1 and X1=-1
        x1_opt = opt_result.optimal_settings['X1']
        assert -1 <= x1_opt <= 1
        
        # Should not be at extreme (would favor one response)
        assert abs(x1_opt) < 0.9  # Not at boundary
    
    def test_importance_weights_affect_optimum(self, simple_factors):
        """Test that importance weights shift optimum."""
        coeff1 = {
            'intercept': 10,
            'X1': 3,
            'X1^2': -1,
            'X2^2': -1
        }
        
        coeff2 = {
            'intercept': 8,
            'X1': -2,
            'X1^2': -1,
            'X2^2': -1
        }
        
        design1, response1, _ = generate_quadratic_data(
            simple_factors, coeff1, noise_std=0.1, seed=42
        )
        design2, response2, _ = generate_quadratic_data(
            simple_factors, coeff2, noise_std=0.1, seed=43
        )
        
        analysis1 = ANOVAAnalysis(design1, response1, simple_factors)
        results1 = analysis1.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        analysis2 = ANOVAAnalysis(design2, response2, simple_factors)
        results2 = analysis2.fit(['X1', 'X2', 'I(X1**2)', 'I(X2**2)'])
        
        models = {'Y1': results1, 'Y2': results2}
        
        # Equal importance
        df_equal = DesirabilityFunction(['Y1', 'Y2'])
        df_equal.add_response('Y1', 'maximize', low=8, high=15, importance=1)
        df_equal.add_response('Y2', 'maximize', low=5, high=10, importance=1)
        
        opt_equal = optimize_desirability(
            models, simple_factors, df_equal, seed=42
        )
        
        # Y1 more important
        df_weighted = DesirabilityFunction(['Y1', 'Y2'])
        df_weighted.add_response('Y1', 'maximize', low=8, high=15, importance=3)
        df_weighted.add_response('Y2', 'maximize', low=5, high=10, importance=1)
        
        opt_weighted = optimize_desirability(
            models, simple_factors, df_weighted, seed=42
        )
        
        # With Y1 more important, optimum should shift toward Y1's optimum
        # Y1 optimized at positive X1, Y2 at negative X1
        assert opt_weighted.optimal_settings['X1'] > opt_equal.optimal_settings['X1']


# ============================================================
# SECTION 4: EDGE CASES AND VALIDATION
# ============================================================


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_target_without_target_value_raises(self, simple_factors):
        """Test that target objective requires target_value."""
        # Create mock results
        mock_model = Mock()
        mock_results = Mock(spec=ANOVAResults)
        mock_results.fitted_model = mock_model
        
        with pytest.raises(ValueError, match="target_value must be provided"):
            optimize_response(
                mock_results, simple_factors, objective='target'
            )
    
    def test_flat_response_surface(self, simple_factors):
        """Test with linear model (no curvature)."""
        # y = 10 + 2*X1 + 3*X2 (no optimum in interior)
        coefficients = {
            'intercept': 10,
            'X1': 2,
            'X2': 3
        }
        
        design, response, _ = generate_quadratic_data(
            simple_factors, coefficients, noise_std=0.1, seed=42
        )
        
        analysis = ANOVAAnalysis(design, response, simple_factors)
        results = analysis.fit(['X1', 'X2'])  # Linear model
        
        # Maximize - should go to corner (1, 1)
        opt_result = optimize_response(
            results, simple_factors, objective='maximize', seed=42
        )
        
        assert opt_result.success
        assert opt_result.optimal_settings['X1'] == pytest.approx(1, abs=0.1)
        assert opt_result.optimal_settings['X2'] == pytest.approx(1, abs=0.1)
    
    def test_three_factor_optimization(self, three_factors):
        """Test optimization with three factors."""
        coefficients = {
            'intercept': 20,
            'X1': 2,
            'X2': 3,
            'X3': 1,
            'X1^2': -1,
            'X2^2': -1,
            'X3^2': -1
        }
        
        design, response, _ = generate_quadratic_data(
            three_factors, coefficients, noise_std=0.2, seed=42
        )
        
        analysis = ANOVAAnalysis(design, response, three_factors)
        results = analysis.fit(['X1', 'X2', 'X3', 'I(X1**2)', 'I(X2**2)', 'I(X3**2)'])
        
        opt_result = optimize_response(
            results, three_factors, objective='maximize', seed=42
        )
        
        assert opt_result.success
        
        # Optimum at X1=1, X2=1.5 (clipped to 1), X3=0.5
        assert opt_result.optimal_settings['X1'] == pytest.approx(1, abs=0.3)
        assert opt_result.optimal_settings['X2'] == pytest.approx(1, abs=0.3)
        assert opt_result.optimal_settings['X3'] == pytest.approx(0.5, abs=0.3)
    
    def test_desirability_with_unconfigured_response_raises(self):
        """Test error when response not configured in desirability."""
        df = DesirabilityFunction(['Y1', 'Y2'])
        df.add_response('Y1', 'maximize', low=5, high=10)
        # Y2 not configured
        
        with pytest.raises(ValueError, match="not configured"):
            df.evaluate({'Y1': 8, 'Y2': 7})
    
    def test_optimize_desirability_missing_model_raises(self, simple_factors):
        """Test error when model missing for a response."""
        mock_results = Mock(spec=ANOVAResults)
        models = {'Y1': mock_results}  # Y2 missing
        
        df = DesirabilityFunction(['Y1', 'Y2'])
        df.add_response('Y1', 'maximize', low=5, high=10)
        df.add_response('Y2', 'maximize', low=3, high=8)
        
        with pytest.raises(ValueError, match="No model provided"):
            optimize_desirability(models, simple_factors, df)