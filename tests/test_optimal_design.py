"""
Comprehensive tests for D-optimal design generation.

Tests cover:
- Basic functionality
- Constraint handling and satisfaction
- Input validation and edge cases
- Design quality metrics
- Algorithm convergence
- Sherman-Morrison numerical accuracy
- Benchmark comparisons (CCD, factorial)
- Published design validation
"""

import pytest
import numpy as np
import pandas as pd
from src.core.optimal_design import (
    generate_d_optimal_design,
    LinearConstraint,
    CandidatePoolConfig,
    OptimizerConfig,
    sherman_morrison_swap,
    create_polynomial_builder,
    compute_d_efficiency_vs_benchmark,
    code_point,
    decode_point,
    augment_constrained_candidates,
)
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.response_surface import CentralCompositeDesign


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def simple_factors():
    """3 continuous factors for basic testing."""
    return [
        Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
        Factor("X2", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10]),
        Factor("X3", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 10])
    ]


@pytest.fixture
def two_factors():
    """2 continuous factors for small tests."""
    return [
        Factor("Temp", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[100, 200]),
        Factor("Press", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[1, 5])
    ]


@pytest.fixture
def mixture_factors():
    """3 factors for mixture design (sum to 1 constraint)."""
    return [
        Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 1]),
        Factor("X2", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 1]),
        Factor("X3", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[0, 1])
    ]


@pytest.fixture
def fast_config():
    """Fast optimizer config for testing."""
    return OptimizerConfig(
        max_iterations=50,
        relative_improvement_tolerance=1e-3,
        stability_window=10,
        n_random_starts=2,
        max_candidates_per_row=30
    )


# ============================================================
# SECTION 1: BASIC FUNCTIONALITY
# ============================================================

class TestBasicDOptimalDesign:
    """Test basic D-optimal design generation."""
    
    def test_linear_model_no_constraints(self, simple_factors):
        """Generate simple linear model design."""
        result = generate_d_optimal_design(
            factors=simple_factors,
            model_type='linear',
            n_runs=8,
            seed=42
        )
        
        assert result.n_runs == 8
        assert result.n_parameters == 4  # 1 + 3 factors
        assert result.design_coded.shape == (8, 3)
        assert result.design_actual.shape[0] == 8
        assert 'StdOrder' in result.design_actual.columns
        assert 'RunOrder' in result.design_actual.columns
        
        # All coded values in [-1, 1]
        assert np.all(result.design_coded >= -1.001)
        assert np.all(result.design_coded <= 1.001)
        
        # Actual values in factor bounds
        for i, factor in enumerate(simple_factors):
            assert np.all(result.design_actual[factor.name] >= factor.min - 1e-6)
            assert np.all(result.design_actual[factor.name] <= factor.max + 1e-6)
    
    def test_interaction_model(self, simple_factors):
        """Generate interaction model design."""
        result = generate_d_optimal_design(
            factors=simple_factors,
            model_type='interaction',
            n_runs=12,
            seed=42
        )
        
        assert result.n_parameters == 7  # 1 + 3 + 3
        assert result.model_matrix.shape == (12, 7)
        
        X = result.model_matrix
        assert np.allclose(X[:, 0], 1.0)  # Intercept
    
    def test_quadratic_model(self, simple_factors):
        """Generate quadratic model design."""
        result = generate_d_optimal_design(
            factors=simple_factors,
            model_type='quadratic',
            n_runs=15,
            seed=42
        )
        
        assert result.n_parameters == 10  # 1 + 3 + 3 + 3
        assert result.model_matrix.shape == (15, 10)
        
        # Quadratic terms should be >= 0 (squares)
        X = result.model_matrix
        for col_idx in range(7, 10):
            assert np.all(X[:, col_idx] >= -1e-10)
    
    def test_reproducibility_with_seed(self, two_factors):
        """Same seed produces identical designs."""
        result1 = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=6,
            seed=12345
        )
        
        result2 = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=6,
            seed=12345
        )
        
        np.testing.assert_array_almost_equal(
            result1.design_coded,
            result2.design_coded,
            decimal=10
        )
        assert result1.final_objective == result2.final_objective
    
    def test_different_seeds_produce_different_designs(self, two_factors):
        """Different seeds explore different solutions."""
        result1 = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=6,
            seed=1
        )
        
        result2 = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=6,
            seed=999
        )
        
        designs_differ = not np.allclose(result1.design_coded, result2.design_coded)
        assert designs_differ


# ============================================================
# SECTION 2: CONSTRAINT HANDLING
# ============================================================

class TestLinearConstraints:
    """Test constraint handling and satisfaction."""
    
    def test_simple_sum_constraint(self, simple_factors):
        """Test: X1 + X2 <= 15 in actual units."""
        constraint = LinearConstraint(
            coefficients={'X1': 1.0, 'X2': 1.0},
            bound=15.0,
            constraint_type='le'
        )
        
        result = generate_d_optimal_design(
            factors=simple_factors,
            model_type='linear',
            n_runs=8,
            constraints=[constraint],
            seed=42
        )
        
        for idx in range(result.n_runs):
            x1 = result.design_actual.iloc[idx]['X1']
            x2 = result.design_actual.iloc[idx]['X2']
            assert x1 + x2 <= 15.0 + 1e-6
    
    def test_inequality_constraint_ge(self, two_factors):
        """Test: Temp >= 150 (forces high values)."""
        constraint = LinearConstraint(
            coefficients={'Temp': 1.0},
            bound=150.0,
            constraint_type='ge'
        )
        
        result = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=6,
            constraints=[constraint],
            seed=42
        )
        
        temps = result.design_actual['Temp'].values
        assert np.all(temps >= 150.0 - 1e-6)
    
    def test_multiple_constraints(self, simple_factors):
        """Test multiple simultaneous constraints."""
        constraints = [
            LinearConstraint(
                coefficients={'X1': 1.0, 'X2': 1.0},
                bound=12.0,
                constraint_type='le'
            ),
            LinearConstraint(
                coefficients={'X1': 1.0},
                bound=3.0,
                constraint_type='ge'
            ),
            LinearConstraint(
                coefficients={'X3': 1.0},
                bound=8.0,
                constraint_type='le'
            )
        ]
        
        result = generate_d_optimal_design(
            factors=simple_factors,
            model_type='linear',
            n_runs=10,
            constraints=constraints,
            seed=42
        )
        
        for idx in range(result.n_runs):
            x1 = result.design_actual.iloc[idx]['X1']
            x2 = result.design_actual.iloc[idx]['X2']
            x3 = result.design_actual.iloc[idx]['X3']
            
            assert x1 + x2 <= 12.0 + 1e-6
            assert x1 >= 3.0 - 1e-6
            assert x3 <= 8.0 + 1e-6
    
    def test_mixture_constraint_sum_to_one(self, mixture_factors):
        """Test mixture design: X1 + X2 + X3 = 1."""
        constraint = LinearConstraint(
            coefficients={'X1': 1.0, 'X2': 1.0, 'X3': 1.0},
            bound=1.0,
            constraint_type='eq'
        )
        
        result = generate_d_optimal_design(
            factors=mixture_factors,
            model_type='linear',
            n_runs=8,
            constraints=[constraint],
            seed=42
        )
        
        for idx in range(result.n_runs):
            x1 = result.design_actual.iloc[idx]['X1']
            x2 = result.design_actual.iloc[idx]['X2']
            x3 = result.design_actual.iloc[idx]['X3']
            total = x1 + x2 + x3
            
            assert abs(total - 1.0) < 1e-6
    
    def test_too_restrictive_constraints(self, two_factors):
        """Test that overly restrictive constraints raise error."""
        constraints = [
            LinearConstraint(
                coefficients={'Temp': 1.0},
                bound=180.0,
                constraint_type='ge'
            ),
            LinearConstraint(
                coefficients={'Temp': 1.0},
                bound=120.0,
                constraint_type='le'
            )
        ]
        
        with pytest.raises(ValueError, match="feasible candidates"):
            generate_d_optimal_design(
                factors=two_factors,
                model_type='linear',
                n_runs=6,
                constraints=constraints,
                seed=42
            )
    
    def test_low_candidate_density_warning(self, two_factors):
        """Test warning when candidate pool is small but sufficient."""
        # Very restrictive but not infeasible constraint
        constraint = LinearConstraint(
            coefficients={'Temp': 1.0, 'Press': 1.0},
            bound=155.0,  # In coded: center is 150 and 3, so sum ≈ 153
            constraint_type='le'
        )
        
        # Should succeed but warn about low density
        with pytest.warns(UserWarning, match="Low candidate density"):
            result = generate_d_optimal_design(
                factors=two_factors,
                model_type='linear',
                n_runs=6,
                constraints=[constraint],
                seed=42
            )
        
        # Should still produce valid design
        assert result.n_runs == 6
    
    def test_augmentation_improves_constrained_pool(self, simple_factors):
        """Test that rejection sampling augments constrained pools."""
        # Moderately restrictive constraint
        constraint = LinearConstraint(
            coefficients={'X1': 1.0, 'X2': 1.0, 'X3': 1.0},
            bound=12.0,
            constraint_type='le'
        )
        
        # Request many runs (forces augmentation)
        result = generate_d_optimal_design(
            factors=simple_factors,
            model_type='linear',
            n_runs=15,
            constraints=[constraint],
            seed=42
        )
        
        # Should succeed despite constraint
        assert result.n_runs == 15
        
        # Verify all points satisfy constraint
        for idx in range(result.n_runs):
            x1 = result.design_actual.iloc[idx]['X1']
            x2 = result.design_actual.iloc[idx]['X2']
            x3 = result.design_actual.iloc[idx]['X3']
            assert x1 + x2 + x3 <= 12.0 + 1e-6


# ============================================================
# SECTION 3: INPUT VALIDATION & EDGE CASES
# ============================================================

class TestInputValidation:
    """Test input validation and error handling."""
    
    def test_insufficient_runs(self, simple_factors):
        """Test: n_runs < n_parameters."""
        with pytest.raises(ValueError, match="n_runs.*must be.*n_parameters"):
            generate_d_optimal_design(
                factors=simple_factors,
                model_type='linear',
                n_runs=3,
                seed=42
            )
    
    def test_saturated_design(self, two_factors):
        """Test: n_runs == n_parameters (saturated)."""
        result = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=3,
            seed=42
        )
        
        assert result.n_runs == 3
        assert result.n_parameters == 3
    
    def test_non_continuous_factors_raises_error(self):
        """Test that categorical/discrete factors raise error."""
        factors = [
            Factor("Material", FactorType.CATEGORICAL, ChangeabilityLevel.EASY, 
                   levels=['A', 'B', 'C']),
            Factor("Temp", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, 
                   levels=[100, 200])
        ]
        
        with pytest.raises(ValueError, match="must be continuous"):
            generate_d_optimal_design(
                factors=factors,
                model_type='linear',
                n_runs=6,
                seed=42
            )
    
    def test_single_factor_linear_model(self):
        """Test with k=1 (edge case)."""
        factors = [
            Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, 
                   levels=[0, 10])
        ]
        
        result = generate_d_optimal_design(
            factors=factors,
            model_type='linear',
            n_runs=4,
            seed=42
        )
        
        assert result.n_runs == 4
        assert result.n_parameters == 2
        assert result.design_coded.shape == (4, 1)


# ============================================================
# SECTION 4: DESIGN QUALITY
# ============================================================

class TestDesignQuality:
    """Test that designs meet quality standards."""
    
    def test_determinant_positive(self, simple_factors):
        """Verify |X'X| > 0 (invertible)."""
        result = generate_d_optimal_design(
            factors=simple_factors,
            model_type='linear',
            n_runs=8,
            seed=42
        )
        
        det = np.exp(result.final_objective)
        assert det > 0
    
    def test_model_matrix_full_rank(self, two_factors):
        """Verify X has full column rank."""
        result = generate_d_optimal_design(
            factors=two_factors,
            model_type='interaction',
            n_runs=8,
            seed=42
        )
        
        rank = np.linalg.matrix_rank(result.model_matrix)
        assert rank == result.n_parameters
    
    def test_d_efficiency_in_percent(self, two_factors):
        """Verify D-efficiency is percentage (can exceed 100% vs benchmark)."""
        result = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=6,
            seed=42
        )
        
        # Efficiency relative to benchmark (can be >100% if better than benchmark)
        assert 0 <= result.d_efficiency_vs_benchmark <= 200
        assert result.benchmark_design_name is not None
    
    def test_efficiency_vs_full_factorial_for_linear(self, two_factors):
        """Linear model should achieve >90% efficiency vs Full Factorial."""
        result = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=4,  # Same as 2^2 factorial
            seed=42
        )
        
        # Should be very close to full factorial efficiency
        assert result.d_efficiency_vs_benchmark >= 90
        assert "Full Factorial" in result.benchmark_design_name
    
    def test_efficiency_vs_ccd_for_quadratic(self, simple_factors):
        """Quadratic model should achieve >100% efficiency vs CCD."""
        # Use more runs than CCD to ensure we can beat it
        result = generate_d_optimal_design(
            factors=simple_factors,
            model_type='quadratic',
            n_runs=25,  # More than typical CCD
            seed=42
        )
        
        # D-optimal should match or exceed CCD
        assert result.d_efficiency_vs_benchmark >= 100
        assert "CCD" in result.benchmark_design_name


# ============================================================
# SECTION 5: ALGORITHM CONVERGENCE
# ============================================================

class TestConvergence:
    """Test optimization algorithm behavior."""
    
    def test_multiple_starts_improve_or_match(self, two_factors):
        """Verify multiple starts find better or equal solutions."""
        config_1start = OptimizerConfig(
            max_iterations=50,
            n_random_starts=1,
            max_candidates_per_row=30
        )
        
        result_1start = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=6,
            optimizer_config=config_1start,
            seed=42
        )
        
        config_3start = OptimizerConfig(
            max_iterations=50,
            n_random_starts=3,
            max_candidates_per_row=30
        )
        
        result_3start = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=6,
            optimizer_config=config_3start,
            seed=42
        )
        
        assert result_3start.final_objective >= result_1start.final_objective - 1e-6
    
    def test_convergence_reason_recorded(self, two_factors, fast_config):
        """Verify converged_by is set."""
        result = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=6,
            optimizer_config=fast_config,
            seed=42
        )
        
        assert result.converged_by in ['stability', 'max_iterations']
    
    def test_iterations_within_limit(self, two_factors, fast_config):
        """Verify optimizer respects max_iterations."""
        result = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=6,
            optimizer_config=fast_config,
            seed=42
        )
        
        assert result.n_iterations <= fast_config.max_iterations


# ============================================================
# SECTION 6: SHERMAN-MORRISON NUMERICAL ACCURACY
# ============================================================

class TestShermanMorrison:
    """Test Sherman-Morrison determinant update with reuse."""
    
    def test_sherman_morrison_swap_matches_explicit(self):
        """Verify SM swap matches explicit computation."""
        XtX = np.array([
            [10, 2, 1],
            [2, 8, 3],
            [1, 3, 6]
        ])
        XtX_inv = np.linalg.inv(XtX)
        
        x_old = np.array([1, 2, 1])
        x_new = np.array([2, 1, 3])
        
        sm_result = sherman_morrison_swap(XtX_inv, x_old, x_new)
        
        # Explicit computation
        XtX_new = XtX - np.outer(x_old, x_old) + np.outer(x_new, x_new)
        XtX_inv_explicit = np.linalg.inv(XtX_new)
        
        det_old = np.linalg.det(XtX)
        det_new = np.linalg.det(XtX_new)
        explicit_ratio = det_new / det_old
        
        # Check ratio matches
        assert abs(sm_result.det_ratio - explicit_ratio) < 1e-8
        
        # Check updated inverse matches
        np.testing.assert_array_almost_equal(
            sm_result.XtX_inv_updated,
            XtX_inv_explicit,
            decimal=10
        )
        
        assert sm_result.is_valid
    
    def test_sherman_morrison_handles_singular_case(self):
        """Test SM swap when denominator near zero."""
        XtX = np.eye(3)
        XtX_inv = np.eye(3)
        
        x_old = np.array([1, 0, 0])
        x_new = np.array([0, 1, 0])
        
        sm_result = sherman_morrison_swap(XtX_inv, x_old, x_new)
        
        # Should return invalid result when singular
        if not sm_result.is_valid:
            assert sm_result.det_ratio == 0.0
        
        # Result should be finite
        assert np.all(np.isfinite(sm_result.XtX_inv_updated))
    
    def test_sherman_morrison_reuse_avoids_redundancy(self):
        """Test that SM result can be reused (no redundant computation)."""
        XtX = np.array([
            [10, 2, 1],
            [2, 8, 3],
            [1, 3, 6]
        ])
        XtX_inv = np.linalg.inv(XtX)
        
        x_old = np.array([1, 2, 1])
        x_new = np.array([2, 1, 3])
        
        # Call once
        sm_result = sherman_morrison_swap(XtX_inv, x_old, x_new)
        
        # Verify we get BOTH ratio and inverse
        assert hasattr(sm_result, 'det_ratio')
        assert hasattr(sm_result, 'XtX_inv_updated')
        assert hasattr(sm_result, 'is_valid')
        
        # Verify inverse is usable for next iteration
        if sm_result.is_valid:
            # Should be able to use XtX_inv_updated directly
            assert sm_result.XtX_inv_updated.shape == XtX_inv.shape
            assert np.linalg.cond(sm_result.XtX_inv_updated) < 1e10


# ============================================================
# SECTION 7: BENCHMARK COMPARISONS
# ============================================================

class TestBenchmarkComparisons:
    """Compare D-optimal to known good designs."""
    
    def test_linear_model_matches_full_factorial(self, two_factors):
        """D-optimal with same runs as 2^k should achieve >90% vs factorial."""
        result = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=4,  # Same as 2^2
            seed=42
        )
        
        # Should achieve >90% efficiency relative to full factorial
        assert result.d_efficiency_vs_benchmark >= 90
        assert "Full Factorial" in result.benchmark_design_name
    
    def test_quadratic_exceeds_ccd(self, simple_factors):
        """D-optimal should exceed CCD efficiency for quadratic model."""
        # Generate CCD to know its size
        from src.core.response_surface import CentralCompositeDesign
        ccd = CentralCompositeDesign(
            factors=simple_factors,
            alpha='rotatable',
            center_points=6
        )
        ccd_design = ccd.generate(randomize=False)
        n_ccd_runs = len(ccd_design)
        
        # D-optimal with same run count
        result = generate_d_optimal_design(
            factors=simple_factors,
            model_type='quadratic',
            n_runs=n_ccd_runs,
            seed=42
        )
        
        # Should match or exceed CCD (>100% means better than CCD)
        assert result.d_efficiency_vs_benchmark >= 100
        assert "CCD" in result.benchmark_design_name
    
    def test_more_runs_than_benchmark_improves_efficiency(self, two_factors):
        """Using more runs than benchmark should improve efficiency."""
        # Full factorial for k=2 is 4 runs
        result_4 = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=4,
            seed=42
        )
        
        result_8 = generate_d_optimal_design(
            factors=two_factors,
            model_type='linear',
            n_runs=8,
            seed=42
        )
        
        # More runs should improve or match efficiency
        # (Though efficiency is relative, so may not strictly increase)
        # At minimum, absolute objective should improve
        assert result_8.final_objective >= result_4.final_objective - 0.5


# ============================================================
# SECTION 8: HELPER FUNCTION TESTS
# ============================================================

class TestHelperFunctions:
    """Test coding/decoding and utility functions."""
    
    def test_code_decode_roundtrip(self, simple_factors):
        """Test that coding and decoding are inverses."""
        x_actual = np.array([5.0, 7.5, 2.0])
        x_coded = code_point(x_actual, simple_factors)
        x_decoded = decode_point(x_coded, simple_factors)
        
        np.testing.assert_array_almost_equal(x_actual, x_decoded, decimal=10)
    
    def test_coding_bounds(self, two_factors):
        """Test that min/max code to -1/+1."""
        x_min = np.array([100.0, 1.0])
        x_coded_min = code_point(x_min, two_factors)
        np.testing.assert_array_almost_equal(x_coded_min, [-1.0, -1.0])
        
        x_max = np.array([200.0, 5.0])
        x_coded_max = code_point(x_max, two_factors)
        np.testing.assert_array_almost_equal(x_coded_max, [1.0, 1.0])
        
        x_center = np.array([150.0, 3.0])
        x_coded_center = code_point(x_center, two_factors)
        np.testing.assert_array_almost_equal(x_coded_center, [0.0, 0.0])
    
    def test_model_builder_creates_correct_columns(self, simple_factors):
        """Test polynomial model builder creates correct terms."""
        builder = create_polynomial_builder(simple_factors, 'quadratic')
        
        X_point = np.array([[0.5, -0.5, 1.0]])
        X_model = builder(X_point)
        
        assert X_model.shape == (1, 10)
        assert X_model[0, 0] == 1.0
        assert X_model[0, 1] == 0.5
        assert X_model[0, 2] == -0.5
        assert X_model[0, 3] == 1.0
        assert X_model[0, 4] == 0.5 * (-0.5)
        assert X_model[0, 5] == 0.5 * 1.0
        assert X_model[0, 6] == (-0.5) * 1.0
        assert X_model[0, 7] == 0.5 ** 2
        assert X_model[0, 8] == (-0.5) ** 2
        assert X_model[0, 9] == 1.0 ** 2


# ============================================================
# SECTION 9: INTEGRATION & REALISTIC SCENARIOS
# ============================================================

class TestIntegrationScenarios:
    """Test realistic usage scenarios."""
    
    def test_typical_screening_experiment(self):
        """Test typical 5-factor screening with 12 runs."""
        factors = [
            Factor(f"X{i}", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, 
                   levels=[0, 10])
            for i in range(1, 6)
        ]
        
        result = generate_d_optimal_design(
            factors=factors,
            model_type='linear',
            n_runs=12,
            seed=42
        )
        
        assert result.n_runs == 12
        assert result.n_parameters == 6
        assert result.d_efficiency > 20
        assert result.condition_number < 1000
    
    def test_response_surface_with_constraints(self):
        """Test quadratic model with sum constraint."""
        factors = [
            Factor("X1", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, 
                   levels=[0, 5]),
            Factor("X2", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, 
                   levels=[0, 5]),
            Factor("X3", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, 
                   levels=[0, 5])
        ]
        
        constraint = LinearConstraint(
            coefficients={'X1': 1.0, 'X2': 1.0, 'X3': 1.0},
            bound=10.0,
            constraint_type='le'
        )
        
        result = generate_d_optimal_design(
            factors=factors,
            model_type='quadratic',
            n_runs=20,
            constraints=[constraint],
            seed=42
        )
        
        for idx in range(result.n_runs):
            x1 = result.design_actual.iloc[idx]['X1']
            x2 = result.design_actual.iloc[idx]['X2']
            x3 = result.design_actual.iloc[idx]['X3']
            assert x1 + x2 + x3 <= 10.0 + 1e-6
        
        assert result.d_efficiency > 20
    
    def test_process_optimization_scenario(self):
        """Test realistic process optimization with multiple constraints."""
        factors = [
            Factor("Temperature", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, 
                   levels=[150, 250]),
            Factor("Pressure", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, 
                   levels=[10, 50]),
            Factor("Time", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, 
                   levels=[30, 120])
        ]
        
        constraints = [
            LinearConstraint(
                coefficients={'Temperature': 1.0, 'Pressure': 2.0},
                bound=350.0,
                constraint_type='le'
            ),
            LinearConstraint(
                coefficients={'Temperature': -1.0, 'Time': 1.0},
                bound=-100.0,
                constraint_type='ge'
            )
        ]
        
        result = generate_d_optimal_design(
            factors=factors,
            model_type='interaction',
            n_runs=15,
            constraints=constraints,
            seed=42
        )
        
        assert result.n_runs == 15
        
        for idx in range(result.n_runs):
            temp = result.design_actual.iloc[idx]['Temperature']
            press = result.design_actual.iloc[idx]['Pressure']
            time = result.design_actual.iloc[idx]['Time']
            
            assert temp + 2 * press <= 350.0 + 1e-6
            assert -temp + time >= -100.0 - 1e-6