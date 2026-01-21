# I-Optimality Extension: Implementation Plan

## 1. Step-by-Step Implementation Plan

### Phase 1: Core Infrastructure (Day 1)

**Step 1.1: Create I-Optimal Criterion Class**
- Location: `src/core/optimal_design.py`
- Add `IOptimalityCriterion` class after `DOptimalityCriterion`
- Implement `objective()` using trace((X'X)^(-1) * M)
- Precompute moment matrix M from prediction grid

**Step 1.2: Create Prediction Grid Generator**
- Add `generate_prediction_grid()` function
- Support factorial and LHS grid types
- Auto-select grid type based on k (factorial for k≤4, LHS for k>4)

**Step 1.3: Create Criterion Factory**
- Add `create_optimality_criterion()` function
- Takes `criterion_type: Literal['D', 'I']`
- Returns appropriate criterion object

**Step 1.4: Update CEXCH Optimizer**
- Change signature to accept `criterion: OptimalityCriterion`
- Replace hardcoded D-optimal logic with `criterion.objective(X)`
- Update Sherman-Morrison usage for I-optimal (reuse inverse for trace computation)

### Phase 2: API Generalization (Day 2)

**Step 2.1: Generalize High-Level API**
- Rename function: `generate_optimal_design()` (was `generate_d_optimal_design`)
- Add `criterion: Literal['D', 'I'] = 'D'` parameter
- Add `prediction_grid_config: Optional[Dict] = None` parameter
- Keep old function as deprecated wrapper

**Step 2.2: Update OptimizationResult**
- Ensure `criterion_type` field uses correct value
- Add `i_criterion` field (optional, only for I-optimal)
- Add `i_efficiency_vs_benchmark` field (optional)

**Step 2.3: Update Augmentation Functions**
- Add `criterion` parameter to `augment_for_model_extension()`
- Add `prediction_grid_config` parameter
- Update `execute_optimal_plan()` to extract criterion from config

### Phase 3: Efficiency Metrics (Day 3)

**Step 3.1: I-Efficiency Computation**
- Add `compute_i_efficiency()` function
- Formula: `(benchmark_I / actual_I) * 100`

**Step 3.2: Generalized Efficiency Function**
- Add `compute_efficiency_vs_benchmark()` dispatcher
- Routes to D or I efficiency based on criterion type

**Step 3.3: Benchmark Computation**
- Update `compute_benchmark_criterion()` to support both D and I
- For I-optimal: compute I-criterion of benchmark design (CCD or FF)

### Phase 4: Call Site Updates (Day 4)

**Step 4.1: Augmentation Module**
```python
# src/core/augmentation/optimal.py
- Update augment_for_model_extension() signature
- Update execute_optimal_plan() to pass criterion
```

**Step 4.2: Goal-Driven Augmentation**
```python
# src/core/augmentation/goal_driven.py
- Add 'i_optimal' strategy in _create_strategy_config()
- Set criterion='I' in OptimalAugmentConfig
```

**Step 4.3: Goals Catalog**
```python
# src/core/augmentation/goals.py
- Update IMPROVE_PREDICTION goal to use I-optimal
- Update typical_strategies to include 'i_optimal'
```

**Step 4.4: Diagnostics**
```python
# src/core/diagnostics/variance.py
- Add I-criterion computation in quality metrics
- Compute average prediction variance
```

### Phase 5: UI Updates (Day 5)

**Step 5.1: Augmentation Wizard**
- Display criterion type in plan details
- Show I-efficiency for I-optimal plans

**Step 5.2: Diagnostics Page**
- Display both D and I metrics
- Show prediction variance profile

### Phase 6: Testing (Day 6)

**Step 6.1: Unit Tests**
- Test I-criterion computation
- Test prediction grid generation
- Test backward compatibility

**Step 6.2: Integration Tests**
- Test D-optimal unchanged
- Test I-optimal produces different designs
- Test augmentation with both criteria

**Step 6.3: Validation**
- Compare I-optimal to known benchmarks
- Verify prediction variance is minimized

---

## 2. Updated Function Signatures

### Core Functions

```python
# NEW FUNCTIONS

class IOptimalityCriterion(OptimalityCriterion):
    def __init__(
        self,
        prediction_points: np.ndarray,
        model_builder: ModelMatrixBuilder,
        ridge: float = 1e-10
    ):
        ...
    
    def objective(self, X_model: np.ndarray) -> float:
        """Return -I for maximization."""
        ...

def generate_prediction_grid(
    factors: List[Factor],
    config: Optional[Dict] = None
) -> np.ndarray:
    """Generate prediction grid for I-optimality."""
    ...

def create_optimality_criterion(
    criterion_type: Literal['D', 'I'],
    model_builder: ModelMatrixBuilder,
    factors: List[Factor],
    prediction_grid_config: Optional[Dict] = None,
    ridge: float = 1e-10
) -> OptimalityCriterion:
    """Factory for criterion objects."""
    ...

def compute_i_efficiency(
    i_criterion: float,
    n_runs: int,
    n_params: int,
    benchmark_i_criterion: float
) -> float:
    """Compute I-efficiency percentage."""
    ...

# MODIFIED FUNCTIONS

def cexch_optimize(
    candidates: np.ndarray,
    n_runs: int,
    model_builder: ModelMatrixBuilder,
    criterion: OptimalityCriterion,  # NEW parameter
    config: OptimizerConfig,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, float, int, str]:
    ...

def generate_optimal_design(  # Renamed from generate_d_optimal_design
    factors: List[Factor],
    model_type: Literal['linear', 'interaction', 'quadratic'],
    n_runs: int,
    criterion: Literal['D', 'I'] = 'D',  # NEW parameter
    constraints: Optional[List[LinearConstraint]] = None,
    prediction_grid_config: Optional[Dict] = None,  # NEW parameter
    candidate_config: Optional[CandidatePoolConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    seed: Optional[int] = None
) -> OptimizationResult:
    ...

def augment_for_model_extension(
    original_design: pd.DataFrame,
    factors: List[Factor],
    current_model_terms: List[str],
    new_model_terms: List[str],
    n_runs_to_add: int,
    criterion: Literal['D', 'I'] = 'D',  # NEW parameter
    prediction_grid_config: Optional[Dict] = None,  # NEW parameter
    seed: Optional[int] = None
) -> AugmentedDesign:
    ...

# DEPRECATED (for backward compatibility)

def generate_d_optimal_design(*args, **kwargs) -> OptimizationResult:
    """DEPRECATED: Use generate_optimal_design(..., criterion='D')."""
    warnings.warn(...)
    kwargs['criterion'] = 'D'
    return generate_optimal_design(*args, **kwargs)
```

---

## 3. Call Site Changes Summary

### Modified Files

**`src/core/optimal_design.py`** (Core implementation)
- Add IOptimalityCriterion class
- Add generate_prediction_grid()
- Add create_optimality_criterion()
- Update cexch_optimize() to accept criterion
- Update generate_optimal_design() (renamed + new params)
- Add compute_i_efficiency()
- Keep generate_d_optimal_design() as deprecated wrapper

**`src/core/augmentation/optimal.py`**
```diff
 def augment_for_model_extension(
     ...,
+    criterion: Literal['D', 'I'] = 'D',
+    prediction_grid_config: Optional[Dict] = None,
     ...
 ):
     ...
-    new_run_indices = _augmented_coordinate_exchange(...)
+    # Create criterion object
+    criterion_obj = create_optimality_criterion(criterion, ...)
+    new_run_indices = _augmented_coordinate_exchange(..., criterion=criterion_obj)
```

**`src/core/augmentation/goal_driven.py`**
```diff
 def _create_strategy_config(...):
     ...
+    elif strategy_name == 'i_optimal':
+        current_terms = _get_current_model_terms(diagnostics)
+        return OptimalAugmentConfig(
+            new_model_terms=current_terms,
+            n_runs_to_add=n_runs,
+            criterion='I'
+        )
```

**`src/core/augmentation/goals.py`**
```diff
     AugmentationGoal.IMPROVE_PREDICTION: GoalDescription(
         ...,
-        typical_strategies=["i_optimal", "space_filling", "uniform_precision"],
+        typical_strategies=["i_optimal", "d_optimal_prediction", "space_filling"],
         ...
     ),
```

**`src/core/diagnostics/variance.py`**
```diff
 def compute_design_quality_metrics(...):
     ...
+    # I-optimality metrics
+    prediction_grid = generate_prediction_grid(factors)
+    X_pred = build_model_matrix(...)
+    M = (X_pred.T @ X_pred) / len(prediction_grid)
+    i_criterion = np.trace(XtX_inv @ M)
+    
     return {
         'd_efficiency': ...,
+        'i_criterion': i_criterion,
+        'avg_prediction_variance': i_criterion,
     }
```

**`src/ui/components/augmentation_wizard.py`**
```diff
 def _display_single_plan(plan, ...):
     config = plan.strategy_config
     
+    if hasattr(config, 'criterion'):
+        criterion_text = {
+            'D': 'D-Optimal (Parameter Precision)',
+            'I': 'I-Optimal (Prediction Quality)'
+        }.get(config.criterion, config.criterion)
+        
+        st.markdown(f"**Optimality Criterion:** {criterion_text}")
```

---

## 4. Compatibility Checklist

### Backward Compatibility

- [ ] `generate_d_optimal_design()` kept as deprecated function
- [ ] Default `criterion='D'` in all new functions
- [ ] `OptimalAugmentConfig.criterion` defaults to 'D'
- [ ] Existing tests pass without modification
- [ ] Deprecation warnings guide users to new API

### Forward Compatibility

- [ ] New `criterion` parameter accepts 'D' and 'I'
- [ ] Can add 'A', 'G', 'V' criteria in future without breaking changes
- [ ] `OptimizationResult` extensible with new fields
- [ ] Criterion objects follow protocol pattern

### Data Compatibility

- [ ] Designs created before I-optimal addition still load
- [ ] Session state handles both D and I designs
- [ ] CSV exports include criterion type in metadata

---

## 5. Test Plan

### Unit Tests (`tests/test_optimal_design_i_criterion.py`)

```python
class TestPredictionGrid:
    def test_factorial_grid_generation(self):
        """Factorial grid for k=3."""
        grid = generate_prediction_grid(factors, {'n_points_per_dim': 5})
        assert grid.shape == (125, 3)  # 5^3
    
    def test_lhs_grid_for_large_k(self):
        """LHS grid auto-selected for k>4."""
        factors_5 = [...]  # 5 factors
        grid = generate_prediction_grid(factors_5)
        # Should use LHS, not 5^5=3125 points

class TestIOptimalCriterion:
    def test_i_criterion_computation(self):
        """I-criterion computed correctly."""
        criterion = IOptimalityCriterion(...)
        X = np.random.randn(10, 5)
        obj = criterion.objective(X)
        assert obj < 0  # Negative for maximization
    
    def test_i_criterion_better_than_random(self):
        """I-optimal design has lower I-criterion than random."""
        design_i = generate_optimal_design(..., criterion='I')
        design_random = generate_random_design(...)
        
        assert design_i.i_criterion < compute_i_criterion(design_random)

class TestBackwardCompatibility:
    def test_d_optimal_still_works(self):
        """Old D-optimal code unchanged."""
        design = generate_optimal_design(factors, 'linear', 12)
        assert design.criterion_type == 'D-optimal'
        assert design.d_efficiency_vs_benchmark > 85
    
    def test_deprecated_function_warns(self):
        """Old API warns about deprecation."""
        with pytest.warns(DeprecationWarning):
            design = generate_d_optimal_design(factors, 'linear', 12)

class TestIDifference:
    def test_d_vs_i_produces_different_designs(self):
        """D and I-optimal give different designs."""
        design_d = generate_optimal_design(..., criterion='D', seed=42)
        design_i = generate_optimal_design(..., criterion='I', seed=42)
        
        # Designs should differ
        assert not np.allclose(
            design_d.design_coded, 
            design_i.design_coded
        )
        
        # I-optimal should have better prediction variance
        i_d = compute_i_criterion(design_d.design_coded, ...)
        i_i = compute_i_criterion(design_i.design_coded, ...)
        assert i_i < i_d
```

### Integration Tests (`tests/test_augmentation_i_optimal.py`)

```python
class TestAugmentationWithICriterion:
    def test_i_optimal_augmentation(self):
        """Augmentation using I-optimal criterion."""
        plan = AugmentationPlan(
            ...,
            strategy_config=OptimalAugmentConfig(
                ...,
                criterion='I'
            )
        )
        
        augmented = plan.execute()
        assert augmented.plan_executed.strategy_config.criterion == 'I'
    
    def test_goal_driven_i_optimal(self):
        """IMPROVE_PREDICTION goal uses I-optimal."""
        request = AugmentationRequest(
            mode='enhance_design',
            diagnostics=diagnostics,
            selected_goal=AugmentationGoal.IMPROVE_PREDICTION
        )
        
        plans = recommend_augmentation(request)
        primary_plan = plans[0]
        
        assert primary_plan.strategy_config.criterion == 'I'
```

### Validation Tests (`tests/test_i_optimal_validation.py`)

```python
class TestIOptimalValidation:
    def test_i_optimal_vs_ccd_benchmark(self):
        """I-optimal should match or beat CCD for prediction."""
        # Generate CCD
        ccd = generate_ccd(factors, ...)
        i_ccd = compute_i_criterion(ccd)
        
        # Generate I-optimal with same number of runs
        design_i = generate_optimal_design(
            factors, 'quadratic', n_runs=len(ccd), criterion='I'
        )
        
        # I-optimal should be at least as good
        assert design_i.i_criterion <= i_ccd * 1.05  # 5% tolerance
    
    def test_prediction_variance_uniformity(self):
        """I-optimal has more uniform prediction variance."""
        design_d = generate_optimal_design(..., criterion='D')
        design_i = generate_optimal_design(..., criterion='I')
        
        # Compute variance at many points
        pred_points = generate_prediction_grid(factors, {'n_points_per_dim': 10})
        
        vars_d = compute_prediction_variances(design_d, pred_points)
        vars_i = compute_prediction_variances(design_i, pred_points)
        
        # I-optimal should have lower std dev of variances
        assert np.std(vars_i) < np.std(vars_d)
```

---

## 6. Documentation Updates

### User-Facing Docs

**File:** `docs/user_guide/optimal_designs.md`

```markdown
# Optimal Design Generation

## D-Optimal vs I-Optimal

**When to use D-optimal:**
- Goal: Precise parameter estimates
- Application: Screening experiments, parameter estimation
- Minimizes: Volume of confidence ellipsoid
- Best for: Understanding which factors matter and by how much

**When to use I-optimal:**
- Goal: Accurate predictions across design space
- Application: Response surface modeling, prediction at untested conditions
- Minimizes: Average prediction variance
- Best for: Creating prediction equations for use in production

## Examples

```python
# D-optimal (default)
design_d = generate_optimal_design(
    factors, 'quadratic', n_runs=20
)

# I-optimal
design_i = generate_optimal_design(
    factors, 'quadratic', n_runs=20,
    criterion='I',
    prediction_grid_config={'n_points_per_dim': 7}
)
```

### API Reference

**File:** `docs/api/optimal_design.md`

```markdown
## generate_optimal_design()

Generate optimal experimental design.

**Parameters:**
- `criterion`: {'D', 'I'} - Optimality criterion (default: 'D')
  - 'D': D-optimal (maximize det(X'X))
  - 'I': I-optimal (minimize avg prediction variance)
- `prediction_grid_config`: dict - I-optimal grid configuration
  - 'n_points_per_dim': int (default: 5)
  - 'grid_type': {'factorial', 'lhs'} (default: auto)

**Returns:**
- `OptimizationResult` with fields:
  - `criterion_type`: 'D-optimal' or 'I-optimal'
  - `d_efficiency_vs_benchmark`: float (for D-optimal)
  - `i_criterion`: float (for I-optimal, optional)
  - `i_efficiency_vs_benchmark`: float (for I-optimal, optional)
```

---

## Summary

This plan provides:

✅ **Clean Extension** - I-optimal added without touching D-optimal implementation
✅ **Code Reuse** - Shared candidates, model matrix, exchange algorithm
✅ **Clear Abstraction** - Criterion protocol separates "what to optimize" from "how to optimize"
✅ **Backward Compatible** - Default to D-optimal, deprecated wrapper for old API
✅ **Fully Tested** - Unit, integration, and validation tests
✅ **Well Documented** - User guide and API reference updates

**Total Implementation Time:** ~6 days for core team member
**Risk Level:** Low (no breaking changes, comprehensive tests)
