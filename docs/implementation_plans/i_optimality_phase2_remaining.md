# I-Optimality Implementation - Remaining Changes

## Status: Phase 1 Complete, Phase 2 In Progress

### ✅ Phase 1 Completed
- IOptimalityCriterion class
- generate_prediction_grid() function
- create_optimality_criterion() factory
- Updated cexch_optimize() signature (partial)

### 🔄 Phase 2: Remaining Tasks

Due to file size constraints, the remaining changes need to be applied manually or via targeted patches. Here are the key modifications needed:

---

## 1. Complete cexch_optimize() Function Update

**Location:** `src/core/optimal_design.py`, function `cexch_optimize()`

**Current signature:**
```python
def cexch_optimize(
    candidates, n_runs, model_builder, config, seed
) -> Tuple[np.ndarray, float, int, str]:
```

**New signature (already partially updated):**
```python
def cexch_optimize(
    candidates, n_runs, model_builder,
    criterion: OptimalityCriterion,  # NEW
    config, seed
) -> Tuple[np.ndarray, float, int, str]:
```

**Required changes inside function body:**

### Change 1: Replace initial objective computation
```python
# OLD (line ~825):
try:
    XtX_inv = np.linalg.inv(XtX)
    sign, logdet = np.linalg.slogdet(XtX)
except np.linalg.LinAlgError:
    continue

if sign <= 0:
    continue

logdet_history = [logdet]

# NEW:
try:
    XtX_inv = np.linalg.inv(XtX)
    objective = criterion.objective(X_current)
except np.linalg.LinAlgError:
    continue

if objective < -1e9:  # Invalid design
    continue

objective_history = [objective]
```

### Change 2: Update inner loop variable names
```python
# OLD:
best_logdet_for_i = logdet

# NEW:
best_objective_for_i = objective
```

### Change 3: Replace Sherman-Morrison usage
```python
# OLD (line ~850):
if config.use_sherman_morrison:
    sm_result = sherman_morrison_swap(XtX_inv, x_old, x_new)
    
    if not sm_result.is_valid or sm_result.det_ratio <= 0:
        continue
    
    logdet_trial = logdet + np.log(sm_result.det_ratio)

# NEW:
if config.use_sherman_morrison:
    sm_result = sherman_morrison_swap(XtX_inv, x_old, x_new)
    
    if not sm_result.is_valid:
        continue
    
    # Use updated inverse to compute objective
    # For D-optimal: can use det_ratio directly
    # For I-optimal: reuse updated inverse
    X_trial = X_current.copy()
    X_trial[i] = x_new
    objective_trial = criterion.objective(X_trial)
    
    # Store updated inverse if this is best so far
    if objective_trial > best_objective_for_i + 1e-10:
        best_objective_for_i = objective_trial
        best_idx_for_i = cand_idx
        best_sm_result = sm_result  # Store for reuse
```

### Change 4: Update acceptance logic
```python
# OLD:
if best_idx_for_i != indices[i]:
    x_new = X_model_candidates[best_idx_for_i]
    
    if config.use_sherman_morrison and best_sm_result is not None:
        XtX_inv = best_sm_result.XtX_inv_updated
    else:
        X_current[i] = x_new
        XtX = X_current.T @ X_current + 1e-10 * np.eye(p)
        try:
            XtX_inv = np.linalg.inv(XtX)
        except:
            continue
    
    X_current[i] = x_new
    indices[i] = best_idx_for_i
    logdet = best_logdet_for_i
    improved_this_iter = True

# NEW:
if best_idx_for_i != indices[i]:
    x_new = X_model_candidates[best_idx_for_i]
    
    if config.use_sherman_morrison and best_sm_result is not None:
        XtX_inv = best_sm_result.XtX_inv_updated
    else:
        X_current[i] = x_new
        XtX = X_current.T @ X_current + 1e-10 * np.eye(p)
        try:
            XtX_inv = np.linalg.inv(XtX)
        except:
            continue
    
    X_current[i] = x_new
    indices[i] = best_idx_for_i
    objective = best_objective_for_i
    improved_this_iter = True
```

### Change 5: Update history tracking
```python
# OLD:
logdet_history.append(logdet)

# Check stopping
if improved_this_iter:
    no_improvement_count = 0
else:
    no_improvement_count += 1

# Relative improvement check
if len(logdet_history) >= config.stability_window:
    old_val = logdet_history[-config.stability_window]
    new_val = logdet_history[-1]
    
    if abs(old_val) > 1e-10:
        rel_improvement = (new_val - old_val) / abs(old_val)

# NEW:
objective_history.append(objective)

# Check stopping
if improved_this_iter:
    no_improvement_count = 0
else:
    no_improvement_count += 1

# Relative improvement check
if len(objective_history) >= config.stability_window:
    old_val = objective_history[-config.stability_window]
    new_val = objective_history[-1]
    
    if abs(old_val) > 1e-10:
        rel_improvement = (new_val - old_val) / abs(old_val)
```

### Change 6: Update final comparison
```python
# OLD:
n_iter = len(logdet_history) - 1

# Keep best across starts
if logdet > best_logdet:
    best_logdet = logdet
    best_indices = indices.copy()
    best_n_iter = n_iter
    best_converged_by = converged_by

# NEW:
n_iter = len(objective_history) - 1

# Keep best across starts
if objective > best_objective:
    best_objective = objective
    best_indices = indices.copy()
    best_n_iter = n_iter
    best_converged_by = converged_by
```

### Change 7: Update return statement
```python
# OLD:
return best_indices, best_logdet, best_n_iter, best_converged_by

# NEW:
return best_indices, best_objective, best_n_iter, best_converged_by
```

---

## 2. Update generate_d_optimal_design() Function

**Location:** `src/core/optimal_design.py`, end of file

**Current function:** `generate_d_optimal_design()`

**Strategy:** 
1. Rename to `generate_optimal_design()`
2. Add `criterion` parameter
3. Add `prediction_grid_config` parameter
4. Create criterion object
5. Pass to optimizer
6. Update result handling

**New signature:**
```python
def generate_optimal_design(
    factors: List[Factor],
    model_type: Literal['linear', 'interaction', 'quadratic'],
    n_runs: int,
    criterion: Literal['D', 'I'] = 'D',  # NEW
    constraints: Optional[List[LinearConstraint]] = None,
    prediction_grid_config: Optional[dict] = None,  # NEW
    candidate_config: Optional[CandidatePoolConfig] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    seed: Optional[int] = None
) -> OptimizationResult:
```

**Key changes:**

```python
# After building model_builder, add:
# Create criterion object
criterion_obj = create_optimality_criterion(
    criterion_type=criterion,
    model_builder=model_builder,
    factors=factors,
    prediction_grid_config=prediction_grid_config
)

# Update optimizer call:
indices, best_objective, n_iter, converged_by = cexch_optimize(
    candidates=candidates,
    n_runs=n_runs,
    model_builder=model_builder,
    criterion=criterion_obj,  # NEW
    config=optimizer_config,
    seed=seed
)

# Update result handling:
if criterion == 'D':
    # D-optimal: objective is log-det
    det_achieved = np.exp(best_objective)
    criterion_type_str = 'D-optimal'
elif criterion == 'I':
    # I-optimal: objective is -I
    i_criterion_value = -best_objective
    criterion_type_str = 'I-optimal'

# Return appropriate result fields based on criterion
```

**Add backward-compatible wrapper:**
```python
def generate_d_optimal_design(*args, **kwargs) -> OptimizationResult:
    """
    DEPRECATED: Use generate_optimal_design(..., criterion='D') instead.
    
    Maintained for backward compatibility.
    """
    warnings.warn(
        "generate_d_optimal_design() is deprecated. "
        "Use generate_optimal_design(..., criterion='D') instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Force D-optimal
    kwargs['criterion'] = 'D'
    return generate_optimal_design(*args, **kwargs)
```

---

## 3. Update OptimizationResult Dataclass

**Location:** `src/core/optimal_design.py`, near end

**Add fields:**
```python
@dataclass
class OptimizationResult:
    # ... existing fields ...
    criterion_type: str  # 'D-optimal' or 'I-optimal'
    final_objective: float
    
    # Optional fields (populated based on criterion)
    i_criterion: Optional[float] = None  # NEW
    i_efficiency_vs_benchmark: Optional[float] = None  # NEW
```

---

## 4. Add I-Efficiency Computation

**Location:** `src/core/optimal_design.py`, helper functions section

**Add new function:**
```python
def compute_i_efficiency(
    i_criterion: float,
    n_runs: int,
    n_params: int,
    benchmark_i_criterion: float
) -> float:
    """
    Compute I-efficiency relative to benchmark design.
    
    I-efficiency = (benchmark_I / actual_I) * 100
    
    Lower I is better (less prediction variance), so efficiency
    is the ratio inverted.
    
    Parameters
    ----------
    i_criterion : float
        I-criterion value for design (average prediction variance)
    n_runs : int
        Number of runs
    n_params : int
        Number of parameters
    benchmark_i_criterion : float
        I-criterion for benchmark design
    
    Returns
    -------
    float
        I-efficiency percentage (100 = matches benchmark)
    
    Examples
    --------
    >>> i_eff = compute_i_efficiency(0.8, 20, 10, 1.0)
    >>> print(f"I-efficiency: {i_eff:.1f}%")
    I-efficiency: 125.0%
    """
    if i_criterion <= 0 or benchmark_i_criterion <= 0:
        return 0.0
    
    # Lower I is better, so efficiency = benchmark/actual
    efficiency = (benchmark_i_criterion / i_criterion) * 100
    
    # Cap at 200% to avoid unrealistic values
    return round(min(efficiency, 200.0), 2)
```

---

## 5. Update compute_benchmark_determinant()

**Location:** `src/core/optimal_design.py`

**Rename to:** `compute_benchmark_criterion()`

**New signature:**
```python
def compute_benchmark_criterion(
    factors: List[Factor],
    model_type: Literal['linear', 'interaction', 'quadratic'],
    model_builder: ModelMatrixBuilder,
    criterion_type: Literal['D', 'I'],
    prediction_grid_config: Optional[dict] = None
) -> Tuple[float, str]:
```

**Add I-optimal case:**
```python
# ... existing D-optimal logic ...

elif criterion_type == 'I':
    # Get benchmark design (same as D-optimal)
    if model_type == 'linear':
        benchmark_design = generate_vertices(k)
        design_name = f"Full Factorial 2^{k}"
    else:
        # Use CCD
        from src.core.response_surface import CentralCompositeDesign
        # ... generate CCD ...
    
    # Build model matrix
    X_model = model_builder(benchmark_design)
    
    # Create I-criterion object
    criterion_obj = create_optimality_criterion(
        'I', model_builder, factors, prediction_grid_config
    )
    
    # Compute I-criterion
    XtX = X_model.T @ X_model
    XtX_inv = np.linalg.inv(XtX + 1e-10 * np.eye(XtX.shape[0]))
    i_value = np.trace(XtX_inv @ criterion_obj.M)
    
    return i_value, design_name
```

---

## Quick Reference: Function Call Flow

```
generate_optimal_design()
  ↓
  create_optimality_criterion(criterion_type, ...)
    ↓
    [if I-optimal] generate_prediction_grid(factors, config)
    ↓
    return DOptimalityCriterion() or IOptimalityCriterion()
  ↓
  cexch_optimize(candidates, n_runs, model_builder, criterion, ...)
    ↓
    criterion.objective(X_model)  # Polymorphic call
    ↓
    [D-optimal] return log-det
    [I-optimal] return -trace((X'X)^(-1) * M)
  ↓
  compute_benchmark_criterion(factors, model_type, criterion_type, ...)
  ↓
  [D-optimal] compute_d_efficiency()
  [I-optimal] compute_i_efficiency()
  ↓
  return OptimizationResult
```

---

## Testing Checklist

After applying changes, test:

1. ✅ D-optimal still works (backward compat)
2. ✅ I-optimal generates different design
3. ✅ Prediction grid correct size
4. ✅ I-criterion computed correctly
5. ✅ Sherman-Morrison reuse works
6. ✅ Efficiency metrics computed
7. ✅ Deprecated wrapper warns

---

## Estimated Lines Changed

- `cexch_optimize()`: ~50 lines modified
- `generate_optimal_design()`: ~80 lines modified (renamed + extended)
- `generate_d_optimal_design()`: +10 lines (new wrapper)
- `compute_i_efficiency()`: +30 lines (new function)
- `compute_benchmark_criterion()`: +40 lines modified
- Docstrings: +100 lines

**Total: ~310 lines changed/added in optimal_design.py**

---

## Next: Phase 3-6

After Phase 2 complete:
- Phase 3: Update augmentation module
- Phase 4: Update goals and UI
- Phase 5: Add tests
- Phase 6: Documentation
