# D-Optimal Design Algorithm Documentation

## Overview

D-optimal designs maximize the determinant of the information matrix (X'X), providing the most precise parameter estimates for a given model and run budget. This module generates D- and I-optimal experimental designs by iteratively swapping runs with points from a candidate pool, with support for linear constraints.

> The implementation is labeled internally as coordinate exchange (CEXCH), but
> it actually operates as **candidate-set row exchange**: each iteration swaps a
> whole run (a full candidate point) into a design row, as described in the
> "Algorithm" section below.

**Key Features:**
- D-optimality and I-optimality (`criterion='D'` or `'I'`)
- Linear constraint support (≤, ≥, = via `LinearConstraint`)
- Candidate pool built from factorial vertices, axial points, the center point, and Latin Hypercube points
- Efficiency benchmarking against Full Factorial 2^k and Face-Centered CCD
- Deterministic with a fixed random seed

---

## Mathematical Foundation

### D-Optimality Criterion

For a design matrix X (n×p), the information matrix is:

```
M = X'X
```

The D-optimality criterion seeks to maximize:

```
Φ_D = det(X'X) = |X'X|
```

**Equivalently (for numerical stability):**
```
Φ_D = log|X'X|
```

**Interpretation:**
- Maximizing |X'X| minimizes the volume of the confidence ellipsoid for parameter estimates
- Provides most precise overall parameter estimation
- Well-suited for prediction and parameter screening

### D-Efficiency

To compare designs of different sizes, we compute D-efficiency relative to a benchmark design:

```
D-efficiency = (|X'X|_achieved / |X'X|_benchmark)^(1/p) × 100%
```

**Benchmark choices** (these are the designs used by the code):
- **Linear models:** Full Factorial 2^k (the gold standard)
- **Interaction and quadratic models:** Face-Centered CCD — `alpha='face'`, `center_points=6` for k ≤ 4 and 5 for k > 4. A face-centered CCD keeps the axial points inside the [-1, 1]^k box (α = 1), which makes it a fair benchmark for designs restricted to the same space. (It is **not** a rotatable CCD.)

**Expected efficiencies:**
- Linear vs Full Factorial: **≥90%** (typically ≈ 100%)
- Interaction/quadratic vs Face-Centered CCD: **≥100%** (D-optimal typically beats CCD; values around 110-120% are typical)

**Note:** Values >100% indicate the design is better than the benchmark. This is expected and desirable for D-optimal designs.

---

## Algorithm: Candidate-Set (Row) Exchange

### Overview

The optimizer iteratively improves a design by swapping individual runs with better candidate points from a feasible candidate pool. Each loop pass tries, for every design row, each candidate point in a per-row subset, keeps the swap that most improves the criterion, and commits it.

**Reference:** Meyer, R. K., & Nachtsheim, C. J. (1995). The coordinate-exchange algorithm for constructing exact optimal experimental designs. *Technometrics*, 37(1), 60-69.

### Algorithm Steps

```
1. Initialize:
   - Generate candidate pool (vertices, axial points, center point, LHS)
   - Filter for feasibility (if constraints present)
   - Randomly select n_runs points as initial design
   - Compute X'X (+ ridge 1e-10 on the diagonal) and (X'X)^(-1)

2. For each iteration:
   For each row i in design:
     - Build the per-row candidate subset:
       * Always include structured (near-boundary) points (max|xᵢ| ≥ 0.99)
       * Sample additional candidates up to max_candidates_per_row (default 50)
     - For each candidate c in the subset:
       - Compute the criterion objective for a trial design with row i → c
       - Track the best improvement
     - If an improvement was found, commit the winning swap
   - Check convergence (see below)

3. Repeat from step 2 with different random starts (default: 3 starts)

4. Return the best design across all starts
```

The number of parameters p is determined by the model type:
- `linear`: p = 1 + k
- `interaction`: p = 1 + k + k(k-1)/2
- `quadratic`: p = 1 + k + k(k-1)/2 + k

### Sherman-Morrison Update

When a row x_old is replaced by x_new, the information matrix changes:

```
X'X_new = X'X_old - x_old·x_old' + x_new·x_new'
```

The **Sherman-Morrison formula** updates (X'X)^(-1) and computes det(X'X_new)/det(X'X_old) in O(p²) rather than a full O(p³) inversion:

```python
# Remove x_old:
v_old = (X'X)^(-1) · x_old
denom_old = 1 - x_old' · v_old
(X'X_1)^(-1) = (X'X)^(-1) + (v_old · v_old') / denom_old

# Add x_new:
v_new = (X'X_1)^(-1) · x_new
denom_new = 1 + x_new' · v_new
(X'X_new)^(-1) = (X'X_1)^(-1) - (v_new · v_new') / denom_new

# Determinant ratio:
det(X'X_new) / det(X'X_old) = denom_old × denom_new
```

**How it is actually used by the code:**
- Candidate *acceptance* still evaluates the full criterion objective (`criterion.objective()`) on each trial design — the determinant ratio is **not** used as the accept/reject predicate.
- The Sherman-Morrison update is applied at **commit time**: the inverse computed along the way is reused for the winning swap instead of re-inverting X'X.
- So the benefit is avoiding a full matrix inversion for every *accepted* swap, not per candidate evaluation. There is no guaranteed ~2× end-to-end speedup.

---

## Constraint Handling

### Supported Constraint Types

**Linear constraints in actual (decoded) factor space:**

```python
# Upper bound: a₁x₁ + a₂x₂ + ... ≤ b
LinearConstraint(
    coefficients={'X1': 1.0, 'X2': 1.0},
    bound=15.0,
    constraint_type='le'  # ≤
)

# Lower bound: x₁ ≥ c
LinearConstraint(
    coefficients={'X1': 1.0},
    bound=3.0,
    constraint_type='ge'  # ≥
)

# Equality: a₁x₁ + a₂x₂ = b
LinearConstraint(
    coefficients={'X1': 1.0, 'X2': 1.0, 'X3': 1.0},
    bound=1.0,
    constraint_type='eq'  # =
)
```

> Note: While equality constraints of the form x1 + x2 + x3 = 1 can be specified,
> full mixture designs (simplex geometry, Scheffé polynomials, mixture model
> parameterization, or mixture-specific candidate sets) are **not** supported.
> The algorithm treats such constraints as standard linear constraints rather
> than a true mixture design, and in practice they can leave too few feasible
> candidates for optimization (see "Warnings and Error Messages").

### Constraint Workflow

```
1. Generate candidate pool in coded space [-1, 1]^k

2. For each candidate point:
   - Decode to actual factor values
   - Check all constraints
   - Keep if feasible, discard if not

3. Multi-layer handling of insufficient feasible candidates:
   - If feasible candidates < n_runs:
     → Warn: "Only N feasible candidates found, need M runs. Attempting augmentation..."
     → Augment via rejection sampling (up to 10000 attempts, target max(2×n_runs, 5×n_runs, 10×p))
     → If STILL < n_runs: raise ValueError
   - Else if feasible candidates < max(5×n_runs, 10×p):
     → Warn: "Low candidate density: N feasible candidates."
     → Proceed with the feasible pool (no augmentation)
   - Else: proceed normally

4. Optimize using only feasible candidates
```

### Augmentation via Rejection Sampling

```python
# Sample uniformly in [-1, 1]^k and accept points that satisfy all constraints.
# Stops when the target size is reached or after max_attempts (default 10000).
augment_constrained_candidates(
    factors=factors,
    existing_candidates=feasible_pool,
    is_feasible=is_feasible,
    target_size=needed_size,
    seed=seed,
)
```

**Acceptance-rate warning:** if the acceptance rate during augmentation is below 1%, a warning is issued:

```
UserWarning: Very low feasible region: 0.12% acceptance rate. Constraints may
be too restrictive. Only found 5 additional feasible points after 10000 attempts.
```

---

## Candidate Pool Generation

### Strategy: Structured Points + Naive LHS

The candidate pool combines structured and random points (no boundary/interior stratification — the optimizer decides):

**1. Structured Points:**
- **Vertices:** all 2^k corners of the [-1, 1]^k hypercube (`include_vertices`, default True)
- **Axial points:** 2k star points at ±α along each axis (`include_axial`, default True; `alpha_axial`, default 1.0 i.e. on the cube faces)
- **Center point:** origin (0, 0, ..., 0) (`include_center`, default True)

**2. Naive LHS:**
- `n_runs × lhs_multiplier` Latin Hypercube points scaled to [-1, 1] (`lhs_multiplier`, default 5)
- **No** stratification by boundary proximity (80/20 split) and **no** dimension-dependent size scaling

**3. Deduplication:**
- Points are rounded to 6 decimals and duplicates removed (`np.unique`)

**Total pool size:** ≈ 2^k + 2k + 1 + n_runs × lhs_multiplier after deduplication
(e.g. k=4, n_runs=20, lhs_multiplier=5 → ≈ 16 + 8 + 1 + 100 = 125).

> The pool defaults (`lhs_multiplier=5`, `alpha_axial=1.0`, structured points
> always on) reflect the current validated implementation defaults and may
> evolve in future releases.

### Why This Strategy?

- **Vertices:** guarantee factorial-like structure (important for linear models)
- **Axial points:** enable CCD-like designs (important for quadratic models)
- **LHS interior fill:** give the optimizer flexible interior coverage
- Constrained designs start from the same pool, decoded and filtered for feasibility

---

## Convergence Criteria

The optimizer stops when any of these conditions is met (all parameters are fields of `OptimizerConfig`):

### 1. Stability (No Improvement)
```
If no improvement for stability_window (default 15) consecutive iterations:
    STOP ("stability")
```

### 2. Relative Improvement Tolerance
```
If relative improvement < relative_improvement_tolerance (default 1e-4)
over the last stability_window iterations:
    rel_improvement = (objective_new - objective_old) / |objective_old|
    STOP ("stability")
```

### 3. Safety Limit
```
If iterations ≥ max_iterations (default 200):
    STOP ("max_iterations")
```

**Multiple starts:** `n_random_starts` (default 3) independent optimizations with different random initializations are run; the best objective across starts is returned. This helps avoid local optima.

---

## Design Quality Metrics

### 1. D-Efficiency vs Benchmark
```
D-efficiency relative to the standard design benchmark:
- Linear models: vs Full Factorial 2^k
- Interaction/quadratic models: vs Face-Centered CCD (alpha='face')

Expected values (warnings are issued when these are missed):
- Linear:    ≥90%   (typical ≈ 100%)
- Quadratic: ≥100%  (typical 110-120%)
```

### 2. Condition Number
```
κ(X'X) = λ_max / λ_min

Interpretation:
- <10:  Excellent (well-conditioned)
- 10-100: Good
- >100:  Warning issued ("High condition number (X.X).")
```

### 3. Determinant
```
|X'X| > 0 required for invertibility

Reported as log|X'X| for numerical stability
```

### 4. Matrix Rank
```
rank(X) = p (full column rank required)

If rank < p: Design is singular (cannot estimate all parameters)
```

---

## Usage Examples

> The primary API is `generate_optimal_design(factors, model_type, n_runs,
> criterion, ...)` where `criterion` is the string `'D'` or `'I'`. Example
> outputs below are from the current implementation defaults and may evolve
> in future releases.

### Example 1: Simple Quadratic Design (D-optimal)

```python
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.optimal.design_generation import generate_optimal_design

# Define factors
factors = [
    Factor("Temperature", FactorType.CONTINUOUS,
           ChangeabilityLevel.EASY, levels=[100, 200]),
    Factor("Pressure", FactorType.CONTINUOUS,
           ChangeabilityLevel.EASY, levels=[10, 50]),
    Factor("Time", FactorType.CONTINUOUS,
           ChangeabilityLevel.EASY, levels=[30, 120])
]

# Generate D-optimal design
result = generate_optimal_design(
    factors=factors,
    model_type='quadratic',
    n_runs=20,
    criterion='D',
    seed=42
)

print(f"Condition number: {result.condition_number:.2f}")
print(f"Converged by: {result.converged_by}")
print(f"Iterations: {result.n_iterations}")
print(f"D-efficiency: {result.d_efficiency_vs_benchmark}%")
print(f"Benchmark: {result.benchmark_design_name}")
print(f"\n{result.design_actual.head()}")
```

**Expected output** (with `seed=42`; the exact values depend on the current
implementation defaults and may shift in future releases):

```
Condition number: 27.34
Converged by: stability
Iterations: 15
D-efficiency: 115.15%
Benchmark: Face-Centered CCD (k=3)

   StdOrder  RunOrder  Temperature  Pressure   Time
0         1         1        100.0      50.0   30.0
1         2         2        100.0      50.0  120.0
2         3         3        100.0      10.0   30.0
3         4         4        200.0      10.0  120.0
...
```

### Example 2: Constrained Design

```python
from src.core.optimal.design_generation import generate_optimal_design
from src.core.optimal.constraints import LinearConstraint

# Chemical process with a safety/temperature-pressure bound
factors = [
    Factor("Temperature", FactorType.CONTINUOUS,
           ChangeabilityLevel.EASY, levels=[150, 250]),
    Factor("Pressure", FactorType.CONTINUOUS,
           ChangeabilityLevel.EASY, levels=[10, 50]),
    Factor("Catalyst", FactorType.CONTINUOUS,
           ChangeabilityLevel.EASY, levels=[0, 5])
]

# High temperature requires low pressure (safety): Temp + 2·Pressure ≤ 350
constraint = LinearConstraint(
    coefficients={'Temperature': 1.0, 'Pressure': 2.0},
    bound=350.0,
    constraint_type='le'
)

result = generate_optimal_design(
    factors=factors,
    model_type='interaction',
    n_runs=15,
    criterion='D',
    constraints=[constraint],
    seed=42
)

# Verify constraint satisfaction
combo = (result.design_actual['Temperature']
         + 2 * result.design_actual['Pressure'])
assert (combo <= 350.0 + 1e-9).all()
print(f"D-efficiency: {result.d_efficiency_vs_benchmark}%")
print(f"Condition number: {result.condition_number:.2f}")
```

For this example the current output is a D-efficiency of ≈117% and a condition
number ≈2.0, with no warnings (the constraint does not degrade the design).

### Example 3: I-Optimal Design

Same API with `criterion='I'`; the design minimizes the average prediction
variance across a prediction grid instead of maximizing |X'X|:

```python
result_i = generate_optimal_design(
    factors=factors,
    model_type='quadratic',
    n_runs=20,
    criterion='I',
    prediction_grid_config={'n_points_per_dim': 7},
    seed=42
)

print(f"Criterion: {result_i.criterion_type}")       # I-optimal
print(f"I-criterion: {result_i.i_criterion:.3f}")
print(f"I-efficiency: {result_i.i_efficiency_vs_benchmark}%")
print(f"D-efficiency (also reported): {result_i.d_efficiency_vs_benchmark}%")
print(f"Benchmark: {result_i.benchmark_design_name}")
```

`generate_d_optimal_design(...)` is a convenience wrapper that calls
`generate_optimal_design(..., criterion='D')`.

---

## Configuration Options

Configuration dataclasses in `src/core/optimal/`:

**`CandidatePoolConfig`** (`src/core/optimal/candidates.py`):
- `lhs_multiplier=5` — generate n_runs × lhs_multiplier LHS points
- `include_vertices=True`, `include_axial=True`, `include_center=True`
- `alpha_axial=1.0` — distance of axial points (1.0 = cube faces)

**`OptimizerConfig`** (`src/core/optimal/optimizer.py`):
- `max_iterations=200`, `relative_improvement_tolerance=1e-4`
- `stability_window=15`, `n_random_starts=3`
- `max_candidates_per_row=50`, `use_sherman_morrison=True`

`candidate_config=` and `optimizer_config=` are passed to
`generate_optimal_design`. Defaults reflect the current validated
implementation and may evolve in future releases.

---

## Warnings and Error Messages

### Error: Insufficient Feasible Candidates

```
ValueError: Even after augmentation, only 8 feasible candidates found (need 12 runs).
```

**Cause:** Constraints eliminate too many candidates, even after rejection
sampling. This typically happens with very tight or equality constraints
(e.g. a mixture-style `x1 + x2 + x3 = 1` on a discrete candidate pool).

**Solutions:**
1. Relax the constraints (widen bounds, remove unnecessary constraints)
2. Reduce n_runs
3. Verify that the constraints are not contradictory
4. Check that a feasible region actually exists

### Warning: Low Candidate Density

```
UserWarning: Low candidate density: 18 feasible candidates.
```

**Cause:** Enough feasible candidates for the runs, but below
`max(5×n_runs, 10×p)` that the optimizer prefers for good swaps. The code
warns and proceeds with the feasible pool (it does not augment in this branch).

**Impact:** Design quality may be suboptimal (lower efficiency, higher condition number).

**Solutions:**
1. Usually no action needed
2. For better quality, relax constraints or increase the pool (`lhs_multiplier`)
3. Monitor D-efficiency and condition number in the results

### Warning: Low D-Efficiency

```
# Linear model, <90%:
UserWarning: D-efficiency: 72.40%. Expected >90% for linear models.

# Interaction/quadratic model, <100%:
UserWarning: D-efficiency: 85.78%. Expected ≥100% for quadratic.
```

**Cause:** Design quality below the expected threshold for the model type.

**Likely reasons:**
- Constraints are too restrictive
- Insufficient runs for model complexity
- Infeasible region poorly sampled

### Warning: High Condition Number

```
UserWarning: High condition number (194.3).
```

**Cause:** X'X is nearly singular (some combinations of parameters are hard to
estimate independently). Issued whenever the condition number exceeds 100.

**Impact:**
- Parameter estimates will have large standard errors
- Predictions may be unreliable
- Numerical instability in ANOVA

**Solutions:**
1. Add more runs
2. Remove highly correlated factors
3. Simplify the model (use 'linear' or 'interaction' instead of 'quadratic')
4. Check whether constraints force correlations between factors

---

## Limitations and Future Enhancements

### Current Limitations

1. **Continuous factors only**
   - Categorical and discrete-numeric factors are not supported
   - Workaround: create separate designs for each categorical level

2. **Linear constraints only**
   - No disallowed combinations (e.g., "If Material=A, then Temp<180")
   - No nonlinear constraints (e.g., x₁² + x₂² ≤ 1)

3. **D-optimality and I-optimality are both supported** (`criterion='D' | 'I'`)
   - A-optimal (minimize trace) not available
   - G-optimal (minimize maximum prediction variance) not available

4. **Mixture designs not supported**
   - Equality constraints like x1 + x2 + x3 = 1 can be specified, but the
     algorithm does not implement mixture-specific model structures (Scheffé
     models), simplex candidate sets, or mixture geometry, and in practice such
     constraints often leave too few feasible candidates

### Planned Enhancements (Post-MVP)

1. **Categorical factor support**
   - Dummy variable encoding
   - Mixed continuous/categorical designs

2. **Disallowed combinations**
   - Logic constraints (if-then rules)
   - Indicator variable approach

3. **Additional optimality criteria**
   - A-optimal (minimize trace, better for parameter estimation)
   - G-optimal (minimize maximum prediction variance, better for prediction)

---

## Validation Against Known Designs

The generated designs are validated against standard references:

**Full Factorial (Linear):**
```
2^k design benchmark: D-optimal linear designs achieve ≈100% of the
full factorial determinant (observed 100.0% for 2^3 with n_runs=8).
```

**Face-Centered CCD (Quadratic):**
```
k=3, 20 runs, 10 parameters: D-optimal designs consistently exceed the
Face-Centered CCD benchmark — observed ≈115% with the default configuration.
```

---

## References

### Core Algorithm

1. **Meyer, R. K., & Nachtsheim, C. J. (1995).** The coordinate-exchange algorithm for constructing exact optimal experimental designs. *Technometrics*, 37(1), 60-69.
   - CEXCH algorithm foundation
   - Convergence properties
   - Comparison to other algorithms

2. **Atkinson, A. C., Donev, A. N., & Tobias, R. D. (2007).** Optimum experimental designs, with SAS. Oxford University Press.
   - D-optimality theory
   - Information matrix
   - Design efficiency

### Sherman-Morrison Formula

3. **Golub, G. H., & Van Loan, C. F. (2013).** Matrix computations (4th ed.). Johns Hopkins University Press.
   - Sherman-Morrison-Woodbury formula
   - Numerical linear algebra
   - Matrix determinant updates

### Constrained Designs

4. **Myers, R. H., Montgomery, D. C., & Anderson-Cook, C. M. (2016).** Response surface methodology: Process and product optimization using designed experiments (4th ed.). Wiley.
   - Constrained optimization
   - Mixture designs
   - Practical applications

5. **Jones, B., & Nachtsheim, C. J. (2011).** Efficient designs with minimal aliasing. *Technometrics*, 53(1), 62-71.
   - Candidate set reduction
   - Constraint handling
   - Computational efficiency

---

## Appendix: Model Matrix Construction

### Linear Model (k factors)

```
Model: y = β₀ + β₁x₁ + β₂x₂ + ... + βₖxₖ + ε

X = [1  x₁  x₂  ...  xₖ]

Parameters: p = 1 + k
```

### Interaction Model (k factors)

```
Model: y = β₀ + Σβᵢxᵢ + ΣΣβᵢⱼxᵢxⱼ + ε

X = [1  x₁  x₂  ...  xₖ  x₁x₂  x₁x₃  ...  xₖ₋₁xₖ]

Parameters: p = 1 + k + k(k-1)/2
```

### Quadratic Model (k factors)

```
Model: y = β₀ + Σβᵢxᵢ + ΣΣβᵢⱼxᵢxⱼ + Σβᵢᵢxᵢ² + ε

X = [1  x₁  x₂  ...  xₖ  x₁x₂  ...  xₖ₋₁xₖ  x₁²  x₂²  ...  xₖ²]

Parameters: p = 1 + k + k(k-1)/2 + k
```

**Column ordering:**
1. Intercept (all 1s)
2. Main effects (k columns)
3. Two-way interactions (k(k-1)/2 columns)
4. Pure quadratic terms (k columns)

**Coded levels:** All factors coded to [-1, +1] for numerical stability and interpretability.