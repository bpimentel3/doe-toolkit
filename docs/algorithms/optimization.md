# Response Optimization Algorithms

## Overview

The optimization module provides methods for finding optimal factor settings based on fitted response surface models. It supports both single-response optimization (maximize, minimize, target) and multi-response optimization using desirability functions.

**Key Features:**
- Single-response optimization with three objectives
- Multi-response desirability functions (Derringer & Suich method)
- Optimization in **actual (natural) factor space** (results are directly interpretable)
- Categorical and discrete-numeric factor support
- Linear constraint support (single-response, numeric factors only)
- Confidence and prediction intervals

> Solver parameters shown below (`maxiter=500`, `popsize=20`, `ftol=1e-9`,
> etc.) reflect the current validated implementation defaults and may evolve
> in future releases.

---

## Single-Response Optimization

### Problem Formulation

Given a fitted model $\hat{y}(x)$ where $x = (x_1, x_2, ..., x_k)$ are factor settings, find optimal settings that:

**Maximize:** $\max_{x} \hat{y}(x)$ subject to constraints

**Minimize:** $\min_{x} \hat{y}(x)$ subject to constraints

**Target:** $\min_{x} (\hat{y}(x) - T)^2$ subject to constraints

where $T$ is the target value.

### Constraints

**Factor Bounds:**
$$x_i^{min} \leq x_i \leq x_i^{max} \quad \forall i$$

The bounds are the factor's declared **actual-space** range: `[min_value, max_value]`
for continuous and discrete-numeric factors, and the index range of the declared
levels for categorical factors (later converted back to level labels).

**Linear Constraints:**
$$\sum_{i} a_i x_i \leq b \quad \text{(inequality, 'le')}$$
$$\sum_{i} a_i x_i \geq b \quad \text{(inequality, 'ge')}$$
$$\sum_{i} a_i x_i = b \quad \text{(equality, 'eq')}$$

Linear constraints **are honored for the pure-numeric single-response path**
(they are converted to SciPy constraint objects for SLSQP). They are **ignored
with a warning** when the design contains categorical factors or when optimizing
desirability (see below).

### Optimization Algorithm

The solver is chosen by the solver by design composition and objective:

1. **Pure-numeric single response:** `scipy.optimize.minimize(method='SLSQP')`
   - Starting point: center of the search space, with an optional random
     perturbation when a `seed` is supplied (uniform in [-0.1, 0.1] of each
     range, clipped to the bounds)
   - Options: `maxiter=500`, `ftol=1e-9`
   - Bounds and linear constraints are passed to SLSQP directly

2. **Categorical factors present (single response):**
   `scipy.optimize.differential_evolution` with `integrality` on the
   categorical index dimensions (`maxiter=500`, `popsize=20`, `polish=False`,
   `seed=seed`). Linear constraints are ignored with a warning. If the
   stochastic search fails to converge, the optimizer falls back to a full
   enumeration of the categorical level combinations
   (`_enumerate_categorical_best`).

3. **Multi-response desirability:** always
   `scipy.optimize.differential_evolution` (`maxiter=500`, `popsize=20`,
   `polish=False`, `seed=seed`). Desirability is optimized globally because
   the hard [0, 1] desirability bounds create a discontinuous objective that
   gradient-based methods cannot navigate (the numerical gradient vanishes in
   the flat d = 0 region).

**Why SLSQP for numeric designs?**
- Handles both bounds and linear constraints natively
- Fast convergence for smooth response surfaces
- Gradient-based (efficient for quadratic models)

**Why differential_evolution for categorical / desirability?**
- No gradient information needed
- `integrality` handles categorical level indices directly
- Global exploration is more robust for discontinuous objectives
- Note: DE is **stochastic** — it explores globally but does not strictly
  guarantee the global optimum; reruns with different seeds can be used to
  confirm a stable solution

### Prediction Uncertainty

After finding optimal settings $x^*$, compute:

**Point Prediction:**
$$\hat{y}(x^*) = E[Y | x^*]$$

**Confidence Interval (95% CI):**
Uncertainty in the **mean response** at $x^*$:
$$\hat{y}(x^*) \pm t_{\alpha/2, df} \cdot SE(\hat{y}(x^*))$$

where $SE(\hat{y}(x^*)) = \sqrt{MSE \cdot x^{*T}(X^TX)^{-1}x^*}$

**Prediction Interval (95% PI):**
Uncertainty for a **single future observation** at $x^*$:
$$\hat{y}(x^*) \pm t_{\alpha/2, df} \cdot SE_{pred}(x^*)$$

where $SE_{pred}(x^*) = \sqrt{MSE \cdot (1 + x^{*T}(X^TX)^{-1}x^*)}$

**Key Difference:**
- CI: "Where will the process average be?" (narrower)
- PI: "Where will my next observation be?" (wider, includes random error)

---

## Multi-Response Optimization with Desirability Functions

### The Desirability Approach

When optimizing multiple responses simultaneously, desirability functions transform each response to a common scale $[0, 1]$:
- $d = 0$: Completely unacceptable
- $d = 1$: Ideal (target achieved)

### Individual Desirability Functions

**Maximize Response:**

For response $y$ where larger is better:

$$d_i(y) = \begin{cases}
0 & y < L \\
\left(\frac{y - L}{U - L}\right)^r & L \leq y \leq U \\
1 & y > U
\end{cases}$$

- $L$: Lower acceptable bound (minimum)
- $U$: Upper target (ideal)
- $r$: Weight (shape parameter)
  - $r = 1$: Linear increase
  - $r > 1$: More emphasis on reaching target
  - $r < 1$: More tolerant of lower values

**Minimize Response:**

For response $y$ where smaller is better:

$$d_i(y) = \begin{cases}
1 & y < L \\
\left(\frac{U - y}{U - L}\right)^r & L \leq y \leq U \\
0 & y > U
\end{cases}$$

- $L$: Lower target (ideal)
- $U$: Upper acceptable bound (maximum)

**Target Response:**

For response $y$ with a specific target $T$:

$$d_i(y) = \begin{cases}
0 & y < L \text{ or } y > U \\
\left(\frac{y - L}{T - L}\right)^{r_1} & L \leq y < T \\
\left(\frac{U - y}{U - T}\right)^{r_2} & T \leq y \leq U
\end{cases}$$

- $L$: Lower acceptable bound
- $T$: Target value (ideal)
- $U$: Upper acceptable bound
- $r_1$, $r_2$: Separate weights below/above target

### Overall Desirability

The overall desirability combines individual desirabilities using **geometric mean** with importance weighting:

$$D = \left(\prod_{i=1}^{n} d_i^{w_i}\right)^{1/\sum w_i}$$

where:
- $d_i$: Individual desirability for response $i$
- $w_i$: Importance weight for response $i$
- $n$: Number of responses

**Properties:**
- If any $d_i = 0$, then $D = 0$ (strict constraint)
- Geometric mean prevents compensation (can't trade off one bad response with many good ones)
- Importance weights shift optimum toward more important responses

### Multi-Response Optimization Algorithm

1. **Configure Desirability:**
   - For each response, specify objective (maximize/minimize/target)
   - Set bounds and weights
   - Assign importance weights

2. **Objective Function:**
   $$\max_{x} D(x) = \max_{x} \left(\prod_{i=1}^{n} d_i(\hat{y}_i(x))^{w_i}\right)^{1/\sum w_i}$$

3. **Optimization:**
   - Use `differential_evolution` to maximize overall desirability
   - Subject to factor bounds (in actual space); linear constraints are
     **ignored with a warning** on this path

4. **Result:**
   - Optimal settings $x^*$
   - Individual desirabilities $d_i(x^*)$
   - Overall desirability $D(x^*)$
   - Predicted responses $\hat{y}_i(x^*)$

---

## Mathematical Foundations

### Quadratic Response Surface Model

Most response surface optimizations use quadratic models:

$$y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i=1}^{k} \beta_{ii} x_i^2 + \sum_{i<j} \beta_{ij} x_i x_j + \varepsilon$$

**Canonical Form** *(theoretical background — not implemented in this app):*

Transform to eliminate cross-product terms:

$$y = y_s + \sum_{i=1}^{k} \lambda_i w_i^2$$

where:
- $y_s$: Stationary point response
- $\lambda_i$: Eigenvalues (curvature along principal axes)
- $w_i$: Canonical variables (rotated coordinates)

**Optimum Classification** *(theoretical background — not implemented in this app):**
- All $\lambda_i < 0$: Maximum (bowl-shaped down)
- All $\lambda_i > 0$: Minimum (bowl-shaped up)
- Mixed signs: Saddle point (minimax)

The optimization module does not compute canonical ridge analysis, stationary
points via eigenvalues, or the classification above. It directly searches for
the optimum numerically over the factor ranges. (Canonical analysis is standard
response-surface methodology — see the references — but is out of scope for the
shipped optimizer, which relies on the numerical methods described in the
"Algorithm" sections.)

### Analytical Optimum (Unconstrained)

For quadratic model without interactions:

$$y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i=1}^{k} \beta_{ii} x_i^2$$

Stationary point found by setting partial derivatives to zero:

$$\frac{\partial y}{\partial x_i} = \beta_i + 2\beta_{ii} x_i = 0$$

$$x_i^* = -\frac{\beta_i}{2\beta_{ii}}$$

*(Used as an analytical check in the test suite; the optimizer itself is numeric.)*

### Constrained Optimization

SLSQP solves constrained problems via a sequential quadratic programming
approach with Lagrange multipliers and Karush-Kuhn-Tucker (KKT) optimality
conditions. This is an internal property of the SLSQP algorithm; the module
does not expose KKT/canonical analysis as a user-facing feature.

---

## Algorithm Details

### SLSQP Method (numeric single-response path)

**Sequential Least Squares Programming:**

At each iteration $k$:

1. **Quadratic Programming Subproblem:**
   $$\min_d \nabla f(x_k)^T d + \frac{1}{2} d^T H_k d$$
   subject to linearized constraints:
   $$\nabla g_j(x_k)^T d + g_j(x_k) \leq 0$$

2. **Update:**
   $$x_{k+1} = x_k + \alpha_k d_k$$
   where $\alpha_k$ is step size from line search

3. **Hessian Approximation:**
   Update $H_k$ using BFGS formula

**Convergence options (as configured):**
- $|\nabla f(x_k)| < \text{ftol}$ (gradient small), `ftol=1e-9`
- $|f(x_{k+1}) - f(x_k)| < \text{ftol}$ (objective change small)
- Maximum iterations (`maxiter=500`)

### Differential Evolution (categorical / desirability paths)

**Global Optimization Strategy:**

1. **Population Initialization:**
   Generate $N_p$ random candidate solutions in feasible region

2. **Mutation:**
   For each candidate $x_i$, create mutant:
   $$v_i = x_{r1} + F \cdot (x_{r2} - x_{r3})$$
   where $r1, r2, r3$ are random distinct indices, $F$ is scaling factor

3. **Crossover:**
   Create trial vector by mixing mutant with target

4. **Selection:**
   Keep trial if better than target:
   $$x_i^{new} = \begin{cases} u_i & f(u_i) < f(x_i) \\ x_i & \text{otherwise} \end{cases}$$

5. **Iteration:**
   Repeat until convergence or max generations

**Why Use Differential Evolution Here?**
- Robust for non-convex, multimodal surfaces
- Doesn't require gradient information
- `integrality` maps cleanly onto categorical level indices
- Handles the discontinuous desirability objective that gradients cannot

**Configuration (current defaults):** `maxiter=500`, `popsize=20`,
`polish=False`, `seed=seed` (reproducible when a seed is given; unseeded
otherwise).

---

## Implementation Notes

### Optimization Space and Scaling

The optimizer works in **actual (natural) factor space** so results are
directly interpretable:

- Continuous factors search over `[min_value, max_value]`
- Discrete-numeric factors search the same range and then **snap** to the
  nearest declared level (`_nearest_level`)
- Categorical factors search an integer index `[0, len(levels) - 1]` and map
  back to level labels

The fitted model, however, may have been trained on **coded** values (e.g.
CCD or fractional-factorial designs, which store coded columns). For those
models pass `model_is_coded=True` to `optimize_response` /
`optimize_desirability`; the optimizer then encodes its actual-space
candidates back to coded space before calling `model.predict`.

### Categorical Defaults and Pinned Levels

- `pinned_levels={factor_name: level}` holds any categorical factor fixed at a
  specified level while the remaining factors are optimized.
- Without explicit pins, categorical factors default to their **first declared
  level** (`_pinned_categorical_defaults`).

### Nuisance Columns

Fitted models from blocked or split-plot designs append nuisance terms such as
`Block`, `WholePlot`, or `VeryHardPlot` to their formula. The optimizer pins
these columns to their reference (first distinct) level in the prediction
frame so their influence is ignored — the block/whole-plot effect is treated
as a nuisance and does not drive the search.

### Ridge Regularization

Ridge regularization `(X'X + εI)^{-1}` with ε = 10⁻¹⁰ is **not** part of this
module. It lives in the **optimal-design generation** machinery
(`src/core/optimal/criteria.py`, `src/core/optimal/optimizer.py`) and in
`src/core/diagnostics/estimability.py`, where it protects determinant
computations for near-singular information matrices. The response optimizer
itself does not perform ridge regularization.

### Convergence Diagnostics

- SLSQP path: report `result.success`, the achieved objective, and the
  iteration count. The `ftol=1e-9` gradient/objective-change threshold applies
  to this path only.
- DE paths: report `result.success` plus the same objective/iteration info; the
  gradient-based first-order condition does **not** apply to the DE/desirability
  paths.

**Typical failure signals and remedies:**
- SLSQP "maximum iterations": increase `maxiter`, or reconsider the model/bounds
- SLSQP "singular matrix": check for factor aliasing or near-collinear terms
- DE fallback enumeration triggered for categorical designs: confirm factor
  ranges and pinned levels so the search region is not degenerate

---

## Validation and Testing

### Test Strategy

1. **Known Analytical Optima:**
   - Simple quadratic functions with closed-form solutions
   - Verify numerical optimizer finds analytical optimum

2. **Constraint Satisfaction:**
   - Test cases verify constraints are satisfied at the optimum
   - Tolerance: $10^{-6}$

3. **Multi-Response Trade-offs:**
   - Conflicting objectives test proper compromise
   - Importance weights shift optimum as expected

4. **Edge Cases:**
   - Linear models (optimum at boundary)
   - Flat regions (multiple optima)
   - Infeasible constraints (proper error handling)

### Example Test Case

**Function:** $y = 10 - x_1^2 - x_2^2$

**Analytical Optimum:**
- $x_1^* = 0$, $x_2^* = 0$
- $y^* = 10$

**Test Procedure:**
1. Generate CCD data from function with noise
2. Fit quadratic model
3. Optimize using `optimize_response(..., objective='maximize')`
4. Verify: $|x_i^* - 0| < 0.2$ and $|y^* - 10| < 0.5$

The module has a dedicated test suite (`tests/test_optimization.py`) covering
these scenarios, including categorical, discrete-numeric, pinned-level, and
desirability cases.

---

## References

### Primary Literature

1. **Myers, R. H., Montgomery, D. C., & Anderson-Cook, C. M. (2016).**
   *Response Surface Methodology: Process and Product Optimization Using Designed Experiments*, 4th Edition. Wiley.
   - Chapter 5: Fitting Response Surfaces
   - Chapter 6: Optimization of Response Surfaces

2. **Derringer, G., & Suich, R. (1980).**
   Simultaneous Optimization of Several Response Variables.
   *Journal of Quality Technology*, 12(4), 214-219.
   - Original desirability function formulation
   - Geometric mean for overall desirability

3. **Box, G. E. P., & Draper, N. R. (2007).**
   *Response Surfaces, Mixtures, and Ridge Analyses*, 2nd Edition. Wiley.
   - Chapter 10: Multiple Response Surface Optimization
   - Canonical and ridge analysis (background for this document)

### Optimization Algorithms

4. **Kraft, D. (1988).**
   A Software Package for Sequential Quadratic Programming.
   *Forschungsbericht DFVLR-FB 88-28*, DLR German Aerospace Center.
   - SLSQP algorithm description

5. **Storn, R., & Price, K. (1997).**
   Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces.
   *Journal of Global Optimization*, 11, 341-359.
   - Differential evolution algorithm

### Statistical Inference

6. **Draper, N. R., & Smith, H. (1998).**
   *Applied Regression Analysis*, 3rd Edition. Wiley.
   - Chapter 2: Prediction and confidence intervals
   - Chapter 3: Multiple regression

---

## Practical Guidelines

### When to Use Each Objective

**Maximize/Minimize:**
- Clear direction of improvement
- No specific target value
- Examples: maximize yield, minimize cost

**Target:**
- Specific target value required
- Deviations in either direction are bad
- Examples: target pH = 7.0, target thickness = 2.5mm

### Choosing Desirability Parameters

**Bounds (L, U):**
- $L$: Minimum acceptable performance
- $U$: Best achievable performance (or target)
- Based on process capability or specifications

**Weights ($r$):**
- $r = 1$: Linear desirability (default)
- $r > 1$: Emphasize reaching target (be more demanding)
- $r < 1$: More tolerant of deviations

**Importance ($w_i$):**
- Equal importance: $w_i = 1$ for all responses
- Quality critical: Higher weight for critical responses
- Cost vs. Performance: Weight reflects relative value

### Interpreting Results

**Overall Desirability:**
- $D > 0.8$: Excellent compromise
- $0.6 < D < 0.8$: Acceptable
- $D < 0.6$: Poor (reconsider specifications or add experiments)

**Individual Desirabilities:**
- Identify which responses are limiting overall performance
- Consider relaxing specifications for limiting responses

**Prediction Intervals:**
- Use PI width to assess process variability
- Wide PI suggests need for variance reduction

---

## Common Issues and Solutions

### Issue: Optimizer finds boundary optimum

**Symptom:** Optimal settings at factor limits ($x_i^* = \min$ or $\max$)

**Diagnosis:**
- Response increasing/decreasing monotonically over the range
- No usable curvature (linear model sufficient)

**Solutions:**
1. Expand the design space (move factor bounds)
2. Add axial/center points to the experiment to estimate curvature
3. Accept the boundary optimum if further expansion is infeasible

### Issue: Multiple local optima suspected

**Symptom:** Different starting points yield different optima for a numeric-only
design (SLSQP is a local optimizer)

**Solutions:**
1. Rerun with different `seed` values and compare (the seed perturbs the
   SLSQP starting point)
2. For categorical or desirability problems, rely on the built-in global
   `differential_evolution` path and confirm stability across seeds
3. Visualize the response surface (contour plots)
4. Narrow the search bounds around the region of interest

### Issue: Conflicting responses with no good compromise

**Symptom:** $D < 0.5$ even with relaxed specifications

**Solutions:**
1. Reassess specifications (are they realistic?)
2. Check for interactions between responses
3. Consider sequential optimization (optimize the primary response first)
4. Add experiments in underexplored regions

### Issue: Prediction intervals very wide

**Symptom:** PI width > 50% of predicted response

**Solutions:**
1. Add replicates to reduce pure error
2. Add center points to estimate curvature better
3. Check for outliers or unusual runs
4. Consider transforming the response (variance stabilization)

---

## Future Enhancements

### Planned Features

1. **Robust Parameter Design:**
   - Minimize variance while optimizing mean
   - Transmit variation functions

2. **Pareto Front Exploration:**
   - Multi-objective optimization without desirability
   - Trade-off curves for conflicting objectives

3. **Bayesian Optimization:**
   - Uncertainty-aware optimization
   - Sequential design for expensive experiments

4. **Mixture Design Optimization:**
   - Simplex constraints ($\sum x_i = 1$)
   - Specialized desirability for mixtures