# Response Optimization Algorithms

## Overview

The optimization module provides methods for finding optimal factor settings based on fitted response surface models. It supports both single-response optimization (maximize, minimize, target) and multi-response optimization using desirability functions.

**Key Features:**
- Single-response optimization with three objectives
- Multi-response desirability functions (Derringer & Suich method)
- Linear constraint support
- Confidence and prediction intervals
- Robust optimization with fallback strategies

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

**Linear Constraints:**
$$\sum_{i} a_i x_i \leq b \quad \text{(inequality)}$$
$$\sum_{i} a_i x_i = b \quad \text{(equality)}$$

### Optimization Algorithm

The module uses **Sequential Least Squares Programming (SLSQP)** from `scipy.optimize`:

1. **Initialization:** Start from center of design space with optional random perturbation
2. **Objective Function:** Construct objective based on model predictions
3. **Optimization:** Use SLSQP with bounds and linear constraints
4. **Fallback:** If SLSQP fails, retry with `differential_evolution` (global optimizer)

**Why SLSQP?**
- Handles both bounds and linear constraints natively
- Fast convergence for smooth response surfaces
- Gradient-based (efficient for quadratic models)

**Fallback Strategy:**
If SLSQP fails to converge (e.g., non-convex surface, multiple local optima):
- Switch to `differential_evolution` (genetic algorithm)
- Slower but more robust for complex surfaces
- Guarantees global search

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
   - Use SLSQP to maximize overall desirability
   - Subject to factor bounds and linear constraints

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

**Canonical Form:**
Transform to eliminate cross-product terms:

$$y = y_s + \sum_{i=1}^{k} \lambda_i w_i^2$$

where:
- $y_s$: Stationary point response
- $\lambda_i$: Eigenvalues (curvature along principal axes)
- $w_i$: Canonical variables (rotated coordinates)

**Optimum Classification:**
- All $\lambda_i < 0$: Maximum (bowl-shaped down)
- All $\lambda_i > 0$: Minimum (bowl-shaped up)
- Mixed signs: Saddle point (minimax)

### Analytical Optimum (Unconstrained)

For quadratic model without interactions:

$$y = \beta_0 + \sum_{i=1}^{k} \beta_i x_i + \sum_{i=1}^{k} \beta_{ii} x_i^2$$

Stationary point found by setting partial derivatives to zero:

$$\frac{\partial y}{\partial x_i} = \beta_i + 2\beta_{ii} x_i = 0$$

$$x_i^* = -\frac{\beta_i}{2\beta_{ii}}$$

**With Interactions:**

For general quadratic model, stationary point:

$$\mathbf{x}^* = -\frac{1}{2} \mathbf{B}^{-1} \mathbf{b}$$

where:
- $\mathbf{b}$: Vector of linear coefficients
- $\mathbf{B}$: Matrix of quadratic/interaction coefficients

### Constrained Optimization

When constraints are active at optimum, use Lagrange multipliers:

$$\mathcal{L}(x, \lambda) = f(x) - \sum_{j} \lambda_j g_j(x)$$

where $g_j(x) \leq 0$ are constraint functions.

**Karush-Kuhn-Tucker (KKT) Conditions:**
1. Stationarity: $\nabla f(x^*) = \sum \lambda_j \nabla g_j(x^*)$
2. Primal feasibility: $g_j(x^*) \leq 0$
3. Dual feasibility: $\lambda_j \geq 0$
4. Complementary slackness: $\lambda_j g_j(x^*) = 0$

SLSQP solves these conditions numerically.

---

## Algorithm Details

### SLSQP Method

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

**Convergence Criteria:**
- $|\nabla f(x_k)| < \text{ftol}$ (gradient small)
- $|f(x_{k+1}) - f(x_k)| < \text{ftol}$ (objective change small)
- Maximum iterations reached

### Differential Evolution (Fallback)

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

**Why Use as Fallback?**
- Robust for non-convex, multimodal surfaces
- Doesn't require gradient information
- Explores globally before converging

---

## Implementation Notes

### Numerical Stability

**Scaling:**
All optimization operates on coded factor values $[-1, 1]$:
$$x_{coded} = \frac{x_{actual} - x_{center}}{x_{range}/2}$$

**Why?**
- Improved numerical conditioning
- Equal sensitivity across factors
- Prevents ill-conditioning from different scales

**Ridge Regularization:**
For near-singular information matrices:
$$(X^TX + \epsilon I)^{-1}$$
where $\epsilon = 10^{-10}$

### Convergence Diagnostics

**Success Indicators:**
- `result.success == True`
- Constraint satisfaction: $|g_j(x^*)| < 10^{-6}$
- First-order conditions: $|\nabla f(x^*)| < 10^{-9}$

**Failure Modes:**
- "Maximum iterations": Increase `maxiter`
- "Singular matrix": Check for factor aliasing
- "Infeasible constraints": Relax constraint bounds

### Computational Complexity

**Per Iteration:**
- Objective evaluation: $O(p)$ where $p$ = number of model parameters
- Gradient evaluation: $O(kp)$ for $k$ factors
- Hessian update (BFGS): $O(k^2)$
- QP subproblem: $O(k^3)$

**Total:** $O(n_{iter} \cdot k^3)$ for SLSQP

**Differential Evolution:** $O(N_p \cdot n_{gen} \cdot p)$ where $N_p$ = population size

---

## Validation and Testing

### Test Strategy

1. **Known Analytical Optima:**
   - Simple quadratic functions with closed-form solutions
   - Verify numerical optimizer finds analytical optimum

2. **Constraint Satisfaction:**
   - All test cases verify constraints satisfied at optimum
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

**Symptom:** Optimal settings at factor limits ($x_i^* = \pm 1$)

**Diagnosis:**
- Response increasing/decreasing monotonically
- No curvature detected (linear model sufficient)

**Solutions:**
1. Expand design space (move factor bounds)
2. Add axial points for curvature estimation
3. Accept boundary optimum if further expansion infeasible

### Issue: Multiple local optima suspected

**Symptom:** Different starting points yield different optima

**Solutions:**
1. Use `differential_evolution` (global optimizer)
2. Run optimization from multiple random starts
3. Visualize response surface (contour plots)
4. Check for ridge systems or saddle points

### Issue: Conflicting responses with no good compromise

**Symptom:** $D < 0.5$ even with relaxed specifications

**Solutions:**
1. Reassess specifications (are they realistic?)
2. Check for interactions between responses
3. Consider sequential optimization (optimize primary first)
4. Add experiments in underexplored regions

### Issue: Prediction intervals very wide

**Symptom:** PI width > 50% of predicted response

**Solutions:**
1. Add replicates to reduce pure error
2. Add center points to estimate curvature better
3. Check for outliers or unusual runs
4. Consider transforming response (variance stabilization)

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

### Implementation Complexity

- **Robust Design:** Medium (requires variance modeling)
- **Pareto Fronts:** High (requires multi-objective algorithms)
- **Bayesian:** Very High (requires Gaussian process models)
- **Mixtures:** Low (constraint modification only)

---

**Last Updated:** Session 10 complete
**Status:** Production ready, all tests passing