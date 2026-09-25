# ANOVA Analysis Algorithm Documentation

## Overview

This document describes the statistical methodology and implementation details for the ANOVA (Analysis of Variance) module in DOE-Toolkit. The module supports regular factorial ANOVA, split-plot ANOVA with proper error terms, and blocked designs.

---

## 1. Statistical Background

### 1.1 Purpose of ANOVA

ANOVA decomposes the total variation in a response into components attributable to:
- **Factor effects**: Main effects and interactions
- **Error**: Unexplained variation

The key question: Are observed differences in response due to factors, or just random variation?

### 1.2 The Linear Model

For a factorial design with factors A, B, C:

```
Y = β₀ + β₁X₁ + β₂X₂ + β₃X₃ + β₁₂X₁X₂ + ... + ε
```

Where:
- Y = response
- β = coefficients (effects)
- X = factor levels
- ε = random error ~ N(0, σ²)

The analysis encodes designs to coded [-1, 1] space internally (via
`DesignSpace` in `src/core/coding.py`) so continuous factors contribute
comparably; natural-unit coefficient values are recovered afterward for
reporting.

### 1.3 ANOVA Table Structure

| Source | SS | df | MS | F | p-value |
|--------|----|----|----|----|---------|
| Factor A | SS_A | df_A | MS_A | F_A | p_A |
| Factor B | SS_B | df_B | MS_B | F_B | p_B |
| A×B | SS_AB | df_AB | MS_AB | F_AB | p_AB |
| Error | SS_E | df_E | MS_E | - | - |
| Total | SS_T | df_T | - | - | - |

**Key formulas:**
- Sum of Squares: `SS = Σ(ŷ - ȳ)²` for each source
- Degrees of Freedom: based on levels and replication
- Mean Square: `MS = SS / df`
- F-statistic: `F = MS_effect / MS_error`
- p-value: from the F-distribution with (df_effect, df_error)

---

## 2. Regular Factorial ANOVA

### 2.1 Algorithm

**Input:**
- Design matrix (n × k): factor settings
- Response vector (n × 1): measured values
- Model terms: which effects to estimate

**Steps:**

1. **Build Model Matrix X** (one row per run):
   ```
   X = [1, x_A, x_B, x_A*x_B]
   ```

2. **Estimate Coefficients (OLS):**
   ```
   β̂ = (X'X)⁻¹X'y
   ```

3. **Compute Fitted Values:**
   ```
   ŷ = Xβ̂
   ```

4. **Compute Residuals:**
   ```
   e = y - ŷ
   ```

5. **Compute Sums of Squares:**
   ```
   SS_total = Σ(y - ȳ)²
   SS_model = Σ(ŷ - ȳ)²
   SS_error = Σ(y - ŷ)² = Σe²
   ```

6. **Compute Type II SS for Each Term:**
   - Fit the full model
   - For each term, obtain its Type II SS by removing only that term from the
     model (all other terms retained)
   - statsmodels computes Type II SS directly via `anova_lm(fit, typ=2)`

7. **Compute F-statistics:**
   ```
   F_term = MS_term / MS_error
   ```

8. **Compute p-values:**
   ```
   p = P(F_{df_term, df_error} > F_term)
   ```

### 2.2 Implementation Notes

**Using statsmodels:**
```python
from statsmodels.formula.api import ols
import statsmodels.api as sm

formula = "Response ~ Temperature + Pressure + Temperature*Pressure"
model = ols(formula, data=df)
results = model.fit()

anova_table = sm.stats.anova_lm(results, typ=2)  # Type II SS
```

**Why Type II SS?**
- Handles unbalanced designs properly
- Each term is tested after adjusting for all others at the same or lower order
- Standard for factorial designs

---

## 3. Split-Plot ANOVA

### 3.1 The Split-Plot Structure

Split-plot designs have **nested error structure**:

```
Whole-Plot (WP) level:
  - Hard- and very-hard-to-change factors (Temperature, Batch, Line, etc.)
  - WP Error: variation between whole-plots

Sub-Plot (SP) level:
  - Easy-to-change factors (Time, Speed, etc.)
  - SP Error: variation within whole-plots
```

**Critical:** different effects test against different error terms!

### 3.2 Error Structure

The shipped analysis uses the **two-strata Yates / expected-mean-squares
approach** — not a mixed-model variance-components fit. For a design with a
hard factor A and an easy factor B, with sub-plot error ε and whole-plot
error γ:

| Source | E(MS) |
|--------|-------|
| Factor A (whole-plot) | σ²_ε + n·σ²_γ + Q_A/(a−1) |
| Whole-plot error | σ²_ε + n·σ²_γ |
| Factor B (sub-plot) | σ²_ε + Q_B/(b−1) |
| A × B interaction | σ²_ε + Q_AB/((a−1)(b−1)) |
| Sub-plot error | σ²_ε |

where n = number of sub-plots per whole-plot and Q denotes the fixed quadratic
form for a factor. The whole-plot and sub-plot effects therefore differ in
their error denominator: whole-plot effects include the between-plot variance
component σ²_γ.

**F-tests:**
- F_A = MS_A / MS_whole-plot error (NOT sub-plot error!)
- F_B = MS_B / MS_sub-plot error
- F_AB = MS_AB / MS_sub-plot error

### 3.3 Implementation: Two-Strata OLS Pipeline

`ANOVAAnalysis.fit()` detects a split-plot structure from factor changeability
and delegates to `fit_split_plot_anova` (`src/core/split_plot_analysis.py`).
The pipeline is:

1. **Classify terms into strata.** A term belongs to the whole-plot stratum
   only if *all* of its constituent factors are hard/very-hard; any term
   touching an easy factor belongs to the sub-plot stratum.

2. **Whole-plot stratum.** Collapse the run data to **one row per whole-plot**
   (mean of the response over its sub-plots). Fit an OLS model of the
   whole-plot terms on those means. The residual from this model is the
   **whole-plot error** (its df = n_whole_plots − rank of the WP model).

3. **Sub-plot stratum.** Fit an OLS model on the **full run data** with all
   terms plus `C(WholePlot)` — the whole-plot ID absorbed as a fixed factor,
   which removes whole-plot variation from the residuals. The residual is the
   **sub-plot error** (df = n_runs − rank of this model).

4. **Assemble the table.** Whole-plot terms are tested against
   MS_whole-plot error; sub-plot terms and cross-strata interactions are
   tested against MS_sub-plot error. Each row is tagged with a `Stratum`
   column (`'Whole-Plot'` or `'Sub-Plot'`).

**What this approach does NOT do:**
- It does **not** estimate random-effect variance components (σ²_γ, σ²_ε are
  not reported separately; the analysis works with the mean-square ratios).
- It does **not** use `MixedLM`. A mixed model with a random whole-plot
  intercept is singular for split-plot data because hard factors do not vary
  within a whole-plot, which raises a `LinAlgError` — the docstring of
  `_fit_mixed_effects_model` documents this explicitly, and the method name is
  retained for compatibility only.

### 3.4 Detection Algorithm

`detect_split_plot_structure(design, factors)` (`src/core/analysis.py`):

```
1. Check factor changeability attributes
2. any HARD or VERY_HARD factor → is_split_plot = True
3. whole_plot_factors = very_hard + hard factors
4. sub_plot_factors = easy factors
5. has_blocking = 'Block' in design.columns
6. whole_plot_column = 'WholePlot' (must exist for a split-plot fit)
```

**Two error strata, regardless of nesting depth.** Even when the *design*
has three nesting levels (very-hard → hard → easy, produced by
`generate_split_plot_design`), the *analysis* collapses the very-hard and hard
factors into a **single whole-plot stratum** — both test their main effects
against the same whole-plot error. There are exactly two error terms, matching
the two-strata table above.

A split-plot fit without a `WholePlot` column raises:

```
ValueError: Split-plot analysis requires a 'WholePlot' column in the design.
Ensure the design was generated with hard-to-change factors so that
whole-plot groupings are recorded.
```

### 3.5 Worked Example (df structure)

Baking experiment: oven temperature (hard, 300/400 °F) × baking time (easy,
10/30 min), run on **three days** (replicates), i.e. 3 × 2 × 2 = 12 runs and
6 whole-plots (one per day-temperature). Fitting
`['Temperature', 'Time', 'Temperature*Time']` produces this structural table
(df/SS/F/P depend on the measured response, so only df/stratum are shown):

```
                  df          Stratum
Temperature        1          Whole-Plot
WholePlot Error    4          Whole-Plot
Time               1          Sub-Plot
Temperature*Time   1          Sub-Plot
SubPlot Error      4          Sub-Plot
```

Checks: whole-plot stratum total = 1 + 4 = 5 = n_whole_plots − 1 = 6 − 1;
sub-plot stratum total = 1 + 1 + 4 = 6 = n_runs − n_whole_plots = 12 − 6.

---

## 4. Blocked Designs

### 4.1 Block as a Factor

Blocking accounts for nuisance variation:
- Different days
- Different batches
- Different operators

### 4.2 Implementation

For non-split-plot designs with a `Block` column, the default is to add
`Block` as a **fixed categorical term** (its values are cast to strings so
patsy treats it as categorical):

```python
formula = "Response ~ Temperature + Pressure + Block"
model = ols(formula, data=df)
results = model.fit()
```

Passing `block_as_random=True` to `ANOVAAnalysis` instead fits Block as a
**random effect**:

```python
model = mixedlm(formula, data=df, groups=df['Block'], re_formula='1')
results = model.fit(method='lbfgs')
```

The mixed-model path is only used for blocked (non-split-plot) designs; the
split-plot path always uses the two-strata OLS pipeline above.

### 4.3 Split-Plot + Blocking

Blocking can be applied at the whole-plot level during design generation
(`generate_split_plot_design(n_blocks=...)`). The `Block` column is carried
into the analysis data, but the shipped two-strata split-plot fit does **not**
currently include `Block` as a model term (a documented limitation). There is
no composite `Block_WholePlot` interaction term in the code.

---

## 5. Model Term Management

### 5.1 Hierarchy Enforcement

**Principle:** if including A×B, must include A and B (and a quadratic I(A**2)
builds on its main effect).

**Why?**
- Interpretability: an interaction is hard to interpret without its parents
- Statistical estimability / correlation between terms without hierarchy
- Standard practice: most software enforces this

**Implementation:** `enforce_hierarchy()` adds any missing lower-order terms
and warns:

```
warnings.warn(f"Added for hierarchy: {added}")
```

### 5.2 Model Term Syntax

**Supported notation (patsy):**

- Main effect: `"Temperature"`
- Interaction: `"Temperature*Pressure"` (patsy expands `A*B` into
  `A + B + A:B` automatically)
- Quadratic: `"I(Temperature**2)"` — the `I()` identity function forces
  Python exponentiation; a bare `A**2` is parsed by patsy as an interaction
  of `A` with itself and is dropped. All quadratic documentation and the
  model builder use the `I(...)` form.
- Intercept: `"1"` (included implicitly by statsmodels)

`generate_model_terms(factors, model_type)` emits `'linear'`,
`'interaction'` or `'quadratic'` term sets directly in this notation,
appending `I({factor}**2)` for every continuous factor when quadratic is
requested.

### 5.3 Validation

`validate_model_terms()` checks before fitting:
1. All factors in every term exist:
   ```
   ValueError: Factor '<name>' in '<term>' not found
   ```
2. Quadratic terms only for continuous factors:
   ```
   ValueError: Quadratic '<term>' requires continuous factor
   ```
3. Transform terms (e.g. `np.log(A)`, `I(1/A)`) require a continuous factor:
   ```
   ValueError: Transform term '<term>' requires a continuous factor;
   '<factor>' is not continuous.
   ```
4. Degrees of freedom are checked; saturation warns:
   ```
   UserWarning: Low df_error = <n>: Inference may be unreliable
   ```

---

## 6. Diagnostic Analysis

### 6.1 Residual Diagnostics

**Purpose:** validate model assumptions
- Normality: ε ~ N(0, σ²)
- Constant variance (homoscedasticity)
- Independence: no patterns in residuals

**Tests implemented (shipped):**

1. **Shapiro-Wilk Test (Normality):**
   ```
   H₀: Residuals are normally distributed
   If p < 0.05 → reject H₀ → non-normal
   ```
   Computed when the number of residuals is ≤ 5000 and reported under
   `results.diagnostics['shapiro_wilk']` as `{'statistic': ..., 'p_value': ...}`.

Only Shapiro-Wilk is currently implemented in `_compute_diagnostics()`; there
is no Breusch-Pagan (or other heteroscedasticity) test in the shipped module.
Homoscedasticity is assessed graphically via the residual plots below.

### 6.2 Diagnostic Plots

Plotting lives in `src/ui/utils/plotting.py` (Streamlit-free, returns
Plotly figures). The functions used by the Analyze page include:

| Plot | Function |
|------|----------|
| Actual vs predicted (parity) | `create_parity_plot(actual, predicted, response_units=...)` |
| Residuals vs fitted | `create_residual_plot(fitted, residuals, response_units=...)` |
| QQ (normality) | `create_qq_plot(residuals)` |
| Residuals vs run order | `create_residual_vs_run_order_plot(...)` |
| Residuals vs factor | `create_residual_vs_factor_plot(...)` |

**1. Parity Plot (Actual vs Predicted)**
- Points should scatter around the 1:1 line
- The 95% band reflects uncertainty in the fit line
- Curvature or drift indicates a missing term

**2. Residuals vs Fitted**
- Expected: random scatter around zero
- Violations: funnel shape (heteroscedasticity), curves (nonlinearity)

**3. QQ Plot**
- Points on the diagonal → residuals approximately normal
- S-curve or heavy tails → non-normality

**4. Residuals vs Run Order**
- Expected: random scatter (no time pattern)
- Violations: trends or cycles (drift, autocorrelation)

Influence is surfaced via the leverage/Cook's-distance machinery in the
diagnostics pages:

```
D_i = (e_i² / (p × MSE)) × (h_i / (1-h_i)²)
```
- h_i = leverage (diagonal of the hat matrix)
- p = number of parameters
- MSE = mean squared error

---

## 7. Effect Visualization

### 7.1 Main Effects Plot

**For each factor:**
1. Group observations by factor level
2. Compute mean response at each level
3. Plot level vs mean response, connected by lines

**Interpretation:**
- Horizontal line → no effect
- Sloped line → main effect present
- Steeper slope → stronger effect

### 7.2 Interaction Plot

**For two factors A and B:**
1. Plot A levels on the x-axis
2. Plot a separate line for each B level (mean response)
3. Non-parallel lines → interaction; parallel lines → no interaction

Implemented as `create_interaction_plot(...)` in `src/ui/utils/plotting.py`.

### 7.3 LogWorth Chart (Significance)

**Purpose:** visual ranking of effect importance

**LogWorth = -log₁₀(p-value)**

**Interpretation:**
- LogWorth > 1.301 → p < 0.05 (significant at α=0.05)
- LogWorth > 2 → p < 0.01 (highly significant)
- Longer bars → more significant effects

Implementations: `create_logworth_plot(logworth_df, p_values)` (term-level
ANOVA p-values) and `create_coefficient_significance_plot(..., alpha=0.05)`
(coefficient-level). The Analyze page also offers a
`create_standardized_effects_plot(...)` (DOE standardized effects) and a
half-normal/probability plot `create_half_normal_plot(effects, effect_names)`.

---

## 8. Model Comparison and Selection

### 8.1 Goodness of Fit Metrics

**R² (Coefficient of Determination):**
```
R² = 1 - SS_error / SS_total
```
- Range: [0, 1]
- Higher is better
- Can be inflated by adding terms

**Adjusted R²:**
```
R²_adj = 1 - (1 - R²) × (n-1) / (n-p-1)
```
- Penalizes model complexity
- For split-plot results, computed over the fixed terms of interest (whole-plot
  absorption columns excluded)

**RMSE (Root Mean Squared Error):**
```
RMSE = √(SS_error / n)
```
- Same units as the response
- Lower is better

### 8.2 Model Selection: BIC Stepwise (shipped)

BIC-based **bidirectional stepwise selection** is implemented in
`src/core/stepwise.py` and exposed on the Analyze page
(`src/ui/components/model_builder.py`):

```
BIC = n·ln(RSS/n) + k·ln(n)        (n = observations, k = parameters)
```

`stepwise_selection(anova_results, ...)` (forward/backward, `bic_threshold`)
adds or removes one term at a time while respecting model hierarchy, keeping
the step that lowers BIC most; a `StepwiseResults` object reports each step's
BIC and ΔBIC. The design-aware automatic model selection
(`src/core/selection.py`) starts from the stepwise result and refines it
against the design in use.

---

## 9. References

### Primary References

[1] **Montgomery, D. C. (2017).** *Design and Analysis of Experiments*, 9th Edition. Wiley.
    - Chapter 5: Factorial Designs
    - Chapter 14: Split-Plot Designs

[2] **Box, G. E. P., Hunter, W. G., & Hunter, J. S. (2005).** *Statistics for Experimenters: Design, Innovation, and Discovery*, 2nd Edition. Wiley-Interscience.
    - Chapter 5: Factorial Designs at Two Levels

[3] **Littell, R. C., Milliken, G. A., Stroup, W. W., Wolfinger, R. D., & Schabenberger, O. (2006).** *SAS for Mixed Models*, 2nd Edition. SAS Institute.
    - Chapter 8: Split-Plot Designs (mixed-model background)

### Statistical Software Documentation

[4] **statsmodels documentation:**
    - https://www.statsmodels.org/stable/regression.html (OLS)
    - https://www.statsmodels.org/stable/mixed_linear.html (MixedLM, blocked designs)

[5] **scipy.stats documentation:**
    - https://docs.scipy.org/doc/scipy/reference/stats.html
    - Statistical tests (Shapiro-Wilk, etc.)

### Online Resources

[6] **NIST Engineering Statistics Handbook:**
    - https://www.itl.nist.gov/div898/handbook/
    - Section 5.3: Full Factorial Designs
    - Section 5.4: Fractional Factorial Designs
    - Section 5.6: Split-Plot Designs

---

## 10. Implementation Details

### 10.1 Module Structure

**Core (Streamlit-free):**
```
analysis.py
├── generate_model_terms()          # 'linear' / 'interaction' / 'quadratic'
├── detect_split_plot_structure()   # changeability → strata + columns
├── prepare_analysis_data()         # factor columns + response + plot/block cols
├── validate_model_terms()          # existence / quadratic / transform rules
├── ANOVAAnalysis
│   ├── __init__()                  # DesignSpace coding, structure detection
│   ├── fit()                       # resolve terms, hierarchy, prune, fit
│   ├── _fit_fixed_effects_model()  # OLS (or mixedlm for blocked random)
│   ├── _fit_mixed_effects_model()  # split-plot → two-strata OLS (legacy name)
│   ├── update_model()              # add/remove terms and refit
│   └── _compute_diagnostics()      # Shapiro-Wilk
├── ANOVAResults (imported from analysis_base)
```

```
analysis_base.py
├── ANOVAResults dataclass
├── parse_model_term() / enforce_hierarchy() / compute_actual_coefficients()
├── build_anova_effect_summary() / build_coefficient_significance()
```

```
split_plot_analysis.py
└── fit_split_plot_anova()          # two-strata Yates/EMS OLS pipeline
```

**Plotting:** `src/ui/utils/plotting.py` — `create_parity_plot`,
`create_residual_plot`, `create_qq_plot`, `create_logworth_plot`,
`create_coefficient_significance_plot`, `create_standardized_effects_plot`,
`create_half_normal_plot`, `create_interaction_plot`, etc. (Plotly figures,
no Streamlit dependency).

### 10.2 Key Design Decisions

**1. Auto-detection with override:**
- Default: detect split-plot from changeability
- Option: user can override with the `is_split_plot` parameter
- Rationale: convenience + flexibility

**2. Separate response and design:**
- Response passed as a separate array/Series
- Validation: length must match (`Response length mismatch: <n> != <m>`)
- Rationale: clean API, common in R/Python stats packages

**3. Model term strings:**
- User-friendly patsy notation: `"A"`, `"A*B"`, `"I(A**2)"`
- Quadratics always use the `I()` identity wrapper
- Rationale: matches R formula syntax (familiar to statisticians)

**4. Split-plot inference is two-strata OLS, not MixedLM:**
- MixedLM is used only for blocked (non-split-plot) designs when
  `block_as_random=True`
- Split-plot uses the Yates / expected-mean-squares two-strata OLS to avoid
  the singular-matrix failure of a random whole-plot intercept model
- Rationale: correct F-tests with per-stratum error terms, no variance
  components needed

### 10.3 Error Handling

**Common errors and solutions (verbatim messages):**

1. **No WholePlot column for split-plot:**
   ```
   ValueError: Split-plot analysis requires a 'WholePlot' column in the design.
   Ensure the design was generated with hard-to-change factors so that
   whole-plot groupings are recorded.
   ```
   Solution: generate the design with `generate_split_plot_design()`.

2. **Response length mismatch:**
   ```
   ValueError: Response length mismatch: <n> != <m>
   ```
   Solution: check that the response matches the design rows.

3. **Invalid factor in term:**
   ```
   ValueError: Factor '<name>' in '<term>' not found
   ```
   Solution: check that factor names match exactly.

4. **Quadratic on a categorical factor:**
   ```
   ValueError: Quadratic '<term>' requires continuous factor
   ```
   Solution: use continuous factors for quadratic models.

---

## 11. Future Enhancements

### Planned Features

1. **Block-as-term support for split-plot fits** — include the `Block` column
   in the two-strata model rather than carrying it in the data only.

2. **More information criteria / selection metrics** — expose AIC in addition
   to the shipped BIC stepwise.

3. **Advanced diagnostics:**
   - Variance Inflation Factor (VIF) for multicollinearity
   - Breusch-Pagan (heteroscedasticity)
   - DFFITS, DFBETAS for influence
   - Partial residual plots

4. **Robust methods:**
   - Resistant regression for outliers
   - Non-parametric alternatives

5. **Aliasing integration:**
   - Read alias structure from fractional factorial designs
   - Warn when fitting aliased terms
   - Suggest alternative models

---

## Appendix: Example Workflows

### Example 1: Simple Factorial ANOVA

```python
import numpy as np
from src.core.factors import Factor, FactorType
from src.core.full_factorial import full_factorial
from src.core.analysis import ANOVAAnalysis, generate_model_terms

factors = [
    Factor("Temperature", FactorType.CONTINUOUS, levels=[150, 200]),
    Factor("Pressure", FactorType.CONTINUOUS, levels=[50, 100]),
]

design = full_factorial(factors, n_center_points=1, randomize=False, random_seed=42)

rng = np.random.default_rng(42)
response = (
    5.0 + 0.3 * design["Temperature"] + 0.04 * design["Pressure"]
    + rng.normal(0, 0.5, len(design))
)

terms = generate_model_terms(factors, "interaction")  # main effects + 2-way

analysis = ANOVAAnalysis(design, response, factors)
results = analysis.fit(terms)

print(results.anova_table)
print(results.effect_estimates[["Coefficient", "p_value"]])
print(f"R² = {results.r_squared:.3f}")
```

### Example 2: Split-Plot ANOVA (two-strata)

```python
import numpy as np
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.split_plot import generate_split_plot_design
from src.core.analysis import ANOVAAnalysis

factors = [
    Factor(name="Temperature", factor_type=FactorType.CONTINUOUS,
           levels=[300, 400], changeability=ChangeabilityLevel.HARD),
    Factor(name="Time", factor_type=FactorType.CONTINUOUS,
           levels=[10, 30], changeability=ChangeabilityLevel.EASY),
]

design = generate_split_plot_design(factors=factors, n_replicates=3, seed=42,
                                    randomize_whole_plots=True,
                                    randomize_sub_plots=True)

rng = np.random.default_rng(7)
response = (
    50.0 + 0.1 * design.design["Temperature"] + 0.5 * design.design["Time"]
    + rng.normal(0, 0.4, len(design.design))
)

analysis = ANOVAAnalysis(design.design, response, factors)  # split-plot auto-detected
results = analysis.fit(["Temperature", "Time", "Temperature*Time"])

print(results.anova_table[["df", "F", "P", "Stratum"]])
```

Residual plots for either example:

```python
from src.ui.utils.plotting import create_residual_plot, create_qq_plot

res_fig = create_residual_plot(results.fitted_values, results.residuals)
qq_fig = create_qq_plot(results.residuals)
```

### Example 3: Model Refinement

```python
# Start with the full interaction model
results1 = analysis.fit(["Temperature", "Pressure", "Temperature*Pressure"])

# Drop a non-significant interaction (hierarchy is re-checked)
results2 = analysis.update_model(terms_to_remove=["Temperature*Pressure"])

print(f"Full model R² = {results1.r_squared:.3f}")
print(f"Reduced model R² = {results2.r_squared:.3f}")
```
