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
- X = factor levels (coded)
- ε = random error ~ N(0, σ²)

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
- Degrees of Freedom: Based on levels and replication
- Mean Square: `MS = SS / df`
- F-statistic: `F = MS_effect / MS_error`
- p-value: From F-distribution with (df_effect, df_error)

---

## 2. Regular Factorial ANOVA

### 2.1 Algorithm

**Input:**
- Design matrix (n × k): factor settings
- Response vector (n × 1): measured values
- Model terms: which effects to estimate

**Steps:**

1. **Build Model Matrix X:**
   ```python
   # For factors A, B with interaction
   X = [1, x_A, x_B, x_A*x_B]  # Each row
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
   - Fit full model
   - Fit model without term
   - SS_term = SS_error(without) - SS_error(with)

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

formula = "Response ~ A + B + A*B"
model = ols(formula, data=data)
results = model.fit()

# ANOVA table
anova_table = sm.stats.anova_lm(results, typ=2)  # Type II SS
```

**Why Type II SS?**
- Handles unbalanced designs properly
- Each term tested after adjusting for all others at same/lower order
- Standard for factorial designs

---

## 3. Split-Plot ANOVA

### 3.1 The Split-Plot Structure

Split-plot designs have **nested error structure**:

```
Whole-Plot (WP) level:
  - Hard-to-change factors (Temperature, Batch, etc.)
  - WP Error: variation between whole-plots

Sub-Plot (SP) level:
  - Easy-to-change factors (Time, Speed, etc.)
  - SP Error: variation within whole-plots
```

**Critical:** Different effects test against different error terms!

### 3.2 Mixed Model Formulation

**Fixed effects:** All factor effects (main effects, interactions)

**Random effects:** Whole-plot ID (accounts for within-WP correlation)

**Model:**
```
Y_ijk = μ + α_i + WP_j(α_i) + β_k + (αβ)_ik + ε_ijk

Where:
- α_i = hard factor (whole-plot level)
- WP_j = random whole-plot effect ~ N(0, σ²_WP)
- β_k = easy factor (sub-plot level)
- (αβ)_ik = interaction
- ε_ijk = sub-plot error ~ N(0, σ²_SP)
```

### 3.3 Proper F-tests

**Whole-plot effects** (hard factors):
```
F = MS_hard / MS_wholeplot
```

**Sub-plot effects** (easy factors):
```
F = MS_easy / MS_subplot
```

**Interactions:**
- Hard × Hard → test against WP error
- Hard × Easy → test against SP error
- Easy × Easy → test against SP error

### 3.4 Implementation with Mixed Models

**Using statsmodels MixedLM:**
```python
from statsmodels.regression.mixed_linear_model import MixedLM

# Fixed effects: all terms
formula = "Response ~ Temperature + Time + Temperature*Time"

# Random effect: whole-plot
model = MixedLM.from_formula(
    formula,
    data=data,
    groups=data['WholePlot'],  # Group by whole-plot ID
    re_formula='1'  # Random intercept
)

results = model.fit(method='lbfgs')
```

**Variance components:**
- σ²_WP: Between whole-plot variance
- σ²_SP: Within whole-plot variance (residual)

### 3.5 Detection Algorithm

**Auto-detect split-plot:**
1. Check factor `changeability` attributes
2. If any factor is HARD or VERY_HARD → split-plot
3. Verify `WholePlot` column exists in design

**Structure classification:**
- VERY_HARD factors → Whole-whole-plots
- HARD factors → Whole-plots
- EASY factors → Sub-plots

---

## 4. Blocked Designs

### 4.1 Block as Random Effect

Blocking accounts for nuisance variation:
- Different days
- Different batches
- Different operators

**Model block as random effect:**
```
Y_ijk = μ + Block_i + α_j + β_k + ... + ε_ijk

Block_i ~ N(0, σ²_Block)
```

### 4.2 Implementation

**Same as split-plot, but group by Block:**
```python
model = MixedLM.from_formula(
    formula,
    data=data,
    groups=data['Block'],
    re_formula='1'
)
```

### 4.3 Split-Plot + Blocking

**Nested structure:**
```
Block → WholePlot(Block) → SubPlot(WholePlot)
```

**Random effects:**
- Block: outermost variation
- WholePlot nested in Block: middle variation
- Residual: innermost variation

**Implementation workaround:**
```python
# Create composite grouping variable
data['Block_WholePlot'] = data['Block'].astype(str) + '_' + data['WholePlot'].astype(str)

model = MixedLM.from_formula(
    formula,
    data=data,
    groups=data['Block_WholePlot'],
    re_formula='1'
)
```

---

## 5. Model Term Management

### 5.1 Hierarchy Enforcement

**Principle:** If including A×B, must include A and B

**Why?** 
- Interpretability: Can't estimate interaction without main effects
- Statistical: Correlation between terms without hierarchy
- Standard practice: All software enforces this

**Implementation:**
1. Parse each term for constituent factors
2. For interactions (A×B) or powers (A²), add main effects
3. Warn user of additions

### 5.2 Model Term Syntax

**Supported notation:**
- Main effect: `"A"`, `"Temperature"`
- Interaction: `"A*B"`, `"Temperature*Pressure"`
- Quadratic: `"A^2"`, `"Temperature^2"`
- Intercept: `"1"` (optional, statsmodels adds by default)

**Conversion for statsmodels:**
```python
# User notation → statsmodels notation
"A^2" → "A**2"  # Python exponentiation
"A*B" → "A:B"   # statsmodels uses colon for interactions (automatic)
```

### 5.3 Validation

**Check before fitting:**
1. All factors in terms exist in factor list
2. Quadratic terms only for continuous factors
3. Quadratic terms only if design has >2 levels
4. All terms are valid syntax

---

## 6. Diagnostic Analysis

### 6.1 Residual Diagnostics

**Purpose:** Validate model assumptions
- Normality: ε ~ N(0, σ²)
- Homoscedasticity: constant variance
- Independence: no patterns in residuals

**Tests:**

1. **Shapiro-Wilk Test (Normality):**
   ```
   H₀: Residuals are normally distributed
   If p < 0.05 → reject H₀ → non-normal
   ```

2. **Breusch-Pagan Test (Homoscedasticity):**
   ```
   H₀: Variance is constant
   If p < 0.05 → reject H₀ → heteroscedastic
   ```

### 6.2 Diagnostic Plots

**1. Normal Probability Plot**
- Plot: Theoretical quantiles vs sample quantiles
- Expected: Points on diagonal line
- Violations: S-curve, outliers

**2. Residuals vs Fitted**
- Plot: e vs ŷ
- Expected: Random scatter around zero
- Violations: Funnel (heteroscedasticity), curves (nonlinearity)

**3. Residuals vs Run Order**
- Plot: e vs time order
- Expected: Random scatter
- Violations: Trends, cycles (process drift, autocorrelation)

**4. Leverage Plot (Cook's Distance)**
- Identifies influential observations
- Cook's D > 1: Very influential
- Cook's D > 4/n: Potentially influential

**Formula:**
```
D_i = (e_i² / (p × MSE)) × (h_i / (1-h_i)²)

Where:
- e_i = residual for observation i
- h_i = leverage (diagonal of hat matrix)
- p = number of parameters
- MSE = mean squared error
```

---

## 7. Effect Visualization

### 7.1 Main Effects Plot

**For each factor:**
1. Group observations by factor level
2. Compute mean response at each level
3. Plot level vs mean response
4. Connect with lines

**Interpretation:**
- Horizontal line → no effect
- Sloped line → main effect present
- Steeper slope → stronger effect

### 7.2 Interaction Plot

**For two factors A and B:**
1. Plot A levels on x-axis
2. Plot separate lines for each B level
3. Each line shows mean response

**Interpretation:**
- Parallel lines → no interaction
- Non-parallel lines → interaction present
- Crossing lines → strong interaction

### 7.3 LogWorth Chart (Significance)

**Purpose:** Visual ranking of effect importance

**LogWorth = -log₁₀(p-value)**

**Interpretation:**
- LogWorth > 1.3 → p < 0.05 (significant at α=0.05)
- LogWorth > 2.0 → p < 0.01 (highly significant)
- Longer bars → more significant effects

**Chart elements:**
- Horizontal bars: LogWorth for each term
- Threshold line: Significance level
- Colors: Red (significant), Gray (not significant)
- Labels: Actual p-values on right side

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
- Use for comparing models

**RMSE (Root Mean Squared Error):**
```
RMSE = √(SS_error / n)
```
- Same units as response
- Lower is better
- Intuitive measure of prediction error

### 8.2 Model Selection Strategy

**Hierarchical approach:**
1. Start with full model (all terms)
2. Remove non-significant terms
3. Keep hierarchy intact
4. Re-fit and check fit metrics

**Information Criteria (future enhancement):**
- AIC: Akaike Information Criterion
- BIC: Bayesian Information Criterion

---

## 9. Computational Complexity

### 9.1 Time Complexity

**Model fitting:**
- OLS: O(n·p² + p³) for n observations, p parameters
- Mixed models: O(n·p² + p³ + n·g) for g groups

**ANOVA table (Type II SS):**
- O(p × [model fitting cost])
- Must refit p times (one per term)

**Diagnostics:**
- O(n) for residual calculations
- O(n²) for leverage (hat matrix diagonal)

### 9.2 Space Complexity

- Design matrix: O(n·p)
- Covariance matrix: O(p²)
- For typical designs: negligible

### 9.3 Numerical Stability

**Concerns:**
- Matrix inversion for (X'X)⁻¹
- Condition number of design matrix

**Solutions:**
- Use coded levels ([-1, 1]) → better conditioning
- Ridge regularization if needed: (X'X + λI)⁻¹
- QR decomposition instead of direct inversion

---

## 10. References

### Primary References

[1] **Montgomery, D. C. (2017).** *Design and Analysis of Experiments*, 9th Edition. Wiley.
    - Chapter 5: Factorial Designs
    - Chapter 14: Split-Plot Designs
    - Gold standard textbook

[2] **Box, G. E. P., Hunter, W. G., & Hunter, J. S. (2005).** *Statistics for Experimenters: Design, Innovation, and Discovery*, 2nd Edition. Wiley-Interscience.
    - Chapter 5: Factorial Designs at Two Levels
    - Practical, example-driven approach

[3] **Littell, R. C., Milliken, G. A., Stroup, W. W., Wolfinger, R. D., & Schabenberger, O. (2006).** *SAS for Mixed Models*, 2nd Edition. SAS Institute.
    - Chapter 8: Split-Plot Designs
    - Detailed mixed model methodology

### Statistical Software Documentation

[4] **statsmodels documentation:**
    - https://www.statsmodels.org/stable/mixed_linear.html
    - Mixed Linear Models (MixedLM)

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

## 11. Implementation Details

### 11.1 Code Architecture

**Module structure:**
```
analysis.py
├── Model Term Generation
│   ├── generate_model_terms()
│   ├── parse_model_term()
│   └── enforce_hierarchy()
├── Structure Detection
│   └── detect_split_plot_structure()
├── Data Preparation
│   ├── prepare_analysis_data()
│   └── validate_model_terms()
├── ANOVAAnalysis Class
│   ├── __init__()
│   ├── fit()
│   ├── _fit_regular_model()
│   ├── _fit_split_plot_model()
│   ├── _build_formula()
│   ├── _extract_results()
│   ├── _compute_diagnostics()
│   ├── update_model()
│   ├── plot_effects()
│   ├── plot_diagnostics()
│   └── plot_significance()
└── ANOVAResults (dataclass)
```

### 11.2 Key Design Decisions

**1. Auto-detection with override:**
- Default: Detect split-plot from changeability
- Option: User can override with `is_split_plot` parameter
- Rationale: Convenience + flexibility

**2. Separate response and design:**
- Response passed as separate array/Series
- Validation: Check length match
- Rationale: Clean API, common in R/Python stats packages

**3. Model term strings:**
- User-friendly notation: "A*B", "A^2"
- Internal conversion to statsmodels format
- Rationale: Matches R formula syntax (familiar to statisticians)

**4. Mixed models for all random effects:**
- Blocks → MixedLM with Block as group
- Split-plot → MixedLM with WholePlot as group
- Rationale: Unified framework, proper inference

### 11.3 Error Handling

**Common errors and solutions:**

1. **No WholePlot column for split-plot:**
   ```
   ValueError: Split-plot analysis requires 'WholePlot' column
   Solution: Use generate_split_plot_design()
   ```

2. **Response length mismatch:**
   ```
   ValueError: Response length must match design length
   Solution: Check that response matches design rows
   ```

3. **Invalid factor in term:**
   ```
   ValueError: Factor 'X' not found in factor list
   Solution: Check factor names match exactly
   ```

4. **Quadratic on categorical:**
   ```
   ValueError: Quadratic term only valid for continuous factors
   Solution: Use continuous factors for quadratic models
   ```

---

## 12. Future Enhancements

### Planned Features

1. **Multiple responses:**
   - Fit multiple models independently
   - Combine in multi-response optimization

2. **More information criteria:**
   - AIC, BIC for model selection
   - Automated stepwise selection

3. **Advanced diagnostics:**
   - Variance Inflation Factor (VIF) for multicollinearity
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
from src.core.analysis import ANOVAAnalysis, generate_model_terms
from src.core.factors import Factor, FactorType, ChangeabilityLevel

# Define factors
factors = [
    Factor("Temperature", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, 
           levels=[150, 200]),
    Factor("Pressure", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
           levels=[50, 100])
]

# Assume we have design and response
# design = ... (from full_factorial)
# response = ... (measured values)

# Generate standard model terms
terms = generate_model_terms(factors, 'interaction')  # Linear + 2-way

# Fit model
analysis = ANOVAAnalysis(design, response, factors)
results = analysis.fit(terms)

# View results
print(results.anova_table)
print(results.effect_estimates)
print(f"R² = {results.r_squared:.3f}")

# Diagnostics
fig = analysis.plot_diagnostics()
```

### Example 2: Split-Plot ANOVA

```python
from src.core.split_plot import generate_split_plot_design

# Define factors with changeability
factors = [
    Factor("Temperature", FactorType.CONTINUOUS, ChangeabilityLevel.HARD,
           levels=[100, 200]),  # Hard to change
    Factor("Time", FactorType.CONTINUOUS, ChangeabilityLevel.EASY,
           levels=[10, 30])  # Easy to change
]

# Generate split-plot design
design = generate_split_plot_design(
    factors=factors,
    n_replicates=3,
    randomize_whole_plots=True
)

# Measure response
# response = ... (collect data)

# Fit split-plot model
analysis = ANOVAAnalysis(design.design, response, factors)
# is_split_plot auto-detected from factor changeability

results = analysis.fit(['Temperature', 'Time', 'Temperature*Time'])

# Results have proper error terms
print(results.anova_table)  # Shows WP and SP error
```

### Example 3: Model Refinement

```python
# Start with full model
results1 = analysis.fit(['A', 'B', 'C', 'A*B', 'A*C', 'B*C'])

# Check significance
fig = analysis.plot_significance(alpha=0.05)

# Remove non-significant terms
results2 = analysis.update_model(terms_to_remove=['B*C'])

# Compare fit
print(f"Full model R² = {results1.r_squared:.3f}")
print(f"Reduced model R² = {results2.r_squared:.3f}")
print(f"Adjusted R² improved: {results2.adj_r_squared > results1.adj_r_squared}")
```

---

**Document Version:** 1.0  
**Last Updated:** Session 9 Implementation  
**Author:** DOE-Toolkit Development Team