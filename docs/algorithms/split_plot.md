# Split-Plot Designs

## Purpose

Split-plot designs are experimental structures used when some factors are harder or more expensive to change than others. They provide proper randomization restrictions and error structure for analyzing experiments where complete randomization is impractical or impossible.

## When to Use Split-Plot Designs

### Common Scenarios

1. **Manufacturing processes** where machine settings are expensive to change
   - Example: Oven temperature (hard) vs. baking time (easy)

2. **Agricultural experiments** with field-level and plot-level factors
   - Example: Irrigation method (hard) vs. fertilizer amount (easy)

3. **Chemical processes** with batch and run-level variables
   - Example: Batch temperature (hard) vs. catalyst amount (easy)

4. **Industrial experiments** with equipment constraints
   - Example: Production line (very hard), die temperature (hard), cycle time (easy)

## Theoretical Foundation

### Factor Changeability Hierarchy

Split-plot designs recognize that factors have different levels of changeability:

| Changeability | Description | Examples | Nesting Level |
|---------------|-------------|----------|---------------|
| **VERY_HARD** | Rarely changed, expensive/time-consuming | Production line, equipment, operator | Whole-whole-plot |
| **HARD** | Changed infrequently, moderate cost | Temperature, pressure, batch | Whole-plot |
| **EASY** | Changed freely, low cost | Time, concentration, speed | Sub-plot |

### Plot Structure

A split-plot design creates a **nested hierarchy**:

```
Replicate
â"œâ"€â"€ Whole-Whole-Plot 1 (Very Hard Factor settings)
â"‚   â"œâ"€â"€ Whole-Plot 1.1 (Hard Factor settings)
â"‚   â"‚   â"œâ"€â"€ Sub-Plot 1.1.1 (Easy Factor settings)
â"‚   â"‚   â""â"€â"€ Sub-Plot 1.1.2 (Different Easy settings)
â"‚   â""â"€â"€ Whole-Plot 1.2
â"‚       â"œâ"€â"€ Sub-Plot 1.2.1
â"‚       â""â"€â"€ Sub-Plot 1.2.2
â""â"€â"€ Whole-Whole-Plot 2
    â""â"€â"€ ...
```

**Key principles:**
1. **Whole-plot factors** remain constant within a whole-plot
2. **Sub-plot factors** vary within each whole-plot
3. **Randomization** respects the nesting structure

### Why Split-Plot Matters: Error Structure

**In a completely randomized design**, all factors are tested against the same experimental error.

**In a split-plot design**, different factors are tested against different error terms:

| Factor Type | Error Term | Degrees of Freedom |
|-------------|------------|-------------------|
| Whole-plot effects | Whole-plot error | (# whole-plots) - (# WP factors) - 1 |
| Sub-plot effects | Sub-plot error | (# sub-plots) - (# SP factors) - (# whole-plots) |
| WP × SP interactions | Sub-plot error | Product of factor df |

**Critical implication:** Whole-plot effects have **less precision** (fewer df for error) than sub-plot effects. This is why you assign factors strategically based on importance and changeability.

## Mathematical Model

### Two-Level Split-Plot

For a design with one hard factor (A) and one easy factor (B):

$$
y_{ijk} = \mu + \alpha_i + \gamma_{j(i)} + \beta_k + (\alpha\beta)_{ik} + \epsilon_{ijk}
$$

where:
- $\mu$ = overall mean
- $\alpha_i$ = effect of whole-plot factor A (level $i$)
- $\gamma_{j(i)}$ = whole-plot error (plot $j$ within level $i$)
- $\beta_k$ = effect of sub-plot factor B (level $k$)
- $(\alpha\beta)_{ik}$ = interaction between A and B
- $\epsilon_{ijk}$ = sub-plot error

**Key distinction:** Two error terms ($\gamma$ and $\epsilon$), not one!

### ANOVA Expected Mean Squares

| Source | E(MS) |
|--------|-------|
| Factor A (whole-plot) | $\sigma_\epsilon^2 + n\sigma_\gamma^2 + \frac{Q_A}{a-1}$ |
| Whole-plot error | $\sigma_\epsilon^2 + n\sigma_\gamma^2$ |
| Factor B (sub-plot) | $\sigma_\epsilon^2 + \frac{Q_B}{b-1}$ |
| A × B interaction | $\sigma_\epsilon^2 + \frac{Q_{AB}}{(a-1)(b-1)}$ |
| Sub-plot error | $\sigma_\epsilon^2$ |

where $n$ = number of sub-plots per whole-plot.

**F-tests:**
- $F_A = MS_A / MS_{\text{whole-plot error}}$ (NOT sub-plot error!)
- $F_B = MS_B / MS_{\text{sub-plot error}}$
- $F_{AB} = MS_{AB} / MS_{\text{sub-plot error}}$

**This is why proper structure matters!** Using wrong error term invalidates F-tests.

## Design Generation Algorithm

### Step 1: Classify Factors by Changeability

```python
very_hard_factors = [f for f in factors if f.changeability == VERY_HARD]
hard_factors = [f for f in factors if f.changeability == HARD]
easy_factors = [f for f in factors if f.changeability == EASY]
```

### Step 2: Generate Factorial Combinations

For each level of nesting, generate all combinations of factors at that level:

```python
# Whole-whole-plot combinations (very hard factors)
vh_combinations = factorial(very_hard_factors)  # e.g., 2 machines

# Whole-plot combinations (hard factors)
h_combinations = factorial(hard_factors)  # e.g., 2 temps × 2 pressures = 4

# Sub-plot combinations (easy factors)
e_combinations = factorial(easy_factors)  # e.g., 2 times × 2 catalysts = 4
```

### Step 3: Nest Combinations

Create nested structure by combining levels:

```python
for vh_combo in vh_combinations:           # 2 iterations
    for h_combo in h_combinations:         # 4 iterations per VH
        # This defines one whole-plot
        for e_combo in e_combinations:     # 4 iterations per H
            # This defines one sub-plot (one experimental run)
            run = {**vh_combo, **h_combo, **e_combo}
```

**Result:** 2 × 4 × 4 = 32 runs in this example.

### Step 4: Apply Randomization

**Restricted randomization:**
1. Randomize order of whole-whole-plots (if applicable)
2. Within each whole-whole-plot, randomize order of whole-plots
3. Within each whole-plot, randomize order of sub-plots

```python
# Randomize whole-plots
random.shuffle(whole_plot_order)

# Within each whole-plot, randomize sub-plots
for wp in whole_plots:
    random.shuffle(wp.sub_plots)
```

**Do NOT randomize across plot boundaries!** This would break the split-plot structure.

### Step 5: Add Center Points (Sub-Plot Level Only)

Center points, if requested, are added **only at the sub-plot level**:

```python
if n_center_points > 0:
    for whole_plot in whole_plots:
        for _ in range(n_center_points):
            center_run = {
                **whole_plot.hard_factor_settings,  # Keep WP factors constant
                **center_values_for_easy_factors     # Easy factors to center
            }
```

**Rationale:** Whole-plot factors cannot be easily changed to center values (defeats the purpose of split-plot).

### Step 6: Add Replicates

Complete replicates multiply the entire structure:

```python
for replicate in range(n_replicates):
    # Generate complete design
    # Assign new whole-plot IDs for this replicate
```

**Result:** Independent whole-plots for each replicate (proper df for whole-plot error).

## Implementation Details

### Minimum Design Requirements

For valid error estimation:

| Requirement | Minimum | Recommended | Reason |
|-------------|---------|-------------|--------|
| Whole-plots | 2 | ≥3 | Need df for whole-plot error |
| Sub-plots per WP | 2 | ≥3 | Need df for sub-plot error |
| Replicates | 1 | ≥2 | Increases power for WP effects |

**If insufficient:**
- Few whole-plots → Add replicates or more levels to hard factors
- Few sub-plots → Add center points or more levels to easy factors

### Blocking in Split-Plots

Blocking can be applied at the **whole-plot level**:

```python
# Divide whole-plots into blocks
n_whole_plots_per_block = n_whole_plots / n_blocks

for block in range(n_blocks):
    whole_plots_in_block = whole_plots[block_start:block_end]
    # Randomize whole-plots within this block
```

**Purpose:** Control for time, operator, or other nuisance variables affecting groups of whole-plots.

### Three-Level Nesting (Very Hard, Hard, Easy)

When very-hard factors are present:

```
VeryHardPlot (e.g., Production Line 1)
â"œâ"€â"€ WholePlot 1 (Line 1, Temp Low)
â"‚   â"œâ"€â"€ SubPlot 1 (Time Low)
â"‚   â""â"€â"€ SubPlot 2 (Time High)
â""â"€â"€ WholePlot 2 (Line 1, Temp High)
    â"œâ"€â"€ SubPlot 3 (Time Low)
    â""â"€â"€ SubPlot 4 (Time High)
```

**Error structure becomes:**
- Very-hard effects tested against very-hard-plot error
- Hard effects tested against whole-plot error
- Easy effects tested against sub-plot error

**Complexity increases**, but model is correct for the experimental restrictions.

## Practical Example: Baking Process

### Scenario

A bakery wants to optimize:
- **Oven temperature** (hard to change - requires heating/cooling oven)
- **Baking time** (easy to change - just set timer differently)

### Design

```python
factors = [
    Factor(name='Temperature', type=CONTINUOUS, 
           min=300, max=400, changeability=HARD),
    Factor(name='Time', type=CONTINUOUS,
           min=10, max=30, changeability=EASY)
]

design = generate_split_plot_design(
    factors=factors,
    n_replicates=2,        # Two days of baking
    n_center_points=2,     # Two center runs per temp
    randomize=True,
    seed=42
)
```

### Resulting Structure

```
Day 1 (Replicate 1):
  Whole-Plot 1: Temp = 300°F
    - Run 1: Time = 10 min
    - Run 2: Time = 30 min
    - Run 3: Time = 20 min (center)
    - Run 4: Time = 20 min (center)
  
  Whole-Plot 2: Temp = 400°F
    - Run 5: Time = 10 min
    - Run 6: Time = 30 min
    - Run 7: Time = 20 min (center)
    - Run 8: Time = 20 min (center)

Day 2 (Replicate 2):
  [Same structure, different randomization]
```

**Total runs:** 2 replicates × 2 temps × 4 times = 16 runs

**Efficiency gain:** Only 4 temperature changes instead of 16 (if fully randomized).

### Analysis Implications

```
Source                  df    F-test denominator
-------------------------------------------------
Temperature             1     Whole-plot error
Whole-plot error        2     (2 WP - 1 Temp - 1 rep)
Time                    3     Sub-plot error
Temp × Time             3     Sub-plot error  
Sub-plot error         6     (16 runs - 2 WP - 4 Time effects)
```

**Note:** Temperature has low df for error (only 2), so less powerful test. This is the **cost** of not randomizing temperature completely.

## Design Evaluation Metrics

### Balance

**Definition:** All whole-plots have the same number of sub-plots.

**Check:**
```python
subplot_counts = design.groupby('WholePlot').size()
is_balanced = subplot_counts.std() == 0
```

**Why it matters:** Unbalanced designs complicate ANOVA (need weighted analysis).

### Efficiency

**Relative efficiency** compared to completely randomized design (CRD):

$$
\text{RE} = \frac{\sigma^2_{\text{CRD}}}{\sigma^2_{\text{WP}} + \sigma^2_{\text{SP}}/n}
$$

where $n$ = sub-plots per whole-plot.

**Interpretation:**
- RE > 1: Split-plot more efficient (whole-plot error small)
- RE < 1: Split-plot less efficient (cost of restricted randomization)

**Practical note:** Split-plot is usually chosen for practical reasons (changeability constraints), not efficiency.

### Power Analysis

**Whole-plot effects** have lower power due to fewer df:

$$
\text{Power}_{\text{WP}} = f(\text{effect size}, \alpha, df_{\text{WP error}})
$$

**Sub-plot effects** have higher power:

$$
\text{Power}_{\text{SP}} = f(\text{effect size}, \alpha, df_{\text{SP error}})
$$

**Design strategy:** Assign **most important** factors to sub-plot level if possible.

## Common Mistakes to Avoid

### 1. Analyzing as Completely Randomized Design

**Wrong:**
```python
# Using single error term for all effects
model = ols('Response ~ Temp + Time + Temp:Time', data=design)
```

**Right:**
```python
# Using mixed model with random whole-plot effect
model = MixedLM(formula='Response ~ Temp + Time + Temp:Time',
                groups=design['WholePlot'],
                data=design)
```

### 2. Randomizing Across Whole-Plot Boundaries

**Wrong:**
```python
# Completely randomizing all runs
design = design.sample(frac=1).reset_index(drop=True)
```

**Right:**
```python
# Randomizing whole-plots, then sub-plots within each WP
for wp in design['WholePlot'].unique():
    wp_data = design[design['WholePlot'] == wp]
    randomized_wp = wp_data.sample(frac=1)
```

### 3. Adding Center Points at Whole-Plot Level

**Wrong:**
```python
# Setting whole-plot factors to center
center_run = {
    'Temperature': (300 + 400) / 2,  # This defeats split-plot!
    'Time': (10 + 30) / 2
}
```

**Right:**
```python
# Only sub-plot factors to center
center_run = {
    'Temperature': existing_wp_temp,  # Keep WP setting
    'Time': (10 + 30) / 2             # Only center easy factor
}
```

### 4. Too Few Whole-Plots

**Problem:** 2 whole-plots → only 1 df for whole-plot error → low power

**Solution:** Add replicates or more levels to hard factors

## Comparison with Other Designs

| Design Type | When to Use | Advantages | Disadvantages |
|-------------|-------------|------------|---------------|
| **Completely Randomized** | All factors easy to change | Simple analysis, maximum power | Impractical if factors costly |
| **Randomized Complete Block** | Nuisance variables | Controls time/operator effects | All factors must be in every block |
| **Split-Plot** | Factors have different changeability | Respects practical constraints | Reduced power for WP effects |
| **Strip-Plot** | Two sets of hard factors (rows and columns) | Good for 2+ hard factor sets | Complex analysis |

## Software Implementation Notes

### Factor Definition

Users specify changeability when defining factors:

```python
Factor(
    name='Temperature',
    type=FactorType.CONTINUOUS,
    min=100,
    max=200,
    changeability=Changeability.HARD  # ← Key attribute!
)
```

### Automatic Structure Inference

Algorithm automatically determines plot structure:

```python
if any VERY_HARD factors:
    structure = 'three_level'  # VH → H → E
elif any HARD factors:
    structure = 'two_level'    # H → E
else:
    raise Error('Need HARD factors for split-plot')
```

### Design Output

Generated design includes plot ID columns:

```
StdOrder  RunOrder  WholePlot  Temp  Time  Response
1         5         1          300   10    ...
2         7         1          300   30    ...
3         2         2          400   10    ...
4         1         2          400   30    ...
```

**Analysis workflow:**
1. Fit mixed model with `WholePlot` as random effect
2. Extract ANOVA table with correct error terms
3. Report effects with appropriate df and p-values

## Future Extensions

Potential enhancements:

1. **Strip-plot designs** (two crossed split-plot structures)
2. **Split-split-plot** (four-level nesting)
3. **Fractional split-plot** (when full factorial too large)
4. **Optimal split-plot** (D-optimal with changeability constraints)
5. **Unbalanced split-plots** (varying sub-plots per whole-plot)

## References

1. **Box, G. E. P., Hunter, W. G., & Hunter, J. S. (2005).** *Statistics for Experimenters: Design, Innovation, and Discovery* (2nd ed.). Wiley.
   - Chapter 13: Split-Plot Designs

2. **Montgomery, D. C. (2017).** *Design and Analysis of Experiments* (9th ed.). Wiley.
   - Chapter 14: Nested and Split-Plot Designs

3. **Goos, P., & Jones, B. (2011).** *Optimal Design of Experiments: A Case Study Approach*. Wiley.
   - Chapter 8: Split-Plot Designs

4. **Bingham, D., & Sitter, R. R. (1999).** "Minimum-aberration two-level fractional factorial split-plot designs." *Technometrics*, 41(1), 62-70.
   - Theory for fractional split-plots

5. **Littell, R. C., et al. (2006).** *SAS for Mixed Models* (2nd ed.). SAS Institute.
   - Practical guidance on mixed model analysis of split-plots