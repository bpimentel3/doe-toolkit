# Split-Plot Designs

## Purpose

Split-plot designs are experimental structures used when some factors are harder or more expensive to change than others. They provide proper randomization restrictions and error structure for analyzing experiments where complete randomization is impractical or impossible.

## When to Use Split-Plot Designs

### Common Scenarios

1. **Manufacturing processes** where machine settings are expensive to change
   - Example: oven temperature (hard) vs. baking time (easy)

2. **Agricultural experiments** with field-level and plot-level factors
   - Example: irrigation method (hard) vs. fertilizer amount (easy)

3. **Chemical processes** with batch and run-level variables
   - Example: batch temperature (hard) vs. catalyst amount (easy)

4. **Industrial experiments** with equipment constraints
   - Example: production line (very hard), die temperature (hard), cycle time (easy)

## Theoretical Foundation

### Factor Changeability Hierarchy

Split-plot designs recognize that factors have different levels of changeability:

| Changeability | Description | Examples | Nesting Level (design) |
|---------------|-------------|----------|------------------------|
| **VERY_HARD** | Rarely changed, expensive/time-consuming | Production line, equipment, operator | Whole-whole-plot |
| **HARD** | Changed infrequently, moderate cost | Temperature, pressure, batch | Whole-plot |
| **EASY** | Changed freely, low cost | Time, concentration, speed | Sub-plot |

### Plot Structure

A split-plot design creates a **nested hierarchy** of factor levels:

```
Replicate
   |- Whole-Whole-Plot 1   (very-hard factor settings)
   |    |- Whole-Plot 1.1   (hard factor settings)
   |    |    |- Sub-Plot 1.1.1  (easy factor settings)
   |    |    `- Sub-Plot 1.1.2  (different easy settings)
   |    `- Whole-Plot 1.2
   |         |- Sub-Plot 1.2.1
   |         `- Sub-Plot 1.2.2
   `- Whole-Whole-Plot 2
        `- ...
```

**Key principles:**
1. **Whole-plot (and whole-whole-plot) factors** remain constant within their plot
2. **Sub-plot factors** vary within each whole-plot
3. **Randomization** respects the nesting structure (restricted randomization)

### Why Split-Plot Matters: Error Structure

**In a completely randomized design**, all factors are tested against the same experimental error.

**In a split-plot design**, different factors are tested against different error terms:

| Factor Type | Error Term | Degrees of Freedom |
|-------------|------------|-------------------|
| Whole-plot effects | Whole-plot error | n_whole_plots − rank(whole-plot model) |
| Sub-plot effects | Sub-plot error | n_runs − n_whole_plots − sub-plot model parameters |
| WP × SP interactions | Sub-plot error | (interaction df) |

Exact degrees of freedom come from the two-strata fit (see the ANOVA
documentation and the worked example below). The **critical implication:** whole-plot
effects have **less precision** (fewer error df) than sub-plot effects. This is why
you assign factors strategically based on importance and changeability.

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

**Key distinction:** two error terms ($\gamma$ and $\epsilon$), not one!

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

**This is why proper structure matters!** Using the wrong error term invalidates F-tests.

## Design Generation Algorithm

`generate_split_plot_design(factors, n_replicates=1, n_center_points=0, n_blocks=1, randomize_whole_plots=True, randomize_sub_plots=True, seed=None)` in `src/core/split_plot.py`:

### Step 1: Classify Factors by Changeability

```python
very_hard_factors = [f for f in factors if f.changeability == ChangeabilityLevel.VERY_HARD]
hard_factors      = [f for f in factors if f.changeability == ChangeabilityLevel.HARD]
easy_factors      = [f for f in factors if f.changeability == ChangeabilityLevel.EASY]
```

If all factors are EASY, generation is rejected:

```
ValueError: All factors are EASY. Use generate_full_factorial() instead.
Split-plot requires at least one HARD or VERY_HARD factor.
```

### Step 2: Generate Factorial Combinations Per Level

```python
vh_combinations = product of all very-hard factor levels  (or [{}] if none)
h_combinations  = product of all hard factor levels       (or [{}] if none)
e_combinations  = product of all easy factor levels
```

Continuous factors contribute their `[min_value, max_value]` endpoints;
discrete-numeric and categorical factors contribute their declared `levels`.

### Step 3: Nest Combinations

Each (very-hard × hard) combination defines **one whole-plot**; its sub-plots are
the easy combinations:

```python
for vh_combo in vh_combinations:
    for h_combo in h_combinations:      # whole-plot containing vh+h settings
        for e_combo in e_combinations:  # one experimental run per easy combo
            run = {**vh_combo, **h_combo, **e_combo,
                   'Replicate': replicate, 'WholePlot': wp_id}
```

Minimum structure is enforced:
- at least **2 whole-plots** per replicate, and
- at least **2 sub-plots** per whole-plot (unless center points are used)

```
ValueError: Design has only 1 whole-plot(s). Need at least 2 whole-plots
for proper error estimation. Add more levels to HARD factors or use replicates.
```

### Step 4: Restricted Randomization

The run order is randomized **within the nesting structure** (never across plot
boundaries):

1. Per replicate: shuffle the order of whole-whole-plot combinations
   (when very-hard factors exist)
2. Within each whole-whole-plot: shuffle the order of whole-plots
3. Within each whole-plot: shuffle the order of sub-plots

`randomize_whole_plots` controls steps 1–2; `randomize_sub_plots` controls
step 3. The RNG is `numpy.random.default_rng(seed)` — reproducibility requires
passing a `seed` (results are not reproducible without one).

### Step 5: Add Center Points (Sub-Plot Level Only)

Center points, if requested, are added **only at the sub-plot level**:

- whole-plot factor values stay at their whole-plot settings
- continuous/discrete sub-plot factors go to their midpoint
  `(min_value + max_value) / 2`
- categorical sub-plot factors go to their first declared level

**Rationale:** whole-plot factors cannot be easily changed to center values
(defeats the purpose of split-plot).

### Step 6: Add Replicates

Each replicate regenerates the nested structure with **fresh whole-plot IDs**,
so replicate effects are absorbed into the whole-plot stratum during analysis
(independently seeded whole-plots give df for whole-plot error).

## Worked Example: Baking Process

### Scenario

A bakery wants to optimize:
- **Oven temperature** (hard to change — requires heating/cooling the oven)
- **Baking time** (easy to change — just set a different timer)

### Design

```python
from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.split_plot import generate_split_plot_design

factors = [
    Factor(name="Temperature", factor_type=FactorType.CONTINUOUS,
           levels=[300, 400], changeability=ChangeabilityLevel.HARD),
    Factor(name="Time", factor_type=FactorType.CONTINUOUS,
           levels=[10, 30], changeability=ChangeabilityLevel.EASY),
]

design = generate_split_plot_design(
    factors=factors,
    n_replicates=3,        # Three days of baking
    seed=42,               # Reproducible (see note below)
)
```

**Structure:** 3 replicates × 2 temperatures × 2 times = **12 runs**, with 6
whole-plots (one per day-temperature). The generated design (seed 42):

```
 StdOrder  RunOrder  Replicate  WholePlot  Temperature  Time
        1         1          1          1          400    10
        2         2          1          1          400    30
        3         3          1          2          300    10
        4         4          1          2          300    30
        5         5          2          3          300    30
        6         6          2          3          300    10
        7         7          2          4          400    30
        8         8          2          4          400    10
        9         9          3          5          300    30
       10        10          3          5          300    10
       11        11          3          6          400    10
       12        12          3          6          400    30
```

Temperature is constant within every whole-plot; Time varies within each.
`StdOrder` is the standard (unrandomized) index; `RunOrder` is the restricted
execution order from the seeded randomization. A different seed reorders the
runs; both columns update accordingly. The example figures replicate the
current implementation defaults (`n_replicates=1`, `n_center_points=0`,
`randomize_whole_plots=True`, `randomize_sub_plots=True`) and may evolve in
future releases.

**Efficiency gain:** only **6 temperature changes** (one per whole-plot) instead
of up to 12 for a fully randomized design.

### Analysis (Two-Strata)

Fit with `ANOVAAnalysis` — the split-plot structure is auto-detected from the
factor changeabilities and the analysis uses the two-strata model (whole-plot
effects vs. whole-plot error; sub-plot effects vs. sub-plot error):

```
                  df          F-test denominator
Temperature        1          Whole-plot error
WholePlot Error    4          6 whole-plots - 2 params (intercept + Temperature)
Time               1          Sub-plot error
Temperature*Time   1          Sub-plot error
SubPlot Error      4          12 runs - 6 whole-plots - 2 sub-plot params
```

Temperature has fewer error df (4) than Time — the statistical **cost** of not
randomizing temperature completely.

## Blocking in Split-Plots

Blocking is supported via `n_blocks`. Blocks are assigned **post-hoc**: the
whole-plots (in their generated order) are divided into contiguous segments of
roughly equal size, each labeled with a block number. There is **no
within-block re-randomization step** — the design is generated first, then the
`Block` column is attached.

```
ValueError: Cannot have more blocks (<n>) than whole-plots (<m>)
```

Note that the shipped split-plot **analysis** does not currently include the
`Block` column as a model term (a documented limitation of the two-strata fit).

## Three-Level Nesting (Very Hard, Hard, Easy)

When very-hard factors are present, the **design generator** produces a
three-level nesting:

```
VeryHardPlot (e.g., Production Line 1)
  |- WholePlot 1 (Line 1, Temp Low)
  |    |- SubPlot 1 (Time Low)
  |    `- SubPlot 2 (Time High)
  `- WholePlot 2 (Line 1, Temp High)
       |- SubPlot 3 (Time Low)
       `- SubPlot 4 (Time High)
```

However, the **analysis** collapses very-hard and hard factors into a **single
whole-plot stratum**: there are still only **two error terms**. Very-hard and
hard main effects (and their interactions with each other) are both tested
against the same whole-plot error; any term involving an easy factor is tested
against the sub-plot error.

## Design Evaluation Metrics

### Balance

**Definition:** all whole-plots have the same number of sub-plots.

**Check:**
```python
subplot_counts = design.design.groupby("WholePlot").size()
is_balanced = subplot_counts.std() == 0
```

Balanced whole-plots are the norm for the factorial nesting this generator
produces (center points are added uniformly per whole-plot).

### Relative Efficiency and Power

Relative efficiency versus a completely randomized design, and formal power
analysis for whole-plot effects — described in the classical literature — are
**not implemented** in this app. The practical guidance stands:

- Whole-plot effects have lower power (fewer error df) than sub-plot effects.
- Assign the **most important** factors to the sub-plot level when possible.

## Common Mistakes to Avoid

### 1. Analyzing as a Completely Randomized Design

**Wrong:** a single error term for everything:

```python
model = ols("Response ~ Temp + Time + Temp:Time", data=design)  # one MS_error
```

**Right (this app):** the two-strata fit, which tests terms in their proper
stratum:

```python
analysis = ANOVAAnalysis(design.design, response, factors)  # auto-detected
results = analysis.fit(["Temperature", "Time", "Temperature*Time"])
```

A mixed model with a random whole-plot intercept is **not** the correct tool
here either: hard-to-change factors do not vary within a whole-plot, so the
design matrix is singular and `MixedLM` raises a `LinAlgError`.

### 2. Randomizing Across Whole-Plot Boundaries

**Wrong:** completely randomizing all 12 runs (breaks the nesting).

**Right:** restricted randomization is built into the generator —
`randomize_whole_plots=True` reorders whole-plots (and whole-whole-plots) and
`randomize_sub_plots=True` shuffles sub-plots within each whole-plot. Never
shuffle across plot boundaries.

### 3. Adding Center Points at the Whole-Plot Level

**Wrong:** setting whole-plot factors to their center (defeats split-plot).

**Right:** the generator keeps whole-plot settings and centers only the
sub-plot factors, so the extra runs re-measure the same whole-plot conditions.

### 4. Too Few Whole-Plots

**Problem:** generating with fewer than 2 whole-plots per replicate leaves no
whole-plot error df.

**Solution:** the generator raises an error in this case — add levels to the
hard factors or use replicates (`n_replicates≥2` is recommended for ≥3
whole-plots per anti-pattern diagnosis; `evaluate_split_plot_design` warns when
a design has under 3 whole-plots or under 3 sub-plots per whole-plot).

## Comparison with Other Designs

| Design Type | When to Use | Advantages | Disadvantages |
|-------------|-------------|------------|---------------|
| **Completely Randomized** | All factors easy to change | Simple analysis, maximum power | Impractical if factors costly |
| **Randomized Complete Block** | Nuisance variables | Controls time/operator effects | All factors must appear in every block |
| **Split-Plot** | Factors have different changeability | Respects practical constraints | Reduced power for WP effects |
| **Strip-Plot** | Two sets of hard factors (rows and columns) | Good for 2+ hard factor sets | Complex analysis |

## Software Implementation Notes

### Factor Definition

Factors use the standard keyword form (note the `factor_type=`, `levels=` and
`changeability=` arguments):

```python
Factor(
    name="Temperature",
    factor_type=FactorType.CONTINUOUS,
    levels=[100, 200],
    changeability=ChangeabilityLevel.HARD,  # key attribute!
)
```

### Automatic Structure Inference

`detect_split_plot_structure()` (`src/core/analysis.py`) classifies from the
factor changeabilities:

```python
any VERY_HARD or HARD factor  -> is_split_plot = True
whole_plot_factors            -> very_hard + hard factor names
sub_plot_factors              -> easy factor names
whole_plot_column             -> 'WholePlot' (must be in the design)
```

### Design Output

The returned `SplitPlotDesign` has a `design` DataFrame with plot-ID and order
columns followed by the factor columns:

```
StdOrder  RunOrder  [Block]  Replicate  [VeryHardPlot]  WholePlot  Temp  Time  ...
```

and a parallel `design_coded` DataFrame with continuous factors on the coded
[-1, 1] scale (natural values remain in `design`). Metadata: `n_runs`,
`n_whole_plots`, `n_sub_plots_per_whole_plot`, `whole_plot_factors`,
`sub_plot_factors`, `has_very_hard_factors`, `n_blocks`.

**Analysis workflow:**
1. Generate the design with `generate_split_plot_design(...)`
2. Collect the response per run
3. Fit with `ANOVAAnalysis(design.design, response, factors)` and `.fit(terms)`
4. Read the two-strata ANOVA table: each row is tagged with its `Stratum`
   (`Whole-Plot` or `Sub-Plot`) and tested against the appropriate error term

## Future Extensions

Potential enhancements:

1. **Strip-plot designs** (two crossed split-plot structures)
2. **Split-split-plot** (four-level nesting)
3. **Block terms in the split-plot ANOVA model** (currently data-only)
4. **Fractional split-plot** (when a full factorial is too large)
5. **Optimal split-plot** (D-optimal with changeability constraints)
6. **Unbalanced split-plots** (varying sub-plots per whole-plot)
7. **Relative-efficiency / power analysis** (theory documented above; not yet implemented)

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
   - Practical guidance on mixed-model analysis of split-plots
