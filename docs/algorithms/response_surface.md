# Response Surface Methods

## Overview

Response surface methodology (RSM) is used to explore relationships between multiple factors and one or more responses, particularly when:
- Curvature (quadratic effects) is expected
- Finding optimal factor settings is the goal
- Building predictive models for optimization

Two most common response surface designs:
1. **Central Composite Design (CCD)**
2. **Box-Behnken Design (BBD)**

Both allow fitting second-order (quadratic) models of the form:

```
y = β₀ + Σβᵢxᵢ + Σβᵢᵢxᵢ² + ΣΣβᵢⱼxᵢxⱼ + ε
```

Where:
- β₀ = intercept
- βᵢ = linear (main) effects
- βᵢᵢ = quadratic effects
- βᵢⱼ = interaction effects
- ε = error

## Central Composite Design (CCD)

### Design Structure

A CCD consists of three types of points:

**1. Factorial Points (2^k or 2^(k-p))**
- Corner points of the design cube
- Coded as ±1 for each factor
- Can use fractional factorial for efficiency

**2. Axial (Star) Points (2k)**
- Points along each factor axis
- Coded as ±α for one factor, 0 for others
- α (alpha) determines design properties

**3. Center Points (replicated)**
- All factors at center (coded as 0)
- Provides pure error estimate
- Tests for curvature

### Mathematical Representation

**Coded factor space:**
```
Factorial points: xᵢ ∈ {-1, +1}
Axial points: xᵢ ∈ {-α, 0, +α}
Center points: xᵢ = 0 for all i
```

### Alpha (α) Selection

Alpha determines the distance of axial points from the center. Three common choices:

**1. Rotatable Design (α = (n_f)^(1/4))**

Makes prediction variance constant at all points equidistant from center.

**Formula:**
```
α = (n_factorial)^(1/4)
```

**Example for k=3:**
```
n_factorial = 2³ = 8
α = 8^(1/4) = 1.682
```

**Properties:**
- Prediction variance depends only on distance from center
- Optimal for exploring unknown response surface
- Most commonly used

**2. Orthogonal Design**

Makes design matrix columns orthogonal (uncorrelated).

**Approximate formula:**
```
α² = √[nf(√(nf + nc) + √(na + nc)) / (2k)]
```

Where:
- nf = number of factorial points
- na = number of axial points
- nc = number of center points

**Properties:**
- Simplifies calculation of regression coefficients
- Reduces correlation between effects
- Good for analysis

**3. Face-Centered Design (α = 1)**

Places axial points on the faces of the design cube.

**Properties:**
- All design points within original factor ranges
- No extrapolation needed
- Useful when factors can't exceed specified ranges
- Less efficient for predicting than rotatable

### Example: 3-Factor Rotatable CCD

**Design:**
```
Point Type    A    B    C
─────────────────────────
Factorial    -1   -1   -1
Factorial    +1   -1   -1
Factorial    -1   +1   -1
Factorial    +1   +1   -1
Factorial    -1   -1   +1
Factorial    +1   -1   +1
Factorial    -1   +1   +1
Factorial    +1   +1   +1
Axial      -1.68   0    0
Axial      +1.68   0    0
Axial         0 -1.68   0
Axial         0 +1.68   0
Axial         0    0 -1.68
Axial         0    0 +1.68
Center        0    0    0
Center        0    0    0
...
Center        0    0    0   (6 total)
```

**Total runs:** 8 + 6 + 6 = 20

### Run Count Formula

```
N = 2^k + 2k + nc

Or for fractional factorial:
N = 2^(k-p) + 2k + nc
```

Where:
- k = number of factors
- p = fraction (for 1/2^p fraction)
- nc = number of center points

**Examples:**

| k | Factorial | Axial | Center | Total |
|---|-----------|-------|--------|-------|
| 2 | 4 | 4 | 5 | 13 |
| 3 | 8 | 6 | 6 | 20 |
| 4 | 16 | 8 | 7 | 31 |
| 5 | 32 | 10 | 10 | 52 |
| 5* | 16 | 10 | 10 | 36 |

*Using 1/2 fraction for factorial portion

## Box-Behnken Design (BBD)

### Design Structure

Box-Behnken designs:
- Place points at **midpoints of edges** of design cube
- Have **no corner points** (no extreme combinations)
- Require **3 or more factors**
- More economical than CCD for 3-4 factors

### Mathematical Representation

For each pair of factors (i, j):
- Create 2² = 4 combinations: (±1, ±1)
- Set all other factors to 0

**Example for k=3:**
```
For pair (A,B) with C=0:
  A   B   C
 -1  -1   0
 +1  -1   0
 -1  +1   0
 +1  +1   0

For pair (A,C) with B=0:
  A   B   C
 -1   0  -1
 +1   0  -1
 -1   0  +1
 +1   0  +1

For pair (B,C) with A=0:
  A   B   C
  0  -1  -1
  0  +1  -1
  0  -1  +1
  0  +1  +1
```

Plus center points (typically 3-5).

### Run Count Formula

```
N = 2k(k-1) + nc
```

**Examples:**

| k | Formula | Factorial | Center | Total |
|---|---------|-----------|--------|-------|
| 3 | 2×3×2 | 12 | 3 | 15 |
| 4 | 2×4×3 | 24 | 3 | 27 |
| 5 | 2×5×4 | 40 | 3 | 43 |
| 6 | 2×6×5 | 60 | 3 | 63 |
| 7 | 2×7×6 | 84 | 3 | 87 |

### Key Properties

**1. Spherical Design**
- Points approximately equidistant from center
- Good prediction properties in all directions

**2. No Extreme Corners**
- Never tests all factors at high/low simultaneously
- Safer when extreme combinations risky
- May miss important regions if true optimum at corner

**3. Three Levels per Factor**
- Each factor tested at: low (-1), middle (0), high (+1)
- Simpler than CCD which uses 5 levels (±α, ±1, 0)

## Design Comparison

### Run Efficiency

For small number of factors, BBD more efficient:

| Factors | CCD Runs | BBD Runs | BBD Advantage |
|---------|----------|----------|---------------|
| 3 | 20 | 15 | 25% fewer |
| 4 | 31 | 27 | 13% fewer |
| 5 | 52 | 43 | 17% fewer |
| 6 | 90 | 63 | 30% fewer |

For 5+ factors, CCD with fractional factorial can be more efficient.

### Prediction Variance

**CCD (Rotatable):**
- Constant variance at all points equidistant from center
- Optimal for exploring unknown surface

**BBD:**
- Near-spherical variance properties
- Slightly worse at extreme points
- Still very good overall

### Practical Considerations

**Use CCD when:**
- Want rotatability properties
- Need to test extreme corners
- Have 2-5 factors
- Can afford more runs
- Can set factors outside [-1, +1] range

**Use BBD when:**
- Want to avoid extreme corners
- Have 3-4 factors (most efficient range)
- Budget limited
- Factors can't exceed specified ranges
- Prefer 3-level design

## Implementation Details

### CCD Algorithm

**Step 1: Generate Factorial Points**
```python
For full factorial:
  Generate all 2^k combinations of ±1

For fractional factorial:
  Use standard generators to get 2^(k-p) runs
```

**Step 2: Generate Axial Points**
```python
For each factor i = 1 to k:
  Create point with xᵢ = +α, all others = 0
  Create point with xᵢ = -α, all others = 0
```

**Step 3: Generate Center Points**
```python
Replicate center point (all xᵢ = 0) nc times
```

**Step 4: Combine and Randomize**
```python
Combine all points
Add standard order
Randomize if requested
Add run order
```

### BBD Algorithm

**Step 1: Generate Edge-Midpoint Runs**
```python
For each pair of factors (i,j):
  For xᵢ in {-1, +1}:
    For xⱼ in {-1, +1}:
      Create run with:
        - xᵢ and xⱼ at specified values
        - All other factors at 0
```

**Step 2: Generate Center Points**
```python
Replicate center point nc times
```

**Step 3: Combine and Randomize**
```python
Combine all points
Add standard order
Randomize if requested
Add run order
```

### Computational Complexity

**CCD:**
- Time: O(2^k + k + nc)
- Space: O((2^k + 2k + nc) × k)

**BBD:**
- Time: O(k² + nc)
- Space: O((2k(k-1) + nc) × k)

BBD generation is faster for large k.

## Decoding to Actual Values

Both CCD and BBD use coded levels (-1, 0, +1, ±α). Convert to actual values:

**Formula:**
```
x_actual = x_center + x_coded × (x_range / 2)
```

Where:
- x_center = (x_min + x_max) / 2
- x_range = x_max - x_min

**Example:**
```
Factor: Temperature
Range: [150, 200]°C
Center: 175°C
Half-range: 25°C

Decoding:
  -1.68 → 175 + (-1.68)(25) = 133°C
  -1.00 → 175 + (-1.00)(25) = 150°C
   0.00 → 175 + (0.00)(25) = 175°C
  +1.00 → 175 + (+1.00)(25) = 200°C
  +1.68 → 175 + (+1.68)(25) = 217°C
```

## Model Fitting

### Full Quadratic Model

After running the experiment, fit:

```
y = β₀ + Σβᵢxᵢ + Σβᵢᵢxᵢ² + ΣΣβᵢⱼxᵢxⱼ + ε
```

**Number of parameters:**
```
p = 1 + k + k + k(k-1)/2
  = 1 + 2k + k(k-1)/2
  = (k+1)(k+2)/2
```

**Examples:**

| Factors | Parameters | Min Runs (saturated) |
|---------|------------|---------------------|
| 2 | 6 | 6 |
| 3 | 10 | 10 |
| 4 | 15 | 15 |
| 5 | 21 | 21 |

Response surface designs have sufficient runs to fit full quadratic model with degrees of freedom for error estimation.

### Model Adequacy Checks

1. **Lack of Fit Test**
   - Use center point replicates for pure error
   - Compare lack of fit to pure error
   - Significant LOF → need higher-order model

2. **Residual Plots**
   - Normal probability plot
   - Residuals vs fitted values
   - Residuals vs factor levels

3. **R² and Adjusted R²**
   - R² > 0.80 generally good
   - Adjusted R² accounts for number of terms

4. **Prediction Variance**
   - Check variance across design space
   - Should be relatively uniform

## Optimization

After fitting model, find optimum:

**For single response:**
```
Maximize/Minimize: ŷ(x)
Subject to: x_min ≤ xᵢ ≤ x_max
```

**Methods:**
1. **Steepest Ascent/Descent**
   - Follow gradient direction
   - Sequential experiments

2. **Canonical Analysis**
   - Transform to principal axes
   - Identify stationary point type (max, min, saddle)

3. **Contour Plots**
   - Visualize response surface
   - Identify optimal region

4. **Desirability Functions** (multiple responses)
   - Convert each response to 0-1 scale
   - Maximize overall desirability

## Advantages and Limitations

### Advantages

1. **Curvature Estimation**
   - Can model quadratic effects
   - Find optimal factor settings
   - More realistic than linear models

2. **Efficiency**
   - Fewer runs than full 3^k factorial
   - CCD can use fractional factorial portion
   - BBD economical for 3-4 factors

3. **Sequential Strategy**
   - Can augment factorial design with axial points
   - Build from screening to optimization

4. **Flexibility**
   - CCD alpha can be adjusted
   - BBD avoids extreme corners

### Limitations

1. **Curvature Assumption**
   - Assumes quadratic model adequate
   - May need higher-order terms

2. **Factor Limits**
   - Practical for 2-7 factors
   - More than 7 factors → too many runs

3. **Continuous Factors Only**
   - Not suitable for categorical factors
   - Use factorial designs instead

4. **Local Optimum**
   - Finds optimum near design center
   - May miss global optimum elsewhere

## When to Use Response Surface Methods

**Use RSM when:**
- Need to model curvature
- Finding optimum is the goal
- Have 2-7 factors
- Screening already done (know important factors)
- Can run sequential experiments

**Choose CCD when:**
- Want rotatability
- 2-5 factors
- Can test outside original ranges (for α > 1)
- Building from factorial design

**Choose BBD when:**
- Want to avoid extremes
- 3-4 factors (most efficient)
- Cannot exceed factor ranges
- Prefer 3-level design

## References

1. Box, G. E. P., and Wilson, K. B. (1951). On the Experimental Attainment of Optimum Conditions. *Journal of the Royal Statistical Society, Series B*, 13, 1-45.

2. Box, G. E. P., and Behnken, D. W. (1960). Some New Three Level Designs for the Study of Quantitative Variables. *Technometrics*, 2, 455-475.

3. Myers, R. H., Montgomery, D. C., and Anderson-Cook, C. M. (2016). *Response Surface Methodology: Process and Product Optimization Using Designed Experiments*, 4th Edition. Wiley.

4. Montgomery, D. C. (2017). *Design and Analysis of Experiments*, 9th Edition. Wiley.

5. Box, G. E. P., Hunter, J. S., and Hunter, W. G. (2005). *Statistics for Experimenters: Design, Innovation, and Discovery*, 2nd Edition. Wiley.