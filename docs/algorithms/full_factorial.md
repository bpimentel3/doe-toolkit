# Full Factorial Design Algorithm

## Overview

A full factorial design systematically explores all possible combinations of factor levels. For k factors with levels L₁, L₂, ..., Lₖ, the design contains:

**Total runs = L₁ × L₂ × ... × Lₖ**

## Mathematical Background

### Cartesian Product

The full factorial design is the Cartesian product of all factor level sets:

```
D = F₁ × F₂ × ... × Fₖ
```

Where:
- D is the design matrix
- Fᵢ is the set of levels for factor i
- × denotes Cartesian product

**Example:** For 2 factors with 2 levels each:
- F₁ = {-1, +1} (Temperature: Low, High)
- F₂ = {-1, +1} (Pressure: Low, High)
- D = {(-1,-1), (-1,+1), (+1,-1), (+1,+1)}

This gives 2² = 4 experimental runs.

### Coded Levels

For continuous factors, we use coded levels to standardize the design space:

**Coding transformation:**
```
x_coded = (x_actual - x_center) / (x_range / 2)
```

Where:
- x_actual is the actual value
- x_center = (x_min + x_max) / 2
- x_range = x_max - x_min

**Standard coded levels:**
- **-1**: Low level (x_min)
- **0**: Center level ((x_min + x_max) / 2)
- **+1**: High level (x_max)

**Example:**
- Temperature range: [150°C, 200°C]
- Center: (150 + 200) / 2 = 175°C
- Half-range: (200 - 150) / 2 = 25°C

Coding:
- x = 150 → x_coded = (150 - 175) / 25 = -1
- x = 175 → x_coded = (175 - 175) / 25 = 0
- x = 200 → x_coded = (200 - 175) / 25 = +1

**Decoding (reverse transformation):**
```
x_actual = x_center + x_coded × (x_range / 2)
```

### Design Properties

**Orthogonality:**
Full factorial designs are orthogonal - the dot product of any two factor columns equals zero (when properly coded). This means:
- Factor effects are independent
- No confounding between main effects
- Simple, unbiased effect estimates

**Balance:**
Each factor level appears an equal number of times:
```
Appearances of level i = (Total runs) / (Levels of factor)
```

**Efficiency:**
Full factorial designs are D-optimal for estimating all main effects and interactions up to order k.

## Implementation Details

### Algorithm Steps

1. **Factor Level Extraction**
   ```
   For each factor:
     - Continuous: Use coded levels [-1, +1]
     - Discrete Numeric: Use actual values
     - Categorical: Use level labels
   ```

2. **Cartesian Product Generation**
   ```python
   combinations = itertools.product(*factor_levels)
   ```
   This efficiently generates all combinations without explicit nested loops.

3. **Center Points** (optional)
   - Added for continuous factors only
   - Coded as 0 for all continuous factors
   - Provides estimate of pure error
   - Allows testing for curvature (quadratic effects)

4. **Randomization** (optional)
   - Protects against time-related trends
   - Standard order preserved in 'StdOrder' column
   - Randomized order shown in 'RunOrder' column

5. **Blocking** (optional)
   - Divides runs into groups to account for nuisance variation
   - Runs assigned to blocks either:
     - Sequentially (if randomized)
     - Interleaved (if not randomized)

### Computational Complexity

**Time Complexity:**
- Generation: O(∏Lᵢ) where Lᵢ is the number of levels for factor i
- This is optimal as we must generate all combinations

**Space Complexity:**
- O(n × k) where n is total runs, k is number of factors
- Storage for the design matrix

### Example Calculation

**Problem:** Design a 2³ factorial experiment

**Factors:**
- A: Temperature [150, 200]°C
- B: Pressure [50, 100] psi  
- C: Time [10, 20] minutes

**Solution:**

1. **Code factors:**
   - A: {-1, +1} for {150, 200}
   - B: {-1, +1} for {50, 100}
   - C: {-1, +1} for {10, 20}

2. **Generate combinations:**
   ```
   Run  A   B   C
   1   -1  -1  -1
   2   +1  -1  -1
   3   -1  +1  -1
   4   +1  +1  -1
   5   -1  -1  +1
   6   +1  -1  +1
   7   -1  +1  +1
   8   +1  +1  +1
   ```

3. **Total runs:** 2³ = 8

4. **If adding 3 center points:** 8 + 3 = 11 total runs

## Advantages and Limitations

### Advantages

1. **Complete Information**
   - All main effects estimable
   - All interactions estimable
   - No confounding

2. **Optimal for Interactions**
   - Best design for detecting and estimating interactions
   - Full interaction hierarchy

3. **Flexible Analysis**
   - Can fit any model up to order k
   - Can pool non-significant effects for error estimation

4. **Robust**
   - No assumptions about which effects are important
   - Protects against model mis-specification

### Limitations

1. **Run Count**
   - Grows exponentially with number of factors: 2^k
   - Becomes impractical for k > 6 or 7
   - Example: 2⁸ = 256 runs (too many for most budgets)

2. **Efficiency for Screening**
   - Estimates many effects that may be negligible
   - Fractional factorial more efficient if only main effects matter

3. **Resource Intensive**
   - May waste resources on unimportant effects
   - Consider fractional factorial or screening design for k > 5

## When to Use Full Factorial

**Use full factorial when:**
- Number of factors is small (k ≤ 5)
- Interactions are expected to be important
- Complete characterization needed
- Resources permit
- Need model flexibility

**Consider alternatives when:**
- Many factors (k > 6): Use fractional factorial or screening design
- Only main effects matter: Use Plackett-Burman
- Non-linear relationships: Use response surface design
- Very limited budget: Use definitive screening design

## Extensions

### Replicated Designs

Adding replicates (complete repeats of the design):
- Provides better estimate of error
- Increases power to detect effects
- Total runs = (Base runs) × (Number of replicates)

### Center Points

Adding center points:
- Tests for curvature without full quadratic model
- Provides pure error estimate
- Efficient way to check model adequacy
- Typical: 3-5 center points

### Blocking

Dividing runs into blocks:
- Controls for known sources of variation
- Blocks should be as homogeneous as possible
- Analysis accounts for block effects
- Typically 2-4 blocks

## References

1. Box, G. E. P., Hunter, J. S., and Hunter, W. G. (2005). *Statistics for Experimenters: Design, Innovation, and Discovery*, 2nd Edition. Wiley.

2. Montgomery, D. C. (2017). *Design and Analysis of Experiments*, 9th Edition. Wiley.

3. Wu, C. F. J., and Hamada, M. S. (2009). *Experiments: Planning, Analysis, and Optimization*, 2nd Edition. Wiley.

4. Myers, R. H., Montgomery, D. C., and Anderson-Cook, C. M. (2016). *Response Surface Methodology: Process and Product Optimization Using Designed Experiments*, 4th Edition. Wiley.

## Implementation Notes

The implementation uses Python's `itertools.product()` for efficient generation of the Cartesian product. This is more efficient than nested loops and scales well to many factors.

For randomization, we use NumPy's random number generator with optional seed for reproducibility. This is important for:
- Validating results
- Debugging
- Comparing designs

The blocking algorithm uses a greedy assignment strategy that balances runs across blocks as evenly as possible when the number of runs is not perfectly divisible by the number of blocks.