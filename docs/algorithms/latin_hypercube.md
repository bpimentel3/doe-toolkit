# Latin Hypercube Sampling (LHS)

## Purpose

Latin Hypercube Sampling is a space-filling design technique that ensures good coverage of the experimental region. It's particularly useful for:

1. **Initial exploration** when little is known about the system
2. **Computer experiments** where running the experiment is cheap
3. **Complex response surfaces** where traditional factorial designs might miss important features
4. **High-dimensional problems** where full factorial designs become prohibitively large

## Mathematical Foundation

### Basic LHS Algorithm

For `n` runs and `k` factors:

1. **Divide each factor's range into `n` equal intervals**
2. **Sample once from each interval** for each factor
3. **Randomly permute** the samples for each factor independently

This ensures that each factor has exactly one sample point in each of its `n` intervals, providing better stratification than pure random sampling.

### Space-Filling Criteria

After generating a basic LHS, we can optimize for additional properties:

#### Maximin Criterion

**Objective:** Maximize the minimum distance between any two design points.

$$
\text{Maximize: } \min_{i \neq j} d(x_i, x_j)
$$

where $d(x_i, x_j)$ is the Euclidean distance between points $i$ and $j$.

**Why this matters:** Ensures no points cluster together, maximizing space coverage.

**Implementation:**
```python
from scipy.spatial.distance import pdist

# For design matrix X (n_runs × n_factors)
distances = pdist(X)  # All pairwise distances
maximin_score = np.min(distances)  # Minimum distance
```

Generate multiple candidate LHS designs and select the one with the largest minimum distance.

#### Correlation Criterion

**Objective:** Minimize correlation between factors.

$$
\text{Minimize: } \max_{i \neq j} |\rho(X_i, X_j)|
$$

where $\rho(X_i, X_j)$ is the Pearson correlation between factors $i$ and $j$.

**Why this matters:** Ensures factors are as independent as possible, simplifying interpretation of effects.

**Implementation:**
```python
corr_matrix = np.corrcoef(X.T)
# Extract upper triangle (exclude diagonal)
correlations = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
correlation_score = -np.max(np.abs(correlations))  # Negative because we minimize
```

### Factor Type Handling

#### Continuous Factors

Standard LHS procedure:
1. Generate uniform random samples in [0, 1]
2. Scale to [min, max] range
3. Each interval gets exactly one sample

#### Discrete Numeric Factors

Modified procedure:
1. Generate continuous LHS samples in [min, max]
2. Round each sample to nearest allowed discrete level
3. This may result in some levels appearing more than once

**Example:** For levels [100, 150, 200] with n=5 runs:
- Continuous LHS might give: [110, 135, 165, 185, 195]
- After rounding: [100, 150, 150, 200, 200]

**Trade-off:** Perfect Latin property is sacrificed to respect discrete constraints.

#### Categorical Factors

Use **stratified random sampling** instead of LHS:
1. Divide `n` runs as evenly as possible across `m` levels
2. Each level appears `floor(n/m)` or `ceil(n/m)` times
3. Randomly assign levels to runs

**Example:** For levels ['A', 'B', 'C'] with n=10 runs:
- 10 ÷ 3 = 3 remainder 1
- Distribution: A appears 4 times, B appears 3 times, C appears 3 times
- Order randomized

**Rationale:** Distance between categorical levels is undefined, so LHS doesn't apply. Stratification ensures balanced representation.

## Design Augmentation

### Problem

After running initial experiments, you want to add more runs while maintaining good space-filling properties.

### Algorithm

1. **Generate candidate sets** of additional runs (e.g., 10 candidates)
2. **For each candidate:**
   - Combine with existing design
   - Compute space-filling criterion
3. **Select candidate** with best combined criterion value

### Why Not Simple Extension?

Naively extending the LHS (adding runs to the end) can create clustering:
- Original design has n₁ runs
- New design should have n₁ + n₂ runs
- Simply generating a separate n₂-run LHS and appending may create clusters

**Better approach:** Evaluate candidates based on combined design quality.

## Implementation Details

### Candidate Generation

**Why generate multiple candidates?**

LHS has inherent randomness (permutation of intervals). Different permutations yield different space-filling properties. By generating multiple (default: 10) and selecting the best, we improve design quality.

**Computational cost:**
- Generating 10 candidates: O(10 × n × k)
- Computing distances: O(n² × k) per candidate
- Total: O(10 × n² × k)

For typical DOE problems (n < 100, k < 20), this is negligible (<1 second).

### Scaling and Coding

**Actual levels** (what user sees):
```
Temperature: [100, 125, 150, 175, 200] °C
```

**Coded levels** (internal computation):
```
Temperature: [-1.0, -0.5, 0.0, 0.5, 1.0]
```

**Why code?**
1. Numerical stability (all factors on same scale)
2. Distance metrics work correctly (no unit bias)
3. Consistent with other DOE designs (factorial, response surface)

**Coding formula:**
$$
x_{\text{coded}} = \frac{x_{\text{actual}} - x_{\text{center}}}{x_{\text{range}} / 2}
$$

where:
- $x_{\text{center}} = \frac{x_{\text{max}} + x_{\text{min}}}{2}$
- $x_{\text{range}} = x_{\text{max}} - x_{\text{min}}$

## Comparison with Other Designs

| Design Type | Structure | Use Case | Advantages | Disadvantages |
|-------------|-----------|----------|------------|---------------|
| **Full Factorial** | Complete grid | All factor combinations | Estimates all interactions | Exponential growth in runs |
| **Fractional Factorial** | Subset of grid | Main effects + key interactions | Fewer runs than full factorial | Some effects confounded |
| **Response Surface** | Factorial + center + axial | Quadratic models | Efficient for optimization | Assumes smooth response |
| **Latin Hypercube** | Space-filling | Exploration, complex surfaces | Good coverage, flexible | No structure for interactions |

**When to use LHS:**
- ✅ Initial screening with many factors
- ✅ Computer experiments (cheap to run)
- ✅ Highly nonlinear or discontinuous responses
- ✅ No prior knowledge of important factors

**When NOT to use LHS:**
- ❌ Need to estimate specific interactions
- ❌ Want to fit polynomial models
- ❌ Runs are expensive (use optimal designs instead)
- ❌ Need blocking or split-plot structure

## Validation and Quality Checks

### Latin Property Check

For each factor, verify that each of `n` intervals contains exactly one sample:

```python
n_runs = 20
intervals = np.linspace(min_value, max_value, n_runs + 1)
counts = np.histogram(design['factor'], bins=intervals)[0]
assert all(counts == 1)  # Each interval has exactly one point
```

### Space-Filling Quality

**Minimum distance metric:**
```python
from scipy.spatial.distance import pdist

distances = pdist(design_matrix)
min_dist = np.min(distances)
max_dist = np.max(distances)

# Good design: min_dist is large relative to max_dist
quality_ratio = min_dist / max_dist
# Typical values: 0.1-0.3 (higher is better)
```

**Coverage metric (discretize space):**
```python
# Divide each factor into bins
# Count how many bins contain at least one point
# Good design: high percentage of bins covered
```

## Example Workflow

### Basic Usage

```python
from factors import Factor, FactorType
from latin_hypercube import generate_latin_hypercube

# Define factors
factors = [
    Factor(name='Temperature', type=FactorType.CONTINUOUS, 
           min=100, max=200, units='°C'),
    Factor(name='Pressure', type=FactorType.CONTINUOUS,
           min=1, max=5, units='bar'),
    Factor(name='Catalyst', type=FactorType.CATEGORICAL,
           levels=['A', 'B', 'C'])
]

# Generate design
design = generate_latin_hypercube(
    factors=factors,
    n_runs=30,
    criterion='maximin',
    n_candidates=10,
    seed=42  # For reproducibility
)

# Access design matrix
print(design.design)  # Actual levels
print(design.design_coded)  # Coded levels

# Check quality
print(f"Minimum distance: {design.criterion_value}")
```

### Augmentation

```python
# Run initial experiments
initial_design = generate_latin_hypercube(factors, n_runs=20, seed=42)

# ... collect data, analyze ...

# Need more data - add 10 runs
from latin_hypercube import augment_latin_hypercube

augmented_design = augment_latin_hypercube(
    existing_design=initial_design.design.drop(columns=['StdOrder', 'RunOrder']),
    factors=factors,
    n_additional_runs=10,
    criterion='maximin',
    seed=42
)

# Total design now has 30 runs
print(f"Total runs: {augmented_design.n_runs}")
```

### Comparing Criteria

```python
# Generate with maximin (better space-filling)
design_maximin = generate_latin_hypercube(
    factors, n_runs=50, criterion='maximin', seed=42
)

# Generate with correlation (more independent factors)
design_corr = generate_latin_hypercube(
    factors, n_runs=50, criterion='correlation', seed=42
)

# Compare
print(f"Maximin score: {design_maximin.criterion_value}")
print(f"Correlation score: {design_corr.criterion_value}")

# Check actual correlation
import numpy as np
corr_maximin = np.corrcoef(design_maximin.design_coded.iloc[:, 2:].T)
corr_corr = np.corrcoef(design_corr.design_coded.iloc[:, 2:].T)

print("Maximin correlations:", corr_maximin[0, 1])
print("Correlation-optimized:", corr_corr[0, 1])
```

## Performance Considerations

### Computational Complexity

| Operation | Complexity | Typical Time |
|-----------|------------|--------------|
| Generate single LHS | O(n × k) | <0.01s |
| Compute distances (maximin) | O(n² × k) | 0.1s for n=100 |
| Generate 10 candidates | O(10 × n² × k) | 1s for n=100 |
| Augmentation | O(m × (n₁+n₂)² × k) | 2s for n₁=n₂=50 |

where:
- n = number of runs
- k = number of factors
- m = number of candidates
- n₁, n₂ = existing and new runs

### Memory Requirements

- Design matrix: O(n × k) = 8 bytes × n × k
- Distance matrix: O(n²) = 8 bytes × n²

For n=1000, k=20: ~160 KB (negligible)

### Optimization Tips

1. **Use fewer candidates** for large designs (n > 1000)
   - 10 candidates is usually sufficient
   - Diminishing returns beyond 20 candidates

2. **Use correlation criterion** for high-dimensional problems (k > 20)
   - Faster than maximin (O(n × k²) vs O(n² × k))
   - Still provides good space-filling

3. **Batch augmentation** rather than sequential
   - Better to add 20 runs once than 5 runs four times

## References

1. **McKay, M. D., Beckman, R. J., & Conover, W. J. (1979).** "A comparison of three methods for selecting values of input variables in the analysis of output from a computer code." *Technometrics*, 21(2), 239-245.
   - Original LHS paper

2. **Morris, M. D., & Mitchell, T. J. (1995).** "Exploratory designs for computational experiments." *Journal of Statistical Planning and Inference*, 43(3), 381-402.
   - Maximin criterion and optimization

3. **Owen, A. B. (1992).** "Orthogonal arrays for computer experiments, integration and visualization." *Statistica Sinica*, 2, 439-452.
   - Correlation criterion and Latin hypercube properties

4. **Jin, R., Chen, W., & Sudjianto, A. (2005).** "An efficient algorithm for constructing optimal design of computer experiments." *Journal of Statistical Planning and Inference*, 134(1), 268-287.
   - Modern algorithms for LHS optimization

## Future Enhancements

Potential improvements for future versions:

1. **Enhanced Latin Hypercube (ELH)**
   - Additional constraints on projections
   - Better performance for low-dimensional projections

2. **Orthogonal Latin Hypercube**
   - Minimize correlation in all 2D projections
   - More computationally intensive

3. **Maximin with constraints**
   - Combine space-filling with feasible region constraints
   - Useful for mixture experiments or process constraints

4. **Parallel candidate generation**
   - Use multiprocessing for large designs
   - 5-10× speedup possible

5. **Progressive augmentation**
   - Add one run at a time optimally
   - Useful for sequential experimentation