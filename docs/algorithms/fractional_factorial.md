# Fractional Factorial Design Algorithm

## Overview

A fractional factorial design uses a carefully chosen fraction of the runs from a full factorial design. Instead of running all 2^k combinations, we run only 2^(k-p) runs, where p is the number of generators.

**Fraction:** 1/2^p of the full factorial

**Example:** A 2^(5-1) design runs 16 experiments instead of 32 (half-fraction)

## Mathematical Background

### Generators and Defining Relation

A fractional factorial is created by selecting p **generators** - algebraic relationships that define which fraction of the design space to explore.

**Example Generator:**
```
E = A × B × C × D
```

This means: When A, B, C, D are all positive (+1), E is positive. When any subset are negative, E follows the multiplication rule.

**Defining Relation:**
The generator E = ABCD implies:
```
I = ABCDE
```

Where I is the identity. This is the **defining relation** or **generator word**.

### Confounding and Aliasing

The cost of running fewer experiments is **confounding** (aliasing) - some effects cannot be separated.

**Alias Formula:**
Any effect X is aliased with:
```
X × (defining relation word)
```

**Example:** For I = ABCDE:
- A is aliased with: A × ABCDE = BCDE
- B is aliased with: B × ABCDE = ACDE  
- AB is aliased with: AB × ABCDE = CDE

**Interpretation:** We cannot distinguish between A and the 4-factor interaction BCDE.

### Resolution

Resolution indicates the degree of confounding:

**Resolution III:**
- Main effects confounded with 2-factor interactions
- Example: A + BC (cannot separate main effect A from interaction BC)
- Useful only for screening when interactions assumed negligible

**Resolution IV:**
- Main effects clear (not confounded with other main effects or 2FI)
- 2-factor interactions confounded with other 2-factor interactions
- Example: AB + CD
- Good for identifying important main effects

**Resolution V:**
- Main effects clear
- 2-factor interactions clear (not confounded with other 2FI)
- 2-factor interactions confounded with 3-factor interactions
- Excellent for estimating main effects and 2FI

**Mathematical Definition:**
Resolution R = minimum word length in defining relation (excluding I)

### Design Selection Strategy

**Common Designs:**

| k | p | Runs | Resolution | Best For |
|---|---|------|-----------|----------|
| 4 | 1 | 8 | IV | Small studies, expect interactions |
| 5 | 1 | 16 | V | Good balance, clear 2FI |
| 6 | 2 | 16 | IV | Many factors, screening |
| 7 | 3 | 16 | IV | Large screening studies |
| 8 | 4 | 16 | IV | Very large screening |

**Selection Rules:**
1. For k ≤ 5: Use Resolution V (full 2FI estimation)
2. For k = 6-7: Use Resolution IV (main effects + some 2FI)
3. For k ≥ 8: Use Resolution III or IV for screening

## Generator Selection

### Standard Generators

Well-established generator sets maximize resolution:

**2^(5-1) Resolution V:**
```
E = ABCD
Defining Relation: I = ABCDE
```

**2^(7-3) Resolution IV:**
```
E = ABC
F = BCD  
G = ACD
Defining Relation: I = ABCE = BCDF = ACDG = ...
```

The complete defining relation includes all products of generators.

### Custom Generators

Users can specify custom generators for special purposes:

**Requirements:**
1. Each generator must involve k-p factors
2. Generators should be independent
3. Higher word length → higher resolution

**Example Custom Generator:**
For a 2^(6-2) design emphasizing factors A and B:
```
E = ACD (avoids B)
F = ABD (includes both A and B)
```

## Implementation Details

### Algorithm Steps

**1. Base Design Generation**
```
Generate 2^(k-p) full factorial for first k-p factors
```

**2. Generated Factor Calculation**
For each generator (e.g., E = ABC):
```python
E_values = A_values × B_values × C_values
```

Where multiplication is element-wise for all runs.

**3. Alias Structure Calculation**

For each effect E:
```
Aliases = {E × W for W in defining_relation if W ≠ I}
```

Simplify using mod-2 algebra:
- A × A = I (any factor squared is identity)
- A × B × A = B (cancellation)
- ABCABC = I

**Example Simplification:**
```
E × ABCDE = ABCD (E cancels)
AB × ABCDE = CDE (A and B cancel)
```

### Computational Complexity

**Time Complexity:**
- Design generation: O(2^(k-p) × k)
- Alias calculation: O(2^p × k^2)
  - Must evaluate all 2^p words in defining relation
  - For each of O(k^2) effects

**Space Complexity:**
- O(2^(k-p) × k) for design matrix
- O(2^p) for defining relation
- O(k^2) for alias structure

### Example Calculation

**Problem:** Create 2^(5-1) design with E = ABCD

**Step 1: Generate base factorial for A, B, C, D**
```
Run  A   B   C   D
1   -1  -1  -1  -1
2   +1  -1  -1  -1
3   -1  +1  -1  -1
...
16  +1  +1  +1  +1
```

**Step 2: Calculate E = A × B × C × D**
```
Run  A   B   C   D   E
1   -1  -1  -1  -1  +1  ((-1)×(-1)×(-1)×(-1) = +1)
2   +1  -1  -1  -1  -1  ((+1)×(-1)×(-1)×(-1) = -1)
3   -1  +1  -1  -1  -1
...
16  +1  +1  +1  +1  +1
```

**Step 3: Defining Relation**
```
I = ABCDE
```

**Step 4: Alias Structure**
```
A: aliased with BCDE
B: aliased with ACDE
C: aliased with ABDE
D: aliased with ABCE
E: aliased with ABCD

AB: aliased with CDE
AC: aliased with BDE
AD: aliased with BCE
AE: aliased with BCD
BC: aliased with ADE
BD: aliased with ACE
BE: aliased with ACD
CD: aliased with ABE
CE: aliased with ABD
DE: aliased with ABC
```

Resolution = 5 (minimum word length = 5)

## Analysis Considerations

### Effect Estimation

When analyzing fractional factorial data:

**Resolution V:**
- Estimate main effects directly
- Estimate 2FI directly  
- Both are unbiased (assuming 3FI+ negligible)

**Resolution IV:**
- Estimate main effects directly (unbiased)
- 2FI estimates are biased by other 2FI
- Use effect sparsity principle and subject matter knowledge

**Resolution III:**
- Main effect estimates biased by 2FI
- Can only separate if strong effect hierarchy
- Typically used for screening only

### Effect Sparsity Principle

**Assumption:** Only a few effects are active (large)

**Implications:**
- In alias strings like AB + CD, typically only one is large
- Use half-normal plots to identify active effects
- Active effects fall off the line
- Follow-up experiments can de-alias if needed

### Sequential Experimentation

**Fold-over Designs:**
Run the opposite fraction to de-alias effects:
```
Original: E = +ABCD
Fold-over: E = -ABCD
Combined: Full 2^5 factorial
```

**Partial fold-over:**
Reverse signs of one factor to de-alias specific interactions.

## Advantages and Limitations

### Advantages

1. **Efficiency**
   - Run half (or less) the experiments
   - Still get main effects (if Resolution ≥ IV)
   
2. **Economical**
   - Fraction cost = Fraction × (Full cost)
   - 2^(8-4) costs 1/16 of 2^8

3. **Sequential Strategy**
   - Start with screening design
   - Augment if needed
   - Never run more than necessary

4. **Effect Hierarchy**
   - Lower-order effects usually larger
   - High-order interactions typically negligible
   - Design takes advantage of this

### Limitations

1. **Confounding**
   - Cannot estimate all effects
   - Some effects inseparable
   - Resolution III: Main effects confounded with 2FI

2. **Assumptions Required**
   - Effect sparsity
   - Hierarchy (lower order > higher order)
   - If violated, misleading results

3. **Model Flexibility**
   - Cannot fit full model
   - Predetermined by resolution
   - May miss important interactions

4. **Analysis Complexity**
   - Must understand alias structure
   - Interpretation requires more care
   - Effect estimates may be biased

## When to Use Fractional Factorial

**Use Fractional Factorial when:**
- Many factors (k ≥ 5)
- Resources limited
- Interactions expected to be small
- Screening is the goal
- Sequential experimentation acceptable

**Use Full Factorial when:**
- Few factors (k ≤ 4)
- Interactions critical
- Complete characterization needed
- Resources permit
- No confounding acceptable

## Standard Generator Tables

### 2^(k-1) Designs (Half-Fractions)

| k | Generator | Resolution | Runs |
|---|-----------|-----------|------|
| 4 | D=ABC | IV | 8 |
| 5 | E=ABCD | V | 16 |
| 6 | F=ABCDE | VI | 32 |
| 7 | G=ABCDEF | VII | 64 |

### 2^(k-2) Designs (Quarter-Fractions)

| k | Generators | Resolution | Runs |
|---|------------|-----------|------|
| 5 | D=AB, E=AC | III | 8 |
| 6 | E=ABC, F=BCD | IV | 16 |
| 7 | F=ABCD, G=ABCE | IV | 32 |

### 2^(k-3) Designs (Eighth-Fractions)

| k | Generators | Resolution | Runs |
|---|------------|-----------|------|
| 7 | E=ABC, F=BCD, G=ACD | IV | 16 |
| 8 | F=ABC, G=ABD, H=ACDE | IV | 32 |

## References

1. Box, G. E. P., Hunter, J. S., and Hunter, W. G. (2005). *Statistics for Experimenters*, 2nd Ed. Wiley.

2. Montgomery, D. C. (2017). *Design and Analysis of Experiments*, 9th Ed. Wiley.

3. Wu, C. F. J., and Hamada, M. S. (2009). *Experiments: Planning, Analysis, and Optimization*, 2nd Ed. Wiley.

4. Box, G. E. P., and Wilson, K. B. (1951). On the Experimental Attainment of Optimum Conditions. *Journal of the Royal Statistical Society, Series B*, 13, 1-45.

5. National Institute of Standards and Technology (NIST). Engineering Statistics Handbook, Section 5.3.3: Fractional Factorial Designs.