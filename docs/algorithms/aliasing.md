# Aliasing and Generator Management

## Overview

This document describes the algorithms used for managing generators, computing alias structures, and validating fractional factorial designs in DOE-Toolkit.

## Problem Statement

Fractional factorial designs use generators to define which subset of the full factorial to run. This creates **aliasing** (confounding) where certain effects cannot be distinguished from each other. The aliasing module:

1. Validates that generators are properly specified
2. Maps between real factor names and algebraic symbols
3. Computes the complete defining relation
4. Calculates which effects are aliased with each other
5. Determines the design resolution

## Factor Name Mapping

### Challenge

Users define factors with meaningful names like "Temperature", "Pressure", "Catalyst_Amount". However, generator algebra uses single-letter notation (A, B, C, ...) following statistical convention.

### Solution: Bidirectional Mapping

```
Real Names          Algebraic Symbols
Temperature    <-->  A
Pressure       <-->  B
Time           <-->  C
Speed          <-->  D
Catalyst       <-->  E
```

**Algorithm:**
```python
# Create mapping in factor order
for i, factor in enumerate(factors):
    symbol = chr(65 + i)  # A=65, B=66, ...
    real_to_algebraic[factor.name] = symbol
    algebraic_to_real[symbol] = factor.name
```

**Use cases:**
- Parser accepts "E=ABCD" (algebraic)
- Design matrix uses "Catalyst=Temperature*Pressure*Time*Speed" (real names)
- Alias tables can show either format

## Generator Validation

### Syntax Validation

**Valid generator format:** `X=YZ...`

Where:
- Left side: single letter (the generated factor)
- Right side: product of 2+ base factors
- No spaces required (but allowed)

**Invalid examples:**
- `EABCD` (missing =)
- `=ABCD` (empty left side)
- `E=` (empty right side)
- `EF=ABC` (multi-character left side)
- `E=123` (non-alphabetic)

### Semantic Validation

**Check 1: Factors exist**
```
Generator: E=ABCD
Base factors available: {A, B, C, D}
✓ All factors in expression exist
```

**Check 2: Correct number of generators**
```
Fraction 1/4 → p=2 → Need exactly 2 generators
Provided: ["E=ABC", "F=BCD"]
✓ Correct count
```

**Check 3: Resolution achievable**
```
Generators: [("D", "ABC")]
Defining relation: I = ABCD
Min word length: 4
Resolution: IV ✓

Claimed resolution: V
Actual resolution: IV
✗ Mismatch → Error
```

## Defining Relation Computation

The **defining relation** is the set of all generator words and their products.

### Algorithm

**Input:** Generators G₁, G₂, ..., Gₚ

**Step 1: Add identity and generators**
```
Words = {I}
For each generator (factor=expression):
    Words ← Words ∪ {factor * expression}
```

**Step 2: Generate all products**
```
For r = 2 to p:
    For each combination of r generators:
        product = multiply all generators in combination
        simplified = simplify_mod2(product)
        Words ← Words ∪ {simplified}
```

**Mod-2 Simplification:**

In mod-2 algebra:
- A × A = I (any factor squared is identity)
- A × B × A = B (cancellation)
- Order doesn't matter: ABC = BAC = CAB

**Algorithm:**
```python
def simplify_word(word: str) -> str:
    # Count occurrences of each letter
    counts = {}
    for letter in word:
        counts[letter] = counts.get(letter, 0) + 1
    
    # Keep only letters with odd count
    result = ""
    for letter in sorted(counts.keys()):
        if counts[letter] % 2 == 1:
            result += letter
    
    return result
```

**Example:**

Generators: E=ABC, F=BCD

```
Initial: I, ABCE, BCDF

Multiply ABCE × BCDF:
= ABCEBCDF
= AABBCCDDEF  (rearrange)
= CDEF  (A×A=I, B×B=I, C×C=I)

Defining Relation: {I, ABCE, BCDF, ADEF}
```

### Computational Complexity

- **Number of words:** 2^p (all subsets of generators)
- **Simplification per word:** O(k log k) for sorting
- **Total:** O(2^p × k log k)

For typical designs (p ≤ 4), this is very fast.

## Alias Structure Computation

### Definition

Effect X is **aliased** with effect Y if:
```
X + Y = W  (mod 2)
```
where W is any word in the defining relation (except I).

Equivalently: X is aliased with X × W for all W ≠ I.

### Algorithm

**Input:** 
- Effects E = {A, B, AB, AC, ...} (all effects up to order 4)
- Defining relation D = {I, W₁, W₂, ...}

**For each effect X in E:**
```
Aliases(X) = {}

For each word W in D:
    if W ≠ I:
        alias = simplify_mod2(X * W)
        if alias ≠ X:
            Aliases(X) ← Aliases(X) ∪ {alias}

Return Aliases(X)
```

**Example:**

Defining relation: {I, ABCDE}

```
Effect A:
  A × ABCDE = BCDE
  Aliases(A) = {BCDE}

Effect B:
  B × ABCDE = ACDE
  Aliases(B) = {ACDE}

Effect AB:
  AB × ABCDE = CDE
  Aliases(AB) = {CDE}
```

### Effect Generation

We generate all effects up to 4th order:

```
Order 1 (main effects): A, B, C, D, E, ...
Order 2 (2FI): AB, AC, AD, AE, BC, BD, ...
Order 3 (3FI): ABC, ABD, ABE, ACD, ...
Order 4 (4FI): ABCD, ABCE, ABDE, ...
```

**Why stop at order 4?**
- Higher-order interactions (5FI+) are typically negligible
- Reduces computation
- Sufficient for practical interpretation

**Number of effects:**
```
Total = C(k,1) + C(k,2) + C(k,3) + C(k,4)
```

For k=8: 8 + 28 + 56 + 70 = 162 effects

### Computational Complexity

- **Effects to consider:** O(k⁴)
- **Words in defining relation:** O(2^p)
- **Simplification per alias:** O(k log k)
- **Total:** O(k⁴ × 2^p × k log k) = O(k⁵ log k × 2^p)

For typical designs (k ≤ 10, p ≤ 4), this completes in milliseconds.

## Resolution Calculation

**Definition:** Resolution R is the minimum word length in the defining relation (excluding I).

**Algorithm:**
```python
def calculate_resolution(defining_relation):
    min_length = infinity
    
    for word in defining_relation:
        if word != "I":
            min_length = min(min_length, len(word))
    
    return min_length
```

**Example:**

Defining relation: {I, ABCDE}
```
Word lengths: [-, 5]
Resolution: 5
```

Defining relation: {I, ABCD, BCDE, ACE}
```
Word lengths: [-, 4, 4, 3]
Resolution: 3
```

### Resolution Interpretation

**Resolution III:**
- Min word length = 3
- Main effects aliased with 2FI
- Example: A + BC

**Resolution IV:**
- Min word length = 4
- Main effects clear
- 2FI aliased with other 2FI
- Example: AB + CD

**Resolution V:**
- Min word length = 5
- Main effects clear
- 2FI clear
- 2FI aliased with 3FI

## Standard Generator Library

We maintain a library of well-established generators from statistical literature (Box-Hunter-Hunter, Montgomery).

### Library Structure

```python
STANDARD_GENERATORS = {
    # Key: (k, p, resolution)
    # Value: [(factor, expression), ...]
    
    (5, 1, 5): [("E", "ABCD")],
    (7, 3, 4): [("E", "ABC"), ("F", "BCD"), ("G", "ACD")],
    (8, 4, 4): [("E", "BCD"), ("F", "ACD"), ("G", "ABC"), ("H", "ABD")],
}
```

### Selection Strategy

When user specifies (k, p, resolution):

1. **Lookup:** Check if (k, p, resolution) exists in library
2. **If found:** Use standard generators (optimal)
3. **If not found:** Return None, require user to provide custom generators

**Why not auto-generate?**

Finding optimal generators is a combinatorial optimization problem. For non-standard designs, we require explicit user specification rather than potentially generating suboptimal designs.

## Validation Workflow

### Complete Validation Sequence

```
User provides: factors, fraction, resolution?, generators?

1. Parse fraction → compute p
   Validate: p < k

2. Create FactorMapper
   Map real names ↔ algebraic symbols

3. Create GeneratorValidator
   Validate:
   - Generator syntax (X=YZ format)
   - Factors exist
   - Generator count matches p

4. Get or validate generators:
   If custom generators provided:
     - Validate each generator
     - Compute actual resolution
     - Verify matches claimed resolution
   
   If resolution specified:
     - Lookup standard generators
     - Error if not found
   
   If neither:
     - Use highest resolution available

5. Create AliasingEngine
   - Compute defining relation
   - Calculate resolution
   - Build alias structure

6. Success → Store results
```

### Error Messages

**Design philosophy:** Concise but actionable

**Examples:**

❌ Bad: "Invalid generator"
✓ Good: "Factor 'X' in 'E=XYZ' not in base factors. Available: A, B, C, D"

❌ Bad: "Wrong number"
✓ Good: "Expected 2 generators for 1/4 fraction, got 3"

❌ Bad: "Low resolution"
✓ Good: "Generators achieve Resolution 4, not 5 as specified"

## Implementation Notes

### Design Principles

1. **Separation of concerns:**
   - Generator validation → GeneratorValidator
   - Name mapping → FactorMapper
   - Aliasing computation → AliasingEngine
   - Each class has single responsibility

2. **Immutability:**
   - Once created, AliasingEngine results don't change
   - Generators validated before use
   - No hidden state changes

3. **Fail fast:**
   - Validate at construction time
   - Don't wait until generate() to find errors
   - Clear error messages at point of failure

### Performance Considerations

**Bottleneck:** Alias structure computation for large designs

**Optimization:**
- Limit effect order to 4
- Cache defining relation
- Use set operations for efficiency

**Typical performance:**
- k=5, p=1: <1ms
- k=8, p=4: ~5ms
- k=10, p=5: ~50ms

All well within acceptable limits for interactive use.

## References

1. Box, G. E. P., Hunter, J. S., and Hunter, W. G. (2005). *Statistics for Experimenters*, 2nd Ed. Wiley. Chapter 8.

2. Montgomery, D. C. (2017). *Design and Analysis of Experiments*, 9th Ed. Wiley. Chapter 8.

3. Wu, C. F. J., and Hamada, M. S. (2009). *Experiments: Planning, Analysis, and Optimization*, 2nd Ed. Wiley. Chapter 5.

4. National Institute of Standards and Technology. *Engineering Statistics Handbook*, Section 5.3.3.6: Fractional Factorial Design Construction.
