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

`FactorMapper` builds a bidirectional mapping in factor order (`src/core/aliasing.py`):

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
for i, factor in enumerate(factors):
    symbol = chr(65 + i)  # A=65, B=66, ...
    real_to_algebraic[factor.name] = symbol
    algebraic_to_real[symbol] = factor.name
```

**Input vs. output convention:** generator **expressions** must be written with
the single-letter **algebraic** symbols (`"E=ABCD"`). Real factor names are the
**output-side** representation:
- Alias tables can be rendered with real names by passing the mapper to
  `format_alias_table(..., mapper=mapper)`.
- Design matrix column labels use the real factor names.
- `FactorMapper.translate_generator()` converts a generator string between
  forms (e.g. `translate_generator("E=ABCD", to_algebraic=False)` →
  `"Catalyst=Temperature*Pressure*Time*Speed"`).

Real-name generator strings are **not** accepted as user input to the design
generator. Passing one fails syntax validation:

```
ValueError: Left side must be single factor: 'Catalyst=Temperature*Pressure*Time*Speed'
```

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

**Check 1: Factors exist — and the LHS is a generated factor**

The left side must be one of the **generated** factors, i.e. the (k − p) new
factors introduced by the fraction. A base factor on the left side is rejected:

```
Generator: A=BCD  (k=5, 1/2 fraction → generated factors are {E})
ValueError: Generator 'A=BCD' must define a generated factor (expected one of E), not base factor 'A'
```

Every letter on the right side must be one of the base factors `A..D`:

```
Generator: E=XYZ
ValueError: Factor 'X' in 'E=XYZ' not in base factors. Available: A, B, C, D
```

**Check 2: Correct number of generators**
```
Fraction 1/4 → p=2 → Need exactly 2 generators
Provided: ["E=ABC", "F=BCD"]
✓ Correct count

Provided: ["E=ABCD"] only
ValueError: Expected 1 generators for 1/2 fraction, got 1
```

**Check 3: Resolution achievable**
```
Generators: [("D", "ABC")]
Defining relation: I = ABCD
Min word length: 4
Resolution: IV ✓

Claimed/generated resolution below target → error
ValueError: Generators achieve Resolution 4, not 5 as specified
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
= A·B·C·E·B·C·D·F
= A·(B·B)·(C·C)·D·E·F     (collect like letters)
= A·D·E·F                  (B×B=I, C×C=I)

Defining Relation: {I, ABCE, BCDF, ADEF}
```

Note the simplification carefully: the B and C factors each appear twice and
cancel, leaving **ADEF** (not CDEF). The same `AliasingEngine` computation
reports `resolution = 4` for this design because every generator word has
length 4.

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
- Higher-order interactions (5FI+) are typically negligible in screening contexts
- Practical size: full enumeration of all k-way products grows combinatorially
- Sufficient for resolution V verification and main-effects-plus-2FI interpretation

**Number of effects:**
```
Total = C(k,1) + C(k,2) + C(k,3) + C(k,4)
```

For k=8: 8 + 28 + 56 + 70 = 162 effects.

**Real output example — 2^(5-1) half fraction (E=ABCD), rendered with real names:**

```
                        Effect                        Aliased_With
                    Temperature  Pressure*Time*Speed*Catalyst
                       Pressure  Temperature*Time*Speed*Catalyst
                           Time  Temperature*Pressure*Speed*Catalyst
                          Speed  Temperature*Pressure*Time*Catalyst
                       Catalyst  Temperature*Pressure*Time*Speed
               Temperature*Pressure          Time*Speed*Catalyst
                 Temperature*Time      Pressure*Speed*Catalyst
                        ...
        Temperature*Pressure*Time*Speed                    Catalyst
```

Each main effect is aliased only with the corresponding 4FI; every 2FI is
aliased with a 3FI — the expected Resolution V structure.

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

The module ships a library of well-established generators from the statistical
literature (Box-Hunter-Hunter, Montgomery). **14 standard designs** are
currently included (`STANDARD_GENERATORS` in `src/core/aliasing.py`):

| (k, p) | Resolution | Generators |
|--------|-----------|------------|
| 2⁴⁻¹ (4,1) | IV | D = ABC |
| 2⁵⁻¹ (5,1) | V | E = ABCD |
| 2⁶⁻¹ (6,1) | VI | F = ABCDE |
| 2⁷⁻¹ (7,1) | VII | G = ABCDEF |
| 2⁵⁻² (5,2) | III | D = AB, E = AC |
| 2⁶⁻² (6,2) | IV | E = ABC, F = BCD |
| 2⁷⁻² (7,2) | IV | F = ABCD, G = ABCE |
| 2⁸⁻² (8,2) | V | G = ABCD, H = ABEF |
| 2⁶⁻³ (6,3) | III | D = AB, E = AC, F = BC |
| 2⁷⁻³ (7,3) | IV | E = ABC, F = BCD, G = ACD |
| 2⁸⁻³ (8,3) | IV | F = ABC, G = ABD, H = ABE |
| 2⁸⁻⁴ (8,4) | IV | E = BCD, F = ACD, G = ABC, H = ABD |
| 2⁹⁻⁴ (9,4) | IV | F = ABCD, G = ABCE, H = ABDE, J = BCDE |
| 2¹⁰⁻⁴ (10,4) | IV | G = ABCD, H = ABEF, J = ACEF, K = BCEF |

The (6,1) Resolution VI and (7,1) Resolution VII half-fractions are included
and reachable (the resolution lookup scans up to VII).

### Selection Strategy

When user specifies (k, p, resolution):

1. **Lookup:** Check if (k, p, resolution) exists in the library
2. **If found:** Use the standard generators
3. **If not found:** Return None, and the design step requires the user to
   provide custom generators

```python
get_standard_generators(5, 1, 5)   # → [('E', 'ABCD')]
get_standard_generators(9, 1, 9)   # → None  (not in library)
```

**Why not auto-generate?**

Finding optimal generators is a combinatorial optimization problem. For
non-standard designs, the app requires explicit user specification rather than
potentially generating suboptimal designs.

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
   - Left side is a generated factor; right side letters are base factors
   - Generator count matches p

4. Get or validate generators:
   If custom generators provided:
     - Validate each generator
     - Compute actual resolution
     - Verify it meets the required resolution

   If resolution specified:
     - Lookup standard generators
     - Error if not found

   If neither:
     - Use the highest resolution available

5. Create AliasingEngine
   - Compute defining relation
   - Calculate resolution
   - Build alias structure

6. Success → Store results
```

### Error Messages

**Design philosophy:** concise but actionable.

**Examples (verbatim from the module):**

❌ Bad: "Invalid generator"
✓ Good: "Factor 'X' in 'E=XYZ' not in base factors. Available: A, B, C, D"

❌ Bad: "Wrong number"
✓ Good: "Expected 1 generators for 1/2 fraction, got 2"

❌ Bad: "Low resolution"
✓ Good: "Generators achieve Resolution 4, not 5 as specified"

❌ Bad: "Invalid left side"
✓ Good: "Generator 'A=BCD' must define a generated factor (expected one of E), not base factor 'A'"

## Implementation Notes

### Design Principles

1. **Separation of concerns:**
   - Generator validation → GeneratorValidator
   - Name mapping → FactorMapper
   - Aliasing computation → AliasingEngine
   - Each class has a single responsibility

2. **Immutability:**
   - Once created, AliasingEngine results don't change
   - Generators validated before use
   - No hidden state changes

3. **Fail fast:**
   - Validate at construction time
   - Don't defer validation to design generation
   - Clear error messages at the point of failure

### Performance

The bottleneck is alias-structure computation for large designs. Mitigations:

- Limit effect order to 4
- Cache the defining relation
- Use set operations for efficiency

**Measured timings on a typical workstation (current implementation):**

```
k=5,  p=1:   30 aliased effects,   2 words, ~0.3 ms
k=8,  p=4:  162 aliased effects,  16 words, ~8 ms
k=10, p=4:  385 aliased effects,  16 words, ~20 ms
```

These figures depend on the machine and release and may evolve; they are all
well within interactive limits.

## References

1. Box, G. E. P., Hunter, J. S., and Hunter, W. G. (2005). *Statistics for Experimenters*, 2nd Ed. Wiley. Chapter 8.

2. Montgomery, D. C. (2017). *Design and Analysis of Experiments*, 9th Ed. Wiley. Chapter 8.

3. Wu, C. F. J., and Hamada, M. S. (2009). *Experiments: Planning, Analysis, and Optimization*, 2nd Ed. Wiley. Chapter 5.

4. National Institute of Standards and Technology. *Engineering Statistics Handbook*, Section 5.3.3.6: Fractional Factorial Design Construction.