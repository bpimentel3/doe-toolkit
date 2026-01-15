"""
Comprehensive test of CSV parser fixes.
Run this to verify all fixes work before running pytest.
"""

from src.ui.utils.csv_parser import parse_doe_csv, generate_doe_csv, validate_csv_structure
from src.core.factors import Factor, FactorType, ChangeabilityLevel
import pandas as pd

print("="*70)
print("CSV PARSER COMPREHENSIVE TEST")
print("="*70)

# Test 1: CSV without RESPONSE DEFINITIONS section
print("\n1. Testing CSV without RESPONSE DEFINITIONS section...")
csv_no_resp = """# DOE-TOOLKIT DESIGN
# Version: 1.0
# Design Type: full_factorial
#
# FACTOR DEFINITIONS
# Name,Type,Changeability,Levels,Units
# Temperature,continuous,easy,150|200,°C
# Pressure,continuous,easy,50|100,psi
#
# DESIGN DATA
StdOrder,RunOrder,Temperature,Pressure
1,1,150,50
2,2,200,100
"""

result = parse_doe_csv(csv_no_resp)
assert result.is_valid, f"Parse failed: {result.error}"
assert len(result.factors) == 2, f"Expected 2 factors, got {len(result.factors)}"
assert len(result.response_definitions) == 0, f"Expected 0 responses, got {len(result.response_definitions)}"
assert len(result.design_data) == 2, f"Expected 2 rows, got {len(result.design_data)}"
print("   ✓ PASSED - Parsed CSV without responses correctly")

# Test 2: Generate CSV without responses and parse it back
print("\n2. Testing generate → parse roundtrip WITHOUT responses...")
design = pd.DataFrame({
    'StdOrder': [1, 2],
    'RunOrder': [1, 2],
    'Temp': [150, 200],
})
factors = [
    Factor(
        name='Temp',
        factor_type=FactorType.CONTINUOUS,
        changeability=ChangeabilityLevel.EASY,
        levels=[150.0, 200.0],
        _validate_on_init=False,
    )
]

csv = generate_doe_csv(design, factors, response_definitions=None)
result2 = parse_doe_csv(csv)
assert result2.is_valid, f"Parse failed: {result2.error}"
assert len(result2.factors) == 1, f"Expected 1 factor, got {len(result2.factors)}"
assert len(result2.response_definitions) == 0, f"Expected 0 responses, got {len(result2.response_definitions)}"
print("   ✓ PASSED - Roundtrip without responses works")

# Test 3: Discrete factors
print("\n3. Testing discrete numeric factors...")
csv_discrete = """# DOE-TOOLKIT DESIGN
# Version: 1.0
# Design Type: custom
#
# FACTOR DEFINITIONS
# Name,Type,Changeability,Levels,Units
# RPM,discrete_numeric,easy,100|150|200,rpm
# Time,discrete_numeric,easy,5|10|15,minutes
#
# DESIGN DATA
StdOrder,RunOrder,RPM,Time
1,1,100,5
2,2,150,10
3,3,200,15
"""

result3 = parse_doe_csv(csv_discrete)
assert result3.is_valid, f"Parse failed: {result3.error}"
assert len(result3.factors) == 2, f"Expected 2 factors, got {len(result3.factors)}"
assert result3.factors[0].factor_type == FactorType.DISCRETE_NUMERIC
assert result3.factors[0].levels == [100.0, 150.0, 200.0]
print("   ✓ PASSED - Discrete factors parsed correctly")

# Test 4: Categorical factors
print("\n4. Testing categorical factors...")
factors_cat = [
    Factor(
        name='Material',
        factor_type=FactorType.CATEGORICAL,
        changeability=ChangeabilityLevel.EASY,
        levels=['A', 'B', 'C'],
        _validate_on_init=False,
    )
]
design_cat = pd.DataFrame({'StdOrder': [1], 'RunOrder': [1], 'Material': ['A']})
csv_cat = generate_doe_csv(design_cat, factors_cat)
result4 = parse_doe_csv(csv_cat)
assert result4.is_valid, f"Parse failed: {result4.error}"
assert result4.factors[0].factor_type == FactorType.CATEGORICAL
assert result4.factors[0].levels == ['A', 'B', 'C']
print("   ✓ PASSED - Categorical factors work correctly")

# Test 5: Validation with matching factors
print("\n5. Testing validation with matching factors...")
is_valid, errors = validate_csv_structure(result2, session_factors=factors)
assert is_valid, f"Validation failed: {errors}"
assert len(errors) == 0
print("   ✓ PASSED - Validation works correctly")

# Test 6: Validation with mismatched factor count
print("\n6. Testing validation with mismatched factor count...")
fewer_factors = []
is_valid, errors = validate_csv_structure(result2, session_factors=fewer_factors)
assert not is_valid, "Validation should have failed"
assert any('count mismatch' in e for e in errors), f"Expected count mismatch error, got: {errors}"
print("   ✓ PASSED - Mismatch detection works")

print("\n" + "="*70)
print("ALL TESTS PASSED! ✓")
print("="*70)
