"""Quick test to verify CSV parser fixes."""

from src.ui.utils.csv_parser import parse_doe_csv, generate_doe_csv
from src.core.factors import Factor, FactorType, ChangeabilityLevel
import pandas as pd

# Test 1: CSV without responses
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

print("Test 1: Parsing CSV without responses...")
result = parse_doe_csv(csv_no_resp)
print(f"  Valid: {result.is_valid}")
print(f"  Factors: {len(result.factors)}")
print(f"  Responses: {len(result.response_definitions)}")
print(f"  Design rows: {len(result.design_data)}")
assert result.is_valid
assert len(result.factors) == 2
assert len(result.response_definitions) == 0
print("  ✓ PASSED\n")

# Test 2: Generate and parse roundtrip
print("Test 2: Generate and parse roundtrip...")
design = pd.DataFrame({
    'StdOrder': [1, 2],
    'RunOrder': [1, 2],
    'Temperature': [150, 200],
})
factors = [
    Factor(
        name='Temperature',
        factor_type=FactorType.CONTINUOUS,
        changeability=ChangeabilityLevel.EASY,
        levels=[150, 200],
        units='°C',
        _validate_on_init=False,
    )
]

csv = generate_doe_csv(design, factors)
print("Generated CSV:")
print(csv)
print()

result2 = parse_doe_csv(csv)
print(f"  Valid: {result2.is_valid}")
print(f"  Factors: {len(result2.factors)}")
print(f"  Responses: {len(result2.response_definitions)}")
if result2.error:
    print(f"  ERROR: {result2.error}")
assert result2.is_valid
assert len(result2.factors) == 1
print("  ✓ PASSED\n")

print("All tests passed!")
