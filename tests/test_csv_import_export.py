"""
Integration tests for CSV export and import workflow.

Tests complete round-trip: generate design → export CSV → import CSV → verify.
"""

import pytest
import pandas as pd
from io import StringIO

from src.ui.utils.csv_parser import (
    generate_doe_csv,
    parse_doe_csv,
    validate_csv_structure,
    CSVParseError,
)
from src.core.factors import Factor, FactorType, ChangeabilityLevel


@pytest.fixture
def simple_design() -> pd.DataFrame:
    """Simple 2^2 full factorial design."""
    return pd.DataFrame({
        'StdOrder': [1, 2, 3, 4],
        'RunOrder': [3, 1, 4, 2],
        'Temperature': [150, 150, 200, 200],
        'Pressure': [50, 100, 50, 100],
    })


@pytest.fixture
def simple_factors() -> list:
    """Two continuous factors."""
    return [
        Factor(
            name='Temperature',
            factor_type=FactorType.CONTINUOUS,
            changeability=ChangeabilityLevel.EASY,
            levels=[150.0, 200.0],
            units='°C',
            _validate_on_init=False,
        ),
        Factor(
            name='Pressure',
            factor_type=FactorType.CONTINUOUS,
            changeability=ChangeabilityLevel.EASY,
            levels=[50.0, 100.0],
            units='psi',
            _validate_on_init=False,
        ),
    ]


@pytest.fixture
def simple_responses() -> list:
    """Two response definitions."""
    return [
        {'name': 'Yield', 'units': '%'},
        {'name': 'Purity', 'units': 'mg/mL'},
    ]


class TestRoundTrip:
    """Tests for complete export → import workflows."""

    def test_generate_and_parse_with_responses(
        self, simple_design: pd.DataFrame, simple_factors: list, simple_responses: list
    ) -> None:
        """Test: Generate CSV with responses → Parse back."""
        
        # Generate
        csv = generate_doe_csv(
            design=simple_design,
            factors=simple_factors,
            response_definitions=simple_responses,
            design_type='full_factorial',
        )
        
        # Parse
        result = parse_doe_csv(csv)
        
        # Verify
        assert result.is_valid
        assert len(result.factors) == 2
        assert len(result.response_definitions) == 2
        assert len(result.design_data) == 4
        
        # Verify factors
        assert result.factors[0].name == 'Temperature'
        assert result.factors[0].min_value == 150
        assert result.factors[1].name == 'Pressure'
        
        # Verify responses
        assert result.response_definitions[0]['name'] == 'Yield'
        assert result.response_definitions[0]['units'] == '%'
        assert result.response_definitions[1]['name'] == 'Purity'

    def test_roundtrip_with_filled_responses(
        self, simple_design: pd.DataFrame, simple_factors: list, simple_responses: list
    ) -> None:
        """Test: Generate → Parse → Design contains response data."""
        
        # Generate CSV with empty response columns
        csv = generate_doe_csv(
            design=simple_design,
            factors=simple_factors,
            response_definitions=simple_responses,
            design_type='full_factorial',
        )
        
        # Simulate analyst filling in response data
        lines = csv.split('\n')
        data_start = None
        for i, line in enumerate(lines):
            if '# DESIGN DATA' in line:
                data_start = i + 1
                break
        
        # Modify design data lines to add response values
        if data_start:
            csv_lines = lines[:data_start+1]  # Keep header
            csv_lines.append('1,3,150,50,85.2,92.1')
            csv_lines.append('2,1,150,100,78.5,94.3')
            csv_lines.append('3,4,200,50,92.1,91.5')
            csv_lines.append('4,2,200,100,88.7,95.2')
            csv_with_data = '\n'.join(csv_lines)
        
        # Parse
        result = parse_doe_csv(csv_with_data)
        
        # Verify design data has responses
        assert 'Yield' in result.design_data.columns
        assert 'Purity' in result.design_data.columns
        assert result.design_data['Yield'].iloc[0] == 85.2
        assert result.design_data['Purity'].iloc[0] == 92.1

    def test_roundtrip_without_responses(
        self, simple_design: pd.DataFrame, simple_factors: list
    ) -> None:
        """Test: Generate → Parse without response definitions."""
        
        csv = generate_doe_csv(
            design=simple_design,
            factors=simple_factors,
            response_definitions=None,
            design_type='full_factorial',
        )
        
        result = parse_doe_csv(csv)
        
        assert result.is_valid
        assert len(result.factors) == 2
        assert len(result.response_definitions) == 0
        assert len(result.design_data) == 4

    def test_roundtrip_preserves_factor_metadata(
        self, simple_design: pd.DataFrame, simple_factors: list
    ) -> None:
        """Test: Factor metadata preserved through export → import."""
        
        csv = generate_doe_csv(
            design=simple_design,
            factors=simple_factors,
            design_type='full_factorial',
        )
        
        result = parse_doe_csv(csv)
        
        # Verify factor properties preserved
        temp_factor = result.factors[0]
        assert temp_factor.name == 'Temperature'
        assert temp_factor.factor_type == FactorType.CONTINUOUS
        assert temp_factor.changeability == ChangeabilityLevel.EASY
        assert temp_factor.min_value == 150
        assert temp_factor.max_value == 200
        assert temp_factor.units == '°C'


class TestMultipleResponses:
    """Tests for multiple response definitions and data."""

    def test_three_responses(
        self, simple_design: pd.DataFrame, simple_factors: list
    ) -> None:
        """Test CSV with 3 responses."""
        
        responses = [
            {'name': 'Yield', 'units': '%'},
            {'name': 'Purity', 'units': 'mg/mL'},
            {'name': 'Cost', 'units': '$'},
        ]
        
        csv = generate_doe_csv(
            design=simple_design,
            factors=simple_factors,
            response_definitions=responses,
            design_type='full_factorial',
        )
        
        result = parse_doe_csv(csv)
        
        assert len(result.response_definitions) == 3
        assert result.response_definitions[2]['name'] == 'Cost'
        assert result.response_definitions[2]['units'] == '$'

    def test_response_without_units(
        self, simple_design: pd.DataFrame, simple_factors: list
    ) -> None:
        """Test response definition without units field."""
        
        responses = [
            {'name': 'Yield', 'units': None},
            {'name': 'Success', 'units': None},
        ]
        
        csv = generate_doe_csv(
            design=simple_design,
            factors=simple_factors,
            response_definitions=responses,
        )
        
        result = parse_doe_csv(csv)
        
        assert result.response_definitions[0]['name'] == 'Yield'
        assert result.response_definitions[0]['units'] is None
        assert result.response_definitions[1]['name'] == 'Success'


class TestFactorTypes:
    """Tests for different factor types in roundtrip."""

    def test_discrete_factors_roundtrip(self) -> None:
        """Test discrete numeric factors in roundtrip."""
        
        factors = [
            Factor(
                name='RPM',
                factor_type=FactorType.DISCRETE_NUMERIC,
                changeability=ChangeabilityLevel.EASY,
                levels=[100.0, 150.0, 200.0],
                units='rpm',
                _validate_on_init=False,
            )
        ]
        
        design = pd.DataFrame({'StdOrder': [1], 'RunOrder': [1], 'RPM': [150]})
        
        csv = generate_doe_csv(design, factors, design_type='custom')
        result = parse_doe_csv(csv)
        
        assert result.factors[0].factor_type == FactorType.DISCRETE_NUMERIC
        assert result.factors[0].levels == [100.0, 150.0, 200.0]

    def test_categorical_factors_roundtrip(self) -> None:
        """Test categorical factors in roundtrip."""
        
        factors = [
            Factor(
                name='Material',
                factor_type=FactorType.CATEGORICAL,
                changeability=ChangeabilityLevel.EASY,
                levels=['A', 'B', 'C'],
                _validate_on_init=False,
            )
        ]
        
        design = pd.DataFrame({'StdOrder': [1], 'RunOrder': [1], 'Material': ['A']})
        
        csv = generate_doe_csv(design, factors, design_type='custom')
        result = parse_doe_csv(csv)
        
        assert result.factors[0].factor_type == FactorType.CATEGORICAL
        assert result.factors[0].levels == ['A', 'B', 'C']

    def test_mixed_factor_types(self) -> None:
        """Test mixed factor types in single design."""
        
        factors = [
            Factor(
                name='Temperature',
                factor_type=FactorType.CONTINUOUS,
                changeability=ChangeabilityLevel.EASY,
                levels=[100.0, 200.0],
                _validate_on_init=False,
            ),
            Factor(
                name='RPM',
                factor_type=FactorType.DISCRETE_NUMERIC,
                changeability=ChangeabilityLevel.EASY,
                levels=[100.0, 200.0],
                _validate_on_init=False,
            ),
            Factor(
                name='Material',
                factor_type=FactorType.CATEGORICAL,
                changeability=ChangeabilityLevel.EASY,
                levels=['A', 'B'],
                _validate_on_init=False,
            ),
        ]
        
        design = pd.DataFrame({
            'StdOrder': [1],
            'RunOrder': [1],
            'Temperature': [150],
            'RPM': [100],
            'Material': ['A'],
        })
        
        csv = generate_doe_csv(design, factors)
        result = parse_doe_csv(csv)
        
        assert len(result.factors) == 3
        assert result.factors[0].factor_type == FactorType.CONTINUOUS
        assert result.factors[1].factor_type == FactorType.DISCRETE_NUMERIC
        assert result.factors[2].factor_type == FactorType.CATEGORICAL


class TestValidation:
    """Tests for validation during import."""

    def test_validate_matching_factors(
        self, simple_design: pd.DataFrame, simple_factors: list
    ) -> None:
        """Test validation passes when factors match."""
        
        csv = generate_doe_csv(simple_design, simple_factors)
        result = parse_doe_csv(csv)
        
        is_valid, errors = validate_csv_structure(result, session_factors=simple_factors)
        
        assert is_valid
        assert errors == []

    def test_validate_mismatched_factor_count(
        self, simple_design: pd.DataFrame, simple_factors: list
    ) -> None:
        """Test validation catches factor count mismatch."""
        
        csv = generate_doe_csv(simple_design, simple_factors)
        result = parse_doe_csv(csv)
        
        # Session has only 1 factor
        is_valid, errors = validate_csv_structure(result, session_factors=simple_factors[:1])
        
        assert not is_valid
        assert any('count mismatch' in e for e in errors)

    def test_validate_mismatched_factor_names(
        self, simple_design: pd.DataFrame, simple_factors: list
    ) -> None:
        """Test validation catches factor name mismatch."""
        
        csv = generate_doe_csv(simple_design, simple_factors)
        result = parse_doe_csv(csv)
        
        # Modify session factor name
        modified_factors = [
            Factor(
                name='Temp',  # Wrong name
                factor_type=simple_factors[0].factor_type,
                changeability=simple_factors[0].changeability,
                levels=simple_factors[0].levels,
                _validate_on_init=False,
            ),
            simple_factors[1],
        ]
        
        is_valid, errors = validate_csv_structure(result, session_factors=modified_factors)
        
        assert not is_valid
        assert any('name mismatch' in e for e in errors)


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_csv_with_empty_response_cells(
        self, simple_design: pd.DataFrame, simple_factors: list, simple_responses: list
    ) -> None:
        """Test CSV with some empty response cells (partial data)."""
        
        csv = generate_doe_csv(simple_design, simple_factors, simple_responses)
        
        # Manually add partial response data
        lines = csv.split('\n')
        data_idx = None
        for i, line in enumerate(lines):
            if '# DESIGN DATA' in line:
                data_idx = i
                break
        
        # Keep header, add partial data
        csv_partial = '\n'.join(lines[:data_idx+2])  # Keep CSV header
        csv_partial += '\n1,3,150,50,85.2,'  # Yield filled, Purity empty
        csv_partial += '\n2,1,150,100,,94.3'  # Yield empty, Purity filled
        csv_partial += '\n3,4,200,50,92.1,91.5'  # Both filled
        csv_partial += '\n4,2,200,100,88.7,'  # Purity empty
        
        result = parse_doe_csv(csv_partial)
        
        assert result.is_valid
        assert len(result.design_data) == 4

    def test_csv_with_block_column(
        self, simple_factors: list, simple_responses: list
    ) -> None:
        """Test CSV with Block column is handled correctly."""
        
        design = pd.DataFrame({
            'StdOrder': [1, 2, 3, 4],
            'RunOrder': [3, 1, 4, 2],
            'Block': [1, 1, 2, 2],
            'Temperature': [150, 150, 200, 200],
            'Pressure': [50, 100, 50, 100],
        })
        
        csv = generate_doe_csv(design, simple_factors, simple_responses)
        result = parse_doe_csv(csv)
        
        # Block column should be in design data but not treated as factor
        assert 'Block' in result.design_data.columns
        assert result.design_data['Block'].iloc[0] == 1

    def test_csv_with_split_plot_columns(
        self, simple_factors: list
    ) -> None:
        """Test CSV with WholePlot column (split-plot design)."""
        
        design = pd.DataFrame({
            'StdOrder': [1, 2, 3, 4],
            'RunOrder': [1, 2, 3, 4],
            'WholePlot': [1, 1, 2, 2],
            'Temperature': [150, 150, 200, 200],
            'Pressure': [50, 100, 50, 100],
        })
        
        csv = generate_doe_csv(design, simple_factors)
        result = parse_doe_csv(csv)
        
        assert 'WholePlot' in result.design_data.columns
        assert result.design_data['WholePlot'].nunique() == 2

    def test_large_design_roundtrip(
        self, simple_factors: list, simple_responses: list
    ) -> None:
        """Test roundtrip with large design (100+ runs)."""
        
        design = pd.DataFrame({
            'StdOrder': range(1, 101),
            'RunOrder': range(1, 101),
            'Temperature': [150 + i % 50 for i in range(100)],
            'Pressure': [50 + i % 50 for i in range(100)],
        })
        
        csv = generate_doe_csv(design, simple_factors, simple_responses)
        result = parse_doe_csv(csv)
        
        assert len(result.design_data) == 100
        assert result.is_valid