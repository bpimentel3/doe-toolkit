"""
Unit tests for CSV parser module.

Tests parsing of DOE-Toolkit CSV format with metadata headers.
"""

import pytest
import pandas as pd

from src.ui.utils.csv_parser import (
    parse_doe_csv,
    extract_metadata_block,
    extract_factor_definitions,
    extract_response_definitions,
    extract_design_data,
    validate_csv_structure,
    generate_doe_csv,
    CSVParseError,
)
from src.core.factors import Factor, FactorType, ChangeabilityLevel


@pytest.fixture
def valid_doe_csv() -> str:
    """Valid DOE-Toolkit CSV with all sections."""
    return """# DOE-TOOLKIT DESIGN
# Version: 1.0
# Generated: 2025-01-10T14:32:15Z
# Design Type: fractional_factorial
#
# FACTOR DEFINITIONS
# Name,Type,Changeability,Levels,Units
# Temperature,continuous,easy,150|200,°C
# Pressure,continuous,easy,50|100,psi
# Material,categorical,easy,A|B|C,
#
# RESPONSE DEFINITIONS
# Name,Units
# Yield,%
# Purity,mg/mL
#
# DESIGN DATA
StdOrder,RunOrder,Temperature,Pressure,Material,Yield,Purity
1,3,150,-1,A,,
2,1,200,1,B,,
3,4,150,1,C,,
4,2,200,-1,A,,
"""


@pytest.fixture
def csv_no_responses() -> str:
    """CSV without response definitions."""
    return """# DOE-TOOLKIT DESIGN
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


@pytest.fixture
def csv_discrete_factors() -> str:
    """CSV with discrete numeric factors."""
    return """# DOE-TOOLKIT DESIGN
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


class TestMetadataExtraction:
    """Tests for extracting metadata headers."""

    def test_extract_valid_metadata(self, valid_doe_csv: str) -> None:
        """Test extracting valid metadata block."""
        lines = valid_doe_csv.split("\n")
        metadata = extract_metadata_block(lines)

        assert metadata["version"] == "1.0"
        assert metadata["design_type"] == "fractional_factorial"
        assert "generated" in metadata

    def test_missing_header_raises_error(self) -> None:
        """Test that missing DOE-TOOLKIT header raises error."""
        lines = [
            "# SOME OTHER CSV",
            "# Name,Type",
            "Factor1,continuous",
        ]

        with pytest.raises(CSVParseError, match="Missing DOE-TOOLKIT header"):
            extract_metadata_block(lines)

    def test_metadata_with_spaces_in_keys(self) -> None:
        """Test metadata keys with spaces are converted to underscores."""
        lines = [
            "# DOE-TOOLKIT DESIGN",
            "# Design Type: full_factorial",
        ]
        metadata = extract_metadata_block(lines)
        assert "design_type" in metadata


class TestFactorExtraction:
    """Tests for extracting factor definitions."""

    def test_extract_continuous_factors(self, valid_doe_csv: str) -> None:
        """Test parsing continuous factor definitions."""
        lines = valid_doe_csv.split("\n")
        factors = extract_factor_definitions(lines)

        assert len(factors) == 3
        assert factors[0].name == "Temperature"
        assert factors[0].factor_type == FactorType.CONTINUOUS
        assert factors[0].min_value == 150
        assert factors[0].max_value == 200
        assert factors[0].units == "°C"

    def test_extract_categorical_factors(self, valid_doe_csv: str) -> None:
        """Test parsing categorical factor definitions."""
        lines = valid_doe_csv.split("\n")
        factors = extract_factor_definitions(lines)

        material_factor = factors[2]
        assert material_factor.name == "Material"
        assert material_factor.factor_type == FactorType.CATEGORICAL
        # Categorical factors store levels in the 'levels' attribute
        assert material_factor.levels == ["A", "B", "C"]

    def test_extract_discrete_factors(self, csv_discrete_factors: str) -> None:
        """Test parsing discrete numeric factor definitions."""
        lines = csv_discrete_factors.split("\n")
        factors = extract_factor_definitions(lines)

        assert len(factors) == 2
        assert factors[0].name == "RPM"
        assert factors[0].factor_type == FactorType.DISCRETE_NUMERIC
        assert factors[0].levels == [100.0, 150.0, 200.0]

    def test_missing_factor_section_raises_error(self) -> None:
        """Test that missing factor section raises error."""
        lines = [
            "# DOE-TOOLKIT DESIGN",
            "# DESIGN DATA",
            "Factor1,Factor2",
        ]

        with pytest.raises(CSVParseError, match="FACTOR DEFINITIONS section missing"):
            extract_factor_definitions(lines)

    def test_empty_factor_section_raises_error(self) -> None:
        """Test that empty factor section raises error."""
        lines = [
            "# DOE-TOOLKIT DESIGN",
            "# FACTOR DEFINITIONS",
            "# Name,Type,Changeability,Levels,Units",
            "#",
            "# DESIGN DATA",
        ]

        with pytest.raises(CSVParseError, match="No factors defined in FACTOR DEFINITIONS section"):
            extract_factor_definitions(lines)


class TestResponseExtraction:
    """Tests for extracting response definitions."""

    def test_extract_response_definitions(self, valid_doe_csv: str) -> None:
        """Test parsing response definitions."""
        lines = valid_doe_csv.split("\n")
        responses = extract_response_definitions(lines)

        assert len(responses) == 2
        assert responses[0]["name"] == "Yield"
        assert responses[0]["units"] == "%"
        assert responses[1]["name"] == "Purity"
        assert responses[1]["units"] == "mg/mL"

    def test_missing_response_section_returns_empty(self, csv_no_responses: str) -> None:
        """Test that missing response section returns empty list."""
        lines = csv_no_responses.split("\n")
        responses = extract_response_definitions(lines)

        assert responses == []

    def test_response_without_units(self) -> None:
        """Test response definition without units."""
        lines = [
            "# DOE-TOOLKIT DESIGN",
            "# RESPONSE DEFINITIONS",
            "# Name,Units",
            "# Yield,",
        ]
        responses = extract_response_definitions(lines)

        assert len(responses) == 1
        assert responses[0]["name"] == "Yield"
        assert responses[0]["units"] is None


class TestDesignDataExtraction:
    """Tests for extracting design data."""

    def test_extract_design_data(self, valid_doe_csv: str) -> None:
        """Test extracting design data section."""
        lines = valid_doe_csv.split("\n")
        design = extract_design_data(lines)

        assert len(design) == 4
        assert list(design.columns) == ["StdOrder", "RunOrder", "Temperature", "Pressure", "Material", "Yield", "Purity"]
        assert design["Temperature"].iloc[0] == 150
        assert design["Material"].iloc[0] == "A"

    def test_missing_design_data_raises_error(self) -> None:
        """Test that missing DESIGN DATA section raises error."""
        lines = [
            "# DOE-TOOLKIT DESIGN",
            "# FACTOR DEFINITIONS",
            "# Name,Type,Changeability,Levels,Units",
        ]

        with pytest.raises(CSVParseError, match="DESIGN DATA section missing"):
            extract_design_data(lines)

    def test_empty_design_data_raises_error(self) -> None:
        """Test that empty DESIGN DATA section raises error."""
        lines = [
            "# DOE-TOOLKIT DESIGN",
            "# DESIGN DATA",
            "# (no data)",
        ]

        with pytest.raises(CSVParseError, match="DESIGN DATA section empty"):
            extract_design_data(lines)


class TestFullParsing:
    """Integration tests for full CSV parsing."""

    def test_parse_valid_doe_csv(self, valid_doe_csv: str) -> None:
        """Test parsing complete valid CSV."""
        result = parse_doe_csv(valid_doe_csv)

        assert result.is_valid
        assert result.error is None
        assert len(result.factors) == 3
        assert len(result.response_definitions) == 2
        assert len(result.design_data) == 4

    def test_parse_csv_no_responses(self, csv_no_responses: str) -> None:
        """Test parsing CSV without response definitions."""
        result = parse_doe_csv(csv_no_responses)

        assert result.is_valid
        assert len(result.response_definitions) == 0

    def test_parse_invalid_csv_returns_error(self) -> None:
        """Test that invalid CSV returns error in result."""
        invalid_csv = "# INVALID HEADER\nsome,data"
        result = parse_doe_csv(invalid_csv)

        assert not result.is_valid
        assert result.error is not None

    def test_parse_discrete_factors(self, csv_discrete_factors: str) -> None:
        """Test parsing CSV with discrete factors."""
        result = parse_doe_csv(csv_discrete_factors)

        assert result.is_valid
        assert result.factors[0].factor_type == FactorType.DISCRETE_NUMERIC


class TestValidation:
    """Tests for CSV structure validation."""

    def test_validate_valid_structure(self, valid_doe_csv: str) -> None:
        """Test validating valid CSV structure."""
        result = parse_doe_csv(valid_doe_csv)
        is_valid, errors = validate_csv_structure(result)

        assert is_valid
        assert errors == []

    def test_validate_matching_factors(self, valid_doe_csv: str) -> None:
        """Test validation with matching session factors."""
        result = parse_doe_csv(valid_doe_csv)
        is_valid, errors = validate_csv_structure(result, session_factors=result.factors)

        assert is_valid
        assert errors == []

    def test_validate_mismatched_factor_count(self, valid_doe_csv: str) -> None:
        """Test validation with factor count mismatch."""
        result = parse_doe_csv(valid_doe_csv)
        fewer_factors = result.factors[:2]

        is_valid, errors = validate_csv_structure(result, session_factors=fewer_factors)

        assert not is_valid
        assert any("Factor count mismatch" in e for e in errors)

    def test_validate_mismatched_factor_names(self, valid_doe_csv: str) -> None:
        """Test validation with factor name mismatch."""
        result = parse_doe_csv(valid_doe_csv)
        
        # Create factors with different names
        different_factors = [
            Factor(
                name="Temp",  # Different name
                factor_type=FactorType.CONTINUOUS,
                levels=[150, 200]
            ),
            Factor(
                name="Press",  # Different name
                factor_type=FactorType.CONTINUOUS,
                levels=[50, 100]
            ),
            Factor(
                name="Mat",  # Different name
                factor_type=FactorType.CATEGORICAL,
                levels=["A", "B", "C"]
            ),
        ]

        is_valid, errors = validate_csv_structure(result, session_factors=different_factors)

        assert not is_valid
        assert any("Factor name mismatch" in e for e in errors)

    def test_validate_with_parse_error(self) -> None:
        """Test validation with parse error."""
        invalid_result = parse_doe_csv("# INVALID")
        is_valid, errors = validate_csv_structure(invalid_result)

        assert not is_valid
        assert any("Parse error" in e for e in errors)


class TestGeneration:
    """Tests for generating DOE CSV files."""

    def test_generate_doe_csv_with_responses(self) -> None:
        """Test generating CSV with response definitions."""
        design = pd.DataFrame({
            "StdOrder": [1, 2],
            "RunOrder": [1, 2],
            "Temperature": [150, 200],
        })
        factors = [
            Factor(
                name="Temperature",
                factor_type=FactorType.CONTINUOUS,
                changeability=ChangeabilityLevel.EASY,
                levels=[150, 200],
                units="°C",
            )
        ]
        responses = [{"name": "Yield", "units": "%"}]

        csv = generate_doe_csv(design, factors, responses, "full_factorial")

        assert "# DOE-TOOLKIT DESIGN" in csv
        assert "# FACTOR DEFINITIONS" in csv
        assert "# RESPONSE DEFINITIONS" in csv
        assert "Temperature,continuous,easy,150|200,°C" in csv
        assert "Yield,%" in csv
        assert "# DESIGN DATA" in csv
        assert "Yield" in csv.split("# DESIGN DATA")[1]  # In data section header

    def test_generate_doe_csv_without_responses(self) -> None:
        """Test generating CSV without response definitions."""
        design = pd.DataFrame({
            "StdOrder": [1],
            "RunOrder": [1],
            "Temp": [150],
        })
        factors = [
            Factor(
                name="Temp",
                factor_type=FactorType.CONTINUOUS,
                changeability=ChangeabilityLevel.EASY,
                levels=[150, 200],
            )
        ]

        csv = generate_doe_csv(design, factors)

        assert "# RESPONSE DEFINITIONS" not in csv
        assert "# DESIGN DATA" in csv

    def test_generate_csv_roundtrip(self) -> None:
        """Test that generated CSV can be parsed back."""
        design = pd.DataFrame({
            "StdOrder": [1, 2],
            "RunOrder": [1, 2],
            "Temperature": [150, 200],
        })
        factors = [
            Factor(
                name="Temperature",
                factor_type=FactorType.CONTINUOUS,
                changeability=ChangeabilityLevel.EASY,
                levels=[150, 200],
                units="°C",
            )
        ]
        responses = [{"name": "Yield", "units": "%"}]

        csv = generate_doe_csv(design, factors, responses)
        result = parse_doe_csv(csv)

        assert result.is_valid
        assert len(result.factors) == 1
        assert result.factors[0].name == "Temperature"
        assert len(result.response_definitions) == 1
        assert result.response_definitions[0]["name"] == "Yield"