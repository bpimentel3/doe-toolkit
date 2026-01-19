"""
CSV parser for DOE-Toolkit design files with metadata headers.

Supports parsing DOE-Toolkit CSV format with metadata block containing
factor definitions, response definitions, and design structure.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import re
from io import StringIO

from src.core.factors import Factor, FactorType, ChangeabilityLevel


@dataclass
class ParseResult:
    """Result of parsing a DOE CSV file."""

    metadata: Dict[str, str]
    factors: List[Factor]
    response_definitions: List[Dict[str, Optional[str]]]
    design_data: pd.DataFrame
    error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        """Check if parse was successful."""
        return self.error is None


class CSVParseError(Exception):
    """Raised when CSV parsing fails."""

    pass


def parse_doe_csv(file_content: str) -> ParseResult:
    """
    Parse DOE-Toolkit CSV with metadata header.

    Attempts to parse metadata block first. If metadata is malformed,
    falls back to plain CSV parsing (assumes factors already defined).

    Parameters
    ----------
    file_content : str
        Raw CSV file content as string.

    Returns
    -------
    ParseResult
        Parsed metadata, factors, responses, and design data.
        If error occurred, error field is populated.

    Notes
    -----
    CSV format:
    # DOE-TOOLKIT DESIGN
    # Version: 1.0
    # FACTOR DEFINITIONS
    # Name,Type,Changeability,Levels,Units
    # Temperature,continuous,easy,150|200,°C
    #
    # RESPONSE DEFINITIONS
    # Name,Units
    # Yield,%
    #
    # DESIGN DATA
    StdOrder,RunOrder,Factor1,Factor2,Response1,Response2
    """
    lines = file_content.strip().split("\n")

    try:
        metadata = extract_metadata_block(lines)
        factors = extract_factor_definitions(lines)
        response_definitions = extract_response_definitions(lines)
        design_data = extract_design_data(lines)

        return ParseResult(
            metadata=metadata,
            factors=factors,
            response_definitions=response_definitions,
            design_data=design_data,
            error=None,
        )

    except CSVParseError as e:
        return ParseResult(
            metadata={},
            factors=[],
            response_definitions=[],
            design_data=pd.DataFrame(),
            error=str(e),
        )


def extract_metadata_block(lines: List[str]) -> Dict[str, str]:
    """
    Extract metadata from header comments.

    Looks for lines like:
    # Version: 1.0
    # Generated: 2025-01-10T14:32:15Z
    # Design Type: fractional_factorial

    Parameters
    ----------
    lines : List[str]
        Raw CSV lines.

    Returns
    -------
    Dict[str, str]
        Metadata key-value pairs.

    Raises
    ------
    CSVParseError
        If no metadata header found.
    """
    metadata = {}

    if not lines or not lines[0].startswith("# DOE-TOOLKIT DESIGN"):
        raise CSVParseError("Missing DOE-TOOLKIT header")

    for line in lines:
        if not line.startswith("#"):
            break

        # Skip section headers and empty comments
        if any(
            x in line
            for x in ["FACTOR DEFINITIONS", "RESPONSE DEFINITIONS", "DESIGN DATA"]
        ):
            continue
        if line.strip() == "#":
            continue

        # Parse key-value pairs
        match = re.match(r"#\s*([A-Za-z ]+):\s*(.+)", line)
        if match:
            key = match.group(1).strip().lower().replace(" ", "_")
            value = match.group(2).strip()
            # Remove trailing commas from value
            value = value.rstrip(',')
            metadata[key] = value

    return metadata


def extract_factor_definitions(lines: List[str]) -> List[Factor]:
    """
    Parse FACTOR DEFINITIONS section and return Factor objects.

    Format:
    # FACTOR DEFINITIONS
    # Name,Type,Changeability,Levels,Units
    # Temperature,continuous,easy,150|200,°C
    # Material,categorical,easy,A|B|C,

    Parameters
    ----------
    lines : List[str]
        Raw CSV lines.

    Returns
    -------
    List[Factor]
        Parsed Factor objects.

    Raises
    ------
    CSVParseError
        If factor definitions malformed or missing.
    """
    factors = []
    in_section = False
    header_found = False

    for i, line in enumerate(lines):
        if "# FACTOR DEFINITIONS" in line:
            in_section = True
            continue

        if in_section and (line.startswith("# RESPONSE DEFINITIONS") or line.startswith("# DESIGN DATA")):
            break

        if not in_section:
            continue
        
        if not line.startswith("#"):
            break

        # Skip header and empty lines
        if "Name,Type" in line:
            header_found = True
            continue
        # Empty comment line (with or without trailing commas)
        if line.lstrip('#').strip().replace(',', '') == '':
            continue

        # Parse factor line
        try:
            factor = _parse_factor_line(line)
            factors.append(factor)
        except ValueError as e:
            raise CSVParseError(f"Invalid factor definition at line {i}: {e}")

    if not header_found:
        raise CSVParseError("FACTOR DEFINITIONS section missing or malformed")
    if not factors:
        raise CSVParseError("No factors defined in FACTOR DEFINITIONS section")

    return factors


def _parse_factor_line(line: str) -> Factor:
    """
    Parse a single factor definition line.

    Format: # Name,Type,Changeability,Levels,Units

    Parameters
    ----------
    line : str
        Comment line like "# Temperature,continuous,easy,150|200,°C"

    Returns
    -------
    Factor
        Parsed Factor object.

    Raises
    ------
    ValueError
        If line malformed.
    """
    # Remove comment prefix
    content = line.lstrip("#").strip()

    # Split and filter out empty trailing fields (from trailing commas)
    parts = [p.strip() for p in content.split(",")]
    # Remove trailing empty strings but keep intentional empty units field
    while len(parts) > 4 and parts[-1] == '':
        parts.pop()
    
    if len(parts) < 4:
        raise ValueError(f"Expected at least 4 fields, got {len(parts)}")

    name, type_str, changeability_str, levels_str = (
        parts[0],
        parts[1],
        parts[2],
        parts[3],
    )
    units = parts[4] if len(parts) > 4 and parts[4] else None

    # Validate and convert type
    try:
        factor_type = FactorType(type_str.lower())
    except ValueError:
        raise ValueError(
            f"Invalid factor type '{type_str}'. "
            f"Must be one of: {[t.value for t in FactorType]}"
        )

    # Validate and convert changeability
    try:
        changeability = ChangeabilityLevel(changeability_str.lower())
    except ValueError:
        raise ValueError(
            f"Invalid changeability '{changeability_str}'. "
            f"Must be one of: {[c.value for c in ChangeabilityLevel]}"
        )

    # Parse levels based on factor type
    if factor_type == FactorType.CONTINUOUS:
        level_values = levels_str.split("|")
        if len(level_values) != 2:
            raise ValueError(f"Continuous factor must have min|max, got {levels_str}")
        try:
            levels = [float(level_values[0]), float(level_values[1])]
        except ValueError:
            raise ValueError(f"Continuous levels must be numeric: {levels_str}")

        if levels[0] >= levels[1]:
            raise ValueError(f"Min must be < max: {levels[0]} >= {levels[1]}")

    elif factor_type == FactorType.DISCRETE_NUMERIC:
        try:
            levels = [float(x.strip()) for x in levels_str.split("|")]
        except ValueError:
            raise ValueError(f"Discrete numeric levels must be numeric: {levels_str}")
        
        if len(levels) < 2:
            raise ValueError(
                f"Discrete factor must have at least 2 levels, got {levels_str}"
            )

    elif factor_type == FactorType.CATEGORICAL:
        levels = [x.strip() for x in levels_str.split("|")]
        if len(levels) < 2:
            raise ValueError(
                f"Categorical factor must have at least 2 levels, got {levels_str}"
            )

    else:
        raise ValueError(f"Unhandled factor type: {factor_type}")

    return Factor(
        name=name,
        factor_type=factor_type,
        changeability=changeability,
        levels=levels,
        units=units if units else None,
        _validate_on_init=False,  # Validation already done during parsing
    )


def extract_response_definitions(lines: List[str]) -> List[Dict[str, Optional[str]]]:
    """
    Parse RESPONSE DEFINITIONS section.

    Format:
    # RESPONSE DEFINITIONS
    # Name,Units
    # Yield,%
    # Purity,mg/mL

    If section missing, returns empty list (responses optional on import).

    Parameters
    ----------
    lines : List[str]
        Raw CSV lines.

    Returns
    -------
    List[Dict[str, Optional[str]]]
        List of dicts with 'name' and 'units' keys.
    """
    responses = []
    in_section = False
    header_found = False

    for line in lines:
        if "# RESPONSE DEFINITIONS" in line:
            in_section = True
            continue

        if in_section and line.startswith("# DESIGN DATA"):
            break

        if not in_section:
            continue
        
        if not line.startswith("#"):
            break

        # Skip header and empty lines
        if "Name,Units" in line:
            header_found = True
            continue
        # Empty comment line (with or without trailing commas)
        if line.lstrip('#').strip().replace(',', '') == '':
            continue

        # Parse response line
        content = line.lstrip("#").strip()
        parts = [p.strip() for p in content.split(",")]
        
        # Remove trailing empty fields
        while len(parts) > 1 and parts[-1] == '':
            parts.pop()

        if len(parts) < 1:
            continue

        name = parts[0]
        units = parts[1] if len(parts) > 1 and parts[1] else None

        responses.append({"name": name, "units": units if units else None})

    return responses


def extract_design_data(lines: List[str]) -> pd.DataFrame:
    """
    Extract design data section (after # DESIGN DATA).

    Parameters
    ----------
    lines : List[str]
        Raw CSV lines.

    Returns
    -------
    pd.DataFrame
        Design data with factor and response columns.

    Raises
    ------
    CSVParseError
        If design data section missing or empty.
    """
    design_start_idx = None

    for i, line in enumerate(lines):
        if "# DESIGN DATA" in line:
            design_start_idx = i + 1
            break

    if design_start_idx is None:
        raise CSVParseError("DESIGN DATA section missing")

    # Extract non-comment lines from design section
    data_lines = []
    for line in lines[design_start_idx:]:
        if line.startswith("#"):
            continue
        if line.strip():
            data_lines.append(line)

    if not data_lines:
        raise CSVParseError("DESIGN DATA section empty")

    # Parse as CSV
    try:
        design_data = pd.read_csv(StringIO("\n".join(data_lines)))
    except Exception as e:
        raise CSVParseError(f"Failed to parse design data: {e}")

    if design_data.empty:
        raise CSVParseError("DESIGN DATA section empty")

    return design_data


def validate_csv_structure(
    parsed: ParseResult, session_factors: Optional[List[Factor]] = None
) -> Tuple[bool, List[str]]:
    """
    Validate parsed CSV structure.

    Parameters
    ----------
    parsed : ParseResult
        Result from parse_doe_csv.
    session_factors : Optional[List[Factor]]
        Factors from current session (for comparison). If None, assumes fresh import.

    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, [error_messages])

    Checks
    ------
    - Metadata section present
    - Factor definitions complete and valid
    - Design data present
    - If session_factors provided: factor count and names match
    """
    errors = []

    if parsed.error:
        errors.append(f"Parse error: {parsed.error}")
        return False, errors

    if not parsed.factors:
        errors.append("No factors defined")

    if parsed.design_data.empty:
        errors.append("No design data present")

    # Check if factors match session (if provided)
    if session_factors is not None:
        if len(parsed.factors) != len(session_factors):
            errors.append(
                f"Factor count mismatch: CSV has {len(parsed.factors)}, "
                f"session has {len(session_factors)}"
            )
        else:
            for csv_factor, session_factor in zip(parsed.factors, session_factors):
                if csv_factor.name != session_factor.name:
                    errors.append(
                        f"Factor name mismatch: CSV has '{csv_factor.name}', "
                        f"session has '{session_factor.name}'"
                    )

    return len(errors) == 0, errors


def generate_doe_csv(
    design: pd.DataFrame,
    factors: List[Factor],
    response_definitions: Optional[List[Dict[str, Optional[str]]]] = None,
    design_type: str = "custom",
    design_metadata: Optional[Dict] = None,
) -> str:
    """
    Generate DOE-Toolkit CSV with metadata header.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with factor columns.
    factors : List[Factor]
        Factor definitions.
    response_definitions : Optional[List[Dict[str, Optional[str]]]]
        Response definitions with name and units.
    design_type : str
        Type of design (e.g., "full_factorial", "optimal").
    design_metadata : Optional[Dict]
        Additional metadata to include in header.

    Returns
    -------
    str
        CSV content with metadata header.
    """
    lines = []

    # Header
    lines.append("# DOE-TOOLKIT DESIGN")
    lines.append("# Version: 1.0")
    lines.append(f"# Design Type: {design_type}")

    if design_metadata:
        for key, value in design_metadata.items():
            if value is not None:
                formatted_key = key.replace("_", " ").title()
                lines.append(f"# {formatted_key}: {value}")

    # Factor definitions
    lines.append("#")
    lines.append("# FACTOR DEFINITIONS")
    lines.append("# Name,Type,Changeability,Levels,Units")

    for factor in factors:
        factor_type_str = factor.factor_type.value
        changeability_str = factor.changeability.value
        units_str = factor.units if factor.units else ""

        # Format levels based on type
        if factor.is_continuous():
            levels_str = f"{factor.levels[0]}|{factor.levels[1]}"
        else:  # discrete or categorical
            levels_str = "|".join(str(v) for v in factor.levels)

        lines.append(
            f"# {factor.name},{factor_type_str},{changeability_str},{levels_str},{units_str}"
        )

    # Response definitions
    if response_definitions:
        lines.append("#")
        lines.append("# RESPONSE DEFINITIONS")
        lines.append("# Name,Units")

        for response in response_definitions:
            units_str = response.get("units") or ""
            lines.append(f"# {response['name']},{units_str}")

    # Design data
    lines.append("#")
    lines.append("# DESIGN DATA")

    # Add response columns (empty) to design
    design_with_responses = design.copy()

    if response_definitions:
        for response in response_definitions:
            design_with_responses[response["name"]] = ""

    csv_data = design_with_responses.to_csv(index=False)
    lines.append(csv_data.rstrip())

    return "\n".join(lines)
