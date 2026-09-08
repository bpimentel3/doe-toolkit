"""
Response name definition helpers (pure, Streamlit-free).

Response names must be safe to embed in Patsy / statsmodels formula strings,
so free-form names are sanitized on save (mirroring the factors grid).  These
helpers are intentionally free of any Streamlit dependency so they can be
unit-tested directly.
"""

import re
from typing import Dict, List, Optional

import pandas as pd

from src.core.factor_naming import sanitize_factor_name, get_sanitization_report


def validate_response_name(name: str, existing_responses: list) -> bool:
    """
    Validate a sanitized response name.

    Rules:
    - Must be alphanumeric + underscore
    - Must not be duplicate (case-insensitive)
    - Must not be reserved word

    Parameters
    ----------
    name : str
        Response name (already sanitized to a valid identifier).
    existing_responses : list
        Iterable of ``{'name': ...}`` dicts (or objects with ``.name``) for the
        names already claimed in this batch.

    Returns
    -------
    bool
        ``True`` if the name may be used.
    """
    # Check format
    if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', name):
        return False

    # Check for duplicates (case-insensitive)
    existing_names = []
    for r in existing_responses:
        n = r['name'] if isinstance(r, dict) else getattr(r, 'name', '')
        existing_names.append(n.lower())
    if name.lower() in existing_names:
        return False

    # Check for reserved words
    reserved = {'I', 'C', 'Q', 'T'}  # patsy/pandas reserved
    if name in reserved:
        return False

    return True


def response_rows_to_definitions(df: pd.DataFrame):
    """
    Convert an editable responses table into sanitized response definitions.

    Accepts free-form response names (mirroring the factors grid): spaces,
    parens, and other formula-unsafe characters are auto-cleaned so the stored
    name is a valid Patsy-safe identifier. Returns the definitions list plus
    validation errors and sanitization warnings.

    Parameters
    ----------
    df : pd.DataFrame
        Editor rows with 'Name' and 'Units' columns.

    Returns
    -------
    definitions : List[Dict[str, Optional[str]]]
        Sanitized ``{'name': ..., 'units': ... or None}`` entries.
    errors : List[str]
        Human-readable validation errors (empty names, duplicates, reserved).
    sanitization_warnings : List[Dict]
        Per-row records of name sanitization for display.
    """
    definitions = []
    errors = []
    sanitization_warnings = []
    used_names = set()

    for idx, row in df.iterrows():
        raw = str(row['Name']).strip() if not pd.isna(row['Name']) else ''
        if not raw or raw.lower() in ('nan', 'none', ''):
            continue  # skip blank rows

        clean, was_modified = sanitize_factor_name(raw)

        if was_modified:
            report = get_sanitization_report(raw)
            sanitization_warnings.append({
                'row': idx + 1,
                'original': raw,
                'sanitized': clean,
                'changes': report['changes'],
            })

        if not validate_response_name(clean, [{'name': n} for n in used_names]):
            errors.append(
                f"Row {idx + 1}: response name '{raw}' "
                f"(cleaned: '{clean}') is invalid or already exists"
            )
            continue

        used_names.add(clean)

        units_raw = row.get('Units')
        units = str(units_raw).strip() if not pd.isna(units_raw) else ''
        definitions.append({
            'name': clean,
            'units': units if units else None,
        })

    return definitions, errors, sanitization_warnings