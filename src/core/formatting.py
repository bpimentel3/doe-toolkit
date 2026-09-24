"""
Presentation-only numeric formatting for statistical tables.

Everything here is display formatting; it never mutates the underlying
numeric values (``format_stat_table`` works on a copy) so exports, stored
values, and calculations are unaffected.

Shared conventions
------------------
- Missing values (``None`` / NaN / ``±inf``) render as ``"—"``.
- Sums of squares and mean squares: 2 decimal places.
- Degrees of freedom: integer when the value is integral.
- F-statistics: 2 decimal places.
- p-values: ``>= 0.0001`` -> 4 decimal places, ``< 0.0001`` -> ``"<0.0001"``.
- Coefficients / standard errors / confidence-interval bounds: 4 decimal
  places.
"""

from typing import Callable, Dict, Optional, Union

import numpy as np
import pandas as pd

NA_DISPLAY = "—"

#: p-values below this threshold render as the ``"<0.0001"`` sentinel.
P_MIN_DISPLAY = 1e-4


def format_na(value) -> str:
    """Return ``"—"`` for missing/non-finite values, else ``None``."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return NA_DISPLAY
    return NA_DISPLAY if not np.isfinite(f) else None


def _signed(value, decimals: int) -> str:
    return f"{value:.{decimals}f}"


def format_ss(value) -> str:
    """Sum of squares with 2 decimal places (missing -> ``"—"``)."""
    na = format_na(value)
    return NA_DISPLAY if na is not None else _signed(float(value), 2)


def format_ms(value) -> str:
    """Mean square with 2 decimal places (missing -> ``"—"``)."""
    na = format_na(value)
    return NA_DISPLAY if na is not None else _signed(float(value), 2)


def format_df(value) -> str:
    """Degrees of freedom as an integer when integral (missing -> ``"—"``)."""
    na = format_na(value)
    if na is not None:
        return NA_DISPLAY
    return str(int(float(value)))


def format_f(value) -> str:
    """F-statistic with 2 decimal places (missing -> ``"—"``)."""
    na = format_na(value)
    return NA_DISPLAY if na is not None else _signed(float(value), 2)


def format_p(value) -> str:
    """p-value: ``< 0.0001`` sentinel or 4 decimal places (missing -> ``"—"``)."""
    na = format_na(value)
    if na is not None:
        return NA_DISPLAY
    v = float(value)
    return "<0.0001" if v < P_MIN_DISPLAY else _signed(v, 4)


def format_coefficient(value) -> str:
    """Coefficient with 4 decimal places (missing -> ``"—"``)."""
    na = format_na(value)
    return NA_DISPLAY if na is not None else _signed(float(value), 4)


def format_std_error(value) -> str:
    """Standard error with 4 decimal places (missing -> ``"—"``)."""
    na = format_na(value)
    return NA_DISPLAY if na is not None else _signed(float(value), 4)


def format_ci_bound(value) -> str:
    """Confidence-interval bound with 4 decimal places (missing -> ``"—"``)."""
    na = format_na(value)
    return NA_DISPLAY if na is not None else _signed(float(value), 4)


#: Per-column formatters for ANOVA-style tables.  Handles both the standard
#: statsmodels ``anova_lm(typ=2)`` headers (``sum_sq``/``PR(>F)``) and the
#: split-plot assembly headers (``SS``/``P``).
ANOVA_TABLE_FORMATTERS: Dict[str, Callable[[object], str]] = {
    "sum_sq": format_ss,
    "SS": format_ss,
    "MS": format_ms,
    "df": format_df,
    "DF": format_df,
    "F": format_f,
    "PR(>F)": format_p,
    "P": format_p,
}

#: Per-column formatters for coefficient/effect tables.
COEFFICIENT_TABLE_FORMATTERS: Dict[str, Callable[[object], str]] = {
    "Coefficient": format_coefficient,
    "Actual_Coefficient": format_coefficient,
    "Std_Error": format_std_error,
    "Actual_Std_Error": format_std_error,
    "t_value": format_coefficient,
    "p_value": format_p,
    "p-value": format_p,
}


def format_stat_table(
    df: pd.DataFrame,
    column_spec: Dict[str, Callable[[object], str]],
) -> pd.DataFrame:
    """
    Return a presentation-only copy of ``df`` with cell-level formatting.

    Each column whose name appears in ``column_spec`` has its cells mapped
    through the matching formatter (missing values become ``"—"``); all other
    columns are left untouched so labels/identifiers stay as-is.  The source
    frame is never modified.
    """
    display = df.copy()
    for column, formatter in column_spec.items():
        if column in display.columns:
            display[column] = display[column].map(
                lambda v: formatter(v) if _is_missing(v) or isinstance(v, (int, float, np.number)) else v
            )
    return display


def format_p_value(value: Optional[Union[float, int]]) -> str:
    """Convenience alias for ``format_p`` (kept for callers that only need p)."""
    return format_p(value)


def _is_missing(value) -> bool:
    try:
        if value is None:
            return True
        f = float(value)
        return bool(np.isnan(f)) or bool(np.isinf(f))
    except (TypeError, ValueError):
        return False