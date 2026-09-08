"""
Alias and correlation display component for design preview and diagnostics.

Provides a shared ``display_alias_correlation`` function that renders:
- A blue-to-red absolute-correlation heatmap for all design types.
- A flagged pairs table for |r| >= 0.5.
- Exact alias chains (for fractional factorial designs).
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.core.factors import Factor

# Threshold above which a term pair is flagged as potentially aliased
_ALIAS_FLAG_THRESHOLD: float = 0.5


def display_alias_correlation(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    design_type: str,
    alias_structure: Optional[Dict[str, List[str]]] = None,
    resolution: Optional[int] = None,
    section_title: str = "🔗 Alias / Correlation Structure",
    expanded: bool = False,
) -> None:
    """
    Render alias and correlation structure inside a Streamlit expander.

    For all design types a model-term correlation heatmap and flagged-pairs
    table are shown.  For fractional factorial designs the exact alias chains
    stored in *alias_structure* are also displayed.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix (actual factor values).
    factors : List[Factor]
        Factor definitions used to build the model matrix.
    model_terms : List[str]
        Model terms (patsy notation) to include in the correlation matrix.
        The intercept term ``'1'`` is automatically excluded.
    design_type : str
        Design type label (e.g. ``'Fractional Factorial'``, ``'D-Optimal'``).
    alias_structure : Dict[str, List[str]], optional
        Pre-computed alias chains from the fractional-factorial engine.
        Displayed only when *design_type* contains ``'Fractional'``.
    resolution : int, optional
        Design resolution; displayed in the header when provided.
    section_title : str, optional
        Title shown on the expander header.
    expanded : bool, optional
        Whether the expander is open by default.

    Returns
    -------
    None
        Renders content directly into the active Streamlit context.

    Examples
    --------
    >>> display_alias_correlation(
    ...     design=design_df,
    ...     factors=factors,
    ...     model_terms=["A", "B", "A*B"],
    ...     design_type="Fractional Factorial",
    ...     alias_structure={"A": ["BCD"], "B": ["ACD"]},
    ...     resolution=4,
    ... )
    """
    with st.expander(section_title, expanded=expanded):
        _render_body(
            design=design,
            factors=factors,
            model_terms=model_terms,
            design_type=design_type,
            alias_structure=alias_structure,
            resolution=resolution,
        )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _render_body(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
    design_type: str,
    alias_structure: Optional[Dict[str, List[str]]],
    resolution: Optional[int],
) -> None:
    """Render the full alias/correlation body (no expander wrapper)."""

    # Header line
    res_label = f" — Resolution {resolution}" if resolution else ""
    st.caption(
        f"Design type: **{design_type or 'Unknown'}**{res_label}. "
        "Values shown are absolute Pearson correlations between model columns."
    )

    # Build model matrix columns
    col_matrix, col_labels = _build_model_columns(design, factors, model_terms)

    if col_matrix is None or col_matrix.shape[1] < 2:
        st.info("Not enough model terms to compute a correlation matrix.")
        return

    # Correlation matrix
    corr = _compute_correlation_matrix(col_matrix)

    # Heatmap
    st.markdown("**Term Correlation Heatmap** (|r|)")
    fig = _build_heatmap(corr, col_labels)
    st.plotly_chart(fig, width="stretch", theme=None)

    # Flagged pairs table
    flagged = _find_flagged_pairs(corr, col_labels)
    if flagged:
        st.markdown(
            f"**Flagged pairs** (|r| ≥ {_ALIAS_FLAG_THRESHOLD:.1f}) — "
            "these terms share substantial variance:"
        )
        flagged_df = pd.DataFrame(
            flagged, columns=["Term A", "Term B", "|r|"]
        ).sort_values("|r|", ascending=False)
        st.dataframe(flagged_df, hide_index=True, width="stretch")
    else:
        st.success(
            f"✅ No term pairs with |r| ≥ {_ALIAS_FLAG_THRESHOLD:.1f}. "
            "Model terms are well-separated."
        )

    # Exact alias chains (fractional factorial only)
    if design_type and "Fractional" in design_type and alias_structure:
        st.divider()
        _render_alias_chains(alias_structure, factors)


def _build_model_columns(
    design: pd.DataFrame,
    factors: List[Factor],
    model_terms: List[str],
) -> tuple:
    """
    Evaluate model terms against the design matrix.

    Parameters
    ----------
    design : pd.DataFrame
        Design matrix with actual factor values.
    factors : List[Factor]
        Factor definitions.
    model_terms : List[str]
        Terms in patsy-style notation; ``'1'`` is skipped.

    Returns
    -------
    tuple[Optional[np.ndarray], List[str]]
        ``(matrix, labels)`` where *matrix* is shape ``(n_runs, n_terms)``
        and *labels* are display-ready term names.  Returns ``(None, [])``
        on failure.

    Notes
    -----
    Terms that cannot be evaluated are silently skipped.
    """
    cols: List[np.ndarray] = []
    labels: List[str] = []

    for term in model_terms:
        if term == "1":
            continue

        col = _evaluate_term(term, design, factors)
        if col is not None:
            cols.append(col)
            labels.append(_format_term_label(term))

    if len(cols) < 2:
        return None, []

    return np.column_stack(cols), labels


def _evaluate_term(
    term: str,
    design: pd.DataFrame,
    factors: List[Factor],
) -> Optional[np.ndarray]:
    """
    Return a numeric column for *term* evaluated on *design*.

    Supports main effects, two-factor interactions (``'A*B'``), and
    quadratic terms (``'I(A**2)'``).  Returns ``None`` for any term that
    cannot be evaluated.

    Parameters
    ----------
    term : str
        Model term in patsy notation.
    design : pd.DataFrame
        Design matrix.
    factors : List[Factor]
        Factor definitions for coding continuous factors.

    Returns
    -------
    Optional[np.ndarray]
        Numeric array of length ``n_runs``, or ``None``.
    """
    try:
        # Quadratic: I(A**2)
        if term.startswith("I(") and "**2" in term:
            factor_name = term[2:].replace("**2)", "").strip()
            if factor_name not in design.columns:
                return None
            return design[factor_name].values.astype(float) ** 2

        # Interaction: A*B
        if "*" in term:
            parts = [p.strip() for p in term.split("*")]
            cols = []
            for part in parts:
                if part not in design.columns:
                    return None
                cols.append(design[part].values.astype(float))
            result = cols[0]
            for c in cols[1:]:
                result = result * c
            return result

        # Main effect
        if term in design.columns:
            return design[term].values.astype(float)

    except Exception:
        pass

    return None


def _compute_correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Compute absolute Pearson correlation matrix.

    Parameters
    ----------
    matrix : np.ndarray
        Shape ``(n_runs, n_terms)``.

    Returns
    -------
    np.ndarray
        Symmetric matrix of shape ``(n_terms, n_terms)`` with values in [0, 1].

    Notes
    -----
    Zero-variance columns yield ``NaN`` correlations, which are replaced with
    ``0.0`` to keep the heatmap renderable.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix.T)
    corr = np.abs(corr)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def _build_heatmap(corr: np.ndarray, labels: List[str]) -> go.Figure:
    """
    Build a Plotly heatmap from an absolute correlation matrix.

    The diagonal is masked (shown as white / 0) so self-correlations do not
    distort the colour scale.

    Parameters
    ----------
    corr : np.ndarray
        Absolute correlation matrix, shape ``(n, n)``.
    labels : List[str]
        Axis tick labels, length ``n``.

    Returns
    -------
    go.Figure
        Configured Plotly figure.
    """
    display = corr.copy()
    np.fill_diagonal(display, np.nan)  # hide diagonal

    n = len(labels)
    cell_size = max(40, min(80, 600 // max(n, 1)))
    fig_size = n * cell_size + 120

    fig = go.Figure(
        go.Heatmap(
            z=display,
            x=labels,
            y=labels,
            colorscale=[
                [0.0, "#2166ac"],   # deep blue  -> low correlation
                [0.5, "#f7f7f7"],   # white      -> moderate
                [1.0, "#b2182b"],   # deep red   -> high correlation
            ],
            zmin=0,
            zmax=1,
            colorbar=dict(title="|r|", thickness=14, len=0.8),
            text=np.where(
                np.isnan(display),
                "",
                np.vectorize(lambda v: f"{v:.2f}")(display),
            ),
            texttemplate="%{text}",
            hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>|r| = %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        height=fig_size,
        width=fig_size,
        margin=dict(l=120, r=40, t=40, b=120),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#fafafa", size=11),
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )

    return fig


def _find_flagged_pairs(
    corr: np.ndarray,
    labels: List[str],
) -> List[tuple]:
    """
    Collect off-diagonal term pairs where |r| >= threshold.

    Parameters
    ----------
    corr : np.ndarray
        Absolute correlation matrix.
    labels : List[str]
        Term labels.

    Returns
    -------
    List[tuple[str, str, float]]
        List of ``(label_a, label_b, abs_r)`` for upper-triangle pairs only.
    """
    flagged = []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            val = float(corr[i, j])
            if val >= _ALIAS_FLAG_THRESHOLD:
                flagged.append((labels[i], labels[j], round(val, 3)))
    return flagged


def _render_alias_chains(
    alias_structure: Dict[str, List[str]],
    factors: List[Factor],
) -> None:
    """
    Display exact alias chains for fractional factorial designs.

    Algebraic symbols are translated to real factor names where possible.
    Only main-effect aliases (i.e. effects aliased with other main effects or
    two-factor interactions) are shown to keep the output concise.

    Parameters
    ----------
    alias_structure : Dict[str, List[str]]
        Mapping from effect string to list of aliased effect strings.
        Keys and values use single-letter algebraic notation (A, B, AB, ...).
    factors : List[Factor]
        Factor definitions used to translate symbols to real names.

    Returns
    -------
    None
    """
    st.markdown("**Exact Alias Chains** (from defining relation)")

    if not alias_structure:
        st.info("No alias chains available.")
        return

    # Build symbol -> name map
    sym_map: Dict[str, str] = {
        chr(65 + i): f.name for i, f in enumerate(factors)
    }

    def _translate(effect: str) -> str:
        """Translate algebraic effect string to factor names."""
        parts = [sym_map.get(ch, ch) for ch in effect]
        return "*".join(parts)

    rows = []
    for effect, aliases in sorted(
        alias_structure.items(), key=lambda x: (len(x[0]), x[0])
    ):
        # Only show up to 3rd-order interactions to avoid clutter
        if len(effect) > 3:
            continue
        if not aliases:
            continue
        rows.append(
            {
                "Effect": _translate(effect),
                "Aliased With": " + ".join(_translate(a) for a in aliases),
            }
        )

    if rows:
        alias_df = pd.DataFrame(rows)
        st.dataframe(alias_df, hide_index=True, width="stretch")
    else:
        st.info("No alias chains up to 3rd order.")


def _format_term_label(term: str) -> str:
    """
    Convert a patsy-notation term to a compact display label.

    Parameters
    ----------
    term : str
        Patsy term, e.g. ``'Temperature*Pressure'`` or ``'I(A**2)'``.

    Returns
    -------
    str
        Short label, e.g. ``'Temp×Press'`` or ``'A²'``.

    Notes
    -----
    Long factor names are truncated to 8 characters to keep the heatmap
    axis labels readable.
    """
    if term.startswith("I(") and "**2" in term:
        name = term[2:].replace("**2)", "").strip()
        return f"{_truncate(name)}²"

    if "*" in term:
        parts = [_truncate(p.strip()) for p in term.split("*")]
        return "×".join(parts)

    return _truncate(term)


def _truncate(name: str, max_len: int = 8) -> str:
    """Truncate a factor name to *max_len* characters."""
    return name if len(name) <= max_len else name[:max_len - 1] + "…"
