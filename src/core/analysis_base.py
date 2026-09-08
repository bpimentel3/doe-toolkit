"""
Shared primitives for ANOVA analysis modules.

This module holds the types and pure-utility functions that are needed by
*both* ``analysis.py`` (fixed-effects / mixed-effects ANOVA) and
``split_plot_analysis.py`` (two-strata split-plot ANOVA).  Keeping them here
breaks the circular import that would otherwise exist between those two
modules.

Contents
--------
- ``ANOVAResults``     – dataclass container for all ANOVA outputs.
- ``parse_model_term`` – parse a patsy-notation term into constituent factors.
- ``enforce_hierarchy``– ensure hierarchical completeness of a model term list.
- ``quadratic``        – convenience helper for quadratic term notation.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.core.factors import Factor


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class ANOVAResults:
    """
    Container for ANOVA model results.

    Attributes
    ----------
    anova_table : pd.DataFrame
        ANOVA table.  For split-plot models this is the two-strata table with
        a ``Stratum`` column; for fixed-effects models it is the standard
        Type II table from statsmodels.
    effect_estimates : pd.DataFrame
        Coefficient table with columns ``Coefficient``, ``Std_Error``,
        ``t_value``, ``p_value``.
    logworth : pd.DataFrame
        Per-term ``LogWorth`` (``-log10(p)``).
    residuals : np.ndarray
        Model residuals (subplot level for split-plot).
    fitted_values : np.ndarray
        Fitted/predicted values corresponding to each observation.
    fitted_model : object
        The underlying statsmodels result object (OLS or MixedLM).
    diagnostics : Dict[str, any]
        Diagnostic test results, e.g. ``{'shapiro_wilk': {'statistic': ..., 'p_value': ...}}``.
    model_terms : List[str]
        Model terms used in the fit (patsy notation, including ``'1'`` for intercept).
    is_split_plot : bool
        ``True`` when the two-strata split-plot estimator was used.
    r_squared : float
        Coefficient of determination.
    adj_r_squared : float
        Adjusted R².
    rmse : float
        Root mean squared error of residuals.
    coefficient_significance : Optional[pd.DataFrame]
        Coefficient-level significance table (one row per fitted coefficient,
        p-values from :attr:`fitted_model.pvalues`).
    anova_effect_summary : Optional[pd.DataFrame]
        Term-level ANOVA effect summary (one row per ANOVA term, p-values and
        F-statistics sourced from :attr:`anova_table`).
    """

    anova_table: pd.DataFrame
    effect_estimates: pd.DataFrame
    logworth: pd.DataFrame
    residuals: np.ndarray
    fitted_values: np.ndarray
    fitted_model: object
    diagnostics: Dict[str, object]
    model_terms: List[str]
    is_split_plot: bool
    r_squared: float
    adj_r_squared: float
    rmse: float
    coefficient_significance: Optional[pd.DataFrame] = None
    anova_effect_summary: Optional[pd.DataFrame] = None

    def predict_from_settings(self, settings: Dict[str, object]) -> float:
        """
        Predict the response for a dict of factor settings without calling patsy.

        Builds the model matrix row directly from ``self.model_terms`` and
        ``self.effect_estimates``, so it works regardless of which underlying
        statsmodels object is stored in ``fitted_model`` and never requires
        nuisance columns such as ``WholePlot`` to be present.

        Supported term types
        --------------------
        - Intercept  : ``'1'``
        - Main effect: ``'A'``
        - Interaction: ``'A*B'``
        - Quadratic  : ``'I(A**2)'``
        - Categorical main effect via dummy coding (patsy ``FactorName[T.Level]`` convention).

        Parameters
        ----------
        settings : Dict[str, object]
            Factor name -> value mapping (coded or actual; must match the
            scale used when the model was fitted).

        Returns
        -------
        float
            Predicted response value.

        Notes
        -----
        Categorical reference levels contribute only through the intercept.
        Non-reference levels are matched via ``FactorName[T.Level]`` keys
        in ``effect_estimates``.

        Examples
        --------
        >>> results.predict_from_settings({'Temperature': -1.0, 'Time': 0.5})
        42.7
        """
        coefficients = self.effect_estimates["Coefficient"]
        prediction = 0.0

        for term in self.model_terms:
            if term == "1":
                intercept_key = next(
                    (k for k in coefficients.index if k in ("Intercept", "const")),
                    None,
                )
                if intercept_key is not None:
                    prediction += float(coefficients[intercept_key])
                continue

            factor_list, operator = parse_model_term(term)

            if operator == "**":
                # Quadratic: I(A**2)
                fname = factor_list[0]
                val = float(settings.get(fname, 0))  # type: ignore[arg-type]
                coef_key = next(
                    (k for k in coefficients.index
                     if k == term or f"I({fname} ** 2)" in k),
                    None,
                )
                if coef_key is not None:
                    prediction += float(coefficients[coef_key]) * val ** 2

            elif operator == "*":
                # Two-way interaction (A*B -> patsy stores as A:B).
                # Continuous factors enter as scalar * value; categorical
                # factors match their dummy key F[T.level]. patsy expands a
                # categorical interaction into dummies with the all-reference
                # baseline suppressed.
                prediction += self._predict_interaction(
                    factor_list, term, settings, coefficients
                )

            else:
                # Main effect - continuous or categorical
                fname = factor_list[0]
                val = settings.get(fname)
                if val is None:
                    continue

                if fname in coefficients.index:
                    # Numeric (continuous / discrete)
                    prediction += float(coefficients[fname]) * float(val)  # type: ignore[arg-type]
                else:
                    # Categorical dummy: "Catalyst[T.B]"
                    dummy_key = f"{fname}[T.{val}]"
                    if dummy_key in coefficients.index:
                        prediction += float(coefficients[dummy_key])
                    # Reference level contributes 0 (absorbed in intercept)

        return prediction

    def _predict_interaction(
        self,
        factor_list: List[str],
        term: str,
        settings: Dict[str, object],
        coefficients: pd.Series,
    ) -> float:
        """
        Predict the contribution of a two-way interaction term.

        Handles continuous/continuous (single ``A:B`` coefficient) as well as
        categorical interactions (patsy expands into ``C(A)[T.x]:C(B)[T.y]``
        dummies with the all-reference baseline suppressed; the plain ``A:B``
        key corresponds to a categorical main effect evaluated at its
        reference level in the other dimension).

        Each factor in the interaction is matched against the levels in
        ``settings``.  Continuous factors are treated as scalar variables; a
        match is required for categorical ones.
        """
        total = 0.0
        f1, f2 = factor_list
        v1 = settings.get(f1, 0)
        v2 = settings.get(f2, 0)

        # Classify each side as continuous (scalar) or categorical (has
        # dummy keys of the form F[T.level] in the coefficient index).
        def is_categorical(name: str) -> bool:
            cat_level_key = f"{name}[T."
            return any(
                str(k).startswith(cat_level_key)
                for k in coefficients.index
            )

        cat1 = is_categorical(f1)
        cat2 = is_categorical(f2)

        if not cat1 and not cat2:
            # Both continuous: single coefficient A:B
            coef_key = next(
                (k for k in coefficients.index
                 if k == term or k == term.replace("*", ":")),
                None,
            )
            if coef_key is not None:
                return float(coefficients[coef_key]) * float(v1) * float(v2)
            return 0.0

        # At least one categorical. Build the exact patsy interaction key(s)
        # for the selected levels and sum the matching coefficients.
        #   categorical-only : A[T.x]:B[T.y]   (order matches term)
        #   mixed            : A[T.x]:B  or  A:B[T.y]
        def side_key(name: str, val: object, cat: bool) -> str:
            return f"{name}[T.{val}]" if cat else name

        k1 = side_key(f1, v1, cat1)
        k2 = side_key(f2, v2, cat2)

        candidates = {f"{k1}:{k2}", f"{k2}:{k1}"}
        for k in coefficients.index:
            if str(k) in candidates:
                total += float(coefficients[k])
        return total


# ---------------------------------------------------------------------------
# Term parsing
# ---------------------------------------------------------------------------

# Transform prefixes recognised by parse_model_term.
# Maps Patsy prefix → (numpy function name used for factor extraction).
_TRANSFORM_PREFIXES: Tuple[str, ...] = (
    "np.log(",
    "np.sqrt(",
    "np.exp(",
    "I(1/",
)


def _extract_transform_factor(term: str) -> Optional[str]:
    """
    Return the raw factor name embedded in a transform term, or None.

    Handles the four recognised transform forms::

        np.log(A)   → 'A'
        np.sqrt(A)  → 'A'
        np.exp(A)   → 'A'
        I(1/A)      → 'A'

    The term must start with one of the known prefixes; the factor name is
    the content between the prefix and the matching closing parenthesis.

    Parameters
    ----------
    term : str
        A single Patsy fragment (no cross ``*`` operators).

    Returns
    -------
    str or None
        Raw factor name, or ``None`` if not a recognised transform.
    """
    for prefix in _TRANSFORM_PREFIXES:
        if term.startswith(prefix):
            # Strip prefix and trailing ')'
            inner = term[len(prefix):]
            if inner.endswith(')'):
                inner = inner[:-1]
            return inner.strip()
    return None


def _split_at_outer_star(term: str) -> Tuple[str, str]:
    """
    Split a term at the first ``*`` that sits outside all parentheses.

    Returns ``(left, right)`` where ``right`` is everything after the ``*``.
    If no outer ``*`` exists, returns ``(term, ''')``.

    Used to separate ``np.log(A)*B`` into ``('np.log(A)', 'B')``.
    """
    depth = 0
    for i, ch in enumerate(term):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '*' and depth == 0:
            return term[:i], term[i + 1:]
    return term, ''


def parse_model_term(term: str) -> Tuple[List[str], str]:
    """
    Parse a patsy-notation model term into its constituent factors and operator.

    Recognised term forms
    ---------------------
    - Main effect         : ``'A'``              → ``(['A'], '')``
    - Interaction         : ``'A*B'``            → ``(['A', 'B'], '*')``
    - Quadratic           : ``'I(A**2)'``        → ``(['A'], '**')``
    - Transform           : ``'np.log(A)'``      → ``(['A'], 'transform')``
    - Transform power     : ``'I(np.log(A)**2)'``→ ``(['A'], 'transform_power')``
    - Transform cross     : ``'np.log(A)*B'``    → ``(['A', 'B'], 'transform_cross')``
    - Transform pwr cross : ``'I(np.log(A)**2)*B'`` → ``(['A', 'B'], 'transform_power_cross')``

    The ``factor_list`` always contains raw design factor names (e.g.
    ``'Temperature'``), never the transform wrapper strings.

    Parameters
    ----------
    term : str
        A single model term in patsy notation.

    Returns
    -------
    factor_list : List[str]
        Names of the constituent factor(s).
    operator : str
        One of ``''``, ``'*'``, ``'**'``, ``'transform'``,
        ``'transform_power'``, ``'transform_cross'``,
        ``'transform_power_cross'``.

    Examples
    --------
    >>> parse_model_term('Temperature')
    (['Temperature'], '')
    >>> parse_model_term('Temperature*Pressure')
    (['Temperature', 'Pressure'], '*')
    >>> parse_model_term('I(Temperature**2)')
    (['Temperature'], '**')
    >>> parse_model_term('np.log(Temperature)')
    (['Temperature'], 'transform')
    >>> parse_model_term('np.log(Temperature)*Pressure')
    (['Temperature', 'Pressure'], 'transform_cross')
    >>> parse_model_term('I(np.log(Temperature)**2)*Pressure')
    (['Temperature', 'Pressure'], 'transform_power_cross')
    """
    # ------------------------------------------------------------------ #
    # I(...**n) — power wrapper; base may be a plain factor or transform  #
    # Examples: I(A**2), I(np.log(A)**2), I(np.log(A)**2)*B              #
    # ------------------------------------------------------------------ #
    if term.startswith('I(') and '**' in term:
        # Find the ')' that closes the outer I( by tracking paren depth.
        # term.index(')') would find the first ')' which may be inside a
        # nested transform such as I(np.log(A)**2).
        depth = 0
        close = -1
        for _i, _ch in enumerate(term):
            if _ch == '(':
                depth += 1
            elif _ch == ')':
                depth -= 1
                if depth == 0:
                    close = _i
                    break
        if close == -1:
            # Malformed term — fall through to main-effect
            return [term.strip()], ''
        power_content = term[2:close]           # e.g. 'A**2' or 'np.log(A)**2'
        remainder = term[close + 1:]            # e.g. '' or '*B'

        base_fragment, _ = power_content.rsplit('**', 1)
        base_fragment = base_fragment.strip()
        raw_factor = _extract_transform_factor(base_fragment)

        if remainder.startswith('*'):
            # Power × cross: I(A**2)*B  or  I(np.log(A)**2)*B
            cross_factor = remainder[1:].strip()
            if raw_factor is not None:
                return [raw_factor, cross_factor], 'transform_power_cross'
            return [base_fragment, cross_factor], '**'

        if raw_factor is not None:
            return [raw_factor], 'transform_power'
        return [base_fragment], '**'

    # ------------------------------------------------------------------ #
    # Transform terms (no I() power wrapper)                              #
    # Examples: np.log(A), np.log(A)*B                                   #
    # ------------------------------------------------------------------ #
    for prefix in _TRANSFORM_PREFIXES:
        if term.startswith(prefix):
            base_fragment, remainder = _split_at_outer_star(term)
            raw_factor = _extract_transform_factor(base_fragment)
            if raw_factor is None:
                break  # Malformed — fall through to main-effect
            if remainder:
                cross_factor = remainder.strip()
                return [raw_factor, cross_factor], 'transform_cross'
            return [raw_factor], 'transform'

    # ------------------------------------------------------------------ #
    # Plain interaction: A*B  (checked AFTER transforms to avoid          #
    # misclassifying np.log(A)*B as an interaction)                      #
    # ------------------------------------------------------------------ #
    if '*' in term:
        factor_list = [_unwrap_c(seg) for seg in term.split('*')]
        return factor_list, '*'

    # ------------------------------------------------------------------ #
    # Plain main effect: A (or C(A) for categorical factors)              #
    # ------------------------------------------------------------------ #
    return [_unwrap_c(term.strip())], ''


def _unwrap_c(segment: str) -> str:
    """
    Strip a patsy ``C(...)`` wrapper from a factor segment so its underlying
    factor name can be found in the factor registry.

    Examples
    --------
    >>> _unwrap_c('C(Egg_lot)')
    'Egg_lot'
    >>> _unwrap_c('Egg_lot')
    'Egg_lot'
    """
    segment = segment.strip()
    if segment.startswith('C(') and segment.endswith(')'):
        return segment[2:-1].strip()
    return segment


# ---------------------------------------------------------------------------
# Coefficient-to-term matching and effect summary builders
#
# These helpers live here so that both the fixed-effects path
# (``src.core.analysis``) and the split-plot path
# (``src.core.split_plot_analysis``) share the same term-matching logic and
# construct the same canonical dataframes.
#
# Two deliberately separate tables are produced:
#   * ``build_coefficient_significance`` — one row per fitted COEFFICIENT,
#     p-values kept from ``fitted_model.pvalues``.
#   * ``build_anova_effect_summary`` — one row per term-level ANOVA test,
#     p-values and F-statistics kept from the displayed ANOVA table.
# The mapping between the two never replaces one statistic with the other.
# ---------------------------------------------------------------------------


#: Centralised policy: fixed Block terms are excluded from the Bonferroni
#: multiplicity family (``m``).  The family is intended for experimental
#: effects, not design/nuisance terms.
BONFERRONI_EXCLUDE_BLOCK = True


def _strip_all_c_wrappers(label: str) -> str:
    """Remove every patsy ``C(...)`` wrapper from a label.

    Examples
    --------
    ``C(Egg_lot)[T.41007666]`` -> ``Egg_lot[T.41007666]``
    """
    return re.sub(r'C\(([^()]*)\)', r'\1', str(label))


def _normalize_term(term) -> str:
    """Collapse a term label to a canonical string for comparisons."""
    return re.sub(r'\s+', '', _strip_all_c_wrappers(str(term)))


def _term_factor_segments(term) -> List[str]:
    """
    Extract the ordered factor segments of a term, dropping categorical dummy
    encodings (``Factor[T.Level]``) and interaction separators.
    """
    t = _normalize_term(term)
    t = re.sub(r'\[T\..*?\]', '', t)
    t = t.replace('*', ':')
    m = re.match(r'I\((.+?)\s*\*\*\s*2\)', t)
    if m:
        return [m.group(1).strip()]
    return [s.strip() for s in t.split(':') if s.strip()]


def _term_sort_key(term):
    """Order-independent key for a term (``A:B`` == ``B:A``)."""
    return tuple(sorted(_term_factor_segments(term)))


def _classify_term_type(term) -> str:
    """Classify a term as main / interaction / quadratic / other."""
    t = re.sub(r'\[T\..*?\]', '', _normalize_term(term))
    if re.match(r'^I\(.+\)$', t):
        return 'quadratic'
    segs = [s for s in t.replace('*', ':').split(':') if s]
    if len(segs) > 1:
        return 'interaction'
    return 'main'


def _is_block_term(term, block_factor_names: Tuple[str, ...] = ('Block',)) -> bool:
    """True when *term* involves one of the design/block factor names."""
    for seg in _term_factor_segments(term):
        if seg in block_factor_names:
            return True
    return False


# Rows in an ANOVA table that never represent experimental effects.
NON_EFFECT_ANOVA_ROWS = frozenset({
    'Intercept', 'const', 'Residual', 'WholePlot Error', 'SubPlot Error',
    'Whole Plot Error', 'Sub Plot Error', 'Pure Error', 'Lack of Fit',
    'Cor Total', 'Model',
})


def _is_non_effect_row(term) -> bool:
    if term in NON_EFFECT_ANOVA_ROWS:
        return True
    low = str(term).lower().replace(' ', '')
    return low in {'intercept', 'residual', 'wholeplaterror', 'subplaterror',
                   'pureerror', 'lackoffit', 'cortotal', 'model'}


def _safe_float(value) -> Optional[float]:
    try:
        f = float(value)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coefficient_logworth(p) -> float:
    """
    LogWorth for a p-value with safe handling of underflow.

    ``p <= 0`` / NaN yields NaN.  Otherwise ``-log10(max(p, tiny))``.
    """
    if p is None:
        return np.nan
    try:
        p = float(p)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(p) or p <= 0:
        return np.nan
    p_safe = max(p, np.finfo(float).tiny)
    return -np.log10(p_safe)


def find_effect_estimate_key(source: str, index: pd.Index) -> Optional[str]:
    """
    Map an ANOVA term / coefficient label back to a coefficient table index.

    Handles patsy/statsmodels conventions:
    - direct matches (``'Temperature'``)
    - ``'*'`` vs ``':'`` interaction separators
    - interaction-order normalisation (``'A:B'`` == ``'B:A'``)
    - categorical dummies (``'Atmosphere'`` -> ``'Atmosphere[T.nitrogen]'`` or
      ``'C(Atmosphere)[T.air]'``)
    - nested/institution interaction dummies
      (``'Temperature:Atmosphere'`` -> ``'Temperature:Atmosphere[T.nitrogen]'``)

    Returns
    -------
    The matching index key, or ``None`` when no safe match exists.
    """
    index = list(index)
    if source in index:
        return source
    patsy_form = str(source).replace('*', ':')
    if patsy_form in index:
        return patsy_form

    src_key = _term_sort_key(source)
    for key in index:
        if _term_sort_key(key) == src_key:
            return key

    base = _strip_all_c_wrappers(patsy_form)
    for key in index:
        if key.startswith(base) or key.startswith(patsy_form):
            return key
    return None


def find_parent_anova_term(coef_name: str, anova_terms) -> Optional[str]:
    """
    Map a fitted coefficient label to its parent ANOVA term.

    Used for labelling / effect-sign recovery only.  Never replaces the
    coefficient's own p-value with the parent term's ANOVA p-value.
    """
    anova_terms = list(anova_terms)
    if coef_name in anova_terms:
        return coef_name
    patsy_form = str(coef_name).replace('*', ':')
    if patsy_form in anova_terms:
        return patsy_form

    coef_key = _term_sort_key(coef_name)
    for term in anova_terms:
        if _term_sort_key(term) == coef_key:
            return term

    base = _normalize_term(coef_name)
    base = re.sub(r'\[T\..*?\]', '', base).replace('*', ':')
    for term in anova_terms:
        if _normalize_term(term) == base:
            return term
    return None


def attach_critical_limits(
    summary: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    (Re)compute per-term ``t_critical`` and ``bonferroni_limit`` columns.

    Residual degrees of freedom are taken per-term from ``residual_df`` so the
    limits are correct for multi-strata (split-plot) models where different
    terms use different error strata.

    ``m`` (the Bonferroni multiplicity family size) is the number of eligible
    experimental effects in *summary*.  Per :data:`BONFERRONI_EXCLUDE_BLOCK`,
    fixed Block terms are excluded from ``m`` by default.  Callers should call
    this again after filtering (e.g. hiding Block) so the limits track the
    displayed effect set.
    """
    if summary is None or summary.empty:
        return summary
    df = summary.copy()
    if BONFERRONI_EXCLUDE_BLOCK:
        m = int(np.sum(~df['is_block'].astype(bool)))
    else:
        m = int(len(df))
    m = max(m, 1)

    t_crits, bonfs = [], []
    for _, row in df.iterrows():
        rd = row.get('residual_df', np.nan)
        if rd is None:
            t_crits.append(np.nan)
            bonfs.append(np.nan)
            continue
        try:
            rd = float(rd)
        except (TypeError, ValueError):
            t_crits.append(np.nan)
            bonfs.append(np.nan)
            continue
        if not np.isfinite(rd) or rd <= 0:
            t_crits.append(np.nan)
            bonfs.append(np.nan)
            continue
        tc = stats.t.ppf(1 - alpha / 2, rd)
        bc = stats.t.ppf(1 - alpha / (2 * m), rd)
        t_crits.append(tc)
        bonfs.append(bc)
    df['t_critical'] = t_crits
    df['bonferroni_limit'] = bonfs
    return df


def build_anova_effect_summary(
    anova_table: Optional[pd.DataFrame],
    effect_estimates: Optional[pd.DataFrame] = None,
    block_factor_names: Tuple[str, ...] = ('Block',),
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Build the canonical term-level ANOVA effect summary.

    Source of truth: the displayed :attr:`anova_table` (``PR(>F)`` for
    fixed-effects statsmodels tables, ``P`` for split-plot tables).  One row
    per eligible ANOVA term (intercept/residual/error rows excluded).

    Standardized statistic:
    - ``df == 1``: ``sign(effect_estimate) * sqrt(F)`` (``|t| = sqrt(F)``),
      with sign recovered from the matched coefficient when available.
    - ``df > 1``: ``sqrt(F)`` labelled ``omnibus sqrt(F)``; multi-DF terms do
      NOT have a single coefficient t-statistic and are never assigned a sign.
    """
    if anova_table is None or anova_table.empty:
        return pd.DataFrame()

    p_col = ('PR(>F)' if 'PR(>F)' in anova_table.columns
             else 'P' if 'P' in anova_table.columns else None)
    has_stratum = 'Stratum' in anova_table.columns
    ss_col = 'sum_sq' if 'sum_sq' in anova_table.columns else 'SS'

    def residual_df_for(stratum) -> Optional[float]:
        if not has_stratum:
            if 'Residual' in anova_table.index:
                return _safe_float(anova_table.loc['Residual', 'df'])
            return None
        err_row = ('WholePlot Error' if stratum == 'Whole-Plot'
                   else 'SubPlot Error')
        if err_row in anova_table.index:
            return _safe_float(anova_table.loc[err_row, 'df'])
        return None

    coef_index = effect_estimates.index if effect_estimates is not None else None

    rows = []
    for term, row in anova_table.iterrows():
        if _is_non_effect_row(term):
            continue

        df = _safe_int(row.get('df'))
        F = _safe_float(row.get('F'))
        p = _safe_float(row.get(p_col)) if p_col else None
        stratum = row.get('Stratum') if has_stratum else None
        residual_df = residual_df_for(stratum)

        effect_estimate = np.nan
        effect_sign = np.nan
        standardized = np.nan
        stat_type = 'sqrt(F) not applicable'
        if df is not None and F is not None and F >= 0:
            if df == 1:
                key = None
                if coef_index is not None and len(coef_index):
                    key = find_effect_estimate_key(term, coef_index)
                if key is not None:
                    try:
                        coeff = float(effect_estimates.loc[key, 'Coefficient'])
                    except (KeyError, TypeError, ValueError):
                        coeff = np.nan
                    if np.isfinite(coeff):
                        effect_estimate = coeff
                        effect_sign = np.sign(coeff)
                standardized = np.sqrt(float(F))
                if effect_sign is not None and np.isfinite(effect_sign):
                    standardized = effect_sign * standardized
                stat_type = '|t| = sqrt(F)'
            elif df > 1:
                standardized = np.sqrt(float(F))
                stat_type = 'omnibus sqrt(F)'
        elif df is None or F is None:
            stat_type = 'unavailable'

        rows.append({
            'term': term,
            'normalized_term': _normalize_term(term),
            'term_type': _classify_term_type(term),
            'sum_sq': _safe_float(row.get(ss_col)),
            'df': df,
            'F': F,
            'p_value': p,
            'effect_estimate': effect_estimate,
            'effect_sign': effect_sign,
            'standardized_statistic': standardized,
            'standardized_statistic_type': stat_type,
            'residual_df': residual_df,
            'is_block': _is_block_term(term, block_factor_names),
            'source': 'displayed term-level ANOVA table',
        })

    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.set_index('term', drop=False)
        summary = attach_critical_limits(summary, alpha=alpha)
    return summary


def build_coefficient_significance(
    effect_estimates: Optional[pd.DataFrame],
    anova_table: Optional[pd.DataFrame] = None,
    block_factor_names: Tuple[str, ...] = ('Block',),
) -> pd.DataFrame:
    """
    Build the canonical coefficient-level significance table.

    One row per fitted coefficient (intercept excluded), p-values kept from
    the fitted model's coefficient table.  ``parent_anova_term`` is populated
    for labelling only and never replaces the coefficient's own statistics.
    """
    if effect_estimates is None or effect_estimates.empty:
        return pd.DataFrame()

    anova_terms = list(anova_table.index) if (anova_table is not None
                                              and not anova_table.empty) else []

    rows = []
    for name, row in effect_estimates.iterrows():
        if name in ('Intercept', 'const'):
            continue
        coeff = _safe_float(row.get('Coefficient'))
        p = _safe_float(row.get('p_value'))
        parent = find_parent_anova_term(name, anova_terms) if anova_terms else None
        parent_df = None
        if parent and anova_table is not None and parent in anova_table.index:
            parent_df = _safe_int(anova_table.loc[parent, 'df'])
        rows.append({
            'coefficient_name': name,
            'normalized_coefficient_name': _normalize_term(name),
            'parent_anova_term': parent,
            'parent_anova_df': parent_df,
            'coefficient_estimate': coeff,
            'standard_error': _safe_float(row.get('Std_Error')),
            't_value': _safe_float(row.get('t_value')),
            'p_value': p,
            'logworth': coefficient_logworth(p),
            'sign': np.sign(coeff) if coeff is not None and np.isfinite(coeff) else np.nan,
            'is_block': _is_block_term(name, block_factor_names),
            'source': 'Fitted-model coefficient test',
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Coded-to-actual-unit coefficient conversion
# ---------------------------------------------------------------------------


def _parse_index_term(term_name: str) -> Tuple[List[str], str]:
    """
    Parse a statsmodels/patsy coefficient index label into factor names and
    operator.

    Handles the following patsy output conventions:
    - ``'Intercept'`` / ``'const'``  → ``([], '')``
    - ``'A'``                         → ``(['A'], '')``
    - ``'A[T.level]'``                → ``(['A'], '')``  (categorical dummy)
    - ``'A:B'``                        → ``(['A', 'B'], ':')``
    - ``'I(A ** 2)'``                 → ``(['A'], '**')``

    Parameters
    ----------
    term_name : str
        A single index label from ``fitted_model.params``.

    Returns
    -------
    factor_names : List[str]
        Extracted factor name(s).
    operator : str
        ``':'`` for interactions, ``'**'`` for quadratics, ``''`` otherwise.
    """
    if term_name in ("Intercept", "const"):
        return [], ""

    quad_match = re.match(r"I\((.+?)\s*\*\*\s*2\)", term_name)
    if quad_match:
        return [quad_match.group(1).strip()], "**"

    if ":" in term_name and not term_name.startswith("I("):
        parts = term_name.split(":")
        clean = [re.sub(r"\[T\..*?\]", "", p).strip() for p in parts]
        return clean, ":"

    cat_match = re.match(r"^([^\[]+)\[T\.", term_name)
    if cat_match:
        return [cat_match.group(1).strip()], ""

    return [term_name.strip()], ""


def compute_actual_coefficients(
    effect_estimates: pd.DataFrame,
    factors: List[Factor],
) -> pd.DataFrame:
    """
    Convert coded-unit coefficients and standard errors to actual units.

    The model is fit on coded factors where each continuous factor ``x`` is
    transformed via ``x_coded = (x_actual - center) / half_range``.  This
    function inverts that scaling so coefficients reflect original units.

    Transformation rules
    --------------------
    - **Intercept**: adjusted for all centering shifts.
    - **Main effect** (continuous): ``b_actual = b_coded / half_range``
    - **Interaction A:B** (both continuous): ``b_actual = b_coded / (hr_A*hr_B)``
    - **Quadratic I(A**2)**: ``b_actual = b_coded / hr_A**2``
    - **Categorical / discrete-numeric** terms: no single scale exists; the
      corresponding ``Actual_Coefficient`` / ``Actual_Std_Error`` are set to
      NaN (an explicit "unavailable" value) rather than a misleading number.

    Parameters
    ----------
    effect_estimates : pd.DataFrame
        Coefficient table with at minimum ``Coefficient`` and ``Std_Error``
        columns and term names as the index (patsy convention).
    factors : List[Factor]
        Factor definitions used to retrieve ``min_value`` / ``max_value``.

    Returns
    -------
    pd.DataFrame
        Copy with added ``Actual_Coefficient`` and ``Actual_Std_Error``.
    """
    scale: Dict[str, Tuple[float, float]] = {}
    for f in factors:
        if f.is_continuous() and f.min_value is not None and f.max_value is not None:
            center = (f.min_value + f.max_value) / 2.0
            half_range = (f.max_value - f.min_value) / 2.0
            if half_range > 0:
                scale[f.name] = (center, half_range)

    result = effect_estimates.copy()
    actual_coefs: List[float] = []
    actual_ses: List[float] = []

    intercept_adjustment = 0.0
    for term_name in effect_estimates.index:
        if term_name in ("Intercept", "const"):
            continue
        _factors, _op = _parse_index_term(term_name)
        if _op == "" and len(_factors) == 1:
            fname = _factors[0]
            if fname in scale:
                center, half_range = scale[fname]
                b_coded = float(effect_estimates.loc[term_name, "Coefficient"])
                intercept_adjustment += b_coded * (-center / half_range)

    for term_name in effect_estimates.index:
        b_coded = float(effect_estimates.loc[term_name, "Coefficient"])
        se_coded = float(effect_estimates.loc[term_name, "Std_Error"])

        if term_name in ("Intercept", "const"):
            actual_coefs.append(b_coded + intercept_adjustment)
            actual_ses.append(se_coded)
            continue

        factor_names_parsed, op = _parse_index_term(term_name)

        if op == "**":
            fname = factor_names_parsed[0]
            if fname in scale:
                _, half_range = scale[fname]
                s = 1.0 / (half_range ** 2)
                actual_coefs.append(b_coded * s)
                actual_ses.append(se_coded * s)
            else:
                actual_coefs.append(float("nan"))
                actual_ses.append(float("nan"))
        elif op == ":":
            if len(factor_names_parsed) == 2:
                fa, fb = factor_names_parsed
                if fa in scale and fb in scale:
                    _, hr_a = scale[fa]
                    _, hr_b = scale[fb]
                    s = 1.0 / (hr_a * hr_b)
                    actual_coefs.append(b_coded * s)
                    actual_ses.append(se_coded * s)
                else:
                    actual_coefs.append(float("nan"))
                    actual_ses.append(float("nan"))
            else:
                actual_coefs.append(float("nan"))
                actual_ses.append(float("nan"))
        elif op == "":
            fname = factor_names_parsed[0]
            if fname in scale:
                _, half_range = scale[fname]
                s = 1.0 / half_range
                actual_coefs.append(b_coded * s)
                actual_ses.append(se_coded * s)
            else:
                actual_coefs.append(float("nan"))
                actual_ses.append(float("nan"))
        else:
            actual_coefs.append(float("nan"))
            actual_ses.append(float("nan"))

    result["Actual_Coefficient"] = actual_coefs
    result["Actual_Std_Error"] = actual_ses
    return result


# ---------------------------------------------------------------------------
# Hierarchy enforcement
# ---------------------------------------------------------------------------


def enforce_hierarchy(
    terms: List[str],
    factor_names: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Ensure hierarchical completeness and canonical ordering of model terms.

    For every interaction or quadratic term present, all lower-order parents
    (main effects) are added if missing.  The returned list is ordered as:
    intercept → main effects → two-way interactions → quadratic terms →
    any remaining terms.

    Parameters
    ----------
    terms : List[str]
        Current model terms (patsy notation).
    factor_names : List[str]
        All factor names in the design (used to validate that auto-added
        parents actually exist in the design).

    Returns
    -------
    ordered_terms : List[str]
        Deduplicated, hierarchy-complete, canonically ordered term list.
    added_terms : List[str]
        Terms that were inserted to satisfy hierarchy (may be empty).

    Examples
    --------
    >>> enforce_hierarchy(['1', 'A*B'], ['A', 'B'])
    (['1', 'A', 'B', 'A*B'], ['A', 'B'])
    """
    complete_terms = list(terms)
    added_terms: List[str] = []
    complete_terms_set: set = set(terms)
    factor_names_set: set = set(factor_names)

    for term in terms:
        if term == '1':
            continue

        factor_list, operator = parse_model_term(term)

        if operator in ('*', '**'):
            for factor_name in factor_list:
                if (
                    factor_name not in complete_terms_set
                    and factor_name in factor_names_set
                ):
                    complete_terms.append(factor_name)
                    added_terms.append(factor_name)
                    complete_terms_set.add(factor_name)

    # Deduplicate while preserving first-seen order
    seen: set = set()
    unique_terms = [t for t in complete_terms if not (t in seen or seen.add(t))]  # type: ignore[func-returns-value]

    # Canonical ordering: intercept → main → interaction → quadratic → other
    ordered: List[str] = []

    if '1' in unique_terms:
        ordered.append('1')

    for term in unique_terms:
        if term == '1':
            continue
        factor_list, op = parse_model_term(term)
        if op == '' and len(factor_list) == 1:
            ordered.append(term)

    for term in unique_terms:
        if term == '1':
            continue
        factor_list, op = parse_model_term(term)
        if op == '*' and len(factor_list) == 2:
            ordered.append(term)

    for term in unique_terms:
        if term == '1':
            continue
        factor_list, op = parse_model_term(term)
        if op == '**':
            ordered.append(term)

    for term in unique_terms:
        if term not in ordered:
            ordered.append(term)

    return ordered, added_terms


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------



def quadratic(factor_name: str) -> str:
    """
    Return the patsy quadratic term notation for a factor.

    Parameters
    ----------
    factor_name : str
        Name of a continuous factor.

    Returns
    -------
    str
        Patsy identity-function notation, e.g. ``'I(Temperature**2)'``.

    Examples
    --------
    >>> quadratic('Temperature')
    'I(Temperature**2)'
    """
    return f"I({factor_name}**2)"
