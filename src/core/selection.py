"""
Automatic Model Selection for DOE designs (design-aware candidate pools).

Implements Design-Expert style forward / backward / stepwise / BIC selection
driven by ANOVA term p-values with STRONG HEREDITY (hierarchy) enforcement.

Design principles
-----------------
- Operates entirely on *term strings* (``'A'``, ``'A*B'``, ``'I(A**2)'``),
  never on coefficient rows, so continuous and categorical factors behave
  identically.
- A single significance basis: term significance always comes from the ANOVA
  p-values of the **maximal fit-able hierarchical candidate model** (the same
  fit that backs the selection table), never from isolated ``['1', term]``
  candidate fits.  This keeps selection consistent with the reported table and
  is robust to response surfaces whose curvature/interactions make marginal
  main-effect fits non-significant even though the effects are strongly
  significant in the full quadratic model.
- Uses the term-level ANOVA F-test p-values from ``ANOVAResults.anova_effect_summary``
  so multi-DF categorical terms have a single, meaningful p-value.
- Strong heredity is preserved: the final model is the hierarchy closure of the
  significant set, so a parent term required by an included interaction or
  quadratic is never dropped even when its own p-value exceeds alpha.
- The candidate pool is derived automatically from the factor mix
  (``candidate_model_pool``): designs with at least one continuous factor use
  mains + 2-way interactions (for all factors) + quadratics (continuous only);
  categorical-only designs use mains + 2-way interactions and never generate
  quadratic terms, so selection never crashes on datasets without continuous
  factors.
- Because the entry/exit criterion is the same full-model p-value, forward,
  backward and stepwise converge to the same significant, hierarchy-closed
  model (backward simply reports the reverse removal path).  Backward is the
  single recommended default for every DOE family.
- ``stepwise_bic`` runs the legacy BIC stepwise engine against the same
  design-aware candidate pool as an optional alternative.
"""

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings

from src.core.analysis import ANOVAAnalysis, ANOVAResults, generate_model_terms
from src.core.analysis_base import (
    enforce_hierarchy,
    find_effect_estimate_key,
    parse_model_term,
)
from src.core.diagnostics.summary import _compute_lof_p_value
from src.core.factor_naming import ALL_RESERVED
from src.core.stepwise import (
    get_candidate_terms_backward,
)

#: Methods supported by :func:`run_model_selection`.
_ALLOWED_METHODS = ('forward', 'backward', 'stepwise', 'stepwise_bic')

#: Default predicted-vs-adjusted R² gap that triggers the overfit warning.
PREDICTED_R2_GAP_THRESHOLD = 0.20

#: R² floor (R² / Adj R² / Pred R²) that triggers the near-perfect fit note.
NEAR_PERFECT_R2_THRESHOLD = 0.999

#: Cell text shown when lack-of-fit cannot be estimated.
LOF_UNAVAILABLE_TEXT = (
    "N/A (insufficient replicated runs to estimate pure error)"
)

#: Tooltip text explaining when lack-of-fit testing is possible.
LOF_HELP = (
    "Lack-of-fit testing requires replicated design points so that pure "
    "error can be estimated."
)


@dataclass
class SelectionStep:
    """A single add / remove step in a selection run.

    Attributes
    ----------
    step_number : int
        1-indexed iteration number.
    action : str
        ``'add'`` or ``'remove'``.
    term : str
        The term string that was added or removed.
    p_value : Optional[float]
        Term p-value that motivated the action.
    current_terms : List[str]
        Model terms after this step.
    """

    step_number: int
    action: str
    term: str
    p_value: Optional[float]
    current_terms: List[str]
    bic: Optional[float] = None
    delta_bic: Optional[float] = None


@dataclass
class ModelSelectionResults:
    """Complete automatic model-selection outcome.

    Attributes
    ----------
    method : str
        One of ``'forward'``, ``'backward'``, ``'stepwise'``.
    alpha : float
        Entry/exit significance threshold used for selection.
    final_terms : List[str]
        Selected hierarchical model terms (``'1'`` included).
    selection_path : List[SelectionStep]
        Record of every add/remove step.
    term_table : pd.DataFrame
        Per-candidate ``Term | Coefficient | p-value | Included | Reason``.
    final_results : ANOVAResults
        ANOVA results for the final model.
    model_quality : Dict[str, Optional[float]]
        Quality metrics for the final model (model p, LOF p, R², Adj R²,
        Predicted R², PRESS, predicted-R² discrepancy).
    n_iterations : int
        Number of steps executed.
    convergence_reason : str
        Why the procedure stopped.
    started_with_intercept_only : bool
        ``False`` when the intercept-only starting model could not be fitted
        and the fallback (best main effect) was used instead.
    oversaturated : bool
        ``True`` when the full candidate pool could not be fitted and the
        candidate table was pruned to a fit-able subset.
    warnings : List[str]
        Advisory messages surfaced to the user.
    bic_threshold : Optional[float]
        BIC improvement threshold for ``'stepwise_bic'`` runs; None otherwise.
    """

    method: str
    alpha: float
    final_terms: List[str]
    selection_path: List[SelectionStep]
    term_table: pd.DataFrame
    final_results: ANOVAResults
    model_quality: Dict[str, Optional[float]]
    n_iterations: int
    convergence_reason: str
    started_with_intercept_only: bool
    oversaturated: bool
    warnings: List[str]
    bic_threshold: Optional[float] = None


def candidate_model_pool(factors) -> Tuple[List[str], str]:
    """Derive the design-aware candidate pool and detected design type.

    Designs with at least one continuous factor use mains + 2-way
    interactions (for all factors) + quadratics for continuous factors only.
    Categorical-only designs fall back to mains + 2-way interactions and never
    generate quadratic terms, so model selection never raises the
    "quadratic requires a continuous factor" error.

    Returns ``(pool_without_intercept, design_type)`` where design_type is one
    of ``'Response Surface'``, ``'Categorical Factorial'`` or ``'Mixed'``.
    """
    has_continuous = any(f.is_continuous() for f in factors)
    has_non_continuous = any(not f.is_continuous() for f in factors)

    if not has_continuous:
        design_type = 'Categorical Factorial'
    elif not has_non_continuous:
        design_type = 'Response Surface'
    else:
        design_type = 'Mixed'

    model_type = 'quadratic' if has_continuous else 'interaction'
    pool = generate_model_terms(factors, model_type, include_intercept=True)
    return [t for t in pool if t != '1'], design_type


def _method_label(method: str) -> str:
    """Human label for a selection method key."""
    return {
        'forward': 'Forward',
        'backward': 'Backward',
        'stepwise': 'Stepwise',
        'stepwise_bic': 'Stepwise (BIC)',
    }.get(method, method.title())


def term_p_value(results: ANOVAResults, term: str) -> Optional[float]:
    """
    Return the term-level ANOVA p-value for *term* in a fitted model.

    Uses ``anova_effect_summary`` so multi-DF categorical terms have a single
    p-value.  Term strings are matched robustly (``*`` -> ``:``,
    ``I(A**2)`` -> ``I(A ** 2)``, order-insensitive).

    Returns None when the term is absent from the fit or has no usable p.
    """
    summary = getattr(results, 'anova_effect_summary', None)
    if summary is None or summary.empty:
        return None
    key = find_effect_estimate_key(term, summary.index)
    if key is None:
        return None
    p = summary.loc[key, 'p_value']
    return _safe_p(p)


def _safe_p(p) -> Optional[float]:
    """Coerce a p-value to float, or None for NaN/None/non-finite."""
    if p is None:
        return None
    try:
        v = float(p)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def _df_resid_ok(results: ANOVAResults) -> bool:
    """True when the fitted model retains residual degrees of freedom."""
    fm = getattr(results, 'fitted_model', None)
    dfr = getattr(fm, 'df_resid', None)
    if dfr is None:
        return True
    try:
        return float(dfr) >= 1
    except (TypeError, ValueError):
        return True


def _fit_candidate(analysis: ANOVAAnalysis, terms: List[str]) -> Optional[ANOVAResults]:
    """Fit a candidate model, returning None when it cannot be estimated.

    Warnings from the many internal candidate fits (low df, two-level
    quadratics, saturated models) are suppressed; the app surfaces warnings
    for the final model through its normal fit path.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            results = analysis.fit(terms, enforce_hierarchy_flag=True)
    except Exception:
        return None
    if not _df_resid_ok(results):
        return None
    return results


def required_by_hierarchy(terms: List[str], factor_names: List[str], term: str) -> bool:
    """True when *term* cannot be removed from ``terms`` without breaking heredity."""
    if term not in terms or term == '1':
        return False
    removable = get_candidate_terms_backward(terms, factor_names, ['1'])
    return term not in removable


def _significant_terms(
    full_results: Optional[ANOVAResults],
    candidate_pool: List[str],
    alpha: float,
) -> set:
    """
    Terms whose ANOVA p-value in the full candidate model is <= alpha.

    Significance is always judged in the *maximal fit-able hierarchical model*
    (the same fit that backs the selection table), never in isolated single-term
    fits.  This keeps selection consistent with what the table reports and is
    robust to response curvature/interactions that marginal entry tests miss.
    Terms absent from the fit (saturated) carry no p-value and are excluded.
    """
    if full_results is None:
        return set()
    return {
        term for term in candidate_pool
        if (p := term_p_value(full_results, term)) is not None and p <= alpha
    }


def _hierarchy_closure(base_terms: List[str], factor_names: List[str]) -> List[str]:
    """
    Minimal hierarchy-complete superset of *base_terms* (intercept included).

    Adds the main-effect parents of any included interaction/quadratic so that
    a significant higher-order term protects its non-significant parents.
    """
    ordered, _ = enforce_hierarchy(['1'] + [t for t in base_terms if t != '1'], factor_names)
    return list(ordered)


def forward_selection(
    anova_analysis: ANOVAAnalysis,
    candidate_pool: List[str],
    alpha: float = 0.10,
    max_iterations: int = 100,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[str], List[SelectionStep], bool, str]:
    """
    Forward selection driven by the full-model significance criterion.

    Term significance is taken from the maximal fit-able hierarchical candidate
    model (one fit), never from isolated ``['1', term]`` fits.  A term is
    selected when its full-model ANOVA p-value is <= alpha; the final model is
    the hierarchy closure of that significant set, so a significant interaction
    or quadratic keeps its (possibly non-significant) parent mains.

    This is robust to response surfaces where curvature/interactions dominate
    and marginal main-effect fits are non-significant even though the effects
    are strongly significant in the full quadratic model.

    Returns
    -------
    final_terms, steps, started_with_intercept_only, convergence_reason, pruned
    """
    factor_names = [f.name for f in anova_analysis.factors]

    full_terms, pruned = _fit_able_model(anova_analysis, candidate_pool)
    full_results = _fit_candidate(anova_analysis, full_terms)
    if full_results is None:
        raise ValueError("The full hierarchical candidate model could not be fitted.")

    significant = _significant_terms(full_results, candidate_pool, alpha)
    target = _hierarchy_closure(sorted(significant, key=lambda t: candidate_pool.index(t)), factor_names)
    target_set = set(target)

    steps: List[SelectionStep] = []
    current_terms = ['1']
    planned = [t for t in target if t != '1']
    total = max(1, len(planned))

    for iteration, term in enumerate(planned, start=1):
        if progress_callback:
            progress_callback(iteration, total)
        parents, _ = parse_model_term(term)
        if all(p in target_set for p in parents):
            p = term_p_value(full_results, term)
            current_terms = list(current_terms) + [term]
            steps.append(SelectionStep(
                len(steps) + 1, 'add', term, p, list(current_terms)
            ))

    convergence_reason = (
        f"All {len(planned)} term(s) significant at p <= alpha ({alpha:g}) "
        "in the full candidate model were selected (strong heredity preserved)."
    )
    return current_terms, steps, True, convergence_reason, pruned


def backward_elimination(
    anova_analysis: ANOVAAnalysis,
    candidate_pool: List[str],
    alpha: float = 0.10,
    max_iterations: int = 100,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[str], List[SelectionStep], List[str], List[str]]:
    """
    Backward elimination driven by the full-model significance criterion.

    Starts from the maximal fit-able hierarchical candidate model and drops,
    in sibling-first order (quadratics/interactions before mains), every term
    whose full-model ANOVA p-value exceeds alpha.  A parent is only dropped
    once all its dependents have been dropped (strong heredity), so significant
    higher-order terms protect their parents.

    Because the criterion is the same fit that backs the selection table,
    backward and forward converge on the same final model.

    Returns
    -------
    final_terms, steps, pruned_terms, warnings
    """
    factor_names = [f.name for f in anova_analysis.factors]

    start_terms, pruned = _fit_able_model(anova_analysis, candidate_pool)
    if pruned:
        warnings = [
            "The full candidate model could not be fitted (degrees of freedom "
            "exhausted). Starting backward elimination from a pruned "
            f"fit-able subset ({len(pruned)} term(s) dropped)."
        ]
    else:
        warnings = []

    full_results = _fit_candidate(anova_analysis, start_terms)
    if full_results is None:
        raise ValueError("The full hierarchical candidate model could not be fitted.")

    significant = _significant_terms(full_results, candidate_pool, alpha)
    target = _hierarchy_closure(
        sorted(significant, key=lambda t: candidate_pool.index(t)), factor_names
    )
    target_set = set(target)

    current_terms = list(start_terms)
    steps: List[SelectionStep] = []

    # Drop non-selected terms sibling-first (children before parents): reverse
    # pool order visits quadratics/interactions before their parent mains.
    for iteration, term in enumerate(reversed(candidate_pool), start=1):
        if progress_callback:
            progress_callback(iteration, max(1, len(candidate_pool)))
        if term == '1' or term in target_set or term not in current_terms:
            continue
        if term not in get_candidate_terms_backward(current_terms, factor_names, ['1']):
            continue
        p = term_p_value(full_results, term)
        current_terms = [t for t in current_terms if t != term]
        steps.append(SelectionStep(
            len(steps) + 1, 'remove', term, p, list(current_terms)
        ))

    return current_terms, steps, pruned, warnings


def stepwise_selection(
    anova_analysis: ANOVAAnalysis,
    candidate_pool: List[str],
    alpha: float = 0.10,
    max_iterations: int = 100,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Tuple[List[str], List[SelectionStep], bool, str, List[str]]:
    """
    Stepwise selection driven by the full-model significance criterion.

    Uses the same significance basis as forward elimination: a term is entered
    when its p-value in the maximal fit-able hierarchical candidate model is
    <= alpha, and the final model is the hierarchy closure of that set.  With the
    full-model criterion the entry/exit p-values are fixed, so stepwise enters
    the same significant set and performs no mid-run removals — it converges to
    the same hierarchical model as forward, which is the desired behavior for
    response-surface workflows.

    Returns
    -------
    final_terms, steps, started_with_intercept_only, convergence_reason, warnings
    """
    warnings: List[str] = []

    full_terms, pruned = _fit_able_model(anova_analysis, candidate_pool)
    if pruned:
        warnings.append(
            "The full candidate model could not be fitted (degrees of freedom "
            "exhausted). Working from a pruned fit-able subset "
            f"({len(pruned)} term(s) dropped)."
        )
    full_results = _fit_candidate(anova_analysis, full_terms)
    if full_results is None:
        raise ValueError("The full hierarchical candidate model could not be fitted.")

    factor_names = [f.name for f in anova_analysis.factors]
    significant = _significant_terms(full_results, candidate_pool, alpha)
    target = _hierarchy_closure(
        sorted(significant, key=lambda t: candidate_pool.index(t)), factor_names
    )
    target_set = set(target)

    steps: List[SelectionStep] = []
    current_terms = ['1']
    planned = [t for t in target if t != '1']
    total = max(1, len(planned))

    for iteration, term in enumerate(planned, start=1):
        if progress_callback:
            progress_callback(iteration, total)
        p = term_p_value(full_results, term)
        current_terms = list(current_terms) + [term]
        steps.append(SelectionStep(
            len(steps) + 1, 'add', term, p, list(current_terms)
        ))

    convergence_reason = (
        f"All {len(planned)} term(s) significant at p <= alpha ({alpha:g}) "
        "in the full candidate model were selected (strong heredity preserved)."
    )
    return current_terms, steps, True, convergence_reason, warnings


def _fit_able_model(
    anova_analysis: ANOVAAnalysis,
    candidate_pool: List[str],
    mandatory_terms: Optional[List[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Return the largest hierarchy-complete subset of *candidate_pool* that fits.

    The intercept ``'1'`` is always included and is never dropped.  Iteratively
    removes the last-added term (quadratics before interactions before mains,
    per generation order) until the model is estimable or only mandatory terms
    remain.  Returns ``(fit_terms, dropped_terms)``.
    """
    mandatory = list(set(mandatory_terms or ['1']) | {'1'})
    factor_names = [f.name for f in anova_analysis.factors]
    terms = list(dict.fromkeys(['1'] + list(candidate_pool)))
    terms, _ = enforce_hierarchy(terms, factor_names)
    while True:
        results = _fit_candidate(anova_analysis, terms)
        if results is not None:
            return terms, _dropped_from(candidate_pool, terms)
        candidates_droppable = [t for t in terms if t not in mandatory]
        if not candidates_droppable:
            raise ValueError("No fit-able model could be constructed from the candidate pool.")
        terms = [t for t in terms if t != candidates_droppable[-1]]


def _dropped_from(pool: List[str], fitted: List[str]) -> List[str]:
    """Terms in *pool* that are absent from the fit-able subset."""
    fit_set = set(fitted)
    return [t for t in pool if t not in fit_set]


def run_model_selection(
    anova_analysis: ANOVAAnalysis,
    method: str = 'forward',
    alpha: float = 0.10,
    bic_threshold: float = 2.0,
    max_iterations: int = 100,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> ModelSelectionResults:
    """
    Run automatic model selection and assemble the full result report.

    Parameters
    ----------
    anova_analysis : ANOVAAnalysis
        Analysis object (fits are re-run internally for candidate evaluation).
    method : str
        ``'forward'`` (default), ``'backward'``, ``'stepwise'``, or
        ``'stepwise_bic'``.
    alpha : float
        Entry/exit significance threshold in ``(0, 1)``.  Default 0.10.
    bic_threshold : float
        Minimum BIC improvement to continue, used only by ``'stepwise_bic'``.
        Default 2.0.
    max_iterations : int
        Iteration cap for the selection loops.
    progress_callback : Callable[[int, int], None], optional
        Progress notification callback.

    Raises
    ------
    ValueError
        If *method* is unknown or the analysis is a split-plot design.
    """
    if method not in _ALLOWED_METHODS:
        raise ValueError(
            f"Unknown selection method '{method}'. Expected one of {_ALLOWED_METHODS}."
        )
    if anova_analysis.design_structure.get('is_split_plot'):
        raise ValueError(
            "Automatic model selection is not supported for split-plot designs."
        )
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1.")
    if method == 'stepwise_bic' and not 0 < bic_threshold:
        raise ValueError("bic_threshold must be positive.")

    factors = anova_analysis.factors
    candidate_pool, _ = candidate_model_pool(factors)

    if method == 'forward':
        final_terms, steps, started_int, convergence, pruned = forward_selection(
            anova_analysis, candidate_pool, alpha=alpha,
            max_iterations=max_iterations, progress_callback=progress_callback,
        )
        run_warnings = [
            "The full candidate model could not be fitted (degrees of freedom "
            "exhausted). Working from a pruned fit-able subset "
            f"({len(pruned)} term(s) dropped)."
        ] if pruned else []
        oversaturated = bool(pruned)
    elif method == 'backward':
        final_terms, steps, pruned, run_warnings = backward_elimination(
            anova_analysis, candidate_pool, alpha=alpha,
            max_iterations=max_iterations, progress_callback=progress_callback,
        )
        convergence = (
            f"Removed all term(s) not significant at p <= alpha ({alpha:g}) "
            "in the full candidate model (strong heredity preserved)."
        ) if steps else (
            f"No candidate term exceeded p = {alpha:g} in the full candidate model."
        )
        started_int = True
        oversaturated = bool(pruned)
    elif method == 'stepwise_bic':
        from src.core.stepwise import stepwise_selection as bic_stepwise_selection

        starting_terms = ['1', factors[0].name] if factors else ['1']
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bic_results = bic_stepwise_selection(
                anova_analysis,
                all_possible_terms=['1'] + candidate_pool,
                starting_terms=starting_terms,
                mandatory_terms=['1'],
                max_iterations=max_iterations,
                bic_threshold=bic_threshold,
                progress_callback=progress_callback,
            )
        steps = [
            SelectionStep(
                s.step_number, s.action, s.term, None, list(s.current_terms),
                bic=s.bic, delta_bic=s.delta_bic,
            )
            for s in bic_results.steps
        ]
        final_terms = list(bic_results.final_terms)
        convergence = bic_results.convergence_reason
        started_int = False
        pruned = []
        oversaturated = False
        run_warnings = []
    else:
        final_terms, steps, started_int, convergence, run_warnings = stepwise_selection(
            anova_analysis, candidate_pool, alpha=alpha,
            max_iterations=max_iterations, progress_callback=progress_callback,
        )
        pruned = []
        oversaturated = False

    final_results = anova_analysis.fit(final_terms, enforce_hierarchy_flag=True)
    quality = compute_model_quality(
        final_results,
        design=anova_analysis.natural_data,
        factors=anova_analysis.factors,
    )

    factor_names = [f.name for f in factors]
    term_table = build_term_table(
        anova_analysis, candidate_pool, final_terms, final_results,
        factor_names, method, alpha, steps, pruned,
    )

    return ModelSelectionResults(
        method=method,
        alpha=alpha,
        final_terms=final_terms,
        selection_path=steps,
        term_table=term_table,
        final_results=final_results,
        model_quality=quality,
        n_iterations=len(steps),
        convergence_reason=convergence,
        started_with_intercept_only=started_int,
        oversaturated=oversaturated,
        warnings=run_warnings,
        bic_threshold=bic_threshold if method == 'stepwise_bic' else None,
    )


def build_term_table(
    anova_analysis: ANOVAAnalysis,
    candidate_pool: List[str],
    final_terms: List[str],
    final_results: ANOVAResults,
    factor_names: List[str],
    method: str,
    alpha: float,
    steps: List[SelectionStep],
    pruned: List[str],
) -> pd.DataFrame:
    """
    Build the per-candidate selection table.

    Coefficients and p-values come from a single full-pool fit when it is
    estimable; otherwise pruned candidates are reported as saturated and the
    final-model fit supplies values for selected terms.
    """
    fit_terms, dropped = _fit_able_model(anova_analysis, candidate_pool)
    full_results = _fit_candidate(anova_analysis, fit_terms)

    full_summary = getattr(full_results, 'anova_effect_summary', None) if full_results is not None else None
    full_estimates = getattr(full_results, 'effect_estimates', None) if full_results is not None else None
    final_summary = getattr(final_results, 'anova_effect_summary', None)

    removed_terms = {s.term for s in steps if s.action == 'remove'}
    added_terms = {s.term for s in steps if s.action == 'add'}
    pruned_set = set(pruned) | set(dropped)
    final_set = set(final_terms)

    rows = []
    for term in candidate_pool:
        if term == '1':
            continue

        p = None
        coefficient = None
        term_df = None
        if full_summary is not None and not full_summary.empty:
            key = find_effect_estimate_key(term, full_summary.index)
            if key is not None:
                row = full_summary.loc[key]
                p = _safe_p(row['p_value'])
                term_df = _safe_float(row.get('df'))
        if term_df is None:
            fkey = find_effect_estimate_key(term, final_summary.index) if final_summary is not None and not final_summary.empty else None
            if fkey is not None:
                term_df = _safe_float(final_summary.loc[fkey, 'df'])
        if term_df is not None and term_df == 1:
            for estimator in (full_estimates, final_results.effect_estimates):
                if estimator is not None and not estimator.empty:
                    k2 = find_effect_estimate_key(term, estimator.index)
                    if k2 is not None:
                        coefficient = _safe_float(estimator.loc[k2, 'Coefficient'])
                    if coefficient is not None:
                        break

        included = term in final_set
        if included:
            if p is None:
                p = term_p_value(final_results, term)

        reason = _reason_for(
            term, included, p, alpha, final_terms, factor_names,
            method, removed_terms, added_terms, pruned_set,
        )

        rows.append({
            'Term': term,
            'Coefficient': coefficient,
            'p-value': p,
            'Included': 'Yes' if included else 'No',
            'Reason': reason,
        })

    return pd.DataFrame(rows)


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _reason_for(
    term: str,
    included: bool,
    p: Optional[float],
    alpha: float,
    final_terms: List[str],
    factor_names: List[str],
    method: str,
    removed_terms: set,
    added_terms: set,
    pruned_set: set,
) -> str:
    """Assign the human-readable reason for a candidate's disposition."""
    if term in pruned_set:
        return 'Excluded (model saturated)'
    if method == 'stepwise_bic':
        if included:
            if term in added_terms:
                return 'Added during stepwise (BIC) selection'
            return 'Selected by BIC stepwise'
        if term in removed_terms:
            return 'Removed during stepwise (BIC) selection'
        return 'Not selected'
    if included:
        if p is not None and p <= alpha:
            return 'Significant'
        if required_by_hierarchy(final_terms, factor_names, term):
            return 'Kept for hierarchy'
        if method in ('forward', 'stepwise') and term in added_terms:
            return f'Added during {method} selection'
        return 'Kept for hierarchy'
    if method == 'backward' or term in removed_terms:
        return 'Removed by backward elimination'
    return 'Not significant'


def compute_model_quality(
    results: ANOVAResults,
    design: Optional[pd.DataFrame] = None,
    factors: Optional[List] = None,
) -> Dict[str, Optional[float]]:
    """
    Compute model-quality metrics for a fitted model.

    Parameters
    ----------
    results : ANOVAResults
        Fitted model results.
    design : pd.DataFrame, optional
        Natural-unit design (factor columns) used to group replicate runs for
        the lack-of-fit test.  When omitted, ``lack_of_fit_p_value`` is None.
    factors : List[Factor], optional
        Factor definitions.  Required when *design* is provided.

    Returns a dict with ``model_p_value``, ``lack_of_fit_p_value``,
    ``r_squared``, ``adj_r_squared``, ``predicted_r_squared``, ``press`` and
    ``predicted_r2_discrepancy`` (``abs(AdjR² - PredR²)``).
    """
    fm = getattr(results, 'fitted_model', None)

    model_p = None
    if fm is not None:
        if hasattr(fm, 'f_pvalue'):
            model_p = _safe_float(fm.f_pvalue)
        elif getattr(results, 'anova_table', None) is not None and not results.anova_table.empty:
            if 'P' in results.anova_table.columns and 'Model' in results.anova_table.index:
                model_p = _safe_float(results.anova_table.loc['Model', 'P'])

    r2 = _safe_float(getattr(results, 'r_squared', None))
    adj_r2 = _safe_float(getattr(results, 'adj_r_squared', None))

    lack_of_fit = None
    if design is not None and factors is not None:
        factor_names = [f.name for f in factors]
        available = [c for c in factor_names if c in design.columns]
        if available:
            try:
                lack_of_fit = _compute_lof_p_value(
                    design, results.residuals, factors, results.model_terms
                )
            except Exception:
                lack_of_fit = None

    press = compute_press(results)
    pred_r2 = compute_predicted_r_squared(results, press)

    discrepancy = None
    if adj_r2 is not None and pred_r2 is not None:
        discrepancy = abs(adj_r2 - pred_r2)

    return {
        'model_p_value': model_p,
        'lack_of_fit_p_value': lack_of_fit,
        'r_squared': r2,
        'adj_r_squared': adj_r2,
        'predicted_r_squared': pred_r2,
        'press': press,
        'predicted_r2_discrepancy': discrepancy,
    }


def compute_press(results: ANOVAResults) -> Optional[float]:
    """
    Prediction error sum of squares via leave-one-out on the hat matrix.

    ``PRESS = Σ (e_i / (1 - h_ii))²`` over all observations of the OLS fit.
    Returns None when the fit is not an OLS-like object or the leverage
    correction is degenerate.
    """
    fm = getattr(results, 'fitted_model', None)
    if fm is None or not hasattr(fm, 'get_influence'):
        return None
    try:
        hat = np.asarray(fm.get_influence().hat_matrix_diag, dtype=float)
    except Exception:
        return None
    resid = np.asarray(results.residuals, dtype=float)
    denom = 1.0 - hat
    if np.any(np.abs(denom) < 1e-12):
        return None
    return float(np.sum((resid / denom) ** 2))


def compute_predicted_r_squared(results: ANOVAResults, press: Optional[float] = None) -> Optional[float]:
    """Predicted R² = 1 - PRESS / TSS, or None when PRESS/TSS unavailable."""
    if press is None:
        press = compute_press(results)
    if press is None:
        return None
    y = np.asarray(results.fitted_values, dtype=float) + np.asarray(results.residuals, dtype=float)
    tss = float(np.sum((y - y.mean()) ** 2))
    if tss <= 0:
        return None
    return 1.0 - press / tss


def predicted_r2_warning(
    adj_r_squared: Optional[float],
    predicted_r_squared: Optional[float],
    threshold: float = PREDICTED_R2_GAP_THRESHOLD,
) -> Optional[str]:
    """
    Return the overfit warning when Predicted R² disagrees with Adjusted R².

    ``abs(AdjR² - PredR²) > threshold`` triggers a cautionary but
    non-blocking warning.
    """
    if adj_r_squared is None or predicted_r_squared is None:
        return None
    diff = abs(float(adj_r_squared) - float(predicted_r_squared))
    if diff > threshold:
        return (
            f"Warning: Predicted R² differs substantially from Adjusted R² "
            f"(difference = {diff:.3f} > {threshold:.2f}). The selected model "
            f"may be overfit and should be interpreted cautiously."
        )
    return None


def near_perfect_fit_message(
    r_squared: Optional[float],
    adj_r_squared: Optional[float],
    predicted_r_squared: Optional[float],
) -> Optional[str]:
    """Return an informational note when all R² measures are near-perfect."""
    if None in (r_squared, adj_r_squared, predicted_r_squared):
        return None
    if min(float(r_squared), float(adj_r_squared), float(predicted_r_squared)) >= NEAR_PERFECT_R2_THRESHOLD:
        return (
            "Model exhibits near-perfect fit. Verify the response is not "
            "deterministic, duplicated, or overparameterized relative to "
            "the design size."
        )
    return None


# -- display-name helpers (streamlit-free) -----------------------------------


def friendly_factor_name(name: str) -> str:
    """Human-friendly display for an internally escaped factor name.

    ``sanitize_factor_name`` prepends ``f_`` when the name is a Patsy
    reserved token (``C``, ``I``, ``np``, …) or starts with a digit.
    This helper reverses that escape for display-only purposes.

    >>> friendly_factor_name('f_C')
    'C'
    >>> friendly_factor_name('Temperature')
    'Temperature'
    """
    if name.startswith('f_') and len(name) > 2:
        rest = name[2:]
        if rest in ALL_RESERVED or (rest and rest[0].isdigit()):
            return rest
    return name


def factor_display_map(
    factors,
    rename_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Build ``{patsy_name → human display}`` for every factor.

    *rename_map* is ``{original_name: sanitized_name}`` as stored on
    ``ANOVAAnalysis.rename_map`` / project-load helpers.  When present the
    reverse mapping (sanitized → original) is preferred over the
    ``f_``-escape heuristic.
    """
    reverse = {v: k for k, v in (rename_map or {}).items()}
    return {f.name: reverse.get(f.name, friendly_factor_name(f.name)) for f in factors}


def map_term_display(term: str, display_map: Optional[Dict[str, str]] = None) -> str:
    """Token-replace factor names inside a patsy term string.

    A factor name is replaced only at identifier boundaries, so ``f_C`` does
    not match inside ``f_C_B`` and ``A`` does not match inside ``AB``.  The
    pattern is still token-insensitive to order/length thanks to the
    boundary lookarounds.

    >>> map_term_display('I(f_C**2)', {'f_C': 'C'})
    'I(C**2)'
    >>> map_term_display('f_C*f_C_B', {'f_C': 'C'})
    'C*f_C_B'
    """
    if not display_map:
        return term
    pattern = '|'.join(re.escape(name) for name in display_map)
    return re.sub(
        rf'(?<!\w)({pattern})(?!\w)',
        lambda m: display_map[m.group(1)],
        term,
    )


def _format_p(p: Optional[float]) -> str:
    if p is None:
        return 'N/A'
    tiny = np.finfo(float).tiny
    v = min(max(float(p), tiny), 1.0 - 1e-16)
    if v < 0.001:
        return f"{v:.1e}".replace('e-0', 'e-')
    return f"{v:.4f}"


def classify_model_type(final_terms: List[str], candidate_pool: List[str]) -> str:
    """Classify a selected model as Linear / Interaction / Reduced / Full.

    ``candidate_pool`` is the full (unpruned) design-aware candidate pool.  A
    model that equals a pool with quadratic terms is *Full Quadratic*; a model
    that equals a pool without quadratics (designs with no continuous factors)
    is *Full Interaction*.  Any quadratic subset is *Reduced Quadratic*;
    interactions without quadratics are *Interaction*.
    """
    final = [t for t in final_terms if t != '1']
    pool = [t for t in candidate_pool if t != '1']
    if not final:
        return 'Linear (intercept only)'
    has_quad = any('I(' in t or '**' in t for t in final)
    has_inter = any('*' in t for t in final)
    if set(final) == set(pool):
        return 'Full Quadratic' if any('I(' in t or '**' in t for t in pool) else 'Full Interaction'
    if has_quad:
        return 'Reduced Quadratic'
    if has_inter:
        return 'Interaction'
    return 'Linear'


def _hierarchy_satisfied(terms: List[str]) -> bool:
    """True when every non-intercept term's parents are also present."""
    included = {t for t in terms if t != '1'}
    for term in included:
        parents, _ = parse_model_term(term)
        for parent in parents:
            if parent not in included:
                return False
    return True


def _display_term(term: str, display_map: Optional[Dict[str, str]] = None) -> str:
    """Tiny streamlit-free pretty-printer for term strings."""
    term = map_term_display(term, display_map)
    superscripts = {'0': '⁰', '1': '¹', '2': '²'}
    if term == '1':
        return 'β₀'
    if term.startswith('I(') and term.endswith('**2)'):
        base = term[2:-4]
        return f"{base}{superscripts.get('2', '2')}"
    if '*' in term:
        return '×'.join(term.split('*'))
    return term


def _subscripted(value) -> str:
    """Render a non-negative integer with unicode subscript digits."""
    subs = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
            '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}
    return ''.join(subs.get(ch, ch) for ch in str(value))


def _symbolic_equation(
    terms: List[str],
    response_name: str = "Y",
    display_map: Optional[Dict[str, str]] = None,
) -> str:
    non_intercept = [t for t in terms if t != '1']
    if not non_intercept:
        return f"{response_name} = β₀"
    parts = ["β₀"] + [f"β{_subscripted(i)}·{_display_term(t, display_map)}" for i, t in enumerate(non_intercept, start=1)]
    return f"{response_name} = " + " + ".join(parts)


def format_selection_summary(
    results: ModelSelectionResults,
    response_name: str = "Y",
    factor_display_map: Optional[Dict[str, str]] = None,
    candidate_pool: Optional[List[str]] = None,
) -> str:
    """
    Format the selection results as a markdown summary.

    Includes the method / criterion / alpha header, a model-metadata block
    (selected terms, model type, hierarchy status), the selection path, and
    the final symbolic model equation.  When *factor_display_map* is provided,
    factor names are rendered human-friendly.  Streamlit-free for testability.
    """
    final_terms = [t for t in results.final_terms if t != '1']
    lines = []
    is_bic = results.method == 'stepwise_bic'
    lines.append("### 🔍 Automatic Model Selection Results")
    lines.append("")
    lines.append(f"**Selection Method:** {_method_label(results.method)}")
    if is_bic:
        lines.append("**Criterion:** BIC stepwise (add/remove by BIC improvement)")
        lines.append(f"**BIC Threshold:** {results.bic_threshold:g}")
    else:
        lines.append("**Criterion:** p-values from the full candidate model")
        lines.append(f"**Alpha:** {results.alpha:g}")
    lines.append(f"**Selected Terms:** {len(final_terms)}")
    if candidate_pool is not None:
        lines.append(f"**Model Type:** {classify_model_type(results.final_terms, candidate_pool)}")
    if _hierarchy_satisfied(results.final_terms):
        lines.append("**Hierarchy Status:** ✓ Strong heredity satisfied")
    else:
        lines.append("**Hierarchy Status:** ⚠️ Not satisfied — parents are missing")
    lines.append("")

    if results.selection_path:
        lines.append("**Selection Path:**")
        lines.append("")
        for step in results.selection_path:
            action = "Added" if step.action == 'add' else "Removed"
            detail = (
                f"(BIC = {step.bic:.2f}, ΔBIC = {step.delta_bic:.2f})"
                if is_bic and step.bic is not None
                else f"(p = {_format_p(step.p_value)})"
            )
            lines.append(
                f"{step.step_number}. {action} {_display_term(step.term, factor_display_map)} {detail}"
            )
        lines.append("")

    lines.append("**Final Model:**")
    lines.append("")
    lines.append(_symbolic_equation(results.final_terms, response_name, factor_display_map))
    lines.append("")
    lines.append(f"*Convergence: {results.convergence_reason}*")
    return "\n".join(lines)