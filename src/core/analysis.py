"""
ANOVA Analysis Module for Design of Experiments.

Model Term Notation:
- Main effects: 'A', 'B', 'Temperature'
- Interactions: 'A*B', 'Temperature*Pressure'
- Quadratic: 'I(A**2)', 'I(Temperature**2)'
  (uses patsy I() identity operator for Python exponentiation)

Shared primitives (ANOVAResults, parse_model_term, enforce_hierarchy,
quadratic) live in ``analysis_base`` to avoid circular imports with
``split_plot_analysis``.
"""

import warnings
import re
from typing import List, Dict, Optional, Union, Literal, Tuple
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
import patsy

from src.core.factors import Factor, FactorType, ChangeabilityLevel
from src.core.coding import DesignSpace
from src.core.analysis_base import (
    ANOVAResults,
    _normalize_term,
    build_anova_effect_summary,
    build_coefficient_significance,
    enforce_hierarchy,
    parse_model_term,
    quadratic,
)
from src.core.split_plot_analysis import fit_split_plot_anova


def generate_model_terms(
    factors: List[Factor],
    model_type: Literal['linear', 'interaction', 'quadratic'],
    include_intercept: bool = True
) -> List[str]:
    """
    Generate standard model terms using Python/patsy notation.
    
    Raises
    ------
    ValueError
        If model_type='quadratic' but no continuous factors present
    
    Examples
    --------
    >>> generate_model_terms(factors, 'quadratic')
    ['1', 'A', 'B', 'A*B', 'I(A**2)', 'I(B**2)']
    """
    # Validate quadratic requirement
    if model_type == 'quadratic':
        if not any(f.is_continuous() for f in factors):
            raise ValueError(
                "Quadratic model requires at least one continuous factor. "
                "Use 'interaction' model for categorical factors only."
            )
        
    terms = []
    
    if include_intercept:
        terms.append('1')
    
    # Main effects
    factor_names = [f.name for f in factors]
    terms.extend(factor_names)
    
    # Two-way interactions
    if model_type in ('interaction', 'quadratic'):
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                terms.append(f"{factor_names[i]}*{factor_names[j]}")
    
    # Quadratic terms - use patsy I() notation directly
    if model_type == 'quadratic':
        for factor in factors:
            if factor.is_continuous():
                terms.append(f"I({factor.name}**2)")
    
    return terms


# parse_model_term is imported from analysis_base


# enforce_hierarchy is imported from analysis_base


# quadratic is imported from analysis_base


def detect_split_plot_structure(design: pd.DataFrame, factors: List[Factor]) -> Dict:
    """Detect split-plot structure from factor changeability."""
    very_hard = [f.name for f in factors if f.changeability == ChangeabilityLevel.VERY_HARD]
    hard = [f.name for f in factors if f.changeability == ChangeabilityLevel.HARD]
    easy = [f.name for f in factors if f.changeability == ChangeabilityLevel.EASY]
    
    return {
        'is_split_plot': len(hard) > 0 or len(very_hard) > 0,
        'whole_plot_factors': very_hard + hard,
        'sub_plot_factors': easy,
        'has_blocking': 'Block' in design.columns,
        'whole_plot_column': 'WholePlot' if 'WholePlot' in design.columns else None
    }


def prepare_analysis_data(
    design: pd.DataFrame,
    response: Union[np.ndarray, pd.Series],
    factors: List[Factor],
    response_name: str = "Response"
) -> pd.DataFrame:
    """Prepare data for analysis."""
    if len(response) != len(design):
        raise ValueError(f"Response length mismatch: {len(response)} != {len(design)}")

    factor_names = [f.name for f in factors]
    analysis_df = design[factor_names].copy()

    analysis_df[response_name] = response
    
    for col in ['Block', 'WholePlot', 'VeryHardPlot', 'Replicate', 'RunOrder', 'StdOrder']:
        if col in design.columns:
            analysis_df[col] = design[col]
    
    return analysis_df



def validate_model_terms(terms: List[str], factors: List[Factor], design: pd.DataFrame) -> None:
    """Validate model terms compatibility."""
    factor_dict = {f.name: f for f in factors}

    # Operators that involve a transform — the primary factor must be continuous
    _transform_operators = {
        'transform', 'transform_power', 'transform_cross', 'transform_power_cross'
    }

    for term in terms:
        if term == '1':
            continue

        factor_list, operator = parse_model_term(term)

        for fname in factor_list:
            if fname not in factor_dict:
                raise ValueError(f"Factor '{fname}' in '{term}' not found")

        if operator == '**':
            # Plain quadratic: I(A**2)
            factor = factor_dict[factor_list[0]]
            if not factor.is_continuous():
                raise ValueError(f"Quadratic '{term}' requires continuous factor")
            unique_vals = design[factor.name].nunique()
            if unique_vals <= 2:
                warnings.warn(f"Quadratic '{term}': only {unique_vals} levels")

        elif operator in _transform_operators:
            # Transform terms: the first factor in factor_list is always the
            # raw factor being transformed — it must be continuous.
            primary_factor = factor_dict[factor_list[0]]
            if not primary_factor.is_continuous():
                raise ValueError(
                    f"Transform term '{term}' requires a continuous factor; "
                    f"'{factor_list[0]}' is not continuous."
                )


# ANOVAResults is imported from analysis_base


class ANOVAAnalysis:
    """ANOVA analysis for experimental designs."""

    def __init__(
        self,
        design: pd.DataFrame,
        response: Union[np.ndarray, pd.Series],
        factors: List[Factor],
        response_name: str = "Response",
        is_split_plot: Optional[bool] = None,
        block_as_random: bool = False,
        include_block_in_prediction: bool = False,
    ):
        self.factors = factors
        self.response = np.array(response)
        self.response_name = response_name
        self.block_as_random = block_as_random
        self.include_block_in_prediction = include_block_in_prediction

        # Build the design space and encode to coded [-1, +1] space for the
        # primary analysis data.  The incoming ``design`` may be in natural
        # units (the canonical session-state format since Phase 2) or already
        # coded (legacy callers).  Using DesignSpace guarantees the coded
        # matrix is always available for polynomial models.
        #
        # Transform terms (np.log, np.sqrt, I(1/x), np.exp) must be evaluated
        # in natural units — applying log to a coded value of -1 produces NaN.
        # self.natural_data retains the original factor values so that Patsy
        # can correctly evaluate transform terms when they appear in a model.
        self._design_space = DesignSpace.from_factors(factors)
        self.design = self._design_space.encode_dataframe(design)

        self.data = prepare_analysis_data(
            self.design, response, factors, response_name
        )
        # Natural-unit data frame: decoded back from coded so it is always
        # consistent with self.design regardless of what the caller passed in.
        self.natural_data = prepare_analysis_data(
            self._design_space.decode_dataframe(self.design), response, factors, response_name
        )
        self.rename_map = {}
        self.design_structure = detect_split_plot_structure(self.design, factors)

        if is_split_plot is not None:
            self.design_structure['is_split_plot'] = is_split_plot

        self.current_model = None
        self.current_results = None
        self.excluded_term_reasons: List[Tuple[str, str]] = []
    
    def fit(self, model_terms: List[str], enforce_hierarchy_flag: bool = True) -> ANOVAResults:
        """Fit ANOVA model. Use I(A**2) notation for quadratic terms."""
        validate_model_terms(model_terms, self.factors, self.design)
        
        if enforce_hierarchy_flag:
            factor_names = [f.name for f in self.factors]
            complete_terms, added = enforce_hierarchy(model_terms, factor_names)
            
            if added:
                warnings.warn(f"Added for hierarchy: {added}")
            
            model_terms = complete_terms
        
        self.current_model = model_terms
        pruned_terms, excluded = self._prune_degenerate_terms(model_terms)
        self.excluded_term_reasons = list(excluded)
        if excluded:
            for term, reason in excluded:
                warnings.warn(f"Excluded term '{term}': {reason}")
        model_terms = pruned_terms

        # Defensive hierarchy cascade: pruning may have removed a parent main
        # effect (e.g. a factor constant across runs).  Any surviving
        # interaction or quadratic built on that parent is itself
        # non-estimable, so drop it too rather than leave an orphan that
        # breaks hierarchical completeness.  Iterate because dropping an
        # interaction can orphan a higher-order term.
        kept_tail = [t for t in model_terms if t != '1']
        orphaned = []
        while True:
            removed_any = False
            for term in list(kept_tail):
                factor_list, operator = parse_model_term(term)
                if operator not in ('*', '**'):
                    continue
                siblings = set(kept_tail) - {term}
                missing = [p for p in factor_list if p not in siblings]
                if missing:
                    kept_tail.remove(term)
                    orphaned.append(
                        (term, f"hierarchical parent '{missing[0]}' pruned as non-estimable")
                    )
                    removed_any = True
            if not removed_any:
                break
        if orphaned:
            excluded.extend(orphaned)
            self.excluded_term_reasons = list(excluded)
            for term, reason in orphaned:
                warnings.warn(f"Excluded term '{term}': {reason}")
            model_terms = ['1'] + kept_tail
        self.current_model = model_terms
        self._validate_degrees_of_freedom(model_terms)
        if self.design_structure['is_split_plot']:
            self._validate_split_plot_degrees_of_freedom()
            results = self._fit_mixed_effects_model(model_terms)
        else:
            results = self._fit_fixed_effects_model(model_terms)
        
        self.current_results = results
        return results
    
    @staticmethod
    def _has_transform_terms(model_terms: List[str]) -> bool:
        """
        Return True if any term in *model_terms* is a transform term.

        Transform terms require evaluation in natural (engineering) units.
        Patsy cannot evaluate ``np.log(A)`` meaningfully when ``A`` is coded
        to ``[-1, +1]`` because negative coded values produce NaN/complex
        results for log and sqrt.

        Parameters
        ----------
        model_terms : List[str]
            Model terms in patsy notation.

        Returns
        -------
        bool
        """
        from src.core.analysis_base import _TRANSFORM_PREFIXES
        return any(
            any(term.startswith(pfx) for pfx in _TRANSFORM_PREFIXES)
            or (term.startswith('I(') and any(
                pfx in term for pfx in _TRANSFORM_PREFIXES
            ))
            for term in model_terms
        )

    def _select_fit_data(self, model_terms: List[str]) -> pd.DataFrame:
        """
        Choose the data frame to pass to Patsy based on whether transform
        terms are present.

        - Polynomial-only models use ``self.data`` (coded space): coefficients
          are comparable across factors and numerically stable.
        - Models containing any transform term use ``self.natural_data``
          (natural units): transforms are applied to meaningful physical
          values, not to arbitrary [-1, +1] coded values.

        Parameters
        ----------
        model_terms : List[str]
            Model terms in patsy notation.

        Returns
        -------
        pd.DataFrame
            The appropriate data frame for OLS fitting.
        """
        if self._has_transform_terms(model_terms):
            return self.natural_data
        return self.data

    def _prune_degenerate_terms(
        self, model_terms: List[str]
    ) -> Tuple[List[str], List[Tuple[str, str]]]:
        """
        Remove model terms whose design columns carry no estimable information.

        The same design matrix patsy will build for the fit is inspected via
        ``patsy.dmatrix`` (identical data frame and term notation), so the
        check reflects exactly the columns the regression would use.  Terms are
        dropped, with a recorded reason, when they are:

        - **constant** — a single value across all runs.  A quadratic on a
          two-level factor produces a column of all ones (identical to the
          intercept); a categorical factor observed at one level produces an
          all-zero dummy column.  Both are unidentifiable.
        - **perfectly aliased** — every column of one term equals every column
          of another kept term (exact linear dependence).
        - **other rank defect** — the remaining columns are linearly dependent
          even though no single term is constant or an exact duplicate.  The
          last-added non-intercept term that resolves the deficit is dropped.

        Removals are iterative: after each drop the matrix is rebuilt, so
        multiple degenerate terms are removed in successive passes.  Never
        drops ``'1'``.  On any unexpected patsy/build failure the pruning
        stops conservatively and returns the terms kept so far.

        Returns ``(kept_terms, [(dropped_term, reason), ...])``.
        """
        terms = [t for t in model_terms if t != '1']
        if not terms:
            return ['1'], []
        fit_data = self._select_fit_data(terms)
        kept = list(terms)
        excluded: List[Tuple[str, str]] = []

        def build_matrix(term_list: List[str]):
            formula_rhs = ' + '.join(self._wrap_categorical_terms(term_list))
            return patsy.dmatrix(formula_rhs, fit_data, return_type='dataframe')

        def near_constant(column) -> bool:
            return bool(np.allclose(column.iloc[0], column))

        def prune_key(name: str) -> str:
            # patsy reports interaction terms in ``term_names`` with ':'
            # separators (``feed_rate:catalyst``) while the wrapped formula
            # side uses '*' (``feed_rate*catalyst``).  Canonicalise both sides
            # so term->slice lookups match for interaction-bearing models.
            return _normalize_term(name).replace('*', ':')

        while kept:
            try:
                design = build_matrix(kept)
            except Exception:
                break
            info = design.design_info
            term_names = list(info.term_names)
            # NOTE: ``len(term_names)`` intentionally is NOT required to equal
            # ``len(kept) + 1``: patsy's ``*`` syntax expands an interaction
            # (``A*B``) into its parent main effects, so a dangling
            # interaction after a parent was pruned yields extra auto-added
            # term names.  Per-term slice lookup below is the real guard; a
            # dropped/renamed term shows up as a missing key and stops
            # conservatively.
            if not np.all(np.isfinite(np.asarray(design))):
                # Non-finite columns (e.g. np.log(0) or I(1/0) outside the
                # transform domain) are a data problem, not a degeneracy: let
                # the regular fit raise its natural error/warning rather than
                # silently dropping the requested term.
                break

            # Map each kept term to its patsy slice by matching the wrapped
            # formula name -- patsy may reorder terms (categoricals first), so
            # a positional zip would misalign the slices.
            wrapped = self._wrap_categorical_terms(kept)
            raw_to_str = dict(zip(kept, wrapped))
            str_to_slice = {
                prune_key(name): info.term_slices[t]
                for t, name in zip(info.terms[1:], info.term_names[1:])
            }
            try:
                slices = [
                    str_to_slice[prune_key(raw_to_str[term])]
                    for term in kept
                ]
            except KeyError:
                break

            # 1) Constant / intercept-aliased columns, or terms that carry no
            #    design columns at all (e.g. a categorical factor observed at
            #    only its reference level).
            drop_term = None
            drop_reason = None
            for term, sl in zip(kept, slices):
                cols = list(design.columns[sl])
                if not cols:
                    drop_term = term
                    drop_reason = "no estimable columns — factor observed at a single level"
                    break
                if all(near_constant(design[col]) for col in cols):
                    drop_term = term
                    if all(np.all(np.asarray(design[col]) == 1.0) for col in cols):
                        drop_reason = (
                            f"constant column — identical to the intercept "
                            f"(factor observed at a single value across runs)"
                        )
                    else:
                        drop_reason = (
                            "constant column — no variation across runs "
                            "(aliased with the intercept)"
                        )
                    break
            if drop_term is not None:
                if len(kept) == 1:
                    # Sole remaining term is unusable; keep it rather than
                    # reducing the model to the intercept.
                    break
                kept.remove(drop_term)
                excluded.append((drop_term, drop_reason))
                continue

            # 2) Exact duplication between terms.
            duplicate = None
            duplicate_reason = None
            for i, (term_a, sl_a) in enumerate(zip(kept, slices)):
                if duplicate is not None:
                    break
                for term_b, sl_b in list(zip(kept, slices))[i + 1:]:
                    cols_a = list(design.columns[sl_a])
                    cols_b = list(design.columns[sl_b])
                    if len(cols_a) == len(cols_b) and all(
                        np.array_equal(design[a], design[b])
                        for a, b in zip(cols_a, cols_b)
                    ):
                        duplicate = term_b
                        duplicate_reason = f"perfectly aliased with '{term_a}'"
                        break
            if duplicate is not None:
                kept.remove(duplicate)
                excluded.append((duplicate, duplicate_reason))
                continue

            # 3) General rank deficiency.
            col_positions = sorted(
                int(i)
                for sl in slices
                for i in range(design.shape[1])[sl]
            )
            matrix = np.asarray(design.iloc[:, col_positions])
            if np.linalg.matrix_rank(matrix) == matrix.shape[1]:
                break
            resolved = None
            for candidate in reversed(kept):
                trial = [t for t in kept if t != candidate]
                try:
                    trial_design = build_matrix(trial)
                except Exception:
                    continue
                data = trial_design.design_info
                # No ``len(term_names) == len(trial) + 1`` requirement here
                # either — ``*`` expands parents — per-term lookup governs.
                trial_wrapped = self._wrap_categorical_terms(trial)
                trial_raw_to_str = dict(zip(trial, trial_wrapped))
                trial_str_to_slice = {
                    prune_key(name): data.term_slices[t]
                    for t, name in zip(data.terms[1:], data.term_names[1:])
                }
                try:
                    trial_slices = [
                        trial_str_to_slice[prune_key(trial_raw_to_str[term])]
                        for term in trial
                    ]
                except KeyError:
                    continue
                trial_cols = sorted(
                    int(i)
                    for sl in trial_slices
                    for i in range(trial_design.shape[1])[sl]
                )
                trial_matrix = np.asarray(trial_design.iloc[:, trial_cols])
                if np.linalg.matrix_rank(trial_matrix) == trial_matrix.shape[1]:
                    resolved = candidate
                    break
            target = resolved if resolved is not None else kept[-1]
            kept.remove(target)
            excluded.append(
                (target, "linearly dependent on the remaining model columns")
            )
        return ['1'] + kept, excluded

    def _fit_fixed_effects_model(self, model_terms: List[str]) -> ANOVAResults:
        """Fit fixed effects ANOVA."""
        formula = self._build_formula(model_terms)
        fit_data = self._select_fit_data(model_terms)

        if self.design_structure['has_blocking'] and not self.block_as_random:
            # Cast Block to str so patsy treats it as categorical without
            # requiring C() notation (which conflicts with factors named "C").
            fit_data = fit_data.copy()
            fit_data['Block'] = fit_data['Block'].astype(str)
            formula += " + Block"
            fitted_model = ols(formula, data=fit_data).fit()
        elif self.design_structure['has_blocking'] and self.block_as_random:
            model = mixedlm(formula, data=fit_data, groups=fit_data['Block'], re_formula='1')
            fitted_model = model.fit(method='lbfgs')
        else:
            fitted_model = ols(formula, data=fit_data).fit()

        return self._build_results_object(fitted_model, model_terms, False)
    
    def _fit_mixed_effects_model(self, model_terms: List[str]) -> ANOVAResults:
        """
        Fit a two-strata split-plot ANOVA model.

        Delegates to split_plot_analysis.fit_split_plot_anova which uses the
        Yates / expected-mean-squares approach:
        - Whole-plot terms tested against whole-plot error (MS_WP)
        - Subplot terms tested against subplot error (MS_SP)

        This avoids the LinAlgError that occurs when whole-plot factors are
        passed to mixedlm alongside a random WholePlot intercept, which makes
        the design matrix singular because hard factors do not vary within any
        whole-plot.
        """
        if self.design_structure['whole_plot_column'] is None:
            raise ValueError(
                "Split-plot analysis requires a 'WholePlot' column in the design. "
                "Ensure the design was generated with hard-to-change factors so that "
                "whole-plot groupings are recorded."
            )

        return fit_split_plot_anova(
            data=self._select_fit_data(model_terms),
            factors=self.factors,
            model_terms=model_terms,
            response_name=self.response_name,
            whole_plot_col=self.design_structure['whole_plot_column'],
            design_structure=self.design_structure,
        )
    
    def _build_formula(self, model_terms: List[str]) -> str:
        """Build formula - terms already in patsy notation."""
        terms = [t for t in self._wrap_categorical_terms(model_terms) if t != '1']
        if not terms:
            # Intercept-only model
            formula_rhs = '1'
        else:
            formula_rhs = ' + '.join(terms)
        return f"{self.response_name} ~ {formula_rhs}"

    def _wrap_categorical_terms(self, model_terms: List[str]) -> List[str]:
        """
        Wrap categorical factor names in patsy ``C(...)`` so they are always
        treated as categorical, regardless of the data column dtype.

        Without this, a categorical factor whose levels are numeric-looking
        labels (e.g. lot/batch IDs) can be silently fitted as a continuous
        predictor: patsy decides categorical-vs-continuous purely from the
        column dtype, and ``float64`` columns get a single linear coefficient
        (DF=1) instead of k-1 dummy columns.
        """
        categorical = {f.name for f in self.factors if f.is_categorical()}
        if not categorical:
            return list(model_terms)

        wrapped_terms = []
        for term in model_terms:
            if term == '1':
                wrapped_terms.append(term)
                continue
            new_term = term
            for name in sorted(categorical, key=len, reverse=True):
                if re.search(r'C\(' + re.escape(name) + r'\)', new_term):
                    continue
                pattern = r'(?<![\w)])' + re.escape(name) + r'(?![\w])'
                new_term = re.sub(pattern, f'C({name})', new_term)
            wrapped_terms.append(new_term)
        return wrapped_terms
    
    @staticmethod
    def _strip_c_wrappers(label: str) -> str:
        """
        Remove patsy ``C(...)`` wrappers from a term label so results are
        reported using plain factor names, e.g.:
          ``C(Egg_lot)`` -> ``Egg_lot``
          ``C(Egg_lot)[T.41007666]`` -> ``Egg_lot[T.41007666]``
          ``C(Egg_lot):Egg_percent`` -> ``Egg_lot:Egg_percent``
        """
        return re.sub(r'C\(([^()]*)\)', r'\1', label)

    def _observation_usage(self, fitted_model, fit_data: pd.DataFrame) -> Dict[str, object]:
        """
        Reconstruct which observations the fitted model actually used.

        ``ols``/``mixedlm`` auto-drop rows with missing values in any formula
        variable, so ``len(fitted_model.resid)`` can be smaller than the
        number of rows in the analysis frame.  The fitted model's row labels
        are the authoritative record of what was used; the excluded rows are
        everything else in the frame.

        Returns
        -------
        Dict[str, object]
            ``n_obs_total``, ``n_obs_used``, ``n_obs_excluded``,
            ``excluded_obs_labels``, ``used_row_indices`` (positions into
            ``fit_data``), and ``used_response`` (observations used, aligned
            with ``residuals``/``fitted_values``).
        """
        n_obs_total = int(len(fit_data))
        used_labels: Optional[List[object]] = None
        try:
            row_labels = getattr(
                getattr(fitted_model, 'model', None), 'data', None
            )
            if row_labels is not None and row_labels.row_labels is not None:
                used_labels = list(row_labels.row_labels)
        except Exception:
            used_labels = None

        if used_labels is None:
            # Fallback (e.g. exotic result objects): derive usage from the
            # response's own missing-value mask.
            response_arr = fit_data[self.response_name].to_numpy()
            used_indices = np.flatnonzero(
                pd.notna(response_arr)
            ).astype(np.int64)
        else:
            positions = fit_data.index.get_indexer(used_labels)
            used_indices = np.asarray(
                [p for p in positions if p >= 0], dtype=np.int64
            )

        used_indices = np.unique(used_indices)
        n_obs_used = int(len(used_indices))
        n_obs_excluded = max(n_obs_total - n_obs_used, 0)

        excluded_indices = [
            i for i in range(n_obs_total) if i not in set(used_indices.tolist())
        ]
        excluded_obs_labels = self._format_excluded_obs_labels(
            excluded_indices
        )

        used_response = None
        if n_obs_used > 0:
            used_response = pd.to_numeric(
                fit_data[self.response_name], errors='coerce'
            ).to_numpy(dtype=float)[used_indices]

        return {
            'n_obs_total': n_obs_total,
            'n_obs_used': n_obs_used,
            'n_obs_excluded': n_obs_excluded,
            'excluded_obs_labels': excluded_obs_labels,
            'used_row_indices': used_indices,
            'used_response': used_response,
        }

    def _format_excluded_obs_labels(self, excluded_indices: List[int]) -> List[str]:
        """
        Format stable, human-readable identifiers for excluded observations.

        Prefers run numbers carried in the analysis frame (``RunOrder`` then
        ``StdOrder``), falling back to 1-based row positions.
        """
        labels = []
        for pos in excluded_indices:
            if 0 <= pos < len(self.data):
                row = self.data.iloc[pos]
                if 'RunOrder' in self.data.columns and pd.notna(row['RunOrder']):
                    labels.append(f"Run {int(row['RunOrder'])}")
                elif 'StdOrder' in self.data.columns and pd.notna(row['StdOrder']):
                    labels.append(f"StdOrder {int(row['StdOrder'])}")
                else:
                    labels.append(f"Row {pos + 1}")
            else:
                labels.append(f"Row {pos + 1}")
        return labels

    def _build_results_object(self, fitted_model, model_terms: List[str], is_split_plot: bool) -> ANOVAResults:
        """Build results from fitted model."""
        fit_data = self._select_fit_data(model_terms)
        usage = self._observation_usage(fitted_model, fit_data)
        try:
            if hasattr(fitted_model, 'anova_table'):
                anova_table = fitted_model.anova_table()
            else:
                anova_table = sm.stats.anova_lm(fitted_model, typ=2)
        except (ValueError, np.linalg.LinAlgError) as e:
            warnings.warn(f"Could not compute ANOVA: {e}")
            anova_table = pd.DataFrame()
        
        if not anova_table.empty:
            anova_table = anova_table.copy()
            anova_table.index = [
                self._strip_c_wrappers(str(i)) for i in anova_table.index
            ]
        
        effect_estimates = pd.DataFrame({
            'Coefficient': fitted_model.params,
            'Std_Error': fitted_model.bse,
            't_value': fitted_model.tvalues,
            'p_value': fitted_model.pvalues
        })
        effect_estimates.index = [
            self._strip_c_wrappers(str(i)) for i in effect_estimates.index
        ]
        residuals = fitted_model.resid
        fitted_values = fitted_model.fittedvalues
        
        if hasattr(fitted_model, 'rsquared'):
            r_squared = fitted_model.rsquared
            adj_r_squared = fitted_model.rsquared_adj
        else:
            ss_total = np.sum((self.data[self.response_name] - self.data[self.response_name].mean())**2)
            ss_resid = np.sum(residuals**2)
            r_squared = 1 - ss_resid / ss_total
            n, p = len(self.data), len(fitted_model.params)
            adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
        
        rmse = np.sqrt(np.mean(residuals**2))
        diagnostics = self._compute_diagnostics(residuals, fitted_values)
        
        # LogWorth
        logworth_df = effect_estimates[effect_estimates.index != 'Intercept'].copy()
        logworth_values = []
        for p in logworth_df['p_value']:
            if pd.isna(p) or p <= 0:
                logworth_values.append(np.nan)
            elif p < 1e-16:
                logworth_values.append(16.0)
            else:
                logworth_values.append(-np.log10(p))
        logworth_df['LogWorth'] = logworth_values

        # Canonical effect tables.  ``coefficient_significance`` keeps the
        # fitted-model coefficient tests; ``anova_effect_summary`` is sourced
        # from the displayed ANOVA table.  The two are deliberately distinct.
        block_names = ('Block',) if (
            self.design_structure.get('has_blocking')
            and not self.block_as_random
        ) else ()
        coefficient_significance = build_coefficient_significance(
            effect_estimates, anova_table, block_factor_names=block_names
        )
        anova_effect_summary = build_anova_effect_summary(
            anova_table, effect_estimates, block_factor_names=block_names
        )

        block_mean_shift = self._block_mean_shift_if_excluded(fitted_model)
        if block_mean_shift != 0.0:
            # Average the fitted equation over blocks: add the size-weighted
            # mean block offset to the intercept so predictions (from either
            # effect_estimates or the statsmodels model) are no longer tied to
            # a specific block's reference level.  The reference block
            # contributes 0 offset and each non-reference block its dummy
            # coefficient, so the reference-level prediction becomes the
            # block-average prediction.
            if 'Intercept' in effect_estimates.index:
                effect_estimates.loc['Intercept', 'Coefficient'] = (
                    effect_estimates.loc['Intercept', 'Coefficient'] + block_mean_shift
                )
            exog_names = list(getattr(fitted_model.model, 'exog_names', ()) or ())
            params_arr = getattr(getattr(fitted_model, '_results', None), 'params', None)
            if isinstance(params_arr, np.ndarray) and 'Intercept' in exog_names:
                params_arr[exog_names.index('Intercept')] += block_mean_shift

        return ANOVAResults(
            anova_table=anova_table,
            effect_estimates=effect_estimates,
            logworth=logworth_df,
            residuals=residuals,
            fitted_values=fitted_values,
            fitted_model=fitted_model,
            diagnostics=diagnostics,
            model_terms=model_terms,
            is_split_plot=is_split_plot,
            r_squared=r_squared,
            adj_r_squared=adj_r_squared,
            rmse=rmse,
            coefficient_significance=coefficient_significance,
            anova_effect_summary=anova_effect_summary,
            block_mean_shift=block_mean_shift,
            blocks_in_predictions=(block_mean_shift == 0.0),
            n_obs_total=usage['n_obs_total'],
            n_obs_used=usage['n_obs_used'],
            n_obs_excluded=usage['n_obs_excluded'],
            excluded_obs_labels=usage['excluded_obs_labels'],
            used_row_indices=usage['used_row_indices'],
            used_response=usage['used_response'],
        )

    def _block_mean_shift_if_excluded(self, fitted_model) -> float:
        """
        Size-weighted mean block offset, or 0 when Block stays predictive.

        Returns a non-zero shift exactly when the design is blocked with a
        fixed Block effect and predictive use of Block is not requested.  The
        offset is the size-weighted mean (over blocks) of each block's fitted
        dummy contribution: the reference block contributes 0, every non-
        reference block its ``Block[T.level]`` coefficient.  Subtracting it
        from the intercept makes the prediction the block-average response,
        which is independent of whichever level was chosen as the reference.
        """
        if not (
            self.design_structure.get('has_blocking') and not self.block_as_random
        ):
            return 0.0
        if self.include_block_in_prediction:
            return 0.0
        model = getattr(fitted_model, 'model', None)
        frame = getattr(getattr(model, 'data', None), 'frame', None)
        if frame is None or 'Block' not in frame.columns:
            return 0.0
        block = frame['Block']
        unique = block.unique()
        if len(unique) < 2:
            return 0.0
        sizes = frame.groupby(block, sort=False).size()
        total = int(sizes.sum())
        if total <= 0:
            return 0.0
        contribution = pd.Series(0.0, index=sizes.index, dtype=float)
        params = pd.Series(fitted_model.params)
        for key in params.index:
            match = re.match(r"^Block\[T\.(.*)\]$", str(key))
            if match and match.group(1) in contribution.index:
                contribution.loc[match.group(1)] = float(params[key])
        return float((contribution * sizes).sum() / total)
    
    def _compute_diagnostics(self, residuals: np.ndarray, fitted_values: np.ndarray) -> Dict:
        """Compute diagnostics."""
        diagnostics = {}
        if len(residuals) <= 5000:
            stat, pval = stats.shapiro(residuals)
            diagnostics['shapiro_wilk'] = {'statistic': stat, 'p_value': pval}
        return diagnostics
    
    def _validate_degrees_of_freedom(self, model_terms: List[str]) -> None:
        """Validate DF and warn about saturation."""
        n_runs = len(self.data)
        n_params = 1 + self._estimate_n_parameters(model_terms)
        df_error = n_runs - n_params

        if df_error < 0:
            raise ValueError(
                f"Model is oversaturated: {n_runs} runs for {n_params} parameters "
                f"(df_error = {df_error}). Reduce model complexity or add more runs."
            )
        elif df_error == 0:
            warnings.warn(
                f"Model is saturated (df_error = 0): No degrees of freedom for error estimation. "
                f"Statistical tests will not be valid. Consider adding runs or simplifying model."
            )
        elif df_error < 3:
            warnings.warn(f"Low df_error = {df_error}: Inference may be unreliable")

    def _estimate_n_parameters(self, model_terms: List[str]) -> int:
        """
        Estimate the number of model parameters consumed by each term,
        counting categorical dummy expansion.

        A categorical main effect with k levels is fit as ``k-1`` dummy
        columns; an interaction ``A*B`` consumes ``df_A * df_B`` columns, where
        ``df`` of a categorical factor is ``k-1`` and ``1`` otherwise.
        """
        factor_df: Dict[str, int] = {}
        for factor in self.factors:
            if factor.is_categorical():
                n_levels = len(factor.levels)
                if n_levels < 2:
                    n_levels = self.data[factor.name].nunique()
                factor_df[factor.name] = max(n_levels - 1, 0)
            else:
                factor_df[factor.name] = 1

        total = 0
        for term in model_terms:
            if term == '1':
                continue
            factor_list, op = parse_model_term(term)
            if op == '*':
                dfs = [factor_df.get(name, 1) for name in factor_list]
                if not dfs or any(df <= 0 for df in dfs):
                    continue
                total += int(np.prod(dfs))
            else:
                name = factor_list[0] if factor_list else ''
                total += factor_df.get(name, 1)
        return total

    def _validate_split_plot_degrees_of_freedom(self) -> None:
        """Warn when the whole-plot stratum has too few groups for reliable inference."""
        whole_plot_col = self.design_structure.get('whole_plot_column')
        if whole_plot_col is None:
            return
        n_wp = self.data[whole_plot_col].nunique()
        if n_wp < 3:
            warnings.warn(f"Only {n_wp} whole-plots, need ≥3 for reliable whole-plot inference.")
    
    def update_model(
        self,
        terms_to_add: Optional[List[str]] = None,
        terms_to_remove: Optional[List[str]] = None,
        enforce_hierarchy_flag: bool = True
    ) -> ANOVAResults:
        """Update model by adding/removing terms."""
        if self.current_model is None:
            raise ValueError("No model fitted yet")
        
        new_terms = self.current_model.copy()
        
        if terms_to_add:
            new_terms.extend(terms_to_add)
        
        if terms_to_remove:
            new_terms = [t for t in new_terms if t not in terms_to_remove]
        
        seen = set()
        new_terms = [t for t in new_terms if not (t in seen or seen.add(t))]
        
        return self.fit(new_terms, enforce_hierarchy_flag)