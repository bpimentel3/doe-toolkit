"""
Model Builder Component - JMP-style term selection interface.

Provides visual term building with mathematical notation display.
"""
from typing import List, Tuple
import streamlit as st
from src.core.factors import Factor


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

# Maps Patsy transform prefix -> (display prefix, display suffix)
# Used by format_term_for_display and by the transform-base option builder.
_TRANSFORM_DISPLAY: dict = {
    "np.log(": ("log(", ")"),
    "np.sqrt(": ("√", ""),
    "np.exp(": ("exp(", ")"),
    "I(1/": ("1/", ""),  # Patsy reciprocal: I(1/A)
}


def _is_transform(fragment: str) -> bool:
    """Return True if fragment starts with a known transform prefix."""
    return any(fragment.startswith(p) for p in _TRANSFORM_DISPLAY)


def _display_fragment(fragment: str) -> str:
    """
    Convert a single Patsy factor fragment to display notation.

    Handles plain factors ('A'), transforms ('np.log(A)'), and
    reciprocals ('I(1/A)').  Does NOT handle power or interaction
    wrapping — that is left to format_term_for_display.

    Parameters
    ----------
    fragment : str
        A single factor fragment in Patsy format.

    Returns
    -------
    str
        Human-readable display string.

    Examples
    --------
    >>> _display_fragment('A')
    'A'
    >>> _display_fragment('np.log(A)')
    'log(A)'
    >>> _display_fragment('np.sqrt(A)')
    '√A'
    >>> _display_fragment('I(1/A)')
    '1/A'
    """
    for prefix, (disp_pre, disp_suf) in _TRANSFORM_DISPLAY.items():
        if fragment.startswith(prefix):
            inner = fragment[len(prefix):]
            # Strip trailing ')' for np.* forms and I(1/ form
            if inner.endswith(')'):
                inner = inner[:-1]
            return f"{disp_pre}{inner}{disp_suf}"
    return fragment


def format_term_for_display(term: str) -> str:
    """
    Convert Patsy notation to mathematical notation.

    Handles plain factors, interactions, power terms, transform terms,
    and combinations thereof (e.g. log(A)², log(A)×B).

    Parameters
    ----------
    term : str
        Term in Patsy format.  Supported forms::

            '1'                     -> 'β₀'
            'A'                     -> 'A'
            'A*B'                   -> 'A×B'
            'I(A**2)'               -> 'A²'
            'I(A**2)*B'             -> 'A²×B'
            'np.log(A)'             -> 'log(A)'
            'np.sqrt(A)'            -> '√A'
            'I(1/A)'                -> '1/A'
            'np.exp(A)'             -> 'exp(A)'
            'I(np.log(A)**2)'       -> 'log(A)²'
            'np.log(A)*B'           -> 'log(A)×B'
            'I(np.log(A)**2)*B'     -> 'log(A)²×B'

    Returns
    -------
    str
        Term in mathematical notation.

    Examples
    --------
    >>> format_term_for_display('1')
    'β₀'
    >>> format_term_for_display('I(A**2)*B')
    'A²×B'
    >>> format_term_for_display('np.log(A)')
    'log(A)'
    >>> format_term_for_display('I(np.log(A)**2)*B')
    'log(A)²×B'
    """
    superscripts = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
    }

    if term == '1':
        return 'β₀'

    # ------------------------------------------------------------------ #
    # I(...**n) — power wrapper, base may itself be a transform            #
    # Examples: I(A**2), I(np.log(A)**2), I(np.log(A)**2)*B              #
    # ------------------------------------------------------------------ #
    if term.startswith('I(') and '**' in term:
        # Find the closing ')' of the I() wrapper
        close = term.index(')')
        power_content = term[2:close]          # e.g. 'A**2' or 'np.log(A)**2'
        remainder = term[close + 1:]           # e.g. '' or '*B'

        base_fragment, exp_str = power_content.rsplit('**', 1)
        exp_display = ''.join(superscripts.get(d, d) for d in exp_str)
        base_display = _display_fragment(base_fragment)
        power_display = f"{base_display}{exp_display}"

        if remainder.startswith('*'):
            # Power × cross: I(A**2)*B  or  I(np.log(A)**2)*B
            cross_factors = remainder[1:].split('*')
            cross_display = '×'.join(_display_fragment(f) for f in cross_factors)
            return f"{power_display}×{cross_display}"

        return power_display

    # ------------------------------------------------------------------ #
    # Transform terms (no power wrapper)                                   #
    # Examples: np.log(A), np.log(A)*B                                    #
    # ------------------------------------------------------------------ #
    if _is_transform(term):
        # May have a trailing cross: np.log(A)*B
        # The transform fragment ends at the first '*' that is OUTSIDE
        # the transform's own parentheses.
        depth = 0
        split_at = len(term)
        for i, ch in enumerate(term):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            elif ch == '*' and depth == 0:
                split_at = i
                break

        base_fragment = term[:split_at]
        remainder = term[split_at:]            # '' or '*B*C'
        base_display = _display_fragment(base_fragment)

        if remainder.startswith('*'):
            cross_factors = remainder[1:].split('*')
            cross_display = '×'.join(_display_fragment(f) for f in cross_factors)
            return f"{base_display}×{cross_display}"

        return base_display

    # ------------------------------------------------------------------ #
    # Plain interaction: A*B                                               #
    # ------------------------------------------------------------------ #
    if '*' in term:
        return '×'.join(_display_fragment(f) for f in term.split('*'))

    # ------------------------------------------------------------------ #
    # Plain main effect: A                                                 #
    # ------------------------------------------------------------------ #
    return term


def format_full_equation(terms: List[str], response_name: str = "Y") -> str:
    """
    Format complete model equation in mathematical notation.
    
    Parameters
    ----------
    terms : List[str]
        Model terms in Patsy format
    response_name : str
        Name of response variable
    
    Returns
    -------
    str
        Formatted equation: Y = β₀ + β₁·A + β₂·B + ...
    
    Examples
    --------
    >>> format_full_equation(['1', 'A', 'B', 'A*B'])
    'Y = β₀ + β₁·A + β₂·B + β₁₂·A×B'
    """
    if not terms:
        return f"{response_name} = (no terms)"
    
    equation_parts = []
    
    for i, term in enumerate(terms):
        display_term = format_term_for_display(term)
        
        if term == '1':
            # Intercept
            equation_parts.append('β₀')
        else:
            # Add coefficient and term
            subscript = str(i) if i < 10 else f"{{{i}}}"
            equation_parts.append(f'β{subscript}·{display_term}')
    
    return f"{response_name} = " + " + ".join(equation_parts)


def get_preset_terms(
    preset: str,
    factors: List[Factor],
    include_intercept: bool
) -> Tuple[List[str], str]:
    """
    Generate preset model terms with validation.
    
    Parameters
    ----------
    preset : str
        One of: 'Linear', 'Quadratic', 'RSM', 'Full Interaction'
    factors : List[Factor]
        Available factors
    include_intercept : bool
        Whether to include intercept term
    
    Returns
    -------
    terms : List[str]
        Generated terms in Patsy format
    message : str
        Warning message if preset modified due to factor types
    
    Examples
    --------
    >>> get_preset_terms('Quadratic', factors, True)
    (['1', 'A', 'B', 'I(A**2)', 'I(B**2)'], '')
    """
    terms = []
    message = ""
    
    factor_names = [f.name for f in factors]
    continuous_factors = [f for f in factors if f.is_continuous()]
    continuous_names = [f.name for f in continuous_factors]
    
    # Intercept
    if include_intercept:
        terms.append('1')
    
    # Main effects (all presets)
    terms.extend(factor_names)
    
    # Preset-specific terms
    if preset == 'Linear':
        # Main effects only
        pass
    
    elif preset == 'Quadratic':
        # Main effects + pure quadratic terms only (no interactions)
        if not continuous_factors:
            message = "⚠️ Quadratic preset requires continuous factors. No quadratic terms added."
        else:
            # Quadratic terms (continuous only) — no interactions
            for name in continuous_names:
                terms.append(f"I({name}**2)")
    
    elif preset == 'RSM':
        # Response Surface Model: interactions + quadratic for continuous
        if not continuous_factors:
            message = "⚠️ RSM preset requires continuous factors. Using Linear model instead."
        elif len(continuous_factors) < 2:
            message = "⚠️ RSM preset requires 2+ continuous factors. Using Quadratic model instead."
            # Add interactions and quadratic for the one continuous factor
            for i in range(len(factor_names)):
                for j in range(i + 1, len(factor_names)):
                    terms.append(f"{factor_names[i]}*{factor_names[j]}")
            for name in continuous_names:
                terms.append(f"I({name}**2)")
        else:
            # Full RSM: all 2-way interactions + quadratic for continuous
            for i in range(len(factor_names)):
                for j in range(i + 1, len(factor_names)):
                    terms.append(f"{factor_names[i]}*{factor_names[j]}")
            
            # Quadratic terms (continuous only)
            for name in continuous_names:
                terms.append(f"I({name}**2)")
    
    elif preset == 'Full Interaction':
        # All two-way interactions
        for i in range(len(factor_names)):
            for j in range(i + 1, len(factor_names)):
                terms.append(f"{factor_names[i]}*{factor_names[j]}")
    
    return terms, message


def display_model_builder(
    factors: List[Factor],
    current_terms: List[str],
    response_name: str,
    key_prefix: str = ""
) -> List[str]:
    """
    Display interactive model builder interface.
    
    Parameters
    ----------
    factors : List[Factor]
        Available factors for model building
    current_terms : List[str]
        Currently selected terms
    response_name : str
        Name of response variable for equation display
    key_prefix : str
        Prefix for Streamlit widget keys to ensure uniqueness
    
    Returns
    -------
    List[str]
        Updated list of selected terms
    """
    st.subheader("🔧 Model Builder")
    
    # Initialize term builder state if needed
    if f'{key_prefix}_selected_factors' not in st.session_state:
        st.session_state[f'{key_prefix}_selected_factors'] = []
    
    factor_names = [f.name for f in factors]
    continuous_names = [f.name for f in factors if f.is_continuous()]
    
    # ========== CONDENSED BUILDER + PRESETS SECTION ==========
    
    # Row 1: Factor checkboxes
    if not factor_names:
        st.warning("⚠️ No factors available. Please define factors in Step 1 first.")
        return current_terms
    
    n_cols = min(6, len(factor_names))
    factor_cols = st.columns(n_cols)
    
    selected_factors = []
    for i, factor_name in enumerate(factor_names):
        col_idx = i % n_cols
        with factor_cols[col_idx]:
            is_continuous = factor_name in continuous_names
            label = f"{factor_name}" + (" •" if is_continuous else "")
            if st.checkbox(
                label,
                value=factor_name in st.session_state[f'{key_prefix}_selected_factors'],
                key=f"{key_prefix}_factor_cb_{factor_name}",
                help="Continuous" if is_continuous else "Categorical"
            ):
                selected_factors.append(factor_name)
    
    st.session_state[f'{key_prefix}_selected_factors'] = selected_factors
    
    # Row 2: Operator buttons (smaller, more compact)
    can_main = len(selected_factors) >= 1
    can_cross = len(selected_factors) >= 2
    can_power = (len(selected_factors) == 1 and
                 selected_factors[0] in continuous_names)
    can_transform = (
        len(selected_factors) == 1
        and selected_factors[0] in continuous_names
    )

    # Custom CSS for smaller buttons
    st.markdown("""
        <style>
        div[data-testid="column"] > div > div > button {
            padding: 0.25rem 0.5rem;
            font-size: 0.85rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Row 2: operator buttons | divider | intercept toggle | preset buttons
    op_row = st.columns([1, 1, 1, 1, 1, 0.5, 1, 1.5, 1.5])

    _tf = selected_factors[0] if can_transform else ""

    with op_row[0]:
        if st.button("Main", disabled=not can_main, key=f"{key_prefix}_main",
                     width='stretch', help="Add as main effects"):
            for factor in selected_factors:
                if factor not in current_terms:
                    current_terms.append(factor)

    with op_row[1]:
        if st.button("×", disabled=not can_cross, key=f"{key_prefix}_cross",
                     width='stretch', help="Cross (interaction)"):
            term = '*'.join(sorted(selected_factors))
            if term not in current_terms:
                current_terms.append(term)

    with op_row[2]:
        if st.button("ln", disabled=not can_transform, key=f"{key_prefix}_tf_log",
                     width='stretch', help="Natural log: log(A)"):
            term = f"np.log({_tf})"
            if term not in current_terms:
                current_terms.append(term)

    with op_row[3]:
        if st.button("√", disabled=not can_transform, key=f"{key_prefix}_tf_sqrt",
                     width='stretch', help="Square root: √A"):
            term = f"np.sqrt({_tf})"
            if term not in current_terms:
                current_terms.append(term)

    with op_row[4]:
        if st.button("1/x", disabled=not can_transform, key=f"{key_prefix}_tf_recip",
                     width='stretch', help="Reciprocal: 1/A"):
            term = f"I(1/{_tf})"
            if term not in current_terms:
                current_terms.append(term)

    with op_row[5]:
        st.markdown("<div style='text-align:center;padding:0.4rem 0;'>|</div>",
                    unsafe_allow_html=True)

    include_intercept = '1' in current_terms

    with op_row[6]:
        if st.button("β₀", key=f"{key_prefix}_intercept_toggle",
                     width='stretch', help="Toggle intercept",
                     type="primary" if include_intercept else "secondary"):
            if include_intercept:
                current_terms = [t for t in current_terms if t != '1']
            else:
                current_terms.insert(0, '1')

    with op_row[7]:
        if st.button("Linear", key=f"{key_prefix}_linear", width='stretch'):
            new_terms, warning = get_preset_terms('Linear', factors, include_intercept)
            if warning:
                st.warning(warning)
            return new_terms

    with op_row[8]:
        if st.button("Quadratic", key=f"{key_prefix}_quad", width='stretch'):
            new_terms, warning = get_preset_terms('Quadratic', factors, include_intercept)
            if warning:
                st.warning(warning)
            return new_terms

    # Row 2b: exp button + preset continuations (exp needs single-factor continuous)
    row2b = st.columns([1, 1, 1, 3])

    with row2b[0]:
        if st.button("exp", disabled=not can_transform, key=f"{key_prefix}_tf_exp",
                     width='stretch', help="Exponential: exp(A)"):
            term = f"np.exp({_tf})"
            if term not in current_terms:
                current_terms.append(term)

    with row2b[1]:
        if st.button("RSM", key=f"{key_prefix}_rsm", width='stretch'):
            new_terms, warning = get_preset_terms('RSM', factors, include_intercept)
            if warning:
                st.warning(warning)
            return new_terms

    with row2b[2]:
        if st.button("2FI", key=f"{key_prefix}_full2fi", width='stretch',
                     help="Full 2-way interactions"):
            new_terms, warning = get_preset_terms('Full Interaction', factors, include_intercept)
            if warning:
                st.warning(warning)
            return new_terms
    
    # Row 3: Custom power term
    preset_row = st.columns([1, 1, 5])

    with preset_row[0]:
        custom_power_val = st.number_input(
            "^",
            min_value=2,
            max_value=10,
            value=2,
            step=1,
            key=f"{key_prefix}_custom_power_input",
            help="Exponent for single-factor power term",
        )

    with preset_row[1]:
        st.markdown("<div style='padding-top: 1.6rem;'>", unsafe_allow_html=True)
        if st.button(
            f"Add ^{custom_power_val}",
            disabled=not can_power,
            key=f"{key_prefix}_custom_power",
            width='stretch',
            help="Raise the selected continuous factor to this power",
        ):
            term = f"I({selected_factors[0]}**{custom_power_val})"
            if term not in current_terms:
                current_terms.append(term)
        st.markdown("</div>", unsafe_allow_html=True)

    # ========== POWER × CROSS TERM EXPANDER ==========

    with st.expander("➕ Advanced Terms", expanded=False):
        # Populate raw continuous options from current selection, else all continuous
        continuous_in_selection = [f for f in selected_factors if f in continuous_names]
        raw_base_options = continuous_in_selection if continuous_in_selection else continuous_names

        cross_candidates = [f for f in selected_factors if f not in continuous_in_selection[:1]]
        cross_factor_options = (
            cross_candidates if cross_candidates
            else [f for f in factor_names if f not in raw_base_options[:1]]
        )

        if not raw_base_options:
            st.caption("⚠️ No continuous factors available to raise to a power.")
        elif not cross_factor_options:
            st.caption("⚠️ Need at least one additional factor to cross with.")
        else:
            # Build (display_label, patsy_fragment) options for the base selectbox.
            # Each continuous factor is offered as raw + all four transforms.
            _tf_specs = [
                ("",        "",       ""),          # raw
                ("ln",      "np.log(", ")"),
                ("√",       "np.sqrt(", ")"),
                ("1/",      "I(1/",    ")"),
                ("exp",     "np.exp(", ")"),
            ]
            base_options: list = []   # list of (label, patsy_fragment)
            for raw_name in raw_base_options:
                for tf_label, tf_pre, tf_suf in _tf_specs:
                    if tf_label == "":
                        label = raw_name
                        fragment = raw_name
                    else:
                        label = f"{tf_label}({raw_name})"
                        fragment = f"{tf_pre}{raw_name}{tf_suf}"
                    base_options.append((label, fragment))

            base_labels = [label for label, _ in base_options]

            pc_col1, pc_col2, pc_col3, pc_col4 = st.columns([2, 1, 2, 2])

            with pc_col1:
                pc_base_idx = st.selectbox(
                    "Raise",
                    options=range(len(base_labels)),
                    format_func=lambda i: base_labels[i],
                    key=f"{key_prefix}_pc_base",
                )
                pc_base_fragment = base_options[pc_base_idx][1]

            with pc_col2:
                pc_exp = st.number_input(
                    "to power",
                    min_value=1,
                    max_value=10,
                    value=2,
                    step=1,
                    key=f"{key_prefix}_pc_exp",
                    help="Use 1 to cross a transform with another factor (e.g. log(A)×B)",
                )

            with pc_col3:
                # Derive raw factor name from the selected base to exclude from cross
                # Raw factor is first token before any '(' in the fragment
                raw_base_name = pc_base_fragment.split('(')[-1].rstrip(')')
                cross_opts = [f for f in cross_factor_options if f != raw_base_name]
                if not cross_opts:
                    cross_opts = [f for f in factor_names if f != raw_base_name]
                pc_cross = st.selectbox(
                    "cross with",
                    options=cross_opts,
                    key=f"{key_prefix}_pc_cross",
                )

            with pc_col4:
                # Build Patsy term. When power=1, skip the I(**1) wrapper —
                # it's redundant for plain factors and cleaner for transforms.
                if pc_exp == 1:
                    pc_patsy = f"{pc_base_fragment}*{pc_cross}"
                else:
                    pc_patsy = f"I({pc_base_fragment}**{pc_exp})*{pc_cross}"
                pc_display = format_term_for_display(pc_patsy)
                st.markdown(
                    f"<div style='padding-top:1.6rem; font-size:1.1em;'>→ "
                    f"<code>{pc_display}</code></div>",
                    unsafe_allow_html=True,
                )

            if st.button(
                "Add Term",
                key=f"{key_prefix}_pc_add",
                type="primary",
                help=f"Add {pc_patsy} to model",
            ):
                if pc_patsy not in current_terms:
                    current_terms.append(pc_patsy)

    st.divider()
    
    # ========== CURRENT MODEL DISPLAY (1.5X FONT SIZE, NO BACKGROUND) ==========
    
    # Live equation preview - reduced font size, no background banner
    equation = format_full_equation(current_terms, response_name)
    st.markdown(
        f"<div style='font-size: 1.5em; padding: 5px 0; margin: 10px 0;'>"
        f"<em>{equation}</em></div>",
        unsafe_allow_html=True
    )
    
    # Display current terms as removable chips
    if len(current_terms) == 0 or (len(current_terms) == 1 and current_terms[0] == '1'):
        st.info("ℹ️ No terms selected. Use builder above.")
    else:
        st.caption(f"{len(current_terms)} terms selected (click ❌ to remove)")
        
        # ------------------------------------------------------------------ #
        # Term classification
        # ------------------------------------------------------------------ #
        # A term is a "transform or power" if it is non-linear and has no
        # cross factor (the rightmost * outside any parens).  This bucket
        # covers: I(A**2), np.log(A), np.sqrt(A), I(1/A), np.exp(A),
        # I(np.log(A)**2), etc.
        #
        # A term is a "transform/power cross" if it is non-linear AND has a
        # cross factor: I(A**2)*B, np.log(A)*B, I(np.log(A)**2)*B.
        #
        # Plain interactions (A*B) stay in their own bucket.
        # Plain main effects (A, Catalyst) stay in theirs.
        _TRANSFORM_STARTS = ("np.log(", "np.sqrt(", "np.exp(", "I(1/")

        def _is_nonlinear(t: str) -> bool:
            """True for any term that is not a plain main effect or plain interaction."""
            return t.startswith("I(") or any(t.startswith(p) for p in _TRANSFORM_STARTS)

        intercept_terms = [t for t in current_terms if t == '1']
        main_effects = [
            t for t in current_terms
            if t != '1' and not _is_nonlinear(t) and '*' not in t
        ]
        interactions = [
            t for t in current_terms
            if not _is_nonlinear(t) and '*' in t
        ]
        # Power/Transform: all non-linear terms (with or without a cross factor)
        powers = [
            t for t in current_terms
            if _is_nonlinear(t)
        ]
        
        # Track if user clicked remove
        remove_term = None
        
        # Create 4-column layout
        cols = st.columns(4)
        
        # Column 0: Intercept
        with cols[0]:
            st.markdown("*Intercept*")
            if intercept_terms:
                for term in intercept_terms:
                    display = format_term_for_display(term)
                    btn_key = f"{key_prefix}_rm_intercept_{hash(term) % 10000}"
                    if st.button(f"❌ {display}", key=btn_key, width='stretch'):
                        remove_term = term
            else:
                st.caption("_(none)_")
        
        # Column 1: Main Effects
        with cols[1]:
            st.markdown("*Main*")
            if main_effects:
                for term in main_effects:
                    display = format_term_for_display(term)
                    btn_key = f"{key_prefix}_rm_main_{term}"
                    if st.button(f"❌ {display}", key=btn_key, width='stretch'):
                        remove_term = term
            else:
                st.caption("_(none)_")
        
        # Column 2: Interactions
        with cols[2]:
            st.markdown("*Interaction*")
            if interactions:
                for idx, term in enumerate(interactions):
                    display = format_term_for_display(term)
                    btn_key = f"{key_prefix}_rm_int_{idx}_{hash(term) % 10000}"
                    if st.button(f"❌ {display}", key=btn_key, width='stretch'):
                        remove_term = term
            else:
                st.caption("_(none)_")
        
        # Column 3: Power / Transform (all non-linear terms)
        with cols[3]:
            st.markdown("*Power/Transform*")
            if powers:
                for idx, term in enumerate(powers):
                    display = format_term_for_display(term)
                    btn_key = f"{key_prefix}_rm_pow_{idx}_{hash(term) % 10000}"
                    if st.button(f"❌ {display}", key=btn_key, width='stretch'):
                        remove_term = term
            else:
                st.caption("_(none)_")
        
        # Handle term removal
        if remove_term:
            current_terms = [t for t in current_terms if t != remove_term]
            # Don't call st.rerun() here - return updated terms and let parent handle it
    
    return current_terms
