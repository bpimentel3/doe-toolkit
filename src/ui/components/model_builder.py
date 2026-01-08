"""
Model Builder Component - JMP-style term selection interface.

Provides visual term building with mathematical notation display.
"""
from typing import List, Set, Tuple
import streamlit as st
from src.core.factors import Factor, FactorType


def format_term_for_display(term: str) -> str:
    """
    Convert Patsy notation to mathematical notation.
    
    Parameters
    ----------
    term : str
        Term in Patsy format ('A', 'A*B', 'I(A**2)', 'I(A**2)*B', '1')
    
    Returns
    -------
    str
        Term in mathematical notation (β₀, A, A×B, A², A²×B)
    
    Examples
    --------
    >>> format_term_for_display('1')
    'β₀'
    >>> format_term_for_display('A*B')
    'A×B'
    >>> format_term_for_display('I(A**2)')
    'A²'
    >>> format_term_for_display('I(A**2)*B')
    'A²×B'
    >>> format_term_for_display('I(A**3)')
    'A³'
    """
    # Superscript mapping
    superscripts = {'0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
                    '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹'}
    
    if term == '1':
        return 'β₀'
    elif term.startswith('I(') and '**' in term:
        # Power term (possibly with interactions)
        # Examples: I(A**2), I(A**3), I(A**2)*B, I(A**3)*B*C
        
        # Extract the power part
        if '*' in term and ')' in term:
            # Power interaction: I(A**2)*B
            power_part = term[2:term.index(')')]
            interaction_part = term[term.index(')')+2:]  # Skip )*
            
            # Format power: A**2 -> A²
            factor, power = power_part.split('**')
            power_display = ''.join(superscripts.get(d, d) for d in power)
            formatted_power = f"{factor}{power_display}"
            
            # Format interaction: B*C -> B×C
            interaction_factors = interaction_part.split('*')
            formatted_interaction = '×'.join(interaction_factors)
            
            return f"{formatted_power}×{formatted_interaction}"
        else:
            # Pure power: I(A**2)
            power_content = term[2:-1]  # Remove I( and )
            factor, power = power_content.split('**')
            power_display = ''.join(superscripts.get(d, d) for d in power)
            return f'{factor}{power_display}'
    elif '*' in term:
        # Interaction: A*B -> A×B
        factors = term.split('*')
        return '×'.join(factors)
    else:
        # Main effect: A -> A
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
    (['1', 'A', 'B', 'A*B', 'I(A**2)', 'I(B**2)'], '')
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
        # Check continuous factors requirement
        if not continuous_factors:
            message = "⚠️ Quadratic preset requires continuous factors. Using Interaction model instead."
            # Fallback to interaction
            for i in range(len(factor_names)):
                for j in range(i + 1, len(factor_names)):
                    terms.append(f"{factor_names[i]}*{factor_names[j]}")
        else:
            # Two-way interactions (all)
            for i in range(len(factor_names)):
                for j in range(i + 1, len(factor_names)):
                    terms.append(f"{factor_names[i]}*{factor_names[j]}")
            
            # Quadratic terms (continuous only)
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
    
    # Custom CSS for smaller buttons
    st.markdown("""
        <style>
        div[data-testid="column"] > div > div > button {
            padding: 0.25rem 0.5rem;
            font-size: 0.85rem;
        }
        </style>
    """, unsafe_allow_html=True)
    
    op_row = st.columns([1, 1, 1, 1, 1, 1, 1.5, 1.5])
    
    with op_row[0]:
        if st.button("Main", disabled=not can_main, key=f"{key_prefix}_main", 
                     use_container_width=True, help="Add as main effects"):
            for factor in selected_factors:
                if factor not in current_terms:
                    current_terms.append(factor)
            st.session_state[f'{key_prefix}_selected_factors'] = []
            return current_terms
    
    with op_row[1]:
        if st.button("×", disabled=not can_cross, key=f"{key_prefix}_cross",
                     use_container_width=True, help="Cross (interaction)"):
            term = '*'.join(sorted(selected_factors))
            if term not in current_terms:
                current_terms.append(term)
            st.session_state[f'{key_prefix}_selected_factors'] = []
            return current_terms
    
    with op_row[2]:
        if st.button("²", disabled=not can_power, key=f"{key_prefix}_square",
                     use_container_width=True, help="Square"):
            term = f"I({selected_factors[0]}**2)"
            if term not in current_terms:
                current_terms.append(term)
            st.session_state[f'{key_prefix}_selected_factors'] = []
            return current_terms
    
    with op_row[3]:
        if st.button("³", disabled=not can_power, key=f"{key_prefix}_cube",
                     use_container_width=True, help="Cube"):
            term = f"I({selected_factors[0]}**3)"
            if term not in current_terms:
                current_terms.append(term)
            st.session_state[f'{key_prefix}_selected_factors'] = []
            return current_terms
    
    # Divider between custom and presets
    with op_row[4]:
        st.markdown("<div style='text-align: center; padding: 0.25rem;'>|</div>", 
                   unsafe_allow_html=True)
    
    # Preset buttons
    include_intercept = '1' in current_terms
    
    with op_row[5]:
        if st.button("β₀", key=f"{key_prefix}_intercept_toggle",
                     use_container_width=True, help="Toggle intercept",
                     type="primary" if include_intercept else "secondary"):
            if include_intercept:
                current_terms = [t for t in current_terms if t != '1']
            else:
                current_terms.insert(0, '1')
            return current_terms
    
    with op_row[6]:
        if st.button("Linear", key=f"{key_prefix}_linear", use_container_width=True):
            new_terms, warning = get_preset_terms('Linear', factors, include_intercept)
            if warning:
                st.warning(warning)
            return new_terms
    
    with op_row[7]:
        if st.button("Quadratic", key=f"{key_prefix}_quad", use_container_width=True):
            new_terms, warning = get_preset_terms('Quadratic', factors, include_intercept)
            if warning:
                st.warning(warning)
            return new_terms
    
    # Row 3: More presets
    preset_row = st.columns([1, 1, 1, 1, 4])
    
    with preset_row[0]:
        if st.button("RSM", key=f"{key_prefix}_rsm", use_container_width=True):
            new_terms, warning = get_preset_terms('RSM', factors, include_intercept)
            if warning:
                st.warning(warning)
            return new_terms
    
    with preset_row[1]:
        if st.button("2FI", key=f"{key_prefix}_full2fi", use_container_width=True,
                     help="Full 2-way interactions"):
            new_terms, warning = get_preset_terms('Full Interaction', factors, include_intercept)
            if warning:
                st.warning(warning)
            return new_terms
    
    with preset_row[2]:
        power_input = st.number_input("^", min_value=2, max_value=10, value=2, step=1,
                                     key=f"{key_prefix}_power_input", label_visibility="collapsed",
                                     help="Custom power")
    
    with preset_row[3]:
        if st.button(f"^{power_input}", disabled=not can_power,
                     key=f"{key_prefix}_custom_power", use_container_width=True):
            term = f"I({selected_factors[0]}**{power_input})"
            if term not in current_terms:
                current_terms.append(term)
            st.session_state[f'{key_prefix}_selected_factors'] = []
            return current_terms
    
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
        
        # Group terms by type
        intercept_terms = [t for t in current_terms if t == '1']
        main_effects = [t for t in current_terms if t not in intercept_terms and '*' not in t and not t.startswith('I(')]
        interactions = [t for t in current_terms if '*' in t and not t.startswith('I(')]
        powers = [t for t in current_terms if t.startswith('I(') and '*' not in t]
        power_interactions = [t for t in current_terms if t.startswith('I(') and '*' in t]
        
        # Track if user clicked remove
        remove_term = None
        
        # Create 5-column layout
        cols = st.columns(5)
        
        # Column 0: Intercept
        with cols[0]:
            st.markdown("*Intercept*")
            if intercept_terms:
                for term in intercept_terms:
                    display = format_term_for_display(term)
                    btn_key = f"{key_prefix}_rm_intercept_{hash(term) % 10000}"
                    if st.button(f"❌ {display}", key=btn_key, use_container_width=True):
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
                    if st.button(f"❌ {display}", key=btn_key, use_container_width=True):
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
                    if st.button(f"❌ {display}", key=btn_key, use_container_width=True):
                        remove_term = term
            else:
                st.caption("_(none)_")
        
        # Column 3: Powers
        with cols[3]:
            st.markdown("*Power*")
            if powers:
                for idx, term in enumerate(powers):
                    display = format_term_for_display(term)
                    btn_key = f"{key_prefix}_rm_pow_{idx}_{hash(term) % 10000}"
                    if st.button(f"❌ {display}", key=btn_key, use_container_width=True):
                        remove_term = term
            else:
                st.caption("_(none)_")
        
        # Column 4: Power Interactions
        with cols[4]:
            st.markdown("*Power×*")
            if power_interactions:
                for idx, term in enumerate(power_interactions):
                    display = format_term_for_display(term)
                    btn_key = f"{key_prefix}_rm_powint_{idx}_{hash(term) % 10000}"
                    if st.button(f"❌ {display}", key=btn_key, use_container_width=True):
                        remove_term = term
            else:
                st.caption("_(none)_")
        
        # Handle term removal
        if remove_term:
            current_terms = [t for t in current_terms if t != remove_term]
            # Don't call st.rerun() here - return updated terms and let parent handle it
    
    return current_terms
