"""
Constraint Builder Component

Reusable UI component for defining linear constraints on factors.
Used in D-Optimal design and augmentation workflows.
"""
import streamlit as st
from typing import List, Dict, Tuple, Optional
from src.core.factors import Factor
from src.core.optimal import LinearConstraint


def format_constraint_preview(
    coefficients: Dict[str, float], 
    bound: float, 
    constraint_type: str
) -> str:
    """
    Format constraint as readable string.
    
    Parameters
    ----------
    coefficients : Dict[str, float]
        Factor coefficients, e.g., {'Temperature': 1.0, 'Pressure': 0.5}
    bound : float
        Right-hand side bound
    constraint_type : str
        One of 'le', 'ge', 'eq'
    
    Returns
    -------
    str
        Formatted constraint string, e.g., "Temperature + 0.5*Pressure ≤ 100"
    
    Examples
    --------
    >>> format_constraint_preview({'A': 1.0, 'B': 0.5}, 100, 'le')
    'A + 0.5*B ≤ 100'
    """
    terms = []
    for factor, coef in coefficients.items():
        if coef == 1.0:
            terms.append(factor)
        elif coef == -1.0:
            terms.append(f"-{factor}")
        elif coef > 0:
            term = f"{coef:+.3g}*{factor}" if terms else f"{coef:.3g}*{factor}"
            terms.append(term)
        else:  # coef < 0
            terms.append(f"{coef:.3g}*{factor}")
    
    lhs = " ".join(terms).lstrip('+').strip()
    
    symbol = {'le': '≤', 'ge': '≥', 'eq': '='}[constraint_type]
    
    return f"{lhs} {symbol} {bound:.3g}"


def validate_constraints(
    constraints: List[LinearConstraint], 
    factors: List[Factor]
) -> Tuple[bool, List[str]]:
    """
    Validate constraints for common issues.
    
    Parameters
    ----------
    constraints : List[LinearConstraint]
        Constraints to validate
    factors : List[Factor]
        Available factors
    
    Returns
    -------
    is_valid : bool
        Whether constraints are valid
    warnings : List[str]
        List of warning/error messages
    """
    warnings = []
    
    # Get continuous factor names
    factor_names = {f.name for f in factors if f.is_continuous()}
    
    # Check: All factors in constraint exist and are continuous
    for i, constraint in enumerate(constraints):
        unknown = set(constraint.coefficients.keys()) - factor_names
        if unknown:
            warnings.append(
                f"Constraint {i+1} references unknown/non-continuous factors: {unknown}"
            )
    
    # Valid if no unknown factor errors
    is_valid = len([w for w in warnings if 'unknown' in w.lower()]) == 0
    
    return is_valid, warnings


def display_constraint_card(
    constraint: LinearConstraint, 
    index: int, 
    factors: List[Factor],
    allow_delete: bool = True
) -> None:
    """
    Display a single constraint with optional delete button.
    
    Parameters
    ----------
    constraint : LinearConstraint
        Constraint to display
    index : int
        Index in constraint list
    factors : List[Factor]
        Available factors (for formatting)
    allow_delete : bool, optional
        Whether to show delete button
    """
    constraint_str = format_constraint_preview(
        constraint.coefficients, 
        constraint.bound, 
        constraint.constraint_type
    )
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.code(constraint_str, language=None)
    
    with col2:
        if allow_delete:
            if st.button("🗑️", key=f"delete_constraint_{index}"):
                del st.session_state['constraints'][index]
                st.rerun()


def show_constraint_form(factors: List[Factor]) -> None:
    """
    Display form to add/edit linear constraint.
    
    Parameters
    ----------
    factors : List[Factor]
        Available factors (only continuous can be used in constraints)
    """
    st.markdown("#### Add Linear Constraint")
    
    st.markdown("""
    Define a linear constraint on factors. Examples:
    - `Temperature + 0.5*Pressure ≤ 100`
    - `Time - 2*Concentration ≥ 10`
    - `Cost_A + Cost_B ≤ 1000`
    """)
    
    # Filter to continuous factors only
    continuous_factors = [f for f in factors if f.is_continuous()]
    
    if not continuous_factors:
        st.error("No continuous factors available for constraints")
        return
    
    with st.form("constraint_form"):
        # Constraint type
        constraint_type = st.selectbox(
            "Constraint Type",
            options=['le', 'ge', 'eq'],
            format_func=lambda x: {
                'le': '≤ (Less than or equal)', 
                'ge': '≥ (Greater than or equal)', 
                'eq': '= (Equal to)'
            }[x]
        )
        
        # Factor coefficients
        st.markdown("**Factor Coefficients:**")
        st.caption("Set coefficient to 0 to exclude a factor from this constraint")
        
        coefficients = {}
        
        # Create two columns for better layout
        n_factors = len(continuous_factors)
        n_cols = 2 if n_factors > 3 else 1
        cols = st.columns(n_cols)
        
        for i, factor in enumerate(continuous_factors):
            col_idx = i % n_cols
            with cols[col_idx]:
                coef = st.number_input(
                    f"{factor.name}",
                    value=0.0,
                    step=0.1,
                    format="%.3f",
                    key=f'coef_{factor.name}'
                )
                if coef != 0:
                    coefficients[factor.name] = coef
        
        # Bound
        bound = st.number_input(
            "Bound (right-hand side)",
            value=0.0,
            step=1.0,
            format="%.3f"
        )
        
        # Preview constraint
        if coefficients:
            constraint_str = format_constraint_preview(
                coefficients, bound, constraint_type
            )
            st.info(f"**Preview:** {constraint_str}")
        else:
            st.warning("⚠️ No factors selected (all coefficients are 0)")
        
        # Submit buttons
        col1, col2 = st.columns(2)
        
        with col1:
            submitted = st.form_submit_button("✓ Add Constraint", type="primary")
        
        with col2:
            cancelled = st.form_submit_button("✗ Cancel")
        
        if submitted:
            if not coefficients:
                st.error("At least one factor must have non-zero coefficient")
            else:
                constraint = LinearConstraint(
                    coefficients=coefficients,
                    bound=bound,
                    constraint_type=constraint_type
                )
                
                st.session_state['constraints'].append(constraint)
                st.session_state['show_constraint_form'] = False
                st.success("✓ Constraint added!")
                st.rerun()
        
        if cancelled:
            st.session_state['show_constraint_form'] = False
            st.rerun()


def show_constraint_builder(factors: List[Factor]) -> None:
    """
    Main constraint builder interface.
    
    Displays existing constraints and provides UI to add/remove constraints.
    Stores constraints in st.session_state['constraints'].
    
    Parameters
    ----------
    factors : List[Factor]
        Available factors
    """
    st.markdown("### 📐 Constraints (Optional)")
    
    # Initialize constraints in session state
    if 'constraints' not in st.session_state:
        st.session_state['constraints'] = []
    
    constraints = st.session_state['constraints']
    
    # Display existing constraints
    if constraints:
        st.markdown(f"**Current Constraints ({len(constraints)}):**")
        for i, constraint in enumerate(constraints):
            display_constraint_card(constraint, index=i, factors=factors)
    else:
        st.info("No constraints defined. Add constraints to restrict the design space.")
    
    # Add constraint button
    if st.button("➕ Add Constraint"):
        st.session_state['show_constraint_form'] = True
    
    # Constraint form (modal-style)
    if st.session_state.get('show_constraint_form'):
        show_constraint_form(factors)


def show_constraint_help() -> None:
    """Display help information about constraints."""
    with st.expander("ℹ️ Constraint Examples & Tips"):
        st.markdown("""
        **Common Use Cases:**
        
        1. **Budget Constraint:**
           - `Material_Cost + Labor_Cost ≤ 1000`
           - Limits total cost of experimental run
        
        2. **Physical Limits:**
           - `Temperature ≤ 200` (equipment maximum)
           - `Pressure ≥ 1` (minimum safe pressure)
        
        3. **Process Constraints:**
           - `Catalyst + Solvent ≤ 100` (total volume)
           - `Time - 2*Temperature ≥ 0` (process relationship)
        
        4. **Safety Constraints:**
           - `Temperature + 0.5*Pressure ≤ 150`
           - Combined thermal stress limit
        
        **Tips:**
        - Use actual factor values (not coded -1/+1)
        - Constraints reduce feasible design space
        - Too many constraints may limit candidate pool
        - Check constraint preview before adding
        - Only continuous factors can be used in constraints
        """)
