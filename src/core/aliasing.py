"""
Aliasing and generator management for fractional factorial designs.

This module handles:
- Generator parsing and validation
- Factor name mapping (real names <-> algebraic symbols)
- Defining relation computation
- Alias structure calculation
- Resolution determination
"""

from typing import List, Dict, Tuple, Optional, Set
import pandas as pd
import itertools

from src.core.factors import Factor


# Standard generators from Box-Hunter-Hunter and Montgomery
STANDARD_GENERATORS = {
    # 2^(k-1) half-fractions
    (4, 1, 4): [("D", "ABC")],
    (5, 1, 5): [("E", "ABCD")],
    (6, 1, 6): [("F", "ABCDE")],
    (7, 1, 7): [("G", "ABCDEF")],
    
    # 2^(k-2) quarter-fractions
    (5, 2, 3): [("D", "AB"), ("E", "AC")],
    (6, 2, 4): [("E", "ABC"), ("F", "BCD")],
    (7, 2, 4): [("F", "ABCD"), ("G", "ABCE")],
    (8, 2, 5): [("G", "ABCD"), ("H", "ABEF")],
    
    # 2^(k-3) eighth-fractions
    (6, 3, 3): [("D", "AB"), ("E", "AC"), ("F", "BC")],
    (7, 3, 4): [("E", "ABC"), ("F", "BCD"), ("G", "ACD")],
    (8, 3, 4): [("F", "ABC"), ("G", "ABD"), ("H", "ABE")],
    
    # 2^(k-4) sixteenth-fractions
    (8, 4, 4): [("E", "BCD"), ("F", "ACD"), ("G", "ABC"), ("H", "ABD")],
    (9, 4, 4): [("F", "ABCD"), ("G", "ABCE"), ("H", "ABDE"), ("J", "BCDE")],
    (10, 4, 4): [("G", "ABCD"), ("H", "ABEF"), ("J", "ACEF"), ("K", "BCEF")],
}


class FactorMapper:
    """
    Bidirectional mapping between real factor names and algebraic symbols.
    
    Parameters
    ----------
    factors : List[Factor]
        List of factors in order
    
    Examples
    --------
    >>> factors = [Factor("Temperature", ...), Factor("Pressure", ...)]
    >>> mapper = FactorMapper(factors)
    >>> mapper.to_algebraic("Temperature")
    'A'
    >>> mapper.to_real("B")
    'Pressure'
    >>> mapper.translate_generator("C=AB", to_algebraic=False)
    'Material=Temperature*Pressure'
    """
    
    def __init__(self, factors: List[Factor]):
        self.factors = factors
        self.n_factors = len(factors)
        
        # Create bidirectional mappings
        self._real_to_algebraic = {}
        self._algebraic_to_real = {}
        
        for i, factor in enumerate(factors):
            symbol = chr(65 + i)  # A, B, C, ...
            self._real_to_algebraic[factor.name] = symbol
            self._algebraic_to_real[symbol] = factor.name
        
        self.algebraic_symbols = list(self._algebraic_to_real.keys())
        self.real_names = list(self._real_to_algebraic.keys())
    
    def to_algebraic(self, factor_name: str) -> str:
        """Convert real factor name to algebraic symbol."""
        if factor_name not in self._real_to_algebraic:
            raise ValueError(f"Unknown factor name: '{factor_name}'")
        return self._real_to_algebraic[factor_name]
    
    def to_real(self, symbol: str) -> str:
        """Convert algebraic symbol to real factor name."""
        if symbol not in self._algebraic_to_real:
            raise ValueError(f"Unknown symbol: '{symbol}'")
        return self._algebraic_to_real[symbol]
    
    def translate_generator(self, gen_str: str, to_algebraic: bool = True) -> str:
        """
        Translate generator between real names and algebraic symbols.
        
        Parameters
        ----------
        gen_str : str
            Generator string (e.g., "D=ABC" or "Pressure=Temperature*Time")
        to_algebraic : bool
            If True, translate to algebraic. If False, translate to real names.
        
        Returns
        -------
        str
            Translated generator string
        """
        if '=' not in gen_str:
            raise ValueError(f"Invalid generator format: '{gen_str}'")
        
        left, right = gen_str.split('=')
        left = left.strip()
        right = right.strip()
        
        if to_algebraic:
            # Real names -> Algebraic
            # Left side: single factor name
            left_trans = self.to_algebraic(left)
            
            # Right side: could be "ABC" (algebraic) or "Temperature*Time*Speed" (real names)
            # Check if it looks algebraic (single letters, no *)
            if '*' in right:
                # Has separators, treat as real names
                right_trans = ''.join([self.to_algebraic(f.strip()) for f in right.split('*')])
            else:
                # No separators - could be algebraic already or single real name
                if len(right) == 1 and right in self.algebraic_symbols:
                    # Single algebraic letter
                    right_trans = right
                elif right in self.real_names:
                    # Single real name
                    right_trans = self.to_algebraic(right)
                else:
                    # Multiple letters, assume algebraic (e.g., "ABC")
                    right_trans = right
        else:
            # Algebraic -> Real names
            left_trans = self.to_real(left)
            right_trans = '*'.join([self.to_real(s) for s in right])
        
        return f"{left_trans}={right_trans}"


class GeneratorValidator:
    """
    Validates generator specifications.
    
    Parameters
    ----------
    k : int
        Number of factors
    p : int
        Number of generators
    mapper : FactorMapper
        Factor name mapper
    """
    
    def __init__(self, k: int, p: int, mapper: FactorMapper):
        self.k = k
        self.p = p
        self.mapper = mapper
    
    def validate_generator_syntax(self, gen_str: str) -> None:
        """Validate generator syntax."""
        if '=' not in gen_str:
            raise ValueError(f"Generator must contain '=': '{gen_str}'")
        
        parts = gen_str.split('=')
        if len(parts) != 2:
            raise ValueError(f"Generator must have exactly one '=': '{gen_str}'")
        
        left, right = parts
        left = left.strip()
        right = right.strip()
        
        if not left:
            raise ValueError(f"Left side empty: '{gen_str}'")
        
        if not right:
            raise ValueError(f"Right side empty: '{gen_str}'")
        
        if len(left) != 1:
            raise ValueError(f"Left side must be single factor: '{gen_str}'")
        
        if not left.isalpha() or not right.replace('*', '').isalpha():
            raise ValueError(f"Generator must contain only letters: '{gen_str}'")
    
    def validate_factors_exist(self, gen_str: str) -> None:
        """Validate that all factors in generator exist."""
        left, right = gen_str.split('=')
        left = left.strip()
        right = right.strip().replace('*', '')
        
        # Check that generated factor is the next one in sequence
        n_base = self.k - self.p
        expected_symbol = chr(65 + n_base)  # First generated factor
        
        available = self.mapper.algebraic_symbols[:n_base]
        
        # Check right side factors exist
        for factor in right:
            if factor not in available:
                raise ValueError(
                    f"Factor '{factor}' in '{gen_str}' not in base factors. "
                    f"Available: {', '.join(available)}"
                )
    
    def validate_generator_count(self) -> None:
        """Validate that number of generators matches fraction."""
        if self.p >= self.k:
            raise ValueError(
                f"Number of generators ({self.p}) must be less than "
                f"number of factors ({self.k})"
            )
    
    def validate_all(self, generators: List[str]) -> None:
        """Run all validations."""
        self.validate_generator_count()
        
        if len(generators) != self.p:
            raise ValueError(
                f"Expected {self.p} generators for 1/{2**self.p} fraction, "
                f"got {len(generators)}"
            )
        
        for gen in generators:
            self.validate_generator_syntax(gen)
            self.validate_factors_exist(gen)


class AliasingEngine:
    """
    Computes defining relations, alias structures, and resolution.
    
    Parameters
    ----------
    k : int
        Number of factors
    generators : List[Tuple[str, str]]
        List of (factor, expression) tuples in algebraic form
    
    Examples
    --------
    >>> engine = AliasingEngine(5, [("E", "ABCD")])
    >>> engine.defining_relation
    ['I', 'ABCDE']
    >>> engine.resolution
    5
    >>> engine.alias_structure['A']
    ['BCDE']
    """
    
    def __init__(self, k: int, generators: List[Tuple[str, str]]):
        self.k = k
        self.generators = generators
        self.p = len(generators)
        
        self.defining_relation = self._build_defining_relation()
        self.resolution = self._calculate_resolution()
        self.alias_structure = self._calculate_alias_structure()
    
    def _build_defining_relation(self) -> List[str]:
        """Build complete defining relation from generators."""
        words = ["I"]
        
        # Add generators
        for factor, expression in self.generators:
            word = self._simplify_word(factor + expression)
            words.append(word)
        
        # Generate all products of generators
        n_generators = len(self.generators)
        for r in range(2, n_generators + 1):
            for combo in itertools.combinations(range(n_generators), r):
                # Multiply the generators
                word = ""
                for idx in combo:
                    factor, expression = self.generators[idx]
                    word += factor + expression
                
                # Simplify
                word = self._simplify_word(word)
                if word and word not in words:
                    words.append(word)
        
        return words
    
    def _simplify_word(self, word: str) -> str:
        """Simplify word using mod-2 algebra (A*A=I)."""
        counts = {}
        for letter in word:
            counts[letter] = counts.get(letter, 0) + 1
        
        # Keep only odd counts, sorted alphabetically
        result = ''.join(sorted([letter for letter in counts if counts[letter] % 2 == 1]))
        
        return result
    
    def _calculate_resolution(self) -> int:
        """Calculate design resolution (min word length excluding I)."""
        min_length = float('inf')
        
        for word in self.defining_relation:
            if word != "I":
                min_length = min(min_length, len(word))
        
        return int(min_length) if min_length != float('inf') else 0
    
    def _calculate_alias_structure(self) -> Dict[str, List[str]]:
        """Calculate complete alias structure."""
        alias_structure = {}
        
        # Factor names
        factor_names = [chr(65 + i) for i in range(self.k)]
        
        # All effects up to 4th order
        all_effects = self._generate_effects(factor_names, max_order=4)
        
        for effect in all_effects:
            aliases = set()
            
            # Multiply by each word in defining relation
            for word in self.defining_relation:
                if word == "I":
                    continue
                
                alias = self._simplify_word(effect + word)
                if alias and alias != effect:
                    aliases.add(alias)
            
            if aliases:
                alias_structure[effect] = sorted(list(aliases), key=lambda x: (len(x), x))
        
        return alias_structure
    
    def _generate_effects(self, factor_names: List[str], max_order: int) -> List[str]:
        """Generate all effects up to specified order."""
        effects = []
        
        for order in range(1, min(max_order, len(factor_names)) + 1):
            for combo in itertools.combinations(factor_names, order):
                effects.append("".join(combo))
        
        return effects


def parse_generators(generator_strings: List[str]) -> List[Tuple[str, str]]:
    """
    Parse generator strings into (factor, expression) tuples.
    
    Parameters
    ----------
    generator_strings : List[str]
        List of generator strings like ["D=ABC", "E=BCD"]
    
    Returns
    -------
    List[Tuple[str, str]]
    List[Tuple[str, str]]
        List of (factor, expression) tuples
    """
    parsed = []
    for gen_str in generator_strings:
        if "=" not in gen_str:
            raise ValueError(f"Invalid generator: '{gen_str}'")
        
        parts = gen_str.split("=")
        if len(parts) != 2:
            raise ValueError(f"Invalid generator: '{gen_str}'")
        
        factor = parts[0].strip()
        expression = parts[1].strip().replace('*', '')
        
        parsed.append((factor, expression))
    
    return parsed


def get_standard_generators(k: int, p: int, resolution: int) -> Optional[List[Tuple[str, str]]]:
    """
    Get standard generators from library.
    
    Parameters
    ----------
    k : int
        Number of factors
    p : int
        Number of generators
    resolution : int
        Desired resolution
    
    Returns
    -------
    Optional[List[Tuple[str, str]]]
        Standard generators if available, None otherwise
    """
    key = (k, p, resolution)
    
    if key in STANDARD_GENERATORS:
        return STANDARD_GENERATORS[key]
    
    return None


def format_alias_table(
    alias_structure: Dict[str, List[str]],
    mapper: Optional[FactorMapper] = None
) -> pd.DataFrame:
    """
    Format alias structure as readable table.
    
    Parameters
    ----------
    alias_structure : Dict[str, List[str]]
        Alias structure from AliasingEngine
    mapper : FactorMapper, optional
        If provided, translates to real factor names
    
    Returns
    -------
    pd.DataFrame
        Formatted alias table
    """
    rows = []
    
    for effect, aliases in sorted(alias_structure.items(), 
                                   key=lambda x: (len(x[0]), x[0])):
        effect_display = effect
        aliases_display = ' + '.join(aliases) if aliases else 'Clear'
        
        # Translate if mapper provided
        if mapper:
            # This is simplified - full translation would expand interactions
            pass
        
        rows.append({
            'Effect': effect_display,
            'Aliased_With': aliases_display
        })
    
    return pd.DataFrame(rows)


def validate_resolution_achievable(
    generators: List[Tuple[str, str]],
    target_resolution: int,
    k: int
) -> None:
    """
    Validate that generators achieve target resolution.
    
    Parameters
    ----------
    generators : List[Tuple[str, str]]
        Generator tuples
    target_resolution : int
        Claimed resolution
    k : int
        Number of factors
    
    Raises
    ------
    ValueError
        If generators don't achieve target resolution
    """
    engine = AliasingEngine(k, generators)
    actual_resolution = engine.resolution
    
    if actual_resolution < target_resolution:
        raise ValueError(
            f"Generators achieve Resolution {actual_resolution}, "
            f"not {target_resolution} as specified"
        )