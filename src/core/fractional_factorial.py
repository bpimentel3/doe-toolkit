"""
Fractional factorial design generation for DOE-Toolkit.

This module implements 2^(k-p) fractional factorial designs with automatic
generator selection and alias structure calculation.
"""

from typing import List, Dict, Optional, Tuple, Set
import pandas as pd
import numpy as np
import itertools

from src.core.factors import Factor, FactorType
from src.core.full_factorial import full_factorial


class FractionalFactorial:
    """
    Generate 2^(k-p) fractional factorial designs.
    
    A fractional factorial design uses a fraction (1/2^p) of the runs from a
    full factorial design. Generators are used to create the fraction, which
    determines the alias structure (confounding pattern).
    
    Parameters
    ----------
    factors : List[Factor]
        List of 2-level factors (must all be 2-level for fractional factorial)
    fraction : str
        Fraction specification, e.g., "1/2", "1/4", "1/8"
    resolution : int, optional
        Desired resolution (III, IV, or V). If provided, generators are chosen
        automatically to achieve this resolution.
    generators : List[str], optional
        Custom generator strings, e.g., ["D=ABC", "E=BCD"]
        If not provided, generators are selected automatically based on resolution.
    
    Attributes
    ----------
    k : int
        Number of factors
    p : int
        Number of generators (fraction = 1/2^p)
    resolution : int
        Design resolution
    defining_relation : List[str]
        Complete defining relation (generator words)
    alias_structure : Dict
        Complete alias structure for all effects
    
    Examples
    --------
    >>> # Create 2^(5-1) design (Resolution V)
    >>> factors = [Factor(f"Factor_{i}", FactorType.CONTINUOUS,
    ...                   ChangeabilityLevel.EASY, levels=[-1, 1])
    ...            for i in range(5)]
    >>> ff = FractionalFactorial(factors, fraction="1/2", resolution=5)
    >>> design = ff.generate()
    
    >>> # Check alias structure
    >>> print(ff.alias_structure['A'])  # Shows what A is aliased with
    
    References
    ----------
    .. [1] Box, G. E. P., Hunter, J. S., and Hunter, W. G. (2005).
           Statistics for Experimenters, 2nd Ed. Wiley.
    .. [2] Montgomery, D. C. (2017). Design and Analysis of Experiments, 9th Ed.
    """
    
    def __init__(
        self,
        factors: List[Factor],
        fraction: str,
        resolution: Optional[int] = None,
        generators: Optional[List[str]] = None
    ):
        # Validate inputs
        self._validate_factors(factors)
        
        self.factors = factors
        self.k = len(factors)
        self.p = self._parse_fraction(fraction)
        
        # Check if fraction is valid
        if self.p >= self.k:
            raise ValueError(
                f"Cannot create 1/{2**self.p} fraction of {self.k} factors. "
                f"Fraction must be less than number of factors."
            )
        
        # Determine generators
        if generators is not None:
            self.generators = self._parse_generators(generators)
            self.resolution = self._calculate_resolution()
        elif resolution is not None:
            self.resolution = resolution
            self.generators = self._select_generators(resolution)
        else:
            # Default: highest resolution possible
            self.resolution = self._get_max_resolution()
            self.generators = self._select_generators(self.resolution)
        
        # Calculate alias structure
        self.defining_relation = self._calculate_defining_relation()
        self.alias_structure = self._calculate_alias_structure()
    
    def _validate_factors(self, factors: List[Factor]) -> None:
        """Validate that all factors are suitable for fractional factorial."""
        if len(factors) < 3:
            raise ValueError("Fractional factorial requires at least 3 factors")
        
        for factor in factors:
            if not factor.is_continuous() and not factor.is_discrete_numeric():
                raise ValueError(
                    f"Factor '{factor.name}' must be continuous or discrete numeric "
                    f"for fractional factorial. Categorical factors not supported."
                )
            
            if factor.is_discrete_numeric() and len(factor.levels) != 2:
                raise ValueError(
                    f"Factor '{factor.name}' must have exactly 2 levels for "
                    f"fractional factorial, got {len(factor.levels)}"
                )
    
    def _parse_fraction(self, fraction: str) -> int:
        """Parse fraction string to get p value."""
        if fraction.startswith("1/"):
            denominator = int(fraction[2:])
            # Check if power of 2
            if denominator & (denominator - 1) != 0:
                raise ValueError(f"Fraction denominator must be power of 2, got {denominator}")
            p = int(np.log2(denominator))
            return p
        else:
            raise ValueError(f"Fraction must be in format '1/2', '1/4', etc., got '{fraction}'")
    
    def _get_max_resolution(self) -> int:
        """Determine maximum achievable resolution for given k and p."""
        # Resolution definitions:
        # III: Main effects clear, main effects aliased with 2FI
        # IV: Main effects clear, 2FI aliased with other 2FI
        # V: Main effects and 2FI clear
        
        n_runs = 2 ** (self.k - self.p)
        
        # Rough heuristic for max resolution
        if self.p == 1:
            if self.k <= 5:
                return 5
            elif self.k <= 7:
                return 4
            else:
                return 3
        elif self.p == 2:
            if self.k <= 7:
                return 4
            else:
                return 3
        else:
            return 3
    
    def _select_generators(self, resolution: int) -> List[Tuple[str, str]]:
        """
        Select generators to achieve desired resolution.
        
        Returns list of (factor, generator_expression) tuples.
        """
        # Standard generators for common designs
        # Format: (k, p, resolution): [(factor, generator), ...]
        
        standard_generators = {
            # 2^(4-1) designs
            (4, 1, 4): [("D", "ABC")],
            
            # 2^(5-1) designs  
            (5, 1, 5): [("E", "ABCD")],
            
            # 2^(5-2) designs
            (5, 2, 3): [("D", "AB"), ("E", "AC")],
            
            # 2^(6-1) designs
            (6, 1, 6): [("F", "ABCDE")],
            
            # 2^(6-2) designs
            (6, 2, 4): [("E", "ABC"), ("F", "BCD")],
            
            # 2^(7-1) designs
            (7, 1, 7): [("G", "ABCDEF")],
            
            # 2^(7-2) designs
            (7, 2, 4): [("F", "ABCD"), ("G", "ABCE")],
            
            # 2^(7-3) designs
            (7, 3, 4): [("E", "ABC"), ("F", "BCD"), ("G", "ACD")],
            
            # 2^(8-2) designs
            (8, 2, 5): [("G", "ABCD"), ("H", "ABEF")],
            
            # 2^(8-4) designs
            (8, 4, 4): [("E", "BCD"), ("F", "ACD"), ("G", "ABC"), ("H", "ABD")],
        }
        
        key = (self.k, self.p, resolution)
        
        if key in standard_generators:
            return standard_generators[key]
        else:
            # Fall back to simple sequential generators
            # This may not achieve the desired resolution
            base_factors = [chr(65 + i) for i in range(self.k - self.p)]
            generators = []
            
            for i in range(self.p):
                generated_factor = chr(65 + self.k - self.p + i)
                # Use simplest generator: product of all base factors
                generator_expr = "".join(base_factors[:i+2])
                generators.append((generated_factor, generator_expr))
            
            return generators
    
    def _parse_generators(self, generator_strings: List[str]) -> List[Tuple[str, str]]:
        """Parse generator strings like ["D=ABC", "E=BCD"] into tuples."""
        parsed = []
        for gen_str in generator_strings:
            if "=" not in gen_str:
                raise ValueError(f"Generator must contain '=', got '{gen_str}'")
            
            parts = gen_str.split("=")
            if len(parts) != 2:
                raise ValueError(f"Invalid generator format: '{gen_str}'")
            
            factor = parts[0].strip()
            expression = parts[1].strip()
            
            parsed.append((factor, expression))
        
        if len(parsed) != self.p:
            raise ValueError(
                f"Number of generators ({len(parsed)}) must equal p ({self.p})"
            )
        
        return parsed
    
    def _calculate_defining_relation(self) -> List[str]:
        """
        Calculate the complete defining relation.
        
        The defining relation includes the generators and all their products.
        """
        # Start with identity
        words = ["I"]
        
        # Add generators
        for factor, expression in self.generators:
            words.append(factor + expression)
        
        # Generate all products of generators
        n_generators = len(self.generators)
        for r in range(2, n_generators + 1):
            for combo in itertools.combinations(range(n_generators), r):
                # Multiply the generators
                word = ""
                for idx in combo:
                    factor, expression = self.generators[idx]
                    word += factor + expression
                
                # Simplify (remove pairs of same letter)
                word = self._simplify_word(word)
                if word and word not in words:
                    words.append(word)
        
        return words
    
    def _simplify_word(self, word: str) -> str:
        """
        Simplify a generator word by removing pairs of identical letters.
        
        In mod-2 algebra: A*A = I, A*B*A = B, etc.
        """
        # Count occurrences of each letter
        counts = {}
        for letter in word:
            counts[letter] = counts.get(letter, 0) + 1
        
        # Keep only letters with odd counts
        result = ""
        for letter in sorted(counts.keys()):
            if counts[letter] % 2 == 1:
                result += letter
        
        return result
    
    def _calculate_resolution(self) -> int:
        """
        Calculate the resolution of the design based on generators.
        
        Resolution = minimum word length in defining relation (excluding I)
        """
        min_length = float('inf')
        
        for word in self.defining_relation:
            if word != "I":
                min_length = min(min_length, len(word))
        
        return int(min_length) if min_length != float('inf') else 0
    
    def _calculate_alias_structure(self) -> Dict[str, List[str]]:
        """
        Calculate the alias structure for all effects.
        
        Each effect is aliased with its product with each word in the
        defining relation.
        """
        alias_structure = {}
        
        # Factor names
        factor_names = [chr(65 + i) for i in range(self.k)]
        
        # All effects to consider (main effects and interactions)
        # For practical purposes, consider up to 4-factor interactions
        all_effects = self._generate_effects(factor_names, max_order=4)
        
        for effect in all_effects:
            aliases = set()
            
            # Multiply effect by each word in defining relation
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
    
    def generate(
        self,
        randomize: bool = True,
        random_seed: Optional[int] = None,
        n_blocks: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate the fractional factorial design.
        
        Parameters
        ----------
        randomize : bool
            Whether to randomize run order
        random_seed : int, optional
            Random seed for reproducibility
        n_blocks : int, optional
            Number of blocks
        
        Returns
        -------
        pd.DataFrame
            Design matrix
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Number of base factors (not generated)
        n_base = self.k - self.p
        base_factors = self.factors[:n_base]
        
        # Generate full factorial for base factors
        base_design = full_factorial(base_factors, randomize=False)
        
        # Remove StdOrder and RunOrder columns
        base_design = base_design[[f.name for f in base_factors]]
        
        # Generate additional factors using generators
        for factor, expression in self.generators:
            # Get the generated factor
            gen_factor = self.factors[ord(factor) - 65]
            
            # Calculate values by multiplying base factors
            values = np.ones(len(base_design))
            for letter in expression:
                col_name = self.factors[ord(letter) - 65].name
                values *= base_design[col_name].values
            
            base_design[gen_factor.name] = values
        
        # Reorder columns to match original factor order
        design = base_design[[f.name for f in self.factors]]
        
        # Add standard order
        design.insert(0, 'StdOrder', range(1, len(design) + 1))
        
        # Randomize if requested
        if randomize:
            design = design.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        
        # Add run order
        design.insert(1, 'RunOrder', range(1, len(design) + 1))
        
        # Add blocking if requested
        if n_blocks is not None:
            design = self._assign_blocks(design, n_blocks)
        
        return design
    
    def _assign_blocks(self, design: pd.DataFrame, n_blocks: int) -> pd.DataFrame:
        """Assign runs to blocks."""
        n_runs = len(design)
        
        if n_blocks > n_runs:
            raise ValueError(f"Cannot have more blocks ({n_blocks}) than runs ({n_runs})")
        
        # Simple sequential assignment
        runs_per_block = n_runs // n_blocks
        extra_runs = n_runs % n_blocks
        
        blocks = []
        for i in range(n_blocks):
            block_size = runs_per_block + (1 if i < extra_runs else 0)
            blocks.extend([i + 1] * block_size)
        
        design.insert(2, 'Block', blocks)
        return design
    
    def get_alias_summary(self) -> pd.DataFrame:
        """
        Get a summary of the alias structure.
        
        Returns
        -------
        pd.DataFrame
            Table showing each effect and what it's aliased with
        """
        rows = []
        
        for effect, aliases in sorted(self.alias_structure.items(), 
                                      key=lambda x: (len(x[0]), x[0])):
            rows.append({
                'Effect': effect,
                'Aliased_With': ' + '.join(aliases) if aliases else 'Clear'
            })
        
        return pd.DataFrame(rows)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FractionalFactorial(k={self.k}, p={self.p}, "
            f"resolution={self.resolution}, n_runs={2**(self.k-self.p)})"
        )