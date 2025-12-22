"""
Fractional factorial design generation for DOE-Toolkit.

This module implements 2^(k-p) fractional factorial designs with automatic
generator selection and alias structure calculation.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from src.core.factors import Factor, FactorType
from src.core.aliasing import (
    FactorMapper,
    GeneratorValidator,
    AliasingEngine,
    parse_generators,
    get_standard_generators,
    format_alias_table,
    validate_resolution_achievable
)
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
    mapper : FactorMapper
        Maps between real factor names and algebraic symbols
    
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
                f"Cannot create 1/{2**self.p} fraction of {self.k} factors"
            )
        
        # Create factor name mapper
        self.mapper = FactorMapper(factors)
        
        # Validate generator count
        validator = GeneratorValidator(self.k, self.p, self.mapper)
        validator.validate_generator_count()
        
        # Determine generators (in algebraic form)
        if generators is not None:
            # Custom generators provided
            validator.validate_all(generators)
            parsed_generators = parse_generators(generators)
            self.resolution = self._get_resolution_from_generators(parsed_generators)
            
            # Validate resolution if specified
            if resolution is not None and self.resolution < resolution:
                raise ValueError(
                    f"Generators achieve Resolution {self.resolution}, "
                    f"not {resolution} as specified"
                )
        elif resolution is not None:
            # Auto-select generators for target resolution
            self.resolution = resolution
            parsed_generators = self._select_generators(resolution)
            
            if parsed_generators is None:
                raise ValueError(
                    f"No standard generators available for k={self.k}, "
                    f"p={self.p}, resolution={resolution}"
                )
        else:
            # Default: highest resolution possible
            self.resolution = self._get_max_resolution()
            parsed_generators = self._select_generators(self.resolution)
            
            if parsed_generators is None:
                raise ValueError(
                    f"No standard generators available for k={self.k}, p={self.p}"
                )
        
        # Store generators and build aliasing structure
        self.generators_algebraic = parsed_generators
        self.engine = AliasingEngine(self.k, parsed_generators)
        self.defining_relation = self.engine.defining_relation
        self.alias_structure = self.engine.alias_structure
    
    def _validate_factors(self, factors: List[Factor]) -> None:
        """Validate that all factors are suitable for fractional factorial."""
        if len(factors) < 3:
            raise ValueError("Fractional factorial requires at least 3 factors")
        
        for factor in factors:
            if not factor.is_continuous() and not factor.is_discrete_numeric():
                raise ValueError(
                    f"Factor '{factor.name}' must be continuous or discrete numeric. "
                    f"Categorical factors not supported."
                )
            
            if factor.is_discrete_numeric() and len(factor.levels) != 2:
                raise ValueError(
                    f"Factor '{factor.name}' must have exactly 2 levels, "
                    f"got {len(factor.levels)}"
                )
    
    def _parse_fraction(self, fraction: str) -> int:
        """Parse fraction string to get p value."""
        if fraction.startswith("1/"):
            denominator = int(fraction[2:])
            # Check if power of 2
            if denominator & (denominator - 1) != 0:
                raise ValueError(f"Fraction must be power of 2, got {denominator}")
            p = int(np.log2(denominator))
            return p
        else:
            raise ValueError(f"Fraction must be in format '1/2', '1/4', etc.")
    
    def _get_max_resolution(self) -> int:
        """Determine maximum achievable resolution for given k and p."""
        # Try resolutions from V down to III
        for res in [5, 4, 3]:
            if get_standard_generators(self.k, self.p, res) is not None:
                return res
        
        # Fallback
        return 3
    
    def _select_generators(self, resolution: int) -> Optional[List[Tuple[str, str]]]:
        """Select generators to achieve desired resolution."""
        return get_standard_generators(self.k, self.p, resolution)
    
    def _get_resolution_from_generators(self, generators: List[Tuple[str, str]]) -> int:
        """Calculate resolution from generators."""
        temp_engine = AliasingEngine(self.k, generators)
        return temp_engine.resolution
    
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
            Design matrix with factor columns (using real names)
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
        for factor_symbol, expression in self.generators_algebraic:
            # Get the factor object
            factor_idx = ord(factor_symbol) - 65
            gen_factor = self.factors[factor_idx]
            
            # Calculate values by multiplying base factors
            values = np.ones(len(base_design))
            for symbol in expression:
                # Map symbol to real factor name
                real_name = self.mapper.to_real(symbol)
                values *= base_design[real_name].values
            
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
        return format_alias_table(self.alias_structure, self.mapper)
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"FractionalFactorial(k={self.k}, p={self.p}, "
            f"resolution={self.resolution}, n_runs={2**(self.k-self.p)})"
        )
