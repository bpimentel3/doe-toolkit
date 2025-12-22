"""
Response surface design generation for DOE-Toolkit.

This module implements Box-Behnken and Central Composite Designs (CCD) for
modeling curvature and finding optimal factor settings.
"""

from typing import List, Dict, Optional, Literal
import pandas as pd
import numpy as np
import itertools

from src.core.factors import Factor, FactorType


class ResponseSurfaceDesign:
    """
    Base class for response surface designs.
    
    Response surface designs are used when:
    - Curvature (quadratic effects) is expected
    - Need to find optimal factor settings
    - Building predictive models
    
    Common designs:
    - Central Composite Design (CCD)
    - Box-Behnken Design (BBD)
    """
    
    def __init__(self, factors: List[Factor]):
        """
        Initialize response surface design.
        
        Parameters
        ----------
        factors : List[Factor]
            List of factors (must be continuous)
        """
        self._validate_factors(factors)
        self.factors = factors
        self.k = len(factors)
    
    def _validate_factors(self, factors: List[Factor]) -> None:
        """Validate that factors are suitable for response surface design."""
        if len(factors) < 2:
            raise ValueError("Response surface design requires at least 2 factors")
        
        if len(factors) > 10:
            raise ValueError(
                f"Response surface design with {len(factors)} factors is impractical. "
                f"Consider screening or sequential approaches."
            )
        
        for factor in factors:
            if not factor.is_continuous():
                raise ValueError(
                    f"Factor '{factor.name}' must be continuous for response surface design. "
                    f"Use full factorial for categorical/discrete factors."
                )


class CentralCompositeDesign(ResponseSurfaceDesign):
    """
    Central Composite Design (CCD).
    
    A CCD consists of:
    1. Factorial points (2^k or 2^(k-p) fractional factorial)
    2. Axial (star) points (2k points along each axis)
    3. Center points (multiple replicates at center)
    
    The design allows estimation of:
    - All main effects
    - All two-factor interactions
    - All quadratic effects
    
    Parameters
    ----------
    factors : List[Factor]
        List of continuous factors
    alpha : float or str, optional
        Distance of axial points from center. Options:
        - "orthogonal": Alpha chosen for orthogonality
        - "rotatable": Alpha = (2^k)^(1/4) for rotatability
        - "face": Alpha = 1 (axial points on cube faces)
        - float: Custom alpha value
    center_points : int, optional
        Number of center point replicates (default: varies by k)
    fraction : str, optional
        Fraction for factorial portion, e.g., "1/2" (default: full factorial)
    
    Attributes
    ----------
    design_type : str
        Type of CCD (orthogonal, rotatable, face-centered)
    alpha : float
        Actual alpha value used
    n_factorial : int
        Number of factorial points
    n_axial : int
        Number of axial points
    n_center : int
        Number of center points
    n_total : int
        Total number of runs
    
    Examples
    --------
    >>> # Create rotatable CCD for 3 factors
    >>> factors = [Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[10, 20]),
    ...            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[50, 100]),
    ...            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[1, 5])]
    >>> ccd = CentralCompositeDesign(factors, alpha="rotatable", center_points=6)
    >>> design = ccd.generate()
    
    References
    ----------
    .. [1] Box, G. E. P., and Wilson, K. B. (1951). On the Experimental Attainment
           of Optimum Conditions. Journal of the Royal Statistical Society, Series B, 13, 1-45.
    .. [2] Myers, R. H., Montgomery, D. C., and Anderson-Cook, C. M. (2016).
           Response Surface Methodology, 4th Ed. Wiley.
    """
    
    def __init__(
        self,
        factors: List[Factor],
        alpha: Literal["orthogonal", "rotatable", "face"] | float = "rotatable",
        center_points: Optional[int] = None,
        fraction: Optional[str] = None
    ):
        super().__init__(factors)
        
        self.fraction = fraction
        
        # Determine number of factorial points
        if fraction is None:
            self.n_factorial = 2 ** self.k
        else:
            # Parse fraction
            p = self._parse_fraction(fraction)
            self.n_factorial = 2 ** (self.k - p)
        
        self.n_axial = 2 * self.k
        
        # Determine center points
        if center_points is None:
            # Default center points for rotatability
            self.n_center = self._default_center_points()
        else:
            self.n_center = center_points
        
        # Calculate alpha
        if isinstance(alpha, str):
            self.alpha = self._calculate_alpha(alpha)
            self.design_type = alpha
        else:
            self.alpha = float(alpha)
            self.design_type = "custom"
        
        self.n_total = self.n_factorial + self.n_axial + self.n_center
    
    def _parse_fraction(self, fraction: str) -> int:
        """Parse fraction string."""
        if fraction.startswith("1/"):
            denominator = int(fraction[2:])
            if denominator & (denominator - 1) != 0:
                raise ValueError(f"Fraction denominator must be power of 2")
            return int(np.log2(denominator))
        else:
            raise ValueError(f"Invalid fraction format: {fraction}")
    
    def _default_center_points(self) -> int:
        """Determine default number of center points."""
        # Standard recommendations for rotatability
        if self.k == 2:
            return 5
        elif self.k == 3:
            return 6
        elif self.k == 4:
            return 7
        elif self.k <= 6:
            return 6
        else:
            return 6
    
    def _calculate_alpha(self, alpha_type: str) -> float:
        """
        Calculate alpha based on design type.
        
        Parameters
        ----------
        alpha_type : str
            Type of design: "rotatable", "orthogonal", or "face"
        
        Returns
        -------
        float
            Alpha value
        """
        if alpha_type == "rotatable":
            # Rotatable: alpha = (n_factorial)^(1/4)
            return self.n_factorial ** 0.25
        
        elif alpha_type == "orthogonal":
            # Orthogonal: more complex calculation
            # Simplified formula for full factorial
            nc = self.n_center
            nf = self.n_factorial
            na = self.n_axial
            
            # Alpha for orthogonality (approximate)
            alpha_squared = (
                np.sqrt(nf * (np.sqrt(nf + nc) + np.sqrt(na + nc)) / (2 * self.k))
            )
            return alpha_squared
        
        elif alpha_type == "face":
            # Face-centered: alpha = 1 (axial points on cube faces)
            return 1.0
        
        else:
            raise ValueError(f"Unknown alpha type: {alpha_type}")
    
    def generate(
        self,
        randomize: bool = True,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate the Central Composite Design.
        
        Parameters
        ----------
        randomize : bool
            Whether to randomize run order
        random_seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        pd.DataFrame
            Design matrix with coded levels
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Collect non-empty DataFrames to concatenate
        dfs_to_concat = []
        point_types = []
        
        # 1. Factorial points (2^k or fractional)
        factorial_points = self._generate_factorial_points()
        if len(factorial_points) > 0:
            dfs_to_concat.append(factorial_points)
            point_types.extend(['Factorial'] * len(factorial_points))
        
        # 2. Axial (star) points
        axial_points = self._generate_axial_points()
        if len(axial_points) > 0:
            dfs_to_concat.append(axial_points)
            point_types.extend(['Axial'] * len(axial_points))
        
        # 3. Center points
        if self.n_center > 0:
            center_points = self._generate_center_points()
            if len(center_points) > 0:
                dfs_to_concat.append(center_points)
                point_types.extend(['Center'] * len(center_points))
        
        # Combine all points (only if we have DataFrames to concat)
        if len(dfs_to_concat) == 0:
            raise ValueError("No design points generated")
        
        design_matrix = pd.concat(dfs_to_concat, ignore_index=True)
        
        # Add point type identifier
        design_matrix.insert(0, 'PointType', point_types)
        
        # Add standard order
        design_matrix.insert(0, 'StdOrder', range(1, len(design_matrix) + 1))
        
        # Randomize if requested
        if randomize:
            design_matrix = design_matrix.sample(
                frac=1, random_state=random_seed
            ).reset_index(drop=True)
        
        # Add run order
        design_matrix.insert(1, 'RunOrder', range(1, len(design_matrix) + 1))
        
        return design_matrix
    
    def _generate_factorial_points(self) -> pd.DataFrame:
        """Generate factorial portion of design."""
        factor_names = [f.name for f in self.factors]
        
        if self.fraction is None:
            # Full factorial: all combinations of -1 and +1
            levels = [[-1, 1] for _ in range(self.k)]
            combinations = list(itertools.product(*levels))
        else:
            # Fractional factorial
            # For simplicity, use a standard fraction
            from src.core.fractional_factorial import FractionalFactorial
            
            # Create temporary factors for fractional factorial
            temp_factors = [
                Factor(f"X{i+1}", FactorType.CONTINUOUS, 
                      self.factors[0].changeability, levels=[-1, 1])
                for i in range(self.k)
            ]
            
            ff = FractionalFactorial(temp_factors, self.fraction)
            ff_design = ff.generate(randomize=False)
            
            # Extract just the factor columns
            temp_names = [f"X{i+1}" for i in range(self.k)]
            combinations = ff_design[temp_names].values.tolist()
        
        # Create DataFrame
        factorial_df = pd.DataFrame(combinations, columns=factor_names)
        
        return factorial_df
    
    def _generate_axial_points(self) -> pd.DataFrame:
        """Generate axial (star) points."""
        factor_names = [f.name for f in self.factors]
        axial_runs = []
        
        # For each factor, create two axial points: +alpha and -alpha
        for i in range(self.k):
            # Positive axial point
            point_pos = [0] * self.k
            point_pos[i] = self.alpha
            axial_runs.append(point_pos)
            
            # Negative axial point
            point_neg = [0] * self.k
            point_neg[i] = -self.alpha
            axial_runs.append(point_neg)
        
        axial_df = pd.DataFrame(axial_runs, columns=factor_names)
        
        return axial_df
    
    def _generate_center_points(self) -> pd.DataFrame:
        """Generate center point replicates."""
        factor_names = [f.name for f in self.factors]
        
        # Center point: all factors at 0
        center_point = [0] * self.k
        center_runs = [center_point] * self.n_center
        
        center_df = pd.DataFrame(center_runs, columns=factor_names)
        
        return center_df
    
    def get_design_properties(self) -> Dict:
        """
        Get properties of the CCD.
        
        Returns
        -------
        dict
            Design properties including alpha, rotatability, etc.
        """
        return {
            'design_type': 'Central Composite Design',
            'variant': self.design_type,
            'n_factors': self.k,
            'alpha': self.alpha,
            'n_factorial_points': self.n_factorial,
            'n_axial_points': self.n_axial,
            'n_center_points': self.n_center,
            'n_total_runs': self.n_total,
            'fraction': self.fraction if self.fraction else 'full',
            'rotatable': abs(self.alpha - self.n_factorial**0.25) < 0.01,
            'face_centered': abs(self.alpha - 1.0) < 0.01
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CentralCompositeDesign(k={self.k}, alpha={self.alpha:.3f}, "
            f"type={self.design_type}, runs={self.n_total})"
        )


class BoxBehnkenDesign(ResponseSurfaceDesign):
    """
    Box-Behnken Design (BBD).
    
    A Box-Behnken design is a spherical design that:
    - Uses 3-level factorial structure
    - Requires 3 or more factors
    - Has no points at the corners of the design space
    - More economical than CCD for 3-4 factors
    
    The design places points at:
    - Midpoints of edges of the design cube
    - Center of the design space
    
    Parameters
    ----------
    factors : List[Factor]
        List of continuous factors (minimum 3)
    center_points : int, optional
        Number of center point replicates (default: 3-5)
    
    Attributes
    ----------
    n_factorial : int
        Number of factorial points
    n_center : int
        Number of center points
    n_total : int
        Total number of runs
    
    Examples
    --------
    >>> # Create Box-Behnken design for 3 factors
    >>> factors = [Factor("A", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[10, 20]),
    ...            Factor("B", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[50, 100]),
    ...            Factor("C", FactorType.CONTINUOUS, ChangeabilityLevel.EASY, levels=[1, 5])]
    >>> bbd = BoxBehnkenDesign(factors, center_points=3)
    >>> design = bbd.generate()
    
    References
    ----------
    .. [1] Box, G. E. P., and Behnken, D. W. (1960). Some New Three Level Designs
           for the Study of Quantitative Variables. Technometrics, 2, 455-475.
    """
    
    def __init__(
        self,
        factors: List[Factor],
        center_points: int = 3
    ):
        super().__init__(factors)
        
        if self.k < 3:
            raise ValueError("Box-Behnken design requires at least 3 factors")
        
        if self.k > 7:
            raise ValueError(
                f"Box-Behnken design with {self.k} factors requires "
                f"{self._calculate_runs()} runs - consider CCD instead"
            )
        
        self.n_center = center_points
        self.n_factorial = self._calculate_runs()
        self.n_total = self.n_factorial + self.n_center
    
    def _calculate_runs(self) -> int:
        """
        Calculate number of factorial points in Box-Behnken design.
        
        Formula: 2k(k-1) for k factors
        """
        return 2 * self.k * (self.k - 1)
    
    def generate(
        self,
        randomize: bool = True,
        random_seed: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Generate the Box-Behnken Design.
        
        Parameters
        ----------
        randomize : bool
            Whether to randomize run order
        random_seed : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        pd.DataFrame
            Design matrix with coded levels (-1, 0, +1)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Collect non-empty DataFrames to concatenate
        dfs_to_concat = []
        point_types = []
        
        # Generate factorial points
        factorial_points = self._generate_factorial_points()
        if len(factorial_points) > 0:
            dfs_to_concat.append(factorial_points)
            point_types.extend(['Factorial'] * len(factorial_points))
        
        # Generate center points
        if self.n_center > 0:
            center_points = self._generate_center_points()
            if len(center_points) > 0:
                dfs_to_concat.append(center_points)
                point_types.extend(['Center'] * len(center_points))
        
        # Combine (only if we have DataFrames)
        if len(dfs_to_concat) == 0:
            raise ValueError("No design points generated")
        
        design_matrix = pd.concat(dfs_to_concat, ignore_index=True)
        
        # Add point type
        design_matrix.insert(0, 'PointType', point_types)
        
        # Add standard order
        design_matrix.insert(0, 'StdOrder', range(1, len(design_matrix) + 1))
        
        # Randomize if requested
        if randomize:
            design_matrix = design_matrix.sample(
                frac=1, random_state=random_seed
            ).reset_index(drop=True)
        
        # Add run order
        design_matrix.insert(1, 'RunOrder', range(1, len(design_matrix) + 1))
        
        return design_matrix
    
    def _generate_factorial_points(self) -> pd.DataFrame:
        """
        Generate Box-Behnken factorial points.
        
        Points are at the midpoint of edges of the design cube.
        For each pair of factors, create 2^2 = 4 combinations at ±1,
        with other factors at 0.
        """
        factor_names = [f.name for f in self.factors]
        runs = []
        
        # For each pair of factors
        for i in range(self.k):
            for j in range(i + 1, self.k):
                # Create 2^2 factorial for factors i and j
                for level_i in [-1, 1]:
                    for level_j in [-1, 1]:
                        run = [0] * self.k
                        run[i] = level_i
                        run[j] = level_j
                        runs.append(run)
        
        factorial_df = pd.DataFrame(runs, columns=factor_names)
        
        return factorial_df
    
    def _generate_center_points(self) -> pd.DataFrame:
        """Generate center point replicates."""
        factor_names = [f.name for f in self.factors]
        
        # Center point: all factors at 0
        center_point = [0] * self.k
        center_runs = [center_point] * self.n_center
        
        center_df = pd.DataFrame(center_runs, columns=factor_names)
        
        return center_df
    
    def get_design_properties(self) -> Dict:
        """
        Get properties of the Box-Behnken design.
        
        Returns
        -------
        dict
            Design properties
        """
        return {
            'design_type': 'Box-Behnken Design',
            'n_factors': self.k,
            'n_factorial_points': self.n_factorial,
            'n_center_points': self.n_center,
            'n_total_runs': self.n_total,
            'spherical': True,
            'levels': 3,
            'no_extreme_points': True
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return f"BoxBehnkenDesign(k={self.k}, runs={self.n_total})"