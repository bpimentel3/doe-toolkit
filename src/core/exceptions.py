"""Custom exceptions for DOE-Toolkit."""

class DOEError(Exception):
    """Base exception for all DOE-Toolkit errors."""
    pass

class DesignError(DOEError):
    """Errors related to design generation."""
    pass

class ValidationError(DOEError):
    """Errors from design or input validation."""
    pass

class AnalysisError(DOEError):
    """Errors from statistical analysis."""
    pass

class OptimizationError(DOEError):
    """Errors from optimization routines."""
    pass

class AugmentationError(DOEError):
    """Errors from design augmentation."""
    pass