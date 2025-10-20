"""
Validation Exception Classes
===========================

Custom exceptions for validation failures with enhanced error reporting.
"""

from typing import Optional
from .base import ValidationResult


class ValidationException(Exception):
    """Base exception for validation failures."""
    
    def __init__(self, message: str, validation_result: Optional[ValidationResult] = None):
        super().__init__(message)
        self.validation_result = validation_result


class ConfigValidationException(ValidationException):
    """Exception raised when configuration validation fails."""
    pass


class ResultValidationException(ValidationException):
    """Exception raised when result validation fails."""
    pass

