"""
Simplified Validation System
===========================

Minimal validation framework for the trading system.
Follows KISS principle - only what's actually needed.
"""

from .base import BaseValidator, ValidationResult, ValidationIssue, ValidationSeverity
from .config.portfolio_validator import PortfolioConfigValidator
from .config.schema_validator import SchemaValidator
from .result.experiment_result_validator import ExperimentResultValidator
from .exceptions import (
    ValidationException,
    ConfigValidationException,
    ResultValidationException
)

__all__ = [
    # Base
    'BaseValidator',
    'ValidationResult',
    'ValidationIssue',
    'ValidationSeverity',
    
    # Config validators
    'PortfolioConfigValidator',
    'SchemaValidator',
    
    # Result validators
    'ExperimentResultValidator',
    
    # Exceptions
    'ValidationException',
    'ConfigValidationException',
    'ResultValidationException',
]
