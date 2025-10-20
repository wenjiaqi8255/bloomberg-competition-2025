"""
Base Validation Framework
========================

Provides the abstract base class and common utilities for all validators.
Follows SOLID principles with clear separation of concerns.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Validation severity levels."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    message: str
    severity: ValidationSeverity
    field: Optional[str] = None
    suggestion: Optional[str] = None
    
    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.field or 'GLOBAL'}: {self.message}"


@dataclass
class ValidationResult:
    """Result of a validation operation."""
    is_valid: bool
    issues: List[ValidationIssue]
    warnings: List[str]
    suggestions: List[str]
    
    def __init__(self, is_valid: bool = True, issues: Optional[List[ValidationIssue]] = None):
        self.is_valid = is_valid
        self.issues = issues or []
        self.warnings = [issue.message for issue in self.issues if issue.severity == ValidationSeverity.WARNING]
        self.suggestions = [issue.suggestion for issue in self.issues if issue.suggestion is not None]
    
    def add_error(self, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        """Add an error issue."""
        self.issues.append(ValidationIssue(message, ValidationSeverity.ERROR, field, suggestion))
        self.is_valid = False
    
    def add_warning(self, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        """Add a warning issue."""
        self.issues.append(ValidationIssue(message, ValidationSeverity.WARNING, field, suggestion))
    
    def add_info(self, message: str, field: Optional[str] = None, suggestion: Optional[str] = None):
        """Add an info issue."""
        self.issues.append(ValidationIssue(message, ValidationSeverity.INFO, field, suggestion))
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return any(issue.severity == ValidationSeverity.ERROR for issue in self.issues)
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)
    
    def get_errors(self) -> List[ValidationIssue]:
        """Get all error issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.ERROR]
    
    def get_warnings(self) -> List[ValidationIssue]:
        """Get all warning issues."""
        return [issue for issue in self.issues if issue.severity == ValidationSeverity.WARNING]
    
    def get_summary(self) -> str:
        """Get a summary of validation results."""
        error_count = len(self.get_errors())
        warning_count = len(self.get_warnings())
        
        if error_count == 0 and warning_count == 0:
            return "✅ Validation passed with no issues"
        elif error_count == 0:
            return f"⚠️  Validation passed with {warning_count} warning(s)"
        else:
            return f"❌ Validation failed with {error_count} error(s) and {warning_count} warning(s)"


class BaseValidator(ABC):
    """
    Abstract base class for all validators.
    
    Provides a consistent interface and common validation utilities.
    All validators should inherit from this class.
    """
    
    def __init__(self, name: str):
        """
        Initialize the validator.
        
        Args:
            name: Human-readable name for this validator
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    def validate(self, data: Any) -> ValidationResult:
        """
        Validate the provided data.
        
        Args:
            data: Data to validate (type depends on validator implementation)
            
        Returns:
            ValidationResult with validation outcome and issues
        """
        pass
    
    def validate_required_field(self, data: Dict[str, Any], field: str, 
                              field_type: type = None) -> bool:
        """
        Validate that a required field exists and optionally check its type.
        
        Args:
            data: Dictionary to validate
            field: Field name to check
            field_type: Expected type (optional)
            
        Returns:
            True if field exists and type matches (if specified)
        """
        if field not in data:
            self.logger.error(f"Missing required field: {field}")
            return False
        
        if field_type is not None and not isinstance(data[field], field_type):
            self.logger.error(f"Field '{field}' must be of type {field_type.__name__}, got {type(data[field]).__name__}")
            return False
        
        return True
    
    def validate_optional_field(self, data: Dict[str, Any], field: str, 
                              field_type: type = None, default_value: Any = None) -> Any:
        """
        Validate an optional field and return its value or default.
        
        Args:
            data: Dictionary to validate
            field: Field name to check
            field_type: Expected type (optional)
            default_value: Default value if field is missing
            
        Returns:
            Field value or default_value
        """
        if field not in data:
            return default_value
        
        if field_type is not None and not isinstance(data[field], field_type):
            self.logger.warning(f"Field '{field}' should be of type {field_type.__name__}, got {type(data[field]).__name__}")
            return default_value
        
        return data[field]
    
    def validate_enum_field(self, data: Dict[str, Any], field: str, 
                          allowed_values: List[str], required: bool = True) -> bool:
        """
        Validate that a field has one of the allowed values.
        
        Args:
            data: Dictionary to validate
            field: Field name to check
            allowed_values: List of allowed values
            required: Whether the field is required
            
        Returns:
            True if field is valid or not required and missing
        """
        if field not in data:
            if required:
                self.logger.error(f"Missing required field: {field}")
                return False
            return True
        
        if data[field] not in allowed_values:
            self.logger.error(f"Field '{field}' must be one of {allowed_values}, got '{data[field]}'")
            return False
        
        return True
    
    def validate_numeric_range(self, data: Dict[str, Any], field: str,
                             min_value: float = None, max_value: float = None,
                             required: bool = True) -> bool:
        """
        Validate that a numeric field is within the specified range.
        
        Args:
            data: Dictionary to validate
            field: Field name to check
            min_value: Minimum allowed value
            max_value: Maximum allowed value
            required: Whether the field is required
            
        Returns:
            True if field is valid or not required and missing
        """
        if field not in data:
            if required:
                self.logger.error(f"Missing required field: {field}")
                return False
            return True
        
        value = data[field]
        if not isinstance(value, (int, float)):
            self.logger.error(f"Field '{field}' must be numeric, got {type(value).__name__}")
            return False
        
        if min_value is not None and value < min_value:
            self.logger.error(f"Field '{field}' must be >= {min_value}, got {value}")
            return False
        
        if max_value is not None and value > max_value:
            self.logger.error(f"Field '{field}' must be <= {max_value}, got {value}")
            return False
        
        return True
    
    def log_validation_start(self, data_type: str):
        """Log the start of validation."""
        self.logger.debug(f"Starting {self.name} validation for {data_type}")
    
    def log_validation_complete(self, result: ValidationResult):
        """Log the completion of validation."""
        if result.is_valid:
            self.logger.info(f"✅ {self.name} validation completed successfully")
        else:
            error_count = len(result.get_errors())
            warning_count = len(result.get_warnings())
            self.logger.warning(f"⚠️  {self.name} validation completed with {error_count} errors, {warning_count} warnings")
