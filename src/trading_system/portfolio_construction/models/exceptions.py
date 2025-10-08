"""
Exception classes for portfolio construction.

Custom exceptions provide better error handling and debugging capabilities
for portfolio construction processes.
"""


class PortfolioConstructionError(Exception):
    """Base exception for portfolio construction errors."""

    def __init__(self, message: str, details: dict = None):
        """
        Initialize portfolio construction error.

        Args:
            message: Error message
            details: Optional additional details dictionary
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        """String representation with details."""
        if self.details:
            details_str = ", ".join(f"{k}: {v}" for k, v in self.details.items())
            return f"{self.message} (Details: {details_str})"
        return self.message


class InsufficientStocksError(PortfolioConstructionError):
    """Raised when a box has insufficient stocks for construction."""

    def __init__(self, box_key: str, available: int, required: int):
        """
        Initialize insufficient stocks error.

        Args:
            box_key: Identifier of the box with insufficient stocks
            available: Number of available stocks
            required: Minimum number of required stocks
        """
        message = f"Box {box_key} has insufficient stocks: {available} < {required}"
        super().__init__(message, {
            'box_key': box_key,
            'available_stocks': available,
            'required_stocks': required
        })


class InvalidConfigError(PortfolioConstructionError):
    """Raised when portfolio construction configuration is invalid."""

    def __init__(self, message: str, config_section: str = None, validation_errors: list = None):
        """
        Initialize invalid config error.

        Args:
            message: Error message
            config_section: Configuration section that caused the error
            validation_errors: List of specific validation errors
        """
        super().__init__(message, {
            'config_section': config_section,
            'validation_errors': validation_errors or []
        })


class ClassificationError(PortfolioConstructionError):
    """Raised when stock classification fails."""

    def __init__(self, symbol: str, error_type: str = None):
        """
        Initialize classification error.

        Args:
            symbol: Stock symbol that failed classification
            error_type: Type of classification error
        """
        message = f"Failed to classify stock {symbol}"
        if error_type:
            message += f" ({error_type})"
        super().__init__(message, {
            'symbol': symbol,
            'error_type': error_type
        })


class OptimizationError(PortfolioConstructionError):
    """Raised when portfolio optimization fails."""

    def __init__(self, message: str, optimizer_type: str = None, constraints: list = None):
        """
        Initialize optimization error.

        Args:
            message: Error message from optimizer
            optimizer_type: Type of optimizer that failed
            constraints: Constraints that were applied
        """
        super().__init__(message, {
            'optimizer_type': optimizer_type,
            'constraints': constraints or []
        })


class DataValidationError(PortfolioConstructionError):
    """Raised when input data validation fails."""

    def __init__(self, data_type: str, issue: str, affected_symbols: list = None):
        """
        Initialize data validation error.

        Args:
            data_type: Type of data that failed validation (signals, price_data, etc.)
            issue: Description of the validation issue
            affected_symbols: List of affected stock symbols
        """
        message = f"Data validation failed for {data_type}: {issue}"
        super().__init__(message, {
            'data_type': data_type,
            'issue': issue,
            'affected_symbols': affected_symbols or []
        })


class WeightAllocationError(PortfolioConstructionError):
    """Raised when weight allocation fails."""

    def __init__(self, allocation_type: str, reason: str, box_key: str = None):
        """
        Initialize weight allocation error.

        Args:
            allocation_type: Type of allocation that failed (box, within_box, etc.)
            reason: Reason for failure
            box_key: Box key if applicable
        """
        message = f"Weight allocation failed for {allocation_type}: {reason}"
        details = {'allocation_type': allocation_type, 'reason': reason}
        if box_key:
            details['box_key'] = box_key
        super().__init__(message, details)