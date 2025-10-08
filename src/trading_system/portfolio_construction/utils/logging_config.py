"""
Logging configuration for portfolio construction module.

Provides centralized logging configuration with structured logging
for better debugging and monitoring of portfolio construction processes.
"""

import logging
import sys
from typing import Dict, Any, Optional
from datetime import datetime
import json


class PortfolioConstructionFormatter(logging.Formatter):
    """
    Custom formatter for portfolio construction logs.

    Adds structured information relevant to portfolio construction
    including method details, processing stages, and performance metrics.
    """

    def __init__(self, include_extra: bool = True):
        """
        Initialize formatter.

        Args:
            include_extra: Whether to include extra fields in log output
        """
        super().__init__()
        self.include_extra = include_extra

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with portfolio construction context."""
        # Basic format
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        level = record.levelname
        name = record.name.split('.')[-1]  # Only show module name
        message = record.getMessage()

        # Build basic log line
        log_line = f"[{timestamp}] {level:8s} {name:25s} {message}"

        # Add extra fields if available
        if self.include_extra and hasattr(record, 'extra_fields'):
            extra = record.extra_fields
            if extra:
                extra_str = " | ".join(f"{k}={v}" for k, v in extra.items())
                log_line += f" | {extra_str}"

        return log_line


class PortfolioConstructionLogger:
    """
    Centralized logger for portfolio construction operations.

    Provides structured logging with context awareness for different
    portfolio construction methods and stages.
    """

    def __init__(self, name: str = None):
        """
        Initialize portfolio construction logger.

        Args:
            name: Logger name (defaults to module name)
        """
        self.logger = logging.getLogger(name or __name__)
        self._setup_logger()

    def _setup_logger(self):
        """Setup logger with custom formatter and handlers."""
        # Clear existing handlers
        self.logger.handlers.clear()

        # Set level
        self.logger.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)

        # Custom formatter
        formatter = PortfolioConstructionFormatter(include_extra=True)
        console_handler.setFormatter(formatter)

        # Add handler
        self.logger.addHandler(console_handler)

        # Prevent propagation to avoid duplicate logs
        self.logger.propagate = False

    def log_construction_start(self, method: str, date: datetime, universe_size: int):
        """Log portfolio construction start."""
        self._log_with_extra(
            level=logging.INFO,
            message=f"Starting portfolio construction",
            extra_fields={
                'method': method,
                'date': date.date().isoformat(),
                'universe_size': universe_size,
                'stage': 'start'
            }
        )

    def log_construction_stage(self, stage: str, method: str, details: Dict[str, Any] = None):
        """Log construction stage progress."""
        extra_fields = {
            'method': method,
            'stage': stage
        }
        if details:
            extra_fields.update(details)

        self._log_with_extra(
            level=logging.INFO,
            message=f"Portfolio construction stage: {stage}",
            extra_fields=extra_fields
        )

    def log_construction_complete(self, method: str, positions: int,
                                processing_time: float = None, details: Dict[str, Any] = None):
        """Log portfolio construction completion."""
        extra_fields = {
            'method': method,
            'positions': positions,
            'stage': 'complete'
        }

        if processing_time is not None:
            extra_fields['processing_time_seconds'] = processing_time

        if details:
            extra_fields.update(details)

        self._log_with_extra(
            level=logging.INFO,
            message=f"Portfolio construction completed",
            extra_fields=extra_fields
        )

    def log_box_coverage(self, method: str, covered_boxes: int, total_boxes: int,
                       covered_weight: float, details: Dict[str, Any] = None):
        """Log box coverage analysis."""
        coverage_ratio = covered_boxes / total_boxes if total_boxes > 0 else 0

        extra_fields = {
            'method': method,
            'covered_boxes': covered_boxes,
            'total_boxes': total_boxes,
            'coverage_ratio': coverage_ratio,
            'covered_weight': covered_weight,
            'stage': 'box_analysis'
        }

        if details:
            extra_fields.update(details)

        self._log_with_extra(
            level=logging.INFO,
            message=f"Box coverage analysis: {coverage_ratio:.1%}",
            extra_fields=extra_fields
        )

    def log_optimization_result(self, method: str, success: bool,
                              iterations: int = None, objective_value: float = None,
                              details: Dict[str, Any] = None):
        """Log optimization results."""
        extra_fields = {
            'method': method,
            'optimization_success': success,
            'stage': 'optimization'
        }

        if iterations is not None:
            extra_fields['iterations'] = iterations
        if objective_value is not None:
            extra_fields['objective_value'] = objective_value

        if details:
            extra_fields.update(details)

        status = "success" if success else "failed"
        self._log_with_extra(
            level=logging.INFO if success else logging.ERROR,
            message=f"Portfolio optimization {status}",
            extra_fields=extra_fields
        )

    def log_error(self, method: str, error_type: str, error_message: str,
                 context: Dict[str, Any] = None):
        """Log portfolio construction error."""
        extra_fields = {
            'method': method,
            'error_type': error_type,
            'stage': 'error'
        }

        if context:
            extra_fields.update(context)

        self._log_with_extra(
            level=logging.ERROR,
            message=f"Portfolio construction error: {error_message}",
            extra_fields=extra_fields
        )

    def log_warning(self, method: str, warning_type: str, warning_message: str,
                   context: Dict[str, Any] = None):
        """Log portfolio construction warning."""
        extra_fields = {
            'method': method,
            'warning_type': warning_type,
            'stage': 'warning'
        }

        if context:
            extra_fields.update(context)

        self._log_with_extra(
            level=logging.WARNING,
            message=f"Portfolio construction warning: {warning_message}",
            extra_fields=extra_fields
        )

    def log_performance_metrics(self, method: str, metrics: Dict[str, Any]):
        """Log performance metrics."""
        extra_fields = {
            'method': method,
            'stage': 'performance_metrics'
        }
        extra_fields.update(metrics)

        self._log_with_extra(
            level=logging.INFO,
            message="Portfolio construction performance metrics",
            extra_fields=extra_fields
        )

    def _log_with_extra(self, level: int, message: str, extra_fields: Dict[str, Any]):
        """Log message with extra fields."""
        if not self.logger.isEnabledFor(level):
            return

        # Create log record
        record = self.logger.makeRecord(
            name=self.logger.name,
            level=level,
            fn="",
            lno=0,
            msg=message,
            args=(),
            exc_info=None
        )

        # Add extra fields
        record.extra_fields = extra_fields

        # Handle log record
        self.logger.handle(record)


class LoggingContext:
    """
    Context manager for portfolio construction logging.

    Provides automatic timing and context tracking for portfolio construction
    operations.
    """

    def __init__(self, logger: PortfolioConstructionLogger, method: str,
                 operation: str, context: Dict[str, Any] = None):
        """
        Initialize logging context.

        Args:
            logger: Portfolio construction logger
            method: Portfolio construction method
            operation: Operation description
            context: Additional context information
        """
        self.logger = logger
        self.method = method
        self.operation = operation
        self.context = context or {}
        self.start_time = None

    def __enter__(self):
        """Enter logging context."""
        self.start_time = datetime.now()

        extra_fields = {
            'operation': self.operation,
            'start_time': self.start_time.isoformat()
        }
        extra_fields.update(self.context)

        self.logger.log_construction_stage(
            stage=f"{self.operation}_start",
            method=self.method,
            details=extra_fields
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit logging context."""
        end_time = datetime.now()
        processing_time = (end_time - self.start_time).total_seconds()

        extra_fields = {
            'operation': self.operation,
            'end_time': end_time.isoformat(),
            'processing_time_seconds': processing_time
        }
        extra_fields.update(self.context)

        if exc_type is None:
            # Success
            self.logger.log_construction_stage(
                stage=f"{self.operation}_complete",
                method=self.method,
                details=extra_fields
            )
        else:
            # Error occurred
            extra_fields.update({
                'error_type': exc_type.__name__ if exc_type else 'Unknown',
                'error_message': str(exc_val) if exc_val else 'Unknown error'
            })

            self.logger.log_error(
                method=self.method,
                error_type=exc_type.__name__ if exc_type else 'Unknown',
                error_message=str(exc_val) if exc_val else 'Unknown error',
                context=extra_fields
            )


# Module-level logger instance
_module_logger = None


def get_logger(name: str = None) -> PortfolioConstructionLogger:
    """
    Get portfolio construction logger instance.

    Args:
        name: Logger name

    Returns:
        Portfolio construction logger
    """
    global _module_logger
    if _module_logger is None:
        _module_logger = PortfolioConstructionLogger(name)
    return _module_logger


def setup_logging(level: str = "INFO", format_type: str = "standard",
                 file_path: Optional[str] = None):
    """
    Setup portfolio construction logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format_type: Format type (standard, json, simple)
        file_path: Optional file path for log output
    """
    # Convert level string to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Setup root logger configuration
    root_logger = logging.getLogger('trading_system.portfolio_construction')
    root_logger.setLevel(numeric_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Choose formatter
    if format_type == "json":
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(name)s", "message": "%(message)s"}'
        )
    elif format_type == "simple":
        formatter = logging.Formatter('%(levelname)s - %(message)s')
    else:  # standard
        formatter = PortfolioConstructionFormatter(include_extra=True)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Optional file handler
    if file_path:
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Configure related modules
    logging.getLogger('trading_system.optimization').setLevel(numeric_level)
    logging.getLogger('trading_system.data.stock_classifier').setLevel(numeric_level)


def log_method_call(method: str, operation: str, context: Dict[str, Any] = None):
    """
    Decorator for logging method calls.

    Args:
        method: Portfolio construction method
        operation: Operation description
        context: Additional context
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger = get_logger()
            with LoggingContext(logger, method, operation, context):
                return func(*args, **kwargs)
        return wrapper
    return decorator