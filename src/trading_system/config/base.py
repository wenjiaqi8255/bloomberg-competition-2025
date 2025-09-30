"""
Base Configuration Class

Abstract base class for all configuration objects following SOLID principles.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class BaseConfig(ABC):
    """
    Abstract base class for all configuration objects.

    Provides common functionality:
    - Parameter validation
    - Serialization/deserialization
    - Default value management
    - Configuration merging
    """

    # Common configuration fields
    name: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_config()
        self._set_defaults()

    @abstractmethod
    def _validate_config(self):
        """Validate configuration parameters. Must be implemented by subclasses."""
        pass

    def _set_defaults(self):
        """Set default values. Can be overridden by subclasses."""
        if not self.name:
            self.name = self.__class__.__name__

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if hasattr(value, 'isoformat'):  # Handle datetime objects
                result[key] = value.isoformat()
            elif hasattr(value, 'to_dict'):  # Handle nested config objects
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseConfig':
        """Create configuration from dictionary."""
        # Handle datetime string conversion
        processed_dict = config_dict.copy()

        for key, value in processed_dict.items():
            if isinstance(value, str):
                # Try to parse as datetime
                try:
                    processed_dict[key] = datetime.fromisoformat(value)
                except ValueError:
                    pass  # Keep as string if not a datetime

        return cls(**processed_dict)

    def update(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                old_value = getattr(self, key)
                setattr(self, key, value)
                logger.info(f"Updated {self.__class__.__name__}.{key}: {old_value} -> {value}")
            else:
                logger.warning(f"Unknown configuration parameter: {key}")

        # Re-validate after updates
        self._validate_config()

    def copy(self) -> 'BaseConfig':
        """Create a copy of the configuration."""
        return self.__class__.from_dict(self.to_dict())

    def merge(self, other: 'BaseConfig') -> 'BaseConfig':
        """Merge with another configuration, other takes precedence."""
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot merge {self.__class__.__name__} with {other.__class__.__name__}")

        current_dict = self.to_dict()
        other_dict = other.to_dict()

        # Remove fields that shouldn't be merged
        for field in ['created_at', 'name', 'version']:
            other_dict.pop(field, None)

        current_dict.update(other_dict)
        return self.__class__.from_dict(current_dict)

    def validate_field(self, field_name: str, value: Any, validator_func=None) -> bool:
        """
        Validate a single field value.

        Args:
            field_name: Name of the field
            value: Value to validate
            validator_func: Optional custom validation function

        Returns:
            True if valid

        Raises:
            ValueError: If validation fails
        """
        if validator_func:
            if not validator_func(value):
                raise ValueError(f"Validation failed for {field_name}: {value}")
        return True

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary for logging."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'fields_count': len(self.__dataclass_fields__) if hasattr(self, '__dataclass_fields__') else 0
        }

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"{self.__class__.__name__}({self.to_dict()})"