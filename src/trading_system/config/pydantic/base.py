"""
Base Pydantic Configuration

Foundation for all Pydantic-based configuration classes.
Follows KISS principle - minimal, focused functionality.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger(__name__)


class BasePydanticConfig(BaseModel):
    """
    Base configuration class using Pydantic.
    
    Provides:
    - Automatic validation
    - Type safety
    - Clear error messages
    - Zero conversion logic
    """
    
    # Common fields
    name: str = Field(default="", description="Configuration name")
    version: str = Field(default="1.0.0", description="Configuration version")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    
    class Config:
        # Enable strict validation
        validate_assignment = True
        # Allow extra fields for flexibility
        extra = "forbid"
        # Use enum values instead of enum objects
        use_enum_values = True
        # Validate default values
        validate_default = True
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v):
        """Validate name field."""
        if v and len(v.strip()) == 0:
            raise ValueError("Name cannot be empty string")
        return v.strip() if v else ""
    
    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        """Validate version format."""
        if v and not isinstance(v, str):
            raise ValueError("Version must be a string")
        return v
    
    def get_summary(self) -> dict:
        """Get configuration summary for logging."""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'version': self.version,
            'created_at': self.created_at.isoformat(),
            'fields_count': len(self.__fields__)
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.version}')"
