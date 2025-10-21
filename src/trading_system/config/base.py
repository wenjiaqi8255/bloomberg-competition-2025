"""
DEPRECATED: Legacy base configuration class

Use pydantic.base instead:
    from trading_system.config.pydantic.base import BaseConfig
"""

import warnings

# Import from Pydantic by default
from .pydantic.base import BaseConfig

# Emit warning when importing from legacy module
warnings.warn(
    "Importing BaseConfig from config.base is deprecated. "
    "Use 'from trading_system.config.pydantic.base import BaseConfig' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['BaseConfig']