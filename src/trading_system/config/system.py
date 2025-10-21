"""
DEPRECATED: Legacy system configuration

Use pydantic.system instead:
    from trading_system.config.pydantic.system import SystemConfig
"""

import warnings

# Import from Pydantic by default
from .pydantic.system import SystemConfig

# Emit warning when importing from legacy module
warnings.warn(
    "Importing SystemConfig from config.system is deprecated. "
    "Use 'from trading_system.config.pydantic.system import SystemConfig' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['SystemConfig']