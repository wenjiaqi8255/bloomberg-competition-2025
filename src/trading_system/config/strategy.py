"""
DEPRECATED: Legacy strategy configuration

Use pydantic.strategy instead:
    from trading_system.config.pydantic.strategy import StrategyConfig, StrategyType
"""

import warnings

# Import from Pydantic by default
from .pydantic.strategy import StrategyConfig, StrategyType

# Emit warning when importing from legacy module
warnings.warn(
    "Importing StrategyConfig from config.strategy is deprecated. "
    "Use 'from trading_system.config.pydantic.strategy import StrategyConfig' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['StrategyConfig', 'StrategyType']