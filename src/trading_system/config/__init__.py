"""
Trading System Configuration (Mixed Legacy + Pydantic)
=====================================================

Recommended imports (Pydantic-based):
    from trading_system.config.pydantic import (
        StrategyConfig,
        BacktestConfig,
        ConfigLoader,
    )

Legacy imports (DEPRECATED):
    from trading_system.config import ConfigFactory  # Don't use
"""

import warnings

# Import from Pydantic by default
from .pydantic.base import BasePydanticConfig as BaseConfig
from .pydantic.strategy import StrategyConfig
from .pydantic.backtest import BacktestConfig
from .pydantic.system import SystemConfig
from .pydantic.loader import ConfigLoader

# Legacy compatibility (will be removed)
from .factory import ConfigFactory

# Emit warning when importing ConfigFactory
def __getattr__(name):
    if name == 'ConfigFactory':
        warnings.warn(
            "Importing ConfigFactory from config is deprecated. "
            "Use ConfigLoader instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return ConfigFactory
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Recommended (Pydantic)
    'BaseConfig',
    'StrategyConfig',
    'BacktestConfig', 
    'SystemConfig',
    'ConfigLoader',
    
    # Deprecated
    'ConfigFactory',
]