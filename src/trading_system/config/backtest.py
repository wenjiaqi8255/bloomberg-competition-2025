"""
DEPRECATED: Legacy backtest configuration

Use pydantic.backtest instead:
    from trading_system.config.pydantic.backtest import BacktestConfig
"""

import warnings

# Import from Pydantic by default
from .pydantic.backtest import BacktestConfig

# Emit warning when importing from legacy module
warnings.warn(
    "Importing BacktestConfig from config.backtest is deprecated. "
    "Use 'from trading_system.config.pydantic.backtest import BacktestConfig' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['BacktestConfig']