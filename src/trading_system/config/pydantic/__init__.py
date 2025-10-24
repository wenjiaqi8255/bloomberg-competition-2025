"""
Pydantic Configuration System

Modern configuration system using Pydantic for automatic validation and type safety.
Replaces the legacy dataclass-based configuration system.

Key Features:
- Automatic validation on creation
- Type safety with IDE support
- Clear error messages
- Zero conversion logic (direct YAML mapping)
- Immediate failure on configuration errors

Usage:
    from trading_system.config.pydantic import ConfigLoader
    
    loader = ConfigLoader()
    config = loader.load_from_yaml("config.yaml")
"""

from .base import BasePydanticConfig
from .portfolio import (
    PortfolioConstructionConfig,
    BoxBasedPortfolioConfig,
    QuantitativePortfolioConfig,
    OptimizerConfig,
    CovarianceConfig
)
from .strategy import StrategyConfig
from .backtest import BacktestConfig
from .loader import ConfigLoader

__all__ = [
    'BasePydanticConfig',
    'PortfolioConstructionConfig',
    'BoxBasedPortfolioConfig',
    'QuantitativePortfolioConfig', 
    'OptimizerConfig',
    'CovarianceConfig',
    'StrategyConfig',
    'BacktestConfig',
    'ConfigLoader'
]
