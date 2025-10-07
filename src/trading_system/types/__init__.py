"""
Unified Type System for Trading System

Single source of truth for all trading system data types.
Following DRY principles and SOLID design.
"""

# Core unified types
from .signals import TradingSignal, SignalType, SignalList, SignalDict
from .portfolio import Position, Trade, PortfolioSnapshot, PositionList, TradeList, PortfolioHistory
from .market_data import PriceData, FactorData, PriceDict, FactorDict
from .enums import (
    AssetClass, DataSource, TimeFrame, RebalanceFrequency,
    TradeSide, OrderType, PositionStatus, RiskLevel,
    ComplianceStatus, LogLevel
)

# Legacy imports for backward compatibility - marked for deprecation
from .portfolio import PortfolioPosition  # @deprecated: Use Position instead

# Import configuration classes from config module
from ..config.backtest import BacktestConfig as NewBacktestConfig
from ..config.system import SystemConfig as NewSystemConfig
from ..config.strategy import StrategyConfig as NewStrategyConfig

# Still use data_types for remaining items until full migration
from .data_types import (
    AssetMetadata,
)

# Validation classes are now imported from utils module
from ..data.validation import DataValidationError, DataValidator

# Strategy classes are now imported from strategies module
# Avoid circular import by using forward reference in type hints
# Import directly when needed: from ..strategies.base_strategy import BaseStrategy, Strategy

# Type aliases
PriceDataFrame = PriceDict
SignalDataFrame = SignalDict

__all__ = [
    # New unified types
    'TradingSignal',
    'SignalType',
    'Position',
    'Trade',
    'PortfolioSnapshot',
    'PriceData',
    'FactorData',
    'AssetClass',
    'DataSource',
    'TimeFrame',
    'RebalanceFrequency',
    'TradeSide',
    'OrderType',
    'PositionStatus',
    'RiskLevel',
    'ComplianceStatus',
    'LogLevel',
  
    # Type aliases
    'SignalList',
    'SignalDict',
    'PositionList',
    'TradeList',
    'PortfolioHistory',
    'PriceDict',
    'FactorDict',
    'PriceDataFrame',
    'SignalDataFrame',

    # Legacy items (to be removed after migration)
    'PortfolioPosition',  # @deprecated: Use Position instead
    'AssetMetadata',
    'NewBacktestConfig',  # Import from config.backtest
    'NewSystemConfig',   # Import from config.system
    'NewStrategyConfig', # Import from config.strategy
    'DataValidationError',
    'DataValidator',
]