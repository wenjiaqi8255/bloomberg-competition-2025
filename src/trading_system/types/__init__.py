"""
Unified Type System for Trading System

Single source of truth for all trading system data types.
Following DRY principles and SOLID design.
"""

# Core unified types
from .signals import TradingSignal, SignalType, SignalList, SignalDict
from .portfolio import Position, Trade, PortfolioSnapshot, PositionList, TradeList, PortfolioHistory
from .market_data import PriceData, FactorData, PriceDict, FactorDict
from .enums import DataSource

# Type aliases
PriceDataFrame = PriceDict
SignalDataFrame = SignalDict

__all__ = [
    # Core types
    'TradingSignal',
    'SignalType',
    'Position',
    'Trade',
    'PortfolioSnapshot',
    'PriceData',
    'FactorData',
    'DataSource',
  
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
]