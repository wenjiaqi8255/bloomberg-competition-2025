"""
Unified Enum Types

Single source of truth for all enumerations in the trading system.
"""

from enum import Enum


class DataSource(Enum):
    """Data source enumeration - covers all major data providers."""
    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    BLOOMBERG = "bloomberg"
    QUANDL = "quandl"
    KENNETH_FRENCH = "kenneth_french"
    POLYGON = "polygon"
    IEX = "iex"
    EXCEL_FILE = "excel_file"


class SignalType(Enum):
    """Trading signal types - simplified and comprehensive."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NEUTRAL = "neutral"