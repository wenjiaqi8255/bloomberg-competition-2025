"""
Unified Enum Types

Single source of truth for all enumerations in the trading system.
"""

from enum import Enum


class AssetClass(Enum):
    """Asset class enumeration - comprehensive and standardized."""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTOCURRENCY = "cryptocurrency"
    ETF = "etf"
    REAL_ESTATE = "real_estate"
    ALTERNATIVE = "alternative"


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


class TimeFrame(Enum):
    """Time frame enumeration for different analysis periods."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    INTRADAY = "intraday"


class RebalanceFrequency(Enum):
    """Rebalancing frequency options."""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    SEMI_ANNUALLY = "semi_annually"
    ANNUALLY = "annually"


class TradeSide(Enum):
    """Trade side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = "open"
    CLOSED = "closed"
    PARTIAL = "partial"


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ComplianceStatus(Enum):
    """Compliance checking status."""
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    WARNING = "warning"
    UNKNOWN = "unknown"


class LogLevel(Enum):
    """Logging levels."""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# Legacy aliases for backward compatibility
Currency = AssetClass.CRYPTOCURRENCY  # @deprecated: Use AssetClass.CRYPTOCURRENCY