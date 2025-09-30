"""
Type definitions for trading system data structures.
"""

from typing import Dict, List, Optional, Union, Any, TypedDict
from datetime import datetime
from dataclasses import dataclass
import pandas as pd

# Import enums from the centralized enums module (DRY principle)
from .enums import AssetClass, DataSource, SignalType


@dataclass
class AssetMetadata:
    """Metadata for financial assets."""
    symbol: str
    name: str
    asset_class: AssetClass
    currency: str = "USD"
    exchange: str = "NYSE"
    is_active: bool = True
    min_lot_size: int = 1
    tick_size: float = 0.01


@dataclass
class PriceData:
    """Standardized price data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None
    source: DataSource = DataSource.YFINANCE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'adj_close': self.adj_close,
            'source': self.source.value
        }


@dataclass
class TradingSignal:
    """Trading signal structure."""
    symbol: str
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    price: float
    confidence: float = 1.0  # 0.0 to 1.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class PortfolioPosition:
    """Portfolio position structure."""
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float  # Portfolio weight (0.0 to 1.0)


@dataclass
class Trade:
    """Trade execution structure."""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    trade_id: Optional[str] = None


@dataclass
class PortfolioSnapshot:
    """Portfolio snapshot at a point in time."""
    timestamp: datetime
    total_value: float
    cash_balance: float
    positions: List[PortfolioPosition]
    daily_return: float
    total_return: float
    drawdown: float


# Configuration classes have been moved to config/ module to follow SOLID principles
# Import from config.backtest, config.system, config.strategy instead


# Data validation classes have been moved to utils/validation.py to follow SOLID principles
# Import from utils.validation instead


# Strategy classes have been moved to strategies/base_module to follow SOLID principles
# Import from strategies.base_strategy instead


@dataclass
class SystemPerformance:
    """System performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    timestamp: datetime


# Type aliases for common data structures
PriceDataFrame = pd.DataFrame  # DataFrame with price data
SignalDataFrame = pd.DataFrame  # DataFrame with trading signals
PortfolioHistory = pd.DataFrame  # DataFrame with portfolio history
TradeList = List[Trade]  # List of trades
SignalDict = Dict[str, float]  # Dictionary of symbol -> signal mappings