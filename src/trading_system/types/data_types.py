"""
Type definitions for trading system data structures.
"""

from typing import Dict, List, Optional, Union, Any, TypedDict
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
from enum import Enum


class AssetClass(Enum):
    """Asset class enumeration."""
    EQUITY = "equity"
    BOND = "bond"
    COMMODITY = "commodity"
    CURRENCY = "currency"
    CRYPTOCURRENCY = "cryptocurrency"
    ETF = "etf"


class DataSource(Enum):
    """Data source enumeration."""
    YFINANCE = "yfinance"
    ALPHA_VANTAGE = "alpha_vantage"
    BLOOMBERG = "bloomberg"
    QUANDL = "quandl"
    KENNETH_FRENCH = "kenneth_french"


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NEUTRAL = "neutral"


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


@dataclass
class BacktestConfig:
    """Backtest configuration structure."""
    initial_capital: float
    start_date: datetime
    end_date: datetime
    symbols: List[str]
    transaction_cost: float = 0.001
    slippage: float = 0.0005
    benchmark_symbol: Optional[str] = None
    rebalance_frequency: str = "daily"


@dataclass
class SystemConfig:
    """System orchestrator configuration."""
    # Basic system parameters
    system_name: str
    core_weight: float = 0.75
    satellite_weight: float = 0.25
    max_positions: int = 20
    rebalance_frequency: int = 30
    risk_budget: float = 0.15
    volatility_target: float = 0.12

    # Advanced parameters
    min_correlation_threshold: float = 0.7
    max_sector_allocation: float = 0.25
    ips_compliance_required: bool = True
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 1000000
    transaction_costs: float = 0.001
    slippage: float = 0.0005  # daily, weekly, monthly


@dataclass
class StrategyConfig:
    """Strategy configuration structure."""
    name: str
    parameters: Dict[str, Any]
    lookback_period: int
    universe: List[str]
    allocation_method: str = "equal_weight"


class DataValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class DataValidator:
    """Data validation utilities."""

    @staticmethod
    def validate_price_data(df: pd.DataFrame, symbol: str) -> bool:
        """
        Validate price data DataFrame.

        Args:
            df: DataFrame with price data
            symbol: Asset symbol for validation

        Returns:
            True if valid, raises DataValidationError otherwise

        Raises:
            DataValidationError: If data is invalid
        """
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

        # Check required columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")

        # Check for empty DataFrame
        if df.empty:
            raise DataValidationError(f"Empty DataFrame for symbol {symbol}")

        # Check for missing values
        if df[required_columns].isnull().all().any():
            raise DataValidationError(f"All values missing in required columns for {symbol}")

        # Validate price ranges
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if (df[col] <= 0).any():
                raise DataValidationError(f"Non-positive prices found in {col} for {symbol}")

        # Validate High-Low relationships
        invalid_hl = (df['High'] < df['Low']).any()
        if invalid_hl:
            raise DataValidationError(f"High < Low found for {symbol}")

        # Validate OHLC relationships
        invalid_ohlc = (
            (df['High'] < df['Open']) |
            (df['High'] < df['Close']) |
            (df['Low'] > df['Open']) |
            (df['Low'] > df['Close'])
        ).any()

        if invalid_ohlc:
            raise DataValidationError(f"Invalid OHLC relationships found for {symbol}")

        # Validate volume
        if (df['Volume'] < 0).any():
            raise DataValidationError(f"Negative volume found for {symbol}")

        return True

    @staticmethod
    def validate_factor_data(df: pd.DataFrame, data_source: str) -> bool:
        """
        Validate factor data DataFrame.

        Args:
            df: DataFrame with factor data
            data_source: Name of the data source for validation

        Returns:
            True if valid, raises DataValidationError otherwise
        """
        # Check required FF5 columns
        required_columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required factor columns: {missing_columns}")

        # Check for empty DataFrame
        if df.empty:
            raise DataValidationError(f"Empty factor DataFrame for source {data_source}")

        # Check for missing values
        if df[required_columns].isnull().all().any():
            raise DataValidationError(f"All values missing in required factor columns for {data_source}")

        # Validate factor return ranges (should be reasonable)
        for col in ['MKT', 'SMB', 'HML', 'RMW', 'CMA']:
            if ((df[col] < -1.0) | (df[col] > 1.0)).any():
                raise DataValidationError(f"Factor returns out of range for {col} in {data_source}")

        # Validate risk-free rate range
        if ((df['RF'] < -0.1) | (df['RF'] > 0.1)).any():
            raise DataValidationError(f"Risk-free rate out of range for {data_source}")

        return True

    @staticmethod
    def validate_signals(signals: pd.DataFrame, symbols: List[str]) -> bool:
        """
        Validate trading signals DataFrame.

        Args:
            signals: DataFrame with trading signals
            symbols: Expected symbols in the signals

        Returns:
            True if valid, raises DataValidationError otherwise
        """
        # Check that all expected symbols are present
        missing_symbols = [sym for sym in symbols if sym not in signals.columns]
        if missing_symbols:
            raise DataValidationError(f"Missing signals for symbols: {missing_symbols}")

        # Check signal values are within valid range
        for symbol in symbols:
            if symbol in signals.columns:
                signal_values = signals[symbol].dropna()
                if ((signal_values < 0) | (signal_values > 1)).any():
                    raise DataValidationError(f"Invalid signal values for {symbol}")

        return True

    @staticmethod
    def validate_portfolio_weights(weights: Dict[str, float]) -> bool:
        """
        Validate portfolio weights.

        Args:
            weights: Dictionary of symbol -> weight mappings

        Returns:
            True if valid, raises DataValidationError otherwise
        """
        # Check weights sum to approximately 1.0
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:  # Allow small rounding errors
            raise DataValidationError(f"Portfolio weights sum to {total_weight}, expected 1.0")

        # Check individual weights are valid
        for symbol, weight in weights.items():
            if weight < 0 or weight > 1:
                raise DataValidationError(f"Invalid weight {weight} for {symbol}")

        return True


class BaseStrategy:
    """Base class for all trading strategies."""

    def __init__(self, config: Dict[str, Any] = None, backtest_config: Dict[str, Any] = None):
        self.config = config or {}
        self.backtest_config = backtest_config or {}

    def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate trading signals. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement generate_signals")

    def get_name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__


# Type aliases for common data structures
PriceDataFrame = pd.DataFrame  # DataFrame with price data
SignalDataFrame = pd.DataFrame  # DataFrame with trading signals
PortfolioHistory = pd.DataFrame  # DataFrame with portfolio history
TradeList = List[Trade]  # List of trades
SignalDict = Dict[str, float]  # Dictionary of symbol -> signal mappings