"""
Base strategy interface for all trading strategies.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd

# Import types for orchestration compatibility
# Use TYPE_CHECKING to avoid circular imports
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..types.signals import TradingSignal


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    All strategies must implement the following methods:
    - generate_signals: Generate trading signals based on market data
    - get_parameters: Get strategy hyperparameters
    - validate_parameters: Validate strategy parameters
    """

    def __init__(self, name: str, **kwargs):
        """
        Initialize the base strategy.

        Args:
            name: Strategy name for identification
            **kwargs: Strategy-specific parameters
        """
        self.name = name
        self.parameters = kwargs
        self.validate_parameters()

    @abstractmethod
    def generate_signals(self, price_data: Dict[str, pd.DataFrame],
                        start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate trading signals for the strategy.

        Args:
            price_data: Dictionary of price DataFrames for each symbol
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with trading signals (columns: symbols, index: dates)
            Positive values = long positions, negative = short, 0 = neutral
        """
        pass

    @abstractmethod
    def validate_parameters(self):
        """Validate strategy parameters. Raises ValueError if invalid."""
        pass

    def get_parameters(self) -> Dict:
        """Get strategy hyperparameters."""
        return self.parameters.copy()

    def set_parameters(self, **kwargs):
        """Update strategy parameters."""
        self.parameters.update(kwargs)
        self.validate_parameters()

    def get_name(self) -> str:
        """Get strategy name."""
        return self.name

    def get_info(self) -> Dict:
        """Get strategy information."""
        return {
            'name': self.name,
            'parameters': self.get_parameters(),
            'type': self.__class__.__name__
        }

    def calculate_returns(self, price_data: pd.DataFrame,
                         lookback_days: int = 252) -> pd.Series:
        """
        Calculate returns over a specified lookback period.

        Args:
            price_data: Price DataFrame
            lookback_days: Number of days to look back for return calculation

        Returns:
            Series with returns for each date
        """
        returns = price_data['Close'].pct_change(periods=lookback_days)
        return returns

    def calculate_volatility(self, price_data: pd.DataFrame,
                           lookback_days: int = 20) -> pd.Series:
        """
        Calculate rolling volatility.

        Args:
            price_data: Price DataFrame
            lookback_days: Number of days for volatility calculation

        Returns:
            Series with volatility for each date
        """
        returns = price_data['Close'].pct_change()
        volatility = returns.rolling(window=lookback_days).std() * (252 ** 0.5)
        return volatility

    def calculate_moving_average(self, price_data: pd.DataFrame,
                                window: int) -> pd.Series:
        """
        Calculate simple moving average.

        Args:
            price_data: Price DataFrame
            window: Moving average window

        Returns:
            Series with moving average values
        """
        return price_data['Close'].rolling(window=window).mean()

    def calculate_exponential_moving_average(self, price_data: pd.DataFrame,
                                           window: int) -> pd.Series:
        """
        Calculate exponential moving average.

        Args:
            price_data: Price DataFrame
            window: EMA window

        Returns:
            Series with EMA values
        """
        return price_data['Close'].ewm(span=window).mean()

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', **{self.parameters})"


class Strategy:
    """Strategy interface for orchestration components."""

    def __init__(self, name: str, **kwargs):
        self.name = name
        self.parameters = kwargs

    def generate_signals(self, date: datetime) -> List['TradingSignal']:
        """Generate trading signals for a given date."""
        raise NotImplementedError("Subclasses must implement generate_signals")

    def get_strategy_name(self) -> str:
        """Get strategy name."""
        return self.name


class LegacyBaseStrategy:
    """
    Legacy BaseStrategy class for backward compatibility.

    This class replicates the interface that was previously in data_types.py
    and is used by strategies like SatelliteStrategy and CoreFFMLStrategy.
    """

    def __init__(self, config: Dict[str, Any] = None, backtest_config: Dict[str, Any] = None):
        self.config = config or {}
        self.backtest_config = backtest_config or {}

    def generate_signals(self, market_data: Dict[str, Any]) -> Dict[str, float]:
        """Generate trading signals. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement generate_signals")

    def get_name(self) -> str:
        """Get strategy name."""
        return self.__class__.__name__