"""
Backtest Configuration

Unified backtest configuration replacing multiple conflicting definitions.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any

from .base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig(BaseConfig):
    """
    Unified backtest configuration.

    Single source of truth for all backtesting parameters,
    consolidating the previous multiple BacktestConfig definitions.
    """

    # Core trading parameters
    initial_capital: float = 1_000_000
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    symbols: List[str] = field(default_factory=list)
    benchmark_symbol: Optional[str] = None

    # Transaction costs (unified from different versions)
    commission_rate: float = 0.001
    spread_rate: float = 0.0005
    slippage_rate: float = 0.0002
    short_borrow_rate: float = 0.02

    # Trading parameters
    rebalance_frequency: str = 'monthly'  # 'daily', 'weekly', 'monthly', 'quarterly'
    position_limit: float = 0.10  # Maximum position size (10% default)
    rebalance_threshold: float = 0.01  # Minimum weight change to trigger rebalance

    # Risk management
    max_drawdown_limit: float = 0.20  # Stop trading if drawdown exceeds this
    volatility_limit: float = 0.30    # Portfolio volatility limit
    enable_stop_loss: bool = True
    stop_loss_threshold: float = -0.07  # -7% stop loss for individual positions

    # Performance calculation
    risk_free_rate: float = 0.02

    # Advanced options
    enable_short_selling: bool = False
    enable_leverage: bool = False
    max_leverage: float = 1.0

    # Output and logging
    save_results: bool = True
    output_directory: str = "results"
    log_level: str = "INFO"

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        # Validate cost parameters are reasonable
        if not 0 <= self.commission_rate <= 0.01:
            logger.warning(f"commission_rate {self.commission_rate} seems unusual")

        if not 0 <= self.spread_rate <= 0.01:
            logger.warning(f"spread_rate {self.spread_rate} seems unusual")

        if not 0 <= self.slippage_rate <= 0.01:
            logger.warning(f"slippage_rate {self.slippage_rate} seems unusual")

        # Validate position limits
        if not 0 <= self.position_limit <= 1:
            raise ValueError("position_limit must be between 0 and 1")

        # Validate dates
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

        # Validate rebalance frequency
        valid_frequencies = ['daily', 'weekly', 'monthly', 'quarterly']
        if self.rebalance_frequency not in valid_frequencies:
            raise ValueError(f"rebalance_frequency must be one of {valid_frequencies}")

        # Validate risk parameters
        if not 0 <= self.max_drawdown_limit <= 1:
            raise ValueError("max_drawdown_limit must be between 0 and 1")

        if not 0 <= self.volatility_limit <= 2:  # Allow up to 200% volatility
            raise ValueError("volatility_limit must be between 0 and 2")

        # Validate stop loss threshold
        if self.enable_stop_loss and not -1 <= self.stop_loss_threshold <= 0:
            raise ValueError("stop_loss_threshold must be between -100% and 0%")

    def _set_defaults(self):
        """Set default values."""
        super()._set_defaults()

        # Set default dates if not provided
        if not self.start_date:
            self.start_date = datetime(2023, 1, 1)

        if not self.end_date:
            self.end_date = datetime(2023, 12, 31)

        logger.info(f"Backtest period: {self.start_date.date()} to {self.end_date.date()}")

    @property
    def total_cost_rate(self) -> float:
        """Calculate total transaction cost rate."""
        return self.commission_rate + self.spread_rate + self.slippage_rate

    @property
    def trading_days_per_year(self) -> int:
        """Get trading days per year based on rebalance frequency."""
        frequency_map = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4
        }
        return frequency_map.get(self.rebalance_frequency, 252)

    def get_position_limit_amount(self, portfolio_value: float) -> float:
        """Calculate maximum position amount in dollar terms."""
        return portfolio_value * self.position_limit

    def is_rebalance_needed(self, current_weight: float, target_weight: float,
                          portfolio_value: float) -> bool:
        """
        Check if rebalancing is needed based on threshold.

        Args:
            current_weight: Current position weight
            target_weight: Target position weight
            portfolio_value: Total portfolio value

        Returns:
            True if rebalancing is needed
        """
        weight_change = abs(target_weight - current_weight)
        dollar_change = weight_change * portfolio_value
        threshold_amount = self.rebalance_threshold * portfolio_value

        return dollar_change >= threshold_amount

    def get_summary(self) -> Dict[str, Any]:
        """Get detailed configuration summary."""
        base_summary = super().get_summary()

        base_summary.update({
            'initial_capital': f"${self.initial_capital:,.0f}",
            'period': f"{self.start_date.date()} to {self.end_date.date()}",
            'symbols_count': len(self.symbols),
            'rebalance_frequency': self.rebalance_frequency,
            'total_cost_rate': f"{self.total_cost_rate:.3%}",
            'risk_management': {
                'position_limit': f"{self.position_limit:.1%}",
                'max_drawdown': f"{self.max_drawdown_limit:.1%}",
                'stop_loss': f"{self.stop_loss_threshold:.1%}" if self.enable_stop_loss else "disabled",
                'volatility_limit': f"{self.volatility_limit:.1%}"
            },
            'trading_options': {
                'short_selling': self.enable_short_selling,
                'leverage': f"{self.max_leverage:.1f}x" if self.enable_leverage else "disabled"
            }
        })

        return base_summary

    @classmethod
    def create_simple(cls,
                     initial_capital: float = 1_000_000,
                     start_date: str = "2023-01-01",
                     end_date: str = "2023-12-31",
                     symbols: List[str] = None) -> 'BacktestConfig':
        """
        Create a simple configuration with minimal parameters.

        Args:
            initial_capital: Starting capital
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to trade

        Returns:
            BacktestConfig instance
        """
        return cls(
            initial_capital=initial_capital,
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            symbols=symbols or []
        )

    @classmethod
    def create_academic(cls,
                       initial_capital: float = 1_000_000,
                       start_date: str = "2018-01-01",
                       end_date: str = "2023-12-31",
                       symbols: List[str] = None) -> 'BacktestConfig':
        """
        Create configuration suitable for academic research.

        Args:
            initial_capital: Starting capital
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to trade

        Returns:
            BacktestConfig instance with academic settings
        """
        return cls(
            initial_capital=initial_capital,
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            symbols=symbols or [],
            commission_rate=0.0005,  # Lower costs for academic analysis
            spread_rate=0.0002,
            slippage_rate=0.0001,
            rebalance_frequency='monthly',
            position_limit=0.05,     # More conservative position sizing
            enable_short_selling=True,
            risk_free_rate=0.02
        )

    @classmethod
    def create_production(cls,
                         initial_capital: float = 10_000_000,
                         start_date: str = "2023-01-01",
                         end_date: str = "2023-12-31",
                         symbols: List[str] = None) -> 'BacktestConfig':
        """
        Create configuration suitable for production trading.

        Args:
            initial_capital: Starting capital
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            symbols: List of symbols to trade

        Returns:
            BacktestConfig instance with production settings
        """
        return cls(
            initial_capital=initial_capital,
            start_date=datetime.fromisoformat(start_date),
            end_date=datetime.fromisoformat(end_date),
            symbols=symbols or [],
            commission_rate=0.001,   # Realistic costs
            spread_rate=0.0005,
            slippage_rate=0.0002,
            short_borrow_rate=0.03,
            rebalance_frequency='weekly',
            position_limit=0.08,
            enable_stop_loss=True,
            stop_loss_threshold=-0.05,
            max_drawdown_limit=0.15,
            volatility_limit=0.25
        )