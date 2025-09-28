"""
Dual Momentum Strategy implementation.

This strategy combines absolute and relative momentum:
1. Absolute momentum: Only consider assets with positive returns
2. Relative momentum: Select top performers from positively trending assets
3. Equal weight selected assets
4. Include cash position when insufficient positive momentum
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class DualMomentumStrategy(BaseStrategy):
    """
    Dual Momentum Strategy combining absolute and relative momentum.

    Strategy Logic:
    1. Calculate total returns over lookback period for all assets
    2. Filter for assets with positive absolute momentum (returns > 0)
    3. If insufficient assets with positive momentum, move to cash
    4. Otherwise, select top N assets by relative momentum (highest returns)
    5. Equal weight selected assets
    6. Rebalance monthly

    This strategy provides built-in downside protection and trend-following characteristics.
    """

    def __init__(self, name: str = "DualMomentum",
                 lookback_days: int = 252,
                 top_n_assets: int = 5,
                 minimum_positive_assets: int = 3,
                 cash_ticker: str = 'SHY',  # Short-term Treasury ETF
                 include_cash_in_universe: bool = True,
                 **kwargs):
        """
        Initialize Dual Momentum Strategy.

        Args:
            name: Strategy name
            lookback_days: Lookback period for momentum calculation (default: 252 = 1 year)
            top_n_assets: Number of top assets to select
            minimum_positive_assets: Minimum assets with positive momentum to stay invested
            cash_ticker: Ticker for cash position (default: SHY)
            include_cash_in_universe: Whether to include cash in asset universe
        """
        # Set attributes before validation
        self.lookback_days = lookback_days
        self.top_n_assets = top_n_assets
        self.minimum_positive_assets = minimum_positive_assets
        self.cash_ticker = cash_ticker
        self.include_cash_in_universe = include_cash_in_universe

        super().__init__(name=name, **kwargs)

    def validate_parameters(self):
        """Validate strategy parameters."""
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")

        if self.top_n_assets <= 0:
            raise ValueError("top_n_assets must be positive")

        if self.minimum_positive_assets <= 0:
            raise ValueError("minimum_positive_assets must be positive")

        if self.minimum_positive_assets > self.top_n_assets:
            raise ValueError("minimum_positive_assets cannot exceed top_n_assets")

        if not self.cash_ticker:
            raise ValueError("cash_ticker cannot be empty")

    def generate_signals(self, price_data: Dict[str, pd.DataFrame],
                        start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate trading signals based on dual momentum.

        Args:
            price_data: Dictionary of price DataFrames for each symbol
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with trading signals (weight allocations for each symbol)
        """
        logger.info(f"Generating dual momentum signals from {start_date} to {end_date}")

        # Create date range for monthly rebalancing
        date_range = pd.date_range(start=start_date, end=end_date, freq='MS')

        signals = pd.DataFrame(index=date_range, columns=list(price_data.keys()))

        for date in date_range:
            try:
                signal = self._generate_signal_for_date(price_data, date)
                signals.loc[date] = signal
            except Exception as e:
                logger.warning(f"Failed to generate signal for {date}: {e}")
                # Use previous signal or neutral position
                if len(signals) > 0 and date in signals.index:
                    prev_idx = signals.index.get_loc(date) - 1
                    if prev_idx >= 0:
                        signals.loc[date] = signals.iloc[prev_idx]

        return signals

    def _generate_signal_for_date(self, price_data: Dict[str, pd.DataFrame],
                                 date: datetime) -> pd.Series:
        """
        Generate trading signal for a specific date.

        Args:
            price_data: Dictionary of price DataFrames
            date: Date for signal generation

        Returns:
            Series with weight allocations
        """
        # Calculate momentum scores for all assets
        momentum_scores = {}

        for symbol, data in price_data.items():
            if self.include_cash_in_universe or symbol != self.cash_ticker:
                momentum = self._calculate_momentum_score(data, date)
                if momentum is not None:
                    momentum_scores[symbol] = momentum

        if not momentum_scores:
            logger.warning(f"No momentum scores calculated for {date}")
            return pd.Series(0, index=list(price_data.keys()))

        # Apply absolute momentum filter
        positive_momentum_assets = {
            symbol: score for symbol, score in momentum_scores.items()
            if score > 0
        }

        # Determine allocation strategy
        if len(positive_momentum_assets) < self.minimum_positive_assets:
            # Move to cash
            logger.debug(f"Only {len(positive_momentum_assets)} assets with positive momentum "
                        f"(< {self.minimum_positive_assets}), moving to cash")
            allocation = self._create_cash_allocation(price_data, 1.0)
        else:
            # Select top N assets by relative momentum
            sorted_assets = sorted(positive_momentum_assets.items(),
                                 key=lambda x: x[1], reverse=True)
            selected_assets = sorted_assets[:self.top_n_assets]

            # Equal weight selected assets
            weight_per_asset = 1.0 / len(selected_assets)
            allocation = pd.Series(0.0, index=list(price_data.keys()), dtype=float)

            for symbol, score in selected_assets:
                allocation[symbol] = weight_per_asset

            logger.debug(f"Selected {len(selected_assets)} assets: {[s[0] for s in selected_assets]}")

        return allocation

    def _calculate_momentum_score(self, price_data: pd.DataFrame,
                                date: datetime) -> Optional[float]:
        """
        Calculate momentum score for an asset.

        Args:
            price_data: Price DataFrame for the asset
            date: Date for momentum calculation

        Returns:
            Momentum score or None if insufficient data
        """
        try:
            # Get data up to the calculation date
            data_up_to_date = price_data[price_data.index <= date]

            if len(data_up_to_date) < self.lookback_days:
                logger.debug(f"Insufficient data for momentum calculation: "
                           f"only {len(data_up_to_date)} days available")
                return None

            # Calculate total return over lookback period
            start_price = data_up_to_date['Close'].iloc[-self.lookback_days]
            end_price = data_up_to_date['Close'].iloc[-1]

            if start_price <= 0:
                logger.warning(f"Invalid start price: {start_price}")
                return None

            momentum_score = (end_price - start_price) / start_price
            return momentum_score

        except Exception as e:
            logger.warning(f"Error calculating momentum score: {e}")
            return None

    def _create_cash_allocation(self, price_data: Dict[str, pd.DataFrame],
                               cash_weight: float) -> pd.Series:
        """
        Create allocation with cash position.

        Args:
            price_data: Price data dictionary
            cash_weight: Weight to allocate to cash (0-1)

        Returns:
            Series with cash allocation
        """
        allocation = pd.Series(0, index=list(price_data.keys()))

        if self.cash_ticker in allocation.index:
            allocation[self.cash_ticker] = cash_weight
        else:
            logger.warning(f"Cash ticker {self.cash_ticker} not in asset universe")
            # If cash ticker not available, distribute equally among all assets
            if len(allocation) > 0:
                allocation[:] = cash_weight / len(allocation)

        return allocation

    def get_momentum_regime(self, price_data: Dict[str, pd.DataFrame],
                           date: datetime) -> str:
        """
        Determine current market regime based on momentum characteristics.

        Args:
            price_data: Price data dictionary
            date: Date for regime analysis

        Returns:
            Regime description: 'bull', 'bear', 'neutral', or 'transition'
        """
        momentum_scores = {}

        for symbol, data in price_data.items():
            if symbol != self.cash_ticker:
                momentum = self._calculate_momentum_score(data, date)
                if momentum is not None:
                    momentum_scores[symbol] = momentum

        if not momentum_scores:
            return 'unknown'

        # Calculate statistics
        positive_count = sum(1 for score in momentum_scores.values() if score > 0)
        total_count = len(momentum_scores)
        positive_ratio = positive_count / total_count if total_count > 0 else 0
        avg_momentum = np.mean(list(momentum_scores.values()))

        # Determine regime
        if positive_ratio >= 0.8 and avg_momentum > 0.1:
            return 'strong_bull'
        elif positive_ratio >= 0.6 and avg_momentum > 0.05:
            return 'bull'
        elif positive_ratio <= 0.2 and avg_momentum < -0.05:
            return 'bear'
        elif positive_ratio <= 0.4:
            return 'weak_bear'
        else:
            return 'neutral'

    def calculate_risk_metrics(self, price_data: Dict[str, pd.DataFrame],
                              signals: pd.DataFrame) -> Dict:
        """
        Calculate risk metrics for the strategy.

        Args:
            price_data: Price data dictionary
            signals: Trading signals DataFrame

        Returns:
            Dictionary with risk metrics
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(price_data, signals)

            if portfolio_returns.empty:
                return {}

            # Basic risk metrics
            volatility = portfolio_returns.std() * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(portfolio_returns)
            var_95 = portfolio_returns.quantile(0.05)
            var_99 = portfolio_returns.quantile(0.01)

            # Momentum-specific metrics
            concentration_risk = self._calculate_concentration_risk(signals)
            turnover = self._calculate_turnover(signals)

            return {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'var_95': var_95,
                'var_99': var_99,
                'concentration_risk': concentration_risk,
                'turnover': turnover,
                'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() > 0 else 0
            }

        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {}

    def _calculate_portfolio_returns(self, price_data: Dict[str, pd.DataFrame],
                                   signals: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns based on signals and price data."""
        returns = pd.Series(dtype=float)

        for date in signals.index:
            if date == signals.index[0]:
                continue  # Skip first period

            # Get previous period's allocation
            prev_date = signals.index[signals.index.get_loc(date) - 1]
            allocation = signals.loc[prev_date]

            # Calculate returns for each position
            period_returns = []
            for symbol, weight in allocation.items():
                if weight > 0 and symbol in price_data:
                    try:
                        # Get price change over the period
                        symbol_data = price_data[symbol]
                        if date in symbol_data.index and prev_date in symbol_data.index:
                            asset_return = (symbol_data.loc[date, 'Close'] - symbol_data.loc[prev_date, 'Close']) / symbol_data.loc[prev_date, 'Close']
                            weighted_return = weight * asset_return
                            period_returns.append(weighted_return)
                    except Exception as e:
                        logger.warning(f"Error calculating return for {symbol}: {e}")

            if period_returns:
                returns[date] = sum(period_returns)

        return returns

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns series."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    def _calculate_concentration_risk(self, signals: pd.DataFrame) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration risk."""
        hhi_values = []
        for _, allocation in signals.iterrows():
            # Only consider non-zero weights
            weights = allocation[allocation > 0]
            if len(weights) > 0:
                # Normalize weights to sum to 1
                weights = weights / weights.sum()
                hhi = (weights ** 2).sum()
                hhi_values.append(hhi)

        return np.mean(hhi_values) if hhi_values else 0

    def _calculate_turnover(self, signals: pd.DataFrame) -> float:
        """Calculate average portfolio turnover."""
        if len(signals) <= 1:
            return 0

        turnover_values = []
        for i in range(1, len(signals)):
            prev_allocation = signals.iloc[i-1]
            curr_allocation = signals.iloc[i]

            # Calculate change in allocation
            change = abs(curr_allocation - prev_allocation).sum()
            turnover_values.append(change / 2)  # Divide by 2 for standard turnover calculation

        return np.mean(turnover_values) if turnover_values else 0

    def get_strategy_info(self) -> Dict:
        """Get detailed strategy information."""
        info = super().get_info()
        info.update({
            'description': 'Dual Momentum Strategy combining absolute and relative momentum',
            'lookback_days': self.lookback_days,
            'top_n_assets': self.top_n_assets,
            'minimum_positive_assets': self.minimum_positive_assets,
            'cash_ticker': self.cash_ticker,
            'include_cash_in_universe': self.include_cash_in_universe,
            'rebalancing_frequency': 'monthly',
            'risk_management': 'Built-in via absolute momentum filter'
        })
        return info