"""
Fama/French 5-factor model strategy implementation.

This strategy implements the Fama/French 5-factor model:
1. Market Risk (MKT) - Market excess returns
2. Size (SMB) - Small minus big market capitalization
3. Value (HML) - High minus low book-to-market ratio
4. Profitability (RMW) - Robust minus weak operating profitability
5. Investment (CMA) - Conservative minus aggressive investment

Strategy selects assets based on factor exposures and expected returns.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class FamaFrench5Strategy(BaseStrategy):
    """
    Fama/French 5-factor model strategy implementation.

    Strategy Logic:
    1. Calculate factor exposures for all assets in universe
    2. Estimate expected returns using factor model
    3. Select assets with highest expected returns
    4. Apply risk management constraints
    5. Rebalance monthly

    The 5 factors are:
    - MKT: Market excess return (market return - risk-free rate)
    - SMB: Small Minus Big (size factor)
    - HML: High Minus Low (value factor)
    - RMW: Robust Minus Weak (profitability factor)
    - CMA: Conservative Minus Aggressive (investment factor)
    """

    def __init__(self, name: str = "FamaFrench5",
                 lookback_days: int = 252,
                 top_n_assets: int = 5,
                 min_factor_score: float = 0.1,
                 max_portfolio_volatility: float = 0.20,
                 rebalance_frequency: str = "monthly",
                 risk_free_rate: float = 0.02,
                 **kwargs):
        """
        Initialize Fama/French 5-factor strategy.

        Args:
            name: Strategy name
            lookback_days: Lookback period for factor calculations (default: 252 = 1 year)
            top_n_assets: Number of top assets to select
            min_factor_score: Minimum factor score for asset selection
            max_portfolio_volatility: Maximum allowed portfolio volatility
            rebalance_frequency: Rebalancing frequency
            risk_free_rate: Annual risk-free rate for calculations
        """
        self.lookback_days = lookback_days
        self.top_n_assets = top_n_assets
        self.min_factor_score = min_factor_score
        self.max_portfolio_volatility = max_portfolio_volatility
        self.rebalance_frequency = rebalance_frequency
        self.risk_free_rate = risk_free_rate

        # Factor weights for scoring (can be calibrated)
        self.factor_weights = {
            'MKT': 0.30,  # Market factor
            'SMB': 0.15,  # Size factor
            'HML': 0.20,  # Value factor
            'RMW': 0.20,  # Profitability factor
            'CMA': 0.15   # Investment factor
        }

        super().__init__(name=name, **kwargs)

    def validate_parameters(self):
        """Validate strategy parameters."""
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")

        if self.top_n_assets <= 0:
            raise ValueError("top_n_assets must be positive")

        if not 0 <= self.max_portfolio_volatility <= 1:
            raise ValueError("max_portfolio_volatility must be between 0 and 1")

        if self.risk_free_rate < 0:
            raise ValueError("risk_free_rate must be non-negative")

        # Validate factor weights sum to 1
        weight_sum = sum(self.factor_weights.values())
        if not np.isclose(weight_sum, 1.0, atol=0.01):
            raise ValueError(f"Factor weights must sum to 1.0, got {weight_sum}")

    def generate_signals(self, price_data: Dict[str, pd.DataFrame],
                        start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Generate trading signals based on Fama/French 5-factor model.

        Args:
            price_data: Dictionary of price DataFrames for each symbol
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with trading signals (weight allocations for each symbol)
        """
        logger.info(f"Generating Fama/French 5-factor signals from {start_date} to {end_date}")

        # Create date range for rebalancing
        if self.rebalance_frequency == "monthly":
            date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
        else:
            date_range = pd.date_range(start=start_date, end=end_date, freq='QS')

        signals = pd.DataFrame(index=date_range, columns=list(price_data.keys()))

        for date in date_range:
            try:
                signal = self._generate_signal_for_date(price_data, date)
                signals.loc[date] = signal
            except Exception as e:
                logger.warning(f"Failed to generate signal for {date}: {e}")
                # Use previous signal or neutral position
                if len(signals) > 0:
                    prev_idx = signals.index.get_loc(date) - 1
                    if prev_idx >= 0:
                        signals.loc[date] = signals.iloc[prev_idx]
                    else:
                        signals.loc[date] = 0
                else:
                    signals.loc[date] = 0

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
        # Calculate factor scores for all assets
        factor_scores = {}

        for symbol, data in price_data.items():
            factor_score = self._calculate_factor_score(data, date)
            if factor_score is not None:
                factor_scores[symbol] = factor_score

        if not factor_scores:
            logger.warning(f"No factor scores calculated for {date}")
            return pd.Series(0, index=list(price_data.keys()))

        # Convert to DataFrame for easier manipulation
        scores_df = pd.DataFrame(list(factor_scores.items()),
                                columns=['symbol', 'factor_score']).set_index('symbol')

        # Filter by minimum factor score
        qualified_assets = scores_df[scores_df['factor_score'] >= self.min_factor_score]

        if len(qualified_assets) < self.top_n_assets:
            logger.debug(f"Only {len(qualified_assets)} assets qualified (need {self.top_n_assets}), "
                        f"reducing selection or moving to cash")
            if len(qualified_assets) == 0:
                return pd.Series(0, index=list(price_data.keys()))
            top_assets = qualified_assets
        else:
            # Select top N assets by factor score
            top_assets = qualified_assets.nlargest(self.top_n_assets, 'factor_score')

        # Equal weight selected assets
        weight_per_asset = 1.0 / len(top_assets)
        allocation = pd.Series(0.0, index=list(price_data.keys()), dtype=float)

        for symbol in top_assets.index:
            allocation[symbol] = weight_per_asset

        logger.debug(f"Selected {len(top_assets)} assets: {list(top_assets.index)}")

        return allocation

    def _calculate_factor_score(self, price_data: pd.DataFrame,
                               date: datetime) -> Optional[float]:
        """
        Calculate factor score for an asset.

        Args:
            price_data: Price DataFrame for the asset
            date: Date for factor calculation

        Returns:
            Factor score or None if insufficient data
        """
        try:
            # Get data up to the calculation date
            data_up_to_date = price_data[price_data.index <= date]

            if len(data_up_to_date) < self.lookback_days:
                logger.debug(f"Insufficient data for factor calculation: "
                           f"only {len(data_up_to_date)} days available, need {self.lookback_days}")
                return None

            # Calculate individual factor scores
            factor_scores = {}

            # Market factor (MKT) - excess returns over market
            market_score = self._calculate_market_factor(data_up_to_date)
            if market_score is not None:
                factor_scores['MKT'] = market_score

            # Size factor (SMB) - volatility as proxy for size
            size_score = self._calculate_size_factor(data_up_to_date)
            if size_score is not None:
                factor_scores['SMB'] = size_score

            # Value factor (HML) - momentum as proxy for value
            value_score = self._calculate_value_factor(data_up_to_date)
            if value_score is not None:
                factor_scores['HML'] = value_score

            # Profitability factor (RMW) - return consistency as proxy
            profitability_score = self._calculate_profitability_factor(data_up_to_date)
            if profitability_score is not None:
                factor_scores['RMW'] = profitability_score

            # Investment factor (CMA) - low volatility as proxy for conservative investment
            investment_score = self._calculate_investment_factor(data_up_to_date)
            if investment_score is not None:
                factor_scores['CMA'] = investment_score

            # Calculate weighted factor score
            total_score = 0
            for factor, score in factor_scores.items():
                if factor in self.factor_weights:
                    total_score += self.factor_weights[factor] * score

            logger.debug(f"Factor scores for {date}: {factor_scores}, total: {total_score:.4f}")
            return total_score

        except Exception as e:
            logger.warning(f"Error calculating factor score: {e}")
            return None

    def _calculate_market_factor(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate market factor (excess returns)."""
        try:
            returns = data['Close'].pct_change().dropna()
            if len(returns) < 20:
                return None

            # Calculate excess return over risk-free rate
            avg_return = returns.mean() * 252  # Annualize
            excess_return = avg_return - self.risk_free_rate

            # Normalize to 0-1 scale
            return max(0, min(1, (excess_return + 0.2) / 0.4))  # Map [-0.2, 0.2] to [0, 1]

        except Exception:
            return None

    def _calculate_size_factor(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate size factor (SMB proxy using volatility)."""
        try:
            returns = data['Close'].pct_change().dropna()
            if len(returns) < 20:
                return None

            # Use volatility as proxy for size (smaller stocks typically more volatile)
            volatility = returns.std() * np.sqrt(252)

            # Normalize to 0-1 scale (higher volatility = higher SMB score)
            return max(0, min(1, volatility / 0.4))  # Map 0-40% vol to [0, 1]

        except Exception:
            return None

    def _calculate_value_factor(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate value factor (HML proxy using momentum)."""
        try:
            # Use momentum as proxy for value
            short_term_return = data['Close'].pct_change(21).dropna()  # 1 month
            long_term_return = data['Close'].pct_change(252).dropna()  # 1 year

            if len(short_term_return) < 5 or len(long_term_return) < 5:
                return None

            # Value factor: long-term momentum relative to short-term
            if long_term_return.iloc[-1] > 0 and short_term_return.iloc[-1] < 0:
                # Good long-term, poor short-term = value opportunity
                return 0.8
            elif long_term_return.iloc[-1] > 0:
                # Good long-term momentum
                return 0.6
            else:
                # Poor long-term momentum
                return 0.2

        except Exception:
            return None

    def _calculate_profitability_factor(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate profitability factor (RMW proxy using return consistency)."""
        try:
            returns = data['Close'].pct_change().dropna()
            if len(returns) < 20:
                return None

            # Calculate profitability as positive return consistency
            positive_returns = returns[returns > 0]
            profitability = len(positive_returns) / len(returns)

            return profitability

        except Exception:
            return None

    def _calculate_investment_factor(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate investment factor (CMA proxy using low volatility)."""
        try:
            returns = data['Close'].pct_change().dropna()
            if len(returns) < 20:
                return None

            # Conservative investment = low volatility
            volatility = returns.std() * np.sqrt(252)

            # Low volatility = high CMA score
            return max(0, 1 - (volatility / 0.5))  # Map 0-50% vol to [1, 0]

        except Exception:
            return None

    def get_factor_exposures(self, price_data: Dict[str, pd.DataFrame],
                           signals: pd.DataFrame) -> Dict:
        """
        Calculate portfolio factor exposures.

        Args:
            price_data: Price data dictionary
            signals: Trading signals DataFrame

        Returns:
            Dictionary with factor exposure metrics
        """
        try:
            factor_exposures = {}

            # Calculate factor exposures for each date
            for date in signals.index:
                if date not in signals.index:
                    continue

                allocation = signals.loc[date]
                date_exposures = {}

                for symbol, weight in allocation.items():
                    if weight > 0 and symbol in price_data:
                        # Get factor exposures for this symbol
                        data_up_to_date = price_data[symbol][price_data[symbol].index <= date]
                        if len(data_up_to_date) >= self.lookback_days:
                            symbol_exposures = self._calculate_factor_exposures(data_up_to_date)
                            for factor, exposure in symbol_exposures.items():
                                date_exposures[factor] = date_exposures.get(factor, 0) + weight * exposure

                factor_exposures[date] = date_exposures

            return factor_exposures

        except Exception as e:
            logger.error(f"Error calculating factor exposures: {e}")
            return {}

    def _calculate_factor_exposures(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate factor exposures for a single asset."""
        exposures = {}

        # Market factor exposure
        returns = data['Close'].pct_change().dropna()
        if len(returns) >= 20:
            beta = returns.rolling(60).cov(returns) / returns.rolling(60).var()
            if len(beta) > 0 and not pd.isna(beta.iloc[-1]):
                exposures['MKT'] = beta.iloc[-1]

        # Other factor exposures (simplified)
        volatility = returns.std() * np.sqrt(252) if len(returns) >= 20 else 0.2
        exposures['SMB'] = min(2.0, volatility / 0.2)  # Size exposure
        exposures['HML'] = 1.0  # Default value exposure
        exposures['RMW'] = 1.0  # Default profitability exposure
        exposures['CMA'] = max(0, 1.0 - volatility)  # Conservative investment

        return exposures

    def get_strategy_info(self) -> Dict:
        """Get detailed strategy information."""
        info = super().get_info()
        info.update({
            'description': 'Fama/French 5-factor model strategy',
            'lookback_days': self.lookback_days,
            'top_n_assets': self.top_n_assets,
            'min_factor_score': self.min_factor_score,
            'max_portfolio_volatility': self.max_portfolio_volatility,
            'rebalance_frequency': self.rebalance_frequency,
            'risk_free_rate': self.risk_free_rate,
            'factors': list(self.factor_weights.keys()),
            'factor_weights': self.factor_weights.copy(),
            'risk_management': 'Factor-based selection with volatility constraints'
        })
        return info