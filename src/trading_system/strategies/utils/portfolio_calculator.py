"""
Portfolio Calculator Utilities

算portfolio的指标
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from ...utils.data_utils import DataProcessor

logger = logging.getLogger(__name__)


class PortfolioCalculator:
    """
    Unified portfolio calculation utilities.

    Provides static methods for calculating portfolio returns, turnover,
    and other portfolio-related metrics consistently across the entire trading system.
    """

    @staticmethod
    def calculate_portfolio_returns(price_data: Dict[str, pd.DataFrame],
                                  signals: pd.DataFrame) -> pd.Series:
        """
        Calculate portfolio returns based on signals and price data.

        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            signals: DataFrame with portfolio weights (symbols in columns, dates in index)

        Returns:
            Series of portfolio returns indexed by date
        """
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
                        # Get price change over the period using DataProcessor
                        symbol_data = price_data[symbol]
                        if date in symbol_data.index and prev_date in symbol_data.index:
                            asset_returns = DataProcessor.calculate_returns(
                                symbol_data.loc[[prev_date, date], ['Close']]
                            )
                            if not asset_returns.empty:
                                asset_return = asset_returns.iloc[-1, 0]
                                weighted_return = weight * asset_return
                                period_returns.append(weighted_return)
                    except Exception as e:
                        logger.warning(f"Error calculating return for {symbol}: {e}")

            if period_returns:
                returns[date] = sum(period_returns)

        return returns

    @staticmethod
    def calculate_turnover(signals: pd.DataFrame) -> float:
        """
        Calculate average portfolio turnover.

        Args:
            signals: DataFrame with portfolio weights (symbols in columns, dates in index)

        Returns:
            Average turnover rate
        """
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

    @staticmethod
    def calculate_concentration_risk(signals: pd.DataFrame) -> float:
        """
        Calculate Herfindahl-Hirschman Index for concentration risk.

        Args:
            signals: DataFrame with portfolio weights

        Returns:
            Concentration risk metric (HHI)
        """
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

    
    @staticmethod
    def calculate_portfolio_metrics(price_data: Dict[str, pd.DataFrame],
                                 signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive portfolio metrics.

        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            signals: DataFrame with portfolio weights

        Returns:
            Dictionary with portfolio metrics
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = PortfolioCalculator.calculate_portfolio_returns(price_data, signals)

            if portfolio_returns.empty:
                return {}

            # Calculate basic statistics
            total_return = (1 + portfolio_returns).prod() - 1
            annualized_return = (1 + portfolio_returns.mean()) ** 252 - 1
            volatility = portfolio_returns.std() * np.sqrt(252)

            # Calculate risk metrics
            max_drawdown = DataProcessor.calculate_drawdown(portfolio_returns)['drawdown'].min()

            # Calculate portfolio-level metrics
            turnover = PortfolioCalculator.calculate_turnover(signals)
            concentration_risk = PortfolioCalculator.calculate_concentration_risk(signals)

            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'turnover': turnover,
                'concentration_risk': concentration_risk,
                'number_of_periods': len(portfolio_returns),
                'portfolio_returns': portfolio_returns
            }

        except Exception as e:
            logger.error(f"Error calculating portfolio metrics: {e}")
            return {}

    @staticmethod
    def calculate_position_metrics(signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate position-related metrics.

        Args:
            signals: DataFrame with portfolio weights

        Returns:
            Dictionary with position metrics
        """
        if signals.empty:
            return {}

        # Calculate position counts and weights
        position_counts = []
        avg_weights = []
        max_weights = []

        for _, allocation in signals.iterrows():
            non_zero_positions = allocation[allocation > 0]
            position_counts.append(len(non_zero_positions))

            if len(non_zero_positions) > 0:
                avg_weights.append(non_zero_positions.mean())
                max_weights.append(non_zero_positions.max())
            else:
                avg_weights.append(0)
                max_weights.append(0)

        return {
            'avg_number_of_positions': np.mean(position_counts),
            'max_number_of_positions': np.max(position_counts),
            'min_number_of_positions': np.min(position_counts),
            'avg_position_weight': np.mean(avg_weights),
            'max_position_weight': np.max(max_weights),
            'avg_concentration': np.mean(max_weights)
        }

    @staticmethod
    def calculate_signal_quality(signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate signal quality metrics.

        Args:
            signals: DataFrame with portfolio weights

        Returns:
            Dictionary with signal quality metrics
        """
        if signals.empty:
            return {}

        # Signal intensity metrics
        signal_intensity = signals.abs().mean()
        signal_consistency = signals.notna().mean()

        # Signal frequency (how often signals change)
        signal_changes = 0
        for i in range(1, len(signals)):
            prev_signals = signals.iloc[i-1] > 0
            curr_signals = signals.iloc[i] > 0
            signal_changes += (prev_signals != curr_signals).sum()

        signal_frequency = signal_changes / (len(signals) - 1) / len(signals.columns)

        return {
            'avg_signal_intensity': signal_intensity.mean(),
            'max_signal_intensity': signal_intensity.max(),
            'avg_signal_consistency': signal_consistency.mean(),
            'signal_frequency': signal_frequency,
            'total_signal_changes': signal_changes
        }

    @staticmethod
    def analyze_portfolio_composition(price_data: Dict[str, pd.DataFrame],
                                   signals: pd.DataFrame,
                                   lookback_days: int = 252) -> Dict[str, Any]:
        """
        Analyze portfolio composition over time.

        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            signals: DataFrame with portfolio weights
            lookback_days: Lookback period for calculations

        Returns:
            Dictionary with composition analysis
        """
        try:
            portfolio_returns = PortfolioCalculator.calculate_portfolio_returns(price_data, signals)

            if portfolio_returns.empty:
                return {}

            # Calculate rolling metrics
            rolling_returns = portfolio_returns.rolling(window=min(lookback_days, len(portfolio_returns)))

            # Calculate contribution analysis
            contributions = {}
            for symbol in signals.columns:
                symbol_weights = signals[symbol]
                if symbol in price_data:
                    symbol_returns = DataProcessor.calculate_returns(
                        price_data[symbol][['Close']]
                    ).iloc[:, 0]

                    # Align and calculate contributions
                    aligned_weights, aligned_returns = symbol_weights.align(symbol_returns, join='inner')
                    contributions[symbol] = (aligned_weights * aligned_returns).sum()

            return {
                'portfolio_returns': portfolio_returns,
                'rolling_volatility': rolling_returns.std() * np.sqrt(252),
                'rolling_sharpe': rolling_returns.mean() / rolling_returns.std() * np.sqrt(252),
                'asset_contributions': contributions,
                'top_contributors': sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:5],
                'worst_contributors': sorted(contributions.items(), key=lambda x: x[1])[:5]
            }

        except Exception as e:
            logger.error(f"Error analyzing portfolio composition: {e}")
            return {}

    @staticmethod
    def create_portfolio_snapshots(price_data: Dict[str, pd.DataFrame],
                                 signals: pd.DataFrame,
                                 initial_capital: float = 1000000) -> pd.DataFrame:
        """
        Create portfolio value snapshots over time.

        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            signals: DataFrame with portfolio weights
            initial_capital: Initial portfolio capital

        Returns:
            DataFrame with portfolio snapshots
        """
        try:
            portfolio_returns = PortfolioCalculator.calculate_portfolio_returns(price_data, signals)

            if portfolio_returns.empty:
                return pd.DataFrame()

            # Calculate portfolio value
            portfolio_value = initial_capital * (1 + portfolio_returns).cumprod()

            # Create snapshots DataFrame
            snapshots = pd.DataFrame({
                'date': portfolio_returns.index,
                'portfolio_value': portfolio_value.values,
                'portfolio_return': portfolio_returns.values,
                'cumulative_return': (portfolio_value / initial_capital - 1).values,
                'daily_return': portfolio_returns.values
            })

            # Add rolling metrics
            snapshots['rolling_20d_vol'] = portfolio_returns.rolling(20).std() * np.sqrt(252)
            snapshots['rolling_20d_return'] = portfolio_returns.rolling(20).mean() * 252

            return snapshots.set_index('date')

        except Exception as e:
            logger.error(f"Error creating portfolio snapshots: {e}")
            return pd.DataFrame()