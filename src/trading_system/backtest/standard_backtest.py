"""
Standard backtest engine with time-weighted returns calculation
and performance metrics computation.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

logger = logging.getLogger(__name__)


class StandardBacktest:
    """
    Standard backtest engine for quantitative trading strategies.

    Features:
    - Time-weighted returns calculation
    - Benchmark-relative performance
    - Comprehensive risk metrics
    - Transaction cost modeling
    - Support for multiple rebalancing frequencies
    """

    def __init__(self, initial_capital: float = 1_000_000,
                 transaction_cost: float = 0.001,
                 benchmark_symbol: str = 'SPY'):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital for the strategy
            transaction_cost: Transaction cost as fraction (0.001 = 0.1%)
            benchmark_symbol: Symbol for benchmark comparison
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.benchmark_symbol = benchmark_symbol

        # Performance tracking
        self.portfolio_values = []
        self.cash_values = []
        self.positions = {}
        self.trades = []
        self.performance_metrics = {}

    def run_backtest(self, strategy_signals: pd.DataFrame,
                    price_data: Dict[str, pd.DataFrame],
                    benchmark_data: pd.DataFrame,
                    start_date: datetime,
                    end_date: datetime,
                    rebalance_frequency: str = 'monthly') -> Dict:
        """
        Run a complete backtest.

        Args:
            strategy_signals: DataFrame with strategy signals
            price_data: Dictionary of price DataFrames for each symbol
            benchmark_data: Benchmark price DataFrame
            start_date: Backtest start date
            end_date: Backtest end date
            rebalance_frequency: 'daily', 'weekly', 'monthly', 'quarterly'

        Returns:
            Dictionary with backtest results and performance metrics
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")

        # Initialize portfolio state
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = {symbol: 0 for symbol in strategy_signals.columns}

        # Generate rebalance dates
        rebalance_dates = self._generate_rebalance_dates(start_date, end_date, rebalance_frequency)

        # Track portfolio evolution
        portfolio_history = []
        positions_history = []
        cash_history = []

        for i, rebalance_date in enumerate(rebalance_dates):
            logger.debug(f"Rebalancing on {rebalance_date}")

            # Get current prices
            current_prices = self._get_current_prices(price_data, rebalance_date)

            if not current_prices:
                logger.warning(f"No price data available for {rebalance_date}")
                continue

            # Get target positions from strategy signals
            target_positions = self._get_target_positions(
                strategy_signals, rebalance_date, portfolio_value, current_prices
            )

            # Execute trades
            trades_executed, trade_costs = self._execute_trades(
                positions, target_positions, current_prices, cash
            )

            # Update portfolio state
            cash -= trade_costs
            positions.update(target_positions)

            # Calculate current portfolio value
            current_portfolio_value = cash + sum(
                positions[symbol] * current_prices.get(symbol, 0)
                for symbol in positions
            )

            # Record state
            portfolio_history.append({
                'date': rebalance_date,
                'portfolio_value': current_portfolio_value,
                'cash': cash,
                'positions': positions.copy(),
                'trades': trades_executed
            })

            logger.debug(f"Portfolio value: ${current_portfolio_value:,.2f}, "
                        f"Cash: ${cash:,.2f}")

        # Convert to DataFrame for analysis
        portfolio_df = pd.DataFrame(portfolio_history)
        portfolio_df.set_index('date', inplace=True)

        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            portfolio_df, benchmark_data, start_date, end_date
        )

        # Prepare results
        results = {
            'portfolio_history': portfolio_df,
            'performance_metrics': performance_metrics,
            'trades': self.trades,
            'initial_capital': self.initial_capital,
            'final_value': portfolio_df['portfolio_value'].iloc[-1],
            'total_return': performance_metrics.get('total_return', 0),
            'annualized_return': performance_metrics.get('annualized_return', 0),
            'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
            'max_drawdown': performance_metrics.get('max_drawdown', 0)
        }

        logger.info(f"Backtest completed. Final value: ${results['final_value']:,.2f}")
        logger.info(f"Total return: {results['total_return']:.2%}")
        logger.info(f"Annualized return: {results['annualized_return']:.2%}")
        logger.info(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
        logger.info(f"Max drawdown: {results['max_drawdown']:.2%}")

        return results

    def _generate_rebalance_dates(self, start_date: datetime, end_date: datetime,
                                 frequency: str) -> List[datetime]:
        """Generate rebalancing dates based on frequency."""
        if frequency == 'daily':
            dates = pd.date_range(start_date, end_date, freq='D')
            # Keep only business days
            dates = dates[dates.dayofweek < 5]
        elif frequency == 'weekly':
            dates = pd.date_range(start_date, end_date, freq='W-MON')
        elif frequency == 'monthly':
            dates = pd.date_range(start_date, end_date, freq='MS')
        elif frequency == 'quarterly':
            dates = pd.date_range(start_date, end_date, freq='QS')
        else:
            raise ValueError(f"Unknown rebalancing frequency: {frequency}")

        # Convert to datetime and ensure within range
        dates = [pd.to_datetime(d) for d in dates if start_date <= pd.to_datetime(d) <= end_date]

        return dates

    def _get_current_prices(self, price_data: Dict[str, pd.DataFrame],
                           date: datetime) -> Dict[str, float]:
        """Get current prices for all symbols on a given date."""
        prices = {}

        for symbol, data in price_data.items():
            if date in data.index:
                prices[symbol] = data.loc[date, 'Close']
            else:
                # Use forward fill to handle non-trading days
                try:
                    data_up_to_date = data[data.index <= date]
                    if not data_up_to_date.empty:
                        prices[symbol] = data_up_to_date['Close'].iloc[-1]
                    else:
                        logger.warning(f"No price data for {symbol} before {date}")
                except Exception as e:
                    logger.warning(f"Error getting price for {symbol} on {date}: {e}")

        return prices

    def _get_target_positions(self, strategy_signals: pd.DataFrame,
                             date: datetime, portfolio_value: float,
                             current_prices: Dict[str, float]) -> Dict[str, float]:
        """Get target positions based on strategy signals."""
        target_positions = {}

        # Get signals for current date
        if date in strategy_signals.index:
            signals = strategy_signals.loc[date]

            # Calculate target allocation based on signals
            for symbol in signals.index:
                if pd.notna(signals[symbol]) and signals[symbol] != 0:
                    target_weight = signals[symbol]
                    target_value = portfolio_value * target_weight

                    if symbol in current_prices and current_prices[symbol] > 0:
                        target_positions[symbol] = target_value / current_prices[symbol]

        return target_positions

    def _execute_trades(self, current_positions: Dict[str, float],
                       target_positions: Dict[str, float],
                       current_prices: Dict[str, float],
                       cash: float) -> Tuple[List[Dict], float]:
        """Execute trades to reach target positions."""
        trades = []
        total_cost = 0

        # Calculate trades needed
        all_symbols = set(current_positions.keys()) | set(target_positions.keys())

        for symbol in all_symbols:
            current_pos = current_positions.get(symbol, 0)
            target_pos = target_positions.get(symbol, 0)

            if abs(current_pos - target_pos) > 1e-6:  # Significant position change
                trade_quantity = target_pos - current_pos
                trade_value = abs(trade_quantity * current_prices.get(symbol, 0))
                trade_cost = trade_value * self.transaction_cost

                trade_record = {
                    'symbol': symbol,
                    'quantity': trade_quantity,
                    'price': current_prices.get(symbol, 0),
                    'value': trade_value,
                    'cost': trade_cost
                }

                trades.append(trade_record)
                total_cost += trade_cost

        self.trades.extend(trades)

        return trades, total_cost

    def _calculate_performance_metrics(self, portfolio_df: pd.DataFrame,
                                     benchmark_data: pd.DataFrame,
                                     start_date: datetime,
                                     end_date: datetime) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        metrics = {}

        # Portfolio returns
        portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()

        # Benchmark returns
        if benchmark_data is not None and not benchmark_data.empty:
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()

            # Align dates
            aligned_returns = pd.DataFrame({
                'portfolio': portfolio_returns,
                'benchmark': benchmark_returns
            }).dropna()

            if not aligned_returns.empty:
                # Calculate excess returns
                excess_returns = aligned_returns['portfolio'] - aligned_returns['benchmark']

                # Alpha (annualized)
                metrics['alpha'] = excess_returns.mean() * 252

                # Beta
                if aligned_returns['benchmark'].std() != 0:
                    metrics['beta'] = (
                        aligned_returns['portfolio'].cov(aligned_returns['benchmark']) /
                        aligned_returns['benchmark'].var()
                    )
                else:
                    metrics['beta'] = 0

                # Information ratio
                if excess_returns.std() != 0:
                    metrics['information_ratio'] = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
                else:
                    metrics['information_ratio'] = 0

                # Tracking error
                metrics['tracking_error'] = excess_returns.std() * np.sqrt(252)

        # Basic return metrics
        if not portfolio_returns.empty:
            # Total return
            metrics['total_return'] = (
                (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0]) - 1
            )

            # Annualized return
            days_held = (end_date - start_date).days
            if days_held > 0:
                years_held = days_held / 365.25
                metrics['annualized_return'] = (1 + metrics['total_return']) ** (1 / years_held) - 1

            # Volatility (annualized)
            metrics['volatility'] = portfolio_returns.std() * np.sqrt(252)

            # Sharpe ratio (assuming risk-free rate of 0 for simplicity)
            if metrics['volatility'] != 0:
                metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility']
            else:
                metrics['sharpe_ratio'] = 0

            # Maximum drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = drawdown.min()

            # Calmar ratio
            if metrics['max_drawdown'] != 0:
                metrics['calmar_ratio'] = abs(metrics['annualized_return'] / metrics['max_drawdown'])
            else:
                metrics['calmar_ratio'] = 0

            # Win rate
            metrics['win_rate'] = (portfolio_returns > 0).mean()

            # Profit factor
            positive_returns = portfolio_returns[portfolio_returns > 0].sum()
            negative_returns = abs(portfolio_returns[portfolio_returns < 0].sum())
            if negative_returns != 0:
                metrics['profit_factor'] = positive_returns / negative_returns
            else:
                metrics['profit_factor'] = float('inf')

        return metrics

    def get_performance_summary(self) -> Dict[str, Union[str, float]]:
        """Get a formatted performance summary."""
        if not self.performance_metrics:
            return {"status": "No backtest results available"}

        metrics = self.performance_metrics

        summary = {
            "Initial Capital": f"${self.initial_capital:,.0f}",
            "Final Value": f"${metrics.get('final_value', 0):,.0f}",
            "Total Return": f"{metrics.get('total_return', 0):.2%}",
            "Annualized Return": f"{metrics.get('annualized_return', 0):.2%}",
            "Volatility": f"{metrics.get('volatility', 0):.2%}",
            "Sharpe Ratio": f"{metrics.get('sharpe_ratio', 0):.3f}",
            "Max Drawdown": f"{metrics.get('max_drawdown', 0):.2%}",
            "Win Rate": f"{metrics.get('win_rate', 0):.1%}",
            "Alpha": f"{metrics.get('alpha', 0):.2%}",
            "Beta": f"{metrics.get('beta', 0):.3f}",
            "Information Ratio": f"{metrics.get('information_ratio', 0):.3f}",
            "Tracking Error": f"{metrics.get('tracking_error', 0):.2%}",
            "Calmar Ratio": f"{metrics.get('calmar_ratio', 0):.3f}"
        }

        return summary