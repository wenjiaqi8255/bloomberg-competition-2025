"""
Performance Metrics Utilities

Consolidated performance calculation methods extracted from multiple locations
to eliminate duplication and provide consistent metrics across the system.

Replaces:
- backtesting/metrics/calculator.py (partial)
- strategy-specific monitor_performance methods
- orchestration/reporter.py performance calculations
- individual strategy performance calculations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from scipy import stats

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """
    Unified performance metrics calculator.

    Provides static methods for calculating various performance metrics
    consistently across the entire trading system.
    """

    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                     periods_per_year: int = 252) -> float:
        """
        Calculate annualized Sharpe ratio.

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year (252 for daily, 12 for monthly)

        Returns:
            Sharpe ratio
        """
        if len(returns) < 2 or returns.std() == 0:
            return 0.0

        daily_risk_free = (1 + risk_free_rate) ** (1/periods_per_year) - 1
        excess_returns = returns - daily_risk_free

        annual_excess_return = excess_returns.mean() * periods_per_year
        annual_excess_vol = excess_returns.std() * np.sqrt(periods_per_year)

        return annual_excess_return / annual_excess_vol if annual_excess_vol > 0 else 0.0

    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02,
                      periods_per_year: int = 252) -> float:
        """
        Calculate Sortino ratio (downside risk-adjusted return).

        Args:
            returns: Return series
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        daily_risk_free = (1 + risk_free_rate) ** (1/periods_per_year) - 1
        excess_returns = returns - daily_risk_free
        negative_returns = excess_returns[excess_returns < 0]

        if len(negative_returns) == 0:
            return float('inf')

        downside_deviation = negative_returns.std()
        annual_excess_return = excess_returns.mean() * periods_per_year
        annual_downside_dev = downside_deviation * np.sqrt(periods_per_year)

        return annual_excess_return / annual_downside_dev if annual_downside_dev > 0 else 0.0

    @staticmethod
    def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate Calmar ratio (return / maximum drawdown).

        Args:
            returns: Return series
            periods_per_year: Number of periods per year

        Returns:
            Calmar ratio
        """
        if len(returns) < 2:
            return 0.0

        max_drawdown = PerformanceMetrics.max_drawdown(returns)
        if max_drawdown == 0:
            return float('inf')

        total_return = (1 + returns).prod() - 1
        years = len(returns) / periods_per_year
        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        return annualized_return / abs(max_drawdown)

    @staticmethod
    def information_ratio(returns: pd.Series, benchmark_returns: pd.Series,
                         periods_per_year: int = 252) -> float:
        """
        Calculate information ratio (active return / tracking error).

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            periods_per_year: Number of periods per year

        Returns:
            Information ratio
        """
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0

        # Align returns
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 2:
            return 0.0

        returns_aligned = returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]

        active_returns = returns_aligned - benchmark_aligned
        tracking_error = active_returns.std() * np.sqrt(periods_per_year)

        if tracking_error == 0:
            return 0.0

        active_return_annual = active_returns.mean() * periods_per_year
        return active_return_annual / tracking_error

    @staticmethod
    def alpha_beta(returns: pd.Series, benchmark_returns: pd.Series,
                   risk_free_rate: float = 0.02, periods_per_year: int = 252) -> Tuple[float, float]:
        """
        Calculate Jensen's alpha and beta.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Tuple of (alpha, beta)
        """
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0, 1.0

        # Align returns
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 2:
            return 0.0, 1.0

        returns_aligned = returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]

        # Calculate beta
        covariance = np.cov(returns_aligned, benchmark_aligned)[0, 1]
        benchmark_variance = np.var(benchmark_aligned)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

        # Calculate alpha
        daily_risk_free = (1 + risk_free_rate) ** (1/periods_per_year) - 1
        portfolio_annual = returns_aligned.mean() * periods_per_year
        benchmark_annual = benchmark_aligned.mean() * periods_per_year

        alpha = portfolio_annual - (risk_free_rate + beta * (benchmark_annual - risk_free_rate))

        return alpha, beta

    @staticmethod
    def max_drawdown(returns: pd.Series) -> float:
        """
        Calculate maximum drawdown.

        Args:
            returns: Return series

        Returns:
            Maximum drawdown (negative value)
        """
        if len(returns) < 2:
            return 0.0

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        return drawdown.min()

    @staticmethod
    def volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized volatility.

        Args:
            returns: Return series
            periods_per_year: Number of periods per year

        Returns:
            Annualized volatility
        """
        if len(returns) < 2:
            return 0.0

        return returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def var(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Return series
            confidence: Confidence level (0.95 for 95% VaR)

        Returns:
            VaR at specified confidence level
        """
        if len(returns) < 2:
            return 0.0

        return returns.quantile(1 - confidence)

    @staticmethod
    def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional Value at Risk).

        Args:
            returns: Return series
            confidence: Confidence level

        Returns:
            Expected shortfall at specified confidence level
        """
        if len(returns) < 2:
            return 0.0

        var_threshold = PerformanceMetrics.var(returns, confidence)
        tail_returns = returns[returns <= var_threshold]

        return tail_returns.mean() if len(tail_returns) > 0 else 0.0

    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """
        Calculate win rate (proportion of positive returns).

        Args:
            returns: Return series

        Returns:
            Win rate as a decimal
        """
        if len(returns) == 0:
            return 0.0

        return (returns > 0).mean()

    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            returns: Return series

        Returns:
            Profit factor
        """
        if len(returns) == 0:
            return 1.0

        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())

        return positive_returns / negative_returns if negative_returns != 0 else float('inf')

    @staticmethod
    def total_return(returns: pd.Series) -> float:
        """
        Calculate total cumulative return.

        Args:
            returns: Return series

        Returns:
            Total return as a decimal
        """
        if len(returns) == 0:
            return 0.0

        return (1 + returns).prod() - 1

    @staticmethod
    def annualized_return(returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calculate annualized return.

        Args:
            returns: Return series
            periods_per_year: Number of periods per year

        Returns:
            Annualized return as a decimal
        """
        if len(returns) < 2:
            return 0.0

        total_return = PerformanceMetrics.total_return(returns)
        years = len(returns) / periods_per_year

        return (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    @staticmethod
    def capture_ratios(returns: pd.Series, benchmark_returns: pd.Series) -> Tuple[float, float]:
        """
        Calculate up-market and down-market capture ratios.

        Args:
            returns: Portfolio returns
            benchmark_returns: Benchmark returns

        Returns:
            Tuple of (up_capture, down_capture)
        """
        if len(returns) < 2 or len(benchmark_returns) < 2:
            return 0.0, 0.0

        # Align returns
        common_index = returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 2:
            return 0.0, 0.0

        returns_aligned = returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]

        up_periods = benchmark_aligned > 0
        down_periods = benchmark_aligned < 0

        # Up-market capture
        if up_periods.sum() > 0:
            portfolio_up = returns_aligned[up_periods].mean()
            benchmark_up = benchmark_aligned[up_periods].mean()
            up_capture = portfolio_up / benchmark_up if benchmark_up > 0 else 0
        else:
            up_capture = 0

        # Down-market capture
        if down_periods.sum() > 0:
            portfolio_down = returns_aligned[down_periods].mean()
            benchmark_down = benchmark_aligned[down_periods].mean()
            down_capture = portfolio_down / benchmark_down if benchmark_down < 0 else 0
        else:
            down_capture = 0

        return up_capture, down_capture

    @staticmethod
    def calculate_all_metrics(returns: pd.Series,
                            benchmark_returns: Optional[pd.Series] = None,
                            risk_free_rate: float = 0.02,
                            periods_per_year: int = 252) -> Dict[str, float]:
        """
        Calculate all comprehensive performance metrics.

        Args:
            returns: Portfolio returns
            benchmark_returns: Optional benchmark returns
            risk_free_rate: Annual risk-free rate
            periods_per_year: Number of periods per year

        Returns:
            Dictionary with all calculated metrics
        """
        if len(returns) < 2:
            logger.warning("Insufficient data for performance calculation")
            return PerformanceMetrics._empty_metrics()

        try:
            metrics = {}

            # Basic return metrics
            metrics['total_return'] = PerformanceMetrics.total_return(returns)
            metrics['annualized_return'] = PerformanceMetrics.annualized_return(returns, periods_per_year)
            metrics['volatility'] = PerformanceMetrics.volatility(returns, periods_per_year)

            # Risk-adjusted ratios
            metrics['sharpe_ratio'] = PerformanceMetrics.sharpe_ratio(returns, risk_free_rate, periods_per_year)
            metrics['sortino_ratio'] = PerformanceMetrics.sortino_ratio(returns, risk_free_rate, periods_per_year)
            metrics['calmar_ratio'] = PerformanceMetrics.calmar_ratio(returns, periods_per_year)
            metrics['max_drawdown'] = PerformanceMetrics.max_drawdown(returns)

            # Risk metrics
            metrics['var_95'] = PerformanceMetrics.var(returns, 0.95)
            metrics['var_99'] = PerformanceMetrics.var(returns, 0.99)
            metrics['expected_shortfall_95'] = PerformanceMetrics.expected_shortfall(returns, 0.95)
            metrics['expected_shortfall_99'] = PerformanceMetrics.expected_shortfall(returns, 0.99)

            # Trading metrics
            metrics['win_rate'] = PerformanceMetrics.win_rate(returns)
            metrics['profit_factor'] = PerformanceMetrics.profit_factor(returns)

            # Relative metrics (if benchmark provided)
            if benchmark_returns is not None:
                alpha, beta = PerformanceMetrics.alpha_beta(returns, benchmark_returns, risk_free_rate, periods_per_year)
                metrics['alpha'] = alpha
                metrics['beta'] = beta
                metrics['information_ratio'] = PerformanceMetrics.information_ratio(returns, benchmark_returns, periods_per_year)
                metrics['tracking_error'] = (returns - benchmark_returns).std() * np.sqrt(periods_per_year)

                up_capture, down_capture = PerformanceMetrics.capture_ratios(returns, benchmark_returns)
                metrics['up_capture'] = up_capture
                metrics['down_capture'] = down_capture
            else:
                # Default values when no benchmark
                metrics.update({
                    'alpha': 0.0,
                    'beta': 1.0,
                    'information_ratio': 0.0,
                    'tracking_error': PerformanceMetrics.volatility(returns, periods_per_year),
                    'up_capture': 0.0,
                    'down_capture': 0.0
                })

            return metrics

        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics._empty_metrics()

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        """Return empty metrics dictionary with default values."""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'var_95': 0.0,
            'var_99': 0.0,
            'expected_shortfall_95': 0.0,
            'expected_shortfall_99': 0.0,
            'win_rate': 0.0,
            'profit_factor': 1.0,
            'alpha': 0.0,
            'beta': 1.0,
            'information_ratio': 0.0,
            'tracking_error': 0.0,
            'up_capture': 0.0,
            'down_capture': 0.0
        }

    @staticmethod
    def format_metrics(metrics: Dict[str, float]) -> Dict[str, str]:
        """
        Format metrics dictionary for display.

        Args:
            metrics: Dictionary of calculated metrics

        Returns:
            Dictionary with formatted string values
        """
        formatted = {}

        for key, value in metrics.items():
            if 'ratio' in key or key in ['alpha', 'beta', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio', 'information_ratio']:
                formatted[key] = f"{value:.3f}"
            elif 'rate' in key or key in ['win_rate', 'up_capture', 'down_capture']:
                formatted[key] = f"{value:.1%}"
            elif key in ['total_return', 'annualized_return', 'volatility', 'max_drawdown',
                         'var_95', 'var_99', 'expected_shortfall_95', 'expected_shortfall_99',
                         'tracking_error']:
                formatted[key] = f"{value:.2%}"
            else:
                formatted[key] = f"{value:.4f}"

        return formatted