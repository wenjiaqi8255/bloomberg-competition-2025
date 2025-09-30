"""
Backtest Results - Unified Data Structure

Simplified results class that provides backward compatibility
while maintaining clean, focused design.
"""

import logging
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """
    Unified backtest results with backward compatibility.

    Provides clean access to all backtest results while maintaining
    compatibility with existing code that expects specific attributes.
    """

    # Core time series data
    portfolio_values: pd.Series
    daily_returns: pd.Series
    benchmark_returns: Optional[pd.Series] = None

    # Performance metrics (calculated, not stored)
    performance_metrics: Dict[str, float] = field(default_factory=dict)

    # Trading information
    trades: List[Any] = field(default_factory=list)  # Will be Trade objects
    transaction_costs: float = 0.0

    # Metadata
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 0.0
    final_value: float = 0.0

    # Additional analytics
    positions_history: List[Any] = field(default_factory=list)  # Portfolio snapshots
    turnover_rate: float = 0.0

    def __post_init__(self):
        """Initialize results and calculate derived values."""
        self._validate_data()
        self._calculate_derived_values()

    def _validate_data(self):
        """Validate input data."""
        if self.portfolio_values is None or len(self.portfolio_values) == 0:
            raise ValueError("portfolio_values cannot be empty")

        if self.daily_returns is None or len(self.daily_returns) == 0:
            raise ValueError("daily_returns cannot be empty")

        # Ensure portfolio_values and daily_returns have same index
        if not self.portfolio_values.index.equals(self.daily_returns.index):
            logger.warning("portfolio_values and daily_returns have different indexes")

        # Set final value if not provided
        if self.final_value == 0.0 and len(self.portfolio_values) > 0:
            self.final_value = self.portfolio_values.iloc[-1]

        # Set dates from data if not provided
        if not self.start_date and len(self.portfolio_values) > 0:
            self.start_date = self.portfolio_values.index[0] if hasattr(self.portfolio_values.index[0], 'date') else datetime.now()

        if not self.end_date and len(self.portfolio_values) > 0:
            self.end_date = self.portfolio_values.index[-1] if hasattr(self.portfolio_values.index[-1], 'date') else datetime.now()

    def _calculate_derived_values(self):
        """Calculate derived values if not already present."""
        if not self.performance_metrics:
            self.performance_metrics = self._calculate_basic_metrics()

    @property
    def total_return(self) -> float:
        """Get total return."""
        if self.initial_capital > 0:
            return (self.final_value - self.initial_capital) / self.initial_capital
        return self.performance_metrics.get('total_return', 0.0)

    @property
    def annualized_return(self) -> float:
        """Get annualized return."""
        return self.performance_metrics.get('annualized_return', 0.0)

    @property
    def sharpe_ratio(self) -> float:
        """Get Sharpe ratio."""
        return self.performance_metrics.get('sharpe_ratio', 0.0)

    @property
    def sortino_ratio(self) -> float:
        """Get Sortino ratio."""
        return self.performance_metrics.get('sortino_ratio', 0.0)

    @property
    def calmar_ratio(self) -> float:
        """Get Calmar ratio."""
        return self.performance_metrics.get('calmar_ratio', 0.0)

    @property
    def max_drawdown(self) -> float:
        """Get maximum drawdown."""
        return self.performance_metrics.get('max_drawdown', 0.0)

    @property
    def volatility(self) -> float:
        """Get annualized volatility."""
        return self.performance_metrics.get('volatility', 0.0)

    @property
    def alpha(self) -> float:
        """Get alpha (if benchmark available)."""
        return self.performance_metrics.get('alpha', 0.0)

    @property
    def beta(self) -> float:
        """Get beta (if benchmark available)."""
        return self.performance_metrics.get('beta', 1.0)

    @property
    def information_ratio(self) -> float:
        """Get information ratio (if benchmark available)."""
        return self.performance_metrics.get('information_ratio', 0.0)

    @property
    def win_rate(self) -> float:
        """Get win rate."""
        return self.performance_metrics.get('win_rate', 0.0)

    @property
    def profit_factor(self) -> float:
        """Get profit factor."""
        return self.performance_metrics.get('profit_factor', 1.0)

    @property
    def var_95(self) -> float:
        """Get 95% Value at Risk."""
        return self.performance_metrics.get('var_95', 0.0)

    @property
    def tracking_error(self) -> float:
        """Get tracking error (if benchmark available)."""
        return self.performance_metrics.get('tracking_error', 0.0)

    def _calculate_basic_metrics(self) -> Dict[str, float]:
        """Calculate basic metrics if not provided."""
        try:
            returns = self.daily_returns.dropna()
            if len(returns) < 2:
                return self._empty_metrics()

            # Total return
            total_return = (self.final_value - self.initial_capital) / self.initial_capital

            # Annualized return
            days_held = len(returns)
            years_held = days_held / 252.0
            annualized_return = (1 + total_return) ** (1 / years_held) - 1 if years_held > 0 else 0

            # Volatility and Sharpe
            volatility = returns.std() * (252 ** 0.5)
            sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()

            return {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': (returns > 0).mean(),
                'profit_factor': self._calculate_profit_factor(returns)
            }

        except Exception as e:
            logger.error(f"Error calculating basic metrics: {e}")
            return self._empty_metrics()

    def _calculate_profit_factor(self, returns: pd.Series) -> float:
        """Calculate profit factor."""
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        return positive_returns / negative_returns if negative_returns != 0 else float('inf')

    def _empty_metrics(self) -> Dict[str, float]:
        """Return empty metrics dictionary."""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0,
            'profit_factor': 1.0
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get formatted summary of key results."""
        return {
            "Initial Capital": f"${self.initial_capital:,.0f}",
            "Final Value": f"${self.final_value:,.0f}",
            "Total Return": f"{self.total_return:.2%}",
            "Annualized Return": f"{self.annualized_return:.2%}",
            "Volatility": f"{self.volatility:.2%}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.3f}",
            "Max Drawdown": f"{self.max_drawdown:.2%}",
            "Win Rate": f"{self.win_rate:.1%}",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Total Trades": len(self.trades),
            "Transaction Costs": f"${self.transaction_costs:,.2f}",
            "Turnover Rate": f"{self.turnover_rate:.1%}",
            "Alpha": f"{self.alpha:.2%}" if self.benchmark_returns is not None else "N/A",
            "Beta": f"{self.beta:.2f}" if self.benchmark_returns is not None else "N/A"
        }

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed performance metrics."""
        summary = self.get_summary()

        # Add additional metrics
        additional_metrics = {
            "Sortino Ratio": f"{self.sortino_ratio:.3f}",
            "Calmar Ratio": f"{self.calmar_ratio:.3f}",
            "95% VaR": f"{self.var_95:.2%}",
            "Information Ratio": f"{self.information_ratio:.3f}" if self.benchmark_returns is not None else "N/A",
            "Tracking Error": f"{self.tracking_error:.2%}" if self.benchmark_returns is not None else "N/A"
        }

        summary.update(additional_metrics)
        return summary

    def save_to_csv(self, output_dir: str = "results") -> None:
        """Save results to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save portfolio values
        portfolio_file = os.path.join(output_dir, "portfolio_values.csv")
        self.portfolio_values.to_csv(portfolio_file)

        # Save daily returns
        returns_file = os.path.join(output_dir, "daily_returns.csv")
        self.daily_returns.to_csv(returns_file)

        # Save benchmark returns if available
        if self.benchmark_returns is not None:
            benchmark_file = os.path.join(output_dir, "benchmark_returns.csv")
            self.benchmark_returns.to_csv(benchmark_file)

        # Save summary
        summary_file = os.path.join(output_dir, "results_summary.json")
        import json
        with open(summary_file, 'w') as f:
            json.dump(self.get_summary(), f, indent=2)

        logger.info(f"Results saved to {output_dir}")

    def plot_performance(self, figsize: tuple = (12, 8), save_path: Optional[str] = None) -> None:
        """Plot performance chart."""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

            # Portfolio value
            ax1.plot(self.portfolio_values.index, self.portfolio_values.values, label='Portfolio', linewidth=2)
            ax1.set_title('Portfolio Value Over Time')
            ax1.set_ylabel('Portfolio Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Drawdown
            portfolio_returns = self.daily_returns.dropna()
            if len(portfolio_returns) > 0:
                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max

                ax2.fill_between(drawdown.index, drawdown.values * 100, 0,
                                alpha=0.3, color='red', label='Drawdown')
                ax2.set_title('Drawdown')
                ax2.set_ylabel('Drawdown (%)')
                ax2.set_xlabel('Date')
                ax2.legend()
                ax2.grid(True, alpha=0.3)

            plt.tight_layout()

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Chart saved to {save_path}")

            plt.show()

        except ImportError:
            logger.warning("Matplotlib not available for plotting")
        except Exception as e:
            logger.error(f"Error plotting performance: {e}")

    def compare_to_benchmark(self) -> Dict[str, Any]:
        """Compare performance to benchmark if available."""
        if self.benchmark_returns is None:
            return {"error": "No benchmark data available"}

        try:
            # Calculate relative performance
            portfolio_returns = self.daily_returns.dropna()
            common_dates = portfolio_returns.index.intersection(self.benchmark_returns.index)

            if len(common_dates) < 2:
                return {"error": "Insufficient overlapping data"}

            portfolio_aligned = portfolio_returns.loc[common_dates]
            benchmark_aligned = self.benchmark_returns.loc[common_dates]

            # Cumulative returns
            portfolio_cumulative = (1 + portfolio_aligned).cumprod()
            benchmark_cumulative = (1 + benchmark_aligned).cumprod()

            # Outperformance
            outperformance = portfolio_cumulative - benchmark_cumulative
            total_outperformance = outperformance.iloc[-1]

            return {
                "total_outperformance": f"{total_outperformance:.2%}",
                "annualized_outperformance": f"{(1 + total_outperformance) ** (252/len(common_dates)) - 1:.2%}",
                "outperformance_volatility": f"{outperformance.std() * (252 ** 0.5):.2%}",
                "tracking_error": f"{(portfolio_aligned - benchmark_aligned).std() * (252 ** 0.5):.2%}",
                "up_capture": self.performance_metrics.get('up_capture', 0),
                "down_capture": self.performance_metrics.get('down_capture', 0)
            }

        except Exception as e:
            logger.error(f"Error comparing to benchmark: {e}")
            return {"error": f"Comparison failed: {e}"}