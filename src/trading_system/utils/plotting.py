"""
Utilities for creating and logging backtest visualization charts.

This module consolidates all plotting logic, separating the concern of
visualization from the orchestration logic in the StrategyRunner.
"""

import logging
from typing import Optional

import pandas as pd

from .experiment_tracking import ExperimentTrackerInterface

logger = logging.getLogger(__name__)

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    logger.warning("Plotly not found. Chart generation will be disabled.")


class BacktestPlotter:
    """Handles the creation and logging of backtest visualization charts."""

    def __init__(self, experiment_tracker: ExperimentTrackerInterface):
        """
        Initializes the plotter with an experiment tracker instance.

        Args:
            experiment_tracker: An object that conforms to the ExperimentTrackerInterface.
        """
        self.experiment_tracker = experiment_tracker

    def create_and_log_charts(self, portfolio_history: pd.DataFrame,
                                     trades_df: pd.DataFrame,
                                     benchmark_df: Optional[pd.DataFrame] = None):
        """
        Creates and logs all standard backtest charts to the experiment tracker.

        Args:
            portfolio_history: DataFrame with portfolio value history.
            trades_df: DataFrame with trade records.
            benchmark_df: Optional DataFrame with benchmark data for comparison.
        """
        if not PLOTLY_AVAILABLE or not self.experiment_tracker:
            return

        try:
            self._create_equity_curve_chart(portfolio_history, benchmark_df)
            self._create_drawdown_chart(portfolio_history)
            self._create_trade_analysis_charts(trades_df)
            self._create_returns_heatmap(portfolio_history)
            logger.info("Backtest charts created and logged successfully.")
        except Exception as e:
            logger.error(f"Failed to create backtest charts: {e}")

    def _create_equity_curve_chart(self, portfolio_history: pd.DataFrame,
                                 benchmark_df: Optional[pd.DataFrame] = None):
        """Create and log equity curve chart."""
        try:
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=portfolio_history.index,
                y=portfolio_history['Portfolio Value'],
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))

            if benchmark_df is not None:
                fig.add_trace(go.Scatter(
                    x=benchmark_df.index,
                    y=benchmark_df['Value'],
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='gray', width=1, dash='dash')
                ))

            fig.update_layout(
                title='Portfolio Equity Curve',
                xaxis_title='Date',
                yaxis_title='Portfolio Value',
                hovermode='x unified',
                template='plotly_white'
            )

            self.experiment_tracker.log_figure(fig, "equity_curve")
        except Exception as e:
            logger.error(f"Failed to create equity curve chart: {e}")

    def _create_drawdown_chart(self, portfolio_history: pd.DataFrame):
        """Create and log drawdown chart."""
        try:
            peak = portfolio_history['Portfolio Value'].expanding().max()
            drawdown = (portfolio_history['Portfolio Value'] - peak) / peak * 100

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=portfolio_history.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red')
            ))

            fig.update_layout(
                title='Portfolio Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white'
            )

            self.experiment_tracker.log_figure(fig, "drawdown")
        except Exception as e:
            logger.error(f"Failed to create drawdown chart: {e}")

    def _create_trade_analysis_charts(self, trades_df: pd.DataFrame):
        """Create and log trade analysis charts."""
        if trades_df.empty:
            return

        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Trade Returns Distribution', 'Trade Volume Over Time',
                              'Win/Loss Ratio', 'Cumulative P&L'),
                specs=[[{"type": "histogram"}, {"type": "scatter"}],
                       [{"type": "pie"}, {"type": "scatter"}]]
            )

            if 'Return' in trades_df.columns:
                fig.add_trace(
                    go.Histogram(x=trades_df['Return'], name='Returns'),
                    row=1, col=1
                )

            if 'Date' in trades_df.columns:
                trade_counts = trades_df.groupby('Date').size()
                fig.add_trace(
                    go.Scatter(x=trade_counts.index, y=trade_counts.values,
                             mode='lines+markers', name='Trade Count'),
                    row=1, col=2
                )

            if 'Return' in trades_df.columns:
                wins = (trades_df['Return'] > 0).sum()
                losses = (trades_df['Return'] <= 0).sum()
                fig.add_trace(
                    go.Pie(labels=['Winning Trades', 'Losing Trades'],
                          values=[wins, losses], name='Win/Loss'),
                    row=2, col=1
                )

            if 'Return' in trades_df.columns:
                cum_returns = trades_df['Return'].cumsum()
                fig.add_trace(
                    go.Scatter(x=trades_df.index, y=cum_returns,
                             mode='lines', name='Cumulative P&L'),
                    row=2, col=2
                )

            fig.update_layout(
                title='Trade Analysis',
                height=600,
                showlegend=False,
                template='plotly_white'
            )

            self.experiment_tracker.log_figure(fig, "trade_analysis")
        except Exception as e:
            logger.error(f"Failed to create trade analysis charts: {e}")

    def _create_returns_heatmap(self, portfolio_history: pd.DataFrame):
        """Create and log monthly returns heatmap."""
        try:
            monthly_returns = portfolio_history['Portfolio Value'].resample('M').last().pct_change()
            monthly_returns = monthly_returns.dropna() * 100

            monthly_returns.index = pd.to_datetime(monthly_returns.index)
            returns_matrix = monthly_returns.groupby([monthly_returns.index.year, monthly_returns.index.month]).first().unstack()

            fig = go.Figure(data=go.Heatmap(
                z=returns_matrix.values,
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=returns_matrix.index,
                colorscale='RdYlGn',
                text=returns_matrix.round(2).values,
                texttemplate="%{text}%",
                textfont={"size": 10},
                hoverongaps=False
            ))

            fig.update_layout(
                title='Monthly Returns Heatmap (%)',
                xaxis_title='Month',
                yaxis_title='Year'
            )

            self.experiment_tracker.log_figure(fig, "monthly_returns_heatmap")
        except Exception as e:
            logger.error(f"Failed to create returns heatmap: {e}")
