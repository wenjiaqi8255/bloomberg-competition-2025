"""
Weights & Biases experiment tracking for trading strategies.

This module provides comprehensive logging of trading experiments,
including hyperparameters, performance metrics, and visualizations.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import wandb

from .secrets_manager import SecretsManager

logger = logging.getLogger(__name__)


class WandBLogger:
    """
    Weights & Biases logger for trading strategy experiments.

    Features:
    - Automatic experiment initialization
    - Hyperparameter tracking
    - Performance metrics logging
    - Interactive visualizations
    - Model and data artifact management
    - Team collaboration support
    """

    def __init__(self, project_name: str = "bloomberg-competition",
                 entity: str = None, config: Dict = None,
                 tags: List[str] = None, group: str = None):
        """
        Initialize WandB logger.

        Args:
            project_name: WandB project name
            entity: WandB entity (team/username)
            config: Configuration dictionary to log
            tags: List of tags for the experiment
            group: Group for organizing related experiments
        """
        self.project_name = project_name
        self.entity = entity
        self.config = config or {}
        self.tags = tags or []
        self.group = group
        self.run = None
        self.is_initialized = False
        self.secrets_manager = SecretsManager()

    def initialize_experiment(self, experiment_name: str = None,
                            run_id: str = None, notes: str = None,
                            **kwargs):
        """
        Initialize a new WandB experiment.

        Args:
            experiment_name: Name of the experiment
            run_id: Specific run ID to use (for resuming)
            notes: Additional notes about the experiment
            **kwargs: Additional wandb.init() parameters
        """
        try:
            if experiment_name is None:
                experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # Check if API key is available using secrets manager
            api_key = self.secrets_manager.get_wandb_api_key()
            if not api_key:
                logger.warning("WANDB_API_KEY not found in environment variables or .env file")
                return False

            # Setup environment variable for wandb
            self.secrets_manager.setup_wandb_environment()

            # Initialize wandb run
            init_kwargs = {
                'project': self.project_name,
                'entity': self.entity,
                'name': experiment_name,
                'id': run_id,
                'config': self.config,
                'tags': self.tags,
                'group': self.group,
                'notes': notes,
                **kwargs
            }

            # Remove None values to avoid conflicts
            init_kwargs = {k: v for k, v in init_kwargs.items() if v is not None}

            self.run = wandb.init(**init_kwargs)

            self.is_initialized = True
            logger.info(f"Initialized WandB experiment: {experiment_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize WandB experiment: {e}")
            return False

    def log_config(self, config: Dict, step: int = None):
        """Log configuration parameters."""
        if not self.is_initialized:
            logger.warning("WandB not initialized. Skipping config logging.")
            return

        try:
            wandb.config.update(config)
            logger.debug("Configuration logged to WandB")
        except Exception as e:
            logger.error(f"Failed to log config to WandB: {e}")

    def log_metrics(self, metrics: Dict, step: int = None):
        """
        Log performance metrics.

        Args:
            metrics: Dictionary of metrics to log
            step: Step number for time-series logging
        """
        if not self.is_initialized:
            logger.warning("WandB not initialized. Skipping metrics logging.")
            return

        try:
            wandb.log(metrics, step=step)
            logger.debug(f"Logged metrics: {list(metrics.keys())}")
        except Exception as e:
            logger.error(f"Failed to log metrics to WandB: {e}")

    def log_portfolio_performance(self, portfolio_df: pd.DataFrame,
                                 benchmark_df: pd.DataFrame = None,
                                 step: int = None):
        """
        Log portfolio performance with visualizations.

        Args:
            portfolio_df: DataFrame with portfolio performance data
            benchmark_df: Optional benchmark performance data
            step: Step number for logging
        """
        if not self.is_initialized:
            logger.warning("WandB not initialized. Skipping performance logging.")
            return

        try:
            # Calculate performance metrics
            portfolio_returns = portfolio_df['portfolio_value'].pct_change().dropna()

            metrics = {
                'final_portfolio_value': portfolio_df['portfolio_value'].iloc[-1],
                'total_return': (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0] - 1),
                'volatility': portfolio_returns.std() * (252 ** 0.5),
                'sharpe_ratio': portfolio_returns.mean() / portfolio_returns.std() * (252 ** 0.5) if portfolio_returns.std() > 0 else 0
            }

            # Add benchmark comparison if available
            if benchmark_df is not None and not benchmark_df.empty:
                benchmark_returns = benchmark_df['Close'].pct_change().dropna()
                metrics['benchmark_return'] = (benchmark_df['Close'].iloc[-1] / benchmark_df['Close'].iloc[0] - 1)
                metrics['alpha'] = metrics['total_return'] - metrics['benchmark_return']

            # Log metrics
            self.log_metrics(metrics, step)

            # Create and log visualizations
            self._log_portfolio_chart(portfolio_df, benchmark_df)
            self._log_drawdown_chart(portfolio_df)
            self._log_returns_distribution(portfolio_returns)

        except Exception as e:
            logger.error(f"Failed to log portfolio performance: {e}")

    def log_trades(self, trades_df: pd.DataFrame, step: int = None):
        """
        Log trading activity statistics.

        Args:
            trades_df: DataFrame with trade data
            step: Step number for logging
        """
        if not self.is_initialized or trades_df.empty:
            return

        try:
            # Calculate trade statistics
            trade_metrics = {
                'total_trades': len(trades_df),
                'winning_trades': len(trades_df[trades_df['pnl'] > 0]),
                'losing_trades': len(trades_df[trades_df['pnl'] < 0]),
                'avg_trade_size': trades_df['value'].mean(),
                'total_transaction_cost': trades_df['cost'].sum(),
                'largest_win': trades_df['pnl'].max(),
                'largest_loss': trades_df['pnl'].min()
            }

            if trade_metrics['total_trades'] > 0:
                trade_metrics['win_rate'] = trade_metrics['winning_trades'] / trade_metrics['total_trades']
                trade_metrics['avg_trade_pnl'] = trades_df['pnl'].mean()

            self.log_metrics(trade_metrics, step)

            # Log trade distribution chart
            self._log_trade_distribution(trades_df)

        except Exception as e:
            logger.error(f"Failed to log trade statistics: {e}")

    def log_hyperparameters(self, hyperparameters: Dict):
        """Log strategy hyperparameters."""
        if not self.is_initialized:
            logger.warning("WandB not initialized. Skipping hyperparameter logging.")
            return

        try:
            wandb.config.update(hyperparameters)
            logger.debug(f"Logged hyperparameters: {list(hyperparameters.keys())}")
        except Exception as e:
            logger.error(f"Failed to log hyperparameters to WandB: {e}")

    def log_artifact(self, artifact_path: str, artifact_name: str,
                    artifact_type: str = "dataset", description: str = ""):
        """
        Log an artifact (file or directory).

        Args:
            artifact_path: Path to the file or directory
            artifact_name: Name for the artifact
            artifact_type: Type of artifact
            description: Description of the artifact
        """
        if not self.is_initialized:
            logger.warning("WandB not initialized. Skipping artifact logging.")
            return

        try:
            artifact = wandb.Artifact(artifact_name, type=artifact_type, description=description)
            if os.path.isfile(artifact_path):
                artifact.add_file(artifact_path)
            elif os.path.isdir(artifact_path):
                artifact.add_dir(artifact_path)

            wandb.log_artifact(artifact)
            logger.info(f"Logged artifact: {artifact_name}")
        except Exception as e:
            logger.error(f"Failed to log artifact {artifact_name}: {e}")

    def log_dataset_info(self, dataset_stats: Dict):
        """Log dataset statistics and information."""
        if not self.is_initialized:
            return

        try:
            self.log_metrics({
                f'dataset_{key}': value for key, value in dataset_stats.items()
            })
        except Exception as e:
            logger.error(f"Failed to log dataset info: {e}")

    def _log_portfolio_chart(self, portfolio_df: pd.DataFrame,
                           benchmark_df: pd.DataFrame = None):
        """Create and log portfolio performance chart."""
        try:
            fig = go.Figure()

            # Portfolio performance
            fig.add_trace(go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['portfolio_value'],
                mode='lines',
                name='Portfolio',
                line=dict(color='blue', width=2)
            ))

            # Benchmark comparison
            if benchmark_df is not None and not benchmark_df.empty:
                # Normalize benchmark to portfolio starting value
                normalized_benchmark = benchmark_df['Close'] * (
                    portfolio_df['portfolio_value'].iloc[0] / benchmark_df['Close'].iloc[0]
                )
                fig.add_trace(go.Scatter(
                    x=benchmark_df.index,
                    y=normalized_benchmark,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color='gray', width=1, dash='dash')
                ))

            fig.update_layout(
                title='Portfolio Performance vs Benchmark',
                xaxis_title='Date',
                yaxis_title='Portfolio Value ($)',
                template='plotly_white'
            )

            wandb.log({"portfolio_performance": fig})

        except Exception as e:
            logger.error(f"Failed to create portfolio chart: {e}")

    def _log_drawdown_chart(self, portfolio_df: pd.DataFrame):
        """Create and log drawdown chart."""
        try:
            # Calculate drawdown series
            portfolio_values = portfolio_df['portfolio_value']
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max

            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown * 100,
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red', width=2)
            ))

            fig.update_layout(
                title='Portfolio Drawdown',
                xaxis_title='Date',
                yaxis_title='Drawdown (%)',
                template='plotly_white',
                yaxis_tickformat='%.1f%%'
            )

            wandb.log({"drawdown_chart": fig})

        except Exception as e:
            logger.error(f"Failed to create drawdown chart: {e}")

    def _log_returns_distribution(self, returns: pd.Series):
        """Create and log returns distribution chart."""
        try:
            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=returns * 100,
                name='Returns Distribution',
                nbinsx=30,
                opacity=0.7
            ))

            fig.add_vline(
                x=returns.mean() * 100,
                line_dash="dash",
                line_color="red",
                annotation_text="Mean"
            )

            fig.update_layout(
                title='Returns Distribution',
                xaxis_title='Daily Return (%)',
                yaxis_title='Frequency',
                template='plotly_white'
            )

            wandb.log({"returns_distribution": fig})

        except Exception as e:
            logger.error(f"Failed to create returns distribution chart: {e}")

    def _log_trade_distribution(self, trades_df: pd.DataFrame):
        """Create and log trade P&L distribution chart."""
        try:
            if 'pnl' not in trades_df.columns:
                return

            fig = go.Figure()

            fig.add_trace(go.Histogram(
                x=trades_df['pnl'],
                name='Trade P&L Distribution',
                nbinsx=20,
                opacity=0.7
            ))

            fig.add_vline(
                x=trades_df['pnl'].mean(),
                line_dash="dash",
                line_color="green",
                annotation_text="Mean"
            )

            fig.update_layout(
                title='Trade P&L Distribution',
                xaxis_title='Trade P&L ($)',
                yaxis_title='Frequency',
                template='plotly_white'
            )

            wandb.log({"trade_distribution": fig})

        except Exception as e:
            logger.error(f"Failed to create trade distribution chart: {e}")

    def finish_experiment(self, exit_code: int = 0):
        """Finish the WandB experiment."""
        if self.is_initialized:
            try:
                wandb.finish(exit_code=exit_code)
                logger.info("WandB experiment finished successfully")
                self.is_initialized = False
            except Exception as e:
                logger.error(f"Failed to finish WandB experiment: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish_experiment()
        return False

    @staticmethod
    def setup_wandb_api_key(api_key: str):
        """Setup WandB API key."""
        os.environ['WANDB_API_KEY'] = api_key
        logger.info("WandB API key configured")