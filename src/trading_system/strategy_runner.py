"""
Main strategy runner that orchestrates the complete trading pipeline.

This module provides a high-level interface for running trading strategies
with configuration-driven parameter management.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

from .config.config_loader import ConfigLoader
from .data.yfinance_provider import YFinanceProvider
from .backtest.standard_backtest import StandardBacktest
from .strategies import DualMomentumStrategy
from .utils.wandb_logger import WandBLogger

logger = logging.getLogger(__name__)


class StrategyRunner:
    """
    Main strategy runner that orchestrates the complete trading pipeline.

    Features:
    - Configuration-driven strategy execution
    - Automatic data acquisition and validation
    - Performance calculation and analysis
    - Experiment tracking with Weights & Biases
    - Error handling and logging
    """

    def __init__(self, config_path: str = None):
        """
        Initialize strategy runner.

        Args:
            config_path: Path to configuration file
        """
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.load_config()

        # Initialize components
        self.data_provider = None
        self.strategy = None
        self.backtest_engine = None
        self.wandb_logger = None

        # Results storage
        self.results = {}
        self.is_initialized = False

    def initialize(self):
        """Initialize all components based on configuration."""
        try:
            logger.info("Initializing strategy runner components...")

            # Initialize data provider
            data_config = self.config_loader.get_data_config()
            self.data_provider = YFinanceProvider(
                max_retries=data_config.get('retry_attempts', 3),
                retry_delay=data_config.get('retry_delay', 1.0),
                request_timeout=data_config.get('timeout', 30)
            )

            # Initialize strategy
            strategy_config = self.config_loader.get_strategy_config()
            strategy_params = self.config_loader.get_strategy_parameters()

            if strategy_config.get('type') == 'DualMomentumStrategy':
                self.strategy = DualMomentumStrategy(
                    name=strategy_config.get('name', 'DualMomentum'),
                    **strategy_params
                )
            else:
                raise ValueError(f"Unknown strategy type: {strategy_config.get('type')}")

            # Initialize backtest engine
            backtest_config = self.config_loader.get_backtest_config()
            self.backtest_engine = StandardBacktest(
                initial_capital=backtest_config.get('initial_capital', 1_000_000),
                transaction_cost=backtest_config.get('transaction_cost', 0.001),
                benchmark_symbol=backtest_config.get('benchmark_symbol', 'SPY')
            )

            # Initialize WandB logger
            experiment_config = self.config_loader.get_experiment_config()
            self.wandb_logger = WandBLogger(
                project_name=experiment_config.get('project_name', 'bloomberg-competition'),
                tags=experiment_config.get('tags', []),
                group=experiment_config.get('group')
            )

            self.is_initialized = True
            logger.info("Strategy runner initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize strategy runner: {e}")
            raise

    def run_strategy(self, experiment_name: str = None) -> Dict[str, Any]:
        """
        Run the complete strategy pipeline.

        Args:
            experiment_name: Name for the experiment (default: strategy name + timestamp)

        Returns:
            Dictionary with complete results
        """
        if not self.is_initialized:
            self.initialize()

        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"{self.strategy.get_name()}_{timestamp}"

        logger.info(f"Running strategy: {experiment_name}")

        try:
            # Initialize WandB experiment
            experiment_config = self.config_loader.get_experiment_config()
            self.wandb_logger.initialize_experiment(
                experiment_name=experiment_name,
                notes=experiment_config.get('notes', ''),
                tags=experiment_config.get('tags', [])
            )

            # Log configuration
            self.wandb_logger.log_config(self.config)

            # Step 1: Get asset universe
            universe = self.config_loader.get_asset_universe()
            logger.info(f"Asset universe: {universe}")

            # Step 2: Fetch data
            backtest_config = self.config_loader.get_backtest_config()
            price_data, benchmark_data = self._fetch_data(
                universe=universe,
                start_date=datetime.strptime(backtest_config['start_date'], '%Y-%m-%d'),
                end_date=datetime.strptime(backtest_config['end_date'], '%Y-%m-%d'),
                benchmark_symbol=backtest_config['benchmark_symbol']
            )

            # Log data statistics
            data_stats = self._calculate_data_statistics(price_data)
            self.wandb_logger.log_dataset_info(data_stats)

            # Step 3: Generate strategy signals
            strategy_signals = self.strategy.generate_signals(
                price_data=price_data,
                start_date=datetime.strptime(backtest_config['start_date'], '%Y-%m-%d'),
                end_date=datetime.strptime(backtest_config['end_date'], '%Y-%m-%d')
            )

            # Step 4: Run backtest
            backtest_results = self.backtest_engine.run_backtest(
                strategy_signals=strategy_signals,
                price_data=price_data,
                benchmark_data=benchmark_data,
                start_date=datetime.strptime(backtest_config['start_date'], '%Y-%m-%d'),
                end_date=datetime.strptime(backtest_config['end_date'], '%Y-%m-%d'),
                rebalance_frequency=backtest_config.get('rebalance_frequency', 'monthly')
            )

            # Step 5: Log performance to WandB
            portfolio_history = backtest_results['portfolio_history']
            self.wandb_logger.log_portfolio_performance(
                portfolio_df=portfolio_history,
                benchmark_df=benchmark_data
            )

            # Step 6: Log trades if available
            if backtest_results.get('trades'):
                trades_df = self._process_trades_to_dataframe(backtest_results['trades'])
                if not trades_df.empty:
                    self.wandb_logger.log_trades(trades_df)

            # Step 7: Calculate and log risk metrics
            risk_metrics = self.strategy.calculate_risk_metrics(price_data, strategy_signals)
            if risk_metrics:
                self.wandb_logger.log_metrics(risk_metrics)

            # Step 8: Log strategy-specific metrics
            strategy_metrics = self._calculate_strategy_specific_metrics(
                strategy_signals, price_data, backtest_results
            )
            self.wandb_logger.log_metrics(strategy_metrics)

            # Compile final results
            self.results = {
                'experiment_name': experiment_name,
                'config': self.config,
                'data_statistics': data_stats,
                'strategy_signals': strategy_signals,
                'backtest_results': backtest_results,
                'performance_metrics': backtest_results['performance_metrics'],
                'risk_metrics': risk_metrics,
                'strategy_metrics': strategy_metrics,
                'execution_timestamp': datetime.now().isoformat()
            }

            # Save results locally
            self._save_results()

            logger.info(f"Strategy execution completed: {experiment_name}")
            return self.results

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            raise

    def _fetch_data(self, universe: List[str], start_date: datetime,
                   end_date: datetime, benchmark_symbol: str) -> tuple:
        """
        Fetch price data for asset universe and benchmark.

        Args:
            universe: List of asset tickers
            start_date: Start date for data fetch
            end_date: End date for data fetch
            benchmark_symbol: Benchmark symbol

        Returns:
            Tuple of (price_data_dict, benchmark_dataframe)
        """
        logger.info(f"Fetching data from {start_date} to {end_date}")

        # Validate symbols first
        valid_symbols = []
        for symbol in universe:
            if self.data_provider.validate_symbol(symbol):
                valid_symbols.append(symbol)
            else:
                logger.warning(f"Symbol {symbol} validation failed, excluding from universe")

        if not valid_symbols:
            raise ValueError("No valid symbols in asset universe")

        logger.info(f"Valid symbols: {valid_symbols}")

        # Fetch price data
        price_data = self.data_provider.get_historical_data(
            symbols=valid_symbols,
            start_date=start_date,
            end_date=end_date
        )

        if not price_data:
            raise ValueError("Failed to fetch price data")

        # Fetch benchmark data
        benchmark_data = None
        if benchmark_symbol and benchmark_symbol in valid_symbols:
            benchmark_data = price_data[benchmark_symbol]
        elif benchmark_symbol:
            # Fetch benchmark separately if not in universe
            benchmark_data_dict = self.data_provider.get_historical_data(
                symbols=[benchmark_symbol],
                start_date=start_date,
                end_date=end_date
            )
            benchmark_data = benchmark_data_dict.get(benchmark_symbol)

        logger.info(f"Successfully fetched data for {len(price_data)} symbols")
        return price_data, benchmark_data

    def _calculate_data_statistics(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate data statistics for logging."""
        stats = {
            'num_symbols': len(price_data),
            'date_range_start': None,
            'date_range_end': None,
            'total_data_points': 0,
            'avg_data_points_per_symbol': 0
        }

        if price_data:
            start_dates = []
            end_dates = []
            total_points = 0

            for symbol, data in price_data.items():
                if data is not None and not data.empty:
                    start_dates.append(data.index.min())
                    end_dates.append(data.index.max())
                    total_points += len(data)

            if start_dates and end_dates:
                stats['date_range_start'] = min(start_dates).isoformat()
                stats['date_range_end'] = max(end_dates).isoformat()
                stats['total_data_points'] = total_points
                stats['avg_data_points_per_symbol'] = total_points / len(price_data)

        return stats

    def _process_trades_to_dataframe(self, trades: List[Dict]) -> pd.DataFrame:
        """Convert trades list to DataFrame for analysis."""
        if not trades:
            return pd.DataFrame()

        # Calculate P&L for each trade
        trades_with_pnl = []
        for trade in trades:
            trade_copy = trade.copy()
            # Simple P&L calculation (would need more sophisticated calculation in practice)
            trade_copy['pnl'] = 0  # Placeholder
            trades_with_pnl.append(trade_copy)

        return pd.DataFrame(trades_with_pnl)

    def _calculate_strategy_specific_metrics(self, strategy_signals: pd.DataFrame,
                                          price_data: Dict[str, pd.DataFrame],
                                          backtest_results: Dict) -> Dict[str, Any]:
        """Calculate strategy-specific performance metrics."""
        metrics = {}

        try:
            # Calculate concentration metrics
            if not strategy_signals.empty:
                # Average number of positions held
                positions_per_period = strategy_signals.apply(
                    lambda row: sum(1 for x in row if x > 0), axis=1
                )
                metrics['avg_positions_held'] = positions_per_period.mean()
                metrics['max_positions_held'] = positions_per_period.max()
                metrics['min_positions_held'] = positions_per_period.min()

                # Cash allocation statistics
                cash_allocations = strategy_signals.get('SHY', pd.Series(0, index=strategy_signals.index))
                metrics['avg_cash_allocation'] = cash_allocations.mean()
                metrics['max_cash_allocation'] = cash_allocations.max()

            # Calculate drawdown duration
            portfolio_values = backtest_results['portfolio_history']['portfolio_value']
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max

            # Find drawdown periods
            is_drawdown = drawdown < 0
            drawdown_periods = []
            current_start = None

            for date, is_dd in is_drawdown.items():
                if is_dd and current_start is None:
                    current_start = date
                elif not is_dd and current_start is not None:
                    drawdown_periods.append((date - current_start).days)
                    current_start = None

            if drawdown_periods:
                metrics['avg_drawdown_duration'] = sum(drawdown_periods) / len(drawdown_periods)
                metrics['max_drawdown_duration'] = max(drawdown_periods)

            # Strategy efficiency metrics
            total_return = backtest_results.get('total_return', 0)
            volatility = backtest_results['performance_metrics'].get('volatility', 0)
            if volatility > 0:
                metrics['return_to_risk_ratio'] = total_return / volatility
            else:
                metrics['return_to_risk_ratio'] = 0

        except Exception as e:
            logger.warning(f"Error calculating strategy-specific metrics: {e}")

        return metrics

    def _save_results(self):
        """Save results to local files."""
        try:
            output_config = self.config_loader.get_output_config()

            if output_config.get('save_results', True):
                # Create results directory
                results_dir = output_config.get('results_path', './results/')
                os.makedirs(results_dir, exist_ok=True)

                # Save results as JSON
                import json
                results_file = os.path.join(results_dir, f"{self.results['experiment_name']}_results.json")

                # Convert numpy types to JSON-serializable
                results_json = json.loads(json.dumps(self.results, default=str))

                with open(results_file, 'w') as f:
                    json.dump(results_json, f, indent=2)

                logger.info(f"Results saved to {results_file}")

                # Save strategy signals
                signals_file = os.path.join(results_dir, f"{self.results['experiment_name']}_signals.csv")
                self.results['strategy_signals'].to_csv(signals_file)
                logger.info(f"Signals saved to {signals_file}")

                # Save portfolio history
                history_file = os.path.join(results_dir, f"{self.results['experiment_name']}_portfolio.csv")
                self.results['backtest_results']['portfolio_history'].to_csv(history_file)
                logger.info(f"Portfolio history saved to {history_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def cleanup(self):
        """Clean up resources."""
        if self.wandb_logger:
            self.wandb_logger.finish_experiment()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False