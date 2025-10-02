"""
Main Strategy Runner for Backtesting and Research
===================================================

This module provides the `StrategyRunner`, a high-level orchestrator designed
specifically for running single-strategy backtests. Its primary purpose is to
facilitate the research and validation phase of a trading strategy's lifecycle.

Core Responsibilities:
----------------------
1.  **Configuration-Driven Setup**: Loads all parameters for a backtest, including
    the strategy to use, from YAML configuration files.
2.  **Component Initialization**: Sets up all necessary components for a backtest,
    such as the data provider, the specified trading strategy (via the `StrategyFactory`),
    and the `BacktestEngine`.
3.  **Data Fetching**: Manages the acquisition of historical market data required
    for the backtest period.
4.  **Backtest Execution**: Orchestrates the main backtesting loop by passing data and
    signals between the strategy and the backtest engine.
5.  **Experiment Tracking**: Integrates with tools like Weights & Biases to log
    configurations, performance metrics, and output artifacts (e.g., charts, trade logs).
6.  **Results Aggregation & Saving**: Compiles a comprehensive report of the backtest
    and saves signals, portfolio history, and performance results to local files.

Usage Example:
--------------
.. code-block:: python

    from trading_system.strategy_runner import create_strategy_runner

    # Create a runner instance, which will handle config loading and tracker setup
    runner = create_strategy_runner(
        config_path="configs/ml_strategy_example.yaml",
        use_wandb=True
    )

    # Run the entire backtest pipeline
    results = runner.run_strategy(experiment_name="ML_Strategy_Volatility_Test_Run_1")

    # The results dictionary contains all data and performance metrics.
    # Results are also saved to files and logged to the experiment tracker.
    print(results['performance_metrics'])

Distinction from SystemOrchestrator:
-----------------------------------
- **StrategyRunner**: Focuses on **one strategy at a time**. It is a tool for
  research, development, and validation.
- **SystemOrchestrator**: Designed to run a **multi-strategy portfolio** in a
  production or near-production environment. It deals with complexities like
  capital allocation between strategies and portfolio-level compliance.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd

from ..config.factory import ConfigFactory
from ..data.yfinance_provider import YFinanceProvider
# New backtesting architecture
from ..backtesting import BacktestEngine
from ..config.backtest import BacktestConfig
from ..strategies import StrategyFactory
from ..utils.experiment_tracking import (
    ExperimentTrackerInterface,
    ExperimentConfig,
    NullExperimentTracker,
    WandBExperimentTracker,
    create_backtest_config
)
from ..types import TradingSignal, SignalType

logger = logging.getLogger(__name__)


def create_strategy_runner(config_path: str = None,
                          experiment_tracker: ExperimentTrackerInterface = None,
                          use_wandb: bool = True) -> 'StrategyRunner':
    """
    Factory function to create StrategyRunner with desired experiment tracking.

    Args:
        config_path: Path to configuration file
        experiment_tracker: Custom experiment tracker instance
        use_wandb: Whether to use WandB tracker (ignored if experiment_tracker is provided)

    Returns:
        StrategyRunner instance with configured experiment tracking
    """
    if experiment_tracker is not None:
        return StrategyRunner(config_path=config_path, experiment_tracker=experiment_tracker)

    if use_wandb:
        try:
            wandb_tracker = WandBExperimentTracker(
                project_name='bloomberg-competition',
                fail_silently=True
            )
            return StrategyRunner(config_path=config_path, experiment_tracker=wandb_tracker)
        except Exception:
            logger.warning("Failed to create WandB tracker, using null tracker")

    # Default to null tracker
    null_tracker = NullExperimentTracker()
    return StrategyRunner(config_path=config_path, experiment_tracker=null_tracker)


class StrategyRunner:
    """
    Orchestrates a complete single-strategy trading backtest from configuration.

    This class acts as the main entry point for running a research backtest.
    It wires together the data provider, a strategy instance, the backtest engine,
    and the experiment tracker based on the settings provided in a configuration file.

    The typical workflow is:
    1. Instantiate via `create_strategy_runner`.
    2. Call `run_strategy()` to execute the backtest.
    3. The runner handles initialization, data fetching, signal generation,
       backtesting, and logging of results.
    """

    def __init__(self, config_path: str = None, experiment_tracker: ExperimentTrackerInterface = None):
        """
        Initialize the StrategyRunner.

        Args:
            config_path: Path to the main YAML configuration file.
            experiment_tracker: An instance of an experiment tracker (e.g., WandB).
                                If not provided, one will be created.
        """
        if config_path:
            self.configs = ConfigFactory.load_all_configs(config_path)
        else:
            # Use default configs
            self.configs = {}

        # Initialize components
        self.data_provider = None
        self.strategy = None
        self.backtest_engine = None
        self.experiment_tracker = experiment_tracker
        self.plotter = None

        # Results storage
        self.results = {}
        self.is_initialized = False

    def _convert_signals_to_unified_format(self, strategy_signals: pd.DataFrame) -> Dict[datetime, List[TradingSignal]]:
        """
        Convert DataFrame strategy signals to unified format Dict[datetime, List[TradingSignal]].

        Args:
            strategy_signals: DataFrame with strategy signals (old format)

        Returns:
            Dictionary mapping dates to lists of TradingSignal objects (new format)
        """
        unified_signals = {}

        if strategy_signals is None or strategy_signals.empty:
            return unified_signals

        for date in strategy_signals.index:
            if not isinstance(date, datetime):
                continue

            signals_for_date = []

            for symbol in strategy_signals.columns:
                signal_value = strategy_signals.loc[date, symbol]

                if pd.isna(signal_value) or signal_value == 0:
                    continue

                # Determine signal type
                signal_type = SignalType.BUY if signal_value > 0 else SignalType.SELL

                # Create TradingSignal object
                trading_signal = TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=abs(float(signal_value)),
                    timestamp=date,
                    confidence=abs(float(signal_value))  # Use strength as confidence
                )

                signals_for_date.append(trading_signal)

            if signals_for_date:
                unified_signals[date] = signals_for_date

        return unified_signals

    def initialize(self):
        """Initialize all components based on configuration."""
        try:
            logger.info("Initializing strategy runner components...")

            # Initialize data provider with defaults
            self.data_provider = YFinanceProvider(
                max_retries=3,
                retry_delay=1.0,
                request_timeout=30
            )

            # Initialize strategy with config objects
            strategy_config = self.configs.get('strategy')
            backtest_config = self.configs.get('backtest')

            if not strategy_config:
                raise ValueError("Strategy configuration not found")

            # Use the StrategyFactory to create the strategy instance
            # This replaces the old if/elif block and makes the runner extensible
            strategy_type = strategy_config.strategy_type.value
            strategy_params = {
                "name": strategy_config.name,
                **strategy_config.parameters
            }
            
            # For MLStrategy, we need to inject dependencies
            if strategy_type == 'ml':
                from ..models.model_persistence import ModelRegistry
                strategy_params['model_registry'] = ModelRegistry()
                # model_id is expected to be in parameters from the config
            
            self.strategy = StrategyFactory.create(
                strategy_type=strategy_type,
                **strategy_params
            )

            # Initialize backtest engine using config object
            if not backtest_config:
                raise ValueError("Backtest configuration not found")

            # Use new BacktestEngine
            self.backtest_engine = BacktestEngine(backtest_config)
            logger.info(f"Initialized new BacktestEngine with ${backtest_config.initial_capital:,.0f} capital")

            # Initialize experiment tracker with new interface
            if self.experiment_tracker is None:
                # Use WandB tracker by default, with fallback to null tracker
                try:
                    self.experiment_tracker = WandBExperimentTracker(
                        project_name='bloomberg-competition',
                        tags=[],
                        group=None,
                        fail_silently=True
                    )
                except Exception:
                    logger.warning("WandB tracker initialization failed, using null tracker")
                    self.experiment_tracker = NullExperimentTracker()

            from ..utils.plotting import BacktestPlotter
            self.plotter = BacktestPlotter(self.experiment_tracker)
  
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
            # Create experiment configuration
            strategy_config = self.configs.get('strategy')
            backtest_config = self.configs.get('backtest')

            # Build hyperparameters from configs
            hyperparameters = {}
            if strategy_config:
                hyperparameters.update(strategy_config.parameters or {})
            if backtest_config:
                hyperparameters.update({
                    'initial_capital': backtest_config.initial_capital,
                    'start_date': backtest_config.start_date,
                    'end_date': backtest_config.end_date,
                    'benchmark_symbol': backtest_config.benchmark_symbol
                })

            experiment_config = create_backtest_config(
                project_name='bloomberg-competition',
                strategy_name=self.strategy.get_name(),
                strategy_config=hyperparameters,
                tags=[strategy_config.strategy_type.value if strategy_config else 'unknown'],
                group=None
            )
            experiment_config.experiment_name = experiment_name

            # Initialize experiment with new interface
            run_id = self.experiment_tracker.init_run(experiment_config)
            logger.info(f"Started experiment run: {run_id}")

            # Log configuration summary
            config_summary = {name: config.get_summary() for name, config in self.configs.items()}
            self.experiment_tracker.log_params(config_summary)

            # Step 1: Get asset universe from strategy config
            strategy_config = self.configs.get('strategy')
            universe = strategy_config.universe if strategy_config else []
            logger.info(f"Asset universe: {universe}")

            # Step 2: Fetch data
            backtest_config = self.configs.get('backtest')
            price_data, benchmark_data = self._fetch_data(
                universe=universe,
                start_date=backtest_config.start_date,
                end_date=backtest_config.end_date,
                benchmark_symbol=backtest_config.benchmark_symbol
            )

            # Log data statistics
            data_stats = self._calculate_data_statistics(price_data)
            self.experiment_tracker.log_dataset_info(data_stats)

            # Step 3: Generate strategy signals
            strategy_signals = self.strategy.generate_signals(
                price_data=price_data,
                start_date=datetime.strptime(backtest_config['start_date'], '%Y-%m-%d'),
                end_date=datetime.strptime(backtest_config['end_date'], '%Y-%m-%d')
            )

            # Step 4: Convert signals to unified format and run backtest
            unified_strategy_signals = self._convert_signals_to_unified_format(strategy_signals)

            backtest_results = self.backtest_engine.run_backtest(
                strategy_signals=unified_strategy_signals,
                price_data=price_data,
                benchmark_data=benchmark_data
            )

            # Step 5: Log performance to experiment tracker
            portfolio_history = backtest_results.portfolio_values if hasattr(backtest_results, 'portfolio_values') else backtest_results.get('portfolio_history')
            if portfolio_history is not None:
                portfolio_df = portfolio_history.to_frame('portfolio_value') if hasattr(portfolio_history, 'to_frame') else portfolio_history
                self.experiment_tracker.log_portfolio_performance(
                    portfolio_df=portfolio_df,
                    benchmark_df=benchmark_data
                )

            # Step 6: Log trades if available
            trades = backtest_results.trades if hasattr(backtest_results, 'trades') else backtest_results.get('trades', [])
            if trades:
                trades_df = self._process_trades_to_dataframe(trades)
                if not trades_df.empty:
                    self.experiment_tracker.log_trades(trades_df)

            # Step 7: Calculate and log risk metrics
            # This is now handled by the BacktestEngine, we just log the results
            if backtest_results.risk_metrics:
                self.experiment_tracker.log_metrics(backtest_results.risk_metrics)

            # Step 8: Log strategy-specific metrics
            strategy_metrics = self._calculate_strategy_specific_metrics(
                strategy_signals, price_data, backtest_results
            )
            self.experiment_tracker.log_metrics(strategy_metrics)

            # Compile final results (adapted for new architecture)
            self.results = {
                'experiment_name': experiment_name,
                'config': self.config,
                'data_statistics': data_stats,
                'strategy_signals': strategy_signals,
                'backtest_results': backtest_results,
                'performance_metrics': backtest_results.performance_metrics,
                'risk_metrics': backtest_results.risk_metrics,
                'strategy_metrics': strategy_metrics,
                'execution_timestamp': datetime.now().isoformat()
            }

            # Phase 6 Enhanced: Log backtest results using experiment tracking interface
            self._log_enhanced_backtest_results(backtest_results)

            # Phase 6 Enhanced: Create and log backtest visualization charts
            self._create_enhanced_backtest_charts(backtest_results)

            # Save results locally
            self._save_results()

            logger.info(f"Strategy execution completed: {experiment_name}")
            return self.results

        except Exception as e:
            logger.error(f"Strategy execution failed: {e}")
            raise

    def _log_enhanced_backtest_results(self, backtest_results):
        """Log enhanced backtest results using experiment tracking interface."""
        try:
            # Extract performance metrics for logging
            performance_metrics = {}
            if hasattr(backtest_results, 'performance_metrics'):
                performance_metrics = backtest_results.performance_metrics
            elif isinstance(backtest_results, dict) and 'performance_metrics' in backtest_results:
                performance_metrics = backtest_results['performance_metrics']

            # Log key performance metrics
            if performance_metrics:
                self.log_backtest_results(performance_metrics)

            # Log portfolio history as artifact
            if hasattr(backtest_results, 'portfolio_history'):
                portfolio_history = backtest_results.portfolio_history
                if isinstance(portfolio_history, pd.DataFrame):
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                            portfolio_history.to_csv(f.name)
                            temp_path = f.name

                        self.experiment_tracker.log_artifact(
                            artifact_path=temp_path,
                            artifact_name="portfolio_history.csv",
                            artifact_type="data",
                            description="Portfolio value history over time"
                        )

                        import os
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.error(f"Failed to log portfolio history artifact: {e}")

            # Log trades as artifact
            if hasattr(backtest_results, 'trades'):
                trades = backtest_results.trades
                if isinstance(trades, pd.DataFrame) and not trades.empty:
                    try:
                        import tempfile
                        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                            trades.to_csv(f.name, index=False)
                            temp_path = f.name

                        self.experiment_tracker.log_artifact(
                            artifact_path=temp_path,
                            artifact_name="trades.csv",
                            artifact_type="data",
                            description="Complete trade log"
                        )

                        import os
                        os.unlink(temp_path)
                    except Exception as e:
                        logger.error(f"Failed to log trades artifact: {e}")

        except Exception as e:
            logger.error(f"Failed to log enhanced backtest results: {e}")

    def _create_enhanced_backtest_charts(self, backtest_results):
        """Create enhanced backtest visualization charts."""
        try:
            # Extract portfolio history
            portfolio_history = None
            if hasattr(backtest_results, 'portfolio_history'):
                portfolio_history = backtest_results.portfolio_history
            elif isinstance(backtest_results, dict) and 'portfolio_history' in backtest_results:
                portfolio_history = backtest_results['portfolio_history']

            if portfolio_history is None or portfolio_history.empty:
                logger.warning("No portfolio history available for charting")
                return

            # Extract trades
            trades = pd.DataFrame()  # Empty DataFrame as default
            if hasattr(backtest_results, 'trades'):
                trades = backtest_results.trades
            elif isinstance(backtest_results, dict) and 'trades' in backtest_results:
                trades = backtest_results['trades']

            # Extract benchmark data if available
            benchmark_df = None
            if hasattr(backtest_results, 'benchmark_data'):
                benchmark_df = backtest_results.benchmark_data
            elif isinstance(backtest_results, dict) and 'benchmark_data' in backtest_results:
                benchmark_df = backtest_results['benchmark_data']

            # Use the enhanced chart creation method
            self.plotter.create_and_log_charts(portfolio_history, trades, benchmark_df)

        except Exception as e:
            logger.error(f"Failed to create enhanced backtest charts: {e}")

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

        # Fetch price data with additional historical data for momentum calculations
        # Add lookback buffer based on strategy requirements
        lookback_buffer = getattr(self.strategy, 'lookback_days', 252)
        buffer_start_date = start_date - pd.Timedelta(days=lookback_buffer)

        logger.info(f"Fetching data from {buffer_start_date} to {end_date} "
                   f"(includes {lookback_buffer} days lookback buffer)")

        price_data = self.data_provider.get_historical_data(
            symbols=valid_symbols,
            start_date=buffer_start_date,
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
        """
        Calculate metrics specific to the strategy's signals, not portfolio performance.

        This method focuses on analyzing the characteristics of the signals generated
        by the strategy, such as position concentration and allocation to cash.
        All portfolio performance metrics (e.g., Sharpe, drawdown) are calculated
        within the BacktestEngine.
        
        NEW: Now leverages the strategy's built-in evaluation methods and
        PortfolioCalculator for comprehensive signal analysis.
        """
        metrics = {}

        if strategy_signals is None or strategy_signals.empty:
            return metrics

        try:
            # === Use PortfolioCalculator for comprehensive metrics ===
            from ..strategies.utils.portfolio_calculator import PortfolioCalculator
            
            # 1. Signal Quality Metrics (from strategy's evaluation)
            signal_quality = self.strategy.evaluate_signal_quality(strategy_signals)
            if signal_quality:
                metrics['signal_quality'] = signal_quality
                # Flatten for easier logging
                for key, value in signal_quality.items():
                    metrics[f'signal_{key}'] = value
            
            # 2. Position Metrics (from strategy's evaluation)
            position_metrics = self.strategy.analyze_positions(strategy_signals)
            if position_metrics:
                metrics['position_metrics'] = position_metrics
                # Flatten for easier logging
                for key, value in position_metrics.items():
                    metrics[f'position_{key}'] = value
            
            # 3. Concentration Risk
            concentration = self.strategy.calculate_concentration_risk(strategy_signals)
            metrics['concentration_risk_hhi'] = concentration
            
            # 4. Turnover Analysis
            turnover = PortfolioCalculator.calculate_turnover(strategy_signals)
            metrics['portfolio_turnover'] = turnover
            
            # 5. Legacy metrics for backward compatibility
            positions_per_period = strategy_signals.gt(0).sum(axis=1)
            metrics['avg_positions_held'] = positions_per_period.mean()
            metrics['max_positions_held'] = positions_per_period.max()
            metrics['min_positions_held'] = positions_per_period.min()

            # Signal turnover (legacy calculation for comparison)
            signal_changes = strategy_signals.diff().abs().sum().sum()
            total_signals = strategy_signals.abs().sum().sum()
            if total_signals > 0:
                metrics['signal_turnover_ratio'] = signal_changes / total_signals

            # Cash allocation statistics (if SHY is used as a cash proxy)
            if 'SHY' in strategy_signals.columns:
                cash_allocations = strategy_signals.get('SHY', pd.Series(0, index=strategy_signals.index))
                metrics['avg_cash_allocation'] = cash_allocations.mean()
                metrics['max_cash_allocation'] = cash_allocations.max()
            
            # 6. Comprehensive portfolio analysis
            portfolio_composition = PortfolioCalculator.analyze_portfolio_composition(
                price_data, strategy_signals, lookback_days=252
            )
            if portfolio_composition:
                # Store top/worst contributors
                metrics['top_contributors'] = portfolio_composition.get('top_contributors', [])
                metrics['worst_contributors'] = portfolio_composition.get('worst_contributors', [])
            
            logger.info("=" * 60)
            logger.info("STRATEGY EVALUATION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Signal Quality: {signal_quality}")
            logger.info(f"Position Metrics: {position_metrics}")
            logger.info(f"Concentration Risk (HHI): {concentration:.3f}")
            logger.info(f"Portfolio Turnover: {turnover:.3f}")
            logger.info("=" * 60)

        except Exception as e:
            logger.warning(f"Error calculating strategy-specific signal metrics: {e}", exc_info=True)

        return metrics

    def _save_results(self):
        """Save results to local files."""
        try:
            # Use backtest config for output settings
            backtest_config = self.configs.get('backtest')

            if backtest_config and backtest_config.save_results:
                # Create results directory
                results_dir = backtest_config.output_directory
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

                # Save portfolio history (adapted for new architecture)
                history_file = os.path.join(results_dir, f"{self.results['experiment_name']}_portfolio.csv")
                portfolio_history = self.results['backtest_results'].portfolio_values if hasattr(self.results['backtest_results'], 'portfolio_values') else self.results['backtest_results'].get('portfolio_history')
                if portfolio_history is not None:
                    if hasattr(portfolio_history, 'to_csv'):
                        portfolio_history.to_csv(history_file)
                    elif isinstance(portfolio_history, pd.DataFrame):
                        portfolio_history.to_csv(history_file)
                    logger.info(f"Portfolio history saved to {history_file}")

        except Exception as e:
            logger.error(f"Failed to save results: {e}")

    def cleanup(self):
        """Clean up resources."""
        if self.experiment_tracker:
            self.experiment_tracker.finish_run()

    def link_to_model_training_run(self, training_run_id: str):
        """
        Link the backtest run to a model training run.

        Args:
            training_run_id: The run ID of the model training experiment
        """
        if self.experiment_tracker:
            self.experiment_tracker.link_to_run(training_run_id, link_type="model_training")
            logger.info(f"Linked backtest to model training run: {training_run_id}")

    def log_backtest_results(self, backtest_results: Dict[str, Any]):
        """
        Log backtest results using the experiment tracking interface.

        Args:
            backtest_results: Dictionary containing backtest performance metrics
        """
        if not self.experiment_tracker:
            return

        # Extract key metrics
        metrics = {}
        for key, value in backtest_results.items():
            if isinstance(value, (int, float)):
                metrics[f"backtest_{key}"] = value

        if metrics:
            self.experiment_tracker.log_metrics(metrics)

        # Log detailed results as artifact
        try:
            import json
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(backtest_results, f, indent=2, default=str)
                temp_path = f.name

            self.experiment_tracker.log_artifact(
                artifact_path=temp_path,
                artifact_name="backtest_results.json",
                artifact_type="results",
                description="Complete backtest results"
            )

            # Clean up temp file
            import os
            os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Failed to log backtest results artifact: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
        return False