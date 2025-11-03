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
from typing import Dict, List, Any, Optional

import pandas as pd

from ..config.factory import ConfigFactory
from ..data.yfinance_provider import YFinanceProvider
# New backtesting architecture
from ..backtesting import BacktestEngine
from ..strategies.factory import StrategyFactory
from src.trading_system.experiment_tracking import (
    ExperimentTrackerInterface,
    NullExperimentTracker,
    WandBExperimentTracker,
    create_backtest_config
)
from ..types import TradingSignal, SignalType
# Portfolio construction imports
from ..portfolio_construction import (
    IPortfolioBuilder, PortfolioConstructionRequest, PortfolioBuilderFactory
)

logger = logging.getLogger(__name__)


def create_strategy_runner(config_path: str = None,
                           config_obj: Dict[str, Any] = None,
                           providers: Dict[str, Any] = None,
                           experiment_tracker: ExperimentTrackerInterface = None,
                           use_wandb: bool = True) -> 'StrategyRunner':
    """
    Factory function to create StrategyRunner with desired experiment tracking.

    Args:
        config_path: Path to configuration file
        config_obj: A dictionary of configuration objects (e.g., from orchestrator)
        providers: A dictionary of pre-initialized data providers.
        experiment_tracker: Custom experiment tracker instance
        use_wandb: Whether to use WandB tracker (ignored if experiment_tracker is provided)

    Returns:
        StrategyRunner instance with configured experiment tracking
    """
    if experiment_tracker is not None:
        return StrategyRunner(
            config_path=config_path,
            config_obj=config_obj,
            providers=providers,
            experiment_tracker=experiment_tracker
        )

    if use_wandb:
        try:
            wandb_tracker = WandBExperimentTracker(
                project_name='bloomberg-competition',
                fail_silently=True
            )
            return StrategyRunner(
                config_path=config_path,
                config_obj=config_obj,
                providers=providers,
                experiment_tracker=wandb_tracker
            )
        except Exception:
            logger.warning("Failed to create WandB tracker, using null tracker")

    # Default to null tracker
    null_tracker = NullExperimentTracker()
    return StrategyRunner(
        config_path=config_path,
        config_obj=config_obj,
        providers=providers,
        experiment_tracker=null_tracker
    )


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

    def __init__(self,
                 config_path: str = None,
                 config_obj: Dict[str, Any] = None,
                 providers: Dict[str, Any] = None,
                 experiment_tracker: ExperimentTrackerInterface = None):
        """
        Initialize the StrategyRunner.

        Args:
            config_path: Path to the main YAML configuration file.
            config_obj: A dictionary of configuration objects, used in priority over config_path.
            providers: A dictionary of pre-initialized data providers.
            experiment_tracker: An instance of an experiment tracker (e.g., WandB).
                               If not provided, one will be created.
        """
        if config_obj:
            logger.info("Initializing StrategyRunner from configuration object.")
            logger.info(f"🔧 DEBUG: config_obj keys = {list(config_obj.keys())}")
            if 'strategy' in config_obj:
                logger.info(f"🔧 DEBUG: strategy config type = {type(config_obj['strategy'])}")
                logger.info(f"🔧 DEBUG: strategy config parameters = {getattr(config_obj['strategy'], 'parameters', 'NO PARAMETERS')}")
            self.configs = config_obj
        elif config_path:
            logger.info(f"Initializing StrategyRunner from configuration file: {config_path}")
            self.configs = ConfigFactory.load_all_configs(config_path)
        else:
            logger.warning("No configuration provided. Using default empty configs.")
            self.configs = {}

        # Initialize components
        self.providers = providers or {}
        self.data_provider = self.providers.get('data_provider')
        self.factor_data_provider = self.providers.get('factor_data_provider')  # ✅ 添加factor_data_provider
        self.strategy = None
        self.backtest_engine = None
        self.experiment_tracker = experiment_tracker
        self.plotter = None
        self.portfolio_builder: Optional[IPortfolioBuilder] = None

        # Results storage
        self.results = {}
        self.is_initialized = False

    def _convert_signals_to_unified_format(self, strategy_signals: pd.DataFrame, price_data: Dict[str, pd.DataFrame] = None) -> Dict[datetime, List[TradingSignal]]:
        """
        Convert DataFrame strategy signals to unified format Dict[datetime, List[TradingSignal]].

        Args:
            strategy_signals: DataFrame with strategy signals (old format)
            price_data: Dictionary of price data for getting signal prices

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

                # Get price for the signal - use closing price on signal date
                signal_price = 0.0
                if price_data and symbol in price_data and date in price_data[symbol].index:
                    signal_price = float(price_data[symbol].loc[date, 'Close'])
                elif price_data and symbol in price_data:
                    # If exact date not found, use the closest previous date
                    symbol_data = price_data[symbol]
                    available_dates = symbol_data.index[symbol_data.index <= date]
                    if len(available_dates) > 0:
                        closest_date = available_dates.max()
                        signal_price = float(symbol_data.loc[closest_date, 'Close'])
                    else:
                        logger.warning(f"No price data available for {symbol} on or before {date}, using default price 1.0")
                        signal_price = 1.0
                else:
                    logger.warning(f"No price data available for {symbol}, using default price 1.0")
                    signal_price = 1.0

                # Calculate signal strength and ensure it's within [0,1] range
                signal_strength = abs(float(signal_value))
                # Clamp to [0,1] range to satisfy TradingSignal validation
                signal_strength = max(0.0, min(1.0, signal_strength))
                confidence = max(0.0, min(1.0, abs(float(signal_value))))  # Also clamp confidence

                # Create TradingSignal object
                trading_signal = TradingSignal(
                    symbol=symbol,
                    signal_type=signal_type,
                    strength=signal_strength,
                    timestamp=date,
                    price=signal_price,
                    confidence=confidence
                )

                signals_for_date.append(trading_signal)

            if signals_for_date:
                unified_signals[date] = signals_for_date

        return unified_signals

    def _initialize_wandb_run(self):
        """Initialize WandB run early for logging initialization process."""
        try:
            # 如果 experiment_tracker 还没有初始化，先初始化它
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
            
            # 创建临时的 experiment config 用于初始化日志
            from ..experiment_tracking import create_backtest_config
            temp_config = create_backtest_config(
                project_name='bloomberg-competition',
                strategy_name='initializing',
                strategy_config={},
                tags=['initialization'],
                group=None
            )
            temp_config.experiment_name = f"initialization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 初始化 WandB run
            run_id = self.experiment_tracker.init_run(temp_config)
            logger.info(f"Started initialization run: {run_id}")
            self._wandb_run_created = True
            
        except Exception as e:
            logger.warning(f"Failed to initialize WandB run for initialization logging: {e}")
            # 继续执行，不中断初始化过程
            self._wandb_run_created = False

    def initialize(self):
        """Initialize all components based on configuration."""
        try:
            # 先创建 WandB run，这样初始化日志就能被记录
            self._initialize_wandb_run()
            
            logger.info("Initializing strategy runner components...")

            # Initialize data provider if not already provided
            if self.data_provider is None:
                logger.info("Data provider not pre-initialized. Creating default YFinanceProvider.")
                self.data_provider = YFinanceProvider(
                    max_retries=3,
                    retry_delay=1.0,
                    request_timeout=30
                )
                self.providers['data_provider'] = self.data_provider

            # Initialize strategy with config objects
            strategy_config_dict = self.configs['strategy'].parameters.copy()  # Make a copy to avoid mutation
            
            # 添加容错处理 strategy_type 访问
            try:
                strategy_config_dict['type'] = self.configs['strategy'].strategy_type.value
            except AttributeError:
                # 如果 strategy_type 不存在，使用 type 字段
                strategy_config_dict['type'] = self.configs['strategy'].type
                logger.warning("strategy_type property not available, using type field directly")
            
            strategy_config_dict['name'] = self.configs['strategy'].name
            # Include universe from config object if not already in parameters
            if 'universe' not in strategy_config_dict and self.configs['strategy'].universe:
                strategy_config_dict['universe'] = self.configs['strategy'].universe
            
            self.strategy = StrategyFactory.create_from_config(
                strategy_config_dict, providers=self.providers
            )

            # Initialize backtest engine using config object
            backtest_config = self.configs.get('backtest')
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
            
            # Initialize portfolio builder if configured
            self._initialize_portfolio_builder()
  
            self.is_initialized = True
            logger.info("Strategy runner initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize strategy runner: {e}", exc_info=True)
            raise

    def _initialize_portfolio_builder(self):
        """Initialize portfolio builder from strategy configuration."""
        strategy_config = self.configs.get('strategy')
        if not strategy_config:
            logger.info("No strategy config found, using default signal processing")
            return
            
        # Get portfolio construction config from top-level attribute
        portfolio_construction_config = getattr(strategy_config, 'portfolio_construction', None)
        
        if not portfolio_construction_config:
            logger.info("No portfolio_construction config found, using default signal processing")
            return
            
        # Create portfolio builder using factory
        self.portfolio_builder = PortfolioBuilderFactory.create_builder(
            portfolio_construction_config,
            factor_data_provider=self.factor_data_provider
        )
        method = getattr(portfolio_construction_config, 'method', 'unknown')
        logger.info(f"✅ Initialized portfolio builder with method: {method}")

    def _apply_portfolio_construction(self, strategy_signals: pd.DataFrame, price_data: Dict[str, pd.DataFrame], start_date: datetime) -> pd.DataFrame:
        """
        Apply portfolio construction to strategy signals.
        
        Args:
            strategy_signals: Raw strategy signals DataFrame
            price_data: Price data for all symbols
            start_date: Start date for portfolio construction
            
        Returns:
            Processed signals DataFrame with portfolio construction applied
        """
        try:
            if self.portfolio_builder is None:
                logger.warning("Portfolio builder not initialized, returning original signals")
                return strategy_signals
                
            # Create portfolio construction request
            # For now, we'll process signals date by date
            processed_signals = pd.DataFrame(index=strategy_signals.index, columns=strategy_signals.columns)
            
            for i, date in enumerate(strategy_signals.index):
                try:
                    # Get signals for this date
                    date_signals = strategy_signals.loc[date]
                    
                    # Log progress for first few dates and every 10th date
                    if i < 3 or i % 10 == 0:
                        logger.info(f"   📅 Processing date {i+1}/{len(strategy_signals)}: {date.date()}")
                    
                    # Create portfolio construction request
                    request = PortfolioConstructionRequest(
                        date=date,
                        universe=list(date_signals.index),
                        signals=date_signals,
                        price_data=price_data,
                        constraints={}
                    )
                    
                    # Build portfolio
                    portfolio_weights = self.portfolio_builder.build_portfolio(request)
                    
                    # Update processed signals
                    processed_signals.loc[date] = portfolio_weights
                    
                except Exception as e:
                    logger.warning(f"Portfolio construction failed for date {date}: {e}")
                    # Fall back to original signals for this date
                    processed_signals.loc[date] = strategy_signals.loc[date]
            
            logger.info(f"Portfolio construction applied successfully to {len(processed_signals)} dates")
            return processed_signals
            
        except Exception as e:
            logger.error(f"Portfolio construction failed: {e}")
            logger.info("Returning original signals")
            return strategy_signals

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
            # 检查是否已经有 WandB run，如果没有则创建一个正式的 run
            if not hasattr(self, '_wandb_run_created') or not self._wandb_run_created:
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

                # 添加容错处理 strategy_type 访问
                try:
                    strategy_type_tag = strategy_config.strategy_type.value if strategy_config else 'unknown'
                except AttributeError:
                    strategy_type_tag = strategy_config.type if strategy_config else 'unknown'

                experiment_config = create_backtest_config(
                    project_name='bloomberg-competition',
                    strategy_name=self.strategy.get_name(),
                    strategy_config=hyperparameters,
                    tags=[strategy_type_tag],
                    group=None
                )
                experiment_config.experiment_name = experiment_name

                # Initialize experiment with new interface
                run_id = self.experiment_tracker.init_run(experiment_config)
                logger.info(f"Started experiment run: {run_id}")
                self._wandb_run_created = True
            else:
                logger.info("Using existing WandB run for strategy execution")

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

            # Step 3: Prepare complete pipeline data (including factor data if available)
            pipeline_data = self._prepare_pipeline_data(price_data, backtest_config.start_date, backtest_config.end_date)

            # Step 4: Generate strategy signals with complete pipeline data
            strategy_signals = self.strategy.generate_signals(
                pipeline_data=pipeline_data,
                start_date=backtest_config.start_date,
                end_date=backtest_config.end_date
            )

            # Step 3.5: Apply portfolio construction if configured
            if self.portfolio_builder and not strategy_signals.empty:
                logger.info("🔧 APPLYING PORTFOLIO CONSTRUCTION...")
                logger.info(f"   📊 Input signals shape: {strategy_signals.shape}")
                logger.info(f"   📅 Date range: {strategy_signals.index[0].date()} to {strategy_signals.index[-1].date()}")
                strategy_signals = self._apply_portfolio_construction(strategy_signals, price_data, backtest_config.start_date)
                logger.info(f"   ✅ Portfolio construction completed. Output shape: {strategy_signals.shape}")

            # Step 4: Convert signals to unified format and run backtest
            unified_strategy_signals = self._convert_signals_to_unified_format(strategy_signals, price_data)
            
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
                'config': self.configs,
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

            # Validate results before saving
            try:
                from trading_system.validation import ExperimentResultValidator

                # Construct validation-compatible result
                validation_dict = {
                    'experiment_name': experiment_name,
                    'trained_model_id': self.results.get('config', {}).get('strategy', {}).get('parameters', {}).get('model_id', 'unknown'),
                    'performance_metrics': self.results['performance_metrics'],
                    'status': 'SUCCESS'
                }

                validator = ExperimentResultValidator()
                validation_result = validator.validate(validation_dict)

                if validation_result.has_warnings():
                    for warning in validation_result.get_warnings():
                        logger.warning(f"Result validation warning: {warning.message}")
            except ImportError:
                logger.warning("Experiment result validation not available, skipping validation")

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

    def _prepare_pipeline_data(self, price_data: Dict[str, pd.DataFrame], start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """
        Prepare complete pipeline data including factor data if available.

        This method implements the elegant architectural solution where StrategyRunner
        prepares all necessary data (price_data + factor_data) for strategies,
        following SOLID, KISS, YAGNI, DRY principles.

        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            start_date: Start date for data preparation
            end_date: End date for data preparation

        Returns:
            Complete pipeline data dictionary with price_data and optionally factor_data
        """
        try:
            logger.info(f"[StrategyRunner] Preparing complete pipeline data...")

            # Start with basic structure
            pipeline_data = {
                'price_data': price_data
            }

            # ✅ REFACTORED: Use StrategyRunner's own factor_data_provider
            # StrategyRunner prepares complete data, strategies don't hold providers
            if self.factor_data_provider is not None:
                logger.info(f"[StrategyRunner] Factor data provider found, fetching factor data...")

                try:
                    # Get symbols from price data
                    symbols = list(price_data.keys())
                    logger.info(f"[StrategyRunner] Fetching factor data for {len(symbols)} symbols...")

                    # Fetch factor data for the same period
                    factor_data = self.factor_data_provider.get_data(
                        start_date=start_date,
                        end_date=end_date
                    )

                    if factor_data is not None and not factor_data.empty:
                        pipeline_data['factor_data'] = factor_data
                        logger.info(f"[StrategyRunner] ✅ Factor data added: {factor_data.shape}")
                        logger.info(f"[StrategyRunner] Factor data columns: {list(factor_data.columns)}")
                    else:
                        logger.warning(f"[StrategyRunner] Factor data provider returned empty data")

                except Exception as e:
                    logger.error(f"[StrategyRunner] Failed to fetch factor data: {e}")
                    logger.info(f"[StrategyRunner] Continuing without factor data...")
            else:
                logger.info(f"[StrategyRunner] No factor data provider available, using price data only")

            logger.info(f"[StrategyRunner] Pipeline data prepared with keys: {list(pipeline_data.keys())}")
            return pipeline_data

        except Exception as e:
            logger.error(f"[StrategyRunner] ❌ Failed to prepare pipeline data: {e}")
            logger.error(f"[StrategyRunner] Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"[StrategyRunner] Traceback: {traceback.format_exc()}")
            # Fallback to just price data
            return {'price_data': price_data}

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
            trade_dict = trade.__dict__.copy()
            # Simple P&L calculation (would need more sophisticated calculation in practice)
            trade_dict['pnl'] = 0  # Placeholder
            trades_with_pnl.append(trade_dict)

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

                # Convert numpy types to JSON-serializable with high precision
                def json_serializer(obj):
                    """JSON serializer that handles numpy types with high precision."""
                    import numpy as np
                    import pandas as pd

                    if isinstance(obj, np.integer):
                        return int(obj)
                    elif isinstance(obj, np.floating):
                        # Convert to float with full precision - don't round small values to zero
                        return float(obj)
                    elif isinstance(obj, np.ndarray):
                        return obj.tolist()
                    elif isinstance(obj, pd.Timestamp):
                        return obj.isoformat()
                    elif hasattr(obj, '__dict__'):
                        # For custom objects, try to convert to dict
                        try:
                            return str(obj)
                        except:
                            return None
                    else:
                        return str(obj)

                # Create a deep copy and manually format performance metrics to ensure precision
                import copy
                results_copy = copy.deepcopy(self.results)

                # Ensure performance metrics retain precision
                if 'performance_metrics' in results_copy:
                    for key, value in results_copy['performance_metrics'].items():
                        if isinstance(value, (int, float)) and abs(value) < 0.0001:
                            # For very small values, ensure they don't get rounded to zero
                            results_copy['performance_metrics'][key] = float(value)

                results_json = json.loads(json.dumps(results_copy, default=json_serializer))

                with open(results_file, 'w') as f:
                    # Use high precision for JSON output to prevent small values being rounded to zero
                    json.dump(results_json, f, indent=2, allow_nan=True)

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