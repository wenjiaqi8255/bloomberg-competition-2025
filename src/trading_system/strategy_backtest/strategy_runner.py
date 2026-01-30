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
from trading_system.experiment_tracking import (
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
from trading_system.portfolio_construction.utils.weight_utils import WeightUtils

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

    def _filter_signals_by_rebalance_frequency(self, strategy_signals: pd.DataFrame, rebalance_frequency: str) -> pd.DataFrame:
        """
        Filter strategy signals DataFrame based on rebalance frequency.
        
        Args:
            strategy_signals: DataFrame with signals (dates as index)
            rebalance_frequency: Frequency string ("weekly", "monthly", "quarterly")
            
        Returns:
            Filtered DataFrame with only dates matching the rebalance frequency
        """
        if strategy_signals.empty:
            return strategy_signals
            
        # Convert index to DatetimeIndex if not already
        if not isinstance(strategy_signals.index, pd.DatetimeIndex):
            strategy_signals.index = pd.to_datetime(strategy_signals.index)
        
        dates = strategy_signals.index
        
        if rebalance_frequency == "weekly":
            # Select the last trading day of each week
            # Group by year-week and take the last date in each week
            # Use strftime('%Y-W%W') to get consistent week grouping
            filtered_dates = []
            current_week_key = None
            current_week_dates = []
            
            for date in dates:
                # Use year-week format for grouping (e.g., "2024-W01")
                week_key = date.strftime('%Y-W%W')
                
                if current_week_key is None:
                    current_week_key = week_key
                    current_week_dates = [date]
                elif week_key == current_week_key:
                    current_week_dates.append(date)
                else:
                    # Week changed, keep the last date of previous week
                    if current_week_dates:
                        filtered_dates.append(current_week_dates[-1])
                    current_week_key = week_key
                    current_week_dates = [date]
            
            # Don't forget the last week
            if current_week_dates:
                filtered_dates.append(current_week_dates[-1])
                
        elif rebalance_frequency == "monthly":
            # Select the last trading day of each month
            filtered_dates = []
            current_month = None
            current_month_dates = []
            
            for date in dates:
                year_month = (date.year, date.month)
                
                if current_month is None:
                    current_month = year_month
                    current_month_dates = [date]
                elif year_month == current_month:
                    current_month_dates.append(date)
                else:
                    # Month changed, keep the last date of previous month
                    if current_month_dates:
                        filtered_dates.append(current_month_dates[-1])
                    current_month = year_month
                    current_month_dates = [date]
            
            # Don't forget the last month
            if current_month_dates:
                filtered_dates.append(current_month_dates[-1])
                
        elif rebalance_frequency == "quarterly":
            # Select the last trading day of each quarter
            filtered_dates = []
            current_quarter = None
            current_quarter_dates = []
            
            for date in dates:
                quarter = (date.year, (date.month - 1) // 3 + 1)  # Q1=1, Q2=2, Q3=3, Q4=4
                
                if current_quarter is None:
                    current_quarter = quarter
                    current_quarter_dates = [date]
                elif quarter == current_quarter:
                    current_quarter_dates.append(date)
                else:
                    # Quarter changed, keep the last date of previous quarter
                    if current_quarter_dates:
                        filtered_dates.append(current_quarter_dates[-1])
                    current_quarter = quarter
                    current_quarter_dates = [date]
            
            # Don't forget the last quarter
            if current_quarter_dates:
                filtered_dates.append(current_quarter_dates[-1])
        else:
            # Unknown frequency, return original
            logger.warning(f"Unknown rebalance_frequency '{rebalance_frequency}', returning original signals")
            return strategy_signals
        
        # Filter the DataFrame to only include selected dates
        filtered_dates = pd.DatetimeIndex(filtered_dates)
        filtered_signals = strategy_signals.loc[filtered_dates]
        
        return filtered_signals

    def _apply_portfolio_construction(self, strategy_signals: pd.DataFrame, price_data: Dict[str, pd.DataFrame], start_date: datetime, rebalance_frequency: str = "daily") -> pd.DataFrame:
        """
        Apply portfolio construction to strategy signals with rebalance optimization.
        
        This method implements the correct portfolio construction logic:
        - Only executes portfolio construction on rebalance dates
        - Forward fills weights to non-rebalance dates
        - Ensures all weights are properly normalized (sum to 1.0)
        - Validates weight constraints (range [0, 1], no NaN values)
        
        Args:
            strategy_signals: Raw strategy signals DataFrame (all dates)
            price_data: Price data for all symbols
            start_date: Start date for portfolio construction
            rebalance_frequency: Rebalance frequency ("daily", "weekly", "monthly", "quarterly")
            
        Returns:
            Processed signals DataFrame with portfolio construction applied to all dates
            All weights are normalized and validated.
        """
        try:
            if self.portfolio_builder is None:
                logger.warning("Portfolio builder not initialized, returning original signals")
                return strategy_signals
            
            if strategy_signals.empty:
                return strategy_signals
            
            # Get rebalance dates based on frequency
            if rebalance_frequency == "daily":
                # For daily rebalancing, process all dates
                rebalance_dates = strategy_signals.index.tolist()
            else:
                # Get rebalance dates using existing filter method
                filtered_signals = self._filter_signals_by_rebalance_frequency(
                    strategy_signals.copy(), rebalance_frequency
                )
                rebalance_dates = filtered_signals.index.tolist()
            
            logger.info(f"   🔄 Portfolio construction: processing {len(rebalance_dates)} rebalance dates out of {len(strategy_signals)} total dates")
            
            # Initialize output DataFrame with same index and columns as input
            processed_signals = pd.DataFrame(index=strategy_signals.index, columns=strategy_signals.columns, dtype=float)
            
            # Dictionary to store portfolio weights for each rebalance date
            rebalance_weights: Dict[datetime, pd.Series] = {}
            
            # Process each rebalance date
            for i, rebalance_date in enumerate(rebalance_dates):
                try:
                    # Get signals for this rebalance date
                    date_signals = strategy_signals.loc[rebalance_date]
                    
                    # Log progress
                    if i < 3 or i % 10 == 0:
                        logger.info(f"   📅 Building portfolio for rebalance date {i+1}/{len(rebalance_dates)}: {rebalance_date.date()}")
                    
                    # Create portfolio construction request
                    request = PortfolioConstructionRequest(
                        date=rebalance_date,
                        universe=list(date_signals.index),
                        signals=date_signals,
                        price_data=price_data,
                        constraints={}
                    )
                    
                    # Build portfolio
                    portfolio_weights = self.portfolio_builder.build_portfolio(request)
                    
                    # CRITICAL FIX: Ensure all columns are properly initialized
                    # Portfolio weights may only contain a subset of symbols, but we need
                    # weights for ALL symbols in the universe (others should be 0.0)
                    if isinstance(portfolio_weights, pd.Series):
                        # Create a full weight vector with all symbols initialized to 0.0
                        full_weights = pd.Series(0.0, index=strategy_signals.columns, dtype=float)
                        
                        # Only update symbols that are in both portfolio_weights and strategy_signals.columns
                        common_symbols = portfolio_weights.index.intersection(strategy_signals.columns)
                        full_weights[common_symbols] = portfolio_weights[common_symbols]
                        
                        # Normalize using the centralized utility
                        portfolio_weights = WeightUtils.normalize_weights(full_weights)

                    elif isinstance(portfolio_weights, dict):
                        # Handle dict format: convert to Series with all symbols
                        full_weights = pd.Series(0.0, index=strategy_signals.columns, dtype=float)
                        for symbol, weight in portfolio_weights.items():
                            if symbol in full_weights.index:
                                full_weights[symbol] = weight
                        
                        # Normalize using the centralized utility
                        portfolio_weights = WeightUtils.normalize_weights(full_weights)
                    
                    # Store weights for this rebalance date
                    rebalance_weights[rebalance_date] = portfolio_weights
                    processed_signals.loc[rebalance_date] = portfolio_weights
                    
                except Exception as e:
                    logger.warning(f"Portfolio construction failed for rebalance date {rebalance_date}: {e}")
                    # Fall back to zero weights (not original signals, as signals are not weights!)
                    zero_weights = pd.Series(0.0, index=strategy_signals.columns, dtype=float)
                    processed_signals.loc[rebalance_date] = zero_weights
                    rebalance_weights[rebalance_date] = zero_weights
            
            # Forward fill weights for non-rebalance dates
            if rebalance_frequency != "daily" and len(rebalance_weights) > 0:
                logger.info(f"   🔄 Forward filling portfolio weights to {len(strategy_signals) - len(rebalance_dates)} non-rebalance dates")
                
                # Weights are already properly formatted above (all columns, normalized to sum to 1.0)
                # Forward fill using pandas (forward fill along index)
                processed_signals = processed_signals.ffill()
                
                # Backward fill for any dates before first rebalance date
                # Use the first rebalance date's weights for dates before it
                if len(rebalance_dates) > 0:
                    first_rebalance_date = rebalance_dates[0]
                    if first_rebalance_date in processed_signals.index:
                        first_weights = processed_signals.loc[first_rebalance_date]
                        # Fill all dates before first rebalance date with first rebalance weights
                        for date in processed_signals.index:
                            if date < first_rebalance_date:
                                processed_signals.loc[date] = first_weights
                
                # CRITICAL FIX: Do NOT use fillna(strategy_signals) as it replaces weights with signals!
                # Signals are NOT weights - they are alpha values or signal strengths!
                # Instead, fill any remaining NaN with 0.0 (no position)
                processed_signals = processed_signals.fillna(0.0)
                
                # Final validation: ensure weights sum to 1.0 for each date (with tolerance)
                weight_sums = processed_signals.sum(axis=1)
                invalid_dates = weight_sums[abs(weight_sums - 1.0) > 0.01]  # 1% tolerance
                if len(invalid_dates) > 0:
                    logger.warning(f"Found {len(invalid_dates)} dates with weights not summing to 1.0 (tolerance: 1%), normalizing...")
                    for date in invalid_dates.index:
                        row_sum = processed_signals.loc[date].sum()
                        if row_sum > 0:
                            processed_signals.loc[date] = processed_signals.loc[date] / row_sum
                        else:
                            # If all weights are 0, keep as 0 (no positions) - this is valid
                            logger.debug(f"Date {date} has all zero weights (no positions)")
                
                logger.debug(f"Forward fill completed. Weight sum stats: min={weight_sums.min():.4f}, max={weight_sums.max():.4f}, mean={weight_sums.mean():.4f}")
                
            else:
                # For daily rebalancing, ensure all dates have proper weights
                # Fill any NaN values with 0.0 (not with original signals!)
                processed_signals = processed_signals.fillna(0.0)
                
                # Validate and normalize weights for daily rebalancing
                for date in processed_signals.index:
                    row_sum = processed_signals.loc[date].sum()
                    if row_sum > 0 and abs(row_sum - 1.0) > 0.01:
                        processed_signals.loc[date] = processed_signals.loc[date] / row_sum
            
            # Validate portfolio weights before returning
            validation_passed = self._validate_portfolio_weights(processed_signals)
            if not validation_passed:
                logger.error("❌ Portfolio weight validation failed! This may indicate a bug.")
                # Still return the weights, but log the error for investigation
            
            logger.info(f"Portfolio construction applied successfully: {len(rebalance_dates)} rebalance dates, {len(processed_signals)} total dates")
            return processed_signals
            
        except Exception as e:
            logger.error(f"Portfolio construction failed: {e}")
            logger.info("Returning original signals")
            return strategy_signals
    
    def _validate_portfolio_weights(self, weights_df: pd.DataFrame) -> bool:
        """
        Validate portfolio weights for correctness.
        
        Checks:
        1. Weights are in range [0, 1]
        2. Weight sums are approximately 1.0 (within tolerance)
        3. No NaN values
        
        Args:
            weights_df: DataFrame with portfolio weights (dates x symbols)
            
        Returns:
            True if validation passes, False otherwise
        """
        if weights_df.empty:
            logger.warning("⚠️  Weight DataFrame is empty")
            return False
        
        # Check 1: Weight range [0, 1]
        if (weights_df < 0).any().any():
            negative_count = (weights_df < 0).sum().sum()
            logger.error(f"❌ Found {negative_count} negative weights!")
            return False
        
        if (weights_df > 1).any().any():
            over_one_count = (weights_df > 1).sum().sum()
            logger.error(f"❌ Found {over_one_count} weights > 1.0!")
            return False
        
        # Check 2: Weight sums approximately 1.0
        weight_sums = weights_df.sum(axis=1)
        tolerance = 0.01  # 1% tolerance
        invalid_sums = weight_sums[(weight_sums < 1 - tolerance) | (weight_sums > 1 + tolerance)]
        
        if len(invalid_sums) > 0:
            # Allow zero-weight dates (no positions)
            non_zero_dates = invalid_sums[invalid_sums.abs() > tolerance]
            if len(non_zero_dates) > 0:
                logger.warning(f"⚠️  Found {len(non_zero_dates)} dates with weight sums outside [0.99, 1.01]")
                logger.debug(f"   Invalid sums: min={invalid_sums.min():.4f}, max={invalid_sums.max():.4f}")
                # Don't return False here, as this might be acceptable (e.g., cash positions)
        
        # Check 3: No NaN values
        if weights_df.isna().any().any():
            nan_count = weights_df.isna().sum().sum()
            logger.error(f"❌ Found {nan_count} NaN values in weights!")
            return False
        
        logger.debug(f"✅ Weight validation passed: {len(weights_df)} dates, weight sum range [{weight_sums.min():.4f}, {weight_sums.max():.4f}]")
        return True
    
    def _sanity_check_weights(self, weights_df: pd.DataFrame, original_signals: pd.DataFrame = None) -> None:
        """
        Sanity check: Ensure weights are not equal to signals.
        
        This helps detect if the bug where signals are used as weights still exists.
        
        Args:
            weights_df: DataFrame with portfolio weights
            original_signals: Original strategy signals (optional, for comparison)
        """
        if original_signals is None or original_signals.empty:
            logger.debug("Skipping sanity check: original signals not available")
            return
        
        if weights_df.empty:
            logger.debug("Skipping sanity check: weights DataFrame is empty")
            return
        
        # Check if weights are identical to signals (would indicate bug)
        try:
            # Align indices and columns
            common_index = weights_df.index.intersection(original_signals.index)
            common_columns = weights_df.columns.intersection(original_signals.columns)
            
            if len(common_index) == 0 or len(common_columns) == 0:
                logger.debug("Skipping sanity check: no common indices/columns")
                return
            
            weights_subset = weights_df.loc[common_index, common_columns]
            signals_subset = original_signals.loc[common_index, common_columns]
            
            # Check if they are approximately equal (within numerical precision)
            # Use a small tolerance to account for floating point errors
            tolerance = 1e-6
            are_equal = (weights_subset - signals_subset).abs().max().max() < tolerance
            
            if are_equal:
                logger.error("❌ CRITICAL BUG DETECTED: Weights are identical to signals!")
                logger.error("   This indicates the bug where signals are used as weights still exists.")
                logger.error("   Portfolio construction may not be working correctly.")
            else:
                logger.debug("✅ Sanity check passed: Weights are different from signals (as expected)")
        
        except Exception as e:
            logger.debug(f"Sanity check encountered an error (non-critical): {e}")

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
                backtest_config=backtest_config
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
            # Portfolio construction implements the correct financial logic:
            # - Only computes weights on rebalance dates (as per rebalance_frequency)
            # - Forward fills weights to non-rebalance dates
            # - Ensures all weights are properly normalized and validated
            # All dates are passed to backtest engine, which handles rebalance threshold logic internally.
            if self.portfolio_builder and not strategy_signals.empty:
                logger.info("🔧 APPLYING PORTFOLIO CONSTRUCTION...")
                logger.info(f"   📊 Input signals shape: {strategy_signals.shape}")
                logger.info(f"   📅 Date range: {strategy_signals.index[0].date()} to {strategy_signals.index[-1].date()}")
                logger.info(f"   🔄 Rebalance frequency: {backtest_config.rebalance_frequency}")
                
                # Save original signals for sanity check
                original_signals = strategy_signals.copy()
                
                strategy_signals = self._apply_portfolio_construction(
                    strategy_signals, 
                    price_data, 
                    backtest_config.start_date,
                    rebalance_frequency=backtest_config.rebalance_frequency
                )
                logger.info(f"   ✅ Portfolio construction completed. Output shape: {strategy_signals.shape}")
                
                # Sanity check: Verify weights are not equal to signals (would indicate bug)
                self._sanity_check_weights(strategy_signals, original_signals)

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
                   end_date: datetime, backtest_config) -> tuple:
        """
        Fetch price data for asset universe and benchmark.
        """
        logger.info(f"Fetching data from {start_date} to {end_date}")

        # Remove the validate_symbol step - it's redundant and bypasses cache
        # get_historical_data already handles errors and returns only successful symbols
        
        # Fetch price data with additional historical data for momentum calculations
        lookback_buffer = getattr(self.strategy, 'lookback_days', 252)
        buffer_start_date = start_date - pd.Timedelta(days=lookback_buffer)

        logger.info(f"Fetching data from {buffer_start_date} to {end_date} "
                   f"(includes {lookback_buffer} days lookback buffer)")

        # Directly fetch data - get_historical_data will:
        # 1. Check L2 cache first (disk cache)
        # 2. Check L1 cache (in-memory)
        # 3. Only make network requests for cache misses
        # 4. Return only successful symbols in the dictionary
        price_data = self.data_provider.get_historical_data(
            symbols=universe,  # ✅ Pass all symbols directly
            start_date=buffer_start_date,
            end_date=end_date
        )

        if not price_data:
            raise ValueError("Failed to fetch price data for any symbols")

        # Log which symbols were successfully fetched
        successful_symbols = list(price_data.keys())
        failed_symbols = [s for s in universe if s not in successful_symbols]
        
        if failed_symbols:
            logger.warning(f"Failed to fetch data for {len(failed_symbols)} symbols: {failed_symbols}")
        
        logger.info(f"Successfully fetched data for {len(successful_symbols)} symbols")

        # Fetch benchmark data
        # Priority: benchmark config > benchmark_symbol (backward compatible)
        benchmark_data = None
        
        if backtest_config.benchmark:
            # Use benchmark configuration (CSV or symbol)
            benchmark_config = backtest_config.benchmark
            
            if benchmark_config.source == "csv":
                # Load from CSV file
                from trading_system.data.utils.benchmark_loader import load_benchmark_from_csv
                
                if not benchmark_config.csv_path:
                    logger.warning("benchmark.csv_path not specified, skipping CSV benchmark")
                else:
                    try:
                        benchmark_data = load_benchmark_from_csv(
                            csv_path=benchmark_config.csv_path,
                            start_date=start_date,
                            end_date=end_date
                        )
                        if benchmark_data is not None:
                            logger.info(f"Loaded benchmark from CSV: {benchmark_config.csv_path}")
                        else:
                            logger.warning(f"Failed to load benchmark from CSV: {benchmark_config.csv_path}")
                    except Exception as e:
                        logger.error(f"Error loading benchmark from CSV: {e}")
                        
            elif benchmark_config.source == "symbol":
                # Use symbol from benchmark config or fallback to benchmark_symbol
                benchmark_symbol = benchmark_config.symbol or backtest_config.benchmark_symbol
                
                if benchmark_symbol and benchmark_symbol in universe:
                    benchmark_data = price_data[benchmark_symbol]
                elif benchmark_symbol:
                    # Fetch benchmark separately if not in universe
                    benchmark_data_dict = self.data_provider.get_historical_data(
                        symbols=[benchmark_symbol],
                        start_date=start_date,
                        end_date=end_date
                    )
                    benchmark_data = benchmark_data_dict.get(benchmark_symbol)
                    
        elif backtest_config.benchmark_symbol:
            # Backward compatible: use benchmark_symbol string
            benchmark_symbol = backtest_config.benchmark_symbol
            if benchmark_symbol in universe:
                benchmark_data = price_data[benchmark_symbol]
            else:
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