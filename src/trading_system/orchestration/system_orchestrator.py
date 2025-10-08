"""
System Orchestrator for Production Trading
==========================================

This module provides the `SystemOrchestrator`, the master coordinator for running
a complete, multi-strategy trading system in a production or simulated live
environment. It embodies the principle of Separation of Concerns by delegating
specific tasks to specialized components.

Core Responsibilities:
----------------------
1.  **Component Coordination**: It does not perform any single task itself, but
    instead manages the flow of data and commands between various sub-components.
2.  **Lifecycle Management**: Manages the step-by-step execution of the trading
    lifecycle for a given point in time (e.g., daily rebalancing).
3.  **State Management**: Holds the current state of the system, including the
    live portfolio and historical execution results.

Trading Lifecycle Orchestrated:
-------------------------------
For each run (e.g., each day), the orchestrator calls its components in a
specific sequence:
1.  `StrategyCoordinator`: Gathers trading signals from all active strategies
    (e.g., a core and a satellite strategy) and resolves any conflicts.
2.  `CapitalAllocator`: Decides how to allocate capital between the different
    strategies based on the system's configuration.
3.  `ComplianceMonitor`: Checks the proposed trades and resulting portfolio
    against a set of rules (e.g., IPS, risk limits).
4.  `TradeExecutor`: Takes the final, approved list of trades and executes them
    (either in simulation or against a live brokerage).
5.  `PerformanceReporter`: Calculates and reports on the performance of the
    overall system.

Usage Example:
--------------
.. code-block:: python

    from trading_system.config.system import SystemConfig
    from trading_system.orchestration import SystemOrchestrator

    # Load system configuration from a file
    system_config = SystemConfig.from_yaml("configs/system_config.yaml")

    # Initialize the orchestrator
    orchestrator = SystemOrchestrator(system_config=system_config)
    orchestrator.initialize_system()

    # Run the system for a specific date
    today = datetime(2025, 10, 3)
    result = orchestrator.run_system(date=today)

    if result.is_successful and not result.has_compliance_issues:
        print("System executed successfully.")

Distinction from StrategyRunner:
--------------------------------
- **SystemOrchestrator**: Manages a **multi-strategy portfolio**. It is designed
  for production-level complexity, including compliance and capital allocation.
- **StrategyRunner**: Focuses on backtesting a **single strategy**. It is a
  tool for research and validation, not for live portfolio management.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd

from .components.coordinator import StrategyCoordinator, CoordinatorConfig
from .components.allocator import CapitalAllocator, AllocationConfig, StrategyAllocation
from .components.compliance import ComplianceMonitor, ComplianceRules, ComplianceReport, ComplianceStatus, StrategyAllocationRule
from .components.executor import TradeExecutor, ExecutionConfig
from .components.reporter import PerformanceReporter, ReportConfig

from ..strategies.base_strategy import BaseStrategy
from ..types.portfolio import Portfolio
from ..types.data_types import TradingSignal
from ..types.signals import SignalType
from ..config.system import SystemConfig
from ..utils.risk import RiskCalculator, LedoitWolfCovarianceEstimator
from ..data.stock_classifier import StockClassifier
from .meta_model import MetaModel
from ..optimization.optimizer import PortfolioOptimizer

logger = logging.getLogger(__name__)


@dataclass
class SystemResult:
    """Result from system execution."""
    timestamp: datetime
    status: str
    signals_summary: Dict[str, Any]
    trades_summary: Dict[str, Any]
    portfolio_summary: Dict[str, Any]
    allocation_summary: Dict[str, Any]
    compliance_report: ComplianceReport
    performance_report: Any
    recommendations: List[str]

    @property
    def is_successful(self) -> bool:
        """Check if system execution was successful."""
        return self.status == "success"

    @property
    def has_compliance_issues(self) -> bool:
        """Check if there are compliance issues."""
        return not self.compliance_report.is_compliant

    @classmethod
    def error_result(cls, error_message: str) -> "SystemResult":
        """Create an error result for the system."""
        return cls(
            timestamp=datetime.now(),
            status="error",
            signals_summary={},
            trades_summary={},
            portfolio_summary={},
            allocation_summary={},
            compliance_report=ComplianceReport(
                timestamp=datetime.now(),
                overall_status=ComplianceStatus.CRITICAL,
                total_violations=1,
                violations=[],
                warnings=[],
                recommendations=[error_message],
                portfolio_summary={}
            ),
            performance_report=None,
            recommendations=[error_message]
        )


class SystemOrchestrator:
    """
    Coordinates specialized components to run a complete multi-strategy system.

    This class acts as the "brain" of the live trading system, but it delegates
    all the "thinking" to its components. Its sole responsibility is to ensure
    that each component is called in the correct order and that data flows
    between them correctly. This design makes the system highly modular,
    testable, and extensible.
    """

    def __init__(self,
                 system_config: SystemConfig,
                 strategies: List[BaseStrategy],
                 meta_model: MetaModel, # New
                 stock_classifier: StockClassifier, # Now required
                 custom_configs: Optional[Dict[str, Any]] = None):
        """
        Initialize System Orchestrator with a modern portfolio construction pipeline.

        Args:
            system_config: System configuration.
            strategies: List of trading strategies.
            meta_model: The meta-model for combining strategy signals.
            stock_classifier: Classifier for box-based constraints.
            custom_configs: Custom configuration overrides for components.
        
        Raises:
            ValueError: If strategies list is empty.
        """
        if not strategies:
            raise ValueError("At least one strategy must be provided")
        
        self.config = system_config
        self.strategies = strategies
        self.meta_model = meta_model
        self.stock_classifier = stock_classifier
        self.custom_configs = custom_configs or {}

        # Merge system config into custom configs to ensure short selling settings are passed through
        self.custom_configs['enable_short_selling'] = system_config.enable_short_selling

        # Initialize specialized components
        self._initialize_components()

        # System state
        self.current_portfolio: Optional[Portfolio] = None
        self.execution_history: List[SystemResult] = []

        logger.info(f"Initialized SystemOrchestrator with {len(self.strategies)} strategies and modern pipeline.")

    def _initialize_components(self) -> None:
        """Initialize all specialized components."""
        # --- Existing Components ---
        # Strategy Coordinator
        coordinator_config = CoordinatorConfig(
            max_signals_per_day=self.custom_configs.get('max_signals_per_day', 50),
            signal_conflict_resolution=self.custom_configs.get('signal_conflict_resolution', 'merge'),
            capacity_scaling=self.custom_configs.get('capacity_scaling', True),
            strategy_priority={s.name: i + 1 for i, s in enumerate(self.strategies)} # Simplified priority
        )
        self.coordinator = StrategyCoordinator(
            strategies=self.strategies,
            config=coordinator_config
        )

        # Trade Executor
        execution_config = ExecutionConfig(
            max_order_size_percent=self.custom_configs.get('max_order_size_percent', 1.0),
            commission_rate=self.custom_configs.get('commission_rate', 0.001),
            max_positions_per_day=self.custom_configs.get('max_positions_per_day', 10)
        )
        self.trade_executor = TradeExecutor(execution_config)

        # Performance Reporter
        report_config = ReportConfig(
            daily_reports=self.custom_configs.get('daily_reports', True),
            save_to_file=self.custom_configs.get('save_reports', True),
            output_directory=self.custom_configs.get('report_output_dir', 'reports')
        )
        self.performance_reporter = PerformanceReporter(report_config)
        
        # --- New & Repurposed Components for 7-Stage Pipeline ---
        
        # Covariance Estimator (from risk.py)
        self.covariance_estimator = LedoitWolfCovarianceEstimator(
            lookback_days=self.custom_configs.get('covariance_lookback_days', 252)
        )
        
        # Portfolio Optimizer (New Component)
        # Support both 'portfolio_optimization' and legacy 'optimizer_config'
        optimizer_config = self.custom_configs.get(
            'portfolio_optimization',
            self.custom_configs.get('optimizer_config', {
                'method': 'mean_variance',
                'risk_aversion': 2.0
            })
        )
        # Add short selling configuration to optimizer config
        optimizer_config['enable_short_selling'] = self.custom_configs.get('enable_short_selling', False)
        self.portfolio_optimizer = PortfolioOptimizer(optimizer_config)
        logger.info(f"Portfolio optimizer initialized with method='{self.portfolio_optimizer.method}'")

        # Compliance Monitor (Modified to use box limits)
        self.box_limits = self.custom_configs.get('box_limits', {})
        compliance_rules = ComplianceRules(
            max_single_position_weight=self.custom_configs.get('max_single_position_weight', 0.15),
            box_exposure_limits=self.box_limits
        )
        self.compliance_monitor = ComplianceMonitor(
            compliance_rules,
            stock_classifier=self.stock_classifier
        )

        logger.info("All specialized components for the 7-stage pipeline initialized")

    def initialize_system(self) -> bool:
        """
        Initialize the complete trading system.

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing trading system components")

            # Initialize all strategies
            failed_strategies = []
            for strategy in self.strategies:
                try:
                    if hasattr(strategy, 'prepare_data'):
                        success = strategy.prepare_data()
                        if not success:
                            failed_strategies.append(strategy.name)
                            logger.warning(f"Strategy '{strategy.name}' preparation returned False")
                except Exception as e:
                    failed_strategies.append(strategy.name)
                    logger.error(f"Failed to initialize strategy '{strategy.name}': {e}")

            if failed_strategies:
                logger.error(f"Failed to initialize strategies: {failed_strategies}")
                return False

            # Initialize portfolio state
            self.current_portfolio = self._create_initial_portfolio()

            logger.info(f"System initialization completed successfully for {len(self.strategies)} strategies")
            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False

    def _create_initial_portfolio(self) -> Portfolio:
        """Create initial portfolio state."""
        # Simplified portfolio creation
        from ..types.portfolio import Portfolio, Position

        portfolio = Portfolio(
            total_value=self.config.initial_capital or 1000000,
            cash_balance=self.config.initial_capital or 1000000,
            positions={},
            last_updated=datetime.now()
        )

        return portfolio

    def run_system(self, date: datetime, price_data: Dict[str, pd.DataFrame]) -> SystemResult:
        """
        Run the complete 7-stage portfolio construction pipeline for a given date.

        Args:
            date: Date to run the system.
            price_data: Dictionary of historical price data for all assets in the universe.

        Returns:
            Complete system result.
        """
        try:
            logger.info(f"--- Running 7-Stage Pipeline for {date.date()} ---")

            # Stage 1: Generate Signals from all strategies
            logger.info("[Stage 1/7] Generating signals from strategies...")
            strategy_signals = self.coordinator.coordinate(date)

            # Stage 2: Combine signals with Meta-Model
            logger.info("[Stage 2/7] Combining signals with Meta-Model...")
            # Convert TradingSignal objects to DataFrames for MetaModel compatibility
            strategy_signal_dfs = self._convert_signals_to_dataframes(strategy_signals, date)
            combined_signal = self.meta_model.combine(strategy_signal_dfs)
            if combined_signal.empty:
                raise ValueError("Meta-Model returned an empty signal.")

            # Stage 3: Dimensionality Reduction
            logger.info("[Stage 3/7] Reducing investment universe dimensionality...")
            universe_size = self.custom_configs.get('optimization_universe_size', 100)
            top_assets = combined_signal.iloc[0].abs().nlargest(universe_size).index
            expected_returns = combined_signal.iloc[0][top_assets]
            logger.info(f"Reduced universe to top {len(expected_returns)} assets.")

            # Stage 4: Estimate Risk (Covariance)
            logger.info("[Stage 4/7] Estimating covariance matrix...")
            reduced_price_data = {symbol: price_data[symbol] for symbol in top_assets if symbol in price_data}
            logger.info(f"Top assets: {list(top_assets)}")
            logger.info(f"Available price data symbols: {list(price_data.keys())}")
            logger.info(f"Reduced price data symbols: {list(reduced_price_data.keys())}")
            try:
                cov_matrix = self.covariance_estimator.estimate(reduced_price_data, date)
                logger.info(f"Covariance matrix shape: {cov_matrix.shape if hasattr(cov_matrix, 'shape') else 'N/A'}")
                if cov_matrix.empty:
                    logger.error("Covariance matrix is empty after estimation")
                    raise ValueError("Covariance estimation failed - empty matrix.")
            except Exception as e:
                logger.error(f"Covariance estimation failed with error: {e}")
                logger.error(f"Reduced price data keys: {list(reduced_price_data.keys())}")
                logger.error(f"Date for estimation: {date}")
                raise ValueError(f"Covariance estimation failed: {e}")

            # Stage 5: Classify Stocks for Box Constraints
            logger.info("[Stage 5/7] Classifying stocks for constraints...")
            investment_boxes = self.stock_classifier.classify_stocks(list(top_assets), price_data, as_of_date=date)

            # Convert InvestmentBox objects to dictionary format for PortfolioOptimizer
            classifications = self._convert_investment_boxes_to_dict(investment_boxes)
            logger.debug(f"Converted classifications: {classifications}")

            # Stage 6: Portfolio Optimization
            logger.info("[Stage 6/7] Optimizing portfolio...")
            logger.info(f"Expected returns for optimizer: {expected_returns.to_dict()}")
            box_constraints = self.portfolio_optimizer.build_box_constraints(classifications, self.box_limits)
            logger.info(f"Box constraints: {box_constraints}")

            # SOLID Principle: Portfolio optimizer handles weight adjustment based on expected returns
            # Strategy layer provides expected returns; optimizer manages risk-adjusted allocation

            final_weights = self.portfolio_optimizer.optimize(expected_returns, cov_matrix, box_constraints)
            logger.info(f"Final weights: {final_weights.to_dict()}")
            logger.info(f"Final weights sum: {final_weights.sum()}")
            logger.info(f"Final weights type: {type(final_weights)}")
            logger.info(f"Final weights empty: {final_weights.empty}")
            if hasattr(final_weights, 'sum') and final_weights.sum() < 0.1: # Check for optimization failure
                 raise ValueError("Portfolio optimization resulted in near-zero weights.")
            
            # Create TradingSignal objects from the final weights
            final_signals = [
                TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,  # All positive weights are buy signals
                    strength=min(weight, 1.0),  # Cap strength at 1.0
                    timestamp=date,
                    price=100.0,  # Mock price, should be obtained from market data
                    confidence=0.8,  # Default confidence
                    metadata={'strategy_name': 'SystemPortfolio', 'weight': weight}
                )
                for symbol, weight in final_weights.items() if weight > 0.0001 # Threshold small weights
            ]

            logger.info(f"Created {len(final_signals)} trading signals from optimization")
            for signal in final_signals:
                logger.info(f"Signal: {signal.symbol} = {signal.metadata.get('weight', 0):.4f}")

            # Execute trades based on the final optimized portfolio
            if self.current_portfolio:
                trades = self.trade_executor.execute(final_signals, self.current_portfolio)
                self._update_portfolio(trades, date)
            else:
                trades = [] # Should not happen if system is initialized

            # Stage 7: Compliance Check
            logger.info("[Stage 7/7] Performing compliance check...")
            if self.current_portfolio:
                compliance_report = self.compliance_monitor.check_compliance(self.current_portfolio)
            else:
                compliance_report = ComplianceReport.empty() # Helper for empty report
            
            # --- Reporting & Result Generation (largely unchanged) ---
            performance_report = self.performance_reporter.generate_report(self.current_portfolio, trades) if self.current_portfolio else None
            recommendations = self._generate_system_recommendations(compliance_report, performance_report)

            result = SystemResult(
                timestamp=datetime.now(),
                status="success",
                signals_summary={'combined_signal_assets': len(final_weights)},
                trades_summary={'executed_count': len(trades)},
                portfolio_summary=self._create_portfolio_summary(),
                allocation_summary={}, # Allocation is now implicit in the optimization
                compliance_report=compliance_report,
                performance_report=performance_report,
                recommendations=recommendations
            )

            self.execution_history.append(result)
            return result

        except Exception as e:
            logger.error(f"System execution failed for {date}: {e}", exc_info=True)
            return SystemResult.error_result(e) # Assuming a helper for error results

    def _update_portfolio(self, trades: List[Any], date: datetime) -> None:
        """Update portfolio state after trades."""
        if not self.current_portfolio:
            return

        # Update positions based on trades
        for trade in trades:
            if trade.symbol not in self.current_portfolio.positions:
                # Create new position
                from ..types.portfolio import Position
                self.current_portfolio.positions[trade.symbol] = Position(
                    symbol=trade.symbol,
                    quantity=trade.quantity,
                    average_cost=trade.price,
                    current_price=trade.price,
                    market_value=trade.quantity * trade.price,
                    unrealized_pnl=0.0,
                    weight=0.0  # Will be calculated below
                )
            else:
                # Update existing position
                position = self.current_portfolio.positions[trade.symbol]
                if trade.side == 'buy':
                    # Buy: increase position
                    total_cost = position.quantity * position.average_cost + trade.quantity * trade.price
                    total_quantity = position.quantity + trade.quantity
                    position.average_cost = total_cost / total_quantity if total_quantity > 0 else position.average_cost
                    position.quantity = total_quantity
                    position.current_price = trade.price
                    position.market_value = position.quantity * position.current_price
                elif trade.side == 'sell':
                    # Sell: reduce position
                    position.quantity -= trade.quantity
                    if position.quantity <= 0:
                        # Position closed
                        position.quantity = 0
                        position.market_value = 0
                    else:
                        position.market_value = position.quantity * trade.price
                        position.current_price = trade.price

        # Update cash balance
        total_cost = sum(trade.quantity * trade.price + trade.commission for trade in trades)
        self.current_portfolio.cash_balance -= total_cost

        # Calculate portfolio weights
        total_market_value = sum(pos.market_value for pos in self.current_portfolio.positions.values())
        total_portfolio_value = total_market_value + self.current_portfolio.cash_balance

        if total_portfolio_value > 0:
            for position in self.current_portfolio.positions.values():
                position.weight = position.market_value / total_portfolio_value

        self.current_portfolio.last_updated = date

        logger.debug(f"Portfolio updated with {len(trades)} trades")
        logger.debug(f"Total portfolio value: ${total_portfolio_value:,.2f}")
        logger.debug(f"Positions updated: {len([p for p in self.current_portfolio.positions.values() if p.quantity > 0])}")

    def _create_portfolio_summary(self) -> Dict[str, Any]:
        """Create portfolio summary."""
        if not self.current_portfolio:
            return {}

        return {
            'total_value': self.current_portfolio.total_value,
            'cash_balance': self.current_portfolio.cash_balance,
            'position_count': len([p for p in self.current_portfolio.positions.values() if p.quantity > 0]),
            'last_updated': self.current_portfolio.last_updated
        }

    def _generate_system_recommendations(self, compliance_report: ComplianceReport,
                                        performance_report: Any) -> List[str]:
        """Generate system-level recommendations."""
        recommendations = []

        # Compliance-based recommendations
        if not compliance_report.is_compliant:
            recommendations.extend(compliance_report.recommendations)
            recommendations.append("Address compliance issues immediately")

        # Performance-based recommendations
        if performance_report and hasattr(performance_report, 'recommendations'):
            recommendations.extend(performance_report.recommendations)

        # System-level recommendations
        if len(self.execution_history) > 0:
            recent_results = self.execution_history[-5:]  # Last 5 executions
            failed_count = len([r for r in recent_results if r.status != "success"])
            if failed_count >= 3:
                recommendations.append("Multiple recent failures - review system configuration")

        return recommendations

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        if not self.execution_history:
            return {
                'status': 'not_initialized',
                'last_execution': None,
                'total_executions': 0,
                'success_rate': 0
            }

        last_result = self.execution_history[-1]
        total_executions = len(self.execution_history)
        successful_executions = len([r for r in self.execution_history if r.is_successful])
        success_rate = successful_executions / total_executions if total_executions > 0 else 0

        return {
            'status': last_result.status,
            'last_execution': last_result.timestamp,
            'total_executions': total_executions,
            'success_rate': success_rate,
            'has_compliance_issues': last_result.has_compliance_issues,
            'component_status': {
                'coordinator': 'active',
                'optimizer': 'active',
                'compliance_monitor': 'active',
                'trade_executor': 'active',
                'performance_reporter': 'active'
            }
        }

    def get_component_info(self) -> Dict[str, Any]:
        """Get information about all system components."""
        return {
            'strategies': {
                strategy.name: {
                    'type': strategy.__class__.__name__,
                    'config': getattr(strategy, 'config', {})
                }
                for strategy in self.strategies
            },
            'strategy_count': len(self.strategies),
            'components': {
                'coordinator': self.coordinator.get_coordination_stats(),
                'optimizer': self.portfolio_optimizer.__class__.__name__,
                'compliance_monitor': self.compliance_monitor.get_compliance_summary(30),
                'trade_executor': self.trade_executor.get_execution_performance(30),
                'performance_reporter': self.performance_reporter.get_performance_summary(30)
            },
            'risk_manager': {
                'status': 'active',
                'type': 'Optimization-Based'
            }
        }

    def validate_system_configuration(self) -> Tuple[bool, List[str]]:
        """Validate entire system configuration."""
        issues = []

        # Validate we have strategies
        if not self.strategies:
            issues.append("No strategies initialized")
            return False, issues

        # Validate components (simplified for this refactor)
        if not all([self.coordinator, self.portfolio_optimizer, self.compliance_monitor, self.trade_executor]):
             issues.append("One or more core components are not initialized.")

        # Validate strategy names are unique
        strategy_names = [s.name for s in self.strategies]
        if len(strategy_names) != len(set(strategy_names)):
            issues.append("Duplicate strategy names detected")

        return len(issues) == 0, issues

    def _convert_signals_to_dataframes(self, strategy_signals: Dict[str, List], date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Convert TradingSignal objects to DataFrames for MetaModel compatibility.

        Args:
            strategy_signals: Dictionary mapping strategy names to lists of TradingSignal objects
            date: Current date for signal generation

        Returns:
            Dictionary mapping strategy names to DataFrames with expected returns
        """
        import pandas as pd

        converted_signals = {}

        for strategy_name, signals in strategy_signals.items():
            if not signals:
                logger.warning(f"No signals from strategy '{strategy_name}'")
                continue

            # Create a DataFrame with signal values
            signals_data = {}
            for signal in signals:
                if hasattr(signal, 'symbol') and hasattr(signal, 'strength'):
                    signals_data[signal.symbol] = signal.strength

            if signals_data:
                # Create DataFrame with single row for the current date
                signal_df = pd.DataFrame([signals_data], index=[date])
                converted_signals[strategy_name] = signal_df
                logger.debug(f"Converted {len(signals_data)} signals from '{strategy_name}' to DataFrame")
            else:
                logger.warning(f"No valid signal data found for strategy '{strategy_name}'")

        return converted_signals

    def _convert_investment_boxes_to_dict(self, investment_boxes: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Convert InvestmentBox objects to dictionary format for PortfolioOptimizer compatibility.

        Note: StockClassifier returns Dict[box_key, InvestmentBox], but we need
        Dict[symbol, classification_dict]. We need to extract symbols from InvestmentBox.stocks.

        Args:
            investment_boxes: Dictionary mapping box_keys to InvestmentBox objects

        Returns:
            Dictionary mapping stock symbols to classification dictionaries
        """
        classifications_dict = {}

        for box_key, investment_box in investment_boxes.items():
            if hasattr(investment_box, 'stocks') and investment_box.stocks:
                # Extract classification info from the InvestmentBox
                if hasattr(investment_box, 'size') and hasattr(investment_box, 'style') and hasattr(investment_box, 'region') and hasattr(investment_box, 'sector'):
                    classification_info = {
                        'size': investment_box.size.value if hasattr(investment_box.size, 'value') else str(investment_box.size),
                        'style': investment_box.style.value if hasattr(investment_box.style, 'value') else str(investment_box.style),
                        'region': investment_box.region.value if hasattr(investment_box.region, 'value') else str(investment_box.region),
                        'sector': investment_box.sector.name if hasattr(investment_box.sector, 'name') else str(investment_box.sector)
                    }
                else:
                    # Fallback classification
                    classification_info = {
                        'size': 'large',
                        'style': 'growth',
                        'region': 'developed',
                        'sector': 'Unknown'
                    }

                # Add classification info for each stock in this box
                for stock_info in investment_box.stocks:
                    if isinstance(stock_info, dict) and 'symbol' in stock_info:
                        symbol = stock_info['symbol']
                        classifications_dict[symbol] = classification_info
                        logger.debug(f"Classified {symbol} as {classification_info}")
                    elif hasattr(stock_info, 'symbol'):
                        symbol = stock_info.symbol
                        classifications_dict[symbol] = classification_info
                        logger.debug(f"Classified {symbol} as {classification_info}")
            else:
                # Handle unexpected format - box_key might be the symbol directly
                logger.warning(f"Unexpected investment box format for {box_key}: {type(investment_box)}")
                # Try to extract symbol from box_key as fallback
                if '_' not in box_key:  # Likely a symbol, not a box_key
                    classifications_dict[box_key] = {
                        'size': 'large',
                        'style': 'growth',
                        'region': 'developed',
                        'sector': 'Unknown'
                    }

        logger.debug(f"Converted {len(classifications_dict)} stocks to dictionary format")
        return classifications_dict