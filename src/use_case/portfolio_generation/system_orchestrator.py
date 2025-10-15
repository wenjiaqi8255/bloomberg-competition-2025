"""
System Orchestrator with Portfolio Construction Integration
=========================================================

This module provides the `SystemOrchestrator`, which integrates with the portfolio 
construction framework while maintaining compatibility with existing components.

Key features:
- Integration with IPortfolioBuilder interface
- Support for both Quantitative and Box-Based construction methods
- Configuration-driven construction method selection
- Unified performance tracking across all components
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass
import pandas as pd

from src.trading_system.orchestration.components.coordinator import StrategyCoordinator, CoordinatorConfig
from src.trading_system.orchestration.components.allocator import CapitalAllocator, AllocationConfig
from src.trading_system.orchestration.components.compliance import ComplianceMonitor, ComplianceRules
from src.trading_system.orchestration.components.executor import TradeExecutor, ExecutionConfig
from src.trading_system.orchestration.components.reporter import PerformanceReporter, ReportConfig
from src.trading_system.strategies.base_strategy import BaseStrategy
from src.trading_system.types.portfolio import Portfolio
from src.trading_system.config.system import SystemConfig
from src.trading_system.data.stock_classifier import StockClassifier
from src.trading_system.metamodel.meta_model import MetaModel
from src.trading_system.orchestration.utils.signal_converters import SignalConverters

# Portfolio construction imports
from src.trading_system.portfolio_construction import (
    IPortfolioBuilder, PortfolioConstructionRequest, PortfolioBuilderFactory
)

logger = logging.getLogger(__name__)


@dataclass
class SystemResult:
    """Result of system execution."""
    timestamp: datetime
    portfolio: Portfolio
    trades: List[Any]
    performance_metrics: Dict[str, Any]
    compliance_report: Optional[Any] = None
    system_recommendations: List[str] = None
    execution_summary: Dict[str, Any] = None

    def __post_init__(self):
        if self.system_recommendations is None:
            self.system_recommendations = []
        if self.execution_summary is None:
            self.execution_summary = {}


class SystemOrchestrator:
    """
    Main system orchestrator that coordinates all trading system components.

    This orchestrator integrates with the portfolio construction framework
    while maintaining compatibility with existing components.

    Responsibilities:
    - Initialize and coordinate all system components
    - Execute the complete trading system pipeline
    - Integrate with portfolio construction framework
    - Generate comprehensive system results
    """

    def __init__(self, config: SystemConfig):
        """
        Initialize the system orchestrator.

        Args:
            config: System configuration
        """
        self.config = config
        self.components = {}
        self.portfolio_builder: Optional[IPortfolioBuilder] = None
        
        # Initialize components
        self._initialize_components()

        logger.info("SystemOrchestrator initialized successfully")

    def _initialize_components(self) -> None:
        """Initialize all system components."""
        try:
            # Initialize strategies
            strategies = self._initialize_strategies()

            # Initialize coordinator
            coordinator_config = CoordinatorConfig(
                max_signals_per_day=self.config.max_signals_per_day,
                signal_conflict_resolution=self.config.signal_conflict_resolution,
                min_signal_strength=self.config.min_signal_strength,
                max_position_size=self.config.max_position_size,
                capacity_scaling=self.config.capacity_scaling
            )
            self.components['coordinator'] = StrategyCoordinator(strategies, coordinator_config)
            
            # Initialize allocator
            allocator_config = AllocationConfig(
                strategy_allocations=self.config.strategy_allocations,
                rebalance_threshold=self.config.rebalance_threshold,
                max_single_position_weight=self.config.max_single_position_weight,
                cash_buffer_weight=self.config.cash_buffer_weight
            )
            self.components['allocator'] = CapitalAllocator(allocator_config)
            
            # Initialize compliance monitor
            compliance_rules = ComplianceRules(
                max_single_position_weight=self.config.max_single_position_weight,
                box_exposure_limits=self.config.box_exposure_limits,
                max_sector_allocation=self.config.max_sector_allocation,
                max_concentration_top5=self.config.max_concentration_top5,
                max_concentration_top10=self.config.max_concentration_top10
            )
            self.components['compliance'] = ComplianceMonitor(compliance_rules)
            
            # Initialize executor
            executor_config = ExecutionConfig(
                max_order_size_percent=self.config.max_order_size_percent,
                min_order_size_usd=self.config.min_order_size_usd,
                max_positions_per_day=self.config.max_positions_per_day,
                commission_rate=self.config.commission_rate,
                cooling_period_hours=self.config.cooling_period_hours,
                default_order_type=self.config.default_order_type,
                expected_slippage_bps=self.config.expected_slippage_bps
            )
            self.components['executor'] = TradeExecutor(executor_config)
            
            # Initialize reporter
            reporter_config = ReportConfig(
                daily_reports=self.config.daily_reports,
                weekly_reports=self.config.weekly_reports,
                monthly_reports=self.config.monthly_reports,
                benchmark_symbol=self.config.benchmark_symbol,
                file_format=self.config.file_format,
                output_directory=self.config.output_directory
            )
            self.components['reporter'] = PerformanceReporter(reporter_config)
            
            # Initialize meta model
            self.components['meta_model'] = MetaModel()
            
            # Initialize stock classifier
            self.components['classifier'] = StockClassifier()
            
            logger.info("All components initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    def _initialize_strategies(self) -> List[BaseStrategy]:
        """Initialize trading strategies."""
        strategies = []
        
        for strategy_config in self.config.strategies:
            try:
                # This would instantiate the actual strategy based on config
                # For now, we'll create a placeholder
                strategy = BaseStrategy(
                    name=strategy_config['name'],
                    config=strategy_config.get('config', {})
                )
                strategies.append(strategy)
                logger.info(f"Initialized strategy: {strategy.name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize strategy {strategy_config.get('name', 'unknown')}: {e}")
                continue
        
        return strategies

    def run_system(self, date: datetime, price_data: Dict[str, pd.DataFrame]) -> SystemResult:
        """
        Execute the complete trading system pipeline.

        Args:
            date: Current date for system execution
            price_data: Market price data

        Returns:
            Complete system execution result
        """
        try:
            logger.info(f"Starting system execution for {date}")
            
            # Step 1: Generate signals from all strategies
            strategy_signals = self.components['coordinator'].coordinate(date)
            logger.info(f"Generated signals from {len(strategy_signals)} strategies")
            
            # Step 2: Combine signals using meta model
            combined_signals = self.components['meta_model'].combine(strategy_signals)
            logger.info(f"Combined signals into {len(combined_signals)} final signals")
            
            # Step 3: Convert signals to DataFrames for portfolio construction
            signal_dataframes = SignalConverters.convert_signals_to_dataframes(
                strategy_signals, date
            )
            
            # Step 4: Portfolio Construction using new framework
            portfolio_weights = self._construct_portfolio(combined_signals, signal_dataframes, date)
            
            # Step 5: Pre-trade compliance check
            compliance_report = self.components['compliance'].check_target_compliance(
                portfolio_weights, date
            )
            
            if compliance_report.has_critical_violations():
                logger.warning("Critical compliance violations detected - adjusting portfolio")
                # In a real system, this would trigger portfolio adjustment logic
                portfolio_weights = self._adjust_portfolio_for_compliance(portfolio_weights, compliance_report)
            
            # Step 6: Execute trades
            trades = self._execute_trades(combined_signals, portfolio_weights, date)
            
            # Step 7: Create final portfolio
            portfolio = self._create_final_portfolio(portfolio_weights, trades, date)
            
            # Step 8: Post-trade compliance check
            post_trade_compliance = self.components['compliance'].check_portfolio_compliance(portfolio)
            
            # Step 9: Generate performance metrics
            performance_metrics = self.components['reporter'].generate_report(
                portfolio, trades, date
            )
            
            # Step 10: Create system result
            result = SystemResult(
                timestamp=date,
                portfolio=portfolio,
                trades=trades,
                performance_metrics=performance_metrics,
                compliance_report=post_trade_compliance,
                system_recommendations=self._generate_system_recommendations(compliance_report, post_trade_compliance),
                execution_summary=self._create_execution_summary(strategy_signals, trades, performance_metrics)
            )
            
            logger.info(f"System execution completed successfully for {date}")
            return result

        except Exception as e:
            logger.error(f"System execution failed for {date}: {e}")
            raise

    def _construct_portfolio(self, signals: pd.DataFrame, signal_dataframes: Dict[str, pd.DataFrame], 
                           date: datetime) -> pd.Series:
        """Construct portfolio using the portfolio construction framework."""
        try:
            # Create portfolio construction request
            request = PortfolioConstructionRequest(
                date=date,
                universe=list(signals.index),
                signals=signals,
                price_data={},  # TODO: Pass actual price data
                constraints={}
            )
            
            # Get portfolio builder from factory
            if not self.portfolio_builder:
                self.portfolio_builder = PortfolioBuilderFactory.create_builder(
                    self.config.portfolio_construction
                )
            
            # Build portfolio
            portfolio_weights = self.portfolio_builder.build_portfolio(request)
            
            logger.info(f"Constructed portfolio with {len(portfolio_weights)} positions")
            return portfolio_weights
            
        except Exception as e:
            logger.error(f"Portfolio construction failed: {e}")
            raise

    def _execute_trades(self, signals: pd.DataFrame, portfolio_weights: pd.Series, 
                       date: datetime) -> List[Any]:
        """Execute trades based on signals and portfolio weights."""
        try:
            # Convert portfolio weights to trading signals
            trading_signals = self._convert_weights_to_signals(portfolio_weights, date)
            
            # Create initial portfolio for execution
            initial_portfolio = Portfolio(
                positions={},
                cash=self.config.initial_capital,
                total_value=self.config.initial_capital
            )
            
            # Execute trades
            trades = self.components['executor'].execute(
                trading_signals, initial_portfolio
            )
            
            logger.info(f"Executed {len(trades)} trades")
            return trades
            
        except Exception as e:
            logger.error(f"Trade execution failed: {e}")
            raise

    def _convert_weights_to_signals(self, portfolio_weights: pd.Series, 
                                   date: datetime) -> List[Any]:
        """Convert portfolio weights to trading signals."""
        # This would convert the portfolio weights to actual TradingSignal objects
        # For now, return empty list as placeholder
        return []

    def _create_final_portfolio(self, portfolio_weights: pd.Series, trades: List[Any], 
                               date: datetime) -> Portfolio:
        """Create final portfolio after trade execution."""
        # This would create the actual portfolio based on executed trades
        # For now, return a placeholder portfolio
        return Portfolio(
            positions={},
            cash=self.config.initial_capital,
            total_value=self.config.initial_capital
        )

    def _adjust_portfolio_for_compliance(self, portfolio_weights: pd.Series, 
                                       compliance_report: Any) -> pd.Series:
        """Adjust portfolio weights to address compliance violations."""
        # This would implement portfolio adjustment logic
        # For now, return the original weights
        return portfolio_weights

    def _generate_system_recommendations(self, pre_trade_compliance: Any, 
                                       post_trade_compliance: Any) -> List[str]:
        """Generate system recommendations based on compliance reports."""
        recommendations = []

        if pre_trade_compliance and pre_trade_compliance.has_critical_violations():
            recommendations.append("Review and adjust portfolio construction parameters")
        
        if post_trade_compliance and post_trade_compliance.has_critical_violations():
            recommendations.append("Implement additional compliance controls")

        return recommendations

    def _create_execution_summary(self, strategy_signals: Dict[str, List], 
                                 trades: List[Any], performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution summary."""
        return {
            'strategies_executed': len(strategy_signals),
            'total_trades': len(trades),
            'execution_success_rate': performance_metrics.get('success_rate', 0.0),
            'components_status': {
                'coordinator': self.components['coordinator'].get_performance_stats(),
                'allocator': self.components['allocator'].get_performance_stats(),
                'compliance': self.components['compliance'].get_performance_stats(),
                'executor': self.components['executor'].get_performance_stats(),
                'reporter': self.components['reporter'].get_performance_stats()
            }
        }

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'timestamp': datetime.now(),
            'components': {
                name: component.get_performance_stats() if hasattr(component, 'get_performance_stats') else {}
                for name, component in self.components.items()
            },
            'config': {
                'strategies_count': len(self.config.strategies),
                'initial_capital': self.config.initial_capital,
                'portfolio_construction_method': self.config.portfolio_construction.get('method', 'unknown')
            }
        }

    def reset_system(self) -> None:
        """Reset all system components."""
        for component in self.components.values():
            if hasattr(component, 'reset_performance_stats'):
                component.reset_performance_stats()
        
        logger.info("System reset completed")

    def get_all_component_stats(self) -> Dict[str, Any]:
        """Get performance stats from all components."""
        return {
            'coordinator': self.components['coordinator'].get_performance_stats() if hasattr(self.components['coordinator'], 'get_performance_stats') else {},
            'allocator': self.components['allocator'].get_performance_stats() if hasattr(self.components['allocator'], 'get_performance_stats') else {},
            'compliance': self.components['compliance'].get_performance_stats() if hasattr(self.components['compliance'], 'get_performance_stats') else {},
            'executor': self.components['executor'].get_performance_stats() if hasattr(self.components['executor'], 'get_performance_stats') else {},
            'reporter': self.components['reporter'].get_performance_stats() if hasattr(self.components['reporter'], 'get_performance_stats') else {}
        }

    def get_trade_history(self) -> List[Any]:
        """Get complete trade history from executor."""
        if hasattr(self.components['executor'], 'completed_trades'):
            return self.components['executor'].completed_trades
        return []

    def get_portfolio_history(self) -> List[Dict[str, Any]]:
        """Get portfolio snapshots from execution history."""
        # This would be populated during backtest execution
        if hasattr(self, '_portfolio_history'):
            return self._portfolio_history
        return []

    def get_compliance_history(self) -> List[Any]:
        """Get compliance check history."""
        if hasattr(self.components['compliance'], 'compliance_history'):
            return self.components['compliance'].compliance_history
        return []

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        trade_history = self.get_trade_history()
        portfolio_history = self.get_portfolio_history()
        component_stats = self.get_all_component_stats()
        
        return {
            'total_trades': len(trade_history),
            'portfolio_snapshots': len(portfolio_history),
            'component_stats': component_stats,
            'system_status': self.get_system_status()
        }
