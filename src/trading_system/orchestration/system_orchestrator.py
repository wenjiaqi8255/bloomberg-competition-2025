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

from .components.coordinator import StrategyCoordinator, CoordinatorConfig
from .components.allocator import CapitalAllocator, AllocationConfig, StrategyAllocation
from .components.compliance import ComplianceMonitor, ComplianceRules, ComplianceReport, ComplianceStatus, StrategyAllocationRule
from .components.executor import TradeExecutor, ExecutionConfig
from .components.reporter import PerformanceReporter, ReportConfig

from ..strategies.base_strategy import BaseStrategy
from ..types.portfolio import Portfolio
from ..types.data_types import TradingSignal
from ..config.system import SystemConfig
from ..utils.risk import RiskCalculator

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
                 allocation_config: AllocationConfig,
                 compliance_rules: Optional[ComplianceRules] = None,
                 custom_configs: Optional[Dict[str, Any]] = None):
        """
        Initialize System Orchestrator with flexible strategy support.
        
        This refactored version supports an arbitrary number of strategies (1, 2, 3, or more)
        instead of being hardcoded to just core and satellite strategies.

        Args:
            system_config: System configuration
            strategies: List of trading strategies (can be 1 or more)
            allocation_config: Capital allocation configuration for all strategies
            compliance_rules: Compliance rules (optional, will be auto-generated from allocation if not provided)
            custom_configs: Custom configuration overrides for components
        
        Raises:
            ValueError: If strategies list is empty or configuration is inconsistent
        """
        if not strategies:
            raise ValueError("At least one strategy must be provided")
        
        self.config = system_config
        self.strategies = strategies
        self.allocation_config = allocation_config
        self.custom_configs = custom_configs or {}
        
        # Validate that strategy names match allocation config
        self._validate_configuration()
        
        # Auto-generate compliance rules from allocation if not provided
        if compliance_rules is None:
            compliance_rules = self._generate_compliance_rules_from_allocation()
        self.compliance_rules = compliance_rules

        # Initialize specialized components
        self._initialize_components()

        # System state
        self.current_portfolio: Optional[Portfolio] = None
        self.execution_history: List[SystemResult] = []

        # Risk management functionality moved to utils/risk.py
        self.risk_calculator = RiskCalculator()

        logger.info(f"Initialized SystemOrchestrator with {len(self.strategies)} strategies:")
        for strategy in self.strategies:
            alloc = self.allocation_config.get_allocation_for_strategy(strategy.name)
            if alloc:
                logger.info(
                    f"  - {strategy.name}: target={alloc.target_weight:.1%}, "
                    f"priority={alloc.priority}"
                )

    def _validate_configuration(self) -> None:
        """
        Validate that strategies and allocation configuration are consistent.
        
        Raises:
            ValueError: If configuration is inconsistent
        """
        strategy_names = {s.name for s in self.strategies}
        config_names = {a.strategy_name for a in self.allocation_config.strategy_allocations}
        
        if strategy_names != config_names:
            raise ValueError(
                f"Strategy names mismatch between strategies and allocation config.\n"
                f"Strategies: {sorted(strategy_names)}\n"
                f"Allocation config: {sorted(config_names)}\n"
                f"All strategy names must match exactly."
            )
        
        logger.info("Configuration validation passed")
    
    def _generate_compliance_rules_from_allocation(self) -> ComplianceRules:
        """
        Auto-generate compliance rules from allocation configuration.
        
        This creates compliance rules that match the allocation constraints,
        so strategies stay within their configured bounds.
        """
        strategy_rules = [
            StrategyAllocationRule(
                strategy_name=alloc.strategy_name,
                min_weight=alloc.min_weight,
                max_weight=alloc.max_weight
            )
            for alloc in self.allocation_config.strategy_allocations
        ]
        
        compliance_rules = ComplianceRules(
            strategy_allocation_rules=strategy_rules
        )
        
        logger.info(f"Auto-generated compliance rules for {len(strategy_rules)} strategies")
        return compliance_rules

    def _initialize_components(self) -> None:
        """Initialize all specialized components."""
        # Strategy Coordinator - extract priorities from allocation config
        strategy_priority = {
            alloc.strategy_name: alloc.priority 
            for alloc in self.allocation_config.strategy_allocations
        }
        
        coordinator_config = CoordinatorConfig(
            max_signals_per_day=self.custom_configs.get('max_signals_per_day', 50),
            signal_conflict_resolution=self.custom_configs.get('signal_conflict_resolution', 'merge'),
            capacity_scaling=self.custom_configs.get('capacity_scaling', True),
            strategy_priority=strategy_priority
        )
        self.coordinator = StrategyCoordinator(
            strategies=self.strategies,  # Use full list of strategies
            config=coordinator_config
        )

        # Capital Allocator - use the provided allocation config
        self.allocator = CapitalAllocator(self.allocation_config)

        # Compliance Monitor - use the compliance rules (auto-generated or provided)
        self.compliance_monitor = ComplianceMonitor(self.compliance_rules)

        # Trade Executor
        execution_config = ExecutionConfig(
            max_order_size_percent=self.custom_configs.get('max_order_size_percent', 0.05),
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

        logger.info("All specialized components initialized")

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

    def run_system(self, date: datetime) -> SystemResult:
        """
        Run the complete system for a given date.

        This is the main orchestration method that coordinates all components.

        Args:
            date: Date to run the system

        Returns:
            Complete system result
        """
        try:
            logger.info(f"Running system for {date}")

            # Step 1: Coordinate strategies and get signals
            strategy_signals = self.coordinator.coordinate(date)
            logger.info(f"Coordinated signals from {len(strategy_signals)} strategies")

            # Step 2: Allocate capital to strategies
            if self.current_portfolio:
                weighted_signals = self.allocator.allocate(
                    strategy_signals,
                    self.current_portfolio,
                    self.current_portfolio.total_value
                )
                logger.info("Capital allocation completed")
            else:
                weighted_signals = strategy_signals
                logger.warning("No current portfolio - using unweighted signals")

            # Step 3: Execute trades
            if self.current_portfolio:
                # Flatten signals for execution
                all_signals = []
                for signals in weighted_signals.values():
                    all_signals.extend(signals)

                trades = self.trade_executor.execute(all_signals, self.current_portfolio)
                logger.info(f"Executed {len(trades)} trades")
            else:
                trades = []
                logger.warning("No current portfolio - skipping trade execution")

            # Step 4: Update portfolio state (simplified)
            if self.current_portfolio and trades:
                self._update_portfolio(trades, date)

            # Step 5: Check IPS compliance
            if self.current_portfolio:
                compliance_report = self.compliance_monitor.check_compliance(self.current_portfolio)
                logger.info(f"Compliance check completed: {compliance_report.overall_status.value}")
            else:
                compliance_report = ComplianceReport(
                    timestamp=datetime.now(),
                    overall_status=ComplianceStatus.COMPLIANT,
                    total_violations=0,
                    violations=[],
                    warnings=[],
                    recommendations=["No portfolio to check"],
                    portfolio_summary={}
                )

            # Step 6: Generate performance report
            if self.current_portfolio:
                performance_report = self.performance_reporter.generate_report(
                    self.current_portfolio,
                    trades,
                    period_days=30
                )
                logger.info("Performance report generated")
            else:
                performance_report = None

            # Step 7: Generate recommendations
            recommendations = self._generate_system_recommendations(compliance_report, performance_report)

            # Step 8: Create system result
            result = SystemResult(
                timestamp=datetime.now(),
                status="success",
                signals_summary={
                    'strategies': list(strategy_signals.keys()),
                    'total_signals': sum(len(signals) for signals in strategy_signals.values())
                },
                trades_summary={
                    'executed_count': len(trades),
                    'trade_list': [trade.__dict__ for trade in trades]
                },
                portfolio_summary=self._create_portfolio_summary(),
                allocation_summary=self.allocator.get_allocation_status(),
                compliance_report=compliance_report,
                performance_report=performance_report,
                recommendations=recommendations
            )

            # Step 9: Store in history
            self.execution_history.append(result)
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]

            logger.info(f"System execution completed for {date}")
            return result

        except Exception as e:
            logger.error(f"System execution failed for {date}: {e}")
            return SystemResult(
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
                    recommendations=["System error - manual review required"],
                    portfolio_summary={}
                ),
                performance_report=None,
                recommendations=[f"System error: {str(e)}"]
            )

    def _update_portfolio(self, trades: List[Any], date: datetime) -> None:
        """Update portfolio state after trades."""
        if not self.current_portfolio:
            return

        # Simplified portfolio update
        # In practice, would properly update positions, cash, and portfolio value
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
                    weight=0.0  # Will be calculated later
                )

        # Update cash balance (simplified)
        total_cost = sum(trade.quantity * trade.price + trade.commission for trade in trades)
        self.current_portfolio.cash_balance -= total_cost
        self.current_portfolio.last_updated = date

        logger.debug(f"Portfolio updated with {len(trades)} trades")

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
                'allocator': 'active',
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
                'allocator': self.allocator.get_allocation_status(),
                'compliance_monitor': self.compliance_monitor.get_compliance_summary(30),
                'trade_executor': self.trade_executor.get_execution_performance(30),
                'performance_reporter': self.performance_reporter.get_performance_summary(30)
            },
            'risk_manager': {
                'status': 'active',
                'type': 'standard'
            }
        }

    def validate_system_configuration(self) -> Tuple[bool, List[str]]:
        """Validate entire system configuration."""
        issues = []

        # Validate we have strategies
        if not self.strategies:
            issues.append("No strategies initialized")
            return False, issues

        # Validate components
        component_validations = [
            self.coordinator.validate_configuration(),
            self.allocator.validate_configuration(),
            self.compliance_monitor.validate_rules(),
            self.trade_executor.validate_configuration(),
            self.performance_reporter.validate_configuration()
        ]

        for is_valid, component_issues in component_validations:
            if not is_valid:
                issues.extend(component_issues)

        # Validate each strategy has allocation config
        for strategy in self.strategies:
            alloc = self.allocation_config.get_allocation_for_strategy(strategy.name)
            if not alloc:
                issues.append(f"Strategy '{strategy.name}' missing allocation configuration")

        # Validate strategy names are unique
        strategy_names = [s.name for s in self.strategies]
        if len(strategy_names) != len(set(strategy_names)):
            issues.append("Duplicate strategy names detected")

        return len(issues) == 0, issues