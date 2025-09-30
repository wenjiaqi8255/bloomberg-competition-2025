"""
System Orchestrator  - Refactored Architecture

This is the refactored version of the SystemOrchestrator that follows the
Single Responsibility Principle. It coordinates the specialized components
while delegating specific responsibilities to them.

"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass

from .components.coordinator import StrategyCoordinator, CoordinatorConfig
from .components.allocator import CapitalAllocator, AllocationConfig
from .components.compliance import ComplianceMonitor, ComplianceRules, ComplianceReport, ComplianceStatus
from .components.executor import TradeExecutor, ExecutionConfig
from .components.reporter import PerformanceReporter, ReportConfig

from ..strategies.core_ffml_strategy import CoreFFMLStrategy
from ..strategies.satellite_strategy import SatelliteStrategy
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
    Refactored System Orchestrator - Coordination Layer Only

    This new version delegates specific responsibilities to specialized components:
    - Strategy coordination: StrategyCoordinator
    - Capital allocation: CapitalAllocator
    - IPS compliance: ComplianceMonitor
    - Trade execution: TradeExecutor
    - Performance reporting: PerformanceReporter

    The orchestrator now only coordinates these components and manages the overall system flow.
    """

    def __init__(self, system_config: SystemConfig,
                 core_strategy: Optional[CoreFFMLStrategy] = None,
                 satellite_strategy: Optional[SatelliteStrategy] = None,
                 custom_configs: Optional[Dict[str, Any]] = None):
        """
        Initialize refactored System Orchestrator.

        Args:
            system_config: System configuration
            core_strategy: Core FFML strategy (optional, will be created if not provided)
            satellite_strategy: Satellite strategy (optional, will be created if not provided)
            custom_configs: Custom configuration overrides for components
        """
        self.config = system_config
        self.custom_configs = custom_configs or {}

        # Initialize strategies
        self.core_strategy = core_strategy or self._create_core_strategy()
        self.satellite_strategy = satellite_strategy or self._create_satellite_strategy()

        # Initialize specialized components
        self._initialize_components()

        # System state
        self.current_portfolio: Optional[Portfolio] = None
        self.execution_history: List[SystemResult] = []

        # Risk management functionality moved to utils/risk.py
        self.risk_calculator = RiskCalculator()

        logger.info("Initialized SystemOrchestrator ")
        logger.info(f"Core strategy: {self.core_strategy.config.name}")
        logger.info(f"Satellite strategy: {self.satellite_strategy.config.name}")

    def _create_core_strategy(self) -> CoreFFMLStrategy:
        """Create default core strategy."""
        return CoreFFMLStrategy(
            strategy_name=f"{self.config.system_name}_Core",
            core_weight=self.config.core_target_weight,
            lookback_window=252,
            max_position_size=0.15
        )

    def _create_satellite_strategy(self) -> SatelliteStrategy:
        """Create default satellite strategy."""
        return SatelliteStrategy(
            strategy_name=f"{self.config.system_name}_Satellite",
            satellite_weight=self.config.satellite_target_weight,
            max_positions=8
        )

    def _initialize_components(self) -> None:
        """Initialize all specialized components."""
        # Strategy Coordinator
        coordinator_config = CoordinatorConfig(
            max_signals_per_day=self.custom_configs.get('max_signals_per_day', 50),
            signal_conflict_resolution=self.custom_configs.get('signal_conflict_resolution', 'merge'),
            capacity_scaling=self.custom_configs.get('capacity_scaling', True)
        )
        self.coordinator = StrategyCoordinator(
            strategies=[self.core_strategy, self.satellite_strategy],
            config=coordinator_config
        )

        # Capital Allocator
        allocation_config = AllocationConfig(
            core_target_weight=self.config.core_target_weight,
            core_min_weight=self.config.core_min_weight,
            core_max_weight=self.config.core_max_weight,
            satellite_target_weight=self.config.satellite_target_weight,
            satellite_min_weight=self.config.satellite_min_weight,
            satellite_max_weight=self.config.satellite_max_weight,
            rebalance_threshold=self.custom_configs.get('rebalance_threshold', 0.05)
        )
        self.allocator = CapitalAllocator(allocation_config)

        # Compliance Monitor
        compliance_rules = ComplianceRules(
            core_min_weight=self.config.core_min_weight,
            core_max_weight=self.config.core_max_weight,
            satellite_min_weight=self.config.satellite_min_weight,
            satellite_max_weight=self.config.satellite_max_weight,
            max_single_position_weight=self.custom_configs.get('max_single_position_weight', 0.15),
            max_portfolio_volatility=self.custom_configs.get('max_portfolio_volatility', 0.15)
        )
        self.compliance_monitor = ComplianceMonitor(compliance_rules)

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

            # Initialize strategies
            core_success = self.core_strategy.prepare_data()
            satellite_success = self.satellite_strategy.prepare_data()

            if not core_success or not satellite_success:
                logger.error("Failed to initialize strategies")
                return False

            # Initialize portfolio state (simplified - would create actual portfolio)
            self.current_portfolio = self._create_initial_portfolio()

            logger.info("System initialization completed successfully")
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
                'core': self.core_strategy.config.name,
                'satellite': self.satellite_strategy.config.name
            },
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

        # Validate strategy configurations
        if not self.core_strategy:
            issues.append("Core strategy not initialized")

        if not self.satellite_strategy:
            issues.append("Satellite strategy not initialized")

        return len(issues) == 0, issues