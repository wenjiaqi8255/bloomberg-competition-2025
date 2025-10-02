"""
Orchestration Layer - Coordinated System Management

This package contains components that coordinate different aspects of the trading system:
- Strategy coordination and signal management
- Capital allocation and portfolio management
- IPS compliance monitoring
- Trade execution
- Performance reporting
- System orchestration coordination

Each component has a single responsibility and can be used independently
or as part of a complete system orchestration.
"""

from .coordinator import StrategyCoordinator, CoordinatorConfig
from .allocator import CapitalAllocator, AllocationConfig
from .compliance import ComplianceMonitor, ComplianceRules, ComplianceReport
from .executor import TradeExecutor, ExecutionConfig
from .reporter import PerformanceReporter, ReportConfig

__all__ = [
    # Strategy coordination
    'StrategyCoordinator',
    'CoordinatorConfig',

    # Capital allocation
    'CapitalAllocator',
    'AllocationConfig',

    # Compliance monitoring
    'ComplianceMonitor',
    'ComplianceRules',
    'ComplianceReport',

    # Trade execution
    'TradeExecutor',
    'ExecutionConfig',

    # Performance reporting
    'PerformanceReporter',
    'ReportConfig',
]