"""
Orchestration Module - Multi-Strategy System Coordination
=========================================================

This module provides the SystemOrchestrator and related components for
coordinating multiple trading strategies in a production environment.
"""

from ...trading_system.metamodel.meta_model import MetaModel
from .components import (
    StrategyCoordinator, CoordinatorConfig,
    CapitalAllocator, AllocationConfig,
    ComplianceMonitor, ComplianceRules, ComplianceReport,
    TradeExecutor, ExecutionConfig,
    PerformanceReporter, ReportConfig
)

__all__ = [
    # Main orchestrator
    'StrategyCoordinator', 'CoordinatorConfig',
    'CapitalAllocator', 'AllocationConfig', 
    'ComplianceMonitor', 'ComplianceRules', 'ComplianceReport',
    'TradeExecutor', 'ExecutionConfig',
    'PerformanceReporter', 'ReportConfig'
    
    # Meta model
    'MetaModel',
    
    # Components
    'StrategyCoordinator', 'CoordinatorConfig',
    'CapitalAllocator', 'AllocationConfig', 
    'ComplianceMonitor', 'ComplianceRules', 'ComplianceReport',
    'TradeExecutor', 'ExecutionConfig',
    'PerformanceReporter', 'ReportConfig'
]

