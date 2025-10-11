"""
Configuration Validation Utilities
=================================

Unified configuration validation for all orchestration components.
Consolidates validation logic to eliminate duplication and ensure consistency.
"""

import logging
from typing import Dict, List, Any, Tuple, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ComponentConfigValidator:
    """
    Unified configuration validator for orchestration components.
    
    Consolidates validation logic from all components to eliminate duplication
    and ensure consistent validation patterns.
    """
    
    @staticmethod
    def validate_coordinator_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate StrategyCoordinator configuration.
        
        Args:
            config: Coordinator configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Validate max_signals_per_day
        max_signals = config.get('max_signals_per_day', 50)
        if not isinstance(max_signals, int) or max_signals <= 0:
            issues.append("max_signals_per_day must be a positive integer")
        
        # Validate signal_conflict_resolution
        resolution = config.get('signal_conflict_resolution', 'merge')
        if resolution not in ['merge', 'priority', 'cancel']:
            issues.append("signal_conflict_resolution must be one of: merge, priority, cancel")
        
        # Validate min_signal_strength
        min_strength = config.get('min_signal_strength', 0.01)
        if not isinstance(min_strength, (int, float)) or min_strength < 0 or min_strength > 1:
            issues.append("min_signal_strength must be between 0 and 1")
        
        # Validate max_position_size
        max_position = config.get('max_position_size', 0.15)
        if not isinstance(max_position, (int, float)) or max_position <= 0 or max_position > 1:
            issues.append("max_position_size must be between 0 and 1")
        
        return len(issues) == 0, issues

    @staticmethod
    def validate_executor_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate TradeExecutor configuration.
        
        Args:
            config: Executor configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Validate max_order_size_percent
        max_order_size = config.get('max_order_size_percent', 1.0)
        if not isinstance(max_order_size, (int, float)) or max_order_size <= 0 or max_order_size > 0.5:
            issues.append("max_order_size_percent must be between 0 and 50%")
        
        # Validate min_order_size_usd
        min_order_size = config.get('min_order_size_usd', 1000)
        if not isinstance(min_order_size, (int, float)) or min_order_size <= 0:
            issues.append("min_order_size_usd must be positive")
        
        # Validate max_positions_per_day
        max_positions = config.get('max_positions_per_day', 10)
        if not isinstance(max_positions, int) or max_positions <= 0:
            issues.append("max_positions_per_day must be a positive integer")
        
        # Validate commission_rate
        commission_rate = config.get('commission_rate', 0.001)
        if not isinstance(commission_rate, (int, float)) or commission_rate < 0 or commission_rate > 0.01:
            issues.append("commission_rate should be between 0 and 1%")
        
        # Validate cooling_period_hours
        cooling_period = config.get('cooling_period_hours', 1)
        if not isinstance(cooling_period, (int, float)) or cooling_period < 0:
            issues.append("cooling_period_hours must be non-negative")
        
        return len(issues) == 0, issues

    @staticmethod
    def validate_allocator_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate CapitalAllocator configuration.
        
        Args:
            config: Allocator configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Validate strategy_allocations
        strategy_allocations = config.get('strategy_allocations', [])
        if not isinstance(strategy_allocations, list) or not strategy_allocations:
            issues.append("strategy_allocations must be a non-empty list")
            return False, issues
        
        # Validate each strategy allocation
        total_target_weight = 0
        for i, alloc in enumerate(strategy_allocations):
            if not isinstance(alloc, dict):
                issues.append(f"Strategy allocation {i} must be a dictionary")
                continue
            
            # Validate target_weight
            target_weight = alloc.get('target_weight', 0)
            if not isinstance(target_weight, (int, float)) or target_weight < 0 or target_weight > 1:
                issues.append(f"Strategy allocation {i} target_weight must be between 0 and 1")
            
            total_target_weight += target_weight
            
            # Validate min_weight and max_weight
            min_weight = alloc.get('min_weight', 0)
            max_weight = alloc.get('max_weight', 1)
            
            if not isinstance(min_weight, (int, float)) or min_weight < 0 or min_weight > 1:
                issues.append(f"Strategy allocation {i} min_weight must be between 0 and 1")
            
            if not isinstance(max_weight, (int, float)) or max_weight < 0 or max_weight > 1:
                issues.append(f"Strategy allocation {i} max_weight must be between 0 and 1")
            
            if min_weight > max_weight:
                issues.append(f"Strategy allocation {i} min_weight must be <= max_weight")
            
            if not (min_weight <= target_weight <= max_weight):
                issues.append(f"Strategy allocation {i} target_weight must be between min_weight and max_weight")
        
        # Validate total target weight
        if total_target_weight > 1.0:
            issues.append(f"Total target weight {total_target_weight:.1%} exceeds 100%")
        
        # Validate rebalance_threshold
        rebalance_threshold = config.get('rebalance_threshold', 0.05)
        if not isinstance(rebalance_threshold, (int, float)) or rebalance_threshold <= 0 or rebalance_threshold > 0.5:
            issues.append("rebalance_threshold must be between 0 and 50%")
        
        # Validate cash_buffer_weight
        cash_buffer = config.get('cash_buffer_weight', 0.05)
        if not isinstance(cash_buffer, (int, float)) or cash_buffer < 0 or cash_buffer > 0.2:
            issues.append("cash_buffer_weight should be between 0 and 20%")
        
        return len(issues) == 0, issues

    @staticmethod
    def validate_reporter_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate PerformanceReporter configuration.
        
        Args:
            config: Reporter configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Validate significant_return_threshold
        return_threshold = config.get('significant_return_threshold', 0.01)
        if not isinstance(return_threshold, (int, float)) or return_threshold <= 0 or return_threshold > 1:
            issues.append("significant_return_threshold must be between 0 and 100%")
        
        # Validate significant_risk_threshold
        risk_threshold = config.get('significant_risk_threshold', 0.02)
        if not isinstance(risk_threshold, (int, float)) or risk_threshold <= 0 or risk_threshold > 1:
            issues.append("significant_risk_threshold must be between 0 and 100%")
        
        # Validate benchmark_symbol
        benchmark_symbol = config.get('benchmark_symbol', 'SPY')
        if not isinstance(benchmark_symbol, str) or not benchmark_symbol:
            issues.append("benchmark_symbol must be a non-empty string")
        
        # Validate file_format
        file_format = config.get('file_format', 'json')
        if file_format not in ['json', 'csv', 'excel']:
            issues.append("file_format must be one of: json, csv, excel")
        
        return len(issues) == 0, issues

    @staticmethod
    def validate_compliance_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate ComplianceMonitor configuration.
        
        Args:
            config: Compliance configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Validate max_single_position_weight
        max_position = config.get('max_single_position_weight', 0.15)
        if not isinstance(max_position, (int, float)) or max_position <= 0 or max_position > 1:
            issues.append("max_single_position_weight must be between 0 and 1")
        
        # Validate box_exposure_limits
        box_limits = config.get('box_exposure_limits', {})
        if isinstance(box_limits, dict):
            for dimension, limit in box_limits.items():
                if not isinstance(limit, (int, float)) or limit <= 0 or limit > 1:
                    issues.append(f"Box exposure limit for {dimension} must be between 0 and 1")
        
        # Validate max_sector_allocation
        max_sector = config.get('max_sector_allocation', 0.25)
        if not isinstance(max_sector, (int, float)) or max_sector <= 0 or max_sector > 1:
            issues.append("max_sector_allocation must be between 0 and 1")
        
        # Validate max_concentration limits
        max_concentration_top5 = config.get('max_concentration_top5', 0.40)
        if not isinstance(max_concentration_top5, (int, float)) or max_concentration_top5 <= 0 or max_concentration_top5 > 1:
            issues.append("max_concentration_top5 must be between 0 and 1")
        
        max_concentration_top10 = config.get('max_concentration_top10', 0.60)
        if not isinstance(max_concentration_top10, (int, float)) or max_concentration_top10 <= 0 or max_concentration_top10 > 1:
            issues.append("max_concentration_top10 must be between 0 and 1")
        
        if max_concentration_top5 > max_concentration_top10:
            issues.append("max_concentration_top5 must be <= max_concentration_top10")
        
        return len(issues) == 0, issues

    @staticmethod
    def validate_portfolio_construction_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate portfolio construction configuration.
        
        Args:
            config: Portfolio construction configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Validate method
        method = config.get('method', 'quantitative')
        if method not in ['quantitative', 'box_based']:
            issues.append("method must be 'quantitative' or 'box_based'")
        
        # Validate universe_size for quantitative method
        if method == 'quantitative':
            universe_size = config.get('universe_size', 100)
            if not isinstance(universe_size, int) or universe_size <= 0:
                issues.append("universe_size must be a positive integer")
        
        # Validate optimizer config
        optimizer_config = config.get('optimizer', {})
        if isinstance(optimizer_config, dict):
            optimizer_method = optimizer_config.get('method', 'mean_variance')
            if optimizer_method not in ['mean_variance', 'risk_parity', 'equal_weight']:
                issues.append("optimizer method must be one of: mean_variance, risk_parity, equal_weight")
            
            risk_aversion = optimizer_config.get('risk_aversion', 2.0)
            if not isinstance(risk_aversion, (int, float)) or risk_aversion <= 0:
                issues.append("risk_aversion must be positive")
        
        # Validate covariance config
        covariance_config = config.get('covariance', {})
        if isinstance(covariance_config, dict):
            lookback_days = covariance_config.get('lookback_days', 252)
            if not isinstance(lookback_days, int) or lookback_days <= 0:
                issues.append("covariance lookback_days must be a positive integer")
            
            cov_method = covariance_config.get('method', 'ledoit_wolf')
            if cov_method not in ['ledoit_wolf', 'sample', 'factor_model']:
                issues.append("covariance method must be one of: ledoit_wolf, sample, factor_model")
        
        return len(issues) == 0, issues

    @staticmethod
    def validate_system_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate complete system configuration.
        
        Args:
            config: Complete system configuration dictionary
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Validate system-level config
        initial_capital = config.get('initial_capital', 1000000)
        if not isinstance(initial_capital, (int, float)) or initial_capital <= 0:
            issues.append("initial_capital must be positive")
        
        enable_short_selling = config.get('enable_short_selling', False)
        if not isinstance(enable_short_selling, bool):
            issues.append("enable_short_selling must be boolean")
        
        # Validate strategies
        strategies = config.get('strategies', [])
        if not isinstance(strategies, list):
            issues.append("strategies must be a list")
        else:
            strategy_names = []
            for i, strategy in enumerate(strategies):
                if not isinstance(strategy, dict):
                    issues.append(f"Strategy {i} must be a dictionary")
                    continue
                
                name = strategy.get('name')
                if not name or not isinstance(name, str):
                    issues.append(f"Strategy {i} must have a valid name")
                else:
                    if name in strategy_names:
                        issues.append(f"Duplicate strategy name: {name}")
                    strategy_names.append(name)
        
        # Validate portfolio construction config
        portfolio_config = config.get('portfolio_construction', {})
        if portfolio_config:
            is_valid, portfolio_issues = ComponentConfigValidator.validate_portfolio_construction_config(portfolio_config)
            if not is_valid:
                issues.extend([f"Portfolio construction: {issue}" for issue in portfolio_issues])
        
        return len(issues) == 0, issues
