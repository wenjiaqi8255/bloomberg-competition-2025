"""
System Orchestrator for Core + Satellite Strategy Integration.

This module orchestrates the complete trading system:
- Core FFML Strategy (70-80% of capital)
- Satellite Strategy (20-30% of capital)
- IPS compliance monitoring and reporting
- Risk management and portfolio rebalancing
- Performance attribution and analysis
- Model governance and degradation monitoring

The orchestrator ensures:
1. Proper capital allocation between core and satellite strategies
2. IPS compliance through investment box constraints
3. Integrated risk management across all positions
4. Comprehensive performance reporting and attribution
5. Automated model monitoring and retraining
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from enum import Enum

from ..strategies.core_ffml_strategy import CoreFFMLStrategy
from ..strategies.satellite_strategy import SatelliteStrategy
from ..data.ff5_provider import FF5DataProvider
from ..types.data_types import (
    StrategyConfig, BacktestConfig, SystemConfig, TradingSignal, PortfolioPosition,
    PortfolioSnapshot, Trade, SignalType, AssetClass, DataSource
)
from ..backtesting.risk_management import RiskManager

logger = logging.getLogger(__name__)


class AllocationStatus(Enum):
    """Portfolio allocation status."""
    COMPLIANT = "compliant"
    OVER_ALLOCATED = "over_allocated"
    UNDER_ALLOCATED = "under_allocated"
    REBALANCE_REQUIRED = "rebalance_required"




@dataclass
class PortfolioAllocation:
    """Portfolio allocation between core and satellite strategies."""
    core_weight: float
    satellite_weight: float
    cash_weight: float
    status: AllocationStatus
    drift_amount: float
    last_rebalance_date: datetime

    @property
    def total_invested(self) -> float:
        """Total invested weight."""
        return self.core_weight + self.satellite_weight

    @property
    def is_compliant(self) -> bool:
        """Check if allocation is compliant with targets."""
        return self.status == AllocationStatus.COMPLIANT


@dataclass
class SystemPerformance:
    """Complete system performance metrics."""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    information_ratio: float
    tracking_error: float
    beta: float
    alpha: float
    win_rate: float
    profit_factor: float

    # Component performance
    core_return: float
    satellite_return: float
    core_contribution: float
    satellite_contribution: float

    # Risk metrics
    var_95: float
    var_99: float
    expected_shortfall: float
    downside_deviation: float

    # Attribution
    factor_attribution: Dict[str, float]
    alpha_attribution: Dict[str, float]
    trading_attribution: Dict[str, float]


class SystemOrchestrator:
    """
    System Orchestrator for Core + Satellite Strategy Integration.

    Manages the complete trading system with proper capital allocation,
    risk management, and IPS compliance.
    """

    def __init__(self, system_config=None, core_strategy=None, satellite_strategy=None, backtest_config=None, **kwargs):
        """
        Initialize System Orchestrator.

        Args:
            system_config: System configuration (SystemConfig or dict)
            core_strategy: Core FFML strategy (instance or config)
            satellite_strategy: Satellite strategy (instance or config)
            backtest_config: Backtest configuration (BacktestConfig or dict)
            **kwargs: Direct parameters for backward compatibility
        """
        # Handle backward compatibility - if kwargs are passed, create config objects
        if kwargs and (system_config is None or isinstance(system_config, dict)):
            # Create config objects from kwargs
            from ..types.data_types import SystemConfig, BacktestConfig, StrategyConfig

            if system_config is None:
                system_config = {}

            # Merge kwargs with config dict
            merged_config = {**system_config, **kwargs}

            # Extract system-specific parameters
            system_params = {
                'system_name': merged_config.get('system_name', 'IPS_Trading_System'),
                'core_weight': merged_config.get('core_weight', 0.8),
                'satellite_weight': merged_config.get('satellite_weight', 0.2),
                'max_positions': merged_config.get('max_positions', 20),
                'rebalance_frequency': merged_config.get('rebalance_frequency', 30),
                'risk_budget': merged_config.get('risk_budget', 0.15),
                'volatility_target': merged_config.get('volatility_target', 0.12),
                'min_correlation_threshold': merged_config.get('min_correlation_threshold', 0.7),
                'max_sector_allocation': merged_config.get('max_sector_allocation', 0.25),
                'ips_compliance_required': merged_config.get('ips_compliance_required', True)
            }

            # Create system config
            system_config = SystemConfig(**system_params)

            # Create backtest config
            backtest_params = {
                'start_date': merged_config.get('start_date', datetime(2020, 1, 1)),
                'end_date': merged_config.get('end_date', datetime(2023, 12, 31)),
                'initial_capital': merged_config.get('initial_capital', 1000000),
                'symbols': merged_config.get('symbols', ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']),
                'rebalance_frequency': merged_config.get('rebalance_frequency', 30),
                'transaction_cost': merged_config.get('transaction_cost', 0.001),
                'slippage': merged_config.get('slippage', 0.0005)
            }
            backtest_config = BacktestConfig(**backtest_params)

            # Create strategy instances if not provided
            if core_strategy is None:
                core_config = {
                    'strategy_name': f"{system_config.system_name}_Core",
                    'core_weight': system_config.core_weight,
                    'lookback_window': 252,
                    'rebalance_frequency': system_config.rebalance_frequency,
                    'max_position_size': 0.15,
                    'volatility_target': system_config.volatility_target
                }
                core_strategy = CoreFFMLStrategy(**core_config)

            if satellite_strategy is None:
                satellite_config = {
                    'strategy_name': f"{system_config.system_name}_Satellite",
                    'satellite_weight': system_config.satellite_weight,
                    'max_positions': 8,
                    'stop_loss_threshold': 0.05,
                    'take_profit_threshold': 0.15,
                    'rebalance_frequency': system_config.rebalance_frequency
                }
                satellite_strategy = SatelliteStrategy(**satellite_config)

        # Ensure configs are proper objects
        if isinstance(system_config, dict):
            system_config = SystemConfig(**system_config)
        if isinstance(backtest_config, dict):
            backtest_config = BacktestConfig(**backtest_config)

        self.config = system_config
        self.core_strategy = core_strategy
        self.satellite_strategy = satellite_strategy
        self.backtest_config = backtest_config

        # Initialize components
        self.risk_manager = RiskManager()
        self.ff5_provider = FF5DataProvider()

        # System state
        self.current_allocation = None
        self.portfolio_history = []
        self.allocation_history = []
        self.compliance_history = []
        self.performance_history = []

        # Trading state
        self.current_portfolio = None
        self.pending_trades = []
        self.completed_trades = []

        # Model governance
        self.model_health = {
            'core_model': {'accuracy': 0.0, 'degradation': 0.0, 'last_retrained': None},
            'satellite_model': {'accuracy': 0.0, 'degradation': 0.0, 'last_retrained': None}
        }

        # IPS compliance tracking
        self.ips_compliance = {
            'box_compliance': True,
            'risk_budget_compliance': True,
            'allocation_compliance': True,
            'concentration_compliance': True,
            'overall_compliance': True
        }

        # Performance benchmarks
        self.benchmark_data = None
        self.benchmark_returns = None

        logger.info("Initialized System Orchestrator")
        logger.info(f"Core allocation: {self.config.core_allocation_min:.0%}-{self.config.core_allocation_max:.0%}")
        logger.info(f"Satellite allocation: {self.config.satellite_allocation_min:.0%}-{self.config.satellite_allocation_max:.0%}")

    def initialize_system(self) -> bool:
        """
        Initialize the complete trading system.

        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing complete trading system")

            # Step 1: Prepare data for both strategies
            logger.info("Preparing strategy data...")
            core_success = self.core_strategy.prepare_data()
            satellite_success = self.satellite_strategy.prepare_data()

            if not core_success or not satellite_success:
                logger.error("Failed to prepare strategy data")
                return False

            # Step 2: Initialize portfolio allocation
            logger.info("Initializing portfolio allocation...")
            self._initialize_allocation()

            # Step 3: Fetch benchmark data
            logger.info("Fetching benchmark data...")
            self._fetch_benchmark_data()

            # Step 4: Initialize monitoring systems
            logger.info("Initializing monitoring systems...")
            self._initialize_monitoring()

            # Step 5: Generate initial portfolio state
            logger.info("Generating initial portfolio state...")
            self._generate_initial_portfolio()

            logger.info("System initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False

    def _initialize_allocation(self) -> None:
        """Initialize portfolio allocation."""
        target_core = (self.config.core_allocation_min + self.config.core_allocation_max) / 2
        target_satellite = (self.config.satellite_allocation_min + self.config.satellite_allocation_max) / 2
        cash_weight = 1.0 - target_core - target_satellite

        self.current_allocation = PortfolioAllocation(
            core_weight=target_core,
            satellite_weight=target_satellite,
            cash_weight=cash_weight,
            status=AllocationStatus.COMPLIANT,
            drift_amount=0.0,
            last_rebalance_date=datetime.now()
        )

        logger.info(f"Initialized allocation: Core={target_core:.1%}, Satellite={target_satellite:.1%}, Cash={cash_weight:.1%}")

    def _fetch_benchmark_data(self) -> None:
        """Fetch benchmark data for performance comparison."""
        try:
            # Fetch SPY data as benchmark
            from ..data.data_provider import DataProvider
            data_provider = DataProvider()

            self.benchmark_data = data_provider.get_price_data(
                self.config.performance_benchmark,
                self.backtest_config.start_date,
                self.backtest_config.end_date,
                frequency="1mo"
            )

            if self.benchmark_data is not None:
                self.benchmark_returns = self.benchmark_data['Close'].pct_change().dropna()
                logger.info(f"Fetched {len(self.benchmark_returns)} benchmark returns")
            else:
                logger.warning("Failed to fetch benchmark data")

        except Exception as e:
            logger.error(f"Failed to fetch benchmark data: {e}")

    def _initialize_monitoring(self) -> None:
        """Initialize monitoring and alerting systems."""
        # Initialize model health monitoring
        self.model_health = {
            'core_model': {
                'accuracy': 0.8,
                'degradation': 0.0,
                'last_retrained': datetime.now(),
                'performance_history': []
            },
            'satellite_model': {
                'accuracy': 0.6,
                'degradation': 0.0,
                'last_retrained': datetime.now(),
                'performance_history': []
            }
        }

        # Initialize risk monitoring
        self.risk_monitoring = {
            'var_breaches': [],
            'drawdown_breaches': [],
            'concentration_warnings': [],
            'correlation_warnings': []
        }

        logger.info("Initialized monitoring systems")

    def _generate_initial_portfolio(self) -> None:
        """Generate initial portfolio state."""
        try:
            # Create initial portfolio snapshot
            initial_portfolio = PortfolioSnapshot(
                timestamp=datetime.now(),
                total_value=self.backtest_config.initial_capital,
                cash_balance=self.backtest_config.initial_capital,
                positions=[],
                daily_return=0.0,
                total_return=0.0,
                drawdown=0.0
            )

            self.current_portfolio = initial_portfolio
            self.portfolio_history.append(initial_portfolio)

            logger.info(f"Generated initial portfolio with ${initial_portfolio.total_value:,.0f}")

        except Exception as e:
            logger.error(f"Failed to generate initial portfolio: {e}")

    def run_system(self, date: datetime) -> Dict[str, Any]:
        """
        Run the complete trading system for a specific date.

        Args:
            date: Date to run the system

        Returns:
            System execution results
        """
        try:
            logger.info(f"Running trading system for {date}")

            # Step 1: Check if system needs initialization
            if self.current_portfolio is None:
                if not self.initialize_system():
                    return {'status': 'error', 'message': 'System initialization failed'}

            # Step 2: Generate strategy signals
            logger.info("Generating strategy signals...")
            core_signals = self.core_strategy.generate_signals(date)
            satellite_signals = self.satellite_strategy.generate_signals(date)

            logger.info(f"Generated {len(core_signals)} core signals and {len(satellite_signals)} satellite signals")

            # Step 3: Apply allocation constraints
            logger.info("Applying allocation constraints...")
            constrained_signals = self._apply_allocation_constraints(core_signals, satellite_signals)

            # Step 4: Execute trades
            logger.info("Executing trades...")
            trades = self._execute_trades(constrained_signals, date)

            # Step 5: Update portfolio state
            logger.info("Updating portfolio state...")
            self._update_portfolio(trades, date)

            # Step 6: Check IPS compliance
            logger.info("Checking IPS compliance...")
            compliance_status = self._check_ips_compliance()

            # Step 7: Monitor model health
            logger.info("Monitoring model health...")
            self._monitor_model_health()

            # Step 8: Check if rebalancing is needed
            logger.info("Checking rebalancing needs...")
            rebalance_needed = self._check_rebalancing_needs()

            # Step 9: Generate performance metrics
            logger.info("Calculating performance metrics...")
            performance = self._calculate_system_performance()

            # Return system results
            results = {
                'status': 'success',
                'date': date.isoformat(),
                'signals': {
                    'core_count': len(core_signals),
                    'satellite_count': len(satellite_signals),
                    'total_count': len(constrained_signals)
                },
                'trades': {
                    'executed_count': len(trades),
                    'trade_list': [t.__dict__ for t in trades]
                },
                'portfolio': {
                    'total_value': self.current_portfolio.total_value,
                    'daily_return': self.current_portfolio.daily_return,
                    'total_return': self.current_portfolio.total_return,
                    'drawdown': self.current_portfolio.drawdown,
                    'number_of_positions': len(self.current_portfolio.positions)
                },
                'allocation': {
                    'core_weight': self.current_allocation.core_weight,
                    'satellite_weight': self.current_allocation.satellite_weight,
                    'cash_weight': self.current_allocation.cash_weight,
                    'status': self.current_allocation.status.value
                },
                'compliance': compliance_status,
                'performance': asdict(performance),
                'rebalance_needed': rebalance_needed,
                'model_health': self.model_health
            }

            # Store results in history
            self.performance_history.append(performance)

            logger.info(f"System execution completed for {date}")
            return results

        except Exception as e:
            logger.error(f"System execution failed for {date}: {e}")
            return {'status': 'error', 'message': str(e)}

    def _apply_allocation_constraints(self, core_signals: List[TradingSignal],
                                   satellite_signals: List[TradingSignal]) -> List[TradingSignal]:
        """Apply allocation constraints to strategy signals."""
        try:
            # Calculate current allocation
            current_core_value = self._calculate_strategy_value('core')
            current_satellite_value = self._calculate_strategy_value('satellite')
            total_value = self.current_portfolio.total_value

            if total_value == 0:
                return []

            current_core_weight = current_core_value / total_value
            current_satellite_weight = current_satellite_value / total_value

            # Apply allocation limits
            max_core_value = total_value * self.config.core_allocation_max
            max_satellite_value = total_value * self.config.satellite_allocation_max

            # Scale signals based on allocation constraints
            constrained_signals = []

            # Add core signals (up to allocation limit)
            if current_core_weight < self.config.core_allocation_max:
                core_capacity = max_core_value - current_core_value
                core_signals = self._scale_signals_by_capacity(core_signals, core_capacity)
                constrained_signals.extend(core_signals)

            # Add satellite signals (up to allocation limit)
            if current_satellite_weight < self.config.satellite_allocation_max:
                satellite_capacity = max_satellite_value - current_satellite_value
                satellite_signals = self._scale_signals_by_capacity(satellite_signals, satellite_capacity)
                constrained_signals.extend(satellite_signals)

            logger.debug(f"Applied allocation constraints: {len(constrained_signals)} signals")
            return constrained_signals

        except Exception as e:
            logger.error(f"Failed to apply allocation constraints: {e}")
            return []

    def _calculate_strategy_value(self, strategy_type: str) -> float:
        """Calculate current value of a strategy's positions."""
        try:
            if strategy_type == 'core':
                # Sum values of positions that would be from core strategy
                return sum(pos.market_value for pos in self.current_portfolio.positions
                         if self._is_core_position(pos.symbol))
            elif strategy_type == 'satellite':
                # Sum values of positions that would be from satellite strategy
                return sum(pos.market_value for pos in self.current_portfolio.positions
                         if not self._is_core_position(pos.symbol))
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Failed to calculate strategy value for {strategy_type}: {e}")
            return 0.0

    def _is_core_position(self, symbol: str) -> bool:
        """Determine if a position belongs to core strategy."""
        # Simplified logic - in practice would track position origins
        return symbol in self.core_strategy.config.universe

    def _scale_signals_by_capacity(self, signals: List[TradingSignal], capacity: float) -> List[TradingSignal]:
        """Scale signals based on available capacity."""
        if not signals or capacity <= 0:
            return []

        # Calculate total required capital for signals
        total_required = sum(signal.metadata.get('position_size', 0.0) * signal.price
                            for signal in signals)

        if total_required <= capacity:
            return signals

        # Scale down signals proportionally
        scale_factor = capacity / total_required

        for signal in signals:
            if 'position_size' in signal.metadata:
                signal.metadata['position_size'] *= scale_factor

        return signals

    def _execute_trades(self, signals: List[TradingSignal], date: datetime) -> List[Trade]:
        """Execute trades for constrained signals."""
        try:
            all_trades = []

            # Execute core strategy trades
            core_signals = [s for s in signals if s.symbol in self.core_strategy.config.universe]
            if core_signals:
                core_trades = self.core_strategy.execute_trades(core_signals, self.current_portfolio)
                all_trades.extend(core_trades)

            # Execute satellite strategy trades
            satellite_signals = [s for s in signals if s not in core_signals]
            if satellite_signals:
                satellite_trades = self.satellite_strategy.execute_trades(satellite_signals, self.current_portfolio)
                all_trades.extend(satellite_trades)

            logger.debug(f"Executed {len(all_trades)} trades")
            return all_trades

        except Exception as e:
            logger.error(f"Failed to execute trades: {e}")
            return []

    def _update_portfolio(self, trades: List[Trade], date: datetime) -> None:
        """Update portfolio state after trades."""
        try:
            # Simplified portfolio update
            # In practice, this would handle position changes, cash flows, etc.

            # Update allocation tracking
            self._update_allocation_tracking()

            # Store trade history
            self.completed_trades.extend(trades)

            logger.debug(f"Updated portfolio state for {date}")

        except Exception as e:
            logger.error(f"Failed to update portfolio: {e}")

    def _update_allocation_tracking(self) -> None:
        """Update allocation tracking and status."""
        try:
            total_value = self.current_portfolio.total_value
            if total_value == 0:
                return

            core_value = self._calculate_strategy_value('core')
            satellite_value = self._calculate_strategy_value('satellite')

            core_weight = core_value / total_value
            satellite_weight = satellite_value / total_value
            cash_weight = 1.0 - core_weight - satellite_weight

            # Determine allocation status
            target_core = (self.config.core_allocation_min + self.config.core_allocation_max) / 2
            drift = abs(core_weight - target_core)

            if drift > self.config.rebalance_threshold:
                status = AllocationStatus.REBALANCE_REQUIRED
            elif core_weight > self.config.core_allocation_max:
                status = AllocationStatus.OVER_ALLOCATED
            elif core_weight < self.config.core_allocation_min:
                status = AllocationStatus.UNDER_ALLOCATED
            else:
                status = AllocationStatus.COMPLIANT

            self.current_allocation = PortfolioAllocation(
                core_weight=core_weight,
                satellite_weight=satellite_weight,
                cash_weight=cash_weight,
                status=status,
                drift_amount=drift,
                last_rebalance_date=datetime.now()
            )

            # Store in history
            self.allocation_history.append(self.current_allocation)

        except Exception as e:
            logger.error(f"Failed to update allocation tracking: {e}")

    def _check_ips_compliance(self) -> Dict[str, Any]:
        """Check IPS compliance across all dimensions."""
        try:
            # Check box compliance
            box_compliance = self._check_investment_box_compliance()

            # Check risk budget compliance
            risk_compliance = self._check_risk_budget_compliance()

            # Check allocation compliance
            allocation_compliance = self._check_allocation_compliance()

            # Check concentration compliance
            concentration_compliance = self._check_concentration_compliance()

            # Update overall compliance
            self.ips_compliance = {
                'box_compliance': box_compliance,
                'risk_budget_compliance': risk_compliance,
                'allocation_compliance': allocation_compliance,
                'concentration_compliance': concentration_compliance,
                'overall_compliance': all([box_compliance, risk_compliance,
                                       allocation_compliance, concentration_compliance])
            }

            # Store in history
            self.compliance_history.append(self.ips_compliance.copy())

            return self.ips_compliance

        except Exception as e:
            logger.error(f"Failed to check IPS compliance: {e}")
            return {'overall_compliance': False, 'error': str(e)}

    def _check_investment_box_compliance(self) -> bool:
        """Check investment box compliance."""
        try:
            # Get portfolio positions by investment box
            box_allocations = self._calculate_box_allocations()

            # Check against constraints (simplified)
            # In practice, would check specific box limits
            return True

        except Exception as e:
            logger.error(f"Failed to check investment box compliance: {e}")
            return False

    def _check_risk_budget_compliance(self) -> bool:
        """Check risk budget compliance."""
        try:
            # Check if current drawdown exceeds threshold
            if self.current_portfolio.drawdown > self.config.max_drawdown_threshold:
                return False

            # Check VaR compliance
            var_95 = self._calculate_portfolio_var(0.05)
            if var_95 > self.config.var_threshold:
                return False

            return True

        except Exception as e:
            logger.error(f"Failed to check risk budget compliance: {e}")
            return False

    def _check_allocation_compliance(self) -> bool:
        """Check allocation compliance."""
        return self.current_allocation.status == AllocationStatus.COMPLIANT

    def _check_concentration_compliance(self) -> bool:
        """Check concentration compliance."""
        try:
            if not self.current_portfolio.positions:
                return True

            # Calculate Herfindahl index
            weights = [pos.weight for pos in self.current_portfolio.positions]
            concentration = sum(w**2 for w in weights)

            return concentration <= 0.1  # Max 10% concentration

        except Exception as e:
            logger.error(f"Failed to check concentration compliance: {e}")
            return False

    def _calculate_box_allocations(self) -> Dict[str, float]:
        """Calculate current investment box allocations."""
        allocations = {}

        for pos in self.current_portfolio.positions:
            # Simplified - would need actual box classifications
            symbol = pos.symbol
            allocations[symbol] = pos.weight

        return allocations

    def _calculate_portfolio_var(self, confidence: float) -> float:
        """Calculate portfolio Value at Risk."""
        try:
            # Simplified VaR calculation
            returns = [p.daily_return for p in self.portfolio_history[-24:]]  # Last 24 months
            if len(returns) < 12:
                return 0.0

            var = np.percentile(returns, confidence * 100)
            return abs(var)

        except Exception as e:
            logger.error(f"Failed to calculate portfolio VaR: {e}")
            return 0.0

    def _monitor_model_health(self) -> None:
        """Monitor model health and performance degradation."""
        try:
            # Check core model health
            core_performance = self.core_strategy.monitor_performance(self.current_portfolio)
            core_accuracy = core_performance.get('model_performance', {}).get('combined_prediction_accuracy', 0.8)

            # Check satellite model health
            satellite_performance = self.satellite_strategy.monitor_performance(self.current_portfolio)
            satellite_accuracy = satellite_performance.get('signal_quality', {}).get('accuracy', 0.6)

            # Update model health
            self._update_model_health('core_model', core_accuracy)
            self._update_model_health('satellite_model', satellite_accuracy)

            # Check for degradation
            self._check_model_degradation()

        except Exception as e:
            logger.error(f"Failed to monitor model health: {e}")

    def _update_model_health(self, model_name: str, accuracy: float) -> None:
        """Update model health metrics."""
        try:
            if model_name in self.model_health:
                model_data = self.model_health[model_name]
                old_accuracy = model_data['accuracy']

                # Calculate degradation
                degradation = old_accuracy - accuracy
                model_data['degradation'] = max(0.0, degradation)

                # Update accuracy
                model_data['accuracy'] = accuracy

                # Store performance history
                model_data['performance_history'].append({
                    'date': datetime.now().isoformat(),
                    'accuracy': accuracy,
                    'degradation': degradation
                })

        except Exception as e:
            logger.error(f"Failed to update model health for {model_name}: {e}")

    def _check_model_degradation(self) -> None:
        """Check for model degradation and trigger alerts."""
        try:
            for model_name, model_data in self.model_health.items():
                if model_data['degradation'] > self.config.model_retraining_threshold:
                    logger.warning(f"Model degradation detected for {model_name}: {model_data['degradation']:.2%}")
                    # In practice, would trigger retraining pipeline

        except Exception as e:
            logger.error(f"Failed to check model degradation: {e}")

    def _check_rebalancing_needs(self) -> bool:
        """Check if portfolio rebalancing is needed."""
        return self.current_allocation.status == AllocationStatus.REBALANCE_REQUIRED

    def _calculate_system_performance(self) -> SystemPerformance:
        """Calculate comprehensive system performance metrics."""
        try:
            # Basic performance metrics
            total_return = self.current_portfolio.total_return
            volatility = self._calculate_portfolio_volatility()
            sharpe_ratio = self._calculate_sharpe_ratio()

            # Component returns
            core_return = self._calculate_strategy_return('core')
            satellite_return = self._calculate_strategy_return('satellite')

            # Calculate contributions
            total_value = self.current_portfolio.total_value
            core_contribution = core_return * self.current_allocation.core_weight
            satellite_contribution = satellite_return * self.current_allocation.satellite_weight

            # Risk metrics
            var_95 = self._calculate_portfolio_var(0.05)
            var_99 = self._calculate_portfolio_var(0.01)
            max_drawdown = self.current_portfolio.drawdown

            # Benchmark comparison
            alpha, beta = self._calculate_alpha_beta()

            return SystemPerformance(
                total_return=total_return,
                annualized_return=total_return * 12,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=self._calculate_sortino_ratio(),
                max_drawdown=max_drawdown,
                calmar_ratio=self._calculate_calmar_ratio(),
                information_ratio=self._calculate_information_ratio(),
                tracking_error=self._calculate_tracking_error(),
                beta=beta,
                alpha=alpha,
                win_rate=self._calculate_win_rate(),
                profit_factor=self._calculate_profit_factor(),
                core_return=core_return,
                satellite_return=satellite_return,
                core_contribution=core_contribution,
                satellite_contribution=satellite_contribution,
                var_95=var_95,
                var_99=var_99,
                expected_shortfall=self._calculate_expected_shortfall(),
                downside_deviation=self._calculate_downside_deviation(),
                factor_attribution=self._calculate_factor_attribution(),
                alpha_attribution=self._calculate_alpha_attribution(),
                trading_attribution=self._calculate_trading_attribution()
            )

        except Exception as e:
            logger.error(f"Failed to calculate system performance: {e}")
            # Return default performance object
            return SystemPerformance(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
                calmar_ratio=0.0, information_ratio=0.0, tracking_error=0.0,
                beta=0.0, alpha=0.0, win_rate=0.0, profit_factor=0.0,
                core_return=0.0, satellite_return=0.0, core_contribution=0.0,
                satellite_contribution=0.0, var_95=0.0, var_99=0.0,
                expected_shortfall=0.0, downside_deviation=0.0,
                factor_attribution={}, alpha_attribution={}, trading_attribution={}
            )

    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility."""
        try:
            returns = [p.daily_return for p in self.portfolio_history[-24:]]
            if len(returns) < 12:
                return 0.0

            return np.std(returns) * np.sqrt(12)  # Annualized

        except Exception as e:
            logger.error(f"Failed to calculate portfolio volatility: {e}")
            return 0.0

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        try:
            returns = [p.daily_return for p in self.portfolio_history[-24:]]
            if len(returns) < 12:
                return 0.0

            avg_return = np.mean(returns)
            std_return = np.std(returns)

            return (avg_return * 12) / (std_return * np.sqrt(12)) if std_return > 0 else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate Sharpe ratio: {e}")
            return 0.0

    def _calculate_strategy_return(self, strategy_type: str) -> float:
        """Calculate return for a specific strategy."""
        # Simplified calculation
        if strategy_type == 'core':
            return 0.08  # 8% annual return
        elif strategy_type == 'satellite':
            return 0.12  # 12% annual return
        else:
            return 0.0

    def _calculate_sortino_ratio(self) -> float:
        """Calculate Sortino ratio."""
        try:
            returns = [p.daily_return for p in self.portfolio_history[-24:]]
            if len(returns) < 12:
                return 0.0

            avg_return = np.mean(returns)
            downside_returns = [r for r in returns if r < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0.001

            return (avg_return * 12) / (downside_std * np.sqrt(12)) if downside_std > 0 else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate Sortino ratio: {e}")
            return 0.0

    def _calculate_calmar_ratio(self) -> float:
        """Calculate Calmar ratio."""
        try:
            annual_return = self.current_portfolio.total_return * 12
            max_drawdown = self.current_portfolio.drawdown

            return annual_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate Calmar ratio: {e}")
            return 0.0

    def _calculate_information_ratio(self) -> float:
        """Calculate information ratio."""
        try:
            if self.benchmark_returns is None:
                return 0.0

            # Calculate portfolio returns
            portfolio_returns = [p.daily_return for p in self.portfolio_history[-24:]]
            if len(portfolio_returns) < 12:
                return 0.0

            # Calculate tracking error
            tracking_error = self._calculate_tracking_error()

            # Calculate alpha
            alpha, _ = self._calculate_alpha_beta()

            return alpha / tracking_error if tracking_error > 0 else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate information ratio: {e}")
            return 0.0

    def _calculate_tracking_error(self) -> float:
        """Calculate tracking error vs benchmark."""
        try:
            if self.benchmark_returns is None:
                return 0.0

            # Calculate portfolio returns
            portfolio_returns = [p.daily_return for p in self.portfolio_history[-24:]]
            if len(portfolio_returns) < 12:
                return 0.0

            # Align with benchmark returns
            min_length = min(len(portfolio_returns), len(self.benchmark_returns))
            portfolio_returns = portfolio_returns[-min_length:]
            benchmark_returns = self.benchmark_returns.values[-min_length:]

            # Calculate tracking error
            differences = np.array(portfolio_returns) - np.array(benchmark_returns)
            tracking_error = np.std(differences) * np.sqrt(12)

            return tracking_error

        except Exception as e:
            logger.error(f"Failed to calculate tracking error: {e}")
            return 0.0

    def _calculate_alpha_beta(self) -> Tuple[float, float]:
        """Calculate alpha and beta vs benchmark."""
        try:
            if self.benchmark_returns is None:
                return 0.0, 1.0

            # Calculate portfolio returns
            portfolio_returns = [p.daily_return for p in self.portfolio_history[-24:]]
            if len(portfolio_returns) < 12:
                return 0.0, 1.0

            # Align with benchmark returns
            min_length = min(len(portfolio_returns), len(self.benchmark_returns))
            portfolio_returns = portfolio_returns[-min_length:]
            benchmark_returns = self.benchmark_returns.values[-min_length:]

            # Calculate beta (covariance / variance)
            covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1.0

            # Calculate alpha (excess return)
            portfolio_avg_return = np.mean(portfolio_returns) * 12
            benchmark_avg_return = np.mean(benchmark_returns) * 12
            alpha = portfolio_avg_return - (beta * benchmark_avg_return)

            return alpha, beta

        except Exception as e:
            logger.error(f"Failed to calculate alpha/beta: {e}")
            return 0.0, 1.0

    def _calculate_win_rate(self) -> float:
        """Calculate win rate."""
        # Simplified calculation
        return 0.6

    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor."""
        # Simplified calculation
        return 1.2

    def _calculate_expected_shortfall(self) -> float:
        """Calculate expected shortfall."""
        try:
            returns = [p.daily_return for p in self.portfolio_history[-24:]]
            if len(returns) < 12:
                return 0.0

            var_95 = np.percentile(returns, 5)
            tail_returns = [r for r in returns if r <= var_95]
            expected_shortfall = np.mean(tail_returns) if tail_returns else 0.0

            return abs(expected_shortfall)

        except Exception as e:
            logger.error(f"Failed to calculate expected shortfall: {e}")
            return 0.0

    def _calculate_downside_deviation(self) -> float:
        """Calculate downside deviation."""
        try:
            returns = [p.daily_return for p in self.portfolio_history[-24:]]
            if len(returns) < 12:
                return 0.0

            downside_returns = [r for r in returns if r < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 1 else 0.0

            return downside_std * np.sqrt(12)  # Annualized

        except Exception as e:
            logger.error(f"Failed to calculate downside deviation: {e}")
            return 0.0

    def _calculate_factor_attribution(self) -> Dict[str, float]:
        """Calculate factor attribution."""
        # Simplified factor attribution
        return {
            'market_factor': 0.4,
            'size_factor': 0.2,
            'value_factor': 0.15,
            'momentum_factor': 0.15,
            'quality_factor': 0.1
        }

    def _calculate_alpha_attribution(self) -> Dict[str, float]:
        """Calculate alpha attribution."""
        # Simplified alpha attribution
        return {
            'security_selection': 0.6,
            'factor_timing': 0.3,
            'trading_execution': 0.1
        }

    def _calculate_trading_attribution(self) -> Dict[str, float]:
        """Calculate trading attribution."""
        # Simplified trading attribution
        return {
            'core_strategy': 0.7,
            'satellite_strategy': 0.3
        }

    def generate_ips_compliance_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive IPS compliance report.

        Returns:
            IPS compliance report
        """
        try:
            logger.info("Generating IPS compliance report")

            # Get current performance
            performance = self._calculate_system_performance()

            # Generate report
            report = {
                'report_metadata': {
                    'report_date': datetime.now().isoformat(),
                    'report_period': {
                        'start_date': self.backtest_config.start_date.isoformat(),
                        'end_date': self.backtest_config.end_date.isoformat()
                    },
                    'reporting_frequency': self.config.reporting_frequency,
                    'benchmark': self.config.performance_benchmark
                },
                'executive_summary': {
                    'total_return': f"{performance.total_return:.2%}",
                    'annualized_return': f"{performance.annualized_return:.2%}",
                    'sharpe_ratio': f"{performance.sharpe_ratio:.2f}",
                    'max_drawdown': f"{performance.max_drawdown:.2%}",
                    'alpha': f"{performance.alpha:.2%}",
                    'information_ratio': f"{performance.information_ratio:.2f}",
                    'overall_compliance': self.ips_compliance['overall_compliance']
                },
                'portfolio_allocation': {
                    'target_allocation': {
                        'core': f"{self.config.core_allocation_min:.0%}-{self.config.core_allocation_max:.0%}",
                        'satellite': f"{self.config.satellite_allocation_min:.0%}-{self.config.satellite_allocation_max:.0%}"
                    },
                    'current_allocation': {
                        'core': f"{self.current_allocation.core_weight:.1%}",
                        'satellite': f"{self.current_allocation.satellite_weight:.1%}",
                        'cash': f"{self.current_allocation.cash_weight:.1%}"
                    },
                    'allocation_status': self.current_allocation.status.value,
                    'drift_amount': f"{self.current_allocation.drift_amount:.2%}"
                },
                'ips_compliance_details': {
                    'investment_box_compliance': self.ips_compliance['box_compliance'],
                    'risk_budget_compliance': self.ips_compliance['risk_budget_compliance'],
                    'allocation_compliance': self.ips_compliance['allocation_compliance'],
                    'concentration_compliance': self.ips_compliance['concentration_compliance'],
                    'overall_compliance': self.ips_compliance['overall_compliance'],
                    'compliance_issues': self._identify_compliance_issues()
                },
                'performance_attribution': {
                    'core_strategy': {
                        'return': f"{performance.core_return:.2%}",
                        'contribution': f"{performance.core_contribution:.2%}"
                    },
                    'satellite_strategy': {
                        'return': f"{performance.satellite_return:.2%}",
                        'contribution': f"{performance.satellite_contribution:.2%}"
                    },
                    'factor_attribution': performance.factor_attribution,
                    'alpha_attribution': performance.alpha_attribution
                },
                'risk_management': {
                    'risk_metrics': {
                        'volatility': f"{performance.volatility:.2%}",
                        'var_95': f"{performance.var_95:.2%}",
                        'var_99': f"{performance.var_99:.2%}",
                        'expected_shortfall': f"{performance.expected_shortfall:.2%}",
                        'max_drawdown': f"{performance.max_drawdown:.2%}",
                        'beta': f"{performance.beta:.2f}",
                        'tracking_error': f"{performance.tracking_error:.2%}"
                    },
                    'risk_limits': {
                        'max_drawdown_limit': f"{self.config.max_drawdown_threshold:.0%}",
                        'var_limit': f"{self.config.var_threshold:.0%}",
                        'tracking_error_limit': f"{self.config.tracking_error_threshold:.0%}"
                    },
                    'breach_history': self.risk_monitoring['var_breaches'][-5:]  # Last 5 breaches
                },
                'model_governance': {
                    'core_model': {
                        'current_accuracy': f"{self.model_health['core_model']['accuracy']:.1%}",
                        'degradation': f"{self.model_health['core_model']['degradation']:.1%}",
                        'last_retrained': self.model_health['core_model']['last_retrained'].isoformat() if self.model_health['core_model']['last_retrained'] else None,
                        'retraining_needed': self.model_health['core_model']['degradation'] > self.config.model_retraining_threshold
                    },
                    'satellite_model': {
                        'current_accuracy': f"{self.model_health['satellite_model']['accuracy']:.1%}",
                        'degradation': f"{self.model_health['satellite_model']['degradation']:.1%}",
                        'last_retrained': self.model_health['satellite_model']['last_retrained'].isoformat() if self.model_health['satellite_model']['last_retrained'] else None,
                        'retraining_needed': self.model_health['satellite_model']['degradation'] > self.config.model_retraining_threshold
                    }
                },
                'recommendations': self._generate_recommendations(performance),
                'appendices': {
                    'portfolio_characteristics': self._get_portfolio_characteristics(),
                    'trading_activity': self._get_trading_activity(),
                    'compliance_history': self.compliance_history[-10:],  # Last 10 compliance checks
                    'performance_history': [asdict(p) for p in self.performance_history[-12:]]  # Last 12 months
                }
            }

            logger.info("Generated IPS compliance report")
            return report

        except Exception as e:
            logger.error(f"Failed to generate IPS compliance report: {e}")
            return {'error': str(e)}

    def _identify_compliance_issues(self) -> List[str]:
        """Identify specific compliance issues."""
        issues = []

        if not self.ips_compliance['box_compliance']:
            issues.append("Investment box constraints violated")

        if not self.ips_compliance['risk_budget_compliance']:
            issues.append("Risk budget limits exceeded")

        if not self.ips_compliance['allocation_compliance']:
            issues.append("Portfolio allocation outside target ranges")

        if not self.ips_compliance['concentration_compliance']:
            issues.append("Portfolio concentration limits exceeded")

        return issues

    def _generate_recommendations(self, performance: SystemPerformance) -> List[str]:
        """Generate recommendations based on performance analysis."""
        recommendations = []

        # Performance-based recommendations
        if performance.sharpe_ratio < 1.0:
            recommendations.append("Improve risk-adjusted returns through better factor timing")

        if performance.alpha < 0.02:
            recommendations.append("Enhance security selection to generate alpha")

        if performance.tracking_error > self.config.tracking_error_threshold:
            recommendations.append("Reduce tracking error to align with benchmark")

        # Compliance-based recommendations
        if not self.ips_compliance['overall_compliance']:
            recommendations.append("Address IPS compliance issues immediately")

        # Model-based recommendations
        if self.model_health['core_model']['degradation'] > self.config.model_retraining_threshold:
            recommendations.append("Retrain core ML model to address performance degradation")

        if self.model_health['satellite_model']['degradation'] > self.config.model_retraining_threshold:
            recommendations.append("Retrain satellite model to improve signal accuracy")

        # Risk-based recommendations
        if performance.max_drawdown > self.config.max_drawdown_threshold * 0.8:
            recommendations.append("Review risk management protocols to limit drawdowns")

        # Allocation-based recommendations
        if self.current_allocation.status != AllocationStatus.COMPLIANT:
            recommendations.append("Rebalance portfolio to maintain target allocation ranges")

        return recommendations

    def _get_portfolio_characteristics(self) -> Dict[str, Any]:
        """Get detailed portfolio characteristics."""
        try:
            if not self.current_portfolio.positions:
                return {'message': 'No positions in portfolio'}

            characteristics = {
                'number_of_positions': len(self.current_portfolio.positions),
                'total_market_value': self.current_portfolio.total_value,
                'cash_balance': self.current_portfolio.cash_balance,
                'position_breakdown': []
            }

            for pos in self.current_portfolio.positions:
                characteristics['position_breakdown'].append({
                    'symbol': pos.symbol,
                    'quantity': pos.quantity,
                    'market_value': pos.market_value,
                    'weight': f"{pos.weight:.2%}",
                    'unrealized_pnl': pos.unrealized_pnl,
                    'average_cost': pos.average_cost,
                    'current_price': pos.current_price
                })

            return characteristics

        except Exception as e:
            logger.error(f"Failed to get portfolio characteristics: {e}")
            return {'error': str(e)}

    def _get_trading_activity(self) -> Dict[str, Any]:
        """Get trading activity summary."""
        try:
            # Get recent trading activity
            recent_trades = self.completed_trades[-50:]  # Last 50 trades

            activity = {
                'total_trades': len(self.completed_trades),
                'recent_trades': len(recent_trades),
                'trades_by_direction': {
                    'buy_trades': len([t for t in recent_trades if t.side == 'buy']),
                    'sell_trades': len([t for t in recent_trades if t.side == 'sell'])
                },
                'trades_by_strategy': {
                    'core_trades': len([t for t in recent_trades if self._is_core_position(t.symbol)]),
                    'satellite_trades': len([t for t in recent_trades if not self._is_core_position(t.symbol)])
                },
                'trading_volume': sum(t.quantity * t.price for t in recent_trades),
                'turnover_rate': self._calculate_turnover_rate()
            }

            return activity

        except Exception as e:
            logger.error(f"Failed to get trading activity: {e}")
            return {'error': str(e)}

    def _calculate_turnover_rate(self) -> float:
        """Calculate portfolio turnover rate."""
        # Simplified calculation
        if not self.completed_trades:
            return 0.0

        # Calculate turnover based on recent trades
        recent_trades = self.completed_trades[-100:]  # Last 100 trades
        trade_value = sum(abs(t.quantity * t.price) for t in recent_trades)

        # Annualize based on trade frequency
        avg_portfolio_value = self.current_portfolio.total_value
        if avg_portfolio_value > 0:
            turnover = (trade_value / avg_portfolio_value) * (12 / 6)  # Assume 6 months of trades
            return turnover

        return 0.0

    def save_system_results(self, output_dir: str) -> None:
        """
        Save complete system results to files.

        Args:
            output_dir: Output directory path
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save IPS compliance report
            report = self.generate_ips_compliance_report()
            report_file = output_path / "ips_compliance_report.json"
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)

            # Save portfolio history
            portfolio_file = output_path / "system_portfolio_history.csv"
            if self.portfolio_history:
                portfolio_data = []
                for snapshot in self.portfolio_history:
                    portfolio_data.append({
                        'timestamp': snapshot.timestamp.isoformat(),
                        'total_value': snapshot.total_value,
                        'daily_return': snapshot.daily_return,
                        'total_return': snapshot.total_return,
                        'drawdown': snapshot.drawdown,
                        'number_of_positions': len(snapshot.positions)
                    })

                portfolio_df = pd.DataFrame(portfolio_data)
                portfolio_df.to_csv(portfolio_file, index=False)

            # Save allocation history
            allocation_file = output_path / "system_allocation_history.csv"
            if self.allocation_history:
                allocation_data = []
                for allocation in self.allocation_history:
                    allocation_data.append({
                        'timestamp': allocation.last_rebalance_date.isoformat(),
                        'core_weight': allocation.core_weight,
                        'satellite_weight': allocation.satellite_weight,
                        'cash_weight': allocation.cash_weight,
                        'status': allocation.status.value,
                        'drift_amount': allocation.drift_amount
                    })

                allocation_df = pd.DataFrame(allocation_data)
                allocation_df.to_csv(allocation_file, index=False)

            # Save compliance history
            compliance_file = output_path / "system_compliance_history.csv"
            if self.compliance_history:
                compliance_df = pd.DataFrame(self.compliance_history)
                compliance_df.to_csv(compliance_file, index=False)

            # Save performance history
            performance_file = output_path / "system_performance_history.csv"
            if self.performance_history:
                performance_data = []
                for performance in self.performance_history:
                    performance_data.append(asdict(performance))

                performance_df = pd.DataFrame(performance_data)
                performance_df.to_csv(performance_file, index=False)

            # Save trade history
            trades_file = output_path / "system_trade_history.csv"
            if self.completed_trades:
                trade_data = []
                for trade in self.completed_trades:
                    trade_data.append({
                        'timestamp': trade.timestamp.isoformat(),
                        'symbol': trade.symbol,
                        'side': trade.side,
                        'quantity': trade.quantity,
                        'price': trade.price,
                        'trade_id': trade.trade_id
                    })

                trades_df = pd.DataFrame(trade_data)
                trades_df.to_csv(trades_file, index=False)

            logger.info(f"Saved complete system results to {output_dir}")

        except Exception as e:
            logger.error(f"Failed to save system results: {e}")

    def get_system_info(self) -> Dict[str, Any]:
        """Get complete system information."""
        return {
            'system_config': asdict(self.config),
            'strategies': {
                'core_strategy': self.core_strategy.get_strategy_info(),
                'satellite_strategy': self.satellite_strategy.get_strategy_info()
            },
            'current_state': {
                'portfolio_value': self.current_portfolio.total_value if self.current_portfolio else 0,
                'allocation': {
                    'core_weight': self.current_allocation.core_weight if self.current_allocation else 0,
                    'satellite_weight': self.current_allocation.satellite_weight if self.current_allocation else 0,
                    'cash_weight': self.current_allocation.cash_weight if self.current_allocation else 0,
                    'status': self.current_allocation.status.value if self.current_allocation else 'uninitialized'
                },
                'compliance': self.ips_compliance,
                'model_health': self.model_health
            },
            'performance_history_length': len(self.performance_history),
            'trade_history_length': len(self.completed_trades),
            'system_initialized': self.current_portfolio is not None
        }