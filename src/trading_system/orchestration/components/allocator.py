"""
Capital Allocator - Strategy Capital Management

Manages capital allocation between different strategies, enforces allocation
constraints, and tracks allocation drift and rebalancing needs.

"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ...types.data_types import TradingSignal
from ...types.portfolio import Portfolio, Position
from ...types.enums import AssetClass
from ..utils.performance_tracker import ComponentPerformanceTrackerMixin
from ..utils.config_validator import ComponentConfigValidator

logger = logging.getLogger(__name__)


class AllocationStatus(Enum):
    """Portfolio allocation status."""
    COMPLIANT = "compliant"
    OVER_ALLOCATED = "over_allocated"
    UNDER_ALLOCATED = "under_allocated"
    REBALANCE_REQUIRED = "rebalance_required"
    DRAIN_EXCEEDED = "drain_exceeded"


@dataclass
class StrategyAllocation:
    """Allocation configuration for a single strategy."""
    strategy_name: str
    target_weight: float
    min_weight: float
    max_weight: float
    priority: int = 1  # Lower number = higher priority (1 is highest)
    
    def __post_init__(self):
        """Validate allocation parameters."""
        if not 0 <= self.target_weight <= 1:
            raise ValueError(f"target_weight must be between 0 and 1, got {self.target_weight}")
        if not 0 <= self.min_weight <= self.max_weight:
            raise ValueError(f"min_weight must be <= max_weight")
        if not self.min_weight <= self.target_weight <= self.max_weight:
            raise ValueError(f"target_weight must be between min_weight and max_weight")


@dataclass
class AllocationTarget:
    """Current allocation state for a strategy."""
    strategy_name: str
    target_weight: float
    min_weight: float
    max_weight: float
    current_weight: float = 0.0
    current_value: float = 0.0

    @property
    def is_within_bounds(self) -> bool:
        """Check if current allocation is within bounds."""
        return self.min_weight <= self.current_weight <= self.max_weight

    @property
    def drift_amount(self) -> float:
        """Calculate drift from target."""
        return abs(self.current_weight - self.target_weight)


@dataclass
class AllocationConfig:
    """
    Configuration for capital allocator - supports arbitrary number of strategies.
    
    This replaces the old hardcoded core/satellite configuration with a flexible
    list-based approach following SOLID and YAGNI principles.
    """
    # Strategy allocations - can be 1, 2, or more strategies
    strategy_allocations: List[StrategyAllocation]
    
    # Rebalancing parameters
    rebalance_threshold: float = 0.05  # 5% drift triggers rebalance
    max_drift_before_rebalance: float = 0.10  # 10% drift forces rebalance
    min_rebalance_interval_days: int = 30  # Minimum days between rebalances

    # Allocation constraints
    max_single_position_weight: float = 0.15
    min_signal_strength_for_allocation: float = 0.1
    cash_buffer_weight: float = 0.05  # Keep 5% cash

    # Risk management
    max_daily_allocation_change: float = 0.20  # Max 20% change per day
    volatility_adjustment_factor: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.strategy_allocations:
            raise ValueError("At least one strategy allocation must be provided")
        
        # Validate total target weight
        total_target = sum(s.target_weight for s in self.strategy_allocations)
        if total_target > 1.0:
            raise ValueError(
                f"Total target weight {total_target:.2%} exceeds 100%. "
                f"Sum of all strategy target weights must be <= 1.0"
            )
        
        # Check for duplicate strategy names
        names = [s.strategy_name for s in self.strategy_allocations]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate strategy names found in allocations")
    
    def get_allocation_for_strategy(self, strategy_name: str) -> Optional[StrategyAllocation]:
        """Get allocation configuration for a specific strategy."""
        for alloc in self.strategy_allocations:
            if alloc.strategy_name == strategy_name:
                return alloc
        return None
    
    @classmethod
    def create_core_satellite(cls, 
                             core_target: float = 0.75,
                             satellite_target: float = 0.25,
                             core_name: str = "core",
                             satellite_name: str = "satellite") -> 'AllocationConfig':
        """
        Factory method to create traditional core-satellite configuration.
        This provides backward compatibility with the old hardcoded approach.
        """
        return cls(
            strategy_allocations=[
                StrategyAllocation(
                    strategy_name=core_name,
                    target_weight=core_target,
                    min_weight=max(0.0, core_target - 0.05),
                    max_weight=min(1.0, core_target + 0.05),
                    priority=1
                ),
                StrategyAllocation(
                    strategy_name=satellite_name,
                    target_weight=satellite_target,
                    min_weight=max(0.0, satellite_target - 0.05),
                    max_weight=min(1.0, satellite_target + 0.05),
                    priority=2
                )
            ]
        )


class CapitalAllocator(ComponentPerformanceTrackerMixin):
    """
    Manages capital allocation between strategies.

    Responsibilities:
    - Calculate optimal allocation based on strategy signals
    - Enforce allocation constraints and limits
    - Monitor allocation drift
    - Generate rebalancing recommendations
    - Scale signals according to available capital
    """

    def __init__(self, config: AllocationConfig):
        """
        Initialize capital allocator.

        Args:
            config: Allocation configuration
        """
        super().__init__()
        self.config = config

        # Current allocation state
        self.current_allocations: Dict[str, AllocationTarget] = {}
        self.allocation_history: List[Dict[str, Any]] = []

        # Rebalancing tracking
        self.last_rebalance_date: Optional[datetime] = None
        self.rebalance_history: List[Dict[str, Any]] = []

        # Performance tracking is now handled by ComponentPerformanceTrackerMixin

        logger.info("Initialized CapitalAllocator")
        logger.info(f"Managing {len(config.strategy_allocations)} strategies:")
        for alloc in config.strategy_allocations:
            logger.info(
                f"  - {alloc.strategy_name}: target={alloc.target_weight:.1%}, "
                f"range=[{alloc.min_weight:.1%}, {alloc.max_weight:.1%}], "
                f"priority={alloc.priority}"
            )

    def allocate(self, strategy_signals: Dict[str, List[TradingSignal]],
                 portfolio: Portfolio, total_capital: float) -> Dict[str, List[TradingSignal]]:
        """
        Allocate capital to strategies based on their signals.

        Args:
            strategy_signals: Dictionary of signals per strategy
            portfolio: Current portfolio state
            total_capital: Total available capital

        Returns:
            Weighted signals per strategy
        """
        operation_id = self.track_operation("allocate_capital", {"strategy_count": len(strategy_signals)})
        
        try:
            logger.info(f"Allocating capital for {len(strategy_signals)} strategies")

            # Step 1: Calculate current allocation
            self._update_current_allocation(portfolio, total_capital)

            # Step 2: Determine allocation targets
            allocation_targets = self._calculate_allocation_targets(strategy_signals, total_capital)

            # Step 3: Apply allocation constraints
            constrained_targets = self._apply_allocation_constraints(allocation_targets, total_capital)

            # Step 4: Scale signals according to allocation
            weighted_signals = self._scale_signals_by_allocation(strategy_signals, constrained_targets)

            # Step 5: Check for rebalancing needs
            rebalance_needed = self._check_rebalancing_needs(constrained_targets)
            if rebalance_needed:
                self.track_counter("rebalance_needed", 1)

            # Step 6: Record allocation decision
            self._record_allocation_decision(strategy_signals, constrained_targets, weighted_signals)

            logger.info(f"Allocation completed: {sum(len(s) for s in weighted_signals.values())} weighted signals")
            self.end_operation(operation_id, success=True, metadata={"weighted_signals": sum(len(s) for s in weighted_signals.values())})
            return weighted_signals

        except Exception as e:
            logger.error(f"Capital allocation failed: {e}")
            self.end_operation(operation_id, success=False, metadata={"error": str(e)})
            return {}

    def _update_current_allocation(self, portfolio: Portfolio, total_capital: float) -> None:
        """Update current allocation based on portfolio state."""
        # Reset current allocations
        self.current_allocations.clear()

        if total_capital == 0:
            return

        # Group positions by strategy (assuming strategy name is stored in position metadata)
        strategy_values: Dict[str, float] = {}

        for position in portfolio.positions.values():
            strategy_name = getattr(position, 'strategy_name', 'unknown')
            position_value = position.quantity * position.current_price

            if strategy_name not in strategy_values:
                strategy_values[strategy_name] = 0
            strategy_values[strategy_name] += position_value

        # Create allocation targets for current state using configuration
        for strategy_name, value in strategy_values.items():
            weight = value / total_capital
            
            # Find strategy configuration
            strategy_config = self.config.get_allocation_for_strategy(strategy_name)
            
            if strategy_config:
                self.current_allocations[strategy_name] = AllocationTarget(
                    strategy_name=strategy_name,
                    target_weight=strategy_config.target_weight,
                    min_weight=strategy_config.min_weight,
                    max_weight=strategy_config.max_weight,
                    current_weight=weight,
                    current_value=value
                )
            else:
                # Strategy not in configuration - log warning
                logger.warning(
                    f"Strategy '{strategy_name}' has positions but no allocation config. "
                    f"Current value: ${value:,.2f} ({weight:.1%})"
                )

    def _calculate_allocation_targets(self, strategy_signals: Dict[str, List[TradingSignal]],
                                    total_capital: float) -> Dict[str, AllocationTarget]:
        """Calculate optimal allocation targets based on signals and configuration."""
        targets = {}

        for strategy_name, signals in strategy_signals.items():
            # Find strategy configuration
            strategy_config = self.config.get_allocation_for_strategy(strategy_name)
            
            if not strategy_config:
                logger.warning(
                    f"Strategy '{strategy_name}' has signals but no allocation config. "
                    f"Signals will be ignored."
                )
                continue
            
            if not signals:
                # No signals for this strategy - use base config
                targets[strategy_name] = AllocationTarget(
                    strategy_name=strategy_name,
                    target_weight=strategy_config.target_weight,
                    min_weight=strategy_config.min_weight,
                    max_weight=strategy_config.max_weight
                )
                continue

            # Calculate signal strength for this strategy
            total_strength = sum(signal.strength for signal in signals)
            avg_strength = total_strength / len(signals) if signals else 0

            # Start with configured target
            target_weight = strategy_config.target_weight
            min_weight = strategy_config.min_weight
            max_weight = strategy_config.max_weight

            # Adjust target based on signal strength (optional dynamic allocation)
            if avg_strength < self.config.min_signal_strength_for_allocation:
                target_weight *= 0.5  # Reduce allocation for weak signals
                logger.debug(
                    f"Reducing allocation for '{strategy_name}' due to weak signals "
                    f"(avg strength: {avg_strength:.2f})"
                )

            targets[strategy_name] = AllocationTarget(
                strategy_name=strategy_name,
                target_weight=target_weight,
                min_weight=min_weight,
                max_weight=max_weight
            )

        return targets

    def _apply_allocation_constraints(self, targets: Dict[str, AllocationTarget],
                                   total_capital: float) -> Dict[str, AllocationTarget]:
        """Apply allocation constraints and rebalance targets."""
        # Calculate total target weight
        total_target_weight = sum(target.target_weight for target in targets.values())

        if total_target_weight > 1.0 - self.config.cash_buffer_weight:
            # Scale down allocations
            scale_factor = (1.0 - self.config.cash_buffer_weight) / total_target_weight
            for target in targets.values():
                target.target_weight *= scale_factor

        # Ensure each target is within bounds
        for target in targets.values():
            target.target_weight = max(target.min_weight, min(target.max_weight, target.target_weight))

        return targets

    def _scale_signals_by_allocation(self, strategy_signals: Dict[str, List[TradingSignal]],
                                   targets: Dict[str, AllocationTarget]) -> Dict[str, List[TradingSignal]]:
        """Scale signals according to allocation targets."""
        weighted_signals = {}

        for strategy_name, signals in strategy_signals.items():
            if strategy_name not in targets or not signals:
                weighted_signals[strategy_name] = signals.copy()
                continue

            target = targets[strategy_name]
            scaling_factor = target.target_weight

            # Scale signal strengths
            scaled_signals = []
            for signal in signals:
                scaled_signal = TradingSignal(
                    symbol=signal.symbol,
                    signal_type=signal.signal_type,
                    strength=signal.strength * scaling_factor,
                    timestamp=signal.timestamp,
                    metadata={**signal.metadata, 'allocation_weight': scaling_factor}
                )
                scaled_signals.append(scaled_signal)

            weighted_signals[strategy_name] = scaled_signals

        return weighted_signals

    def _check_rebalancing_needs(self, targets: Dict[str, AllocationTarget]) -> bool:
        """Check if rebalancing is needed based on allocation drift."""
        if not self.last_rebalance_date:
            return True

        # Check minimum time interval
        days_since_rebalance = (datetime.now() - self.last_rebalance_date).days
        if days_since_rebalance < self.config.min_rebalance_interval_days:
            return False

        # Check drift threshold
        max_drift = 0.0
        for strategy_name, target in targets.items():
            if strategy_name in self.current_allocations:
                current = self.current_allocations[strategy_name]
                drift = abs(current.current_weight - target.target_weight)
                max_drift = max(max_drift, drift)

        return max_drift >= self.config.rebalance_threshold

    def _record_allocation_decision(self, strategy_signals: Dict[str, List[TradingSignal]],
                                  targets: Dict[str, AllocationTarget],
                                  weighted_signals: Dict[str, List[TradingSignal]]) -> None:
        """Record allocation decision for tracking."""
        record = {
            'timestamp': datetime.now(),
            'strategies': list(strategy_signals.keys()),
            'total_signals': sum(len(signals) for signals in strategy_signals.values()),
            'total_weighted_signals': sum(len(signals) for signals in weighted_signals.values()),
            'targets': {
                name: {
                    'target_weight': target.target_weight,
                    'min_weight': target.min_weight,
                    'max_weight': target.max_weight
                } for name, target in targets.items()
            }
        }

        self.allocation_history.append(record)

        # Keep only last 100 records
        if len(self.allocation_history) > 100:
            self.allocation_history = self.allocation_history[-100:]

    def get_allocation_status(self) -> Dict[str, Any]:
        """Get current allocation status."""
        return {
            'current_allocations': {
                name: {
                    'current_weight': target.current_weight,
                    'target_weight': target.target_weight,
                    'drift': target.drift_amount,
                    'within_bounds': target.is_within_bounds,
                    'current_value': target.current_value
                } for name, target in self.current_allocations.items()
            },
            'rebalance_status': {
                'last_rebalance': self.last_rebalance_date,
                'needs_rebalance': self._check_rebalancing_needs(self.current_allocations),
                'days_since_rebalance': (datetime.now() - self.last_rebalance_date).days if self.last_rebalance_date else None
            },
            'stats': self.get_performance_stats(),
            'config': {
                'strategies': {
                    alloc.strategy_name: {
                        'target_weight': alloc.target_weight,
                        'min_weight': alloc.min_weight,
                        'max_weight': alloc.max_weight,
                        'priority': alloc.priority
                    } for alloc in self.config.strategy_allocations
                },
                'rebalance_threshold': self.config.rebalance_threshold,
                'cash_buffer': self.config.cash_buffer_weight
            }
        }

    def execute_rebalance(self, portfolio: Portfolio) -> List[str]:
        """Execute rebalancing and return list of actions taken."""
        actions = []

        # This would typically generate trades to rebalance
        # For now, just update the rebalance timestamp
        self.last_rebalance_date = datetime.now()
        self.track_counter("total_rebalances", 1)

        actions.append(f"Rebalanced portfolio on {self.last_rebalance_date}")

        self.rebalance_history.append({
            'timestamp': self.last_rebalance_date,
            'actions': actions,
            'portfolio_value': portfolio.total_value
        })

        logger.info(f"Executed portfolio rebalancing: {len(actions)} actions")
        return actions

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate allocator configuration."""
        config_dict = {
            'strategy_allocations': [
                {
                    'strategy_name': alloc.strategy_name,
                    'target_weight': alloc.target_weight,
                    'min_weight': alloc.min_weight,
                    'max_weight': alloc.max_weight,
                    'priority': alloc.priority
                }
                for alloc in self.config.strategy_allocations
            ],
            'rebalance_threshold': self.config.rebalance_threshold,
            'max_single_position_weight': self.config.max_single_position_weight,
            'cash_buffer_weight': self.config.cash_buffer_weight
        }
        
        return ComponentConfigValidator.validate_allocator_config(config_dict)