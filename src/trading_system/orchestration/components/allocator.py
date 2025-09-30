"""
Capital Allocator - Strategy Capital Management

Manages capital allocation between different strategies, enforces allocation
constraints, and tracks allocation drift and rebalancing needs.

Extracted from SystemOrchestrator to follow Single Responsibility Principle.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from ...types.data_types import TradingSignal
from ...types.portfolio import Portfolio, Position
from ...types.enums import AssetClass

logger = logging.getLogger(__name__)


class AllocationStatus(Enum):
    """Portfolio allocation status."""
    COMPLIANT = "compliant"
    OVER_ALLOCATED = "over_allocated"
    UNDER_ALLOCATED = "under_allocated"
    REBALANCE_REQUIRED = "rebalance_required"
    DRAIN_EXCEEDED = "drain_exceeded"


@dataclass
class AllocationTarget:
    """Target allocation for a strategy."""
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
    """Configuration for capital allocator."""
    # Core-Satellite allocation
    core_target_weight: float = 0.75
    core_min_weight: float = 0.70
    core_max_weight: float = 0.80

    satellite_target_weight: float = 0.25
    satellite_min_weight: float = 0.20
    satellite_max_weight: float = 0.30

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


class CapitalAllocator:
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
        self.config = config

        # Current allocation state
        self.current_allocations: Dict[str, AllocationTarget] = {}
        self.allocation_history: List[Dict[str, Any]] = []

        # Rebalancing tracking
        self.last_rebalance_date: Optional[datetime] = None
        self.rebalance_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.allocation_stats = {
            'total_rebalances': 0,
            'allocation_changes': 0,
            'constraint_violations': 0,
            'drift_events': 0
        }

        logger.info("Initialized CapitalAllocator")
        logger.info(f"Core allocation: {config.core_target_weight:.1%} ±{config.rebalance_threshold:.1%}")
        logger.info(f"Satellite allocation: {config.satellite_target_weight:.1%} ±{config.rebalance_threshold:.1%}")

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

            # Step 6: Record allocation decision
            self._record_allocation_decision(strategy_signals, constrained_targets, weighted_signals)

            logger.info(f"Allocation completed: {sum(len(s) for s in weighted_signals.values())} weighted signals")
            return weighted_signals

        except Exception as e:
            logger.error(f"Capital allocation failed: {e}")
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

        # Create allocation targets for current state
        for strategy_name, value in strategy_values.items():
            weight = value / total_capital

            if 'core' in strategy_name.lower():
                self.current_allocations[strategy_name] = AllocationTarget(
                    strategy_name=strategy_name,
                    target_weight=self.config.core_target_weight,
                    min_weight=self.config.core_min_weight,
                    max_weight=self.config.core_max_weight,
                    current_weight=weight,
                    current_value=value
                )
            elif 'satellite' in strategy_name.lower():
                self.current_allocations[strategy_name] = AllocationTarget(
                    strategy_name=strategy_name,
                    target_weight=self.config.satellite_target_weight,
                    min_weight=self.config.satellite_min_weight,
                    max_weight=self.config.satellite_max_weight,
                    current_weight=weight,
                    current_value=value
                )

    def _calculate_allocation_targets(self, strategy_signals: Dict[str, List[TradingSignal]],
                                    total_capital: float) -> Dict[str, AllocationTarget]:
        """Calculate optimal allocation targets based on signals."""
        targets = {}

        for strategy_name, signals in strategy_signals.items():
            if not signals:
                continue

            # Calculate signal strength for this strategy
            total_strength = sum(signal.strength for signal in signals)
            avg_strength = total_strength / len(signals) if signals else 0

            # Determine strategy type and target allocation
            if 'core' in strategy_name.lower():
                target_weight = self.config.core_target_weight
                min_weight = self.config.core_min_weight
                max_weight = self.config.core_max_weight
            elif 'satellite' in strategy_name.lower():
                target_weight = self.config.satellite_target_weight
                min_weight = self.config.satellite_min_weight
                max_weight = self.config.satellite_max_weight
            else:
                # Default to equal allocation
                target_weight = 0.5
                min_weight = 0.3
                max_weight = 0.7

            # Adjust target based on signal strength
            if avg_strength < self.config.min_signal_strength_for_allocation:
                target_weight *= 0.5  # Reduce allocation for weak signals

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
            'stats': self.allocation_stats.copy(),
            'config': {
                'core_target': self.config.core_target_weight,
                'satellite_target': self.config.satellite_target_weight,
                'rebalance_threshold': self.config.rebalance_threshold
            }
        }

    def execute_rebalance(self, portfolio: Portfolio) -> List[str]:
        """Execute rebalancing and return list of actions taken."""
        actions = []

        # This would typically generate trades to rebalance
        # For now, just update the rebalance timestamp
        self.last_rebalance_date = datetime.now()
        self.allocation_stats['total_rebalances'] += 1

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
        issues = []

        # Check allocation weights sum
        total_target = self.config.core_target_weight + self.config.satellite_target_weight
        if total_target > 0.95:  # Allow for cash buffer
            issues.append(f"Target allocations sum to {total_target:.1%}, should be ≤ 95%")

        # Check bounds logic
        if not (self.config.core_min_weight <= self.config.core_target_weight <= self.config.core_max_weight):
            issues.append("Core allocation bounds are inconsistent")

        if not (self.config.satellite_min_weight <= self.config.satellite_target_weight <= self.config.satellite_max_weight):
            issues.append("Satellite allocation bounds are inconsistent")

        # Check thresholds
        if self.config.rebalance_threshold <= 0 or self.config.rebalance_threshold > 0.5:
            issues.append("rebalance_threshold must be between 0 and 50%")

        if self.config.max_single_position_weight <= 0 or self.config.max_single_position_weight > 1:
            issues.append("max_single_position_weight must be between 0 and 100%")

        return len(issues) == 0, issues