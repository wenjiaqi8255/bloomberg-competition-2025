"""
Strategy Coordinator - Multi-Strategy Signal Management

Coordinates multiple trading strategies, merges conflicting signals, and applies
capacity constraints to ensure proper signal integration.

"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass

from ...types.signals import TradingSignal
from ...types.enums import SignalType
from ...strategies.base_strategy import BaseStrategy
from ...types.enums import AssetClass

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorConfig:
    """Configuration for strategy coordinator - supports flexible strategy prioritization."""
    max_signals_per_day: int = 50
    signal_conflict_resolution: str = "merge"  # "merge", "priority", "cancel"
    capacity_scaling: bool = True
    min_signal_strength: float = 0.01
    max_position_size: float = 0.15

    # Priority ordering for conflict resolution
    # Maps strategy name to priority (lower number = higher priority)
    # e.g., {"FF5_Core": 1, "ML_Satellite": 2, "Tech_Tactical": 3}
    strategy_priority: Dict[str, int] = None

    def __post_init__(self):
        # If no priorities specified, all strategies have equal priority
        if self.strategy_priority is None:
            self.strategy_priority = {}


class StrategyCoordinator:
    """
    Coordinates multiple strategies and manages signal integration.

    Responsibilities:
    - Collect signals from multiple strategies
    - Resolve signal conflicts
    - Apply capacity constraints
    - Filter weak signals
    """

    def __init__(self, strategies: List[BaseStrategy], config: CoordinatorConfig):
        """
        Initialize strategy coordinator.

        Args:
            strategies: List of trading strategies
            config: Coordinator configuration
        """
        self.strategies = strategies
        self.config = config

        # Signal history for debugging
        self.signal_history: List[Dict[str, Any]] = []

        # Performance tracking
        self.coordination_stats = {
            'total_signals_processed': 0,
            'conflicts_resolved': 0,
            'signals_filtered': 0,
            'capacity_rejections': 0
        }

        logger.info(f"Initialized StrategyCoordinator with {len(strategies)} strategies")
        logger.info(f"Conflict resolution method: {config.signal_conflict_resolution}")

    def coordinate(self, date: datetime) -> Dict[str, List[TradingSignal]]:
        """
        Coordinate strategies and return merged signals.

        Args:
            date: Date for signal generation

        Returns:
            Dictionary mapping strategy names to their signals
        """
        try:
            logger.info(f"Coordinating strategies for {date}")

            # Step 1: Collect signals from all strategies
            all_signals = self._collect_strategy_signals(date)

            # Step 2: Apply capacity constraints if enabled
            if self.config.capacity_scaling:
                all_signals = self._apply_capacity_constraints(all_signals, date)

            # Step 3: Filter weak signals
            all_signals = self._filter_weak_signals(all_signals)

            # Step 4: Group signals by strategy
            strategy_signals = self._group_signals_by_strategy(all_signals)

            # Step 5: Record coordination history
            self._record_coordination(date, all_signals, strategy_signals)

            logger.info(f"Coordination completed: {len(all_signals)} total signals")
            return strategy_signals

        except Exception as e:
            logger.error(f"Strategy coordination failed for {date}: {e}")
            return {}

    def _collect_strategy_signals(self, date: datetime) -> List[TradingSignal]:
        """Collect signals from all strategies."""
        all_signals = []

        for strategy in self.strategies:
            try:
                # Generate signals from strategy
                strategy_signals = strategy.generate_signals(date)

                # Add strategy metadata
                for signal in strategy_signals:
                    signal.strategy_name = strategy.name
                    signal.strategy_type = getattr(strategy, 'strategy_type', 'unknown')

                all_signals.extend(strategy_signals)
                logger.debug(f"Collected {len(strategy_signals)} signals from {strategy.name}")

            except Exception as e:
                logger.error(f"Failed to get signals from {strategy.name}: {e}")
                continue

        self.coordination_stats['total_signals_processed'] += len(all_signals)
        return all_signals

    def _apply_capacity_constraints(self, signals: List[TradingSignal], date: datetime) -> List[TradingSignal]:
        """Apply capacity constraints to limit signal volume."""
        if len(signals) <= self.config.max_signals_per_day:
            return signals

        # Sort signals by strength (descending)
        signals.sort(key=lambda s: s.strength, reverse=True)

        # Take top N signals
        constrained_signals = signals[:self.config.max_signals_per_day]

        rejected_count = len(signals) - len(constrained_signals)
        self.coordination_stats['capacity_rejections'] += rejected_count

        logger.warning(f"Capacity constraint: rejected {rejected_count} signals (kept {len(constrained_signals)})")
        return constrained_signals

    def _filter_weak_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Filter signals below minimum strength threshold."""
        filtered_signals = [
            signal for signal in signals
            if signal.strength >= self.config.min_signal_strength
        ]

        filtered_count = len(signals) - len(filtered_signals)
        self.coordination_stats['signals_filtered'] += filtered_count

        if filtered_count > 0:
            logger.debug(f"Filtered {filtered_count} weak signals")

        return filtered_signals

    def _group_signals_by_strategy(self, signals: List[TradingSignal]) -> Dict[str, List[TradingSignal]]:
        """Group signals by strategy name."""
        strategy_signals = {}

        for signal in signals:
            strategy_name = getattr(signal, 'strategy_name', 'unknown')
            if strategy_name not in strategy_signals:
                strategy_signals[strategy_name] = []
            strategy_signals[strategy_name].append(signal)

        return strategy_signals

    def _record_coordination(self, date: datetime, all_signals: List[TradingSignal],
                           strategy_signals: Dict[str, List[TradingSignal]]) -> None:
        """Record coordination history for debugging."""
        record = {
            'date': date,
            'total_signals': len(all_signals),
            'strategies': len(strategy_signals),
            'signals_per_strategy': {
                name: len(signals) for name, signals in strategy_signals.items()
            },
            'coordination_stats': self.coordination_stats.copy()
        }

        self.signal_history.append(record)

        # Keep only last 100 records
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]

    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get coordination statistics."""
        return {
            'stats': self.coordination_stats.copy(),
            'recent_history': self.signal_history[-10:],  # Last 10 records
            'strategies_count': len(self.strategies),
            'config': {
                'max_signals_per_day': self.config.max_signals_per_day,
                'signal_conflict_resolution': self.config.signal_conflict_resolution,
                'min_signal_strength': self.config.min_signal_strength
            }
        }

    def reset_stats(self) -> None:
        """Reset coordination statistics."""
        self.coordination_stats = {
            'total_signals_processed': 0,
            'conflicts_resolved': 0,
            'signals_filtered': 0,
            'capacity_rejections': 0
        }
        self.signal_history.clear()
        logger.info("Reset coordination statistics")

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """Validate coordinator configuration."""
        issues = []

        if self.config.max_signals_per_day <= 0:
            issues.append("max_signals_per_day must be positive")

        if self.config.min_signal_strength < 0 or self.config.min_signal_strength > 1:
            issues.append("min_signal_strength must be between 0 and 1")

        if self.config.max_position_size <= 0 or self.config.max_position_size > 1:
            issues.append("max_position_size must be between 0 and 1")

        if self.config.signal_conflict_resolution not in ["merge", "priority", "cancel"]:
            issues.append("signal_conflict_resolution must be one of: merge, priority, cancel")

        if not self.strategies:
            issues.append("At least one strategy must be provided")

        return len(issues) == 0, issues