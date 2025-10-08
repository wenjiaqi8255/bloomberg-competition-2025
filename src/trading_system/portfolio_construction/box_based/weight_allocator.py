"""
Weight allocation strategies for within-box distribution.

Provides different strategies for allocating a box's total weight
among the selected stocks within that box.
"""

import logging
from typing import Dict, List, Any
import pandas as pd

from src.trading_system.portfolio_construction.interface.interfaces import IWithinBoxAllocator
from src.trading_system.portfolio_construction.models.exceptions import WeightAllocationError

logger = logging.getLogger(__name__)


class EqualWeightAllocator(IWithinBoxAllocator):
    """
    Allocates equal weights to all stocks within a box.

    Simple and robust allocation strategy that ensures
    equal representation of all selected stocks.
    """

    def allocate(self, stocks: List[str], total_weight: float,
                signals: pd.Series) -> Dict[str, float]:
        """
        Allocate equal weights to all stocks.

        Args:
            stocks: List of selected stocks in the box
            total_weight: Total weight allocated to this box
            signals: Signal strengths for all stocks (unused in equal allocation)

        Returns:
            Dictionary mapping stock symbols to equal weights
        """
        if not stocks:
            logger.warning("No stocks provided for equal weight allocation")
            return {}

        if total_weight <= 0:
            raise WeightAllocationError("Total weight must be positive", "within_box", str(total_weight))

        equal_weight = total_weight / len(stocks)
        weights = {stock: equal_weight for stock in stocks}

        logger.debug(f"Allocated equal weights to {len(stocks)} stocks: {equal_weight:.6f} each")
        return weights


class SignalProportionalAllocator(IWithinBoxAllocator):
    """
    Allocates weights proportional to signal strengths.

    Stocks with stronger signals receive larger allocations,
    while weaker signals receive smaller allocations.
    """

    def __init__(self, min_signal_threshold: float = 1e-6):
        """
        Initialize signal proportional allocator.

        Args:
            min_signal_threshold: Minimum signal threshold for allocation
        """
        self.min_signal_threshold = min_signal_threshold

    def allocate(self, stocks: List[str], total_weight: float,
                signals: pd.Series) -> Dict[str, float]:
        """
        Allocate weights proportional to signal strengths.

        Args:
            stocks: List of selected stocks in the box
            total_weight: Total weight allocated to this box
            signals: Signal strengths for all stocks

        Returns:
            Dictionary mapping stock symbols to proportional weights
        """
        if not stocks:
            logger.warning("No stocks provided for signal proportional allocation")
            return {}

        if total_weight <= 0:
            raise WeightAllocationError("Total weight must be positive", "within_box", str(total_weight))

        # Extract signals for stocks in this box
        stock_signals = {}
        for stock in stocks:
            signal = signals.get(stock, 0.0)
            stock_signals[stock] = signal

        # Filter out stocks with very low or negative signals
        valid_stocks = {
            stock: signal for stock, signal in stock_signals.items()
            if signal > self.min_signal_threshold
        }

        if not valid_stocks:
            logger.warning("No stocks with positive signals found, falling back to equal allocation")
            return EqualWeightAllocator().allocate(stocks, total_weight, signals)

        # Calculate total signal for normalization
        total_signal = sum(valid_stocks.values())
        if total_signal <= 0:
            logger.warning("Total signal is non-positive, falling back to equal allocation")
            return EqualWeightAllocator().allocate(stocks, total_weight, signals)

        # Allocate weights proportional to signals
        weights = {}
        for stock, signal in valid_stocks.items():
            weight = total_weight * (signal / total_signal)
            weights[stock] = weight

        # Allocate remaining weight to excluded stocks (if any)
        excluded_stocks = [s for s in stocks if s not in valid_stocks]
        if excluded_stocks:
            remaining_weight = total_weight - sum(weights.values())
            if remaining_weight > 0 and excluded_stocks:
                fallback_weight = remaining_weight / len(excluded_stocks)
                for stock in excluded_stocks:
                    weights[stock] = fallback_weight

        logger.debug(f"Allocated signal-proportional weights to {len(weights)} stocks "
                    f"(valid signals: {len(valid_stocks)}, excluded: {len(excluded_stocks)})")
        return weights


class OptimizedAllocator(IWithinBoxAllocator):
    """
    Performs mini-optimization within a box.

    Placeholder for future implementation of box-level optimization
    strategies that consider risk and return within the box.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize optimized allocator.

        Args:
            config: Configuration for optimization parameters
        """
        self.config = config or {}
        self.fallback_allocator = SignalProportionalAllocator()

    def allocate(self, stocks: List[str], total_weight: float,
                signals: pd.Series) -> Dict[str, float]:
        """
        Perform mini-optimization within the box.

        Args:
            stocks: List of selected stocks in the box
            total_weight: Total weight allocated to this box
            signals: Signal strengths for all stocks

        Returns:
            Dictionary mapping stock symbols to optimized weights
        """
        # For now, fall back to signal proportional allocation
        # TODO: Implement box-level optimization
        logger.info("Box-level optimization not yet implemented, using signal proportional allocation")
        return self.fallback_allocator.allocate(stocks, total_weight, signals)


class WeightAllocatorFactory:
    """
    Factory for creating weight allocators based on configuration.

    Provides a clean interface for creating different allocation
    strategies while maintaining consistency.
    """

    @staticmethod
    def create_allocator(allocation_method: str, config: Dict[str, Any] = None) -> IWithinBoxAllocator:
        """
        Create weight allocator based on method.

        Args:
            allocation_method: Allocation method name
            config: Optional configuration for the allocator

        Returns:
            Configured weight allocator
        """
        config = config or {}

        if allocation_method == 'equal':
            return EqualWeightAllocator()
        elif allocation_method == 'signal_proportional':
            min_threshold = config.get('min_signal_threshold', 1e-6)
            return SignalProportionalAllocator(min_threshold)
        elif allocation_method == 'optimized':
            return OptimizedAllocator(config)
        else:
            raise ValueError(f"Unknown allocation method: {allocation_method}")

    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available allocation methods."""
        return ['equal', 'signal_proportional', 'optimized']

    @staticmethod
    def validate_method(allocation_method: str) -> bool:
        """Validate allocation method name."""
        return allocation_method in WeightAllocatorFactory.get_available_methods()