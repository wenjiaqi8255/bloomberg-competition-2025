"""
Weight allocation strategies for within-box distribution.

Provides different strategies for allocating a box's total weight
among the selected stocks within that box.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from trading_system.portfolio_construction.interface.interfaces import IWithinBoxAllocator
from trading_system.portfolio_construction.models.exceptions import WeightAllocationError
from trading_system.optimization.optimizer import PortfolioOptimizer
from trading_system.utils.risk import CovarianceEstimator
from trading_system.portfolio_construction.utils.weight_utils import WeightUtils

logger = logging.getLogger(__name__)


class EqualWeightAllocator(IWithinBoxAllocator):
    """
    Allocates equal weights to all stocks within a box.

    Simple and robust allocation strategy that ensures
    equal representation of all selected stocks.
    """

    def allocate(self, stocks: List[str], total_weight: float,
                signals: pd.Series,
                price_data: Optional[Dict[str, pd.DataFrame]] = None,
                date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Allocate equal weights to all stocks.

        Args:
            stocks: List of selected stocks in the box
            total_weight: Total weight allocated to this box
            signals: Signal strengths for all stocks (unused in equal allocation)
            price_data: Optional price data (unused in equal allocation)
            date: Optional date (unused in equal allocation)

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
                signals: pd.Series,
                price_data: Optional[Dict[str, pd.DataFrame]] = None,
                date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Allocate weights proportional to signal strengths.

        Args:
            stocks: List of selected stocks in the box
            total_weight: Total weight allocated to this box
            signals: Signal strengths for all stocks
            price_data: Optional price data (unused in signal proportional allocation)
            date: Optional date (unused in signal proportional allocation)

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
            return EqualWeightAllocator().allocate(stocks, total_weight, signals, price_data, date)

        # Calculate total signal for normalization
        total_signal = sum(valid_stocks.values())
        if total_signal <= 0:
            logger.warning("Total signal is non-positive, falling back to equal allocation")
            return EqualWeightAllocator().allocate(stocks, total_weight, signals, price_data, date)

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


class MeanVarianceAllocator(IWithinBoxAllocator):
    """
    Allocates weights using mean-variance optimization within a box.
    
    Uses Markowitz mean-variance optimization to optimize risk-return
    tradeoff for stocks within a box.
    """
    
    def __init__(self, config: Dict[str, Any], covariance_estimator: CovarianceEstimator):
        """
        Initialize mean-variance allocator.
        
        Args:
            config: Configuration dictionary with parameters:
                - risk_aversion: Risk aversion parameter (default: 2.0)
                - lookback_days: Lookback period for covariance (default: 252)
                - covariance_method: 'simple', 'ledoit_wolf', or 'factor_model' (default: 'ledoit_wolf')
                - min_regression_obs: Minimum observations for factor model (default: 24)
            covariance_estimator: A CovarianceEstimator instance.
        """
        self.config = config or {}
        self.risk_aversion = self.config.get('risk_aversion', 2.0)
        self.lookback_days = self.config.get('lookback_days', 252)
        self.covariance_estimator = covariance_estimator
        
        # Initialize optimizer
        optimizer_config = {
            'method': 'mean_variance',
            'risk_aversion': self.risk_aversion,
            'enable_short_selling': False,
            'max_position_weight': None  # Will be applied at box level
        }
        self.optimizer = PortfolioOptimizer(optimizer_config)
        
        # Fallback allocator (default to equal weight)
        self.fallback_allocator = EqualWeightAllocator()
        
        logger.info(f"Initialized MeanVarianceAllocator with risk_aversion={self.risk_aversion}, "
                   f"lookback_days={self.lookback_days}, "
                   f"cache_enabled={hasattr(self.covariance_estimator, '_cache')}")
    
    def allocate(self, stocks: List[str], total_weight: float,
                signals: pd.Series,
                price_data: Optional[Dict[str, pd.DataFrame]] = None,
                date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Allocate weights using mean-variance optimization.
        
        Args:
            stocks: List of selected stocks in the box
            total_weight: Total weight allocated to this box
            signals: Signal strengths for all stocks (used as expected returns)
            price_data: Price data for covariance estimation
            date: Date for data filtering
            
        Returns:
            Dictionary mapping stock symbols to optimized weights
        """
        if not stocks:
            logger.warning("No stocks provided for mean-variance allocation")
            return {}
        
        if total_weight <= 0:
            raise WeightAllocationError("Total weight must be positive", "within_box", str(total_weight))
        
        # If no price data, fall back to equal weight
        if price_data is None or date is None:
            logger.warning("Mean-variance requires price_data and date, falling back to equal weight")
            return self.fallback_allocator.allocate(stocks, total_weight, signals, price_data, date)
        
        # Filter stocks to those with price data
        available_stocks = [s for s in stocks if s in signals.index and s in price_data and not price_data[s].empty]

        if not available_stocks:
            logger.warning(f"No valid data for any stocks in {stocks}, cannot allocate.")
            return {}

        # Estimate covariance matrix using the provided estimator
        cov_matrix = self.covariance_estimator.estimate(
            {s: price_data[s] for s in available_stocks}, date
        )

        if cov_matrix.empty or not all(s in cov_matrix.columns for s in available_stocks):
            logger.warning(f"Covariance matrix is empty or does not contain all stocks for box. "
                         f"Falling back to equal weight for stocks: {available_stocks}")
            return {stock: total_weight / len(available_stocks) for stock in available_stocks}
        
        # Align signals and covariance matrix
        aligned_signals = signals.loc[available_stocks]
        aligned_cov = cov_matrix.loc[available_stocks, available_stocks]

        # Perform optimization
        try:
            # Use an empty list for constraints as box-level optimization
            # typically does not have complex constraints.
            weights = self.optimizer.optimize(aligned_signals, aligned_cov, [])
            
            # Scale weights to the total allocated weight for the box
            scaled_weights = (weights * total_weight).to_dict()
            return scaled_weights
            
        except Exception as e:
            logger.error(f"Mean-variance allocation failed for stocks {available_stocks}: {e}. "
                         f"Falling back to equal weight.")
            return {stock: total_weight / len(available_stocks) for stock in available_stocks}


class WeightAllocatorFactory:
    """Factory for creating weight allocators."""

    @staticmethod
    def create_allocator(method: str, config: Dict[str, Any]) -> IWithinBoxAllocator:
        """
        Create a weight allocator based on the specified method.
        
        Args:
            method: Allocation method ('equal', 'signal_proportional', etc.)
            config: Configuration for the allocator
            
        Returns:
            An instance of a weight allocator.
        """
        if method == 'equal':
            return EqualWeightAllocator()
        elif method == 'signal_proportional':
            return SignalProportionalAllocator()
        else:
            raise ValueError(f"Unknown or unsupported allocation method for this factory: {method}")