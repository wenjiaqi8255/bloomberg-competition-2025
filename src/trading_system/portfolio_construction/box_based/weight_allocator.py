"""
Weight allocation strategies for within-box distribution.

Provides different strategies for allocating a box's total weight
among the selected stocks within that box.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from src.trading_system.portfolio_construction.interface.interfaces import IWithinBoxAllocator
from src.trading_system.portfolio_construction.models.exceptions import WeightAllocationError
from src.trading_system.optimization.optimizer import PortfolioOptimizer

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
                signals: pd.Series,
                price_data: Optional[Dict[str, pd.DataFrame]] = None,
                date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Perform mini-optimization within the box.

        Args:
            stocks: List of selected stocks in the box
            total_weight: Total weight allocated to this box
            signals: Signal strengths for all stocks
            price_data: Optional price data (unused in optimized allocation)
            date: Optional date (unused in optimized allocation)

        Returns:
            Dictionary mapping stock symbols to optimized weights
        """
        # For now, fall back to signal proportional allocation
        # TODO: Implement box-level optimization
        logger.info("Box-level optimization not yet implemented, using signal proportional allocation")
        return self.fallback_allocator.allocate(stocks, total_weight, signals, price_data, date)


class MeanVarianceAllocator(IWithinBoxAllocator):
    """
    Allocates weights using mean-variance optimization within a box.
    
    Uses Markowitz mean-variance optimization to optimize risk-return
    tradeoff for stocks within a box.
    """
    
    def __init__(self, config: Dict[str, Any] = None, factor_data_provider=None):
        """
        Initialize mean-variance allocator.
        
        Args:
            config: Configuration dictionary with parameters:
                - risk_aversion: Risk aversion parameter (default: 2.0)
                - lookback_days: Lookback period for covariance (default: 252)
                - covariance_method: 'simple', 'ledoit_wolf', or 'factor_model' (default: 'ledoit_wolf')
                - min_regression_obs: Minimum observations for factor model (default: 24)
            factor_data_provider: Optional factor data provider for factor model covariance
        """
        self.config = config or {}
        self.risk_aversion = self.config.get('risk_aversion', 2.0)
        self.lookback_days = self.config.get('lookback_days', 252)
        self.covariance_method = self.config.get('covariance_method', 'ledoit_wolf')
        self.factor_data_provider = factor_data_provider
        
        # Initialize optimizer
        optimizer_config = {
            'method': 'mean_variance',
            'risk_aversion': self.risk_aversion,
            'enable_short_selling': False,
            'max_position_weight': None  # Will be applied at box level
        }
        self.optimizer = PortfolioOptimizer(optimizer_config)
        
        # Initialize covariance estimator
        if self.covariance_method == 'factor_model':
            if self.factor_data_provider is None:
                logger.warning("factor_model requires factor_data_provider, falling back to ledoit_wolf")
                from src.trading_system.utils.risk import LedoitWolfCovarianceEstimator
                self.covariance_estimator = LedoitWolfCovarianceEstimator(lookback_days=self.lookback_days)
            else:
                from src.trading_system.utils.risk import FactorModelCovarianceEstimator
                min_regression_obs = self.config.get('min_regression_obs', 24)
                self.covariance_estimator = FactorModelCovarianceEstimator(
                    factor_data_provider=self.factor_data_provider,
                    lookback_days=self.lookback_days,
                    min_regression_obs=min_regression_obs
                )
                logger.info(f"Using FactorModelCovarianceEstimator with {self.lookback_days} days lookback")
        elif self.covariance_method == 'ledoit_wolf':
            from src.trading_system.utils.risk import LedoitWolfCovarianceEstimator
            self.covariance_estimator = LedoitWolfCovarianceEstimator(lookback_days=self.lookback_days)
        else:
            from src.trading_system.utils.risk import SimpleCovarianceEstimator
            self.covariance_estimator = SimpleCovarianceEstimator(lookback_days=self.lookback_days)
        
        # Fallback allocator (default to equal weight)
        self.fallback_allocator = EqualWeightAllocator()
        
        logger.info(f"Initialized MeanVarianceAllocator with risk_aversion={self.risk_aversion}, "
                   f"lookback_days={self.lookback_days}, method={self.covariance_method}")
    
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
        available_stocks = [s for s in stocks if s in price_data and price_data[s] is not None]
        if len(available_stocks) < 2:
            logger.warning(f"Need at least 2 stocks with price data for mean-variance, "
                         f"got {len(available_stocks)}, falling back to equal weight")
            return self.fallback_allocator.allocate(stocks, total_weight, signals, price_data, date)
        
        try:
            # Estimate covariance matrix
            relevant_price_data = {s: price_data[s] for s in available_stocks}
            cov_matrix = self.covariance_estimator.estimate(relevant_price_data, date)
            
            if cov_matrix.empty or len(cov_matrix) < 2:
                logger.warning("Covariance estimation failed, falling back to equal weight")
                return self.fallback_allocator.allocate(stocks, total_weight, signals, price_data, date)
            
            # Use signals as expected returns
            expected_returns = pd.Series({s: signals.get(s, 0.0) for s in available_stocks})
            
            # Run optimization with no additional constraints (box-level constraints already applied)
            optimized_weights = self.optimizer.optimize(
                expected_returns=expected_returns,
                cov_matrix=cov_matrix,
                constraints=[]  # No box-level constraints needed here
            )
            
            # Scale to box's total weight
            scaled_weights = {stock: weight * total_weight 
                            for stock, weight in optimized_weights.items()}
            
            # Handle stocks not in optimized weights (shouldn't happen, but safety check)
            for stock in stocks:
                if stock not in scaled_weights:
                    scaled_weights[stock] = 0.0
            
            logger.debug(f"Mean-variance optimization allocated weights to {len(scaled_weights)} stocks")
            return scaled_weights
            
        except Exception as e:
            logger.error(f"Mean-variance optimization failed: {e}, falling back to equal weight")
            return self.fallback_allocator.allocate(stocks, total_weight, signals, price_data, date)


class WeightAllocatorFactory:
    """
    Factory for creating weight allocators based on configuration.

    Provides a clean interface for creating different allocation
    strategies while maintaining consistency.
    """

    @staticmethod
    def create_allocator(allocation_method: str, config: Dict[str, Any] = None,
                        factor_data_provider=None) -> IWithinBoxAllocator:
        """
        Create weight allocator based on method.

        Args:
            allocation_method: Allocation method name
            config: Optional configuration for the allocator
            factor_data_provider: Optional factor data provider for factor model covariance

        Returns:
            Configured weight allocator
        """
        config = config or {}

        if allocation_method == 'equal':
            return EqualWeightAllocator()
        elif allocation_method == 'signal_proportional':
            min_threshold = config.get('min_signal_threshold', 1e-6)
            return SignalProportionalAllocator(min_threshold)
        elif allocation_method == 'mean_variance':
            return MeanVarianceAllocator(config, factor_data_provider=factor_data_provider)
        elif allocation_method == 'optimized':
            return OptimizedAllocator(config)
        else:
            raise ValueError(f"Unknown allocation method: {allocation_method}")

    @staticmethod
    def get_available_methods() -> List[str]:
        """Get list of available allocation methods."""
        return ['equal', 'signal_proportional', 'mean_variance', 'optimized']

    @staticmethod
    def validate_method(allocation_method: str) -> bool:
        """Validate allocation method name."""
        return allocation_method in WeightAllocatorFactory.get_available_methods()