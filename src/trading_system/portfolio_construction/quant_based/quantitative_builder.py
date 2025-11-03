"""
Quantitative portfolio construction strategy.

Wraps the existing quantitative optimization pipeline into the new
portfolio construction framework, maintaining backward compatibility
while providing optional box-aware enhancements.
"""

import logging
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

from src.trading_system.portfolio_construction.interface.interfaces import IPortfolioBuilder
from src.trading_system.portfolio_construction.models.types import PortfolioConstructionRequest, BoxConstructionResult
from src.trading_system.portfolio_construction.utils.adapters import ClassificationAdapter
from src.trading_system.portfolio_construction.models.exceptions import (
    PortfolioConstructionError, OptimizationError,
    ClassificationError, InvalidConfigError
)
from src.trading_system.optimization.optimizer import PortfolioOptimizer
from src.trading_system.data.stock_classifier import StockClassifier
from src.trading_system.utils.risk import (
    LedoitWolfCovarianceEstimator,
    SimpleCovarianceEstimator,
    FactorModelCovarianceEstimator
)
from src.trading_system.data.box_sampling_provider import BoxSamplingProvider

logger = logging.getLogger(__name__)


class QuantitativePortfolioBuilder(IPortfolioBuilder):
    """
    Quantitative portfolio construction strategy.

    Implements the traditional quantitative approach:
    1. Universe selection (with optional box-aware sampling)
    2. Signal filtering and dimensionality reduction
    3. Risk estimation (covariance matrix)
    4. Box classification for constraints
    5. Mathematical optimization
    6. Final portfolio weights

    Maintains compatibility with existing 7-stage pipeline while
    providing optional box-aware enhancements.
    """

    def __init__(self, config: Dict[str, Any], factor_data_provider=None):
        """
        Initialize quantitative portfolio builder.

        Args:
            config: Configuration for quantitative construction
            factor_data_provider: Optional factor data provider for factor model covariance
        """
        self.config = config
        self.factor_data_provider = factor_data_provider
        self._initialize_components()
        self._validate_configuration()

        logger.info(f"Initialized QuantitativePortfolioBuilder with optimizer: {self.optimizer.method}")

    def _initialize_components(self) -> None:
        """Initialize all sub-components."""
        # Core optimization parameters
        self.universe_size = self.config.get('universe_size', 100)
        self.enable_short_selling = self.config.get('enable_short_selling', False)

        # Portfolio optimizer - pass constraints
        optimizer_config = self.config.get('optimizer', {}).copy()
        max_position_weight = self.config.get('constraints', {}).get('max_position_weight')
        if max_position_weight is not None:
            optimizer_config['max_position_weight'] = max_position_weight
        self.optimizer = PortfolioOptimizer(optimizer_config)

        # Covariance estimator
        cov_config = self.config.get('covariance', {})
        lookback_days = cov_config.get('lookback_days', 252)
        covariance_method = cov_config.get('method', 'ledoit_wolf')
        
        if covariance_method == 'factor_model':
            if self.factor_data_provider is None:
                logger.warning("factor_model requires factor_data_provider, falling back to ledoit_wolf")
                self.covariance_estimator = LedoitWolfCovarianceEstimator(lookback_days=lookback_days)
            else:
                min_regression_obs = cov_config.get('min_regression_obs', 24)
                self.covariance_estimator = FactorModelCovarianceEstimator(
                    factor_data_provider=self.factor_data_provider,
                    lookback_days=lookback_days,
                    min_regression_obs=min_regression_obs
                )
                logger.info(f"Using FactorModelCovarianceEstimator with {lookback_days} days lookback")
        elif covariance_method == 'simple':
            self.covariance_estimator = SimpleCovarianceEstimator(lookback_days=lookback_days)
        else:
            # Default to Ledoit-Wolf
            self.covariance_estimator = LedoitWolfCovarianceEstimator(lookback_days=lookback_days)

        # Stock classifier for box constraints
        classifier_config = self.config.get('classifier', {})
        self.stock_classifier = StockClassifier(classifier_config)

        # Box constraints
        self.box_limits = self.config.get('box_limits', {})

        # Optional box-aware sampling
        self.use_box_sampling = self.config.get('use_box_sampling', False)
        if self.use_box_sampling:
            box_sampling_config = self.config.get('box_sampling', {})
            self.box_sampling_provider = BoxSamplingProvider(
                box_sampling_config,
                stock_classifier=self.stock_classifier
            )

        # Liquidity filtering
        self.min_history_days = self.config.get('min_history_days', 252)
        
        # Centralized constraints
        self.constraints = self.config.get('constraints', {})

    def _validate_configuration(self) -> None:
        """Validate builder configuration."""
        # Validate universe size
        if self.universe_size <= 0:
            raise InvalidConfigError("universe_size must be positive")

        # Validate optimizer configuration
        if not hasattr(self.optimizer, 'method'):
            raise InvalidConfigError("Invalid optimizer configuration")

        # Validate box limits if present
        self._validate_box_limits()

        logger.info("Quantitative builder configuration validation passed")

    def _validate_box_limits(self) -> None:
        """Validate box limits configuration."""
        for dimension, limits in self.box_limits.items():
            if not isinstance(limits, dict):
                raise InvalidConfigError(f"Box limits for {dimension} must be a dictionary")

            for box_name, limit in limits.items():
                if not isinstance(limit, (int, float)):
                    raise InvalidConfigError(f"Box limit for {dimension}:{box_name} must be numeric")
                if limit < 0 or limit > 1:
                    raise InvalidConfigError(f"Box limit for {dimension}:{box_name} must be between 0 and 1")

    def build_portfolio(self, request: PortfolioConstructionRequest) -> pd.Series:
        """
        Build portfolio using quantitative optimization.

        Args:
            request: Portfolio construction request

        Returns:
            Series of optimized portfolio weights
        """
        logger.info(f"Building quantitative portfolio for {request.date.date()}")

        try:
            # Step 1: Universe selection (with optional box sampling)
            universe = self._select_universe(request.universe, request.price_data,
                                           request.signals, request.date)
            logger.info(f"Selected universe: {len(universe)} stocks")

            # Step 2: Filter signals to selected universe
            filtered_signals = request.signals[request.signals.index.isin(universe)]
            logger.info(f"Filtered signals: {len(filtered_signals)} stocks")

            # Step 3: Dimensionality reduction (top N selection)
            top_signals = self._select_top_signals(filtered_signals)
            logger.info(f"Top signals: {len(top_signals)} stocks")

            # Step 4: Risk estimation
            cov_matrix = self._estimate_covariance_matrix(top_signals.index, request.price_data, request.date)
            logger.info(f"Covariance matrix: {cov_matrix.shape}")

            # Step 5: Box classification for constraints
            classifications = self._classify_stocks(top_signals.index, request.price_data, request.date)

            # Step 6: Build optimization constraints
            box_constraints = self._build_box_constraints(classifications)

            # Step 7: Portfolio optimization
            weights = self._optimize_portfolio(top_signals, cov_matrix, box_constraints)

            logger.info(f"Quantitative portfolio construction completed: {len(weights)} positions")
            return weights

        except Exception as e:
            logger.error(f"Quantitative portfolio construction failed: {e}")
            raise PortfolioConstructionError(f"Quantitative construction failed: {e}")

    def build_portfolio_with_result(self, request: PortfolioConstructionRequest) -> BoxConstructionResult:
        """
        Build portfolio with detailed construction information.

        Args:
            request: Portfolio construction request

        Returns:
            Detailed construction result
        """
        weights = self.build_portfolio(request)

        # Create basic result for quantitative method
        return BoxConstructionResult(
            weights=weights,
            box_coverage={},  # Not applicable for quantitative method
            selected_stocks={},
            target_weights={},
            construction_log=[f"Built portfolio using quantitative optimization ({self.optimizer.method})"]
        )

    def _select_universe(self, full_universe: List[str], price_data: Dict[str, pd.DataFrame],
                        signals: Dict[str, float], date: datetime) -> List[str]:
        """
        Select universe using configured method.

        Args:
            full_universe: Complete available universe
            price_data: Price data for all stocks
            signals: Signal strengths
            date: Selection date

        Returns:
            Selected universe
        """
        if self.use_box_sampling and hasattr(self, 'box_sampling_provider'):
            # Use box-aware sampling
            logger.info("Using box-aware universe sampling")
            return self.box_sampling_provider.sample_universe(
                full_universe, price_data, signals, date
            )
        else:
            # Use traditional liquidity filtering
            return self._filter_liquid_stocks(full_universe, price_data, date)

    def _filter_liquid_stocks(self, universe: List[str], price_data: Dict[str, pd.DataFrame],
                              date: datetime) -> List[str]:
        """
        Filter stocks based on data availability and liquidity.

        Args:
            universe: List of stocks to filter
            price_data: Price data for all stocks
            date: Filtering date

        Returns:
            Filtered list of liquid stocks
        """
        liquid_stocks = []

        for symbol in universe:
            if symbol not in price_data or price_data[symbol] is None:
                continue

            stock_data = price_data[symbol]
            data_up_to_date = stock_data[stock_data.index <= date]

            # Check minimum data history
            if len(data_up_to_date) >= self.min_history_days:
                liquid_stocks.append(symbol)

        logger.debug(f"Liquidity filter: {len(liquid_stocks)}/{len(universe)} stocks")
        return liquid_stocks

    def _select_top_signals(self, signals: pd.Series) -> pd.Series:
        """
        Select top N signals for optimization.

        Args:
            signals: Filtered signals for all stocks

        Returns:
            Top N signals
        """
        if len(signals) <= self.universe_size:
            return signals

        # Select by absolute signal value to include both strong long and short signals
        abs_signals = signals.abs()
        top_symbols = abs_signals.nlargest(self.universe_size).index
        top_signals = signals[top_symbols]

        logger.debug(f"Selected top {len(top_signals)} signals from {len(signals)}")
        return top_signals

    def _estimate_covariance_matrix(self, symbols: List[str], price_data: Dict[str, pd.DataFrame],
                                   date: datetime) -> pd.DataFrame:
        """
        Estimate covariance matrix for risk modeling.

        Args:
            symbols: Symbols to include in covariance matrix
            price_data: Price data for all symbols
            date: Estimation date

        Returns:
            Covariance matrix
        """
        try:
            # Prepare price data for covariance estimation
            relevant_price_data = {}
            for symbol in symbols:
                if symbol in price_data and price_data[symbol] is not None:
                    relevant_price_data[symbol] = price_data[symbol]

            if len(relevant_price_data) < 2:
                raise OptimizationError("Insufficient data for covariance estimation")

            # Estimate covariance matrix
            cov_matrix = self.covariance_estimator.estimate(relevant_price_data, date)

            if cov_matrix.empty:
                raise OptimizationError("Covariance estimation resulted in empty matrix")

            logger.debug(f"Covariance matrix estimated successfully: {cov_matrix.shape}")
            return cov_matrix

        except Exception as e:
            raise OptimizationError(f"Covariance estimation failed: {e}", "covariance_estimation")

    def _classify_stocks(self, symbols: List[str], price_data: Dict[str, pd.DataFrame],
                        date: datetime) -> Dict[str, Dict[str, str]]:
        """
        Classify stocks into boxes for constraint building.

        Args:
            symbols: Symbols to classify
            price_data: Price data for classification
            date: Classification date

        Returns:
            Dictionary mapping symbols to classification dictionaries
        """
        try:
            investment_boxes = self.stock_classifier.classify_stocks(symbols, price_data, as_of_date=date)
            classifications = ClassificationAdapter.convert_to_classification_dict(
                ClassificationAdapter.convert_investment_boxes_to_box_keys(investment_boxes)
            )

            logger.debug(f"Classified {len(classifications)} stocks into boxes")
            return classifications

        except Exception as e:
            raise ClassificationError("Stock classification for constraints failed", str(e))

    def _build_box_constraints(self, classifications: Dict[str, Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Build box constraints for optimization.

        Args:
            classifications: Stock classifications

        Returns:
            List of box constraint dictionaries
        """
        if not self.box_limits:
            logger.debug("No box limits configured")
            return []

        try:
            box_constraints = self.optimizer.build_box_constraints(classifications, self.box_limits)
            logger.debug(f"Built {len(box_constraints)} box constraints")
            return box_constraints

        except Exception as e:
            logger.error(f"Failed to build box constraints: {e}")
            return []

    def _optimize_portfolio(self, signals: pd.Series, cov_matrix: pd.DataFrame,
                          constraints: List[Dict[str, Any]]) -> pd.Series:
        """
        Perform portfolio optimization.

        Args:
            signals: Expected returns for all assets
            cov_matrix: Covariance matrix
            constraints: Optimization constraints

        Returns:
            Optimized portfolio weights
        """
        try:
            weights = self.optimizer.optimize(signals, cov_matrix, constraints)

            if weights.empty:
                raise OptimizationError("Optimization resulted in empty weights")

            # Validate optimization result
            total_weight = weights.sum()
            if abs(total_weight - 1.0) > 0.1:  # 10% tolerance
                logger.warning(f"Optimization weights sum to {total_weight:.4f}, normalizing")
                weights = weights / total_weight

            logger.debug(f"Portfolio optimization successful: {len(weights)} positions")
            return weights

        except Exception as e:
            raise OptimizationError(f"Portfolio optimization failed: {e}",
                                  self.optimizer.method, constraints)

    def get_method_name(self) -> str:
        """Get the method name."""
        return f"Quantitative({self.optimizer.method})"

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate the configuration."""
        try:
            # Create a temporary builder to validate config
            temp_builder = QuantitativePortfolioBuilder(config)
            return True
        except Exception as e:
            logger.error(f"Quantitative configuration validation failed: {e}")
            return False

    def get_construction_info(self) -> Dict[str, Any]:
        """Get detailed information about the construction method."""
        return {
            'method_name': self.get_method_name(),
            'method_type': self.__class__.__name__,
            'description': "Traditional quantitative optimization with optional box-aware sampling",
            'configuration': {
                'universe_size': self.universe_size,
                'optimizer_method': self.optimizer.method,
                'enable_short_selling': self.enable_short_selling,
                'use_box_sampling': self.use_box_sampling,
                'min_history_days': self.min_history_days,
                'box_limits': self.box_limits
            },
            'components': {
                'optimizer': self.optimizer.__class__.__name__,
                'covariance_estimator': self.covariance_estimator.__class__.__name__,
                'stock_classifier': self.stock_classifier.__class__.__name__
            }
        }