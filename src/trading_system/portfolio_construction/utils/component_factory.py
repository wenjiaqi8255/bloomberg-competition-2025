"""
Component Factory for Portfolio Construction
============================================

Provides a centralized factory for creating and configuring common components
used across different portfolio construction strategies. This ensures
consistent instantiation and adheres to the DRY principle.

Components created by this factory include:
- StockClassifier
- CovarianceEstimator (with caching)
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd

from src.trading_system.data.stock_classifier import StockClassifier
from src.trading_system.utils.risk import (
    CovarianceEstimator,
    SimpleCovarianceEstimator,
    LedoitWolfCovarianceEstimator,
    FactorModelCovarianceEstimator,
    CachedCovarianceEstimator
)
from src.trading_system.data.offline_stock_metadata_provider import OfflineStockMetadataProvider

logger = logging.getLogger(__name__)


class ComponentFactory:
    """Factory for creating shared portfolio construction components."""

    @staticmethod
    def create_stock_classifier(config: Dict[str, Any]) -> StockClassifier:
        """
        Create a StockClassifier instance.

        Args:
            config: Classifier configuration dictionary.

        Returns:
            An instance of StockClassifier.
        """
        classifier_config = config.copy()
        cache_enabled = classifier_config.pop('cache_enabled', True)
        
        offline_metadata_provider = None
        offline_csv_path = classifier_config.get('offline_metadata_csv_path')
        if offline_csv_path:
            try:
                offline_metadata_provider = OfflineStockMetadataProvider(offline_csv_path)
                logger.info(f"Initialized offline metadata provider from {offline_csv_path}")
            except Exception as e:
                logger.warning(f"Failed to initialize offline metadata provider: {e}")
        
        logger.info(f"Creating StockClassifier with cache_enabled={cache_enabled}")
        return StockClassifier(
            classifier_config,
            cache_enabled=cache_enabled,
            offline_metadata_provider=offline_metadata_provider
        )

    @staticmethod
    def create_covariance_estimator(
        config: Dict[str, Any],
        factor_data_provider: Optional[Any] = None
    ) -> CovarianceEstimator:
        """
        Create a covariance estimator based on the configuration.

        Args:
            config: A dictionary specifically for covariance estimation, containing:
                - method: 'simple', 'ledoit_wolf', 'factor_model'
                - lookback_days: Lookback period in days
                - covariance_cache: Cache configuration
            factor_data_provider: Optional factor data provider for factor models.

        Returns:
            An instance of a CovarianceEstimator.
        """
        method = config.get('method', 'ledoit_wolf')
        lookback_days = config.get('lookback_days', 252)

        # Create the base estimator
        if method == 'factor_model':
            if factor_data_provider is None:
                raise ValueError("Factor data provider is required for factor model covariance.")
            min_obs = config.get('min_regression_obs', 24)
            base_estimator = FactorModelCovarianceEstimator(
                factor_data_provider, lookback_days, min_obs
            )
            logger.info(f"Created base covariance estimator: FactorModelCovarianceEstimator")
        elif method == 'ledoit_wolf':
            base_estimator = LedoitWolfCovarianceEstimator(lookback_days)
            logger.info(f"Created base covariance estimator: LedoitWolfCovarianceEstimator")
        else:
            base_estimator = SimpleCovarianceEstimator(lookback_days)
            logger.info(f"Created base covariance estimator: SimpleCovarianceEstimator")
        
        # Wrap the estimator with a cache if enabled in the config
        cache_config = config.get('covariance_cache', {})
        if cache_config.get('enabled', True): # Caching is on by default
            max_cache_size = cache_config.get('max_cache_size', 50)
            logger.info(f"Covariance caching is enabled (max_size={max_cache_size}).")
            return CachedCovarianceEstimator(base_estimator, max_cache_size=max_cache_size)

        logger.info("Covariance caching is disabled.")
        return base_estimator
