"""
Simplified Feature Engine - Core Implementation.

This is the new core feature engineering implementation that replaces
the complex orchestration system with a simple, clean interface.
"""

import logging
import warnings
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from datetime import datetime

from .models.data_types import (
    IFeatureEngine, FeatureResult, FeatureMetrics,
    PriceData, ForwardReturns, FeatureData, ValidationData,
    validate_price_data, align_data, create_default_config
)
from ..config.feature import FeatureType, FeatureConfig
from .utils.technical_features import TechnicalIndicatorCalculator
from .utils.validation import FeatureValidator

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class FeatureEngine(IFeatureEngine):
    """
    Simplified feature engineering engine.

    This engine provides a clean, unified interface for computing
    technical features with validation.
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        """
        Initialize feature engine.

        Args:
            config: Optional configuration, uses defaults if None
        """
        self.config = config or create_default_config()
        self.validator = FeatureValidator(
            min_ic_threshold=self.config.min_ic_threshold,
            min_significance=self.config.min_significance
        )
        self.technical_calculator = TechnicalIndicatorCalculator()

        # Cache for performance
        self._feature_names_cache: Optional[List[str]] = None
        self._last_result: Optional[FeatureResult] = None

        logger.info(f"Initialized FeatureEngine with {len(self.config.enabled_features)} feature types")

    def compute_features(self,
                        price_data: PriceData,
                        forward_returns: Optional[ForwardReturns] = None,
                        config: Optional[FeatureConfig] = None) -> FeatureResult:
        """
        Compute features for all symbols.

        Args:
            price_data: Dictionary of price DataFrames by symbol
            forward_returns: Optional forward returns for validation
            config: Optional configuration override

        Returns:
            FeatureResult object with features and metrics
        """
        # Update config if provided
        if config:
            self.config = config
            self.validator = FeatureValidator(
                min_ic_threshold=self.config.min_ic_threshold,
                min_significance=self.config.min_significance
            )

        # Validate input data
        if not validate_price_data(price_data):
            raise ValueError("Invalid price data format")

        # Align data
        aligned_price_data, aligned_forward_returns = align_data(price_data, forward_returns)

        logger.info(f"Computing features for {len(aligned_price_data)} symbols")

        # Compute features for each symbol
        all_features = {}
        for symbol, data in aligned_price_data.items():
            try:
                symbol_features = self._compute_symbol_features(data, symbol)
                all_features[symbol] = symbol_features
                logger.debug(f"Computed {len(symbol_features.columns)} features for {symbol}")
            except Exception as e:
                logger.error(f"Failed to compute features for {symbol}: {e}")
                continue

        # Combine features across symbols
        combined_features = self._combine_features(all_features)

        # Validate features if forward returns provided
        metrics = {}
        if aligned_forward_returns:
            logger.info("Validating features...")
            metrics = self.validator.validate_features(
                combined_features, aligned_forward_returns
            )

        # Get accepted features
        accepted_features = [
            name for name, metric in metrics.items()
            if metric.recommendation in ["ACCEPT", "MARGINAL"]
        ]

        # Create result object
        result = FeatureResult(
            features=combined_features,
            metrics=metrics,
            config=self.config,
            symbols=list(aligned_price_data.keys()),
            feature_names=list(combined_features.columns),
            accepted_features=accepted_features,
            total_features=len(combined_features.columns),
            accepted_count=len(accepted_features),
            acceptance_rate=len(accepted_features) / max(1, len(combined_features.columns))
        )

        # Cache result
        self._last_result = result
        self._feature_names_cache = list(combined_features.columns)

        logger.info(f"Feature computation completed: {len(combined_features.columns)} total, "
                   f"{len(accepted_features)} accepted ({result.acceptance_rate:.1%})")

        return result

    def _compute_symbol_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Compute features for a single symbol."""
        features = pd.DataFrame(index=data.index)

        # Compute enabled feature types
        for feature_type in self.config.enabled_features:
            try:
                if feature_type == FeatureType.MOMENTUM:
                    momentum_features = self.technical_calculator.compute_momentum_features(
                        data, self.config.momentum_periods
                    )
                    features = pd.concat([features, momentum_features], axis=1)

                elif feature_type == FeatureType.VOLATILITY:
                    volatility_features = self.technical_calculator.compute_volatility_features(
                        data, self.config.volatility_windows
                    )
                    features = pd.concat([features, volatility_features], axis=1)

                elif feature_type == FeatureType.TECHNICAL and self.config.include_technical:
                    technical_features = self.technical_calculator.compute_technical_indicators(data)
                    features = pd.concat([features, technical_features], axis=1)

                elif feature_type == FeatureType.VOLUME:
                    volume_features = self.technical_calculator.compute_volume_features(data)
                    features = pd.concat([features, volume_features], axis=1)

                elif feature_type == FeatureType.LIQUIDITY:
                    liquidity_features = self.technical_calculator.compute_liquidity_features(data)
                    features = pd.concat([features, liquidity_features], axis=1)

                elif feature_type == FeatureType.MEAN_REVERSION:
                    mean_reversion_features = self.technical_calculator.compute_mean_reversion_features(
                        data, self.config.lookback_periods
                    )
                    features = pd.concat([features, mean_reversion_features], axis=1)

                elif feature_type == FeatureType.TREND:
                    trend_features = self.technical_calculator.compute_trend_features(
                        data, self.config.lookback_periods
                    )
                    features = pd.concat([features, trend_features], axis=1)

            except Exception as e:
                logger.warning(f"Failed to compute {feature_type.value} features for {symbol}: {e}")
                continue

        # Clean up features
        features = self._clean_features(features)

        return features

    def _combine_features(self, symbol_features: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Combine features from all symbols into a single DataFrame."""
        if not symbol_features:
            return pd.DataFrame()

        # Create a unified feature DataFrame with symbol prefixes
        combined_features_list = []

        for symbol, features in symbol_features.items():
            # Add symbol prefix to avoid column name conflicts
            prefixed_features = features.copy()
            prefixed_features.columns = [f"{symbol}_{col}" for col in prefixed_features.columns]
            combined_features_list.append(prefixed_features)

        # Concatenate all features
        combined_features = pd.concat(combined_features_list, axis=1, sort=False)

        return combined_features

    def _clean_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess features."""
        # Handle infinite values
        features = features.replace([np.inf, -np.inf], np.nan)

        # Apply feature lag to prevent look-ahead bias
        if self.config.feature_lag > 0:
            features = features.shift(self.config.feature_lag)

        # Fill NaN values
        features = features.ffill().bfill().fillna(0)

        # Remove constant features (zero variance)
        constant_cols = []
        for col in features.columns:
            if features[col].var() == 0:
                constant_cols.append(col)

        if constant_cols:
            features = features.drop(columns=constant_cols)
            logger.debug(f"Removed {len(constant_cols)} constant features")

        return features

    def get_feature_names(self) -> List[str]:
        """Get list of feature names this engine generates."""
        if self._feature_names_cache is None:
            if self._last_result:
                self._feature_names_cache = self._last_result.feature_names
            else:
                # Generate sample data to get feature names
                sample_data = self._generate_sample_data()
                sample_result = self.compute_features(sample_data)
                self._feature_names_cache = sample_result.feature_names

        return self._feature_names_cache

    def get_config(self) -> FeatureConfig:
        """Get current configuration."""
        return self.config

    def _generate_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Generate sample price data for feature name extraction."""
        dates = pd.date_range('2023-01-01', periods=500, freq='D')
        sample_symbol = 'SAMPLE'

        # Generate realistic OHLCV data
        np.random.seed(42)
        base_price = 100
        returns = np.random.normal(0, 0.02, len(dates))
        prices = base_price * np.cumprod(1 + returns)

        high = prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates))))
        low = prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates))))
        open_price = np.roll(prices, 1)
        open_price[0] = prices[0]
        volume = np.random.lognormal(15, 1, len(dates))

        sample_data = pd.DataFrame({
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': prices,
            'Volume': volume
        }, index=dates)

        return {sample_symbol: sample_data}

    def get_feature_summary(self) -> Optional[pd.DataFrame]:
        """Get summary of last computed features."""
        if not self._last_result:
            return None

        summary_data = []
        for feature_name, metrics in self._last_result.metrics.items():
            summary_data.append({
                'feature_name': feature_name,
                'ic': metrics.information_coefficient,
                'ic_p_value': metrics.ic_p_value,
                'positive_ic_ratio': metrics.positive_ic_ratio,
                'stability': metrics.feature_stability,
                'economic_significance': metrics.economic_significance,
                'recommendation': metrics.recommendation,
                'accepted': feature_name in self._last_result.accepted_features
            })

        return pd.DataFrame(summary_data).sort_values('ic', ascending=False)


# ============================================================================
# Convenience Functions
# ============================================================================

def create_feature_engine(config: Optional[FeatureConfig] = None) -> FeatureEngine:
    """Create a feature engine with default or custom configuration."""
    return FeatureEngine(config)


def compute_features(price_data: PriceData,
                    forward_returns: Optional[ForwardReturns] = None,
                    config: Optional[FeatureConfig] = None) -> FeatureResult:
    """
    Convenience function to compute features without explicitly creating engine.

    Args:
        price_data: Dictionary of price DataFrames by symbol
        forward_returns: Optional forward returns for validation
        config: Optional configuration

    Returns:
        FeatureResult object with features and metrics
    """
    engine = FeatureEngine(config)
    return engine.compute_features(price_data, forward_returns)