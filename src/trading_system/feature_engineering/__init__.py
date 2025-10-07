"""
Simplified Feature Engineering Module.

This module provides a clean, unified interface for feature engineering
with comprehensive technical indicators and validation.

Key Features:
- Simplified interface with single compute_features method
- Comprehensive technical indicators
- Information Coefficient validation
- Academic-grade feature metrics
- High performance implementation

Usage:
    # Basic usage
    from src.trading_system.feature_engineering import compute_features
    result = compute_features(price_data, forward_returns)

    # Advanced usage with custom configuration
    from src.trading_system.feature_engineering import FeatureEngine, FeatureConfig, FeatureType
    config = FeatureConfig(
        enabled_features=[FeatureType.MOMENTUM, FeatureType.VOLATILITY],
        min_ic_threshold=0.05
    )
    engine = FeatureEngine(config)
    result = engine.compute_features(price_data, forward_returns)

    # Access results
    features = result.features
    metrics = result.metrics
    accepted_features = result.accepted_features
"""

# Core imports
import pandas as pd
from typing import Optional, List, Dict
from .models.data_types import (
    FeatureMetrics, FeatureResult,
    PriceData, ForwardReturns, ValidationData, FeatureData,
    validate_price_data, align_data, create_default_config
)
from ..config.feature import FeatureType, FeatureConfig
from .utils.technical_features import TechnicalIndicatorCalculator


# ============================================================================
# Public API - Simplified Interface
# ============================================================================

def compute_technical_features(price_data: PriceData,
                             forward_returns: Optional[ForwardReturns] = None,
                             config: Optional[FeatureConfig] = None) -> FeatureResult:
    """
    Compute technical features with validation.

    This is the main entry point for feature engineering.

    Args:
        price_data: Dictionary of price DataFrames by symbol
        forward_returns: Optional forward returns for validation
        config: Optional configuration, uses defaults if None

    Returns:
        FeatureResult object with features and validation metrics

    Example:
        >>> import pandas as pd
        >>> from src.trading_system.feature_engineering import compute_technical_features
        >>>
        >>> # Prepare price data
        >>> price_data = {
        ...     'AAPL': pd.DataFrame({
        ...         'Open': [150, 151, 152],
        ...         'High': [151, 152, 153],
        ...         'Low': [149, 150, 151],
        ...         'Close': [150.5, 151.5, 152.5],
        ...         'Volume': [1000000, 1100000, 1200000]
        ...     }, index=pd.date_range('2023-01-01', periods=3))
        ... }
        >>>
        >>> # Compute features
        >>> result = compute_technical_features(price_data)
        >>> print(f"Computed {len(result.features.columns)} features")
        >>> print(f"Accepted {len(result.accepted_features)} features")
    """
    return compute_features(price_data, forward_returns, config)


def create_momentum_features(price_data: PriceData,
                           periods: Optional[List[int]] = None) -> FeatureData:
    """
    Compute only momentum features.

    Args:
        price_data: Dictionary of price DataFrames by symbol
        periods: Optional list of momentum periods

    Returns:
        DataFrame with momentum features
    """
    config = FeatureConfig(
        enabled_features=[FeatureType.MOMENTUM],
        momentum_periods=periods or [21, 63, 126, 252]
    )
    result = compute_features(price_data, config=config)
    return result.features


def create_volatility_features(price_data: PriceData,
                             windows: Optional[List[int]] = None) -> FeatureData:
    """
    Compute only volatility features.

    Args:
        price_data: Dictionary of price DataFrames by symbol
        windows: Optional list of volatility windows

    Returns:
        DataFrame with volatility features
    """
    config = FeatureConfig(
        enabled_features=[FeatureType.VOLATILITY],
        volatility_windows=windows or [20, 60]
    )
    result = compute_features(price_data, config=config)
    return result.features


def create_technical_indicators(price_data: PriceData) -> FeatureData:
    """
    Compute only technical indicators (RSI, MACD, etc.).

    Args:
        price_data: Dictionary of price DataFrames by symbol

    Returns:
        DataFrame with technical indicators
    """
    config = FeatureConfig(
        enabled_features=[FeatureType.TECHNICAL],
        include_technical=True
    )
    result = compute_features(price_data, config=config)
    return result.features


def validate_feature_performance(features: FeatureData,
                              forward_returns: ForwardReturns,
                              min_ic_threshold: float = 0.03) -> Dict[str, FeatureMetrics]:
    """
    Validate feature performance using Information Coefficient analysis.

    Args:
        features: Feature DataFrame
        forward_returns: Forward returns by symbol
        min_ic_threshold: Minimum IC threshold for acceptance

    Returns:
        Dictionary of feature validation metrics
    """
    return validate_features(features, forward_returns, min_ic_threshold)


# ============================================================================
# Configuration Helpers
# ============================================================================

def create_momentum_config() -> FeatureConfig:
    """Create configuration optimized for momentum strategies."""
    return FeatureConfig(
        enabled_features=[FeatureType.MOMENTUM, FeatureType.TREND],
        momentum_periods=[21, 63, 126, 252],
        min_ic_threshold=0.03,
        feature_lag=1
    )


def create_volatility_config() -> FeatureConfig:
    """Create configuration optimized for volatility-based strategies."""
    return FeatureConfig(
        enabled_features=[FeatureType.VOLATILITY, FeatureType.MEAN_REVERSION],
        volatility_windows=[20, 60, 120],
        min_ic_threshold=0.02,
        feature_lag=1
    )


def create_technical_config() -> FeatureConfig:
    """Create configuration optimized for technical analysis strategies."""
    return FeatureConfig(
        enabled_features=[FeatureType.TECHNICAL, FeatureType.VOLUME, FeatureType.LIQUIDITY],
        include_technical=True,
        min_ic_threshold=0.025,
        feature_lag=1
    )


def create_academic_config() -> FeatureConfig:
    """Create configuration for academic-grade validation."""
    return FeatureConfig(
        enabled_features=[FeatureType.MOMENTUM, FeatureType.VOLATILITY, FeatureType.TECHNICAL],
        momentum_periods=[21, 63, 126, 252],
        volatility_windows=[20, 60],
        include_technical=True,
        min_ic_threshold=0.05,
        min_significance=0.01,
        feature_lag=1,
        normalize_features=True,
        normalization_method="robust"
    )


def create_production_config() -> FeatureConfig:
    """Create configuration optimized for production trading."""
    return FeatureConfig(
        enabled_features=[FeatureType.MOMENTUM, FeatureType.VOLATILITY, FeatureType.TECHNICAL,
                         FeatureType.VOLUME, FeatureType.TREND],
        momentum_periods=[21, 63],
        volatility_windows=[20, 60],
        include_technical=True,
        min_ic_threshold=0.03,
        min_significance=0.05,
        feature_lag=1,
        normalize_features=True,
        normalization_method="robust",
        max_features=30
    )


# ============================================================================
# Version Info
# ============================================================================

__version__ = "3.0.0"
__author__ = "Bloomberg Competition Team"

# Export main classes and functions
__all__ = [
    # Core classes
    'FeatureConfig',
    'FeatureType',
    'FeatureResult',
    'FeatureMetrics',
    'TechnicalIndicatorCalculator',

    # Main functions
    'compute_technical_features',
    'compute_features',
    'validate_feature_performance',

    # Specialized feature creators
    'create_momentum_features',
    'create_volatility_features',
    'create_technical_indicators',

    # Configuration helpers
    'create_momentum_config',
    'create_volatility_config',
    'create_technical_config',
    'create_academic_config',
    'create_production_config',

    # Types
    'PriceData',
    'ForwardReturns',
    'FeatureData',
    'ValidationData',

    # Version
    '__version__',
]

# Set up logging
import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())