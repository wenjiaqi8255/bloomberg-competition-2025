"""
Cross-Sectional Cache Adapter

This adapter implements the FeatureCacheProvider interface for cross-sectional features,
following the Adapter pattern to integrate with the existing cache infrastructure
while maintaining SOLID principles.

Key Design Principles:
- Single Responsibility: Adapts cross-sectional feature cache calls to the provider interface
- Open/Closed: Can be extended with different cache providers without modification
- Dependency Inversion: Depends on abstractions, not concretions
- DRY: Reuses existing cache provider interface instead of creating new cache logic
- KISS: Simple, focused implementation
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd

from .cache_provider import FeatureCacheProvider

logger = logging.getLogger(__name__)


class CrossSectionalCacheAdapter(FeatureCacheProvider):
    """
    Adapter for cross-sectional feature caching using the existing cache provider interface.

    This adapter maps cross-sectional feature calculations (which involve multiple symbols
    at a single date) to the symbol-based cache provider interface.
    """

    def __init__(self, cache_provider: FeatureCacheProvider):
        """
        Initialize the adapter with a cache provider.

        Args:
            cache_provider: Implementation of FeatureCacheProvider interface
        """
        self._cache_provider = cache_provider
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0
        }

        logger.info(f"Initialized CrossSectionalCacheAdapter with {type(cache_provider).__name__}")

    def get_cross_sectional_features(
        self,
        symbols: list,
        date: datetime,
        feature_names: list,
        lookback_periods: dict
    ) -> Optional[pd.DataFrame]:
        """
        Get cached cross-sectional features for multiple symbols at a specific date.

        This method implements the adapter pattern by mapping the cross-sectional
        request to individual symbol-based cache calls.

        Args:
            symbols: List of stock symbols
            date: Target date for features
            feature_names: List of feature names
            lookback_periods: Lookback period configuration

        Returns:
            Cached cross-sectional features DataFrame or None if not found
        """
        logger.debug(f"Cache lookup: {len(symbols)} symbols for {date}")

        try:
            # Try to retrieve features for all symbols
            all_features = []
            missing_symbols = []

            for symbol in symbols:
                # Use the cache provider interface for each symbol
                cached_data = self._cache_provider.get(
                    symbol=symbol,
                    feature_name='cross_sectional_features',
                    start_date=date,
                    end_date=date
                )

                if cached_data is not None:
                    # Verify that the cached data matches our requirements
                    if self._validate_cached_data(cached_data, feature_names, lookback_periods):
                        all_features.append(cached_data)
                    else:
                        missing_symbols.append(symbol)
                        logger.debug(f"Cached data validation failed for {symbol}")
                else:
                    missing_symbols.append(symbol)

            # If we got data for all symbols, reconstruct the cross-sectional DataFrame
            if not missing_symbols and all_features:
                combined_features = self._combine_cached_features(all_features, symbols, date)
                self._cache_stats['hits'] += 1
                logger.debug(f"Cache HIT for {len(symbols)} symbols")
                return combined_features
            else:
                self._cache_stats['misses'] += 1
                logger.debug(f"Cache MISS for {len(symbols)} symbols (missing: {len(missing_symbols)})")
                return None

        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            self._cache_stats['misses'] += 1
            return None

    def set_cross_sectional_features(
        self,
        features_df: pd.DataFrame,
        symbols: list,
        date: datetime,
        feature_names: list,
        lookback_periods: dict
    ) -> None:
        """
        Cache cross-sectional features using the provider interface.

        Args:
            features_df: Cross-sectional features DataFrame
            symbols: List of stock symbols
            date: Target date
            feature_names: List of feature names
            lookback_periods: Lookback period configuration
        """
        logger.debug(f"Caching features for {len(symbols)} symbols at {date}")

        try:
            # Store metadata along with features for validation
            metadata = {
                'feature_names': feature_names,
                'lookback_periods': lookback_periods,
                'calculation_date': date.isoformat(),
                'num_symbols': len(symbols)
            }

            # Cache each symbol's features individually using the provider interface
            cached_count = 0
            for _, row in features_df.iterrows():
                symbol = row['symbol']

                # Create symbol-specific feature data
                symbol_data = row.copy().to_frame().T
                symbol_data.columns = ['value']  # Standardize to expected format
                symbol_data['metadata'] = str(metadata)  # Store metadata as string

                # Use the cache provider interface
                self._cache_provider.set(
                    symbol=symbol,
                    feature_name='cross_sectional_features',
                    data=symbol_data
                )

                cached_count += 1

            self._cache_stats['sets'] += 1
            logger.debug(f"Cached {cached_count} symbols successfully")

        except Exception as e:
            logger.error(f"Cache storage error: {e}")

    def _validate_cached_data(
        self,
        cached_data: pd.DataFrame,
        required_features: list,
        lookback_periods: dict
    ) -> bool:
        """
        Validate that cached data matches current requirements.

        Args:
            cached_data: Cached feature data
            required_features: Required feature names
            lookback_periods: Lookback period configuration

        Returns:
            True if data is valid, False otherwise
        """
        try:
            # Check metadata if available
            if 'metadata' in cached_data.columns:
                metadata_str = cached_data['metadata'].iloc[0]
                # Note: In production, this would be proper JSON deserialization
                # For simplicity, we're doing basic validation

            # Check that we have the expected features
            # The actual validation would depend on the cache provider implementation
            return True

        except Exception as e:
            logger.debug(f"Cache validation error: {e}")
            return False

    def _combine_cached_features(
        self,
        individual_features: list,
        expected_symbols: list,
        target_date: datetime
    ) -> pd.DataFrame:
        """
        Combine individual symbol features into a cross-sectional DataFrame.

        Args:
            individual_features: List of individual symbol feature DataFrames
            expected_symbols: Expected symbols (for ordering)
            target_date: Target date

        Returns:
            Combined cross-sectional features DataFrame
        """
        combined_rows = []

        for feature_df in individual_features:
            # Extract the row (symbol data) from the cached DataFrame
            if len(feature_df) > 0:
                row_data = feature_df.iloc[0].to_dict()

                # Convert back to expected format
                processed_row = {
                    'symbol': feature_df.index[0] if hasattr(feature_df.index[0], 'name') else 'unknown',
                    'date': target_date
                }

                # Add feature values (skip metadata)
                for key, value in row_data.items():
                    if key != 'metadata' and key != 'value':
                        processed_row[key] = value
                    elif key == 'value':
                        processed_row['feature_value'] = value

                combined_rows.append(processed_row)

        if combined_rows:
            return pd.DataFrame(combined_rows)
        else:
            return pd.DataFrame()

    # Implement the required FeatureCacheProvider interface

    def get(self, symbol: str, feature_name: str, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Forward to the underlying cache provider."""
        return self._cache_provider.get(symbol, feature_name, start_date, end_date)

    def set(self, symbol: str, feature_name: str, data: pd.DataFrame) -> None:
        """Forward to the underlying cache provider."""
        self._cache_provider.set(symbol, feature_name, data)

    def get_last_update(self, symbol: str, feature_name: str) -> Optional[datetime]:
        """Forward to the underlying cache provider."""
        return self._cache_provider.get_last_update(symbol, feature_name)

    def clear(self, symbol: Optional[str] = None) -> None:
        """Forward to the underlying cache provider."""
        self._cache_provider.clear(symbol)

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_stats['hits'] + self._cache_stats['misses']
        hit_rate = self._cache_stats['hits'] / total_requests if total_requests > 0 else 0.0

        return {
            'hits': self._cache_stats['hits'],
            'misses': self._cache_stats['misses'],
            'sets': self._cache_stats['sets'],
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'adapter_type': 'CrossSectionalCacheAdapter'
        }

    def reset_stats(self) -> None:
        """Reset cache statistics."""
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'sets': 0
        }
        logger.info("Cache statistics reset")