"""
Refactored Feature Engineering Pipeline

This module provides a simplified pipeline that leverages existing components
instead of reimplementing functionality.

Key Principles:
- KISS: Simple coordination of existing components
- SRP: Pipeline only coordinates, doesn't implement
- DRY: Reuses all existing components
- DIP: Depends on abstractions, not concrete implementations
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from .models.data_types import FeatureConfig
from .utils.technical_features import TechnicalIndicatorCalculator
from .utils.cross_sectional_features import CrossSectionalFeatureCalculator
from .utils.panel_data_transformer import PanelDataTransformer
from .cache_provider import FeatureCacheProvider
from .local_cache_provider import LocalCacheProvider

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Simplified feature engineering pipeline that coordinates existing components.

    This pipeline acts as a coordinator rather than reimplementing functionality,
    leveraging the existing ecosystem of feature engineering components.
    """

    def __init__(self,
                 config: FeatureConfig,
                 feature_cache: Optional[FeatureCacheProvider] = None,
                 auto_cleanup_cache: bool = True,
                 model_type: Optional[str] = None):
        """
        Initialize the pipeline with dependency injection of existing components.

        Args:
            config: Feature configuration
            feature_cache: Optional cache provider (uses LocalCacheProvider if None)
            auto_cleanup_cache: Whether to auto-clean invalid cache on init
            model_type: Model type for data format determination
        """
        self.config = config
        self.model_type = model_type
        self.scalers: Dict[str, StandardScaler] = {}
        self._is_fitted = False

        # Use existing cache provider
        self.feature_cache = feature_cache or LocalCacheProvider()

        # Initialize existing calculators
        self.technical_calculator = TechnicalIndicatorCalculator(config)

        # Initialize cross-sectional calculator if needed
        if hasattr(config, 'include_cross_sectional') and config.include_cross_sectional:
            lookback = getattr(config, 'cross_sectional_lookback', None)
            winsorize = getattr(config, 'winsorize_percentile', 0.01)
            self.cross_sectional_calculator = CrossSectionalFeatureCalculator(
                lookback_periods=lookback,
                winsorize_percentile=winsorize
            )
            logger.info("Initialized CrossSectionalFeatureCalculator")
        else:
            self.cross_sectional_calculator = None

        # Auto cleanup cache if requested
        if auto_cleanup_cache and hasattr(self.feature_cache, 'clear_invalid_cache'):
            try:
                cleared_count = self.feature_cache.clear_invalid_cache()
                if cleared_count > 0:
                    logger.info(f"Auto-cleaned {cleared_count} invalid cache files")
            except Exception as e:
                logger.warning(f"Failed to auto-clean cache: {e}")

        logger.info(f"Initialized pipeline with existing components, "
                   f"cross_sectional={'enabled' if self.cross_sectional_calculator else 'disabled'}, "
                   )

    def _get_index_format_for_model(self) -> Tuple[str, str]:
        """
        Determine the appropriate MultiIndex format based on model type.

        Returns:
            Tuple of (level_0_name, level_1_name) for the MultiIndex
        """
        if self.model_type is None:
            return ('date', 'symbol')

        # Model types that should use time series format (symbol, date)
        time_series_models = {
            'ff5_regression', 'linear_regression', 'ridge_regression', 'lasso_regression',
            'xgboost', 'lightgbm', 'random_forest', 'lstm', 'gru'
        }

        # Model types that should use panel data format (date, symbol)
        panel_data_models = {
            'fama_macbeth', 'panel_regression', 'cross_sectional', 'panel_ml'
        }

        model_type_lower = self.model_type.lower()

        if model_type_lower in time_series_models:
            return ('symbol', 'date')
        elif model_type_lower in panel_data_models:
            return ('date', 'symbol')
        else:
            # Default to time series format for unknown models
            return ('symbol', 'date')

    def fit(self, data: Dict[str, pd.DataFrame]):
        """
        Learn scaling parameters from training data using existing components.

        Args:
            data: Dictionary of DataFrames with 'price_data' key
        """
        logger.info("Fitting FeatureEngineeringPipeline...")

        # Compute features using existing components
        features = self.transform(data, is_fitting=True)

        # Fit scalers for numeric features if normalization is enabled
        if self.config.normalize_features and not features.empty:
            numeric_features = features.select_dtypes(include=['number']).columns
            for col in numeric_features:
                if not features[col].isna().all():
                    scaler = StandardScaler()
                    self.scalers[col] = scaler.fit(features[[col]].dropna())
                    logger.info(f"Fitted scaler for feature: {col}")

        self._is_fitted = True
        logger.info("FeatureEngineeringPipeline fitting complete.")

    def transform(self, data: Dict[str, pd.DataFrame], is_fitting: bool = False) -> pd.DataFrame:
        """
        Apply feature engineering using existing components.

        Args:
            data: Dictionary of DataFrames with 'price_data' and optionally 'factor_data'
            is_fitting: If True, skips scaling during fit phase

        Returns:
            DataFrame with all computed features
        """
        if not data or 'price_data' not in data:
            raise ValueError("Input data dictionary must contain 'price_data'.")

        price_data = data['price_data']
        logger.info(f"Processing features for {len(price_data)} symbols")

        # Step 1: Compute technical features using existing TechnicalIndicatorCalculator
        technical_features = self._compute_technical_features_with_cache(price_data)

        # Step 2: Compute cross-sectional features using existing CrossSectionalFeatureCalculator
        cross_sectional_features = pd.DataFrame()
        if self.cross_sectional_calculator is not None:
            cross_sectional_features = self._compute_cross_sectional_features_with_cache(price_data)

        # Step 3: Merge features using existing PanelDataTransformer
        features = self._merge_features_using_transformer(
            technical_features, cross_sectional_features
        )

        # Step 4: Add factor data if available
        if 'factor_data' in data and data['factor_data'] is not None:
            features = self._merge_factor_data(features, data['factor_data'])

        # Step 5: Handle NaN values using simplified approach
        if not features.empty:
            features = self._handle_nan_values(features)

        # Step 6: Apply scaling if fitted and not in fitting mode
        if self._is_fitted and not is_fitting:
            features = self._apply_scaling(features)

        # Step 7: Return processed features
        logger.info(f"Feature transformation complete: {features.shape}")
        return features

    def _compute_technical_features_with_cache(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute technical features using existing TechnicalIndicatorCalculator.

        This method leverages the existing TechnicalIndicatorCalculator instead of
        reimplementing technical indicator calculations.
        """
        logger.info("Computing technical features using TechnicalIndicatorCalculator...")
        all_features = []

        for symbol, data in price_data.items():
            logger.debug(f"Processing technical features for symbol: {symbol}")

            try:
                symbol_features = pd.DataFrame(index=data.index)

                # Validate input data using simplified validation
                self._validate_input_data(data, symbol)

                # Use existing TechnicalIndicatorCalculator for all technical indicators
                if hasattr(self.config, 'include_technical') and self.config.include_technical:
                    # Use existing _calculate_technical_for_group method
                    technical = self.technical_calculator._calculate_technical_for_group(data)

                    # Use existing _calculate_momentum_for_group method if configured
                    if hasattr(self.config, 'momentum_periods') and self.config.momentum_periods:
                        momentum_methods = getattr(self.config, 'momentum_methods', ['simple'])
                        momentum = self.technical_calculator._calculate_momentum_for_group(
                            data, self.config.momentum_periods, momentum_methods
                        )
                    else:
                        momentum = pd.DataFrame(index=data.index)

                    # Use existing _calculate_volatility_for_group method if configured
                    if hasattr(self.config, 'volatility_windows') and self.config.volatility_windows:
                        volatility_methods = getattr(self.config, 'volatility_methods', ['std'])
                        volatility = self.technical_calculator._calculate_volatility_for_group(
                            data, self.config.volatility_windows, volatility_methods
                        )
                    else:
                        volatility = pd.DataFrame(index=data.index)

                    # Combine all technical features for this symbol
                    combined_technical = pd.concat([technical, momentum, volatility], axis=1)

                    # Remove any all-NaN columns
                    valid_cols = [col for col in combined_technical.columns
                                if not combined_technical[col].isna().all()]
                    combined_technical = combined_technical[valid_cols]

                    if not combined_technical.empty:
                        symbol_features = pd.concat([symbol_features, combined_technical], axis=1)
                        logger.debug(f"Added {len(combined_technical.columns)} technical features for {symbol}")

                # Create MultiIndex for this symbol's features
                index_format = self._get_index_format_for_model()
                level_0_name, level_1_name = index_format

                # Convert date index to datetime
                date_index = pd.to_datetime(symbol_features.index)

                if index_format == ('symbol', 'date'):
                    # Time series format: (symbol, date)
                    symbol_multiindex = pd.MultiIndex.from_arrays([
                        [symbol] * len(date_index), date_index
                    ], names=[level_0_name, level_1_name])
                else:
                    # Panel data format: (date, symbol)
                    symbol_multiindex = pd.MultiIndex.from_arrays([
                        date_index, [symbol] * len(date_index)
                    ], names=[level_0_name, level_1_name])

                symbol_features.index = symbol_multiindex
                all_features.append(symbol_features)

                logger.debug(f"Completed technical features for {symbol}: {len(symbol_features.columns)} features")

            except Exception as e:
                logger.error(f"Failed to compute technical features for {symbol}: {e}")
                continue

        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=0)
            combined_features.sort_index(inplace=True)
            logger.info(f"Technical features computation complete: {combined_features.shape}")
            return combined_features
        else:
            logger.warning("No technical features computed from any symbols")
            return pd.DataFrame()

    def _compute_cross_sectional_features_with_cache(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute cross-sectional features using existing CrossSectionalFeatureCalculator.

        This method leverages the existing CrossSectionalFeatureCalculator instead of
        reimplementing cross-sectional feature calculations.
        """
        if self.cross_sectional_calculator is None:
            return pd.DataFrame()

        logger.info("Computing cross-sectional features using CrossSectionalFeatureCalculator...")

        try:
            # Get all unique dates across all symbols
            all_dates = set()
            for symbol, data in price_data.items():
                all_dates.update(data.index.tolist())
            all_dates = sorted(list(all_dates))

            # Determine which features to calculate from config
            feature_names = None
            if hasattr(self.config, 'cross_sectional_features'):
                feature_names = self.config.cross_sectional_features

            # Use existing CrossSectionalFeatureCalculator to compute panel features
            panel_features = self.cross_sectional_calculator.calculate_panel_features(
                price_data=price_data,
                dates=all_dates,
                feature_names=feature_names
            )

            # Use existing PanelDataTransformer to ensure consistent format
            if not panel_features.empty:
                try:
                    index_format = self._get_index_format_for_model()
                    if index_format == ('date', 'symbol'):
                        # Already in correct format
                        standardized_features = panel_features
                    else:
                        # Convert to time series format
                        standardized_features, _ = PanelDataTransformer.to_time_series_format(
                            panel_features
                        )

                    logger.info(f"Cross-sectional features computation complete: {standardized_features.shape}")
                    return standardized_features

                except Exception as e:
                    logger.warning(f"Cross-sectional feature format standardization failed: {e}")
                    return panel_features
            else:
                logger.warning("No cross-sectional features computed")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to compute cross-sectional features: {e}")
            return pd.DataFrame()

    def _merge_features_using_transformer(self,
                                        technical_features: pd.DataFrame,
                                        cross_sectional_features: pd.DataFrame) -> pd.DataFrame:
        """
        Merge features using existing PanelDataTransformer for format standardization.

        This method uses the existing PanelDataTransformer instead of manually
        handling MultiIndex format conversions.
        """
        if technical_features.empty and cross_sectional_features.empty:
            return pd.DataFrame()

        if technical_features.empty:
            features = cross_sectional_features
        elif cross_sectional_features.empty:
            features = technical_features
        else:
            # Merge features, handling different index formats
            try:
                # Use existing PanelDataTransformer for format standardization
                index_format = self._get_index_format_for_model()

                if index_format == ('date', 'symbol'):
                    # Convert technical features to panel format if needed
                    if not technical_features.empty:
                        technical_panel, _ = PanelDataTransformer.to_panel_format(technical_features)
                    else:
                        technical_panel = technical_features

                    # Cross-sectional features should already be in panel format
                    # Ensure both have the same index structure before concatenation
                    if not technical_panel.empty and not cross_sectional_features.empty:
                        # Check if indices are compatible
                        if isinstance(technical_panel.index, pd.MultiIndex) and isinstance(cross_sectional_features.index, pd.MultiIndex):
                            # Use concat with join='outer' to handle any index mismatches
                            features = pd.concat([technical_panel, cross_sectional_features], axis=1, join='outer')
                        else:
                            # Fallback: convert both to same format
                            features = pd.concat([technical_panel, cross_sectional_features], axis=1, ignore_index=False)
                    else:
                        # One of them is empty, just return the non-empty one
                        features = technical_panel if not technical_panel.empty else cross_sectional_features
                else:
                    # Convert cross-sectional features to time series format if needed
                    if not cross_sectional_features.empty:
                        cross_sectional_ts, _ = PanelDataTransformer.to_time_series_format(cross_sectional_features)
                    else:
                        cross_sectional_ts = cross_sectional_features

                    # Technical features should already be in time series format
                    # Ensure both have the same index structure before concatenation
                    if not technical_features.empty and not cross_sectional_ts.empty:
                        features = pd.concat([technical_features, cross_sectional_ts], axis=1, join='outer')
                    else:
                        # One of them is empty, just return the non-empty one
                        features = technical_features if not technical_features.empty else cross_sectional_ts

                # Remove duplicate columns if any
                if features.columns.duplicated().any():
                    features = features.loc[:, ~features.columns.duplicated()]

                logger.debug(f"Features merged successfully: {features.shape}")
                return features

            except Exception as e:
                logger.warning(f"Feature merging with PanelDataTransformer failed: {e}, using robust merge")
                # Robust fallback: reset indices and merge
                try:
                    if not technical_features.empty:
                        tech_reset = technical_features.reset_index()
                    else:
                        tech_reset = pd.DataFrame()

                    if not cross_sectional_features.empty:
                        cross_reset = cross_sectional_features.reset_index()
                    else:
                        cross_reset = pd.DataFrame()

                    # Merge and recreate index
                    merged = pd.concat([tech_reset, cross_reset], axis=1, ignore_index=False)

                    # Recreate MultiIndex if needed
                    if 'date' in merged.columns and 'symbol' in merged.columns:
                        merged = merged.set_index(['date', 'symbol'])

                    logger.debug(f"Robust merge completed: {merged.shape}")
                    return merged

                except Exception as e2:
                    logger.error(f"Robust merge also failed: {e2}")
                    # Final fallback: return non-empty features
                    return technical_features if not technical_features.empty else cross_sectional_features

        logger.debug(f"Single feature type returned: {features.shape}")
        return features

    def _merge_factor_data(self, features: pd.DataFrame, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge factor data with features (simplified implementation).

        This is a simplified version that focuses on the essential merging logic.
        """
        if factor_data is None or not isinstance(factor_data, pd.DataFrame):
            logger.info("No factor data available or invalid format, skipping merge")
            return features

        if factor_data.empty:
            logger.info("Factor data is empty, skipping merge")
            return features

        try:
            # Simplified factor data merging
            # This is a basic implementation - could be enhanced with more sophisticated alignment
            logger.info(f"Merging factor data with shape {factor_data.shape}")

            # Filter out non-numeric columns (metadata like DataSource, Provider, etc.)
            # Only keep numeric factor data for feature engineering
            numeric_factor_data = factor_data.select_dtypes(include=[np.number])
            non_numeric_cols = set(factor_data.columns) - set(numeric_factor_data.columns)
            if non_numeric_cols:
                logger.info(f"Filtering out non-numeric factor columns: {non_numeric_cols}")
                logger.debug(f"Keeping numeric columns: {list(numeric_factor_data.columns)}")
            factor_data = numeric_factor_data

            # Ensure factor data index is datetime
            factor_data.index = pd.to_datetime(factor_data.index)

            # Get date range from features
            if isinstance(features.index, pd.MultiIndex):
                feature_dates = features.index.get_level_values('date').unique()
            else:
                feature_dates = features.index.unique()

            # Resample factor data to match feature dates
            factor_data_resampled = factor_data.reindex(feature_dates, method='ffill')

            # For each symbol, add factor data
            if isinstance(features.index, pd.MultiIndex):
                symbols = features.index.get_level_values('symbol').unique()
                all_features = []

                for symbol in symbols:
                    symbol_features = features.xs(symbol, level='symbol')

                    # Add factor columns
                    for factor_col in factor_data_resampled.columns:
                        symbol_features[factor_col] = factor_data_resampled[factor_col].values

                    # Recreate MultiIndex
                    symbol_multiindex = pd.MultiIndex.from_product(
                        [[symbol], symbol_features.index], names=['symbol', 'date']
                    )
                    symbol_features.index = symbol_multiindex
                    all_features.append(symbol_features)

                merged_features = pd.concat(all_features, axis=0)
            else:
                # Simple concatenation for non-MultiIndex data
                merged_features = features.copy()
                for factor_col in factor_data_resampled.columns:
                    merged_features[factor_col] = factor_data_resampled[factor_col].values

            logger.info(f"Factor data merge complete: {merged_features.shape}")
            return merged_features

        except Exception as e:
            logger.error(f"Failed to merge factor data: {e}")
            return features

    def _handle_nan_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Simplified NaN handling using existing validation patterns.

        This is a simplified version that uses the most robust NaN handling strategies.
        """
        if features.empty:
            return features

        logger.info(f"Handling NaN values in features with shape {features.shape}")

        # Drop features with >95% NaN values (too corrupted to be useful)
        nan_percentages = features.isnull().sum() / len(features)
        cols_to_drop = nan_percentages[nan_percentages > 0.95].index.tolist()

        if cols_to_drop:
            logger.warning(f"Dropping {len(cols_to_drop)} features with >95% NaN values")
            features = features.drop(columns=cols_to_drop)

        # Apply robust NaN handling strategy
        # Forward fill, then backward fill, then interpolate, then median fill
        features_clean = features.ffill().bfill().interpolate(method='linear')

        # Final median fill for any remaining NaNs
        median_values = features_clean.median()
        features_clean = features_clean.fillna(median_values)

        # Handle infinite values
        features_clean = features_clean.replace([float('inf'), float('-inf')], [1e10, -1e10])

        # Clip extreme values to prevent numerical issues
        max_val = features_clean.abs().max().max()
        if max_val > 1e9:
            features_clean = features_clean.clip(-1e9, 1e9)
            logger.warning(f"Clipped extreme values to prevent numerical issues")

        # Log results
        remaining_nan = features_clean.isnull().sum().sum()
        if remaining_nan > 0:
            logger.warning(f"Still have {remaining_nan} NaN values after cleaning, applying zero fill")
            features_clean = features_clean.fillna(0)

        logger.info(f"NaN handling complete: final shape {features_clean.shape}")
        return features_clean

    def _apply_scaling(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned scaling to features.
        """
        if not features.empty and self.scalers:
            logger.debug("Applying feature scaling...")
            for col, scaler in self.scalers.items():
                if col in features.columns:
                    try:
                        features[col] = scaler.transform(features[[col]])
                    except Exception as e:
                        logger.warning(f"Failed to apply scaling to {col}: {e}")

        return features

    def _validate_input_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Simplified input data validation.
        """
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")

        if len(data) < 10:
            logger.warning(f"Very short data series for {symbol}: only {len(data)} rows")

    def save(self, file_path: Path):
        """Save the fitted pipeline to a file."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save a pipeline that has not been fitted yet.")
        logger.info(f"Saving feature pipeline to {file_path}")
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path: Path) -> 'FeatureEngineeringPipeline':
        """Load a fitted pipeline from a file."""
        logger.info(f"Loading feature pipeline from {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"Feature pipeline file not found at {file_path}")
        return joblib.load(file_path)

    @classmethod
    def from_config(cls, config: Dict[str, Any], model_type: Optional[str] = None) -> 'FeatureEngineeringPipeline':
        """Create pipeline from dictionary configuration."""
        feature_config = FeatureConfig(**config)
        return cls(feature_config, model_type=model_type)

    @classmethod
    def from_yaml(cls, file_path: Path, model_type: Optional[str] = None) -> 'FeatureEngineeringPipeline':
        """Create pipeline from YAML configuration file."""
        logger.info(f"Creating feature pipeline from config: {file_path}")
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if 'feature_engineering' not in config_dict:
            raise ValueError(f"Key 'feature_engineering' not found in {file_path}")

        return cls.from_config(config_dict['feature_engineering'], model_type=model_type)

    def get_max_lookback(self) -> int:
        """Determine maximum lookback period required by the configuration."""
        max_lookback = 0

        if hasattr(self.config, 'momentum_periods') and self.config.momentum_periods:
            max_lookback = max(max_lookback, max(self.config.momentum_periods))

        if hasattr(self.config, 'volatility_windows') and self.config.volatility_windows:
            max_lookback = max(max_lookback, max(self.config.volatility_windows))

        # Add lookback from technical indicators (SMA 200 is typically the longest)
        max_lookback = max(max_lookback, 200)

        # Add cross-sectional lookback if enabled
        if self.cross_sectional_calculator:
            cross_sectional_lookback = max(self.cross_sectional_calculator.lookback_periods.values())
            max_lookback = max(max_lookback, cross_sectional_lookback)

        # Add buffer
        return max_lookback + 5

    def clear_cache(self, symbol: Optional[str] = None):
        """Clear feature cache using existing cache provider."""
        self.feature_cache.clear(symbol)
        logger.info(f"Feature cache cleared{' for ' + symbol if symbol else ''}")