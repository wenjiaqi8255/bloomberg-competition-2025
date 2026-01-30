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
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from .models.data_types import FeatureConfig
from trading_system.feature_engineering.components.technical_features import TechnicalIndicatorCalculator
from trading_system.feature_engineering.components.cross_sectional_features import CrossSectionalFeatureCalculator
from .utils.panel_data_transformer import PanelDataTransformer
from trading_system.feature_engineering.utils.cache_provider import FeatureCacheProvider
from trading_system.feature_engineering.utils.local_cache_provider import LocalCacheProvider
from .box_feature_provider import BoxFeatureProvider, BoxFeatureConfig

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

        # Initialize box feature provider if configured
        self.box_feature_provider = None
        if hasattr(config, 'box_features') and config.box_features:
            box_config = BoxFeatureConfig(**config.box_features)
            if box_config.enabled:
                self.box_feature_provider = BoxFeatureProvider(box_config)
                logger.info("Initialized BoxFeatureProvider")
            else:
                logger.info("Box features are disabled in configuration")
        else:
            logger.info("No box features configuration found")

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
                   f"box_features={'enabled' if self.box_feature_provider else 'disabled'}"
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
        Learn parameters from training data using existing components.

        This method fits all sub-components and learns statistics needed for
        preventing data leakage during backtesting.

        Args:
            data: Dictionary of DataFrames with 'price_data' key
        """
        logger.info("Fitting FeatureEngineeringPipeline...")

        # Step 1: Fit cross-sectional calculator if available
        if self.cross_sectional_calculator is not None:
            logger.info("Fitting CrossSectionalFeatureCalculator...")
            # Extract training dates from price data
            all_dates = set()
            for symbol, symbol_data in data['price_data'].items():
                all_dates.update(symbol_data.index.tolist())
            train_dates = sorted(list(all_dates))
            
            # Extract country risk provider from data if available
            country_risk_provider = None
            if 'country_risk_provider' in data:
                country_risk_provider = data['country_risk_provider']
            
            # Fit cross-sectional calculator
            self.cross_sectional_calculator.fit(
                data['price_data'], 
                train_dates,
                country_risk_data=country_risk_provider.get_symbol_country_risk_data() if country_risk_provider else None
            )
            logger.info("CrossSectionalFeatureCalculator fitted successfully")

        # Step 2: Fit box feature provider if available
        if self.box_feature_provider is not None:
            logger.info("Fitting BoxFeatureProvider...")
            # Use the latest date as training end date
            all_dates = set()
            for symbol, symbol_data in data['price_data'].items():
                all_dates.update(symbol_data.index.tolist())
            train_end_date = max(all_dates)

            # Fit box feature provider
            self.box_feature_provider.fit(data['price_data'], train_end_date)
            logger.info("BoxFeatureProvider fitted successfully")

        # Note: FF5 betas computation moved to FF5Model class
        # Pipeline is now responsible only for feature extraction (algorithms), not parameter estimation

        # Step 3: Compute features to learn NaN fill statistics
        logger.info("Computing features to learn NaN fill statistics...")
        features = self.transform(data, is_fitting=True)

        # Step 4: Learn NaN fill statistics
        if not features.empty:
            self.nan_fill_values = features.median()
            logger.info(f"Learned NaN fill statistics for {len(self.nan_fill_values)} features")
        else:
            logger.warning("No features computed during fitting, cannot learn NaN fill statistics")
            self.nan_fill_values = {}

        # Step 5: Fit scalers for numeric features if normalization is enabled
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
            # Extract country risk provider from data if available
            country_risk_provider = None
            if 'country_risk_provider' in data:
                country_risk_provider = data['country_risk_provider']
            cross_sectional_features = self._compute_cross_sectional_features_with_cache(
                price_data, country_risk_provider, use_transform=not is_fitting
            )

        # Step 3: Compute box features if configured
        box_features = pd.DataFrame()
        if self.box_feature_provider is not None:
            box_features = self._compute_box_features_with_cache(price_data, use_transform=not is_fitting)
            logger.info(f"Box features computed: {box_features.shape}")

        # Step 4: Create factor features if available
        factor_features = pd.DataFrame()
        if 'factor_data' in data and isinstance(data['factor_data'], pd.DataFrame) and not data['factor_data'].empty:
            logger.info("Factor data available, creating factor features")
            factor_features = self._create_factor_features(price_data, data['factor_data'])
        else:
            logger.info("No factor data available, skipping factor features")

        # Step 5: Collect all feature types and merge once
        all_feature_dfs = []
        if not technical_features.empty:
            all_feature_dfs.append(technical_features)
        if not cross_sectional_features.empty:
            all_feature_dfs.append(cross_sectional_features)
        if not box_features.empty:
            all_feature_dfs.append(box_features)
        if not factor_features.empty:
            all_feature_dfs.append(factor_features)

        # Merge all features using existing transformer
        if all_feature_dfs:
            if len(all_feature_dfs) == 1:
                features = all_feature_dfs[0]
            else:
                features = self._merge_features_using_transformer(all_feature_dfs[0], all_feature_dfs[1])
                for i in range(2, len(all_feature_dfs)):
                    features = self._merge_features_using_transformer(features, all_feature_dfs[i])
        else:
            features = pd.DataFrame()

        # Step 7: Handle NaN values using learned statistics
        if not features.empty:
            features = self._handle_nan_values(features, use_learned_stats=not is_fitting)

        # Step 8: Apply scaling if fitted and not in fitting mode
        if self._is_fitted and not is_fitting:
            features = self._apply_scaling(features)

        # Step 9: Validate that features were generated when expected
        if features.empty and (self.config.include_technical or 
                              (self.cross_sectional_calculator is not None) or 
                              (self.box_feature_provider is not None)):
            error_msg = ("No features were generated despite feature engineering being enabled. "
                        "This may be due to insufficient data for technical indicators or "
                        "configuration issues. Check your data length and feature configuration.")
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Step 10: Return processed features
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
                    else:
                        logger.warning(f"No valid technical features generated for {symbol} - all features were NaN")

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

    def _compute_cross_sectional_features_with_cache(
        self,
        price_data: Dict[str, pd.DataFrame],
        country_risk_provider=None,
        use_transform: bool = False
    ) -> pd.DataFrame:
        """
        Compute cross-sectional features using existing CrossSectionalFeatureCalculator.

        This method leverages the existing CrossSectionalFeatureCalculator instead of
        reimplementing cross-sectional feature calculations.
        """
        if self.cross_sectional_calculator is None:
            return pd.DataFrame()

        logger.info("Computing cross-sectional features using CrossSectionalFeatureCalculator...")

        try:
            # Get country risk data if provider is available
            country_risk_data = None
            if country_risk_provider:
                try:
                    country_risk_data = country_risk_provider.get_symbol_country_risk_data()
                    logger.info(f"Loaded country risk data for {len(country_risk_data)} symbols")
                except Exception as e:
                    logger.warning(f"Failed to load country risk data: {e}")
                    country_risk_data = None

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
                feature_names=feature_names,
                country_risk_data=country_risk_data,
                use_transform=use_transform
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

    def _compute_box_features_with_cache(self, price_data: Dict[str, pd.DataFrame], 
                                        use_transform: bool = False) -> pd.DataFrame:
        """
        Compute box features using BoxFeatureProvider.

        Args:
            price_data: Dictionary of price data for all symbols

        Returns:
            DataFrame with box classification features
        """
        if self.box_feature_provider is None:
            return pd.DataFrame()

        logger.info("Computing box features using BoxFeatureProvider...")

        try:
            # Get symbols from price data
            symbols = list(price_data.keys())
            if not symbols:
                logger.warning("No symbols found in price data")
                return pd.DataFrame()

            # Generate box features
            box_features = self.box_feature_provider.generate_box_features(
                price_data=price_data,
                symbols=symbols,
                use_transform=use_transform
            )

            if not box_features.empty:
                logger.info(f"Box features computed successfully: {box_features.shape}")
                return box_features
            else:
                logger.warning("Box features computation returned empty result")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to compute box features: {e}")
            return pd.DataFrame()

    def _merge_features_with_box_features(self,
                                         features: pd.DataFrame,
                                         box_features: pd.DataFrame) -> pd.DataFrame:
        """
        Merge existing features with box features.

        Args:
            features: Existing feature DataFrame
            box_features: Box feature DataFrame to merge

        Returns:
            Merged feature DataFrame
        """
        if box_features.empty:
            return features

        if features.empty:
            return box_features

        try:
            logger.info(f"Merging features {features.shape} with box features {box_features.shape}")

            # Ensure both have the same index structure
            if isinstance(features.index, pd.MultiIndex) and isinstance(box_features.index, pd.MultiIndex):
                # Both have MultiIndex, use outer join
                merged_features = pd.concat([features, box_features], axis=1, join='outer')
            else:
                # Handle different index formats
                if isinstance(features.index, pd.MultiIndex):
                    # Reset box_features index to match features format
                    if not box_features.empty:
                        box_features_reset = box_features.reset_index()
                        box_features_reset.set_index(['date', 'symbol'], inplace=True)
                        merged_features = pd.concat([features, box_features_reset], axis=1, join='outer')
                    else:
                        merged_features = features
                else:
                    # Convert both to simple format and merge
                    features_reset = features.reset_index()
                    box_features_reset = box_features.reset_index()
                    merged_features = pd.concat([features_reset, box_features_reset], axis=1, join='outer')

            # Remove duplicate columns if any
            if merged_features.columns.duplicated().any():
                merged_features = merged_features.loc[:, ~merged_features.columns.duplicated()]

            logger.debug(f"Features merged with box features successfully: {merged_features.shape}")
            return merged_features

        except Exception as e:
            logger.error(f"Failed to merge features with box features: {e}")
            # Fallback: return original features
            return features

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

    def _merge_country_risk_features(
        self,
        panel_features: pd.DataFrame,
        country_risk_data: Dict[str, Dict[str, float]],
        dates: List
    ) -> pd.DataFrame:
        """
        Merge country risk features into panel features.

        Args:
            panel_features: Existing panel features DataFrame
            country_risk_data: Country risk data mapped to symbols
            dates: List of dates in the panel

        Returns:
            Panel features DataFrame with country risk features added
        """
        try:
            logger.info(f"Merging country risk features into panel data with shape {panel_features.shape}")

            # Create country risk panel data
            country_risk_records = []

            for date in dates:
                for symbol, risk_data in country_risk_data.items():
                    record = {
                        'date': date,
                        'symbol': symbol,
                        'country_risk_premium': risk_data.get('country_risk_premium', 0.0),
                        'equity_risk_premium': risk_data.get('equity_risk_premium', 0.0),
                        'default_spread': risk_data.get('default_spread', 0.0),
                        'corporate_tax_rate': risk_data.get('corporate_tax_rate', 0.0)
                    }
                    country_risk_records.append(record)

            # Create country risk panel DataFrame
            if country_risk_records:
                country_risk_panel = pd.DataFrame(country_risk_records)
                country_risk_panel = country_risk_panel.set_index(['date', 'symbol'])

                # Merge with existing panel features
                merged_panel = pd.concat([panel_features, country_risk_panel], axis=1, join='outer')

                logger.info(f"Successfully merged country risk features: {merged_panel.shape}")
                return merged_panel
            else:
                logger.warning("No country risk records to merge")
                return panel_features

        except Exception as e:
            logger.error(f"Failed to merge country risk features: {e}")
            return panel_features

    def _create_factor_features(self, price_data: Dict[str, pd.DataFrame], factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        只提取因子值，不计算betas

        Betas应该由模型自己计算，不是特征工程的职责。
        Pipeline只负责提供"算法"（因子值提取），而不是"值"（betas）。

        Args:
            price_data: Dictionary of price DataFrames by symbol
            factor_data: Factor data DataFrame with numeric columns

        Returns:
            DataFrame with factor values only in proper MultiIndex format
        """
        if factor_data is None or not isinstance(factor_data, pd.DataFrame) or factor_data.empty:
            logger.warning("No factor data available, returning empty DataFrame")
            return pd.DataFrame()

        try:
            logger.info("Creating factor features (values only)...")

            # 根据模型类型选择因子
            # FF3模型只需要3个因子：MKT, SMB, HML
            # FF5模型需要5个因子：MKT, SMB, HML, RMW, CMA
            if self.model_type and self.model_type.lower() in ['ff3_regression', 'fama_french_3']:
                factor_cols = ['MKT', 'SMB', 'HML']  # FF3只用3个因子
                logger.info("Using FF3 factors: MKT, SMB, HML")
            else:
                factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']  # FF5用5个因子
                logger.info("Using FF5 factors: MKT, SMB, HML, RMW, CMA")
            
            missing_factors = [col for col in factor_cols if col not in factor_data.columns]
            if missing_factors:
                logger.error(f"Missing required factors in factor_data: {missing_factors}. Required: {factor_cols}")
                return pd.DataFrame()

            # 只选择数值型的因子数据
            factor_data_numeric = factor_data[factor_cols].select_dtypes(include=[np.number])

            # 确保因子数据索引是datetime
            factor_data_numeric.index = pd.to_datetime(factor_data_numeric.index)

            # 获取所有日期和符号
            all_dates = set()
            for symbol, data in price_data.items():
                if not data.empty:
                    all_dates.update(data.index.tolist())

            all_dates = sorted(list(all_dates))

            if not all_dates:
                logger.warning("No valid dates found in price data")
                return pd.DataFrame()

            # 对齐因子数据到所有日期
            factor_data_resampled = factor_data_numeric.reindex(all_dates, method='ffill')

            # 处理缺失值
            if factor_data_resampled.isna().any().any():
                logger.warning("Factor data contains missing values, applying forward fill")
                factor_data_resampled = factor_data_resampled.fillna(method='ffill').fillna(0)

            # 为每个符号创建特征（只包含因子值）
            all_features = []
            index_format = self._get_index_format_for_model()

            for symbol in price_data.keys():
                try:
                    # 为每个日期创建特征
                    symbol_features = pd.DataFrame(index=all_dates)

                    # 只保留因子值（这是"算法"而不是"值"）
                    for factor_col in factor_cols:
                        symbol_features[factor_col] = factor_data_resampled[factor_col].values

                    # 创建MultiIndex
                    if index_format == ('symbol', 'date'):
                        # Time series format: (symbol, date)
                        symbol_multiindex = pd.MultiIndex.from_arrays([
                            [symbol] * len(all_dates),
                            pd.to_datetime(all_dates)
                        ], names=['symbol', 'date'])
                    else:
                        # Panel data format: (date, symbol)
                        symbol_multiindex = pd.MultiIndex.from_arrays([
                            pd.to_datetime(all_dates),
                            [symbol] * len(all_dates)
                        ], names=['date', 'symbol'])

                    symbol_features.index = symbol_multiindex
                    all_features.append(symbol_features)

                    logger.debug(f"Created factor features for {symbol}: {symbol_features.shape}")

                except Exception as e:
                    logger.error(f"Failed to create factor features for {symbol}: {e}")
                    continue

            # 合并所有特征
            if all_features:
                combined_features = pd.concat(all_features, axis=0)
                combined_features.sort_index(inplace=True)

                logger.info(f"Factor features created: {combined_features.shape}")
                logger.info(f"Factor columns: {list(combined_features.columns)}")
                return combined_features
            else:
                logger.warning("No factor features created")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Failed to create factor features: {e}")
            return pd.DataFrame()

    def _handle_nan_values(self, features: pd.DataFrame, use_learned_stats: bool = False) -> pd.DataFrame:
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
        if use_learned_stats and hasattr(self, 'nan_fill_values') and not self.nan_fill_values.empty:
            # Use learned statistics for backtesting
            logger.debug("Using learned NaN fill statistics")
            features_clean = features_clean.fillna(self.nan_fill_values)
        else:
            # Use current data statistics for training
            logger.debug("Using current data NaN fill statistics")
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