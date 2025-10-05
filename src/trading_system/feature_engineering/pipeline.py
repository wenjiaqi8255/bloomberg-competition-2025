"""
Centralized Feature Engineering Pipeline

This module provides a unified pipeline for feature engineering that ensures
consistency across training, backtesting, and live execution environments.

Key Features:
- Versionable and configurable from a file.
- `fit` method to learn parameters (e.g., for scaling) from training data.
- `transform` method to apply the learned transformations consistently.
- Saves/loads its state to be bundled with a trained model.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import yaml
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

from .models.data_types import FeatureConfig
from .utils.technical_features import TechnicalIndicatorCalculator
from .cache_provider import FeatureCacheProvider
from .local_cache_provider import LocalCacheProvider

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Orchestrates the entire feature engineering process, ensuring consistency.
    """

    def __init__(self, config: FeatureConfig, feature_cache: Optional[FeatureCacheProvider] = None):
        """
        Initialize the pipeline with a feature configuration.

        Args:
            config: A FeatureConfig object detailing which features to compute.
            feature_cache: Optional cache provider for feature caching
        """
        self.config = config
        self.scalers: Dict[str, StandardScaler] = {}
        self._is_fitted = False

        # 使用提供的缓存，或创建默认的本地缓存
        self.feature_cache = feature_cache or LocalCacheProvider()

        # 定义哪些特征是"慢变"的，需要缓存
        self.SLOW_FEATURES = {
            'sma_200', 'sma_50', 'ema_200', 'ema_50',
            'volatility_60d', 'volatility_30d',
            'bb_upper_20', 'bb_lower_20', 'bb_middle_20'
        }

        logger.info(f"Initialized pipeline with {self.feature_cache.__class__.__name__}")

    def fit(self, data: Dict[str, pd.DataFrame]):
        """
        Learn scaling parameters from the training data.

        Args:
            data: A dictionary of DataFrames, expecting keys like 'price_data'.
                   The data should cover the entire training period.
        """
        logger.info("Fitting FeatureEngineeringPipeline...")
        # First, compute the features on the training data to know what to scale
        features = self.transform(data, is_fitting=True)

        # Now, fit scalers for the columns that are configured to be scaled
        if self.config.normalize_features and self.config.normalization_method == 'robust':
            # For now, we'll scale all numeric features when normalization is enabled
            # This could be made more configurable in the future
            pass  # Robust scaling will be handled differently if needed
        elif self.config.normalize_features:
            # Use standard scaling for all numeric features when normalize_features is True
            features_to_scale = features.select_dtypes(include=['number']).columns.tolist()
            for col in features_to_scale:
                if col in features.columns:
                    scaler = StandardScaler()
                    # Reshape is needed as StandardScaler expects 2D array
                    self.scalers[col] = scaler.fit(features[[col]])
                    logger.info(f"Fitted scaler for feature: {col}")
                else:
                    logger.warning(f"Column '{col}' not found in features for scaling.")
        
        self._is_fitted = True
        logger.info("FeatureEngineeringPipeline fitting complete.")

    def transform(self, data: Dict[str, pd.DataFrame], is_fitting: bool = False) -> pd.DataFrame:
        """
        Apply feature engineering and scaling transformations.

        Args:
            data: A dictionary of DataFrames, expecting keys like 'price_data' and optionally 'factor_data'.
            is_fitting: If True, skips scaling as it's done post-transform during the fit phase.

        Returns:
            A DataFrame with all computed features.
        """
        if not data or 'price_data' not in data:
            raise ValueError("Input data dictionary must contain 'price_data'.")

        price_data = data['price_data']

        # Step 1: Compute technical features using TechnicalIndicatorCalculator
        calculator = TechnicalIndicatorCalculator()
        features = self._compute_all_features(price_data, calculator)

        # Step 2: Apply NaN handling strategies
        logger.info("Applying NaN handling strategies...")
        features = self._handle_nan_values(features)

        # Step 3: Add factor data if available
        if 'factor_data' in data and data['factor_data'] is not None:
            logger.info("Merging factor data with features...")
            factor_data = data['factor_data']
            features = self._merge_factor_data(features, factor_data)

        # Step 4: Apply scaling if the pipeline is already fitted
        if self._is_fitted and not is_fitting:
            logger.debug("Applying learned scaling to features...")
            for col, scaler in self.scalers.items():
                if col in features.columns:
                    # Transform returns a 2D array, so we flatten and assign
                    features[col] = scaler.transform(features[[col]])
                else:
                    logger.warning(f"Scaled column '{col}' not found during transform.")

        # Step 5: Add more feature steps here in the future (e.g., PCA, feature selection)

        return features

    def _handle_nan_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply multi-level NaN handling strategies based on feature analysis.

        This method implements the optimal strategies identified by the debugger:
        - Forward fill for time-series features
        - Interpolation for gap filling
        - Median fill for remaining values
        - Drop features with >80% NaN values

        Args:
            features: DataFrame with computed features that may contain NaN values

        Returns:
            DataFrame with NaN values handled appropriately
        """
        if features.empty:
            return features

        logger.info(f"Starting NaN handling on {features.shape[0]} rows, {features.shape[1]} columns")

        # Analyze NaN percentages for each column
        nan_analysis = {}
        for col in features.columns:
            nan_count = features[col].isnull().sum()
            # Ensure we have a scalar value (handle MultiIndex case)
            if isinstance(nan_count, pd.Series):
                nan_count = nan_count.iloc[0] if len(nan_count) > 0 else 0

            nan_pct = (nan_count / len(features)) * 100
            nan_analysis[col] = {'count': nan_count, 'percentage': nan_pct}

        # Identify columns to drop (>98% NaN) - Very permissive threshold for ML models that can handle missing data patterns
        cols_to_drop = [col for col, info in nan_analysis.items() if info['percentage'] > 98]
        if cols_to_drop:
            logger.warning(f"Dropping {len(cols_to_drop)} features with >98% NaN values: {cols_to_drop}")
            features = features.drop(columns=cols_to_drop)
            # Remove from analysis
            for col in cols_to_drop:
                del nan_analysis[col]
        else:
            logger.info("No features exceeded 98% NaN threshold - keeping all features for ML model to handle")

        # Log initial NaN state with detailed breakdown
        initial_total_nan = sum(info['count'] for info in nan_analysis.values())
        logger.info(f"Initial NaN values: {initial_total_nan} across {len(features.columns)} features")

        # Log top 10 features with highest NaN percentages for debugging
        sorted_features = sorted(nan_analysis.items(), key=lambda x: x[1]['percentage'], reverse=True)
        high_nan_features = [f"{col} ({info['percentage']:.1f}%)" for col, info in sorted_features[:10]]
        logger.info(f"Top 10 features by NaN percentage: {high_nan_features}")

        # Apply multi-level NaN handling strategy
        # Step 1: Forward fill (time series appropriate)
        logger.debug("Applying forward fill for time-series features...")
        features_ffill = features.ffill()

        # Step 2: Backward fill for leading NaNs
        logger.debug("Applying backward fill for leading NaNs...")
        features_bfill = features_ffill.bfill()

        # Step 3: Linear interpolation for remaining gaps
        logger.debug("Applying linear interpolation for remaining gaps...")
        features_interp = features_bfill.interpolate(method='linear')

        # Step 4: Median fill for any remaining NaNs (most robust)
        logger.debug("Applying median fill for remaining NaNs...")
        median_values = features_interp.median()
        features_clean = features_interp.fillna(median_values)

        # Verify no NaNs remain
        remaining_nan = features_clean.isnull().sum().sum()
        if remaining_nan > 0:
            logger.warning(f"Still have {remaining_nan} NaN values after all strategies - applying zero fill as last resort")
            features_clean = features_clean.fillna(0)

        # Handle infinite values (XGBoost can't handle inf/-inf)
        logger.info("Checking for infinite values...")
        inf_count = (features_clean == float('inf')).sum().sum() + (features_clean == float('-inf')).sum().sum()
        if inf_count > 0:
            logger.warning(f"Found {inf_count} infinite values - clipping to reasonable range")
            # Clip infinite values to reasonable bounds
            features_clean = features_clean.replace([float('inf'), float('-inf')], [1e10, -1e10])

        # Additional check for extreme values that might cause numerical issues
        logger.info("Checking for extreme values...")
        max_val = features_clean.abs().max().max()
        if max_val > 1e9:
            logger.warning(f"Found extreme values (max abs: {max_val}) - clipping to prevent numerical issues")
            features_clean = features_clean.clip(-1e9, 1e9)

        # Log final state
        final_total_nan = features_clean.isnull().sum().sum()
        nan_reduction = ((initial_total_nan - final_total_nan) / initial_total_nan * 100) if initial_total_nan > 0 else 100

        logger.info(f"NaN handling complete:")
        logger.info(f"  - Initial NaN: {initial_total_nan}")
        logger.info(f"  - Final NaN: {final_total_nan}")
        logger.info(f"  - NaN reduction: {nan_reduction:.1f}%")
        logger.info(f"  - Features dropped: {len(cols_to_drop)}")
        logger.info(f"  - Infinite values fixed: {inf_count}")
        logger.info(f"  - Final shape: {features_clean.shape}")
        logger.info(f"  - Max absolute value: {features_clean.abs().max().max():.2e}")

        return features_clean

    def save(self, file_path: Path):
        """Saves the fitted pipeline (including scalers) to a file."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save a pipeline that has not been fitted yet.")
        logger.info(f"Saving feature pipeline to {file_path}")
        joblib.dump(self, file_path)

    @staticmethod
    def load(file_path: Path) -> 'FeatureEngineeringPipeline':
        """Loads a fitted pipeline from a file."""
        logger.info(f"Loading feature pipeline from {file_path}")
        if not file_path.exists():
            raise FileNotFoundError(f"Feature pipeline file not found at {file_path}")
        return joblib.load(file_path)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'FeatureEngineeringPipeline':
        """Creates a pipeline instance from a dictionary configuration."""
        feature_config = FeatureConfig(**config)
        return cls(feature_config)

    @classmethod
    def from_yaml(cls, file_path: Path) -> 'FeatureEngineeringPipeline':
        """Creates a pipeline instance from a YAML configuration file."""
        logger.info(f"Creating feature pipeline from config: {file_path}")
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Assuming the feature config is under a 'feature_engineering' key
        if 'feature_engineering' not in config_dict:
            raise ValueError(f"Key 'feature_engineering' not found in {file_path}")
            
        return cls.from_config(config_dict['feature_engineering'])
    
    def get_max_lookback(self) -> int:
        """
        Determines the maximum lookback period required by the feature configuration.

        This is crucial for ensuring enough historical data is loaded to prevent
        NaNs at the start of the training period.

        Returns:
            The maximum number of days required for feature calculation.
        """
        max_lookback = 0
        if self.config.momentum_periods:
            max_lookback = max(max_lookback, max(self.config.momentum_periods))
        
        if self.config.volatility_windows:
            max_lookback = max(max_lookback, max(self.config.volatility_windows))
            
        # Add lookbacks from other feature types as they are implemented
        # e.g., technical indicators might have their own windows
        
        # A small buffer is often a good idea
        return max_lookback + 5

    def _compute_all_features(self, price_data: Dict[str, pd.DataFrame],
                              calculator: TechnicalIndicatorCalculator) -> pd.DataFrame:
        """
        Compute all features from price data using optimized calculation order.

        This method implements the optimized feature computation order identified
        by the debugger to minimize NaN propagation and improve data quality.

        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            calculator: Technical indicator calculator instance

        Returns:
            DataFrame with all computed features
        """
        logger.info("Starting optimized feature computation with data quality checkpoints...")
        all_features = []

        for symbol, data in price_data.items():
            logger.debug(f"Processing features for symbol: {symbol}")
            symbol_features = pd.DataFrame(index=data.index)

            # === Data Quality Checkpoint 1: Input Data Validation ===
            self._validate_input_data(data, symbol)

            # === Optimized Feature Computation Order ===
            # Order: Basic indicators -> Momentum -> Volatility -> Advanced technical -> Derived features
            # This minimizes NaN propagation by computing stable features first

            # Step 1: Basic technical indicators (lowest NaN generation) - with caching
            if hasattr(self.config, 'include_technical') and self.config.include_technical:
                logger.debug(f"Step 1: Computing basic technical indicators for {symbol}")

                # Try to compute technical indicators with caching for individual features
                technical_cached = pd.DataFrame(index=data.index)

                # Try to get individual technical features from cache
                for feature_name in ['sma_200', 'sma_50', 'ema_200', 'ema_50', 'rsi_14', 'bb_upper_20', 'bb_lower_20', 'bb_middle_20']:
                    cached_feature = self._get_single_feature_cached(symbol, feature_name, data.index[0], data.index[-1])
                    if cached_feature is not None:
                        technical_cached[feature_name] = cached_feature['value']
                        logger.debug(f"Used cached {feature_name} for {symbol}")

                # Compute remaining features that weren't cached
                if len(technical_cached.columns) < 8:  # If not all features were cached
                    technical = calculator._calculate_technical_for_group(data)
                    # Cache individual features that are in SLOW_FEATURES
                    for feature_name in self.SLOW_FEATURES:
                        if feature_name in technical.columns and feature_name not in technical_cached.columns:
                            self._cache_single_feature(symbol, feature_name, technical[feature_name])

                    # Combine cached and newly computed features
                    symbol_features = pd.concat([symbol_features, technical], axis=1)
                else:
                    # Use all cached features
                    symbol_features = pd.concat([symbol_features, technical_cached], axis=1)

                # Quality checkpoint after basic indicators
                self._quality_checkpoint(symbol_features, f"{symbol}_basic_technical", step=1)

            # Step 2: Volatility features (moderate NaN generation) - with caching
            if hasattr(self.config, 'volatility_windows') and self.config.volatility_windows:
                logger.debug(f"Step 2: Computing volatility features for {symbol}")
                volatility_cached = pd.DataFrame(index=data.index)

                # Try to get volatility features from cache
                for window in self.config.volatility_windows:
                    feature_name = f'volatility_{window}d'
                    cached_feature = self._get_single_feature_cached(symbol, feature_name, data.index[0], data.index[-1])
                    if cached_feature is not None:
                        volatility_cached[feature_name] = cached_feature['value']
                        logger.debug(f"Used cached {feature_name} for {symbol}")

                # Compute remaining volatility features
                missing_windows = [w for w in self.config.volatility_windows
                                 if f'volatility_{w}d' not in volatility_cached.columns]
                if missing_windows:
                    volatility_methods = getattr(self.config, 'volatility_methods', ['std'])
                    volatility = calculator._calculate_volatility_for_group(data, missing_windows, volatility_methods)
                    # Cache individual volatility features
                    for window in missing_windows:
                        feature_name = f'volatility_{window}d'
                        if feature_name in volatility.columns:
                            self._cache_single_feature(symbol, feature_name, volatility[feature_name])

                    symbol_features = pd.concat([symbol_features, volatility], axis=1)
                else:
                    symbol_features = pd.concat([symbol_features, volatility_cached], axis=1)

                # Quality checkpoint after volatility features
                self._quality_checkpoint(symbol_features, f"{symbol}_volatility", step=2)

            # Step 3: Momentum features (highest NaN generation, but computed after volatility)
            if hasattr(self.config, 'momentum_periods'):
                logger.debug(f"Step 3: Computing momentum features for {symbol}")
                momentum_methods = getattr(self.config, 'momentum_methods', ['simple'])
                momentum = calculator._calculate_momentum_for_group(data, self.config.momentum_periods, momentum_methods)
                symbol_features = pd.concat([symbol_features, momentum], axis=1)
                # Quality checkpoint after momentum features
                self._quality_checkpoint(symbol_features, f"{symbol}_momentum", step=3)

            # Step 4: Cross-feature calculations (use all previous features)
            if not symbol_features.empty:
                logger.debug(f"Step 4: Computing cross-features for {symbol}")
                cross_features = self._compute_cross_features(symbol_features, data)
                if not cross_features.empty:
                    symbol_features = pd.concat([symbol_features, cross_features], axis=1)
                    # Final quality checkpoint
                    self._quality_checkpoint(symbol_features, f"{symbol}_final", step=4)

            # Create MultiIndex for this symbol's features (universal feature names)
            symbol_multiindex = pd.MultiIndex.from_product(
                [[symbol], symbol_features.index],
                names=['symbol', 'date']
            )
            symbol_features.index = symbol_multiindex

            all_features.append(symbol_features)

            logger.info(f"Completed feature computation for {symbol}: {len(symbol_features.columns)} features")

        # Combine all features with proper MultiIndex structure
        if all_features:
            combined_features = pd.concat(all_features, axis=0)
            combined_features.sort_index(inplace=True)

            # Remove duplicate columns (can happen during concatenation)
            if combined_features.columns.duplicated().any():
                logger.warning(f"Found {combined_features.columns.duplicated().sum()} duplicate columns, removing duplicates")
                combined_features = combined_features.loc[:, ~combined_features.columns.duplicated()]

            # Final global quality checkpoint
            self._final_quality_checkpoint(combined_features)

            return combined_features
        else:
            logger.warning("No features computed from any symbols")
            return pd.DataFrame()

    def _merge_factor_data(self, features: pd.DataFrame, factor_data: pd.DataFrame) -> pd.DataFrame:
        """
        Merge factor data with features.

        Args:
            features: DataFrame with technical features (MultiIndex: symbol, date)
            factor_data: DataFrame with factor data (index: dates, columns: factors)

        Returns:
            DataFrame with features and factor data merged
        """
        # Check if factor_data is valid (not None, not empty dict, has proper structure)
        if factor_data is None or not isinstance(factor_data, pd.DataFrame):
            logger.info("No factor data available or factor_data is not a DataFrame, skipping merge")
            return features

        if factor_data.empty:
            logger.info("Factor data is empty DataFrame, skipping merge")
            return features

        logger.info(f"Merging factor data with shape {factor_data.shape} into features with shape {features.shape}")

        # Factor data has a simple date index, we need to broadcast it to all symbols
        factor_columns = factor_data.columns.tolist()
        logger.info(f"Available factor columns: {factor_columns}")

        # Resample factor data to daily frequency using forward fill
        # This ensures we maintain daily granularity for features
        features_date_range = features.index.get_level_values('date')
        min_date, max_date = features_date_range.min(), features_date_range.max()

        # Create daily date range
        daily_range = pd.date_range(start=min_date, end=max_date, freq='D')

        # Filter to trading days (weekdays) - approximate by excluding weekends
        trading_days = daily_range[daily_range.dayofweek < 5]

        # Reindex factor data to trading days with forward fill
        # First, ensure factor data index is datetime
        factor_data.index = pd.to_datetime(factor_data.index)

        # Reindex to trading days and forward fill
        daily_factor_data = factor_data.reindex(trading_days, method='ffill')

        # Drop any remaining NaNs at the beginning
        daily_factor_data = daily_factor_data.dropna()

        logger.info(f"Resampled factor data from {factor_data.shape[0]} monthly rows to {daily_factor_data.shape[0]} daily rows")
        logger.info(f"Factor data date range after resampling: {daily_factor_data.index.min().date()} to {daily_factor_data.index.max().date()}")

        # For each symbol in features, add the factor data
        symbols = features.index.get_level_values(0).unique()
        all_features_with_factors = []

        for symbol in symbols:
            # Get features for this symbol
            symbol_features = features.loc[symbol]

            # Align factor data with symbol's dates
            symbol_dates = symbol_features.index

            # Find common dates between symbol dates and daily factor data
            common_dates = symbol_dates.intersection(daily_factor_data.index)

            if len(common_dates) > 0:
                # Get factor data for common dates
                aligned_factor_data = daily_factor_data.loc[common_dates]

                # Reindex symbol_features to only include common dates
                aligned_features = symbol_features.loc[common_dates]

                # Add factor columns to the symbol's features
                for factor_col in factor_columns:
                    aligned_features[factor_col] = aligned_factor_data[factor_col].values

                # Create MultiIndex for this symbol
                symbol_multiindex = pd.MultiIndex.from_product(
                    [[symbol], aligned_features.index],
                    names=['symbol', 'date']
                )
                aligned_features.index = symbol_multiindex
                all_features_with_factors.append(aligned_features)
            else:
                logger.warning(f"No common dates found for symbol {symbol}")
                # Keep original features without factor data
                symbol_multiindex = pd.MultiIndex.from_product(
                    [[symbol], symbol_features.index],
                    names=['symbol', 'date']
                )
                symbol_features.index = symbol_multiindex
                all_features_with_factors.append(symbol_features)

        # Combine all features with factor data
        if all_features_with_factors:
            combined_features = pd.concat(all_features_with_factors, axis=0)
            logger.info(f"After merging factor data: shape {combined_features.shape}")
            logger.info(f"Factor columns added: {factor_columns}")
            return combined_features
        else:
            logger.warning("No features to merge with factor data")
            return features

    def _validate_input_data(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Validate input price data before feature computation.

        Args:
            data: OHLCV DataFrame for a symbol
            symbol: Symbol name for logging
        """
        required_columns = ['Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in data.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")

        # Check for data continuity
        if len(data) < 10:
            logger.warning(f"Very short data series for {symbol}: only {len(data)} rows")

        # Check for excessive NaN values in input
        total_values = len(data) * len(required_columns)
        nan_values = data[required_columns].isnull().sum().sum()
        nan_percentage = (nan_values / total_values) * 100

        if nan_percentage > 20:
            logger.warning(f"High NaN percentage in input data for {symbol}: {nan_percentage:.1f}%")
        elif nan_percentage > 5:
            logger.info(f"Moderate NaN percentage in input data for {symbol}: {nan_percentage:.1f}%")

        logger.debug(f"Input data validation for {symbol}: {len(data)} rows, {nan_percentage:.1f}% NaN")

    def _quality_checkpoint(self, features: pd.DataFrame, context: str, step: int) -> None:
        """
        Quality checkpoint to monitor feature quality at each computation step.

        Args:
            features: DataFrame with computed features
            context: Context name for logging
            step: Step number in computation process
        """
        if features.empty:
            logger.warning(f"Quality checkpoint {step} - {context}: No features computed")
            return

        total_values = len(features) * len(features.columns)
        nan_values = features.isnull().sum().sum()
        nan_percentage = (nan_values / total_values) * 100

        # Find features with high NaN percentages
        problematic_features = []
        for col in features.columns:
            col_nan_count = features[col].isnull().sum()
            # Ensure we have a scalar value (handle MultiIndex case)
            if isinstance(col_nan_count, pd.Series):
                col_nan_count = col_nan_count.iloc[0] if len(col_nan_count) > 0 else 0

            col_nan_pct = (col_nan_count / len(features)) * 100
            if col_nan_pct > 50:
                problematic_features.append((col, col_nan_pct))

        # Log quality metrics
        if nan_percentage > 30:
            logger.warning(f"Quality checkpoint {step} - {context}: High NaN percentage {nan_percentage:.1f}%")
        elif nan_percentage > 15:
            logger.info(f"Quality checkpoint {step} - {context}: Moderate NaN percentage {nan_percentage:.1f}%")
        else:
            logger.debug(f"Quality checkpoint {step} - {context}: Low NaN percentage {nan_percentage:.1f}%")

        # Log problematic features
        if problematic_features:
            logger.warning(f"Quality checkpoint {step} - {context}: {len(problematic_features)} features with >50% NaN:")
            for feature, pct in problematic_features[:5]:  # Log top 5
                logger.warning(f"  - {feature}: {pct:.1f}% NaN")

        # Additional quality metrics
        finite_values = np.isfinite(features.to_numpy()).sum()
        finite_percentage = (finite_values / total_values) * 100
        logger.debug(f"Quality checkpoint {step} - {context}: {finite_percentage:.1f}% finite values")

    def _compute_cross_features(self, features: pd.DataFrame, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        Compute cross-features that combine multiple basic features.

        These features often provide additional predictive power by capturing
        interactions between different technical indicators.

        Args:
            features: DataFrame with basic technical features
            price_data: Original OHLCV data

        Returns:
            DataFrame with cross-features
        """
        cross_features = pd.DataFrame(index=features.index)

        try:
            # Price-to-moving average ratios (if SMAs are available)
            sma_columns = [col for col in features.columns if 'sma_' in col and '_200' not in col]
            if sma_columns and 'Close' in price_data.columns:
                for sma_col in sma_columns[:3]:  # Limit to first 3 to avoid explosion
                    if len(sma_col.split('_')) >= 2:
                        period = sma_col.split('_')[1]
                        ratio_col = f'price_sma_ratio_{period}'
                        cross_features[ratio_col] = price_data['Close'] / features[sma_col]

            # Momentum-volatility interaction (if both are available)
            momentum_cols = [col for col in features.columns if 'momentum_' in col and 'rank' not in col]
            vol_cols = [col for col in features.columns if 'volatility_' in col and 'rank' not in col]

            if momentum_cols and vol_cols:
                # Use shortest period momentum and volatility
                mom_col = min(momentum_cols, key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 999)
                vol_col = min(vol_cols, key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 999)

                cross_features['momentum_vol_ratio'] = features[mom_col] / (features[vol_col] + 1e-8)
                cross_features['momentum_vol_interaction'] = features[mom_col] * features[vol_col]

            # RSI divergence (if RSI and momentum are available)
            rsi_cols = [col for col in features.columns if 'rsi_' in col]
            if rsi_cols and momentum_cols:
                rsi_col = rsi_cols[0]
                mom_col = momentum_cols[0]  # Use first momentum column

                # Simple divergence: high RSI with negative momentum or vice versa
                rsi_normalized = (features[rsi_col] - 50) / 50  # Normalize to [-1, 1]
                momentum_normalized = np.sign(features[mom_col])  # Just use direction

                cross_features['rsi_momentum_divergence'] = rsi_normalized * momentum_normalized

            # Bollinger Band mean reversion strength (if available)
            bb_cols = [col for col in features.columns if 'bb_position' in col]
            if bb_cols:
                bb_pos = features[bb_cols[0]]
                # Distance from neutral position (0.5)
                cross_features['bb_mean_reversion_strength'] = abs(bb_pos - 0.5) * 2

            # Volume-price trend (if volume and price features are available)
            if 'Volume' in price_data.columns and len(features.columns) > 0:
                price_change = price_data['Close'].pct_change()
                volume_ratio = price_data['Volume'] / price_data['Volume'].rolling(20).mean()
                cross_features['volume_price_trend'] = price_change * volume_ratio

            logger.debug(f"Computed {len(cross_features.columns)} cross-features")

        except Exception as e:
            logger.warning(f"Error computing cross-features: {e}")
            return pd.DataFrame(index=features.index)

        return cross_features

    def _final_quality_checkpoint(self, combined_features: pd.DataFrame) -> None:
        """
        Final quality checkpoint on the combined feature set.

        Args:
            combined_features: DataFrame with all features for all symbols
        """
        if combined_features.empty:
            logger.error("Final quality checkpoint: No features computed")
            return

        # Global quality metrics
        total_values = len(combined_features) * len(combined_features.columns)
        nan_values = combined_features.isnull().sum().sum()
        nan_percentage = (nan_values / total_values) * 100

        finite_values = np.isfinite(combined_features.to_numpy()).sum()
        finite_percentage = (finite_values / total_values) * 100

        # Symbol-wise quality analysis
        symbols = combined_features.index.get_level_values('symbol').unique()
        symbol_quality = {}

        for symbol in symbols:
            symbol_data = combined_features.loc[symbol]
            symbol_nan_pct = (symbol_data.isnull().sum().sum() / (len(symbol_data) * len(symbol_data.columns))) * 100
            symbol_quality[symbol] = symbol_nan_pct

        # Log comprehensive quality report
        logger.info("=== FINAL QUALITY CHECKPOINT ===")
        logger.info(f"Total features: {len(combined_features.columns)}")
        logger.info(f"Total data points: {len(combined_features)}")
        logger.info(f"Overall NaN percentage: {nan_percentage:.2f}%")
        logger.info(f"Overall finite percentage: {finite_percentage:.2f}%")

        # Symbol quality summary
        avg_symbol_quality = np.mean(list(symbol_quality.values()))
        logger.info(f"Average symbol quality: {avg_symbol_quality:.2f}% NaN")

        # Problematic symbols
        problematic_symbols = [(sym, pct) for sym, pct in symbol_quality.items() if pct > 25]
        if problematic_symbols:
            logger.warning(f"Found {len(problematic_symbols)} symbols with >25% NaN:")
            for symbol, pct in problematic_symbols[:5]:  # Log top 5
                logger.warning(f"  - {symbol}: {pct:.1f}% NaN")

        # Feature quality summary
        feature_nan_pct = (combined_features.isnull().sum() / len(combined_features)) * 100
        high_nan_features = feature_nan_pct[feature_nan_pct > 30].sort_values(ascending=False)

        if len(high_nan_features) > 0:
            logger.warning(f"Found {len(high_nan_features)} features with >30% NaN:")
            for feature, pct in high_nan_features.head(10).items():  # Log top 10
                logger.warning(f"  - {feature}: {pct:.1f}% NaN")

        # Overall quality assessment
        if nan_percentage < 10 and finite_percentage > 95:
            logger.info("✅ Overall feature quality: EXCELLENT")
        elif nan_percentage < 20 and finite_percentage > 90:
            logger.info("✅ Overall feature quality: GOOD")
        elif nan_percentage < 35 and finite_percentage > 80:
            logger.warning("⚠️  Overall feature quality: ACCEPTABLE")
        else:
            logger.error("❌ Overall feature quality: POOR - Consider reviewing feature engineering")

        logger.info("=== END FINAL QUALITY CHECKPOINT ===")

    # Cache-related methods
    def _get_single_feature_cached(
        self,
        symbol: str,
        feature_name: str,
        start_date: datetime,
        end_date: datetime
    ) -> Optional[pd.DataFrame]:
        """尝试从缓存获取单个特征"""
        # 只缓存慢变特征
        if feature_name not in self.SLOW_FEATURES:
            return None

        return self.feature_cache.get(symbol, feature_name, start_date, end_date)

    def _cache_single_feature(
        self,
        symbol: str,
        feature_name: str,
        data: pd.Series
    ) -> None:
        """缓存单个特征"""
        if feature_name not in self.SLOW_FEATURES:
            return

        # 转换为 DataFrame
        df = data.to_frame(name='value')
        self.feature_cache.set(symbol, feature_name, df)
        logger.debug(f"Cached {feature_name} for {symbol}")

    def clear_cache(self, symbol: Optional[str] = None):
        """清空特征缓存"""
        self.feature_cache.clear(symbol)
        logger.info(f"Feature cache cleared{' for ' + symbol if symbol else ''}")

    def _compute_feature_with_cache(
        self,
        symbol: str,
        data: pd.DataFrame,
        feature_name: str,
        compute_func: callable
    ) -> Optional[pd.DataFrame]:
        """
        使用缓存计算单个特征

        Args:
            symbol: 股票代码
            data: 价格数据
            feature_name: 特征名称
            compute_func: 计算函数，接受data返回DataFrame

        Returns:
            包含特征的DataFrame，或None如果计算失败
        """
        start_date = data.index[0]
        end_date = data.index[-1]

        # 尝试从缓存获取
        cached_feature = self._get_single_feature_cached(symbol, feature_name, start_date, end_date)
        if cached_feature is not None:
            logger.debug(f"Using cached {feature_name} for {symbol}")
            return cached_feature

        # 缓存未命中，计算特征
        try:
            computed_features = compute_func(data)
            if computed_features is not None and not computed_features.empty:
                if feature_name in computed_features.columns:
                    # 缓存这个特征
                    self._cache_single_feature(symbol, feature_name, computed_features[feature_name])
                    logger.debug(f"Computed and cached {feature_name} for {symbol}")
                    return computed_features[[feature_name]]
        except Exception as e:
            logger.warning(f"Failed to compute {feature_name} for {symbol}: {e}")

        return None
