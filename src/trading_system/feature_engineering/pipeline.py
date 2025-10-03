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
import yaml
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from .models.data_types import FeatureConfig
from .utils.technical_features import TechnicalIndicatorCalculator

logger = logging.getLogger(__name__)


class FeatureEngineeringPipeline:
    """
    Orchestrates the entire feature engineering process, ensuring consistency.
    """

    def __init__(self, config: FeatureConfig):
        """
        Initialize the pipeline with a feature configuration.

        Args:
            config: A FeatureConfig object detailing which features to compute.
        """
        self.config = config
        self.scalers: Dict[str, StandardScaler] = {}
        self._is_fitted = False

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

        # Step 2: Add factor data if available
        if 'factor_data' in data and data['factor_data'] is not None:
            logger.info("Merging factor data with features...")
            factor_data = data['factor_data']
            features = self._merge_factor_data(features, factor_data)

        # Step 3: Apply scaling if the pipeline is already fitted
        if self._is_fitted and not is_fitting:
            logger.debug("Applying learned scaling to features...")
            for col, scaler in self.scalers.items():
                if col in features.columns:
                    # Transform returns a 2D array, so we flatten and assign
                    features[col] = scaler.transform(features[[col]])
                else:
                    logger.warning(f"Scaled column '{col}' not found during transform.")

        # Step 4: Add more feature steps here in the future (e.g., PCA, feature selection)

        return features

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
    
    def _compute_all_features(self, price_data: Dict[str, pd.DataFrame],
                              calculator: TechnicalIndicatorCalculator) -> pd.DataFrame:
        """
        Compute all features from price data using configured feature types.

        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            calculator: Technical indicator calculator instance

        Returns:
            DataFrame with all computed features
        """
        all_features = []

        for symbol, data in price_data.items():
            symbol_features = pd.DataFrame(index=data.index)

            # Compute based on config
            if hasattr(self.config, 'momentum_periods'):
                momentum = calculator.compute_momentum_features(data, self.config.momentum_periods)
                symbol_features = pd.concat([symbol_features, momentum], axis=1)

            if hasattr(self.config, 'volatility_windows'):
                volatility = calculator.compute_volatility_features(data, self.config.volatility_windows)
                symbol_features = pd.concat([symbol_features, volatility], axis=1)

            if hasattr(self.config, 'include_technical') and self.config.include_technical:
                technical = calculator.compute_technical_indicators(data)
                symbol_features = pd.concat([symbol_features, technical], axis=1)

            # Add symbol prefix to feature names and create MultiIndex
            symbol_features.columns = [f"{symbol}_{col}" for col in symbol_features.columns]

            # Create MultiIndex for this symbol's features
            symbol_multiindex = pd.MultiIndex.from_product(
                [[symbol], symbol_features.index],
                names=['symbol', 'date']
            )
            symbol_features.index = symbol_multiindex
            all_features.append(symbol_features)

        # Combine all features with proper MultiIndex structure
        if all_features:
            combined_features = pd.concat(all_features, axis=0)
            return combined_features
        else:
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

        # For each symbol in features, add the factor data
        symbols = features.index.get_level_values(0).unique()
        all_features_with_factors = []

        for symbol in symbols:
            # Get features for this symbol
            symbol_features = features.loc[symbol]

            # Align factor data with symbol's dates
            symbol_dates = symbol_features.index

            # Find common dates between symbol dates and factor data
            common_dates = symbol_dates.intersection(factor_data.index)

            if len(common_dates) > 0:
                # Get factor data for common dates
                aligned_factor_data = factor_data.loc[common_dates]

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
