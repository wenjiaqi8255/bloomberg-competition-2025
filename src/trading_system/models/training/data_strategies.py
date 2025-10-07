"""
Data Processing Strategies for Different Model Types

This module implements the Strategy pattern to handle different data formats
required by various model types while maintaining SOLID principles.

Design Principles:
- Single Responsibility: Each strategy handles one data format type
- Open/Closed: Easy to add new model types without modifying existing code
- Dependency Inversion: TrainingPipeline depends on abstraction, not concrete implementations
- KISS: Simple, clear logic for each strategy
- DRY: No duplicate data format conversion code
"""

import logging
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class DataProcessingStrategy(ABC):
    """
    Abstract base class for data processing strategies.

    Each strategy knows how to align and slice data for its specific
    model type requirements.
    """

    @abstractmethod
    def align_and_slice_data(
        self,
        features: pd.DataFrame,
        targets: Dict[str, pd.Series],
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align features with targets and slice to the specified date range.

        Args:
            features: DataFrame with MultiIndex (either symbol/date or date/symbol)
            targets: Dictionary mapping symbols to target Series
            start_date: Start date for training data
            end_date: End date for training data

        Returns:
            Tuple of (aligned_features, aligned_target)
        """
        pass

    @abstractmethod
    def validate_input_format(self, features: pd.DataFrame) -> bool:
        """
        Validate that features are in the expected format for this strategy.

        Args:
            features: DataFrame to validate

        Returns:
            True if format is correct, False otherwise
        """
        pass

    @abstractmethod
    def get_expected_index_order(self) -> Tuple[str, str]:
        """
        Get the expected index order for this strategy.

        Returns:
            Tuple of (level_0_name, level_1_name)
        """
        pass


class TimeSeriesDataStrategy(DataProcessingStrategy):
    """
    Data processing strategy for time series models.

    Expects and works with (symbol, date) index order.
    Suitable for models that process each symbol independently.
    """

    def align_and_slice_data(
        self,
        features: pd.DataFrame,
        targets: Dict[str, pd.Series],
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align and slice data for time series models.

        Works with (symbol, date) index order.
        """
        logger.debug("Using TimeSeriesDataStrategy for data processing")

        # Validate input format
        if not self.validate_input_format(features):
            logger.warning("Features not in expected (symbol, date) format, attempting to convert")
            features = self._ensure_symbol_date_format(features)

        # Adjust end_date for forward return window (21 days)
        adjusted_end_date = end_date - timedelta(days=21)

        # Slice features to the training period (symbol, date format)
        features_in_period = self._slice_features_time_series(features, start_date, adjusted_end_date)

        # Prepare targets in (symbol, date) format
        y_full = self._prepare_targets_time_series(targets)

        # Merge features and targets
        aligned_data = features_in_period.merge(
            y_full.rename('target'),
            left_index=True,
            right_index=True,
            how='inner'
        )

        if aligned_data.empty:
            raise ValueError("No overlapping data found between features and targets after alignment")

        # Clean data
        aligned_data = self._clean_data(aligned_data)

        # Split into features and target
        y = aligned_data['target']
        X = aligned_data.drop(columns=['target'])

        logger.info(f"Time series data alignment complete: {len(X)} samples, {len(X.columns)} features")
        return X, y

    def validate_input_format(self, features: pd.DataFrame) -> bool:
        """Check if features are in (symbol, date) format."""
        if not isinstance(features.index, pd.MultiIndex):
            return False

        index_names = features.index.names
        return index_names[0] == 'symbol' and index_names[1] == 'date'

    def get_expected_index_order(self) -> Tuple[str, str]:
        """Return expected index order for time series."""
        return ('symbol', 'date')

    def _ensure_symbol_date_format(self, features: pd.DataFrame) -> pd.DataFrame:
        """Convert features to (symbol, date) format if needed."""
        try:
            from feature_engineering.utils.panel_formatter import PanelDataFormatter
            return PanelDataFormatter.ensure_panel_format(
                features,
                index_order=('symbol', 'date'),
                validate=True,
                auto_fix=True
            )
        except Exception as e:
            logger.error(f"Failed to convert features to (symbol, date) format: {e}")
            raise ValueError(f"Cannot convert features to required format: {e}")

    def _slice_features_time_series(
        self,
        features: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Slice features in (symbol, date) format."""
        try:
            return features.loc[(slice(None), slice(start_date, end_date)), :]
        except Exception as e:
            logger.error(f"Failed to slice time series features: {e}")
            raise

    def _prepare_targets_time_series(self, targets: Dict[str, pd.Series]) -> pd.Series:
        """Prepare targets in (symbol, date) format."""
        all_targets = []
        for symbol, target_series in targets.items():
            target_df = target_series.to_frame(name='target')
            target_df['symbol'] = symbol
            target_df.index.name = 'date'
            target_df = target_df.set_index('symbol', append=True).reorder_levels(['symbol', 'date'])
            all_targets.append(target_df)

        if not all_targets:
            raise ValueError("No target data could be processed")

        return pd.concat(all_targets)['target']

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean aligned data by removing NaN values."""
        # Remove rows where target is NaN
        data = data.dropna(subset=['target'])

        # Remove rows where ALL features are NaN (keep partial feature rows)
        feature_columns = [col for col in data.columns if col != 'target']
        data = data.dropna(how='all', subset=feature_columns)

        return data


class PanelDataStrategy(DataProcessingStrategy):
    """
    Data processing strategy for panel data models.

    Expects and works with (date, symbol) index order.
    Suitable for models like Fama-MacBeth that process cross-sections.
    """

    def align_and_slice_data(
        self,
        features: pd.DataFrame,
        targets: Dict[str, pd.Series],
        start_date: datetime,
        end_date: datetime
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Align and slice data for panel data models.

        Works with (date, symbol) index order.
        """
        logger.debug("Using PanelDataStrategy for data processing")

        # Validate input format
        if not self.validate_input_format(features):
            logger.warning("Features not in expected (date, symbol) format, attempting to convert")
            features = self._ensure_date_symbol_format(features)

        # Adjust end_date for forward return window (21 days)
        adjusted_end_date = end_date - timedelta(days=21)

        # Slice features to the training period (date, symbol format)
        features_in_period = self._slice_features_panel(features, start_date, adjusted_end_date)

        # Prepare targets in (date, symbol) format
        y_full = self._prepare_targets_panel(targets)

        # Merge features and targets
        aligned_data = features_in_period.merge(
            y_full.rename('target'),
            left_index=True,
            right_index=True,
            how='inner'
        )

        if aligned_data.empty:
            raise ValueError("No overlapping data found between features and targets after alignment")

        # Clean data
        aligned_data = self._clean_data(aligned_data)

        # Split into features and target
        y = aligned_data['target']
        X = aligned_data.drop(columns=['target'])

        logger.info(f"Panel data alignment complete: {len(X)} samples, {len(X.columns)} features")
        return X, y

    def validate_input_format(self, features: pd.DataFrame) -> bool:
        """Check if features are in (date, symbol) format."""
        if not isinstance(features.index, pd.MultiIndex):
            return False

        index_names = features.index.names
        return index_names[0] == 'date' and index_names[1] == 'symbol'

    def get_expected_index_order(self) -> Tuple[str, str]:
        """Return expected index order for panel data."""
        return ('date', 'symbol')

    def _ensure_date_symbol_format(self, features: pd.DataFrame) -> pd.DataFrame:
        """Convert features to (date, symbol) format if needed."""
        try:
            from feature_engineering.utils.panel_formatter import PanelDataFormatter
            return PanelDataFormatter.ensure_panel_format(
                features,
                index_order=('date', 'symbol'),
                validate=True,
                auto_fix=True
            )
        except Exception as e:
            logger.error(f"Failed to convert features to (date, symbol) format: {e}")
            raise ValueError(f"Cannot convert features to required format: {e}")

    def _slice_features_panel(
        self,
        features: pd.DataFrame,
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Slice features in (date, symbol) format."""
        try:
            logger.debug(f"PanelDataStrategy: Slicing features with shape {features.shape}")
            logger.debug(f"PanelDataStrategy: Features index type: {type(features.index)}")
            if hasattr(features.index, 'names'):
                logger.debug(f"PanelDataStrategy: Features index names: {features.index.names}")
                logger.debug(f"PanelDataStrategy: Expected (date, symbol), actual: {tuple(features.index.names)}")
            logger.debug(f"PanelDataStrategy: Date range: {start_date} to {end_date}")

            if len(features) > 0:
                logger.debug(f"PanelDataStrategy: Sample index: {features.index[:3]}")

                # Check for mixed types in date level and fix if needed
                if hasattr(features.index, 'levels'):
                    date_level = features.index.levels[0]
                    logger.debug(f"PanelDataStrategy: Date level dtype: {date_level.dtype}")
                    if date_level.dtype == 'object':
                        logger.warning(f"PanelDataStrategy: Date level is object type, fixing mixed content...")
                        unique_types = set(type(x).__name__ for x in date_level[:10])
                        logger.warning(f"PanelDataStrategy: Date level content types: {unique_types}")

                        # Convert index to proper datetime format
                        logger.info(f"PanelDataStrategy: Converting date index to datetime...")
                        features = features.copy()
                        features.index = features.index.set_levels(
                            pd.to_datetime(features.index.levels[0]),
                            level=0
                        )
                        logger.info(f"PanelDataStrategy: Date level converted to: {features.index.levels[0].dtype}")

            result = features.loc[(slice(start_date, end_date), slice(None)), :]
            logger.debug(f"PanelDataStrategy: Slice result shape: {result.shape}")
            return result
        except Exception as e:
            logger.error(f"Failed to slice panel features: {e}")
            logger.error(f"Error details: {type(e).__name__}: {e}")
            logger.error(f"Features index type: {type(features.index)}")
            if hasattr(features.index, 'names'):
                logger.error(f"Features index names: {features.index.names}")
            if hasattr(features.index, 'levels'):
                for i, level in enumerate(features.index.levels):
                    level_name = features.index.names[i] if features.index.names[i] else f"Level_{i}"
                    logger.error(f"Level {level_name} dtype: {level.dtype}")
                    sample_values = level[:3] if len(level) > 0 else []
                    logger.error(f"Level {level_name} sample: {sample_values}")
            raise

    def _prepare_targets_panel(self, targets: Dict[str, pd.Series]) -> pd.Series:
        """Prepare targets in (date, symbol) format."""
        all_targets = []
        for symbol, target_series in targets.items():
            target_df = target_series.to_frame(name='target')
            target_df['symbol'] = symbol
            target_df.index.name = 'date'
            target_df = target_df.set_index('symbol', append=True).reorder_levels(['date', 'symbol'])
            all_targets.append(target_df)

        if not all_targets:
            raise ValueError("No target data could be processed")

        return pd.concat(all_targets)['target']

    def _clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean aligned data by removing NaN values."""
        # Remove rows where target is NaN
        data = data.dropna(subset=['target'])

        # Remove rows where ALL features are NaN (keep partial feature rows)
        feature_columns = [col for col in data.columns if col != 'target']
        data = data.dropna(how='all', subset=feature_columns)

        return data


class DataStrategyFactory:
    """
    Factory class for creating appropriate data processing strategies
    based on model type.

    Implements the Factory pattern for strategy selection.
    """

    # Model types that use panel data processing
    PANEL_DATA_MODELS = {
        'fama_macbeth',
        'panel_regression',
        'cross_sectional',
        'panel_ml'
    }

    # Default strategy for unspecified model types
    DEFAULT_STRATEGY = TimeSeriesDataStrategy

    @classmethod
    def create_strategy(cls, model_type: str) -> DataProcessingStrategy:
        """
        Create the appropriate data processing strategy for the given model type.

        Args:
            model_type: The type of model (e.g., 'fama_macbeth', 'xgboost', 'lstm')

        Returns:
            DataProcessingStrategy instance appropriate for the model type
        """
        model_type_lower = model_type.lower()

        if model_type_lower in cls.PANEL_DATA_MODELS:
            logger.debug(f"Creating PanelDataStrategy for model type: {model_type}")
            return PanelDataStrategy()
        else:
            logger.debug(f"Creating TimeSeriesDataStrategy for model type: {model_type}")
            return TimeSeriesDataStrategy()

    @classmethod
    def register_panel_model(cls, model_type: str):
        """
        Register a new model type as requiring panel data processing.

        Args:
            model_type: Model type to register
        """
        cls.PANEL_DATA_MODELS.add(model_type.lower())
        logger.info(f"Registered {model_type} as panel data model")

    @classmethod
    def set_default_strategy(cls, strategy_class: type):
        """
        Set a different default strategy.

        Args:
            strategy_class: Class to use as default strategy
        """
        if not issubclass(strategy_class, DataProcessingStrategy):
            raise ValueError("Default strategy must inherit from DataProcessingStrategy")

        cls.DEFAULT_STRATEGY = strategy_class
        logger.info(f"Set default strategy to {strategy_class.__name__}")