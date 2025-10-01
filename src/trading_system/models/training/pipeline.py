"""
Training Pipeline

This module provides end-to-end training pipelines that orchestrate
the complete machine learning workflow from data preparation to
model registration.

Key Features:
- End-to-end workflow orchestration
- Data preparation and feature engineering integration
- Automatic model registration and versioning
- Experiment tracking and logging
- Error handling and recovery
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import pandas as pd

from .trainer import ModelTrainer, TrainingResult, TrainingConfig
from ..base.model_factory import ModelFactory, ModelRegistry
from ...feature_engineering import compute_technical_features, FeatureConfig

logger = logging.getLogger(__name__)


class TrainingPipeline:
    """
    End-to-end training pipeline for ML models.

    This pipeline orchestrates the complete workflow:
    1. Data loading and validation
    2. Feature engineering
    3. Model training (with CV if requested)
    4. Model evaluation and validation
    5. Model registration
    6. Experiment logging
    """

    def __init__(self,
                 model_type: str,
                 config: Optional[TrainingConfig] = None,
                 registry_path: Optional[str] = None):
        """
        Initialize the training pipeline.

        Args:
            model_type: Type of model to train
            config: Training configuration
            registry_path: Path for model registry
        """
        self.model_type = model_type
        self.config = config or TrainingConfig()
        self.registry = ModelRegistry(registry_path or "./models/")
        self.trainer = ModelTrainer(self.config)

        # Initialize data providers (to be configured)
        self.data_provider = None
        self.feature_config = None

    def configure_data(self,
                      data_provider,
                      feature_config: Optional[FeatureConfig] = None) -> 'TrainingPipeline':
        """
        Configure data sources for the pipeline.

        Args:
            data_provider: Data provider instance
            feature_config: Feature engineering configuration

        Returns:
            Self for method chaining
        """
        self.data_provider = data_provider
        self.feature_config = feature_config
        return self

    def run_pipeline(self,
                    start_date: datetime,
                    end_date: datetime,
                    symbols: List[str],
                    model_name: Optional[str] = None,
                    **kwargs) -> Dict[str, Any]:
        """
        Run the complete training pipeline.

        Args:
            start_date: Training start date
            end_date: Training end date
            symbols: List of symbols to train on
            model_name: Optional name for model registration
            **kwargs: Additional pipeline parameters

        Returns:
            Pipeline execution results

        Raises:
            ValueError: If configuration is incomplete
            RuntimeError: If pipeline fails
        """
        logger.info(f"Starting training pipeline for {self.model_type}")

        if not self.data_provider:
            raise ValueError("Data provider must be configured")

        # Generate model name if not provided
        if model_name is None:
            model_name = f"{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        try:
            # Step 1: Load and prepare data
            logger.info("Step 1: Loading data...")
            price_data, factor_data, target_data = self._load_data(
                start_date, end_date, symbols, **kwargs
            )

            # Step 2: Feature engineering
            logger.info("Step 2: Computing features...")
            features = self._compute_features(price_data, target_data)

            # Step 3: Prepare training data
            logger.info("Step 3: Preparing training data...")
            X, y = self._prepare_training_data(features, target_data)

            # Step 4: Create and train model
            logger.info("Step 4: Training model...")
            model = ModelFactory.create(self.model_type, config=kwargs.get('model_config'))
            training_result = self.trainer.train(model, X, y)

            # Step 5: Register model (only if successfully trained)
            logger.info("Step 5: Registering model...")
            if hasattr(model, 'status') and model.status == "trained":
                model_id = self.registry.save_model(
                    model,
                    model_name,
                    tags={
                        'pipeline_version': '1.0.0',
                        'training_date': datetime.now().isoformat(),
                        'data_period': f"{start_date.date()}_{end_date.date()}",
                        'symbols': ','.join(symbols)
                    }
                )
            else:
                logger.warning("Model not trained, skipping registration")
                model_id = None

            # Step 6: Generate report
            logger.info("Step 6: Generating pipeline report...")
            pipeline_result = self._generate_pipeline_report(
                model_id, training_result, price_data, features
            )

            logger.info(f"Training pipeline completed successfully. Model ID: {model_id}")
            return pipeline_result

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise RuntimeError(f"Pipeline failed: {e}")

    def _load_data(self,
                   start_date: datetime,
                   end_date: datetime,
                   symbols: List[str],
                   **kwargs) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, pd.Series]]:
        """
        Load and prepare data for training.

        Args:
            start_date: Start date for data
            end_date: End date for data
            symbols: List of symbols
            **kwargs: Additional loading parameters

        Returns:
            Tuple of (price_data, factor_data, target_data)
        """
        # Load price data
        price_data = self.data_provider.get_price_data(symbols, start_date, end_date)

        # Load factor data if needed
        factor_data = {}
        if self.model_type == "ff5_regression" or self.model_type == "residual_predictor":
            factor_data = self.data_provider.get_factor_data(start_date, end_date)

        # Load/calculate target data (forward returns)
        target_data = {}
        for symbol in symbols:
            if symbol in price_data:
                prices = price_data[symbol]['Close']
                # Calculate forward returns (e.g., 21-day forward return)
                forward_returns = prices.pct_change(21).shift(-21)
                target_data[symbol] = forward_returns.dropna()

        return price_data, factor_data, target_data

    def _compute_features(self,
                         price_data: Dict[str, pd.DataFrame],
                         target_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Compute features for training.

        Args:
            price_data: Price data by symbol
            target_data: Target data by symbol

        Returns:
            Feature DataFrame
        """
        if self.feature_config:
            # Use provided feature configuration
            result = compute_technical_features(price_data, config=self.feature_config)
        else:
            # Use default configuration
            result = compute_technical_features(price_data)

        features = result.features

        # For residual_predictor, we need to add factor features
        if self.model_type == "residual_predictor":
            # Get factor data that was already loaded in _load_data
            factor_data = self.data_provider.get_factor_data(None, None)  # Should return cached data

            if not factor_data.empty:
                # Reindex factor data to match features
                factor_features = factor_data.reindex(features.index)

                # Forward fill factor data (usually monthly, need to fill daily)
                factor_features = factor_features.ffill().bfill()

                # Combine technical features with factor features
                features = pd.concat([features, factor_features], axis=1)
                logger.info(f"Added {len(factor_features.columns)} factor features: {list(factor_features.columns)}")
            else:
                logger.warning("No factor data available for residual_predictor")

        return features

    def _prepare_training_data(self,
                              features: pd.DataFrame,
                              target_data: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data by aligning features and targets.

        Args:
            features: Feature DataFrame
            target_data: Target data by symbol

        Returns:
            Tuple of (X, y) for training
        """
        logger.info(f"Features shape: {features.shape}")
        logger.info(f"Features index sample: {features.index[:5].tolist()}")
        logger.info(f"Available symbols in target data: {list(target_data.keys())}")

        # For debugging, check one symbol's target data
        if target_data:
            first_symbol = list(target_data.keys())[0]
            first_target = target_data[first_symbol]
            logger.info(f"First symbol '{first_symbol}' target shape: {first_target.shape}")
            logger.info(f"First symbol target index sample: {first_target.index[:5].tolist()}")

        # The issue is that features and targets use different indexing schemes
        # Features use a multi-index (symbol-level) while targets are concatenated
        # Let's create proper alignment by rebuilding targets with the same structure as features

        # First, check if features have a multi-index structure (symbol-date)
        if isinstance(features.index, pd.MultiIndex):
            logger.info("Features have MultiIndex, aligning targets accordingly")

            # Rebuild target data to match feature structure
            all_targets = []
            for symbol, target_series in target_data.items():
                if symbol in features.index.get_level_values(0):
                    # Create MultiIndex entries for this symbol
                    symbol_indices = features.index.get_level_values(0) == symbol
                    symbol_dates = features.index[symbol_indices].get_level_values(1)

                    # Align target series with these dates
                    aligned_target = target_series.reindex(symbol_dates)
                    aligned_target.index = pd.MultiIndex.from_product(
                        [[symbol], symbol_dates],
                        names=['symbol', 'date']
                    )
                    all_targets.append(aligned_target)

            if all_targets:
                y = pd.concat(all_targets).dropna()
                X = features.loc[y.index].dropna()
            else:
                raise ValueError("No matching symbols between features and targets")
        else:
            # Use the original approach for single-index features
            all_targets = []
            for symbol, target_series in target_data.items():
                target_series.name = symbol
                all_targets.append(target_series)

            y = pd.concat(all_targets).dropna()
            X = features.loc[y.index].dropna()

        # Ensure alignment
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        logger.info(f"After alignment - training data: {len(X)} samples, {len(X.columns)} features")

        if len(X) == 0:
            raise ValueError("No training samples after alignment. Check data structure and index compatibility.")

        return X, y

    def _generate_pipeline_report(self,
                                 model_id: str,
                                 training_result: TrainingResult,
                                 price_data: Dict[str, pd.DataFrame],
                                 features: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a comprehensive pipeline execution report.

        Args:
            model_id: ID of the trained model
            training_result: Training results
            price_data: Price data used
            features: Features computed

        Returns:
            Pipeline report dictionary
        """
        report = {
            'pipeline_info': {
                'model_id': model_id,
                'model_type': self.model_type,
                'execution_time': datetime.now().isoformat(),
                'training_time': training_result.training_time
            },
            'data_info': {
                'symbols': list(price_data.keys()),
                'price_data_points': sum(len(df) for df in price_data.values()),
                'feature_count': len(features.columns),
                'training_samples': training_result.model.metadata.training_samples
            },
            'training_results': training_result.get_summary(),
            'model_info': {
                'model_config': training_result.model.config,
                'model_metadata': training_result.model.metadata.to_dict(),
                'feature_importance': training_result.model.get_feature_importance()
            }
        }

        # Log to experiment tracking if available
        try:
            import wandb
            if wandb.run:
                wandb.log(report)
                logger.info("Pipeline report logged to Weights & Biases")
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to log pipeline report: {e}")

        return report

    def list_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List all models in the registry.

        Returns:
            Dictionary of model information
        """
        return self.registry.list_models()

    def load_model(self, model_id: str):
        """
        Load a model from the registry.

        Args:
            model_id: Model ID to load

        Returns:
            Loaded model instance
        """
        return self.registry.load_model(model_id)