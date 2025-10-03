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
from ...feature_engineering.pipeline import FeatureEngineeringPipeline
from ...config.feature import FeatureConfig

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
                 feature_pipeline: FeatureEngineeringPipeline,
                 config: Optional[TrainingConfig] = None,
                 registry_path: Optional[str] = None):
        """
        Initialize the training pipeline.

        Args:
            model_type: Type of model to train
            feature_pipeline: The unified feature engineering pipeline.
            config: Training configuration
            registry_path: Path for model registry
        """
        self.model_type = model_type
        self.feature_pipeline = feature_pipeline
        self.config = config or TrainingConfig()
        self.registry = ModelRegistry(registry_path or "./models/")
        self.trainer = ModelTrainer(self.config)

        # Data providers are configured separately
        self.data_provider = None
        self.factor_data_provider = None

    def configure_data(self, data_provider, factor_data_provider=None) -> 'TrainingPipeline':
        """
        Configure data sources for the pipeline.

        Args:
            data_provider: Price data provider instance
            factor_data_provider: Optional factor data provider instance

        Returns:
            Self for method chaining
        """
        self.data_provider = data_provider
        self.factor_data_provider = factor_data_provider
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
            
            # Encapsulate data for the feature pipeline
            feature_input_data = {'price_data': price_data, 'factor_data': factor_data}

            # Step 2: Fit the feature pipeline and then transform the data
            logger.info("Step 2: Fitting and transforming features...")
            self.feature_pipeline.fit(feature_input_data)
            features = self.feature_pipeline.transform(feature_input_data)

            # Step 3: Prepare training data
            logger.info("Step 3: Preparing training data...")
            X, y = self._prepare_training_data(features, target_data)

            # Step 4: Create and train model
            logger.info("Step 4: Training model...")
            model = ModelFactory.create(self.model_type, config=kwargs.get('model_config'))
            training_result = self.trainer.train(model, X, y)

            # Step 5: Register model and artifacts
            logger.info("Step 5: Registering model and artifacts...")
            if hasattr(model, 'status') and model.status == "trained":
                # Save the model with tags
                model_id = self.registry.save_model(
                    model=model,
                    model_name=model_name,
                    tags={
                        'pipeline_version': '2.0.0', # Updated version
                        'training_date': datetime.now().isoformat(),
                        'data_period': f"{start_date}_{end_date}",
                        'symbols': ','.join(symbols)
                    }
                )

                # TODO: Save feature pipeline separately if needed
                # For now, the feature pipeline is saved with the model if the model supports it
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
        price_data = self.data_provider.get_historical_data(symbols, start_date, end_date)

        # Load factor data if needed
        factor_data = {}
        if self.model_type == "ff5_regression" or self.model_type == "residual_predictor":
            if self.factor_data_provider is not None:
                logger.info(f"Loading factor data using {type(self.factor_data_provider).__name__}")
                factor_data = self.factor_data_provider.get_factor_returns(start_date, end_date)
            elif hasattr(self.data_provider, 'get_factor_returns'):
                factor_data = self.data_provider.get_factor_returns(start_date, end_date)
            else:
                logger.warning(f"No factor data provider available. Factor data will be empty.")
                factor_data = {}

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
        [DEPRECATED] This method's logic has been moved to FeatureEngineeringPipeline.
        """
        raise NotImplementedError("This method is deprecated. Use FeatureEngineeringPipeline.")

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
        logger.info(f"Features index type: {type(features.index)}")
        logger.info(f"Features index sample: {features.index[:5].tolist()}")
        logger.info(f"Is MultiIndex: {isinstance(features.index, pd.MultiIndex)}")
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

                    # Align target series with these dates for forward return prediction
                    # target_series contains future returns (shifted -21 days)
                    # We need to align current features with future targets
                    # Find target dates that correspond to our feature dates
                    future_target_dates = symbol_dates + pd.Timedelta(days=21)

                    # Get target values for these future dates
                    future_targets = target_series.reindex(future_target_dates)

                    # Only keep non-NaN targets (where we have future data)
                    valid_mask = future_targets.notna()
                    valid_feature_dates = symbol_dates[valid_mask]
                    valid_targets = future_targets[valid_mask]

                    # Create MultiIndex for the valid pairs
                    aligned_target = pd.Series(
                        valid_targets.values,
                        index=pd.MultiIndex.from_product(
                            [[symbol], valid_feature_dates],
                            names=['symbol', 'date']
                        )
                    )
                    all_targets.append(aligned_target)

            if all_targets:
                y = pd.concat(all_targets).dropna()
                logger.info(f"MultiIndex y shape: {y.shape}")
                logger.info(f"MultiIndex y sample: {y.index[:5].tolist()}")
                logger.info(f"Features index sample: {features.index[:5].tolist()}")

                # Check if indices match
                common_indices = features.index.intersection(y.index)
                logger.info(f"Common indices count: {len(common_indices)}")

                # Debug: Check index details
                logger.info(f"Features index names: {features.index.names}")
                logger.info(f"Y index names: {y.index.names}")
                logger.info(f"Features index levels: {features.index.levels}")
                logger.info(f"Y index levels: {y.index.levels}")

                # Try direct index comparison
                logger.info(f"Index objects equal: {features.index.equals(y.index)}")

                # Use common indices approach - simpler and more reliable
                common_indices = features.index.intersection(y.index)
                logger.info(f"Common indices count: {len(common_indices)}")

                if len(common_indices) > 0:
                    X = features.loc[common_indices]
                    y = y.loc[common_indices]
                    logger.info(f"After common index alignment - X shape: {X.shape}, y shape: {y.shape}")
                else:
                    raise ValueError("No common indices found between features and targets.")
            else:
                raise ValueError("No matching symbols between features and targets")
        else:
            # Use the original approach for single-index features
            all_targets = []
            for symbol, target_series in target_data.items():
                target_series.name = symbol
                all_targets.append(target_series)

            y = pd.concat(all_targets).dropna()

            # For forward return prediction: features at time t predict returns at time t+21
            # Align features(t) with targets(t+21)
            target_dates = y.index
            feature_dates = target_dates - pd.Timedelta(days=21)  # Look back 21 days for features

            # Find matching feature dates
            valid_feature_dates = features.index.intersection(feature_dates)
            valid_target_dates = valid_feature_dates + pd.Timedelta(days=21)

            # Get the corresponding target dates that exist
            final_target_dates = y.index.intersection(valid_target_dates)
            final_feature_dates = final_target_dates - pd.Timedelta(days=21)

            if len(final_feature_dates) > 0:
                X = features.loc[final_feature_dates]
                y = y.loc[final_target_dates]
            else:
                raise ValueError("No matching indices between features and targets after forward alignment")

        # Ensure alignment
        common_index = X.index.intersection(y.index)
        X = X.loc[common_index]
        y = y.loc[common_index]

        logger.info(f"After alignment - training data: {len(X)} samples, {len(X.columns)} features")
        logger.info(f"Target data length: {len(y)}")
        logger.info(f"X and y have same index: {X.index.equals(y.index)}")

        if len(X) != len(y):
            logger.error(f"Length mismatch: X={len(X)}, y={len(y)}")
            logger.error(f"X index type: {type(X.index)}")
            logger.error(f"y index type: {type(y.index)}")
            logger.error(f"X index sample: {X.index[:5] if len(X) > 0 else 'empty'}")
            logger.error(f"y index sample: {y.index[:5] if len(y) > 0 else 'empty'}")

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