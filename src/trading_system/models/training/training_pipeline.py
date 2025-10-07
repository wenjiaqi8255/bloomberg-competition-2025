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
from ..base.model_factory import ModelFactory
from ..model_persistence import ModelRegistry
from ...feature_engineering.pipeline import FeatureEngineeringPipeline
from ...config.feature import FeatureConfig
from .data_strategies import DataProcessingStrategy, DataStrategyFactory

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
                 registry_path: Optional[str] = None,
                 data_strategy: Optional[DataProcessingStrategy] = None):
        """
        Initialize the training pipeline.

        Args:
            model_type: Type of model to train
            feature_pipeline: The unified feature engineering pipeline.
            config: Training configuration
            registry_path: Path for model registry
            data_strategy: Strategy for data processing (auto-selected if None)
        """
        self.model_type = model_type
        self.feature_pipeline = feature_pipeline
        self.config = config or TrainingConfig()
        self.registry = ModelRegistry(registry_path or "./models/")
        self.trainer = ModelTrainer(self.config)

        # Select appropriate data processing strategy based on model type
        if data_strategy is None:
            self.data_strategy = DataStrategyFactory.create_strategy(model_type)
            logger.info(f"Auto-selected {self.data_strategy.__class__.__name__} for model type: {model_type}")
        else:
            self.data_strategy = data_strategy
            logger.info(f"Using provided {self.data_strategy.__class__.__name__} for model type: {model_type}")

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
            model_name = self.model_type

        try:
            # ** THE FIX: Ensure date arguments are datetime objects **
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)

            # Step 1: Load and prepare data
            # ** THE FIX: Extend the data loading window to accommodate feature lookbacks **
            logger.info("Step 1: Loading data...")
            # Determine the longest lookback period required by the feature pipeline.
            max_lookback = self.feature_pipeline.get_max_lookback()
            extended_start_date = start_date - pd.Timedelta(days=max_lookback * 1.5) # 1.5x buffer for non-trading days

            logger.info(f"Max feature lookback is {max_lookback} days. Extending data loading from {start_date.date()} to {extended_start_date.date()}.")
            
            price_data, factor_data, target_data = self._load_data(
                extended_start_date, end_date, symbols, **kwargs
            )
            
            # Encapsulate data for the feature pipeline
            feature_input_data = {'price_data': price_data, 'factor_data': factor_data}

            # Step 2: Fit the feature pipeline and then transform the data
            logger.info("Step 2: Fitting and transforming features...")
            self.feature_pipeline.fit(feature_input_data)
            features = self.feature_pipeline.transform(feature_input_data)

            # Step 3: Prepare training data
            logger.info("Step 3: Preparing training data...")
            logger.info(f"DEBUG: Features shape: {features.shape}, Features date range: {features.index.min()} to {features.index.max()}")
            logger.info(f"DEBUG: Target data shape: {[len(v) for v in target_data.values()] if isinstance(target_data, dict) else 'N/A'}")
            if isinstance(target_data, dict) and target_data:
                first_symbol = list(target_data.keys())[0]
                logger.info(f"DEBUG: Sample target data for {first_symbol}: shape {target_data[first_symbol].shape}, range {target_data[first_symbol].index.min()} to {target_data[first_symbol].index.max()}")
                logger.info(f"DEBUG: Target {first_symbol} stats: mean={target_data[first_symbol].mean():.6f}, std={target_data[first_symbol].std():.6f}, min={target_data[first_symbol].min():.6f}, max={target_data[first_symbol].max():.6f}")

            X, y = self._prepare_training_data(features, target_data, start_date, end_date)

            logger.info(f"DEBUG: After _prepare_training_data:")
            logger.info(f"DEBUG: X shape: {X.shape}, y shape: {y.shape}")
            logger.info(f"DEBUG: X date range: {X.index.min()} to {X.index.max()}")
            logger.info(f"DEBUG: y date range: {y.index.min()} to {y.index.max()}")
            logger.info(f"DEBUG: y stats: mean={y.mean():.6f}, std={y.std():.6f}, min={y.min():.6f}, max={y.max():.6f}")
            logger.info(f"DEBUG: X has NaN values: {X.isnull().any().any()}, Total NaN count: {X.isnull().sum().sum()}")
            logger.info(f"DEBUG: y has NaN values: {y.isnull().any()}, NaN count: {y.isnull().sum()}")
            logger.info(f"DEBUG: X feature types sample: {dict(list(X.dtypes.head(10).items()))}")

            # DEBUG: Check for any DataFrame columns (shouldn't exist but let's verify)
            problematic_cols = []
            for col in X.columns:
                if isinstance(X[col], pd.DataFrame):
                    problematic_cols.append(col)
                    logger.error(f"DEBUG: Column '{col}' is a DataFrame with shape {X[col].shape}")
                    logger.error(f"DEBUG: Column '{col}' columns: {list(X[col].columns)}")
                    logger.error(f"DEBUG: Column '{col}' index: {X[col].index}")

            if problematic_cols:
                logger.error(f"DEBUG: Found {len(problematic_cols)} DataFrame columns that should be Series: {problematic_cols}")
                # Try to fix the issue by taking the first column if it's a single-column DataFrame
                for col in problematic_cols:
                    if isinstance(X[col], pd.DataFrame) and len(X[col].columns) == 1:
                        logger.info(f"DEBUG: Attempting to fix column '{col}' by extracting first column")
                        X[col] = X[col].iloc[:, 0]
            else:
                logger.info("DEBUG: All columns are properly formatted as Series")

            # Step 4: Create and train model
            logger.info("Step 4: Training model...")
            model = ModelFactory.create(self.model_type, config=kwargs.get('model_config'))
            logger.info(f"DEBUG: Created model type: {self.model_type}")
            logger.info(f"DEBUG: Model config: {kwargs.get('model_config', 'None')}")

            training_result = self.trainer.train(model, X, y)

            logger.info(f"DEBUG: Training completed. Training result keys: {training_result.__dict__.keys() if hasattr(training_result, '__dict__') else 'No __dict__'}")
            if hasattr(training_result, 'validation_metrics') and training_result.validation_metrics:
                logger.info(f"DEBUG: Training validation metrics: {training_result.validation_metrics}")
            else:
                logger.warning("DEBUG: No validation metrics found in training result")

            # Step 5: Register model and artifacts
            logger.info("Step 5: Registering model and artifacts...")
            model_id = None
            if hasattr(training_result.model, 'is_trained') and training_result.model.is_trained:
                # ** THE FIX: Transfer training metrics to model metadata before saving **
                if training_result.validation_metrics:
                    # logger.info(f"DEBUG: Transferring validation metrics to model metadata: {training_result.validation_metrics}")
                    training_result.model.update_metadata(
                        performance_metrics=training_result.validation_metrics
                    )

                # Also store cross-validation results if available
                if training_result.cv_results:
                    # logger.info(f"DEBUG: Storing CV results in model metadata: {training_result.cv_results}")
                    training_result.model.update_metadata(
                        cv_results=training_result.cv_results
                    )

                # Re-apply the fix: Use `save_model_with_artifacts` to bundle the pipeline
                model_id = self.registry.save_model_with_artifacts(
                    model=training_result.model,
                    model_name=model_name,
                    artifacts={
                        'feature_pipeline': self.feature_pipeline,
                        'training_result': training_result  # Also save the complete training result
                    },
                    tags={
                        'pipeline_version': '3.0.0', # Mark new saving format
                        'training_date': datetime.now().isoformat(),
                        'data_period': f"{start_date.date()}_{end_date.date()}",
                        'symbols': ','.join(symbols)
                    }
                )
            else:
                logger.warning("Model was not successfully trained, skipping registration.")

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

    def _prepare_training_data(self,
                              features: pd.DataFrame,
                              target_data: Dict[str, pd.Series],
                              start_date: datetime,
                              end_date: datetime) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare training data using the configured data processing strategy.

        This method delegates to the appropriate strategy based on model type,
        following the Strategy pattern for clean separation of concerns.
        """
        logger.info(f"Preparing training data using {self.data_strategy.__class__.__name__}")
        logger.info(f"Strategy expects index order: {self.data_strategy.get_expected_index_order()}")

        # Delegate data alignment and slicing to the strategy
        X, y = self.data_strategy.align_and_slice_data(
            features=features,
            targets=target_data,
            start_date=start_date,
            end_date=end_date
        )

        logger.info(f"Strategy completed: {len(X)} samples, {len(X.columns)} features prepared")
        return X, y

    def _compute_features(self,
                         price_data: Dict[str, pd.DataFrame],
                         target_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        [DEPRECATED] This method's logic has been moved to FeatureEngineeringPipeline.
        """
        raise NotImplementedError("This method is deprecated. Use FeatureEngineeringPipeline.")


    def _generate_pipeline_report(self,
                                 model_id: Optional[str],
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
        Load a model and its artifacts from the registry.
        """
        loaded = self.registry.load_model_with_artifacts(model_id)
        if loaded:
            model, artifacts = loaded
            # Attach artifacts to the model for easy access
            if 'feature_pipeline' in artifacts:
                model.feature_pipeline = artifacts['feature_pipeline']
                logger.info(f"Attached feature_pipeline artifact to model '{model_id}'")
            return model
        return None