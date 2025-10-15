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

import pandas as pd
import numpy as np

from .trainer import ModelTrainer, TrainingResult, TrainingConfig
from ..base.model_factory import ModelFactory
from ..model_persistence import ModelRegistry
from ...feature_engineering.pipeline import FeatureEngineeringPipeline
from ...experiment_tracking.interface import ExperimentTrackerInterface
from ...experiment_tracking.config import create_training_config

# DataStrategies已删除 - 简化版本不需要复杂的数据策略

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
                 data_strategy: Optional[Any] = None,  # 简化版本，不再依赖DataProcessingStrategy
                 model_config: Optional[Dict[str, Any]] = None,
                 experiment_tracker: Optional[ExperimentTrackerInterface] = None):
        """
        Initialize the training pipeline.

        Args:
            model_type: Type of model to train
            feature_pipeline: The unified feature engineering pipeline.
            config: Training configuration
            registry_path: Path for model registry
            data_strategy: Strategy for data processing (auto-selected if None)
            model_config: Model-specific configuration (e.g., sequence_length for LSTM)
        """
        self.model_type = model_type
        self.feature_pipeline = feature_pipeline
        self.config = config or TrainingConfig()
        self.model_config = model_config or {}
        self.registry = ModelRegistry(registry_path or "./models/")
        self.trainer = ModelTrainer(self.config)

        # 简化版本 - 不再使用复杂的数据策略
        self.data_strategy = None
        logger.info(f"Using simplified data handling for model type: {model_type}")

        # Data providers are configured separately
        self.data_provider = None
        self.factor_data_provider = None
        # Experiment tracker (optional)
        self.experiment_tracker = experiment_tracker

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

        # Initialize experiment tracking run if available
        run_id = None
        if self.experiment_tracker is not None:
            try:
                exp_cfg = create_training_config(
                    project_name="bloomberg-competition",
                    model_type=self.model_type,
                    hyperparameters={
                        "model_config": kwargs.get("model_config", {}),
                        "symbols": symbols,
                        "start_date": str(start_date),
                        "end_date": str(end_date)
                    },
                    tags=["training"],
                    notes="Unified training run from TrainingPipeline"
                )
                run_id = self.experiment_tracker.init_run(exp_cfg)
                self.experiment_tracker.update_run_status("training")
                self.experiment_tracker.log_params(exp_cfg.hyperparameters)
            except Exception as e:
                logger.warning(f"Experiment tracker init failed: {e}")

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
            feature_input_data = {'price_data': price_data, 'factor_data': factor_data }

            # Step 2: Create model and delegate training to trainer with CV
            logger.info("Step 2: Creating model and training with cross-validation...")
            model = ModelFactory.create(self.model_type, config=kwargs.get('model_config'))
            logger.info(f"DEBUG: Created model type: {self.model_type}")
            logger.info(f"DEBUG: Model config: {kwargs.get('model_config', 'None')}")

            # ** KEY CHANGE: Don't fit pipeline here! Pass raw data and unfitted pipeline to trainer
            # The trainer will handle CV and fit the pipeline for each fold independently
            logger.info("Delegating training to ModelTrainer with CV and independent pipeline fitting...")
            training_result = self.trainer.train_with_cv(
                model=model,
                data={
                    'price_data': price_data,
                    'factor_data': factor_data,
                    'target_data': target_data
                },
                feature_pipeline=self.feature_pipeline,  # Pass unfitted pipeline
                date_range=(start_date, end_date)  # Actual training date range
            )

            logger.info(f"DEBUG: Training completed. Training result keys: {training_result.__dict__.keys() if hasattr(training_result, '__dict__') else 'No __dict__'}")
            if hasattr(training_result, 'validation_metrics') and training_result.validation_metrics:
                logger.info(f"DEBUG: Training validation metrics: {training_result.validation_metrics}")
            else:
                logger.warning("DEBUG: No validation metrics found in training result")

            # Log metrics to experiment tracker if available
            if self.experiment_tracker is not None and getattr(training_result, 'validation_metrics', None):
                try:
                    flat_metrics = {}
                    for k, v in training_result.validation_metrics.items():
                        if isinstance(v, (int, float)):
                            flat_metrics[k] = v
                    if flat_metrics:
                        self.experiment_tracker.log_metrics(flat_metrics)
                except Exception as e:
                    logger.warning(f"Experiment tracker log_metrics failed: {e}")

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

                # ** KEY CHANGE: Use the fitted pipeline from training result
                # This ensures the saved pipeline has been properly fitted during CV
                fitted_pipeline = getattr(training_result, 'feature_pipeline', self.feature_pipeline)
                model_id = self.registry.save_model_with_artifacts(
                    model=training_result.model,
                    model_name=model_name,
                    artifacts={
                        'feature_pipeline': fitted_pipeline,  # Use fitted pipeline from CV
                        'training_result': training_result  # Also save the complete training result
                    },
                    tags={
                        'pipeline_version': '4.0.0', # Mark new CV-aware saving format
                        'training_date': datetime.now().isoformat(),
                        'data_period': f"{start_date.date()}_{end_date.date()}",
                        'symbols': ','.join(symbols),
                        'cv_method': self.config.cv_method if hasattr(self.config, 'cv_method') else 'unknown'
                    }
                )

                # Log basic training artifacts/metadata
                if self.experiment_tracker is not None:
                    try:
                        meta = {
                            "model_id": model_id,
                            "model_type": self.model_type,
                            "feature_count": int(features.shape[1]) if isinstance(features, pd.DataFrame) else None,
                            "training_samples": int(getattr(training_result.model.metadata, "training_samples", 0))
                        }
                        self.experiment_tracker.log_artifact_from_dict(meta, artifact_name=f"{model_id}_training_metadata")
                    except Exception as e:
                        logger.warning(f"Experiment tracker log_artifact_from_dict failed: {e}")
            else:
                logger.warning("Model was not successfully trained, skipping registration.")

            # Step 6: Generate report
            logger.info("Step 6: Generating pipeline report...")
            # Get features from the fitted pipeline for reporting
            fitted_pipeline = getattr(training_result, 'feature_pipeline', self.feature_pipeline)
            if fitted_pipeline and hasattr(fitted_pipeline, 'transform'):
                try:
                    # Transform data to get features for reporting
                    report_features = fitted_pipeline.transform(feature_input_data)
                except Exception as e:
                    logger.warning(f"Could not generate features for report: {e}")
                    report_features = None
            else:
                report_features = None
            
            pipeline_result = self._generate_pipeline_report(
                model_id, training_result, price_data, report_features
            )

            logger.info(f"Training pipeline completed successfully. Model ID: {model_id}")

            # Mark run completed
            if self.experiment_tracker is not None:
                try:
                    self.experiment_tracker.update_run_status("completed")
                    self.experiment_tracker.finish_run(exit_code=0)
                except Exception as e:
                    logger.warning(f"Experiment tracker finish_run failed: {e}")
            return pipeline_result

        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            if self.experiment_tracker is not None:
                try:
                    self.experiment_tracker.update_run_status("failed")
                    self.experiment_tracker.log_alert("training_failed", str(e), level="error")
                    self.experiment_tracker.finish_run(exit_code=1)
                except Exception:
                    pass
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
        if self.model_type == "ff5_regression":
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
        Prepare training data using simple MVP approach (no complex data strategy).

        Handles alignment between features and targets, supporting both cross-sectional
        and time series data formats based on model type.
        """
        logger.info(f"Preparing training data using simple MVP approach for {self.model_type}")

        # Simple MVP data preparation - no complex strategy pattern
        if self.model_type in ['lstm_model']:
            return self._prepare_lstm_data(features, target_data, start_date, end_date)
        else:
            return self._prepare_tabular_data(features, target_data, start_date, end_date)

    def _generate_pipeline_report(self,
                                 model_id: Optional[str],
                                 training_result: TrainingResult,
                                 price_data: Dict[str, pd.DataFrame],
                                 features: Optional[pd.DataFrame]) -> Dict[str, Any]:
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
                'feature_count': len(features.columns) if features is not None else 0,
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

    def _prepare_tabular_data(self,
                             features: pd.DataFrame,
                             target_data: Dict[str, pd.Series],
                             start_date: datetime,
                             end_date: datetime) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare tabular data for cross-sectional models (XGBoost, FF5, etc.).
        """
        logger.info("Preparing tabular (cross-sectional) data")

        # Handle MultiIndex features using PanelDataFormatter
        if isinstance(features.index, pd.MultiIndex):
            try:
                from ...feature_engineering.utils.panel_formatter import PanelDataFormatter

                # Standardize to panel format (date, symbol) using PanelDataFormatter
                features_standardized = PanelDataFormatter.ensure_panel_format(
                    features,
                    index_order=('date', 'symbol'),
                    validate=True,
                    auto_fix=True
                )

                # Filter by date range on standardized format
                date_mask = (features_standardized.index.get_level_values('date') >= start_date) & \
                           (features_standardized.index.get_level_values('date') <= end_date)
                features_filtered = features_standardized[date_mask]

                logger.debug(f"Standardized features to panel format: {features_filtered.shape}")

            except ImportError:
                logger.warning("PanelDataFormatter not available, using fallback logic")
                # Fallback to simple date filtering
                features_filtered = features.loc[start_date:end_date]
        else:
            # Filter regular index by date range
            features_filtered = features.loc[start_date:end_date]

        # Prepare target data using standardized panel format
        all_targets = []
        all_features = []

        # Use PanelDataFormatter to standardize target data as well
        try:
            from ...feature_engineering.utils.panel_formatter import PanelDataFormatter

            # Convert target data to panel format for easier merging
            target_records = []
            for symbol, target_series in target_data.items():
                for date, value in target_series.items():
                    target_records.append({
                        'date': pd.to_datetime(date),
                        'symbol': symbol,
                        'target': value
                    })

            if target_records:
                target_panel = pd.DataFrame(target_records)
                target_panel = target_panel.set_index(['date', 'symbol'])
                target_panel = target_panel['target']  # Convert to Series

                # Filter target panel by date range
                target_mask = (target_panel.index.get_level_values('date') >= start_date) & \
                             (target_panel.index.get_level_values('date') <= end_date)
                target_filtered = target_panel[target_mask]

                # Merge features and targets using the panel structure
                merged_data = pd.concat([features_filtered, target_filtered], axis=1, join='inner')

                if not merged_data.empty:
                    X = merged_data.drop(columns=['target'])
                    y = merged_data['target']

                    # Remove any remaining NaN values
                    valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
                    X = X[valid_mask]
                    y = y[valid_mask]

                    logger.info(f"Panel data prepared: {len(X)} samples, {len(X.columns)} features")
                    return X, y
                else:
                    logger.warning("No overlapping data between features and targets")
                    raise ValueError("No valid training data available after alignment")

        except ImportError:
            logger.warning("PanelDataFormatter not available for target data, using fallback logic")

            # Fallback to original logic for backward compatibility
            for symbol, target_series in target_data.items():
                if isinstance(features.index, pd.MultiIndex):
                    # For panel format, extract symbol data
                    try:
                        symbol_features = features_filtered.loc[symbol] if symbol in features_filtered.index.get_level_values('symbol') else pd.DataFrame()
                    except:
                        symbol_features = pd.DataFrame()
                else:
                    # For regular index, assume features are already symbol-specific
                    symbol_features = features_filtered

                if len(symbol_features) == 0:
                    continue

                # Align target with feature dates
                if isinstance(symbol_features.index, pd.MultiIndex):
                    feature_dates = symbol_features.index.get_level_values('date')
                else:
                    feature_dates = symbol_features.index

                target_aligned = target_series.reindex(feature_dates, method='ffill').dropna()
                symbol_features = symbol_features.reindex(target_aligned.index)

                if len(target_aligned) > 0:
                    all_targets.append(target_aligned)
                    all_features.append(symbol_features)

            if all_targets:
                y = pd.concat(all_targets).sort_index()
                X = pd.concat(all_features).sort_index()

                # Remove any remaining NaN values
                valid_mask = ~(X.isnull().any(axis=1) | y.isnull())
                X = X[valid_mask]
                y = y[valid_mask]

                logger.info(f"Tabular data prepared: {len(X)} samples, {len(X.columns)} features")
                return X, y
            else:
                raise ValueError("No valid training data available after alignment")

    def _prepare_lstm_data(self,
                          features: pd.DataFrame,
                          target_data: Dict[str, pd.Series],
                          start_date: datetime,
                          end_date: datetime) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequence data for LSTM models.
        """
        logger.info("Preparing LSTM (time series) data")

        # Filter features by date range
        features_filtered = features.loc[start_date:end_date]

        # For LSTM, we need sequences per symbol
        sequence_length = 30  # Default sequence length
        all_sequences = []
        all_targets = []

        for symbol, target_series in target_data.items():
            # Get symbol-specific features (assuming multi-index)
            if symbol in features_filtered.columns.get_level_values(0):
                symbol_features = features_filtered[symbol]
                target_aligned = target_series.reindex(features_filtered.index).dropna()

                if len(target_aligned) > sequence_length:
                    # Create sequences
                    for i in range(sequence_length, len(target_aligned)):
                        seq_features = symbol_features.iloc[i-sequence_length:i]
                        seq_target = target_aligned.iloc[i]

                        if not seq_features.isnull().any().any() and not np.isnan(seq_target):
                            all_sequences.append(seq_features.values)
                            all_targets.append(seq_target)

        if all_sequences:
            X = np.array(all_sequences)
            y = np.array(all_targets)

            logger.info(f"LSTM data prepared: {X.shape[0]} sequences, {X.shape[1]} timesteps, {X.shape[2]} features")
            return X, y
        else:
            raise ValueError("No valid sequence data available after alignment")