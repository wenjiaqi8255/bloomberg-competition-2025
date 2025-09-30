"""
Model Trainer

This module provides a unified training interface for all ML models.
It handles the complete training workflow while keeping model logic
separate from training orchestration.

Key Features:
- Separation of training logic from model implementation
- Built-in cross-validation support
- Performance evaluation and reporting
- Early stopping and model selection
- Comprehensive logging and tracking
"""

import logging
import time
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from ..base.base_model import BaseModel, ModelStatus
from ...validation.time_series_cv import TimeSeriesCV
from ...utils.performance import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Training parameters
    use_cross_validation: bool = True
    cv_folds: int = 5
    purge_period: int = 21  # trading days
    embargo_period: int = 5  # trading days

    # Validation
    validation_split: float = 0.2
    early_stopping: bool = True
    early_stopping_patience: int = 10

    # Performance metrics
    metrics_to_compute: List[str] = field(default_factory=lambda: ['r2', 'mse', 'mae'])

    # Experiment tracking
    log_experiment: bool = True
    experiment_name: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")

        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")


@dataclass
class TrainingResult:
    """Results from model training."""
    model: BaseModel
    training_time: float
    cv_results: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    training_history: List[Dict[str, float]] = field(default_factory=list)
    best_params: Optional[Dict[str, Any]] = None

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of training results."""
        return {
            'model_type': self.model.model_type,
            'status': self.model.status,
            'training_time': self.training_time,
            'validation_metrics': self.validation_metrics,
            'test_metrics': self.test_metrics,
            'best_params': self.best_params
        }


class ModelTrainer:
    """
    Unified trainer for all ML models.

    This trainer handles the complete training workflow while keeping
    concerns properly separated:
    - Model logic stays in the model
    - Training orchestration stays here
    - Validation uses TimeSeriesCV
    - Performance evaluation uses dedicated metrics
    """

    def __init__(self,
                 config: Optional[TrainingConfig] = None,
                 cv: Optional[TimeSeriesCV] = None):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            cv: Cross-validation instance (created if None)
        """
        self.config = config or TrainingConfig()
        self.cv = cv or TimeSeriesCV()
        self.performance_calculator = PerformanceMetrics()

    def train(self,
              model: BaseModel,
              X: pd.DataFrame,
              y: pd.Series,
              X_test: Optional[pd.DataFrame] = None,
              y_test: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train a model with the specified configuration.

        Args:
            model: Model to train (must inherit from BaseModel)
            X: Training features
            y: Training targets
            X_test: Optional test features
            y_test: Optional test targets

        Returns:
            TrainingResult with comprehensive information

        Raises:
            ValueError: If data is invalid
            RuntimeError: If training fails
        """
        logger.info(f"Starting training for {model.model_type}")
        start_time = time.time()

        # Validate inputs
        self._validate_training_data(X, y)
        model.validate_data(X, y)

        training_history = []
        cv_results = None

        # Cross-validation if requested
        if self.config.use_cross_validation:
            logger.info("Performing cross-validation...")
            cv_results = self._cross_validate(model, X, y)
            training_history.append({'stage': 'cross_validation', **cv_results})

        # Train final model on full dataset
        logger.info("Training final model...")
        try:
            model.fit(X, y)
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")

        # Calculate validation metrics
        validation_metrics = self._calculate_metrics(
            model, X, y, split_name="training"
        )

        # Calculate test metrics if provided
        test_metrics = None
        if X_test is not None and y_test is not None:
            test_metrics = self._calculate_metrics(
                model, X_test, y_test, split_name="test"
            )

        training_time = time.time() - start_time

        # Log experiment if requested
        if self.config.log_experiment:
            self._log_experiment(model, training_history, cv_results)

        result = TrainingResult(
            model=model,
            training_time=training_time,
            cv_results=cv_results,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            training_history=training_history
        )

        logger.info(f"Training completed in {training_time:.2f} seconds")
        return result

    def _validate_training_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate training data.

        Args:
            X: Feature DataFrame
            y: Target Series

        Raises:
            ValueError: If data is invalid
        """
        if X.empty or y.empty:
            raise ValueError("Training data cannot be empty")

        if len(X) != len(y):
            raise ValueError("X and y must have the same length")

        if len(X) < self.config.cv_folds:
            raise ValueError(f"Not enough data for {self.config.cv_folds}-fold CV")

        # Check for sufficient data after CV splits
        min_samples_per_split = len(X) // self.config.cv_folds
        required_samples = self.config.purge_period + self.config.embargo_period + 10

        if min_samples_per_split < required_samples:
            logger.warning(
                f"Cross-validation may be unreliable. "
                f"Only {min_samples_per_split} samples per split, "
                f"but need at least {required_samples}"
            )

    def _cross_validate(self,
                        model: BaseModel,
                        X: pd.DataFrame,
                        y: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation.

        Args:
            model: Model to validate
            X: Features
            y: Targets

        Returns:
            Cross-validation results
        """
        cv_scores = []
        fold_results = []

        # Perform time series cross-validation
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            # Create a copy of the model for this fold
            fold_model = self._create_model_copy(model)

            # Train the fold model
            fold_model.fit(X_train_fold, y_train_fold)

            # Evaluate on validation set
            fold_metrics = self._calculate_metrics(
                fold_model, X_val_fold, y_val_fold, split_name=f"fold_{fold_idx}"
            )
            fold_results.append(fold_metrics)
            cv_scores.append(fold_metrics.get('r2', 0.0))

        # Aggregate CV results
        cv_result = {
            'mean_r2': np.mean(cv_scores),
            'std_r2': np.std(cv_scores),
            'fold_results': fold_results,
            'cv_scores': cv_scores
        }

        logger.info(f"Cross-validation R²: {cv_result['mean_r2']:.4f} ± {cv_result['std_r2']:.4f}")
        return cv_result

    def _calculate_metrics(self,
                          model: BaseModel,
                          X: pd.DataFrame,
                          y: pd.Series,
                          split_name: str = "default") -> Dict[str, float]:
        """
        Calculate performance metrics.

        Args:
            model: Trained model
            X: Features
            y: True targets
            split_name: Name for this data split

        Returns:
            Dictionary of metrics
        """
        try:
            predictions = model.predict(X)

            metrics = {}

            # Calculate requested metrics
            if 'r2' in self.config.metrics_to_compute:
                metrics['r2'] = r2_score(y, predictions)

            if 'mse' in self.config.metrics_to_compute:
                metrics['mse'] = mean_squared_error(y, predictions)

            if 'mae' in self.config.metrics_to_compute:
                metrics['mae'] = mean_absolute_error(y, predictions)

            # Calculate additional metrics
            metrics['rmse'] = np.sqrt(metrics.get('mse', 0))

            # Information coefficient (correlation)
            if len(predictions) > 1:
                ic = np.corrcoef(y, predictions)[0, 1]
                metrics['ic'] = ic if not np.isnan(ic) else 0.0

            logger.info(f"{split_name} metrics: {metrics}")
            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate metrics for {split_name}: {e}")
            return {}

    def _create_model_copy(self, model: BaseModel) -> BaseModel:
        """
        Create a copy of the model for cross-validation.

        Args:
            model: Original model

        Returns:
            Model copy
        """
        # Use the factory to create a new instance of the same type
        from ..base.model_factory import ModelFactory

        return ModelFactory.create(
            model.model_type,
            config=model.config.copy()
        )

    def _log_experiment(self,
                       model: BaseModel,
                       training_history: List[Dict[str, float]],
                       cv_results: Optional[Dict[str, Any]]) -> None:
        """
        Log experiment information.

        Args:
            model: Trained model
            training_history: Training history
            cv_results: Cross-validation results
        """
        try:
            import wandb

            if wandb.run:
                # Log model parameters
                wandb.config.update({
                    'model_type': model.model_type,
                    'model_config': model.config,
                    'training_config': self.config.__dict__
                })

                # Log metrics
                if cv_results:
                    wandb.log({
                        'cv_mean_r2': cv_results['mean_r2'],
                        'cv_std_r2': cv_results['std_r2']
                    })

                if model.metadata.performance_metrics:
                    wandb.log(model.metadata.performance_metrics)

                logger.info("Experiment logged to Weights & Biases")

        except ImportError:
            logger.debug("wandb not available, skipping experiment logging")
        except Exception as e:
            logger.warning(f"Failed to log experiment: {e}")