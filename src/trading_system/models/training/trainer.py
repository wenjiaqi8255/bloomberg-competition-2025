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
from ..utils.performance_evaluator import PerformanceEvaluator
from ...utils.experiment_tracking import ExperimentTrackerInterface
from .types import TrainingConfig, TrainingResult
from .experiment_logger import ExperimentLogger

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified trainer for all ML models.

    This trainer handles the complete training workflow while keeping
    concerns properly separated:
    - Model logic stays in the model
    - Training orchestration stays here
    - Validation uses TimeSeriesCV
    - Performance evaluation uses a dedicated evaluator
    """

    def __init__(self,
                 config: Optional[TrainingConfig] = None,
                 cv: Optional[TimeSeriesCV] = None,
                 logger: Optional[ExperimentLogger] = None,
                 performance_evaluator: Optional[PerformanceEvaluator] = None):
        """
        Initialize the trainer.

        Args:
            config: Training configuration
            cv: Cross-validation instance (created if None)
            logger: Logger for experiment tracking (created if None)
            performance_evaluator: Evaluator for model performance (created if None)
        """
        self.config = config or TrainingConfig()
        self.cv = cv or TimeSeriesCV()
        self.evaluator = performance_evaluator or PerformanceEvaluator()
        self.logger = logger or ExperimentLogger()

    def train(self,
              model: BaseModel,
              X: pd.DataFrame,
              y: pd.Series,
              X_test: Optional[pd.DataFrame] = None,
              y_test: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train a model with the specified configuration.

        This is a simplified entry point that delegates to the main training
        loop and optionally logs the results if configured.

        Args:
            model: Model to train
            X: Training features
            y: Training targets
            X_test: Optional test features
            y_test: Optional test targets

        Returns:
            TrainingResult with comprehensive information
        """
        logger.info(f"Starting training for {model.model_type}")
        start_time = time.time()

        result = self._perform_training(model, X, y, X_test, y_test)
        
        training_time = time.time() - start_time
        result.training_time = training_time

        if self.config.log_experiment:
            self.logger.init_run(model, X, self.config)
            if result.cv_results:
                self.logger.log_cv_summary(result.cv_results)
            if result.validation_metrics:
                self.logger.log_metrics(result.validation_metrics, split_name="training")
            if result.test_metrics:
                self.logger.log_metrics(result.test_metrics, split_name="test")
            self.logger.finish_run(training_time)

        logger.info(f"Training completed in {result.training_time:.2f} seconds")
        return result

    def train_with_tracking(self,
                           model: BaseModel,
                           X: pd.DataFrame,
                           y: pd.Series,
                           experiment_config: Optional[Dict[str, Any]] = None,
                           X_test: Optional[pd.DataFrame] = None,
                           y_test: Optional[pd.Series] = None) -> TrainingResult:
        """
        Train a model with comprehensive experiment tracking.

        Args:
            model: Model to train
            X: Training features
            y: Training targets
            experiment_config: Configuration for the experiment
            X_test: Optional test features
            y_test: Optional test targets

        Returns:
            TrainingResult with comprehensive information
        """
        logger.info(f"Starting tracked training for {model.model_type}")
        total_start_time = time.time()

        self.logger.init_run(model, X, self.config, experiment_config)

        try:
            self.logger.log_data_statistics(X, y)

            result = self._perform_training(model, X, y, X_test, y_test, track_cv=True)

            total_training_time = time.time() - total_start_time
            result.training_time = total_training_time

            self.logger.log_metrics(result.validation_metrics, split_name="training")
            if result.test_metrics:
                self.logger.log_metrics(result.test_metrics, split_name="test")

            self.logger.log_model_information(model)
            self.logger.finish_run(total_training_time)

            logger.info(f"Tracked training completed in {total_training_time:.2f} seconds")
            return result

        except Exception as e:
            self.logger.log_failure(model, e)
            raise RuntimeError(f"Training failed: {e}")

    def _perform_training(self,
                          model: BaseModel,
                          X: pd.DataFrame,
                          y: pd.Series,
                          X_test: Optional[pd.DataFrame] = None,
                          y_test: Optional[pd.Series] = None,
                          track_cv: bool = False) -> TrainingResult:
        """
        Core internal training loop.

        Args:
            model: Model to train
            X: Training features
            y: Training targets
            X_test: Optional test features
            y_test: Optional test targets
            track_cv: Whether to log detailed CV progress

        Returns:
            TrainingResult object (without training_time set)
        """
        self._validate_training_data(X, y)
        model.validate_data(X, y)

        training_history = []
        cv_results = None

        if self.config.use_cross_validation:
            logger.info("Performing cross-validation...")
            cv_method = self._cross_validate_with_tracking if track_cv else self._cross_validate
            cv_results = cv_method(model, X, y)
            training_history.append({'stage': 'cross_validation', **cv_results})

        logger.info("Training final model...")
        try:
            model.fit(X, y)
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            raise RuntimeError(f"Training failed: {e}")

        validation_metrics = self._calculate_metrics(model, X, y, split_name="training")

        test_metrics = None
        if X_test is not None and y_test is not None:
            test_metrics = self._calculate_metrics(model, X_test, y_test, split_name="test")

        return TrainingResult(
            model=model,
            training_time=0.0,  # Will be set by the caller
            cv_results=cv_results,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            training_history=training_history
        )

    def _log_data_statistics(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Log data statistics to experiment tracker."""
        self.logger.log_data_statistics(X, y)

    def _cross_validate_with_tracking(self,
                                     model: BaseModel,
                                     X: pd.DataFrame,
                                     y: pd.Series) -> Dict[str, Any]:
        """
        Perform cross-validation with experiment tracking for each fold.
        """
        cv_scores = []
        fold_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.cv.split(X)):
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

            fold_model = self._create_model_copy(model)
            fold_model.fit(X_train_fold, y_train_fold)

            fold_metrics = self._calculate_metrics(
                fold_model, X_val_fold, y_val_fold, split_name=f"fold_{fold_idx}"
            )
            fold_results.append(fold_metrics)
            cv_scores.append(fold_metrics.get('r2', 0.0))

            self.logger.log_cv_fold(fold_idx, fold_metrics)

        cv_result = {
            'mean_r2': np.mean(cv_scores) if cv_scores else 0.0,
            'std_r2': np.std(cv_scores) if cv_scores else 0.0,
            'fold_results': fold_results,
            'cv_scores': cv_scores
        }

        self.logger.log_cv_summary(cv_result)

        logger.info(f"Cross-validation R²: {cv_result['mean_r2']:.4f} ± {cv_result['std_r2']:.4f}")
        return cv_result

    def _log_model_information(self, model: BaseModel) -> None:
        """Log model-specific information to experiment tracker."""
        self.logger.log_model_information(model)

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
        Calculate performance metrics using the dedicated evaluator.

        Args:
            model: Trained model
            X: Features
            y: True targets
            split_name: Name for this data split

        Returns:
            Dictionary of metrics
        """
        try:
            logger.info(f"Calculating metrics for {split_name} split...")
            
            # Note: evaluate_model returns more metrics than before.
            # We can filter them here if needed, or just return all of them.
            # For now, returning all seems fine.
            metrics = self.evaluator.evaluate_model(model, X, y)

            # Filter to only include metrics relevant to the old implementation if necessary
            # For example: metrics = {k: v for k, v in metrics.items() if k in ['r2', 'mse', 'mae', 'rmse', 'ic']}

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
