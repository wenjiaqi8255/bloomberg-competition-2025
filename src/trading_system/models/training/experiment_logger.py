"""
Experiment Logger for Model Training

This module provides a dedicated logger to handle all interactions
with experiment tracking services like Weights & Biases or MLflow.
It decouples the training logic from the logging implementation.
"""
import logging
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np

from ...utils.experiment_tracking import (
    ExperimentTrackerInterface,
    ExperimentConfig,
    NullExperimentTracker
)
from ..base.base_model import BaseModel
from .types import TrainingConfig, TrainingResult

logger = logging.getLogger(__name__)


class ExperimentLogger:
    """Encapsulates all experiment tracking logic for model training."""

    def __init__(self, tracker: Optional[ExperimentTrackerInterface] = None):
        """
        Initialize the logger with an experiment tracker.

        Args:
            tracker: An instance of a class that implements ExperimentTrackerInterface.
        """
        self.tracker = tracker or NullExperimentTracker()

    def init_run(self, model: BaseModel, X: pd.DataFrame,
                 training_config: TrainingConfig, experiment_config: Optional[Dict[str, Any]] = None):
        """
        Initialize a new experiment run.

        Args:
            model: The model being trained.
            X: The training features.
            training_config: The configuration for the training process.
            experiment_config: High-level configuration for the experiment.
        """
        if isinstance(self.tracker, NullExperimentTracker):
            return

        experiment_config = experiment_config or {}
        self.tracker.init_run(
            ExperimentConfig(
                project_name=experiment_config.get('project_name', 'model-training'),
                experiment_name=experiment_config.get('experiment_name', f'{model.model_type}_training'),
                run_type='training',
                tags=experiment_config.get('tags', [model.model_type]),
                hyperparameters=model.config,
                metadata={
                    'model_type': model.model_type,
                    'training_samples': len(X),
                    'feature_count': len(X.columns),
                    'use_cross_validation': training_config.use_cross_validation,
                    'cv_folds': training_config.cv_folds
                }
            )
        )

    def log_data_statistics(self, X: pd.DataFrame, y: pd.Series):
        """Log summary statistics of the training data."""
        if isinstance(self.tracker, NullExperimentTracker):
            return
        
        try:
            stats = {
                'dataset_shape': X.shape,
                'feature_count': len(X.columns),
                'target_mean': float(y.mean()),
                'target_std': float(y.std()),
                'target_min': float(y.min()),
                'target_max': float(y.max()),
                'missing_values_X': int(X.isnull().sum().sum()),
                'missing_values_y': int(y.isnull().sum())
            }

            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                correlations = X[numeric_cols].corrwith(y).abs().describe()
                stats.update({
                    'avg_feature_correlation': float(correlations['mean']),
                    'max_feature_correlation': float(correlations['max']),
                    'min_feature_correlation': float(correlations['min'])
                })
            self.tracker.log_params(stats)
        except Exception as e:
            logger.warning(f"Failed to log data statistics: {e}")

    def log_cv_fold(self, fold_idx: int, fold_metrics: Dict[str, float]):
        """Log metrics for a single cross-validation fold."""
        if isinstance(self.tracker, NullExperimentTracker):
            return
        self.tracker.log_metrics(fold_metrics, step=fold_idx)

    def log_cv_summary(self, cv_results: Dict[str, Any]):
        """Log the summary of the entire cross-validation process."""
        if isinstance(self.tracker, NullExperimentTracker):
            return
        
        cv_scores = cv_results.get('cv_scores', [0])
        self.tracker.log_metrics({
            'cv_mean_r2': cv_results.get('mean_r2', 0.0),
            'cv_std_r2': cv_results.get('std_r2', 0.0),
            'cv_min_r2': np.min(cv_scores),
            'cv_max_r2': np.max(cv_scores)
        })

    def log_metrics(self, metrics: Dict[str, float], split_name: str):
        """Log performance metrics for a given data split (e.g., training, test)."""
        if isinstance(self.tracker, NullExperimentTracker):
            return
        
        step = 1 if split_name == "test" else 0
        self.tracker.log_metrics(metrics, step=step)

    def log_model_information(self, model: BaseModel):
        """Log model-specific information, like feature importance."""
        if isinstance(self.tracker, NullExperimentTracker):
            return

        # Log feature importance if available
        if not (hasattr(model, 'get_feature_importance') and callable(getattr(model, 'get_feature_importance'))):
            return

        try:
            importance = model.get_feature_importance()
            if importance is None:
                return

            if isinstance(importance, dict):
                importance_df = pd.DataFrame([
                    {'feature': name, 'importance': value}
                    for name, value in sorted(importance.items(), key=lambda x: x[1], reverse=True)
                ])
            else:
                importance_df = pd.DataFrame({
                    'feature': importance.index if hasattr(importance, 'index') else range(len(importance)),
                    'importance': importance.values if hasattr(importance, 'values') else importance
                })
            
            self.tracker.log_table(importance_df, "feature_importance")

            from ...utils.experiment_tracking import ExperimentVisualizer
            visualizer = ExperimentVisualizer()
            fig = visualizer.create_feature_importance(importance, top_n=20)
            if fig:
                self.tracker.log_figure(fig, "feature_importance_chart")
        except Exception as e:
            logger.warning(f"Failed to log feature importance: {e}")

    def finish_run(self, training_time: float):
        """
        Finalize and close the experiment run.
        
        Args:
            training_time: The total time taken for training.
        """
        if isinstance(self.tracker, NullExperimentTracker):
            return

        try:
            self.tracker.log_metrics({
                'training_completed': 1,
                'total_training_time': training_time
            })
            self.tracker.finish_run()
        except Exception as e:
            logger.warning(f"Failed to finish experiment tracking cleanly: {e}")

    def log_failure(self, model: BaseModel, error: Exception):
        """Log a training failure."""
        if isinstance(self.tracker, NullExperimentTracker):
            return
        
        try:
            self.tracker.log_alert(
                title="Training Failed",
                text=f"Training failed for {model.model_type}: {str(error)}",
                level="error"
            )
            self.tracker.finish_run(exit_code=1)
        except Exception as e:
            logger.warning(f"Failed to log experiment failure: {e}")

