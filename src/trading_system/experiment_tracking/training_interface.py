"""
Training-specific experiment tracking interface.

This module defines specialized interfaces for training experiment tracking,
extending the base experiment tracking interface with training-specific
methods and utilities.

Key Features:
- Training-specific logging methods
- Model lifecycle tracking
- Performance metrics aggregation
- Training progress visualization
- Model artifact management
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

import pandas as pd

from .interface import ExperimentTrackerInterface, ExperimentConfig
from src.trading_system.utils.performance import PerformanceMetrics


@dataclass
class TrainingMetrics:
    """Training-specific metrics."""
    epoch: Optional[int] = None
    step: Optional[int] = None
    loss: Optional[float] = None
    training_score: Optional[float] = None
    validation_score: Optional[float] = None
    learning_rate: Optional[float] = None
    training_time: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[f"training_{key}"] = float(value)
        return result


@dataclass
class ModelLifecycleEvent:
    """Model lifecycle event for tracking."""
    event_type: str  # 'created', 'training_started', 'training_completed', 'saved', 'deployed'
    timestamp: str
    model_id: Optional[str] = None
    model_type: Optional[str] = None
    model_config: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    metadata: Optional[Dict[str, Any]] = None


class TrainingExperimentTrackerInterface(ExperimentTrackerInterface, ABC):
    """
    Abstract interface for training experiment tracking.

    This interface extends the base experiment tracking interface with
    training-specific methods for comprehensive model training tracking.
    """

    @abstractmethod
    def log_training_metrics(self, metrics: TrainingMetrics) -> None:
        """
        Log training-specific metrics.

        Args:
            metrics: Training metrics to log
        """
        pass

    @abstractmethod
    def log_model_lifecycle_event(self, event: ModelLifecycleEvent) -> None:
        """
        Log a model lifecycle event.

        Args:
            event: Model lifecycle event
        """
        pass

    @abstractmethod
    def log_feature_importance(self, importance: Union[Dict[str, float], pd.Series],
                              top_n: int = 20) -> None:
        """
        Log feature importance with visualization.

        Args:
            importance: Feature importance data
            top_n: Number of top features to highlight
        """
        pass

    @abstractmethod
    def log_cross_validation_results(self, cv_results: Dict[str, Any]) -> None:
        """
        Log cross-validation results.

        Args:
            cv_results: Cross-validation results
        """
        pass

    @abstractmethod
    def log_training_progress(self, epoch: int, metrics: Dict[str, float]) -> None:
        """
        Log training progress over epochs.

        Args:
            epoch: Current epoch
            metrics: Metrics for this epoch
        """
        pass

    @abstractmethod
    def create_child_run(self, name: str, config: Optional[Dict[str, Any]] = None) -> 'TrainingExperimentTrackerInterface':
        """
        Create a child run for nested experiments (e.g., hyperparameter trials).

        Args:
            name: Name for the child run
            config: Configuration for the child run

        Returns:
            Child experiment tracker
        """
        pass

    @abstractmethod
    def log_model_artifact(self, model_path: str, model_info: Dict[str, Any]) -> None:
        """
        Log a trained model artifact.

        Args:
            model_path: Path to the saved model
            model_info: Information about the model
        """
        pass

    @abstractmethod
    def log_dataset_info(self, dataset_info: Dict[str, Any]) -> None:
        """
        Log dataset information.

        Args:
            dataset_info: Dataset statistics and information
        """
        pass

    @abstractmethod
    def log_model_comparison(self, models: List[Dict[str, Any]]) -> None:
        """
        Log comparison between multiple models.

        Args:
            models: List of model information to compare
        """
        pass

    @abstractmethod
    def log_training_alert(self, title: str, message: str,
                          level: str = "info", metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a training-specific alert.

        Args:
            title: Alert title
            message: Alert message
            level: Alert level ('info', 'warning', 'error')
            metadata: Additional metadata
        """
        pass


class TrainingMetricsAggregator:
    """
    Utility class for aggregating training metrics across runs.

    This class provides methods to collect, aggregate, and analyze
    training metrics from multiple experiments.
    """

    def __init__(self):
        """Initialize the metrics aggregator."""
        self.metrics_history: List[TrainingMetrics] = []
        self.lifecycle_events: List[ModelLifecycleEvent] = []

    def add_metrics(self, metrics: TrainingMetrics) -> None:
        """Add training metrics to the history."""
        self.metrics_history.append(metrics)

    def add_lifecycle_event(self, event: ModelLifecycleEvent) -> None:
        """Add a lifecycle event to the history."""
        self.lifecycle_events.append(event)

    def get_best_epoch(self, metric: str = "validation_score") -> Optional[TrainingMetrics]:
        """
        Get the epoch with the best score for a given metric.

        Args:
            metric: Metric to optimize

        Returns:
            TrainingMetrics for best epoch, or None if no metrics
        """
        if not self.metrics_history:
            return None

        best_metrics = None
        best_score = float('-inf') if metric.endswith('_score') else float('inf')

        for metrics in self.metrics_history:
            score = getattr(metrics, metric, None)
            if score is None:
                continue

            if metric.endswith('_score'):
                if score > best_score:
                    best_score = score
                    best_metrics = metrics
            else:
                if score < best_score:
                    best_score = score
                    best_metrics = metrics

        return best_metrics

    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all training metrics.

        Returns:
            Dictionary with training summary
        """
        if not self.metrics_history:
            return {}

        summary = {
            'total_epochs': len(self.metrics_history),
            'training_completed': len([e for e in self.lifecycle_events
                                     if e.event_type == 'training_completed']) > 0
        }

        # Aggregate metrics
        metrics_df = pd.DataFrame([m.to_dict() for m in self.metrics_history])
        if not metrics_df.empty:
            numeric_cols = metrics_df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                summary[f'{col}_mean'] = float(metrics_df[col].mean())
                summary[f'{col}_std'] = float(metrics_df[col].std())
                summary[f'{col}_min'] = float(metrics_df[col].min())
                summary[f'{col}_max'] = float(metrics_df[col].max())

        return summary

    def get_lifecycle_timeline(self) -> List[Dict[str, Any]]:
        """
        Get a timeline of model lifecycle events.

        Returns:
            List of lifecycle events as dictionaries
        """
        return [
            {
                'event_type': event.event_type,
                'timestamp': event.timestamp,
                'model_type': event.model_type,
                'metrics': event.metrics
            }
            for event in self.lifecycle_events
        ]


class TrainingExperimentConfig(ExperimentConfig):
    """
    Configuration specifically for training experiments.

    Extends the base experiment configuration with training-specific
    fields and defaults.
    """

    def __init__(self,
                 project_name: str = "model-training",
                 experiment_name: str = "",
                 run_type: str = "training",
                 tags: Optional[List[str]] = None,
                 hyperparameters: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None,
                 run_id: Optional[str] = None,
                 notes: str = "",
                 group: Optional[str] = None,
                 entity: Optional[str] = None,
                 # Training-specific fields
                 model_type: Optional[str] = None,
                 training_config: Optional[Dict[str, Any]] = None,
                 dataset_info: Optional[Dict[str, Any]] = None,
                 save_models: bool = True,
                 track_feature_importance: bool = True,
                 enable_early_stopping: bool = True):
        """
        Initialize training experiment configuration.

        Args:
            project_name: WandB project name
            experiment_name: Name of the experiment
            run_type: Type of run (training, evaluation, etc.)
            tags: List of tags for the experiment
            hyperparameters: Model hyperparameters
            metadata: Additional metadata
            run_id: Specific run ID
            notes: Notes about the experiment
            group: Group for organizing experiments
            entity: WandB entity
            model_type: Type of model being trained
            training_config: Training-specific configuration
            dataset_info: Dataset information
            save_models: Whether to save model artifacts
            track_feature_importance: Whether to track feature importance
            enable_early_stopping: Whether to enable early stopping
        """
        super().__init__(
            project_name=project_name,
            experiment_name=experiment_name,
            run_type=run_type,
            tags=tags or [],
            hyperparameters=hyperparameters or {},
            metadata=metadata or {},
            run_id=run_id,
            notes=notes,
            group=group,
            entity=entity,
            data_info=dataset_info or {}
        )

        self.model_type = model_type
        self.training_config = training_config or {}
        self.save_models = save_models
        self.track_feature_importance = track_feature_importance
        self.enable_early_stopping = enable_early_stopping

        # Add training-specific metadata
        self.metadata.update({
            'save_models': save_models,
            'track_feature_importance': track_feature_importance,
            'enable_early_stopping': enable_early_stopping,
            'model_type': model_type,
            'training_config': training_config
        })


def create_training_config(model_type: str,
                          model_config: Dict[str, Any],
                          training_params: Optional[Dict[str, Any]] = None,
                          dataset_info: Optional[Dict[str, Any]] = None,
                          tags: Optional[List[str]] = None,
                          **kwargs) -> TrainingExperimentConfig:
    """
    Factory function to create a training experiment configuration.

    Args:
        model_type: Type of model being trained
        model_config: Model configuration
        training_params: Training-specific parameters
        dataset_info: Dataset information
        tags: Additional tags for the experiment
        **kwargs: Additional configuration parameters

    Returns:
        TrainingExperimentConfig instance
    """
    # Merge provided tags with default tags
    default_tags = [model_type, "training"]
    if tags:
        all_tags = list(set(default_tags + tags))
    else:
        all_tags = default_tags

    return TrainingExperimentConfig(
        experiment_name=f"{model_type}_training",
        model_type=model_type,
        hyperparameters=model_config,
        training_config=training_params or {},
        dataset_info=dataset_info or {},
        tags=all_tags,
        **kwargs
    )