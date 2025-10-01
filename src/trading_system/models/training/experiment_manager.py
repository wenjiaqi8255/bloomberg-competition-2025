"""
Training Experiment Manager

This module provides a high-level experiment management interface for
training ML models with comprehensive experiment tracking. It orchestrates
the complete training experiment lifecycle while separating concerns:

- Manager: Experiment-level orchestration
- Trainer: Training-level orchestration
- Tracker: Data recording
- Visualizer: Chart generation

Key Features:
- Complete experiment lifecycle management
- Automatic artifact saving and logging
- Comprehensive visualization generation
- Model registration and versioning
- Error handling and recovery
"""

import logging
import pickle
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import pandas as pd
from dataclasses import dataclass

from ..base.base_model import BaseModel
from .trainer import ModelTrainer, TrainingConfig, TrainingResult
from ...utils.experiment_tracking import (
    ExperimentTrackerInterface,
    ExperimentConfig,
    ExperimentVisualizer
)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentMetadata:
    """Metadata for training experiments."""
    experiment_name: str
    run_id: str
    model_type: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    dataset_info: Dict[str, Any]
    timestamp: datetime
    status: str = "running"


class TrainingExperimentManager:
    """
    High-level manager for training experiments.

    This class handles the complete experiment lifecycle including
    initialization, training orchestration, artifact management,
    and cleanup. It provides a clean interface for running
    comprehensive training experiments with full tracking.

    Responsibilities:
    - Experiment initialization and configuration
    - Data preparation and validation
    - Training orchestration (delegates to ModelTrainer)
    - Artifact collection and logging
    - Visualization generation
    - Model saving and registration
    - Error handling and cleanup
    """

    def __init__(self,
                 tracker: ExperimentTrackerInterface,
                 artifact_dir: Optional[str] = None,
                 save_models: bool = True):
        """
        Initialize the experiment manager.

        Args:
            tracker: Experiment tracker for logging
            artifact_dir: Directory to save artifacts (auto-created if None)
            save_models: Whether to save trained models
        """
        self.tracker = tracker
        self.save_models = save_models
        self.visualizer = ExperimentVisualizer()

        # Setup artifact directory
        if artifact_dir is None:
            self.artifact_dir = tempfile.mkdtemp(prefix="training_exp_")
        else:
            self.artifact_dir = Path(artifact_dir)
            self.artifact_dir.mkdir(parents=True, exist_ok=True)

        self.current_experiment: Optional[ExperimentMetadata] = None

    def run_training_experiment(self,
                              model: BaseModel,
                              X: pd.DataFrame,
                              y: pd.Series,
                              experiment_config: Optional[ExperimentConfig] = None,
                              training_config: Optional[TrainingConfig] = None,
                              X_test: Optional[pd.DataFrame] = None,
                              y_test: Optional[pd.Series] = None) -> TrainingResult:
        """
        Run a complete training experiment.

        Args:
            model: Model to train
            X: Training features
            y: Training targets
            experiment_config: Experiment configuration
            training_config: Training configuration
            X_test: Optional test features
            y_test: Optional test targets

        Returns:
            TrainingResult with comprehensive information

        Raises:
            ValueError: If inputs are invalid
            RuntimeError: If experiment fails
        """
        logger.info(f"Starting training experiment: {model.model_type}")

        # Use provided config or create default
        experiment_config = experiment_config or self._create_default_experiment_config(model)
        training_config = training_config or TrainingConfig()

        # Initialize experiment
        self._initialize_experiment(experiment_config, model, X, y)

        try:
            # Log dataset information
            self._log_dataset_info(X, y, X_test, y_test)

            # Create trainer with experiment tracking
            trainer = ModelTrainer(
                config=training_config,
                experiment_tracker=self.tracker
            )

            # Run training with comprehensive tracking
            result = trainer.train_with_tracking(
                model=model,
                X=X,
                y=y,
                experiment_config=experiment_config.__dict__,
                X_test=X_test,
                y_test=y_test
            )

            # Save model artifacts
            model_path = None
            if self.save_models:
                model_path = self._save_model_artifact(model, result)

            # Generate and log visualizations
            self._create_and_log_visualizations(result, X, y)

            # Log final summary
            self._log_experiment_summary(result, model_path)

            # Update experiment status
            if self.current_experiment:
                self.current_experiment.status = "completed"

            logger.info(f"Training experiment completed successfully")
            return result

        except Exception as e:
            # Log error and update status
            logger.error(f"Training experiment failed: {e}")
            if self.current_experiment:
                self.current_experiment.status = "failed"

            self.tracker.log_alert(
                title="Experiment Failed",
                text=f"Training experiment failed: {str(e)}",
                level="error"
            )
            self.tracker.finish_run(exit_code=1)
            raise RuntimeError(f"Experiment failed: {e}")

    def _create_default_experiment_config(self, model: BaseModel) -> ExperimentConfig:
        """Create a default experiment configuration."""
        return ExperimentConfig(
            project_name="model-training",
            experiment_name=f"{model.model_type}_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            run_type="training",
            tags=[model.model_type, "automated"],
            hyperparameters=model.config,
            metadata={}
        )

    def _initialize_experiment(self,
                              experiment_config: ExperimentConfig,
                              model: BaseModel,
                              X: pd.DataFrame,
                              y: pd.Series) -> None:
        """Initialize the experiment tracking."""
        # Initialize experiment run
        run_id = self.tracker.init_run(experiment_config)

        # Create experiment metadata
        self.current_experiment = ExperimentMetadata(
            experiment_name=experiment_config.experiment_name,
            run_id=run_id,
            model_type=model.model_type,
            model_config=model.config,
            training_config={},
            dataset_info={
                'samples': len(X),
                'features': len(X.columns),
                'target_mean': float(y.mean()),
                'target_std': float(y.std())
            },
            timestamp=datetime.now()
        )

        # Log experiment metadata
        self.tracker.log_params({
            'experiment_metadata': self.current_experiment.__dict__
        })

        logger.info(f"Initialized experiment: {experiment_config.experiment_name}")

    def _log_dataset_info(self,
                         X: pd.DataFrame,
                         y: pd.Series,
                         X_test: Optional[pd.DataFrame] = None,
                         y_test: Optional[pd.Series] = None) -> None:
        """Log comprehensive dataset information."""
        # Training data info
        training_info = {
            'dataset_shape': X.shape,
            'feature_count': len(X.columns),
            'sample_count': len(X),
            'target_mean': float(y.mean()),
            'target_std': float(y.std()),
            'target_range': [float(y.min()), float(y.max())],
            'missing_values': int(X.isnull().sum().sum()),
            'feature_types': X.dtypes.value_counts().to_dict()
        }

        # Test data info if provided
        if X_test is not None and y_test is not None:
            test_info = {
                'test_dataset_shape': X_test.shape,
                'test_sample_count': len(X_test),
                'test_target_mean': float(y_test.mean()),
                'test_target_std': float(y_test.std()),
                'test_missing_values': int(X_test.isnull().sum().sum())
            }
            training_info.update(test_info)

        # Log as parameters
        self.tracker.log_params(training_info)

        # Log sample of the data as table
        sample_data = pd.concat([
            X.head(10),
            y.head(10)
        ], axis=1)
        sample_data.columns = list(X.columns) + ['target']
        self.tracker.log_table(sample_data, "data_sample")

        logger.info("Logged dataset information")

    def _save_model_artifact(self, model: BaseModel, result: TrainingResult) -> str:
        """Save the trained model as an artifact."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model.model_type}_model_{timestamp}.pkl"
        model_path = self.artifact_dir / model_filename

        # Save the model
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Log model as artifact
        self.tracker.log_artifact(
            artifact_path=str(model_path),
            artifact_name=f"trained_model_{model.model_type}",
            artifact_type="model",
            description=f"Trained {model.model_type} model"
        )

        # Also log model metadata
        model_metadata = {
            'model_type': model.model_type,
            'model_config': model.config,
            'training_metrics': result.validation_metrics,
            'model_path': str(model_path),
            'timestamp': timestamp
        }

        self.tracker.log_table(pd.DataFrame([model_metadata]), "model_info")

        logger.info(f"Saved model artifact: {model_path}")
        return str(model_path)

    def _create_and_log_visualizations(self,
                                     result: TrainingResult,
                                     X: pd.DataFrame,
                                     y: pd.Series) -> None:
        """Create and log comprehensive visualizations."""
        try:
            visualizations = []

            # 1. Feature importance (if available)
            if hasattr(result.model, 'get_feature_importance'):
                importance = result.model.get_feature_importance()
                if importance is not None:
                    fig = self.visualizer.create_feature_importance(importance, top_n=20)
                    if fig:
                        self.tracker.log_figure(fig, "feature_importance")
                        visualizations.append("feature_importance")

            # 2. Training history (if available)
            if result.training_history:
                metrics_history = {}
                for entry in result.training_history:
                    if 'stage' in entry:
                        continue
                    for metric, value in entry.items():
                        if metric not in metrics_history:
                            metrics_history[metric] = []
                        metrics_history[metric].append(value)

                if metrics_history:
                    fig = self.visualizer.create_training_curve(metrics_history, "Training History")
                    if fig:
                        self.tracker.log_figure(fig, "training_history")
                        visualizations.append("training_history")

            # 3. CV results (if available)
            if result.cv_results and 'cv_scores' in result.cv_results:
                cv_scores = result.cv_results['cv_scores']
                cv_data = pd.DataFrame({
                    'fold': range(len(cv_scores)),
                    'r2_score': cv_scores
                })

                # Create simple training curve for CV results
                fig = self.visualizer.create_training_curve(
                    {'cv_r2': cv_scores},
                    "Cross-Validation Results"
                )
                if fig:
                    self.tracker.log_figure(fig, "cv_results")
                    visualizations.append("cv_results")

            # 4. Data distribution (target variable)
            target_dist = pd.DataFrame({'target': y})
            try:
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.hist(target_dist['target'], bins=50, alpha=0.7, edgecolor='black')
                ax.set_title('Target Variable Distribution')
                ax.set_xlabel('Target Value')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                plt.tight_layout()

                self.tracker.log_figure(fig, "target_distribution")
                visualizations.append("target_distribution")
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to create target distribution plot: {e}")

            logger.info(f"Created {len(visualizations)} visualizations")

        except Exception as e:
            logger.warning(f"Failed to create visualizations: {e}")

    def _log_experiment_summary(self, result: TrainingResult, model_path: Optional[str]) -> None:
        """Log final experiment summary."""
        summary = {
            'experiment_status': 'completed',
            'model_type': result.model.model_type,
            'training_time_seconds': result.training_time,
            'validation_r2': result.validation_metrics.get('r2', 0.0) if result.validation_metrics else 0.0,
            'validation_ic': result.validation_metrics.get('ic', 0.0) if result.validation_metrics else 0.0,
            'cv_mean_r2': result.cv_results.get('mean_r2', 0.0) if result.cv_results else 0.0,
            'cv_std_r2': result.cv_results.get('std_r2', 0.0) if result.cv_results else 0.0,
            'model_saved': model_path is not None,
            'artifact_count': 1 if model_path else 0
        }

        # Add test metrics if available
        if result.test_metrics:
            summary.update({
                f'test_{k}': v for k, v in result.test_metrics.items()
            })

        # Log final summary
        self.tracker.log_metrics(summary)
        self.tracker.log_alert(
            title="Experiment Completed",
            text=f"Training experiment completed successfully for {result.model.model_type}. "
                 f"R²: {summary['validation_r2']:.4f}, IC: {summary['validation_ic']:.4f}",
            level="info"
        )

        logger.info(f"Logged experiment summary: {summary}")

    def get_experiment_history(self) -> List[ExperimentMetadata]:
        """Get history of experiments run by this manager."""
        # Note: This is a simplified implementation
        # In a real system, this would query a database or file storage
        if self.current_experiment:
            return [self.current_experiment]
        return []

    def cleanup(self) -> None:
        """Cleanup resources and finish any active runs."""
        try:
            if hasattr(self.tracker, 'is_active') and self.tracker.is_active():
                self.tracker.finish_run()
        except Exception as e:
            logger.warning(f"Failed to cleanup experiment tracking: {e}")

        # Clean up temporary files if created
        if self.artifact_dir and self.artifact_dir.name.startswith("training_exp_"):
            try:
                import shutil
                shutil.rmtree(self.artifact_dir)
                logger.info(f"Cleaned up temporary artifact directory: {self.artifact_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup artifact directory: {e}")