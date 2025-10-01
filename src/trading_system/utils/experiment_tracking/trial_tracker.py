"""
Trial-Level Experiment Tracking

This module provides specialized tracking for hyperparameter optimization trials,
including trial metadata, performance tracking, and optimization analysis.

Key Features:
- Trial-specific metadata management
- Trial performance aggregation
- Optimization history tracking
- Trial comparison and analysis
- Hyperparameter impact analysis
- Pruning and failure tracking
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import logging
import time
from datetime import datetime
import json

import pandas as pd
import numpy as np

from .interface import ExperimentTrackerInterface, ExperimentConfig
from .training_interface import TrainingMetrics, ModelLifecycleEvent
from ...utils.performance import PerformanceMetrics

logger = logging.getLogger(__name__)


@dataclass
class TrialMetadata:
    """Metadata for a single optimization trial."""
    trial_number: int
    study_name: str
    parameters: Dict[str, Any]
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"  # "running", "completed", "pruned", "failed"
    score: Optional[float] = None
    objective_value: Optional[float] = None

    # Intermediate results
    intermediate_values: List[Tuple[int, float]] = field(default_factory=list)
    metrics_history: List[Dict[str, float]] = field(default_factory=list)

    # Additional metadata
    trial_config: Optional[Dict[str, Any]] = None
    pruned_step: Optional[int] = None
    failure_reason: Optional[str] = None
    evaluation_time: Optional[float] = None
    model_path: Optional[str] = None

    # Optimization context
    best_so_far: Optional[float] = None
    improvement: Optional[float] = None
    percentile_rank: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "trial_number": self.trial_number,
            "study_name": self.study_name,
            "parameters": self.parameters,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "status": self.status,
            "score": self.score,
            "objective_value": self.objective_value,
            "intermediate_values": self.intermediate_values,
            "metrics_history": self.metrics_history,
            "trial_config": self.trial_config,
            "pruned_step": self.pruned_step,
            "failure_reason": self.failure_reason,
            "evaluation_time": self.evaluation_time,
            "model_path": self.model_path,
            "best_so_far": self.best_so_far,
            "improvement": self.improvement,
            "percentile_rank": self.percentile_rank
        }


@dataclass
class StudyMetadata:
    """Metadata for the entire optimization study."""
    study_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_trials: int = 0
    completed_trials: int = 0
    pruned_trials: int = 0
    failed_trials: int = 0

    # Performance metrics
    best_score: Optional[float] = None
    best_params: Optional[Dict[str, Any]] = None
    worst_score: Optional[float] = None
    mean_score: Optional[float] = None
    std_score: Optional[float] = None

    # Search space info
    search_space: Optional[Dict[str, Any]] = None
    n_parameters: int = 0

    # Configuration
    optimization_config: Optional[Dict[str, Any]] = None
    sampler_type: Optional[str] = None
    pruner_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "study_name": self.study_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_trials": self.total_trials,
            "completed_trials": self.completed_trials,
            "pruned_trials": self.pruned_trials,
            "failed_trials": self.failed_trials,
            "best_score": self.best_score,
            "best_params": self.best_params,
            "worst_score": self.worst_score,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "search_space": self.search_space,
            "n_parameters": self.n_parameters,
            "optimization_config": self.optimization_config,
            "sampler_type": self.sampler_type,
            "pruner_type": self.pruner_type
        }


class TrialTracker:
    """
    Specialized tracker for hyperparameter optimization trials.

    Provides comprehensive trial-level tracking with metadata management,
    performance aggregation, and optimization analysis.
    """

    def __init__(self,
                 base_tracker: ExperimentTrackerInterface,
                 study_name: str,
                 optimization_config: Optional[Dict[str, Any]] = None):
        """
        Initialize trial tracker.

        Args:
            base_tracker: Base experiment tracker for logging
            study_name: Name of the optimization study
            optimization_config: Optimization configuration metadata
        """
        self.base_tracker = base_tracker
        self.study_name = study_name
        self.optimization_config = optimization_config or {}

        # Trial tracking
        self.current_trial: Optional[TrialMetadata] = None
        self.trials_history: List[TrialMetadata] = []
        self.study_metadata = StudyMetadata(
            study_name=study_name,
            start_time=datetime.now(),
            optimization_config=optimization_config
        )

        # Performance tracking
        self.best_score: Optional[float] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.scores_history: List[float] = []

        # Statistics
        self.trial_times: List[float] = []
        self.evaluation_times: List[float] = []

        logger.info(f"TrialTracker initialized for study '{study_name}'")

    def start_trial(self,
                   trial_number: int,
                   parameters: Dict[str, Any],
                   trial_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Start tracking a new trial.

        Args:
            trial_number: Trial number
            parameters: Trial hyperparameters
            trial_config: Additional trial configuration

        Returns:
            Trial run ID
        """
        # Create trial metadata
        self.current_trial = TrialMetadata(
            trial_number=trial_number,
            study_name=self.study_name,
            parameters=parameters.copy(),
            start_time=datetime.now(),
            trial_config=trial_config
        )

        # Create trial experiment config
        experiment_config = ExperimentConfig(
            project_name=f"trial_optimization_{self.study_name}",
            experiment_name=f"trial_{trial_number:04d}",
            run_type="hyperparameter_trial",
            tags=["trial", f"trial_{trial_number}", self.study_name],
            hyperparameters=parameters,
            metadata={
                "trial_number": trial_number,
                "study_name": self.study_name,
                "start_time": self.current_trial.start_time.isoformat(),
                **(trial_config or {})
            }
        )

        # Initialize base tracker run
        run_id = self.base_tracker.init_run(experiment_config)

        # Log trial start
        self.base_tracker.log_params({
            "trial_number": trial_number,
            "study_name": self.study_name,
            "parameters": parameters,
            "start_time": self.current_trial.start_time.isoformat(),
            **(trial_config or {})
        })

        # Update study metadata
        self.study_metadata.total_trials += 1
        if self.study_metadata.n_parameters == 0:
            self.study_metadata.n_parameters = len(parameters)

        logger.info(f"Started trial {trial_number} with run_id: {run_id}")
        return run_id

    def log_intermediate_value(self, step: int, value: float, metrics: Optional[Dict[str, float]] = None) -> None:
        """
        Log intermediate value for current trial.

        Args:
            step: Step number
            value: Intermediate value
            metrics: Additional metrics
        """
        if not self.current_trial:
            logger.warning("No current trial to log intermediate value")
            return

        # Store intermediate value
        self.current_trial.intermediate_values.append((step, value))

        # Log to base tracker
        log_metrics = {
            "intermediate_value": value,
            "step": step
        }
        if metrics:
            log_metrics.update(metrics)

        # Add metrics to history
        if metrics:
            self.current_trial.metrics_history.append({
                "step": step,
                **metrics
            })

        self.base_tracker.log_metrics(log_metrics, step=step)

    def complete_trial(self,
                      score: float,
                      objective_value: Optional[float] = None,
                      metrics: Optional[Dict[str, float]] = None,
                      model_path: Optional[str] = None,
                      evaluation_time: Optional[float] = None) -> None:
        """
        Complete the current trial with results.

        Args:
            score: Final trial score
            objective_value: Objective value (may differ from score)
            metrics: Additional metrics
            model_path: Path to saved model
            evaluation_time: Time taken for evaluation
        """
        if not self.current_trial:
            logger.warning("No current trial to complete")
            return

        # Update trial metadata
        self.current_trial.end_time = datetime.now()
        self.current_trial.status = "completed"
        self.current_trial.score = score
        self.current_trial.objective_value = objective_value or score
        self.current_trial.model_path = model_path
        self.current_trial.evaluation_time = evaluation_time

        # Calculate trial time
        trial_time = (self.current_trial.end_time - self.current_trial.start_time).total_seconds()
        self.current_trial.evaluation_time = trial_time
        self.trial_times.append(trial_time)

        if evaluation_time:
            self.evaluation_times.append(evaluation_time)

        # Calculate improvement
        if self.best_score is not None:
            self.current_trial.best_so_far = self.best_score
            self.current_trial.improvement = score - self.best_score
        else:
            self.current_trial.best_so_far = score
            self.current_trial.improvement = score

        # Update best tracking
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_params = self.current_trial.parameters.copy()
            self.current_trial.best_so_far = score
            self.current_trial.improvement = 0.0

        # Update scores history
        self.scores_history.append(score)

        # Calculate percentile rank
        if self.scores_history:
            better_scores = sum(1 for s in self.scores_history if s > score)
            self.current_trial.percentile_rank = better_scores / len(self.scores_history)

        # Log completion metrics
        completion_metrics = {
            "score": score,
            "objective_value": objective_value or score,
            "trial_time": trial_time,
            "best_so_far": self.best_score,
            "improvement": self.current_trial.improvement,
            "percentile_rank": self.current_trial.percentile_rank
        }

        if metrics:
            completion_metrics.update(metrics)

        self.base_tracker.log_metrics(completion_metrics)

        # Log model artifact if provided
        if model_path:
            self.base_tracker.log_artifact(model_path, "trained_model")

        # Log success alert
        self.base_tracker.log_alert(
            "Trial Completed",
            f"Trial {self.current_trial.trial_number} completed with score: {score:.4f}",
            "info"
        )

        # Update study metadata
        self.study_metadata.completed_trials += 1
        self._update_study_metadata()

        # Store trial in history
        self.trials_history.append(self.current_trial)

        # Log trial metadata as artifact
        self._log_trial_metadata()

        # Finish trial run
        self.base_tracker.finish_run()

        self.current_trial = None

        logger.info(f"Completed trial with score: {score:.4f}")

    def prune_trial(self,
                    step: int,
                    reason: str = "Pruned by algorithm",
                    intermediate_value: Optional[float] = None) -> None:
        """
        Prune the current trial.

        Args:
            step: Step at which pruning occurred
            reason: Reason for pruning
            intermediate_value: Last intermediate value
        """
        if not self.current_trial:
            logger.warning("No current trial to prune")
            return

        # Update trial metadata
        self.current_trial.end_time = datetime.now()
        self.current_trial.status = "pruned"
        self.current_trial.pruned_step = step
        self.current_trial.score = intermediate_value

        # Calculate trial time
        trial_time = (self.current_trial.end_time - self.current_trial.start_time).total_seconds()
        self.current_trial.evaluation_time = trial_time
        self.trial_times.append(trial_time)

        # Log pruning
        self.base_tracker.log_metrics({
            "pruned": True,
            "pruned_step": step,
            "trial_time": trial_time
        })

        self.base_tracker.log_alert(
            "Trial Pruned",
            f"Trial {self.current_trial.trial_number} pruned at step {step}: {reason}",
            "warning"
        )

        # Update study metadata
        self.study_metadata.pruned_trials += 1

        # Store trial in history
        self.trials_history.append(self.current_trial)

        # Finish trial run
        self.base_tracker.finish_run(exit_code=1)

        self.current_trial = None

        logger.info(f"Pruned trial at step {step}: {reason}")

    def fail_trial(self,
                   error_message: str,
                   exception_type: Optional[str] = None) -> None:
        """
        Mark the current trial as failed.

        Args:
            error_message: Error message
            exception_type: Type of exception
        """
        if not self.current_trial:
            logger.warning("No current trial to fail")
            return

        # Update trial metadata
        self.current_trial.end_time = datetime.now()
        self.current_trial.status = "failed"
        self.current_trial.failure_reason = error_message

        # Calculate trial time
        trial_time = (self.current_trial.end_time - self.current_trial.start_time).total_seconds()
        self.current_trial.evaluation_time = trial_time
        self.trial_times.append(trial_time)

        # Log failure
        self.base_tracker.log_metrics({
            "failed": True,
            "error_message": error_message,
            "exception_type": exception_type,
            "trial_time": trial_time
        })

        self.base_tracker.log_alert(
            "Trial Failed",
            f"Trial {self.current_trial.trial_number} failed: {error_message}",
            "error"
        )

        # Update study metadata
        self.study_metadata.failed_trials += 1

        # Store trial in history
        self.trials_history.append(self.current_trial)

        # Finish trial run
        self.base_tracker.finish_run(exit_code=1)

        self.current_trial = None

        logger.error(f"Trial failed: {error_message}")

    def _update_study_metadata(self) -> None:
        """Update study metadata with current statistics."""
        if self.scores_history:
            completed_scores = [t.score for t in self.trials_history if t.status == "completed" and t.score is not None]

            if completed_scores:
                self.study_metadata.best_score = max(completed_scores)
                self.study_metadata.worst_score = min(completed_scores)
                self.study_metadata.mean_score = np.mean(completed_scores)
                self.study_metadata.std_score = np.std(completed_scores)

                # Find best parameters
                best_trial = max(self.trials_history, key=lambda t: t.score if t.score is not None else float('-inf'))
                if best_trial.score is not None:
                    self.study_metadata.best_params = best_trial.parameters

    def _log_trial_metadata(self) -> None:
        """Log trial metadata as artifact."""
        if self.current_trial:
            metadata = self.current_trial.to_dict()
            self.base_tracker.log_artifact_from_dict(metadata, "trial_metadata")

    def log_study_progress(self) -> None:
        """Log overall study progress."""
        progress_metrics = {
            "study_progress": {
                "total_trials": self.study_metadata.total_trials,
                "completed_trials": self.study_metadata.completed_trials,
                "pruned_trials": self.study_metadata.pruned_trials,
                "failed_trials": self.study_metadata.failed_trials,
                "completion_rate": self.study_metadata.completed_trials / max(1, self.study_metadata.total_trials)
            },
            "performance": {
                "best_score": self.best_score,
                "mean_score": self.study_metadata.mean_score,
                "std_score": self.study_metadata.std_score,
                "mean_trial_time": np.mean(self.trial_times) if self.trial_times else None,
                "mean_evaluation_time": np.mean(self.evaluation_times) if self.evaluation_times else None
            }
        }

        self.base_tracker.log_metrics(progress_metrics)

    def get_trials_dataframe(self) -> pd.DataFrame:
        """Get all trials as DataFrame."""
        if not self.trials_history:
            return pd.DataFrame()

        trials_data = []
        for trial in self.trials_history:
            trial_dict = trial.to_dict()
            # Flatten parameters
            if trial_dict["parameters"]:
                for param, value in trial_dict["parameters"].items():
                    trial_dict[f"param_{param}"] = value
            trials_data.append(trial_dict)

        return pd.DataFrame(trials_data)

    def get_best_trial(self) -> Optional[TrialMetadata]:
        """Get the best trial."""
        completed_trials = [t for t in self.trials_history if t.status == "completed" and t.score is not None]
        if not completed_trials:
            return None
        return max(completed_trials, key=lambda t: t.score)

    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Estimate parameter importance from completed trials.

        Returns:
            Dictionary of parameter importance scores
        """
        completed_trials = [t for t in self.trials_history if t.status == "completed" and t.score is not None]
        if len(completed_trials) < 5:
            logger.warning("Not enough completed trials for parameter importance analysis")
            return {}

        # Create DataFrame for analysis
        trials_df = self.get_trials_dataframe()
        if trials_df.empty:
            return {}

        # Calculate correlation for each parameter
        importance = {}
        numeric_params = []

        # Find numeric parameters
        for col in trials_df.columns:
            if col.startswith("param_"):
                param_name = col[6:]  # Remove "param_" prefix
                param_values = trials_df[col]
                if param_values.dtype in ['int64', 'float64'] and len(param_values.unique()) > 1:
                    numeric_params.append((param_name, col))

        # Calculate correlation with score
        for param_name, col in numeric_params:
            correlation = trials_df[col].corr(trials_df["score"])
            if not np.isnan(correlation):
                importance[param_name] = abs(correlation)

        # For categorical parameters, use one-way ANOVA
        for col in trials_df.columns:
            if col.startswith("param_"):
                param_name = col[6:]
                if param_name not in importance and trials_df[col].dtype == 'object':
                    # Calculate ANOVA F-statistic
                    groups = trials_df.groupby(col)["score"].apply(list)
                    if len(groups) > 1:
                        # Simple F-statistic calculation
                        group_means = [np.mean(group) for group in groups]
                        group_sizes = [len(group) for group in groups]
                        overall_mean = np.mean(trials_df["score"])

                        between_group_variance = sum(
                            size * (mean - overall_mean) ** 2
                            for size, mean in zip(group_sizes, group_means)
                        ) / (len(groups) - 1)

                        within_group_variance = sum(
                            sum((x - group_mean) ** 2 for x in group)
                            for group, group_mean in zip(groups, group_means)
                        ) / (len(trials_df) - len(groups))

                        if within_group_variance > 0:
                            f_statistic = between_group_variance / within_group_variance
                            importance[param_name] = f_statistic

        # Normalize importance scores
        if importance:
            max_importance = max(importance.values())
            if max_importance > 0:
                importance = {k: v / max_importance for k, v in importance.items()}

        return importance

    def generate_study_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive study report.

        Returns:
            Study report dictionary
        """
        # Update study metadata
        self._update_study_metadata()

        # Get parameter importance
        param_importance = self.get_parameter_importance()

        # Calculate statistics
        completed_trials = [t for t in self.trials_history if t.status == "completed"]
        score_stats = {}
        if completed_trials:
            scores = [t.score for t in completed_trials if t.score is not None]
            if scores:
                score_stats = {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "min": np.min(scores),
                    "max": np.max(scores),
                    "median": np.median(scores),
                    "q25": np.percentile(scores, 25),
                    "q75": np.percentile(scores, 75)
                }

        # Time statistics
        time_stats = {}
        if self.trial_times:
            time_stats = {
                "mean_trial_time": np.mean(self.trial_times),
                "std_trial_time": np.std(self.trial_times),
                "min_trial_time": np.min(self.trial_times),
                "max_trial_time": np.max(self.trial_times)
            }

        if self.evaluation_times:
            time_stats.update({
                "mean_evaluation_time": np.mean(self.evaluation_times),
                "std_evaluation_time": np.std(self.evaluation_times)
            })

        report = {
            "study_summary": self.study_metadata.to_dict(),
            "score_statistics": score_stats,
            "time_statistics": time_stats,
            "parameter_importance": param_importance,
            "best_trial": self.get_best_trial().to_dict() if self.get_best_trial() else None,
            "n_trials": len(self.trials_history),
            "completion_rate": self.study_metadata.completed_trials / max(1, self.study_metadata.total_trials),
            "pruning_rate": self.study_metadata.pruned_trials / max(1, self.study_metadata.total_trials),
            "failure_rate": self.study_metadata.failed_trials / max(1, self.study_metadata.total_trials)
        }

        return report

    def log_study_report(self) -> None:
        """Log comprehensive study report to base tracker."""
        report = self.generate_study_report()

        # Log as artifact
        self.base_tracker.log_artifact_from_dict(report, "study_report")

        # Log key metrics
        self.base_tracker.log_metrics({
            "study_summary": {
                "n_trials": report["n_trials"],
                "completion_rate": report["completion_rate"],
                "best_score": report["study_summary"]["best_score"],
                "mean_score": report["score_statistics"].get("mean"),
                "std_score": report["score_statistics"].get("std")
            }
        })

        logger.info(f"Logged study report for '{self.study_name}'")

    def finish_study(self) -> None:
        """Finish the study and log final report."""
        self.study_metadata.end_time = datetime.now()

        # Log final progress
        self.log_study_progress()

        # Log final study report
        self.log_study_report()

        # Log study completion alert
        self.base_tracker.log_alert(
            "Study Completed",
            f"Study '{self.study_name}' completed with {self.study_metadata.total_trials} trials",
            "info"
        )

        logger.info(f"Finished study '{self.study_name}'")

    def cleanup(self) -> None:
        """Cleanup trial tracker resources."""
        if self.current_trial:
            logger.warning("Cleaning up with active trial - completing with failure")
            self.fail_trial("Study cleanup interrupted active trial")

        if hasattr(self.base_tracker, 'is_active') and self.base_tracker.is_active():
            self.base_tracker.finish_run()

        logger.info("TrialTracker cleanup completed")