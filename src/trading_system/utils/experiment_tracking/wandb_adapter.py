"""
WandB adapter for the experiment tracking interface.

This module provides a clean adapter that implements ExperimentTrackerInterface
using WandB as the backend. It focuses purely on tracking functionality
and delegates visualization to the dedicated visualizer component.
"""

import logging
import os
import tempfile
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..wandb_logger import WandBLogger
from .interface import ExperimentTrackerInterface, ExperimentTrackingError
from .config import ExperimentConfig
from .visualizer import ExperimentVisualizer

logger = logging.getLogger(__name__)


class WandBExperimentTracker(ExperimentTrackerInterface):
    """
    Clean WandB implementation of ExperimentTrackerInterface.

    This adapter focuses on tracking functionality and uses the ExperimentVisualizer
    for creating visualizations. It provides better error handling and
    graceful degradation compared to the original WandBLogger.
    """

    def __init__(self, project_name: str = "bloomberg-competition",
                 entity: Optional[str] = None, tags: Optional[List[str]] = None,
                 group: Optional[str] = None, config: Optional[Dict] = None,
                 visualizer: Optional[ExperimentVisualizer] = None,
                 fail_silently: bool = True):
        """
        Initialize WandB experiment tracker.

        Args:
            project_name: WandB project name
            entity: WandB entity (team/username)
            tags: List of tags for experiments
            group: Group for organizing experiments
            config: Default configuration dictionary
            visualizer: Custom visualizer instance
            fail_silently: If True, tracking failures don't raise exceptions
        """
        self.project_name = project_name
        self.entity = entity
        self.tags = tags or []
        self.group = group
        self.config = config or {}
        self.fail_silently = fail_silently

        # Initialize components
        self.visualizer = visualizer or ExperimentVisualizer()
        self.wandb_logger: Optional[WandBLogger] = None
        self.current_config: Optional[ExperimentConfig] = None
        self.child_runs: List['WandBExperimentTracker'] = []
        self._initialize_wandb()

    def _initialize_wandb(self) -> None:
        """Initialize WandB logger with error handling."""
        try:
            self.wandb_logger = WandBLogger(
                project_name=self.project_name,
                entity=self.entity,
                tags=self.tags,
                group=self.group,
                config=self.config
            )
        except Exception as e:
            if self.fail_silently:
                logger.warning(f"Failed to initialize WandB: {e}. Running in offline mode.")
                self.wandb_logger = None
            else:
                raise ExperimentTrackingError(f"Failed to initialize WandB: {e}")

    def _handle_wandb_error(self, operation: str, error: Exception) -> None:
        """Handle WandB errors according to fail_silently setting."""
        if self.fail_silently:
            logger.error(f"WandB {operation} failed: {error}")
        else:
            raise ExperimentTrackingError(f"WandB {operation} failed: {error}")

    def init_run(self, config: ExperimentConfig) -> str:
        """
        Initialize a new WandB experiment run.

        Args:
            config: Experiment configuration

        Returns:
            Run identifier (WandB run ID if available, otherwise generated)

        Raises:
            ExperimentTrackingError: If initialization fails and fail_silently is False
        """
        self.current_config = config

        if self.wandb_logger is None:
            # WandB not available, generate fake run ID
            run_id = config.run_id or f"offline_run_{config.experiment_name}"
            logger.info(f"Running offline with run ID: {run_id}")
            return run_id

        try:
            # Initialize WandB experiment
            success = self.wandb_logger.initialize_experiment(
                experiment_name=config.experiment_name,
                run_id=config.run_id,
                notes=config.notes,
                tags=config.tags,
                group=config.group
            )

            if not success:
                raise ExperimentTrackingError("Failed to initialize WandB experiment")

            # Log configuration parameters
            self.wandb_logger.log_config(config.hyperparameters)

            # Log metadata as additional config
            if config.metadata:
                self.wandb_logger.log_config({"metadata": config.metadata})

            # Log data and model info
            if config.data_info:
                self.wandb_logger.log_dataset_info(config.data_info)

            # Generate a run ID if WandB doesn't provide one
            run_id = config.run_id or f"wandb_run_{config.experiment_name}"

            logger.info(f"Initialized WandB run: {run_id}")
            return run_id

        except Exception as e:
            self._handle_wandb_error("run initialization", e)
            # Fallback to offline mode
            run_id = config.run_id or f"offline_run_{config.experiment_name}"
            logger.info(f"Falling back to offline mode with run ID: {run_id}")
            return run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log experiment parameters."""
        if self.wandb_logger is None:
            logger.debug(f"Offline mode: would log params {list(params.keys())}")
            return

        try:
            self.wandb_logger.log_hyperparameters(params)
        except Exception as e:
            self._handle_wandb_error("parameter logging", e)

    def log_metrics(self, metrics: Dict[str, Union[int, float]],
                   step: Optional[int] = None) -> None:
        """Log performance metrics."""
        if self.wandb_logger is None:
            logger.debug(f"Offline mode: would log metrics {list(metrics.keys())} at step {step}")
            return

        try:
            self.wandb_logger.log_metrics(metrics, step=step)
        except Exception as e:
            self._handle_wandb_error("metrics logging", e)

    def log_artifact(self, artifact_path: str, artifact_name: str,
                    artifact_type: str = "model", description: str = "") -> None:
        """Log an artifact."""
        if self.wandb_logger is None:
            logger.debug(f"Offline mode: would log artifact {artifact_name}")
            return

        try:
            self.wandb_logger.log_artifact(
                artifact_path=artifact_path,
                artifact_name=artifact_name,
                artifact_type=artifact_type,
                description=description
            )
        except Exception as e:
            self._handle_wandb_error("artifact logging", e)

    def log_figure(self, figure: Any, figure_name: str) -> None:
        """Log a visualization figure."""
        if self.wandb_logger is None:
            logger.debug(f"Offline mode: would log figure {figure_name}")
            return

        try:
            # Use visualizer to save figure temporarily, then log as artifact
            import tempfile
            import os

            # Try to save as HTML for web-based figures
            with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
                if hasattr(figure, 'write_html'):  # Plotly figure
                    f.write(figure.write_html(include_plotlyjs='cdn'))
                else:
                    # Fallback to image conversion
                    image_bytes = self.visualizer.figure_to_image_bytes(figure, format='png')
                    f.write(f'<html><body><img src="data:image/png;base64,{image_bytes.hex()}"></body></html>')
                temp_path = f.name

            self.wandb_logger.log_artifact(
                artifact_path=temp_path,
                artifact_name=f"{figure_name}.html",
                artifact_type="visualization",
                description=f"Visualization: {figure_name}"
            )

            # Clean up temporary file
            os.unlink(temp_path)

        except Exception as e:
            self._handle_wandb_error("figure logging", e)

    def log_table(self, data: Any, table_name: str) -> None:
        """Log tabular data."""
        if self.wandb_logger is None:
            logger.debug(f"Offline mode: would log table {table_name}")
            return

        try:
            import tempfile
            import os

            # Convert data to DataFrame and save as CSV
            if not isinstance(data, pd.DataFrame):
                try:
                    df = pd.DataFrame(data)
                except Exception:
                    df = pd.DataFrame({"value": [data]})
            else:
                df = data

            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                df.to_csv(f.name, index=False)
                temp_path = f.name

            self.wandb_logger.log_artifact(
                artifact_path=temp_path,
                artifact_name=f"{table_name}.csv",
                artifact_type="table",
                description=f"Table data: {table_name}"
            )

            os.unlink(temp_path)

        except Exception as e:
            self._handle_wandb_error("table logging", e)

    def log_alert(self, title: str, text: str, level: str = "info") -> None:
        """Log an alert/notification."""
        if self.wandb_logger is None:
            logger.debug(f"Offline mode: would log alert {title}: {text}")
            return

        try:
            import wandb

            # Log alert as a special metric with metadata
            alert_metric = {
                f"alert_{title.lower().replace(' ', '_')}": 1 if level != "info" else 0,
                "alert_level": {"info": 0, "warning": 1, "error": 2}.get(level, 0)
            }

            self.wandb_logger.log_metrics(alert_metric)

            # Also log as text message
            alert_text = f"[{level.upper()}] {title}: {text}"
            wandb.log({"alert_message": alert_text})

        except Exception as e:
            self._handle_wandb_error("alert logging", e)

    def create_child_run(self, name: str, config: Optional[Dict[str, Any]] = None) -> 'WandBExperimentTracker':
        """Create a child run for hierarchical experiment tracking."""
        try:
            # Create child tracker with same project but different experiment name
            child_tracker = WandBExperimentTracker(
                project_name=self.project_name,
                entity=self.entity,
                group=self.current_config.experiment_name if self.current_config else name,
                tags=self.current_config.tags if self.current_config else [],
                visualizer=self.visualizer,
                fail_silently=self.fail_silently
            )

            # Store reference to parent for cleanup
            self.child_runs.append(child_tracker)

            return child_tracker

        except Exception as e:
            self._handle_wandb_error("child run creation", e)
            # Return null tracker if child creation fails
            from .interface import NullExperimentTracker
            return NullExperimentTracker()

    def link_to_run(self, run_id: str, link_type: str = "parent") -> None:
        """Link this run to another run."""
        if self.wandb_logger is None:
            logger.debug(f"Offline mode: would link to run {run_id} as {link_type}")
            return

        try:
            link_info = {
                f"linked_run_{link_type}": run_id,
                "link_timestamp": pd.Timestamp.now().isoformat()
            }
            self.wandb_logger.log_config(link_info)

        except Exception as e:
            self._handle_wandb_error("run linking", e)

    def get_run_url(self) -> Optional[str]:
        """Get the URL for viewing this run in the WandB UI."""
        if self.wandb_logger is None or not self.wandb_logger.is_initialized:
            return None

        try:
            if hasattr(self.wandb_logger, 'run') and self.wandb_logger.run:
                return getattr(self.wandb_logger.run, 'url', None)
            return None
        except Exception:
            return None

    def finish_run(self, exit_code: int = 0) -> None:
        """Finish the current WandB experiment run."""
        # Finish child runs first
        for child_tracker in self.child_runs:
            if child_tracker.is_active():
                child_tracker.finish_run(exit_code)
        self.child_runs.clear()

        if self.wandb_logger is None:
            logger.debug("Offline mode: would finish run")
            self.current_config = None
            return

        try:
            self.wandb_logger.finish_experiment(exit_code=exit_code)
            self.current_config = None
            logger.info("WandB run finished successfully")

        except Exception as e:
            self._handle_wandb_error("run finishing", e)
            self.current_config = None

    def is_active(self) -> bool:
        """Check if the tracker is currently active."""
        return self.wandb_logger is not None and self.wandb_logger.is_initialized

    # Enhanced convenience methods with error handling
    def log_portfolio_performance(self, portfolio_df, benchmark_df=None, step=None):
        """Log portfolio performance with visualization."""
        if self.wandb_logger is None:
            logger.debug("Offline mode: would log portfolio performance")
            return

        try:
            # Use WandBLogger for portfolio-specific logging
            self.wandb_logger.log_portfolio_performance(portfolio_df, benchmark_df, step)

            # Also create and log a visualization using our visualizer
            portfolio_fig = self.visualizer.create_portfolio_performance(portfolio_df, benchmark_df)
            if portfolio_fig:
                self.log_figure(portfolio_fig, "portfolio_performance")

            # Create drawdown chart
            drawdown_fig = self.visualizer.create_drawdown_chart(portfolio_df)
            if drawdown_fig:
                self.log_figure(drawdown_fig, "drawdown")

        except Exception as e:
            self._handle_wandb_error("portfolio performance logging", e)

    def log_trades(self, trades_df, step=None):
        """Log trading activity with visualization."""
        if self.wandb_logger is None:
            logger.debug("Offline mode: would log trades")
            return

        try:
            self.wandb_logger.log_trades(trades_df, step)

        except Exception as e:
            self._handle_wandb_error("trade logging", e)

    def log_dataset_info(self, dataset_stats):
        """Log dataset statistics."""
        if self.wandb_logger is None:
            logger.debug(f"Offline mode: would log dataset info {list(dataset_stats.keys())}")
            return

        try:
            self.wandb_logger.log_dataset_info(dataset_stats)
        except Exception as e:
            self._handle_wandb_error("dataset info logging", e)

    def log_training_metrics(self, metrics_history: Dict[str, List[float]], step: Optional[int] = None):
        """Log training metrics with visualization."""
        # Log individual metric points
        if step is not None:
            for metric_name, values in metrics_history.items():
                if step < len(values):
                    self.log_metrics({metric_name: values[step]}, step=step)

        # Create and log training curve visualization
        training_curve = self.visualizer.create_training_curve(metrics_history)
        if training_curve:
            self.log_figure(training_curve, "training_curve")

    def log_feature_importance(self, importance_data: Union[Dict[str, float], pd.Series], top_n: int = 20):
        """Log feature importance with visualization."""
        importance_fig = self.visualizer.create_feature_importance(importance_data, top_n)
        if importance_fig:
            self.log_figure(importance_fig, "feature_importance")

        # Also log as table
        if isinstance(importance_data, dict):
            df = pd.DataFrame([
                {"feature": name, "importance": value}
                for name, value in sorted(importance_data.items(), key=lambda x: x[1], reverse=True)
            ])
        else:
            df = pd.DataFrame({
                "feature": importance_data.index,
                "importance": importance_data.values
            })
        self.log_table(df.head(top_n), "feature_importance")


def create_wandb_tracker_from_config(config: ExperimentConfig) -> WandBExperimentTracker:
    """
    Factory function to create a WandB tracker from ExperimentConfig.

    Args:
        config: Experiment configuration

    Returns:
        Configured WandBExperimentTracker
    """
    return WandBExperimentTracker(
        project_name=config.project_name,
        entity=config.entity,
        tags=config.tags,
        group=config.group,
        config=config.hyperparameters
    )