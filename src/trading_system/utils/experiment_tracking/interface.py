"""
Experiment tracking interface for the trading system.

This module defines the abstract interface for experiment tracking,
following the dependency inversion principle. Components should depend
on this interface, not on concrete implementations like WandB.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Union

from .config import ExperimentConfig

logger = logging.getLogger(__name__)


class ExperimentTrackerInterface(ABC):
    """
    Abstract interface for experiment tracking systems.

    This interface defines the contract that all experiment tracking
    implementations must follow, enabling:
    - Dependency injection for testing
    - Swappable tracking backends (WandB, MLflow, etc.)
    - Graceful degradation when tracking is unavailable
    """

    @abstractmethod
    def init_run(self, config: ExperimentConfig) -> str:
        """
        Initialize a new experiment run.

        Args:
            config: Experiment configuration including project name,
                   run type, hyperparameters, and metadata.

        Returns:
            Unique run identifier that can be used to reference this run.

        Raises:
            ExperimentTrackingError: If initialization fails.
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log experiment parameters/hyperparameters.

        Args:
            params: Dictionary of parameter names to values.
                   Should be logged once at the start of the experiment.

        Raises:
            ExperimentTrackingError: If logging fails.
        """
        pass

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Union[int, float]],
                   step: Optional[int] = None) -> None:
        """
        Log performance metrics.

        Args:
            metrics: Dictionary of metric names to numeric values.
            step: Optional step number for time-series logging.

        Raises:
            ExperimentTrackingError: If logging fails.
        """
        pass

    @abstractmethod
    def log_artifact(self, artifact_path: str, artifact_name: str,
                    artifact_type: str = "model", description: str = "") -> None:
        """
        Log an artifact (file or directory).

        Args:
            artifact_path: Path to the file or directory.
            artifact_name: Name for the artifact.
            artifact_type: Type of artifact (e.g., "model", "dataset", "config").
            description: Human-readable description of the artifact.

        Raises:
            ExperimentTrackingError: If logging fails.
        """
        pass

    @abstractmethod
    def log_figure(self, figure: Any, figure_name: str) -> None:
        """
        Log a visualization figure.

        Args:
            figure: Figure object (plotly Figure, matplotlib Figure, etc.).
            figure_name: Name for the figure.

        Raises:
            ExperimentTrackingError: If logging fails.
        """
        pass

    @abstractmethod
    def log_table(self, data: Any, table_name: str) -> None:
        """
        Log tabular data.

        Args:
            data: Table data (pandas DataFrame, list of dicts, etc.).
            table_name: Name for the table.

        Raises:
            ExperimentTrackingError: If logging fails.
        """
        pass

    @abstractmethod
    def log_alert(self, title: str, text: str, level: str = "info") -> None:
        """
        Log an alert/notification.

        Args:
            title: Alert title.
            text: Alert message.
            level: Alert level ("info", "warning", "error").

        Raises:
            ExperimentTrackingError: If logging fails.
        """
        pass

    def log_artifact_from_dict(self, data: Dict[str, Any], artifact_name: str) -> None:
        """
        Log a dictionary as an artifact.

        This is a convenience method that converts a dictionary to a temporary
        file and logs it as an artifact. Not all tracking backends support this.

        Args:
            data: Dictionary data to log.
            artifact_name: Name for the artifact.

        Raises:
            ExperimentTrackingError: If logging fails.
        """
        import tempfile
        import json

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data, f, indent=2, default=str)
                temp_path = f.name

            # Log as artifact
            self.log_artifact(temp_path, artifact_name, artifact_type="metadata")

            # Clean up
            import os
            os.unlink(temp_path)

        except Exception as e:
            logger.warning(f"Failed to log dictionary as artifact {artifact_name}: {e}")
            # Don't raise exception to avoid breaking the flow

    @abstractmethod
    def create_child_run(self, name: str, config: Optional[Dict[str, Any]] = None) -> 'ExperimentTrackerInterface':
        """
        Create a child run for hierarchical experiment tracking.

        Useful for hyperparameter optimization where each trial
        should be a separate run but linked to the parent optimization run.

        Args:
            name: Name for the child run.
            config: Optional configuration for the child run.

        Returns:
            New tracker instance for the child run.

        Raises:
            ExperimentTrackingError: If child run creation fails.
        """
        pass

    @abstractmethod
    def link_to_run(self, run_id: str, link_type: str = "parent") -> None:
        """
        Link this run to another run.

        Args:
            run_id: ID of the run to link to.
            link_type: Type of relationship ("parent", "child", "related").

        Raises:
            ExperimentTrackingError: If linking fails.
        """
        pass

    @abstractmethod
    def get_run_url(self) -> Optional[str]:
        """
        Get the URL for viewing this run in the tracking UI.

        Returns:
            URL string if available, None otherwise.
        """
        pass

    @abstractmethod
    def finish_run(self, exit_code: int = 0) -> None:
        """
        Finish the current experiment run.

        Args:
            exit_code: Exit code indicating success (0) or failure (non-zero).

        Raises:
            ExperimentTrackingError: If finishing fails.
        """
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """
        Check if the tracker is currently active and ready to log data.

        Returns:
            True if tracker is active, False otherwise.
        """
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if exc_type is not None:
            self.finish_run(exit_code=1)
        else:
            self.finish_run(exit_code=0)
        return False


class ExperimentTrackingError(Exception):
    """Exception raised for experiment tracking errors."""
    pass


class NullExperimentTracker(ExperimentTrackerInterface):
    """
    Null object implementation of ExperimentTrackerInterface.

    This implementation does nothing and is used when experiment tracking
    is unavailable or disabled. It allows the system to continue working
    without errors even when the tracking backend is not configured.
    """

    def __init__(self):
        self._run_count = 0

    def init_run(self, config: ExperimentConfig) -> str:
        """Generate a fake run ID."""
        run_id = f"null_run_{self._run_count}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._run_count += 1
        logger.debug(f"Null tracker: initialized run {run_id}")
        return run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Do nothing."""
        logger.debug(f"Null tracker: would log params {list(params.keys())}")

    def log_metrics(self, metrics: Dict[str, Union[int, float]],
                   step: Optional[int] = None) -> None:
        """Do nothing."""
        logger.debug(f"Null tracker: would log metrics {list(metrics.keys())} at step {step}")

    def log_artifact(self, artifact_path: str, artifact_name: str,
                    artifact_type: str = "model", description: str = "") -> None:
        """Do nothing."""
        logger.debug(f"Null tracker: would log artifact {artifact_name}")

    def log_figure(self, figure: Any, figure_name: str) -> None:
        """Do nothing."""
        logger.debug(f"Null tracker: would log figure {figure_name}")

    def log_table(self, data: Any, table_name: str) -> None:
        """Do nothing."""
        logger.debug(f"Null tracker: would log table {table_name}")

    def log_alert(self, title: str, text: str, level: str = "info") -> None:
        """Do nothing."""
        logger.debug(f"Null tracker: would log alert {title}: {text}")

    def create_child_run(self, name: str, config: Optional[Dict[str, Any]] = None) -> 'NullExperimentTracker':
        """Return another null tracker."""
        logger.debug(f"Null tracker: would create child run {name}")
        return NullExperimentTracker()

    def link_to_run(self, run_id: str, link_type: str = "parent") -> None:
        """Do nothing."""
        logger.debug(f"Null tracker: would link to run {run_id}")

    def get_run_url(self) -> Optional[str]:
        """Return None."""
        return None

    def finish_run(self, exit_code: int = 0) -> None:
        """Do nothing."""
        logger.debug("Null tracker: would finish run")

    def is_active(self) -> bool:
        """Always return False."""
        return False