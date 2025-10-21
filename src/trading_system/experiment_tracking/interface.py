"""
Simplified experiment tracking interface for the trading system.

This module defines a minimal interface for experiment tracking,
following the KISS principle. Only includes methods that are actually used.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, Union

from .config import ExperimentConfig

logger = logging.getLogger(__name__)


class ExperimentTrackerInterface(ABC):
    """
    Simplified interface for experiment tracking systems.
    
    Only includes methods that are actually used in the codebase.
    """

    @abstractmethod
    def init_run(self, config: ExperimentConfig) -> str:
        """
        Initialize a new experiment run.

        Args:
            config: Experiment configuration

        Returns:
            Unique run identifier
        """
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log experiment parameters/hyperparameters.

        Args:
            params: Dictionary of parameter names to values.
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
        """
        pass

    @abstractmethod
    def update_run_status(self, status: str) -> None:
        """
        Update the status of the current run.

        Args:
            status: New status value (e.g., "running", "completed", "failed").
        """
        pass

    def log_artifact_from_dict(self, data: Dict[str, Any], artifact_name: str) -> None:
        """
        Log a dictionary as an artifact.

        Args:
            data: Dictionary data to log.
            artifact_name: Name for the artifact.
        """
        import tempfile
        import json
        import os

        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                json.dump(data, f, indent=2, default=str)
                temp_path = f.name

            # Log as artifact
            self.log_artifact(temp_path, artifact_name, artifact_type="metadata")

            # Clean up
            os.unlink(temp_path)

        except Exception as e:
            logger.warning(f"Failed to log dictionary as artifact {artifact_name}: {e}")

    @abstractmethod
    def create_child_run(self, name: str, config: Optional[Dict[str, Any]] = None) -> 'ExperimentTrackerInterface':
        """
        Create a child run for hierarchical experiment tracking.

        Args:
            name: Name for the child run.
            config: Optional configuration for the child run.

        Returns:
            New tracker instance for the child run.
        """
        pass

    @abstractmethod
    def finish_run(self, exit_code: int = 0) -> None:
        """
        Finish the current experiment run.

        Args:
            exit_code: Exit code indicating success (0) or failure (non-zero).
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
    is unavailable or disabled.
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

    def update_run_status(self, status: str) -> None:
        """Do nothing."""
        logger.debug(f"Null tracker: would update run status to {status}")

    def create_child_run(self, name: str, config: Optional[Dict[str, Any]] = None) -> 'NullExperimentTracker':
        """Return another null tracker."""
        logger.debug(f"Null tracker: would create child run {name}")
        return NullExperimentTracker()

    def finish_run(self, exit_code: int = 0) -> None:
        """Do nothing."""
        logger.debug("Null tracker: would finish run")