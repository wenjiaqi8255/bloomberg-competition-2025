"""
Tests for experiment tracking interface.

This module tests the ExperimentTrackerInterface and its implementations,
ensuring they properly handle experiment tracking operations.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.trading_system.utils.experiment_tracking.interface import (
    ExperimentTrackerInterface,
    NullExperimentTracker,
    ExperimentTrackingError
)
from src.trading_system.utils.experiment_tracking.config import ExperimentConfig


class MockExperimentTracker(ExperimentTrackerInterface):
    """Mock implementation of ExperimentTrackerInterface for testing."""

    def __init__(self):
        self.runs = {}
        self.current_run_id = None
        self.is_initialized_flag = False

    def init_run(self, config: ExperimentConfig) -> str:
        run_id = f"mock_run_{len(self.runs)}_{hash(str(config)) % 10000}"
        self.runs[run_id] = {
            "config": config,
            "params": {},
            "metrics": [],
            "artifacts": [],
            "figures": [],
            "tables": [],
            "alerts": [],
            "finished": False
        }
        self.current_run_id = run_id
        self.is_initialized_flag = True
        return run_id

    def log_params(self, params):
        if not self.is_active():
            raise ExperimentTrackingError("No active run")
        self.runs[self.current_run_id]["params"].update(params)

    def log_metrics(self, metrics, step=None):
        if not self.is_active():
            raise ExperimentTrackingError("No active run")
        self.runs[self.current_run_id]["metrics"].append({"metrics": metrics, "step": step})

    def log_artifact(self, artifact_path, artifact_name, artifact_type="model", description=""):
        if not self.is_active():
            raise ExperimentTrackingError("No active run")
        self.runs[self.current_run_id]["artifacts"].append({
            "path": artifact_path,
            "name": artifact_name,
            "type": artifact_type,
            "description": description
        })

    def log_figure(self, figure, figure_name):
        if not self.is_active():
            raise ExperimentTrackingError("No active run")
        self.runs[self.current_run_id]["figures"].append({"figure": figure, "name": figure_name})

    def log_table(self, data, table_name):
        if not self.is_active():
            raise ExperimentTrackingError("No active run")
        self.runs[self.current_run_id]["tables"].append({"data": data, "name": table_name})

    def log_alert(self, title, text, level="info"):
        if not self.is_active():
            raise ExperimentTrackingError("No active run")
        self.runs[self.current_run_id]["alerts"].append({"title": title, "text": text, "level": level})

    def create_child_run(self, name, config=None):
        if not self.is_active():
            raise ExperimentTrackingError("No active run")
        child_tracker = MockExperimentTracker()
        child_config = ExperimentConfig(
            project_name=self.runs[self.current_run_id]["config"].project_name,
            experiment_name=name,
            run_type="training"
        )
        if config:
            child_config.hyperparameters.update(config)
        child_tracker.init_run(child_config)
        return child_tracker

    def link_to_run(self, run_id, link_type="parent"):
        if not self.is_active():
            raise ExperimentTrackingError("No active run")
        # Mock implementation - just store the link
        if "links" not in self.runs[self.current_run_id]:
            self.runs[self.current_run_id]["links"] = []
        self.runs[self.current_run_id]["links"].append({"run_id": run_id, "type": link_type})

    def get_run_url(self):
        if not self.is_active():
            return None
        return f"https://mock.example.com/runs/{self.current_run_id}"

    def finish_run(self, exit_code=0):
        if not self.is_active():
            raise ExperimentTrackingError("No active run")
        self.runs[self.current_run_id]["finished"] = True
        self.current_run_id = None
        self.is_initialized_flag = False

    def is_active(self):
        return self.is_initialized_flag


class TestExperimentTrackerInterface:
    """Test cases for the abstract interface."""

    def test_interface_is_abstract(self):
        """Test that the interface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ExperimentTrackerInterface()

    def test_context_manager(self):
        """Test context manager functionality."""
        tracker = MockExperimentTracker()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        with tracker as t:
            run_id = t.init_run(config)
            assert run_id is not None
            assert t.is_active()

        # After context, should be finished
        assert not t.is_active()


class TestNullExperimentTracker:
    """Test cases for NullExperimentTracker."""

    def test_init_run(self):
        """Test that null tracker generates fake run IDs."""
        tracker = NullExperimentTracker()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        run_id1 = tracker.init_run(config)
        run_id2 = tracker.init_run(config)

        assert run_id1.startswith("null_run_")
        assert run_id2.startswith("null_run_")
        assert run_id1 != run_id2

    def test_log_operations_do_nothing(self):
        """Test that logging operations don't raise errors."""
        tracker = NullExperimentTracker()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        # Initialize run
        run_id = tracker.init_run(config)

        # All logging operations should not raise errors
        tracker.log_params({"param1": "value1"})
        tracker.log_metrics({"metric1": 1.0}, step=1)
        tracker.log_artifact("/path/to/model.pkl", "model")
        tracker.log_figure(None, "chart")
        tracker.log_table([{"a": 1}], "data")
        tracker.log_alert("Test Alert", "Test message", "warning")

        # Should not be active
        assert not tracker.is_active()

    def test_create_child_run(self):
        """Test creating child runs."""
        tracker = NullExperimentTracker()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        child_tracker = tracker.create_child_run("child_run")
        assert isinstance(child_tracker, NullExperimentTracker)
        assert child_tracker is not tracker

    def test_link_to_run(self):
        """Test linking to other runs."""
        tracker = NullExperimentTracker()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        # Should not raise error
        tracker.link_to_run("parent_run_123", "parent")

    def test_get_run_url(self):
        """Test getting run URL."""
        tracker = NullExperimentTracker()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        url = tracker.get_run_url()
        assert url is None

    def test_finish_run(self):
        """Test finishing run."""
        tracker = NullExperimentTracker()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        # Should not raise error
        tracker.finish_run()

    def test_context_manager(self):
        """Test context manager functionality."""
        tracker = NullExperimentTracker()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        with tracker as t:
            run_id = t.init_run(config)
            assert run_id is not None

        # Should complete without errors

    @patch('src.trading_system.utils.experiment_tracking.interface.logger')
    def test_debug_logging(self, mock_logger):
        """Test that operations are logged at debug level."""
        tracker = NullExperimentTracker()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        tracker.init_run(config)
        tracker.log_params({"param1": "value1"})
        tracker.log_metrics({"metric1": 1.0})

        # Should have called logger.debug
        assert mock_logger.debug.called
        debug_calls = [call[0][0] for call in mock_logger.debug.call_args_list]
        assert any("Null tracker: initialized run" in call for call in debug_calls)
        assert any("would log params" in call for call in debug_calls)
        assert any("would log metrics" in call for call in debug_calls)


class TestMockExperimentTracker:
    """Test cases for MockExperimentTracker used in testing."""

    def test_full_workflow(self):
        """Test a complete experiment workflow."""
        tracker = MockExperimentTracker()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training",
            hyperparameters={"learning_rate": 0.01}
        )

        # Initialize run
        run_id = tracker.init_run(config)
        assert run_id.startswith("mock_run_")
        assert tracker.is_active()

        # Log parameters
        tracker.log_params({"n_estimators": 100})
        assert tracker.runs[run_id]["params"]["n_estimators"] == 100

        # Log metrics
        tracker.log_metrics({"loss": 0.5}, step=1)
        tracker.log_metrics({"loss": 0.3}, step=2)
        assert len(tracker.runs[run_id]["metrics"]) == 2

        # Log artifact
        tracker.log_artifact("/path/to/model.pkl", "model", "model", "Trained model")
        assert len(tracker.runs[run_id]["artifacts"]) == 1

        # Log figure
        tracker.log_figure({"type": "plotly"}, "performance_chart")
        assert len(tracker.runs[run_id]["figures"]) == 1

        # Log table
        tracker.log_table([{"metric": "accuracy", "value": 0.95}], "results")
        assert len(tracker.runs[run_id]["tables"]) == 1

        # Log alert
        tracker.log_alert("Training Complete", "Model training finished successfully", "info")
        assert len(tracker.runs[run_id]["alerts"]) == 1

        # Get run URL
        url = tracker.get_run_url()
        assert url == f"https://mock.example.com/runs/{run_id}"

        # Finish run
        tracker.finish_run()
        assert not tracker.is_active()
        assert tracker.runs[run_id]["finished"] is True

    def test_error_on_inactive_operations(self):
        """Test that operations raise errors when no active run."""
        tracker = MockExperimentTracker()

        # Should raise errors when no run is active
        with pytest.raises(ExperimentTrackingError, match="No active run"):
            tracker.log_params({"param1": "value1"})

        with pytest.raises(ExperimentTrackingError, match="No active run"):
            tracker.log_metrics({"metric1": 1.0})

        with pytest.raises(ExperimentTrackingError, match="No active run"):
            tracker.log_artifact("/path", "artifact")

        with pytest.raises(ExperimentTrackingError, match="No active run"):
            tracker.log_figure(None, "figure")

        with pytest.raises(ExperimentTrackingError, match="No active run"):
            tracker.log_table([], "table")

        with pytest.raises(ExperimentTrackingError, match="No active run"):
            tracker.log_alert("title", "text")

        with pytest.raises(ExperimentTrackingError, match="No active run"):
            tracker.create_child_run("child")

        with pytest.raises(ExperimentTrackingError, match="No active run"):
            tracker.link_to_run("run_id", "parent")

        assert tracker.get_run_url() is None

        with pytest.raises(ExperimentTrackingError, match="No active run"):
            tracker.finish_run()

    def test_child_runs(self):
        """Test creating and using child runs."""
        parent_tracker = MockExperimentTracker()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="parent_experiment",
            run_type="optimization"
        )

        # Initialize parent run
        parent_run_id = parent_tracker.init_run(config)

        # Create child run
        child_tracker = parent_tracker.create_child_run("child_experiment", {"param1": "value1"})
        assert child_tracker.is_active()
        assert parent_tracker.is_active()  # Parent should still be active

        # Child should have its own run
        child_run_id = child_tracker.current_run_id
        assert child_run_id != parent_run_id

        # Child can log its own data
        child_tracker.log_metrics({"loss": 0.1})
        assert len(child_tracker.runs[child_run_id]["metrics"]) == 1

        # Finish child run
        child_tracker.finish_run()
        assert not child_tracker.is_active()
        assert parent_tracker.is_active()  # Parent should still be active

    def test_link_runs(self):
        """Test linking runs together."""
        tracker = MockExperimentTracker()
        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        )

        run_id = tracker.init_run(config)
        tracker.link_to_run("parent_run_123", "parent")
        tracker.link_to_run("related_run_456", "related")

        links = tracker.runs[run_id]["links"]
        assert len(links) == 2
        assert links[0]["run_id"] == "parent_run_123"
        assert links[0]["type"] == "parent"
        assert links[1]["run_id"] == "related_run_456"
        assert links[1]["type"] == "related"

    def test_multiple_runs(self):
        """Test managing multiple runs."""
        tracker = MockExperimentTracker()
        config1 = ExperimentConfig(
            project_name="test_project",
            experiment_name="experiment_1",
            run_type="training"
        )
        config2 = ExperimentConfig(
            project_name="test_project",
            experiment_name="experiment_2",
            run_type="training"
        )

        # First run
        run1 = tracker.init_run(config1)
        tracker.log_params({"param1": "value1"})

        # Finish first run
        tracker.finish_run()

        # Second run
        run2 = tracker.init_run(config2)
        tracker.log_params({"param2": "value2"})

        # Should have two separate runs
        assert len(tracker.runs) == 2
        assert run1 != run2
        assert tracker.runs[run1]["params"]["param1"] == "value1"
        assert tracker.runs[run2]["params"]["param2"] == "value2"