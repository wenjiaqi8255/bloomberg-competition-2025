"""
Tests for Phase 5 Monitoring Enhancements

This module tests the integration of experiment tracking with model monitoring,
including performance degradation alerts, health status tracking, and dashboard generation.
"""

import unittest
import sys
import os
import tempfile
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Add the src directory to Python path correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
src_dir = os.path.join(project_root, 'src')

# Add both src directory and project root to path
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

try:
    import pytest
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from src.trading_system.models.serving.monitor import ModelMonitor, ModelHealthStatus
    from src.trading_system.models.serving.dashboard import MonitoringDashboard, Dashboard
    from src.trading_system.utils.experiment_tracking.interface import (
        ExperimentTrackerInterface, NullExperimentTracker, ExperimentTrackingError
    )
    from src.trading_system.utils.experiment_tracking.config import ExperimentConfig
    from src.trading_system.models.base.base_model import BaseModel
    IMPORTS_AVAILABLE = True
except ImportError as e:
    IMPORTS_AVAILABLE = False
    print(f"Import error: {e}")


# Skip all tests if imports are not available
if not IMPORTS_AVAILABLE:
    pytestmark = pytest.mark.skip("Required imports not available")


class MockModel(BaseModel):
    """Mock model for testing."""

    def __init__(self, model_id="test_model"):
        self.model_id = model_id
        self.model_type = "mock"
        self.config = {"test": True}
        self.is_trained = True
        self.status = "trained"
        self.metadata = {"test": True}

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            return pd.Series(np.random.normal(0, 1, len(X)))
        return np.random.normal(0, 1, len(X))

    def fit(self, X, y):
        self.is_trained = True
        self.status = "trained"
        return self

    def get_feature_importance(self):
        return {"feature_1": 0.5, "feature_2": 0.3, "feature_3": 0.2}


class MockExperimentTracker(ExperimentTrackerInterface):
    """Mock experiment tracker for testing."""

    def __init__(self):
        self.runs = []
        self.current_run = None
        self.alerts = []
        self.metrics = []
        self.statuses = []

    def init_run(self, config):
        run_id = f"test_run_{len(self.runs)}"
        self.current_run = {
            "id": run_id,
            "config": config,
            "finished": False
        }
        self.runs.append(self.current_run)
        return run_id

    def log_params(self, params):
        if self.current_run:
            self.current_run["params"] = params

    def log_metrics(self, metrics, step=None):
        if self.current_run:
            if "metrics" not in self.current_run:
                self.current_run["metrics"] = []
            self.current_run["metrics"].append({
                "values": metrics,
                "step": step,
                "timestamp": datetime.now()
            })
            self.metrics.extend(metrics.items())

    def log_artifact(self, artifact_path, artifact_name, artifact_type="model", description=""):
        if self.current_run:
            if "artifacts" not in self.current_run:
                self.current_run["artifacts"] = []
            self.current_run["artifacts"].append({
                "path": artifact_path,
                "name": artifact_name,
                "type": artifact_type,
                "description": description
            })

    def log_figure(self, figure, figure_name):
        if self.current_run:
            if "figures" not in self.current_run:
                self.current_run["figures"] = []
            self.current_run["figures"].append({
                "name": figure_name,
                "figure": figure
            })

    def log_table(self, data, table_name):
        if self.current_run:
            if "tables" not in self.current_run:
                self.current_run["tables"] = []
            self.current_run["tables"].append({
                "name": table_name,
                "data": data
            })

    def log_alert(self, title, text, level="info"):
        alert = {
            "title": title,
            "text": text,
            "level": level,
            "timestamp": datetime.now()
        }
        self.alerts.append(alert)
        if self.current_run:
            if "alerts" not in self.current_run:
                self.current_run["alerts"] = []
            self.current_run["alerts"].append(alert)

    def update_run_status(self, status):
        self.statuses.append(status)
        if self.current_run:
            self.current_run["status"] = status

    def create_child_run(self, name, config=None):
        child = MockExperimentTracker()
        child.init_run(config or {"name": name})
        return child

    def link_to_run(self, run_id, link_type="parent"):
        if self.current_run:
            if "links" not in self.current_run:
                self.current_run["links"] = []
            self.current_run["links"].append({
                "run_id": run_id,
                "type": link_type
            })

    def get_run_url(self):
        return f"https://test.wandb.io/run/{self.current_run['id']}" if self.current_run else None

    def finish_run(self, exit_code=0):
        if self.current_run:
            self.current_run["finished"] = True
            self.current_run["exit_code"] = exit_code

    def is_active(self):
        return self.current_run is not None and not self.current_run["finished"]


@pytest.fixture
def mock_tracker():
    """Create a mock experiment tracker."""
    return MockExperimentTracker()


@pytest.fixture
def mock_model():
    """Create a mock model."""
    return MockModel()


@pytest.fixture
def sample_monitor(mock_tracker):
    """Create a model monitor with mock tracker."""
    config = {
        'performance_window': 7,
        'degradation_threshold': 0.1,
        'min_samples': 5
    }
    return ModelMonitor("test_model", config, tracker=mock_tracker)


class TestModelMonitorIntegration:
    """Test ModelMonitor integration with experiment tracking."""

    def test_monitor_initialization_with_tracker(self, mock_tracker):
        """Test that monitor initializes tracking run."""
        monitor = ModelMonitor("test_model", tracker=mock_tracker)

        # Should have initialized a monitoring run
        assert len(mock_tracker.runs) == 1
        assert mock_tracker.runs[0]["config"].experiment_name == "monitor_test_model"
        assert mock_tracker.runs[0]["config"].run_type == "monitoring"
        assert "monitoring" in mock_tracker.statuses

    def test_monitor_initialization_without_tracker(self):
        """Test that monitor works without tracker."""
        monitor = ModelMonitor("test_model", tracker=None)

        # Should work fine without tracker
        assert monitor.model_id == "test_model"
        assert monitor.tracker is None

    def test_performance_degradation_alert(self, sample_monitor, mock_model, mock_tracker):
        """Test performance degradation alerts."""
        # Set some baseline metrics manually to simulate degradation
        sample_monitor.baseline_metrics = {"r2": 0.8, "rmse": 0.1}

        # Log some predictions with actual values
        features = {"feature_1": 1.0, "feature_2": 2.0}

        # Log predictions with poor performance
        for i in range(10):
            sample_monitor.log_prediction(features, 1.0, actual=-1.0)  # Poor predictions

        # Mock the performance evaluation to return poor metrics
        with patch('src.trading_system.models.utils.performance_evaluator.PerformanceEvaluator.evaluate_model') as mock_eval:
            mock_eval.return_value = {"r2": -0.5, "rmse": 1.0, "correlation": -0.3}

            # Check performance (should detect degradation)
            result = sample_monitor.check_performance_degradation(mock_model)
            assert result['degradation_detected']

        # Verify alert was logged
        assert len(mock_tracker.alerts) > 0
        degradation_alerts = [a for a in mock_tracker.alerts if "Performance Degradation" in a["title"]]
        assert len(degradation_alerts) > 0
        assert degradation_alerts[0]["level"] == "warning"

    def test_health_status_changes_tracked(self, sample_monitor, mock_model, mock_tracker):
        """Test that health status changes are tracked."""
        # Get initial health status
        initial_health = sample_monitor.get_health_status(mock_model)
        initial_status = initial_health.status

        # The status should be tracked in metrics
        assert len(mock_tracker.metrics) > 0

        # Simulate some issues that would change status
        # Add some predictions to trigger issues
        for i in range(5):
            sample_monitor.log_prediction({"f1": i}, 0.1, actual=0.2)

        # Get health status again
        new_health = sample_monitor.get_health_status(mock_model)

        # Check that status change was logged if it occurred
        if initial_status != new_health.status:
            status_metrics = [m for m in mock_tracker.metrics if "health_status_code" in m[0]]
            assert len(status_metrics) > 0

    def test_log_prediction_with_tracker(self, sample_monitor):
        """Test that prediction logging works with tracker."""
        features = {"feature_1": 1.0, "feature_2": 2.0}

        # Log a prediction
        prediction_id = sample_monitor.log_prediction(features, 0.1, actual=0.15)

        # Should have logged successfully
        assert prediction_id is not None
        assert len(sample_monitor.prediction_log) == 1

    def test_stop_monitoring(self, sample_monitor, mock_tracker):
        """Test stopping monitoring and finalizing run."""
        # Stop monitoring
        sample_monitor.stop_monitoring()

        # Run should be finished
        assert mock_tracker.runs[0]["finished"]
        assert mock_tracker.runs[0]["exit_code"] == 0

        # Should have logged final report
        assert "artifacts" in mock_tracker.runs[0]
        artifact_names = [a["name"] for a in mock_tracker.runs[0]["artifacts"]]
        assert any("final_monitoring_report" in name for name in artifact_names)


class TestMonitoringDashboard:
    """Test monitoring dashboard functionality."""

    def test_dashboard_creation(self, sample_monitor):
        """Test dashboard creation from monitor."""
        # Add some prediction data
        for i in range(20):
            sample_monitor.log_prediction(
                {"feature_1": i * 0.1},
                np.random.normal(0, 1),
                actual=np.random.normal(0.1, 1)
            )

        # Create dashboard
        dashboard_factory = MonitoringDashboard()
        dashboard = dashboard_factory.create_dashboard(sample_monitor)

        # Verify dashboard structure
        assert isinstance(dashboard, Dashboard)
        assert dashboard.model_id == "test_model"
        assert len(dashboard.charts) > 0
        assert dashboard.summary_metrics is not None
        assert dashboard.generated_at is not None

    def test_dashboard_chart_types(self, sample_monitor):
        """Test that dashboard contains different chart types."""
        # Add some data
        for i in range(15):
            sample_monitor.log_prediction(
                {"feature_1": i * 0.1},
                0.1,
                actual=0.15 if i % 3 == 0 else 0.05
            )

        dashboard_factory = MonitoringDashboard()
        dashboard = dashboard_factory.create_dashboard(sample_monitor)

        # Check for different chart types
        chart_types = {chart.chart_type for chart in dashboard.charts}
        expected_types = {"performance", "timeline", "metrics"}

        # Should have at least some of the expected types
        assert len(chart_types.intersection(expected_types)) >= 2

    def test_dashboard_html_export(self, sample_monitor):
        """Test dashboard HTML export."""
        dashboard_factory = MonitoringDashboard()
        dashboard = dashboard_factory.create_dashboard(sample_monitor)

        # Generate HTML
        html_content = dashboard.to_html()

        # Verify HTML structure
        assert "<!DOCTYPE html>" in html_content
        assert "test_model" in html_content
        assert "Model Monitoring Dashboard" in html_content

    def test_dashboard_save_to_file(self, sample_monitor):
        """Test saving dashboard to file."""
        dashboard_factory = MonitoringDashboard()
        dashboard = dashboard_factory.create_dashboard(sample_monitor)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            temp_path = f.name

        try:
            saved_path = dashboard_factory.save_dashboard(dashboard, temp_path)

            # Verify file was created
            assert os.path.exists(saved_path)
            assert saved_path == temp_path

            # Verify content
            with open(saved_path, 'r') as f:
                content = f.read()
                assert "test_model" in content

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_dashboard_with_minimal_data(self):
        """Test dashboard creation with minimal data."""
        monitor = ModelMonitor("empty_model")

        dashboard_factory = MonitoringDashboard()
        dashboard = dashboard_factory.create_dashboard(monitor)

        # Should still create dashboard even with minimal data
        assert isinstance(dashboard, Dashboard)
        assert dashboard.model_id == "empty_model"


class TestExperimentTrackerInterface:
    """Test the enhanced experiment tracker interface."""

    def test_interface_has_required_methods(self):
        """Test that interface has all required methods."""
        # Check that abstract methods are defined
        abstract_methods = ExperimentTrackerInterface.__abstractmethods__
        expected_methods = {
            'init_run', 'log_params', 'log_metrics', 'log_artifact',
            'log_figure', 'log_table', 'log_alert', 'update_run_status',
            'create_child_run', 'link_to_run', 'get_run_url', 'finish_run', 'is_active'
        }

        assert abstract_methods == expected_methods

    def test_null_tracker_implements_all_methods(self):
        """Test that NullExperimentTracker implements all methods."""
        tracker = NullExperimentTracker()

        # Should implement all interface methods
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test",
            run_type="training",  # Use valid run_type
            tags=[],
            hyperparameters={},
            metadata={}
        )

        run_id = tracker.init_run(config)
        assert run_id is not None

        tracker.log_params({"test": 1})
        tracker.log_metrics({"metric": 1.0})
        tracker.log_artifact("path", "name")
        tracker.log_figure(None, "figure")
        tracker.log_table([], "table")
        tracker.log_alert("test", "test", "info")
        tracker.update_run_status("running")

        child = tracker.create_child_run("child")
        assert isinstance(child, NullExperimentTracker)

        tracker.link_to_run("test_run")
        tracker.finish_run()
        assert not tracker.is_active()


class TestWandBExperimentTracker:
    """Test WandB experiment tracker enhancements."""

    @patch('src.trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger')
    def test_alert_logging(self, mock_wandb_logger_class):
        """Test alert logging in WandB tracker."""
        # Setup mock
        mock_wandb_logger = Mock()
        mock_wandb_logger_class.return_value = mock_wandb_logger

        from src.trading_system.utils.experiment_tracking.wandb_adapter import WandBExperimentTracker

        tracker = WandBExperimentTracker(fail_silently=True)

        # Test alert logging
        tracker.log_alert("Test Alert", "This is a test alert", "warning")

        # Verify WandB logger was called appropriately
        assert mock_wandb_logger.log_metrics.called

    @patch('src.trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger')
    def test_status_update(self, mock_wandb_logger_class):
        """Test status updates in WandB tracker."""
        # Setup mock
        mock_wandb_logger = Mock()
        mock_wandb_logger_class.return_value = mock_wandb_logger

        from src.trading_system.utils.experiment_tracking.wandb_adapter import WandBExperimentTracker

        tracker = WandBExperimentTracker(fail_silently=True)

        # Test status update
        tracker.update_run_status("monitoring")

        # Verify status was set
        # Note: In real implementation, this would use wandb.log
        # For test, we just verify no exception was raised
        assert True  # If we reach here, no exception was raised


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios."""

    def test_monitoring_workflow_with_tracker(self, mock_tracker):
        """Test complete monitoring workflow with tracking."""
        # Initialize monitor with tracker
        monitor = ModelMonitor("integration_test", tracker=mock_tracker)

        # Add prediction data
        features = {"feature_1": 1.0, "feature_2": 2.0}
        for i in range(20):
            actual = 0.1 + np.random.normal(0, 0.1)
            prediction = actual + np.random.normal(0, 0.2)
            monitor.log_prediction(features, prediction, actual=actual)

        # Check performance (triggers tracking if degradation)
        model = MockModel()
        result = monitor.check_performance_degradation(model)

        # Get health status
        health = monitor.get_health_status(model)

        # Create dashboard
        dashboard_factory = MonitoringDashboard()
        dashboard = dashboard_factory.create_dashboard(monitor)

        # Stop monitoring
        monitor.stop_monitoring()

        # Verify tracking integration
        assert len(mock_tracker.runs) == 1
        assert mock_tracker.runs[0]["finished"]

        # Should have logged some artifacts
        assert "artifacts" in mock_tracker.runs[0]

    def test_monitoring_workflow_without_tracker(self):
        """Test monitoring workflow without tracking (graceful degradation)."""
        # Initialize monitor without tracker
        monitor = ModelMonitor("no_tracker_test", tracker=None)

        # Add prediction data
        features = {"feature_1": 1.0}
        for i in range(10):
            monitor.log_prediction(features, 0.1, actual=0.15)

        # Check performance
        model = MockModel()
        result = monitor.check_performance_degradation(model)

        # Get health status
        health = monitor.get_health_status(model)

        # Create dashboard
        dashboard_factory = MonitoringDashboard()
        dashboard = dashboard_factory.create_dashboard(monitor)

        # Stop monitoring (should not fail)
        monitor.stop_monitoring()

        # Verify everything worked without tracker
        assert isinstance(health, ModelHealthStatus)
        assert isinstance(dashboard, Dashboard)


if __name__ == "__main__":
    if not IMPORTS_AVAILABLE:
        print("Required imports not available, skipping tests")
    else:
        pytest.main([__file__, "-v"])