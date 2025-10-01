"""
Simplified Phase 3: TrainingExperimentManager Concept Tests

This module tests the core concepts of experiment management
without complex import dependencies.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
import numpy as np
import tempfile

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Simple imports that should work
from utils.experiment_tracking import (
    ExperimentTrackerInterface,
    NullExperimentTracker,
    ExperimentConfig
)


class MockModel:
    """Simple mock model for testing."""

    def __init__(self, model_type="mock", config=None):
        self.model_type = model_type
        self.config = config or {"param1": "value1"}
        self._trained = False

    def fit(self, X, y):
        self._trained = True
        return self

    def predict(self, X):
        if not self._trained:
            raise ValueError("Model not trained")
        return np.random.normal(0, 1, len(X))

    def get_feature_importance(self):
        return {"feature1": 0.9, "feature2": 0.7}


class MockTrainingResult:
    """Mock training result."""

    def __init__(self, model):
        self.model = model
        self.training_time = 1.5
        self.validation_metrics = {"r2": 0.85, "ic": 0.12}
        self.cv_results = {"mean_r2": 0.83, "std_r2": 0.05}
        self.training_history = []


class ExperimentMetadata:
    """Mock experiment metadata."""

    def __init__(self, experiment_name, run_id, model_type, **kwargs):
        self.experiment_name = experiment_name
        self.run_id = run_id
        self.model_type = model_type
        self.status = "running"
        self.timestamp = datetime.now()
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockTrainingExperimentManager:
    """Mock training experiment manager for testing concepts."""

    def __init__(self, tracker, artifact_dir=None, save_models=True):
        self.tracker = tracker
        self.artifact_dir = artifact_dir or tempfile.mkdtemp()
        self.save_models = save_models
        self.current_experiment = None

    def run_training_experiment(self, model, X, y, experiment_config=None, **kwargs):
        """Mock experiment execution."""
        # Initialize experiment
        config = experiment_config or ExperimentConfig(
            project_name="test",
            experiment_name=f"{model.model_type}_experiment",
            run_type="training"
        )

        run_id = self.tracker.init_run(config)

        # Create metadata
        self.current_experiment = ExperimentMetadata(
            experiment_name=config.experiment_name,
            run_id=run_id,
            model_type=model.model_type,
            model_config=model.config,
            dataset_info={"samples": len(X), "features": len(X.columns)}
        )

        try:
            # Log dataset info
            self._log_dataset_info(X, y)

            # Mock training
            model.fit(X, y)

            # Log results
            self.tracker.log_metrics({"training_completed": 1, "r2": 0.85})

            # Log model info
            self._log_model_info(model)

            # Update status
            self.current_experiment.status = "completed"

            # Finish run
            self.tracker.finish_run()

            return MockTrainingResult(model)

        except Exception as e:
            self.current_experiment.status = "failed"
            self.tracker.log_alert("Experiment Failed", str(e), level="error")
            self.tracker.finish_run(exit_code=1)
            raise

    def _log_dataset_info(self, X, y):
        """Log dataset information."""
        stats = {
            "dataset_shape": X.shape,
            "feature_count": len(X.columns),
            "target_mean": float(y.mean()),
            "target_std": float(y.std())
        }
        self.tracker.log_params(stats)

    def _log_model_info(self, model):
        """Log model information."""
        importance = model.get_feature_importance()
        if importance:
            importance_df = pd.DataFrame([
                {"feature": name, "importance": value}
                for name, value in importance.items()
            ])
            self.tracker.log_table(importance_df, table_name="feature_importance")

    def get_experiment_history(self):
        """Get experiment history."""
        return [self.current_experiment] if self.current_experiment else []

    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self.tracker, 'is_active') and self.tracker.is_active():
            self.tracker.finish_run()


class TestTrainingExperimentManagerConcept(unittest.TestCase):
    """Test TrainingExperimentManager concepts."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock tracker
        self.mock_tracker = Mock(spec=ExperimentTrackerInterface)
        self.mock_tracker.init_run.return_value = "test_run_id"
        self.mock_tracker.is_active.return_value = True

        # Configure mock to return consistent values
        self.mock_tracker.log_table.return_value = None
        self.mock_tracker.log_params.return_value = None
        self.mock_tracker.log_metrics.return_value = None
        self.mock_tracker.finish_run.return_value = None

        # For multiple experiments, make init_run return different values
        self.call_count = 0
        def init_run_side_effect(config):
            self.call_count += 1
            return f"test_run_id_{self.call_count}"
        self.mock_tracker.init_run.side_effect = init_run_side_effect

        # Create test data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50)
        })
        self.y = pd.Series(np.random.normal(0, 1, 50))

        self.model = MockModel()

    def test_experiment_manager_concept(self):
        """Test the experiment manager concept."""
        manager = MockTrainingExperimentManager(
            tracker=self.mock_tracker,
            save_models=True
        )

        # Test initialization
        self.assertEqual(manager.tracker, self.mock_tracker)
        self.assertTrue(manager.save_models)
        self.assertIsNone(manager.current_experiment)

    def test_successful_experiment_execution(self):
        """Test successful experiment execution."""
        manager = MockTrainingExperimentManager(tracker=self.mock_tracker)

        # Create experiment config
        config = ExperimentConfig(
            project_name="test-project",
            experiment_name="test-experiment",
            run_type="training",
            tags=["test"]
        )

        # Run experiment
        result = manager.run_training_experiment(
            model=self.model,
            X=self.X,
            y=self.y,
            experiment_config=config
        )

        # Verify result
        self.assertIsNotNone(result)
        self.assertTrue(result.model._trained)

        # Verify tracker interactions
        self.mock_tracker.init_run.assert_called_once_with(config)
        self.mock_tracker.log_params.assert_called()
        self.mock_tracker.log_metrics.assert_called()
        self.mock_tracker.log_table.assert_called()
        self.mock_tracker.finish_run.assert_called_once_with()

        # Verify experiment metadata
        self.assertIsNotNone(manager.current_experiment)
        self.assertEqual(manager.current_experiment.status, "completed")
        self.assertEqual(manager.current_experiment.model_type, "mock")

    def test_experiment_with_dataset_logging(self):
        """Test dataset information logging."""
        manager = MockTrainingExperimentManager(tracker=self.mock_tracker)

        manager.run_training_experiment(self.model, self.X, self.y)

        # Verify dataset statistics were logged
        log_params_call = self.mock_tracker.log_params.call_args
        args, kwargs = log_params_call
        stats = args[0] if args else kwargs

        self.assertIn('dataset_shape', stats)
        self.assertEqual(stats['dataset_shape'], (50, 2))
        self.assertIn('feature_count', stats)
        self.assertEqual(stats['feature_count'], 2)
        self.assertIn('target_mean', stats)
        self.assertIn('target_std', stats)

    def test_experiment_with_feature_importance_logging(self):
        """Test feature importance logging."""
        manager = MockTrainingExperimentManager(tracker=self.mock_tracker)

        manager.run_training_experiment(self.model, self.X, self.y)

        # Verify feature importance was logged
        self.mock_tracker.log_table.assert_called_once()
        table_call = self.mock_tracker.log_table.call_args
        args, kwargs = table_call
        table_name = kwargs.get('table_name', '')

        self.assertEqual(table_name, "feature_importance")

        # Check table content
        table_data = args[0] if args else kwargs.get('data', pd.DataFrame())
        self.assertIsInstance(table_data, pd.DataFrame)
        self.assertIn('feature', table_data.columns)
        self.assertIn('importance', table_data.columns)
        self.assertEqual(len(table_data), 2)  # Two features

    def test_experiment_error_handling(self):
        """Test experiment error handling."""
        # Create manager with failing tracker
        failing_tracker = Mock(spec=ExperimentTrackerInterface)
        failing_tracker.init_run.return_value = "test_run_id"
        failing_tracker.log_metrics.side_effect = Exception("Logging failed")

        manager = MockTrainingExperimentManager(tracker=failing_tracker)

        # Mock model that fails during fit
        failing_model = Mock()
        failing_model.model_type = "failing_model"
        failing_model.config = {}
        failing_model.fit.side_effect = Exception("Training failed")

        # Should raise exception
        with self.assertRaises(Exception):
            manager.run_training_experiment(failing_model, self.X, self.y)

        # Verify error was logged
        failing_tracker.log_alert.assert_called_once()
        alert_call = failing_tracker.log_alert.call_args
        args, kwargs = alert_call
        self.assertEqual(args[0], "Experiment Failed")
        self.assertIn("Training failed", args[1])
        self.assertEqual(kwargs.get('level'), "error")

        # Verify experiment status
        self.assertEqual(manager.current_experiment.status, "failed")

    def test_experiment_history_tracking(self):
        """Test experiment history tracking."""
        manager = MockTrainingExperimentManager(tracker=self.mock_tracker)

        # Initially no history
        history = manager.get_experiment_history()
        self.assertEqual(len(history), 0)

        # Run experiment
        manager.run_training_experiment(self.model, self.X, self.y)

        # Should have one experiment in history
        history = manager.get_experiment_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].model_type, "mock")
        self.assertEqual(history[0].status, "completed")

    def test_experiment_cleanup(self):
        """Test experiment cleanup."""
        manager = MockTrainingExperimentManager(tracker=self.mock_tracker)

        # Run experiment
        manager.run_training_experiment(self.model, self.X, self.y)

        # Test cleanup
        manager.cleanup()

        # Verify cleanup was called
        self.mock_tracker.finish_run.assert_called()

    def test_experiment_with_null_tracker(self):
        """Test experiment with NullExperimentTracker."""
        null_tracker = NullExperimentTracker()
        manager = MockTrainingExperimentManager(tracker=null_tracker)

        # Should not raise exceptions
        result = manager.run_training_experiment(self.model, self.X, self.y)

        self.assertIsNotNone(result)
        self.assertTrue(result.model._trained)
        self.assertEqual(manager.current_experiment.status, "completed")

    def test_experiment_config_defaults(self):
        """Test experiment configuration defaults."""
        manager = MockTrainingExperimentManager(tracker=self.mock_tracker)

        # Run without explicit config
        result = manager.run_training_experiment(self.model, self.X, self.y)

        # Should use default config
        init_call = self.mock_tracker.init_run.call_args
        args, kwargs = init_call
        config = args[0] if args else kwargs.get('config', {})

        self.assertEqual(config.project_name, "test")
        self.assertIn("mock_experiment", config.experiment_name)
        self.assertEqual(config.run_type, "training")

    def test_multiple_experiments(self):
        """Test running multiple experiments."""
        manager = MockTrainingExperimentManager(tracker=self.mock_tracker)

        # Run first experiment
        result1 = manager.run_training_experiment(self.model, self.X, self.y)
        run_id1 = manager.current_experiment.run_id

        # Run second experiment
        model2 = MockModel(model_type="mock2")
        result2 = manager.run_training_experiment(model2, self.X, self.y)
        run_id2 = manager.current_experiment.run_id

        # Should have different run IDs
        self.assertNotEqual(run_id1, run_id2)

        # Should have called init_run twice
        self.assertEqual(self.mock_tracker.init_run.call_count, 2)

        # Should have called finish_run twice
        self.assertEqual(self.mock_tracker.finish_run.call_count, 2)

    def test_experiment_artifact_directory(self):
        """Test artifact directory handling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = MockTrainingExperimentManager(
                tracker=self.mock_tracker,
                artifact_dir=temp_dir
            )

            self.assertEqual(manager.artifact_dir, temp_dir)

            # Run experiment
            manager.run_training_experiment(self.model, self.X, self.y)

            # Directory should exist
            self.assertTrue(os.path.exists(temp_dir))


if __name__ == "__main__":
    unittest.main()