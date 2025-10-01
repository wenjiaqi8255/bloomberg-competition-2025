"""
Phase 3: TrainingExperimentManager Tests

This module tests the TrainingExperimentManager functionality,
including experiment lifecycle management, artifact handling,
and comprehensive tracking integration.
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import pandas as pd
import numpy as np
import tempfile
import pathlib

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
        self._feature_importance = {"feature1": 0.9, "feature2": 0.7, "feature3": 0.5}

    def fit(self, X, y):
        self._trained = True
        return self

    def predict(self, X):
        if not self._trained:
            raise ValueError("Model not trained")
        return np.random.normal(0, 1, len(X))

    def get_feature_importance(self):
        return self._feature_importance

    def validate_data(self, X, y):
        if len(X) != len(y):
            raise ValueError("X and y must have same length")


class MockTrainingResult:
    """Mock training result for testing."""

    def __init__(self, model, validation_metrics=None, cv_results=None):
        self.model = model
        self.training_time = 1.5
        self.validation_metrics = validation_metrics or {"r2": 0.85, "ic": 0.12}
        self.cv_results = cv_results or {"mean_r2": 0.83, "std_r2": 0.05}
        self.training_history = [{"epoch": 1, "loss": 0.5}, {"epoch": 2, "loss": 0.3}]


class MockModelTrainer:
    """Mock model trainer for testing."""

    def __init__(self, config=None, experiment_tracker=None):
        self.config = config
        self.experiment_tracker = experiment_tracker

    def train_with_tracking(self, model, X, y, experiment_config=None, X_test=None, y_test=None):
        """Mock training with tracking."""
        model.fit(X, y)

        # Simulate tracking calls
        if self.experiment_tracker:
            self.experiment_tracker.log_metrics({"training_completed": 1})
            self.experiment_tracker.finish_run()

        return MockTrainingResult(model)


class TestTrainingExperimentManager(unittest.TestCase):
    """Test TrainingExperimentManager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock tracker
        self.mock_tracker = Mock(spec=ExperimentTrackerInterface)
        self.mock_tracker.init_run.return_value = "test_run_id"
        self.mock_tracker.is_active.return_value = True

        # Create test data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50),
            'feature3': np.random.normal(0, 1, 50)
        })
        self.y = pd.Series(np.random.normal(0, 1, 50))

        self.X_test = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 20),
            'feature2': np.random.normal(0, 1, 20),
            'feature3': np.random.normal(0, 1, 20)
        })
        self.y_test = pd.Series(np.random.normal(0, 1, 20))

        self.model = MockModel()

    @patch('src.trading_system.testing.test_phase3_experiment_manager.ModelTrainer')
    @patch('src.trading_system.testing.test_phase3_experiment_manager.ExperimentVisualizer')
    def test_experiment_manager_initialization(self, mock_visualizer_class, mock_trainer_class):
        """Test experiment manager initialization."""
        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer

        from models.training.experiment_manager import TrainingExperimentManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TrainingExperimentManager(
                tracker=self.mock_tracker,
                artifact_dir=temp_dir,
                save_models=True
            )

            self.assertEqual(manager.tracker, self.mock_tracker)
            self.assertEqual(manager.save_models, True)
            self.assertEqual(manager.visualizer, mock_visualizer)
            self.assertTrue(pathlib.Path(temp_dir).exists())

    @patch('src.trading_system.testing.test_phase3_experiment_manager.ModelTrainer')
    @patch('src.trading_system.testing.test_phase3_experiment_manager.ExperimentVisualizer')
    def test_run_training_experiment_success(self, mock_visualizer_class, mock_trainer_class):
        """Test successful experiment execution."""
        # Setup mocks
        mock_trainer = MockModelTrainer(experiment_tracker=self.mock_tracker)
        mock_trainer_class.return_value = mock_trainer

        mock_visualizer = Mock()
        mock_visualizer.create_feature_importance.return_value = Mock()
        mock_visualizer.create_training_curve.return_value = Mock()
        mock_visualizer_class.return_value = mock_visualizer

        # Import here to avoid import issues
        from models.training.experiment_manager import TrainingExperimentManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TrainingExperimentManager(
                tracker=self.mock_tracker,
                artifact_dir=temp_dir,
                save_models=False  # Disable model saving for simplicity
            )

            # Create experiment config
            experiment_config = ExperimentConfig(
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
                experiment_config=experiment_config,
                X_test=self.X_test,
                y_test=self.y_test
            )

            # Verify result
            self.assertIsNotNone(result)
            self.assertTrue(result.model._trained)

            # Verify tracker was called
            self.mock_tracker.init_run.assert_called_once_with(experiment_config)
            self.mock_tracker.log_params.assert_called()
            self.mock_tracker.log_metrics.assert_called()
            self.mock_tracker.finish_run.assert_called_once()

            # Verify experiment metadata was created
            self.assertIsNotNone(manager.current_experiment)
            self.assertEqual(manager.current_experiment.status, "completed")

    @patch('src.trading_system.testing.test_phase3_experiment_manager.ModelTrainer')
    @patch('src.trading_system.testing.test_phase3_experiment_manager.ExperimentVisualizer')
    @patch('builtins.open', create=True)
    @patch('pickle.dump')
    def test_run_experiment_with_model_saving(self, mock_pickle_dump, mock_open, mock_visualizer_class, mock_trainer_class):
        """Test experiment execution with model saving."""
        # Setup mocks
        mock_trainer = MockModelTrainer(experiment_tracker=self.mock_tracker)
        mock_trainer_class.return_value = mock_trainer

        mock_visualizer = Mock()
        mock_visualizer_class.return_value = mock_visualizer

        from models.training.experiment_manager import TrainingExperimentManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TrainingExperimentManager(
                tracker=self.mock_tracker,
                artifact_dir=temp_dir,
                save_models=True
            )

            # Run experiment
            result = manager.run_training_experiment(
                model=self.model,
                X=self.X,
                y=self.y
            )

            # Verify model saving was attempted
            mock_pickle_dump.assert_called()
            mock_open.assert_called()

            # Verify artifact logging
            self.mock_tracker.log_artifact.assert_called()

    @patch('src.trading_system.testing.test_phase3_experiment_manager.ModelTrainer')
    def test_experiment_error_handling(self, mock_trainer_class):
        """Test experiment error handling."""
        # Setup trainer to raise exception
        mock_trainer = Mock()
        mock_trainer.train_with_tracking.side_effect = Exception("Training failed")
        mock_trainer_class.return_value = mock_trainer

        from models.training.experiment_manager import TrainingExperimentManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TrainingExperimentManager(
                tracker=self.mock_tracker,
                artifact_dir=temp_dir
            )

            # Should raise exception
            with self.assertRaises(RuntimeError):
                manager.run_training_experiment(
                    model=self.model,
                    X=self.X,
                    y=self.y
                )

            # Verify error was logged
            self.mock_tracker.log_alert.assert_called_once()
            alert_call = self.mock_tracker.log_alert.call_args
            args, kwargs = alert_call
            self.assertEqual(args[0], "Experiment Failed")
            self.assertIn("Training failed", args[1])
            self.assertEqual(kwargs.get('level'), "error")

            # Verify run finished with error code
            finish_call = self.mock_tracker.finish_run.call_args
            args, kwargs = finish_call
            self.assertEqual(kwargs.get('exit_code'), 1)

            # Verify experiment status updated
            self.assertIsNotNone(manager.current_experiment)
            self.assertEqual(manager.current_experiment.status, "failed")

    def test_create_default_experiment_config(self):
        """Test default experiment configuration creation."""
        from models.training.experiment_manager import TrainingExperimentManager

        manager = TrainingExperimentManager(tracker=self.mock_tracker)
        config = manager._create_default_experiment_config(self.model)

        self.assertEqual(config.project_name, "model-training")
        self.assertIn("mock_experiment", config.experiment_name)
        self.assertEqual(config.run_type, "training")
        self.assertIn("mock", config.tags)
        self.assertEqual(config.hyperparameters, self.model.config)

    @patch('src.trading_system.testing.test_phase3_experiment_manager.ModelTrainer')
    def test_log_dataset_info(self, mock_trainer_class):
        """Test dataset information logging."""
        from models.training.experiment_manager import TrainingExperimentManager

        manager = TrainingExperimentManager(tracker=self.mock_tracker)
        manager._log_dataset_info(self.X, self.y, self.X_test, self.y_test)

        # Verify dataset info was logged
        log_params_calls = self.mock_tracker.log_params.call_args_list
        self.assertTrue(len(log_params_calls) >= 1)

        # Check that dataset statistics are included
        params_logged = False
        for call in log_params_calls:
            args, kwargs = call
            params = args[0] if args else kwargs
            if 'dataset_shape' in params:
                params_logged = True
                self.assertEqual(params['dataset_shape'], (50, 3))
                self.assertEqual(params['feature_count'], 3)
                self.assertEqual(params['sample_count'], 50)
                break

        self.assertTrue(params_logged, "Dataset parameters should be logged")

        # Verify table logging for data sample
        self.mock_tracker.log_table.assert_called()
        table_call = self.mock_tracker.log_table.call_args
        args, kwargs = table_call
        self.assertEqual(kwargs.get('table_name'), "data_sample")

    @patch('src.trading_system.testing.test_phase3_experiment_manager.ExperimentVisualizer')
    def test_create_and_log_visualizations(self, mock_visualizer_class):
        """Test visualization creation and logging."""
        # Setup mock visualizer
        mock_visualizer = Mock()
        mock_fig = Mock()
        mock_visualizer.create_feature_importance.return_value = mock_fig
        mock_visualizer.create_training_curve.return_value = mock_fig
        mock_visualizer_class.return_value = mock_visualizer

        from models.training.experiment_manager import TrainingExperimentManager

        manager = TrainingExperimentManager(tracker=self.mock_tracker, visualizer=mock_visualizer)

        # Create mock result
        result = MockTrainingResult(self.model)

        # Test visualization creation
        manager._create_and_log_visualizations(result, self.X, self.y)

        # Verify visualizations were created
        mock_visualizer.create_feature_importance.assert_called_once()
        mock_visualizer.create_training_curve.assert_called_once()

        # Verify figures were logged
        self.assertEqual(self.mock_tracker.log_figure.call_count, 2)  # feature importance + training history

    def test_get_experiment_history(self):
        """Test getting experiment history."""
        from models.training.experiment_manager import TrainingExperimentManager

        manager = TrainingExperimentManager(tracker=self.mock_tracker)

        # Initially no history
        history = manager.get_experiment_history()
        self.assertEqual(len(history), 0)

        # Set current experiment
        from models.training.experiment_manager import ExperimentMetadata
        manager.current_experiment = ExperimentMetadata(
            experiment_name="test",
            run_id="test_run",
            model_type="mock",
            model_config={},
            training_config={},
            dataset_info={},
            timestamp=datetime.now(),
            status="completed"
        )

        # Should have one experiment in history
        history = manager.get_experiment_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].experiment_name, "test")

    def test_cleanup(self):
        """Test cleanup functionality."""
        from models.training.experiment_manager import TrainingExperimentManager

        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TrainingExperimentManager(
                tracker=self.mock_tracker,
                artifact_dir=temp_dir
            )

            # Test cleanup
            manager.cleanup()

            # Verify tracker finish was called
            if hasattr(self.mock_tracker, 'is_active') and self.mock_tracker.is_active():
                self.mock_tracker.finish_run.assert_called_once()


class TestExperimentMetadata(unittest.TestCase):
    """Test ExperimentMetadata dataclass."""

    def test_experiment_metadata_creation(self):
        """Test experiment metadata creation."""
        from models.training.experiment_manager import ExperimentMetadata

        timestamp = datetime.now()
        metadata = ExperimentMetadata(
            experiment_name="test_experiment",
            run_id="test_run_123",
            model_type="mock_model",
            model_config={"param1": "value1"},
            training_config={"cv_folds": 5},
            dataset_info={"samples": 1000, "features": 10},
            timestamp=timestamp,
            status="running"
        )

        self.assertEqual(metadata.experiment_name, "test_experiment")
        self.assertEqual(metadata.run_id, "test_run_123")
        self.assertEqual(metadata.model_type, "mock_model")
        self.assertEqual(metadata.status, "running")
        self.assertEqual(metadata.timestamp, timestamp)


if __name__ == "__main__":
    unittest.main()