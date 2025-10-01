"""
Phase 3: ModelTrainer Integration Tests

This module tests the integration of experiment tracking with the ModelTrainer.
It verifies that training can be properly tracked with different tracker implementations
and that all training-related information is correctly logged.

Test Coverage:
- ModelTrainer with NullExperimentTracker
- ModelTrainer with WandBExperimentTracker
- ModelTrainer.train_with_tracking() method
- Cross-validation tracking
- Feature importance logging
- Training metrics aggregation
- Error handling and cleanup
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime
import pandas as pd
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.training.trainer import ModelTrainer, TrainingConfig, TrainingResult
from models.base.base_model import BaseModel, ModelStatus
from utils.experiment_tracking import (
    ExperimentTrackerInterface,
    NullExperimentTracker,
    WandBExperimentTracker,
    ExperimentConfig,
    ExperimentTrackingError
)
from utils.experiment_tracking.training_interface import (
    TrainingMetrics,
    ModelLifecycleEvent,
    TrainingExperimentConfig
)


class MockModel(BaseModel):
    """Mock model for testing purposes."""

    def __init__(self, model_type="mock", config=None):
        super().__init__(model_type, config or {})
        self._is_trained = False
        self._feature_importance = {"feature1": 0.9, "feature2": 0.7, "feature3": 0.5}

    def fit(self, X, y):
        """Mock fitting."""
        self._is_trained = True
        self.status = ModelStatus.TRAINED
        return self

    def predict(self, X):
        """Mock prediction."""
        if not self._is_trained:
            raise ValueError("Model must be trained before prediction")
        return np.random.normal(0, 1, len(X))

    def get_feature_importance(self):
        """Return mock feature importance."""
        return self._feature_importance

    def validate_data(self, X, y):
        """Mock data validation."""
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        if len(X) < 10:
            raise ValueError("Not enough data")


class TestModelTrainerWithNullTracker(unittest.TestCase):
    """Test ModelTrainer with NullExperimentTracker."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TrainingConfig(
            use_cross_validation=False,
            log_experiment=False
        )
        self.null_tracker = NullExperimentTracker()
        self.trainer = ModelTrainer(
            config=self.config,
            experiment_tracker=self.null_tracker
        )

        # Create test data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        })
        self.y = pd.Series(np.random.normal(0, 1, 100))

        self.model = MockModel()

    def test_trainer_initialization_with_null_tracker(self):
        """Test trainer can be initialized with NullExperimentTracker."""
        self.assertIsInstance(self.trainer.experiment_tracker, NullExperimentTracker)

    def test_training_without_tracking_works_normally(self):
        """Test that training works normally with NullExperimentTracker."""
        result = self.trainer.train(self.model, self.X, self.y)

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.model.status, ModelStatus.TRAINED)
        self.assertIsNotNone(result.validation_metrics)
        self.assertGreater(result.training_time, 0)

    def test_training_with_tracking_and_null_tracker(self):
        """Test train_with_tracking with NullExperimentTracker."""
        experiment_config = {
            'project_name': 'test-project',
            'experiment_name': 'test-experiment',
            'tags': ['test']
        }

        result = self.trainer.train_with_tracking(
            self.model, self.X, self.y, experiment_config
        )

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.model.status, ModelStatus.TRAINED)
        self.assertIsNotNone(result.validation_metrics)

    def test_null_tracker_methods_dont_raise_exceptions(self):
        """Test that NullExperimentTracker methods don't raise exceptions."""
        # All these should work without errors
        self.null_tracker.log_params({"test": "value"})
        self.null_tracker.log_metrics({"accuracy": 0.9})
        self.null_tracker.log_artifact("/tmp/test.pkl", "test_model")
        self.null_tracker.log_figure(None, "test_chart")
        self.null_tracker.log_table(pd.DataFrame({"a": [1, 2]}), "test_table")
        self.null_tracker.log_alert("Test Alert", "Test message", "info")
        self.null_tracker.finish_run()

        # Should return false for most operations
        self.assertFalse(self.null_tracker.is_active())


class TestModelTrainerWithMockTracker(unittest.TestCase):
    """Test ModelTrainer with a mock experiment tracker."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TrainingConfig(
            use_cross_validation=False,
            log_experiment=False
        )

        # Create mock tracker
        self.mock_tracker = Mock(spec=ExperimentTrackerInterface)
        self.mock_tracker.init_run.return_value = "test_run_id"
        self.mock_tracker.is_active.return_value = True

        self.trainer = ModelTrainer(
            config=self.config,
            experiment_tracker=self.mock_tracker
        )

        # Create test data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 50),
            'feature2': np.random.normal(0, 1, 50),
            'feature3': np.random.normal(0, 1, 50)
        })
        self.y = pd.Series(np.random.normal(0, 1, 50))

        self.model = MockModel()

    def test_trainer_with_mock_tracker_initialization(self):
        """Test trainer initialization with mock tracker."""
        self.assertEqual(self.trainer.experiment_tracker, self.mock_tracker)

    def test_train_with_tracking_calls_tracker_methods(self):
        """Test that train_with_tracking calls tracker methods correctly."""
        experiment_config = {
            'project_name': 'test-project',
            'experiment_name': 'test-experiment',
            'tags': ['test']
        }

        result = self.trainer.train_with_tracking(
            self.model, self.X, self.y, experiment_config
        )

        # Verify tracker was called
        self.mock_tracker.init_run.assert_called_once()
        self.mock_tracker.log_params.assert_called()
        self.mock_tracker.log_metrics.assert_called()
        self.mock_tracker.finish_run.assert_called_once()

        # Verify result
        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.model.status, ModelStatus.TRAINED)

    def test_train_with_tracking_logs_data_statistics(self):
        """Test that data statistics are logged correctly."""
        self.trainer.train_with_tracking(self.model, self.X, self.y)

        # Verify log_params was called with data statistics
        log_params_calls = self.mock_tracker.log_params.call_args_list
        self.assertTrue(len(log_params_calls) >= 1)

        # Check that data statistics are included
        stats_call = None
        for call in log_params_calls:
            args, kwargs = call
            params = args[0] if args else kwargs
            if 'dataset_shape' in params:
                stats_call = params
                break

        self.assertIsNotNone(stats_call, "Data statistics should be logged")
        self.assertEqual(stats_call['dataset_shape'], (50, 3))
        self.assertIn('feature_count', stats_call)
        self.assertIn('target_mean', stats_call)

    def test_train_with_tracking_logs_feature_importance(self):
        """Test that feature importance is logged."""
        self.trainer.train_with_tracking(self.model, self.X, self.y)

        # Should log feature importance as table
        self.mock_tracker.log_table.assert_called()

        # Check the table call arguments
        table_calls = self.mock_tracker.log_table.call_args_list
        importance_call = None
        for call in table_calls:
            args, kwargs = call
            table_name = kwargs.get('table_name', '')
            if 'importance' in table_name.lower():
                importance_call = call
                break

        self.assertIsNotNone(importance_call, "Feature importance should be logged as table")

    def test_train_with_cross_validation_tracking(self):
        """Test cross-validation with tracking."""
        cv_config = TrainingConfig(
            use_cross_validation=True,
            cv_folds=3,
            log_experiment=False
        )

        trainer_cv = ModelTrainer(
            config=cv_config,
            experiment_tracker=self.mock_tracker
        )

        result = trainer_cv.train_with_tracking(self.model, self.X, self.y)

        # Should log CV configuration
        log_params_calls = self.mock_tracker.log_params.call_args_list
        cv_logged = any(
            'cv_folds' in str(call)
            for call in log_params_calls
        )
        self.assertTrue(cv_logged, "CV configuration should be logged")

        # Should log CV results
        log_metrics_calls = self.mock_tracker.log_metrics.call_args_list
        cv_metrics_logged = any(
            'cv_mean_r2' in str(call)
            for call in log_metrics_calls
        )
        self.assertTrue(cv_metrics_logged, "CV results should be logged")

        self.assertIsNotNone(result.cv_results)
        self.assertIn('mean_r2', result.cv_results)

    def test_train_with_test_data_tracking(self):
        """Test training with test data tracking."""
        X_test = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 20),
            'feature2': np.random.normal(0, 1, 20),
            'feature3': np.random.normal(0, 1, 20)
        })
        y_test = pd.Series(np.random.normal(0, 1, 20))

        result = self.trainer.train_with_tracking(
            self.model, self.X, self.y, X_test=X_test, y_test=y_test
        )

        # Should have test metrics
        self.assertIsNotNone(result.test_metrics)
        self.assertIn('r2', result.test_metrics)

        # Verify test metrics were logged
        log_metrics_calls = self.mock_tracker.log_metrics.call_args_list
        test_metrics_logged = any(
            'step' in str(call) and '1' in str(call)  # Test metrics logged at step 1
            for call in log_metrics_calls
        )
        self.assertTrue(test_metrics_logged, "Test metrics should be logged")

    def test_error_handling_with_tracking(self):
        """Test error handling when tracking is enabled."""
        # Mock model that raises an exception during training
        failing_model = Mock()
        failing_model.model_type = "failing_model"
        failing_model.config = {}
        failing_model.fit.side_effect = Exception("Training failed")

        with self.assertRaises(RuntimeError):
            self.trainer.train_with_tracking(failing_model, self.X, self.y)

        # Should log error alert
        self.mock_tracker.log_alert.assert_called()
        alert_call = self.mock_tracker.log_alert.call_args
        args, kwargs = alert_call
        self.assertEqual(args[0], "Training Failed")  # title
        self.assertIn("Training failed", args[1])  # text contains error
        self.assertEqual(kwargs.get('level'), "error")

        # Should finish run with exit code 1
        finish_call = self.mock_tracker.finish_run.call_args
        args, kwargs = finish_call
        self.assertEqual(kwargs.get('exit_code'), 1)

    def test_legacy_train_method_uses_tracker(self):
        """Test that legacy train method also uses the tracker."""
        config_with_logging = TrainingConfig(log_experiment=True)
        trainer_legacy = ModelTrainer(
            config=config_with_logging,
            experiment_tracker=self.mock_tracker
        )

        result = trainer_legacy.train(self.model, self.X, self.y)

        # Should call log experiment (legacy method)
        # Note: This depends on the implementation of _log_experiment
        self.assertTrue(self.mock_tracker.log_params.called or
                       self.mock_tracker.log_metrics.called)

        # Should finish the run
        self.mock_tracker.finish_run.assert_called()

        self.assertIsInstance(result, TrainingResult)


class TestModelTrainerWithWandBTracker(unittest.TestCase):
    """Test ModelTrainer integration with WandBExperimentTracker."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = TrainingConfig(log_experiment=False)

        # Create test data
        np.random.seed(42)
        self.X = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 30),
            'feature2': np.random.normal(0, 1, 30)
        })
        self.y = pd.Series(np.random.normal(0, 1, 30))

        self.model = MockModel()

    @patch('models.training.trainer.WandBLogger')
    def test_trainer_with_wandb_tracker_offline_mode(self, mock_wandb_logger_class):
        """Test trainer with WandB tracker in offline mode."""
        # Mock WandBLogger to simulate offline mode
        mock_wandb_logger = Mock()
        mock_wandb_logger.is_initialized = False
        mock_wandb_logger_class.return_value = mock_wandb_logger

        # Create trainer with WandB tracker
        wandb_tracker = WandBExperimentTracker(fail_silently=True)
        trainer = ModelTrainer(
            config=self.config,
            experiment_tracker=wandb_tracker
        )

        # Should not raise exception even in offline mode
        result = trainer.train_with_tracking(self.model, self.X, self.y)

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.model.status, ModelStatus.TRAINED)

    @patch('models.training.trainer.WandBLogger')
    def test_trainer_with_wandb_tracker_success(self, mock_wandb_logger_class):
        """Test trainer with working WandB tracker."""
        # Mock successful WandB logger
        mock_wandb_logger = Mock()
        mock_wandb_logger.is_initialized = True
        mock_wandb_logger.log_metrics.return_value = True
        mock_wandb_logger.log_hyperparameters.return_value = True
        mock_wandb_logger.log_artifact.return_value = True
        mock_wandb_logger_class.return_value = mock_wandb_logger

        wandb_tracker = WandBExperimentTracker(fail_silently=True)
        trainer = ModelTrainer(
            config=self.config,
            experiment_tracker=wandb_tracker
        )

        result = trainer.train_with_tracking(self.model, self.X, self.y)

        self.assertIsInstance(result, TrainingResult)
        self.assertEqual(result.model.status, ModelStatus.TRAINED)

        # Verify WandB logger was used
        self.assertTrue(mock_wandb_logger.log_metrics.called or
                       mock_wandb_logger.log_hyperparameters.called)


class TestTrainingMetricsAggregation(unittest.TestCase):
    """Test training metrics aggregation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        from utils.experiment_tracking.training_interface import TrainingMetricsAggregator
        self.aggregator = TrainingMetricsAggregator()

    def test_add_training_metrics(self):
        """Test adding training metrics."""
        metrics1 = TrainingMetrics(step=1, loss=0.5, training_score=0.7)
        metrics2 = TrainingMetrics(step=2, loss=0.3, training_score=0.8)

        self.aggregator.add_metrics(metrics1)
        self.aggregator.add_metrics(metrics2)

        self.assertEqual(len(self.aggregator.metrics_history), 2)
        self.assertEqual(self.aggregator.metrics_history[0].loss, 0.5)
        self.assertEqual(self.aggregator.metrics_history[1].loss, 0.3)

    def test_add_lifecycle_events(self):
        """Test adding lifecycle events."""
        event1 = ModelLifecycleEvent(
            event_type="training_started",
            timestamp="2023-01-01T10:00:00",
            model_type="test_model"
        )
        event2 = ModelLifecycleEvent(
            event_type="training_completed",
            timestamp="2023-01-01T10:05:00",
            model_type="test_model"
        )

        self.aggregator.add_lifecycle_event(event1)
        self.aggregator.add_lifecycle_event(event2)

        self.assertEqual(len(self.aggregator.lifecycle_events), 2)
        self.assertEqual(self.aggregator.lifecycle_events[0].event_type, "training_started")

    def test_get_best_epoch(self):
        """Test getting best epoch."""
        metrics1 = TrainingMetrics(step=1, validation_score=0.7)
        metrics2 = TrainingMetrics(step=2, validation_score=0.8)
        metrics3 = TrainingMetrics(step=3, validation_score=0.75)

        self.aggregator.add_metrics(metrics1)
        self.aggregator.add_metrics(metrics2)
        self.aggregator.add_metrics(metrics3)

        best = self.aggregator.get_best_epoch("validation_score")
        self.assertIsNotNone(best)
        self.assertEqual(best.step, 2)
        self.assertEqual(best.validation_score, 0.8)

    def test_get_training_summary(self):
        """Test getting training summary."""
        metrics1 = TrainingMetrics(step=1, loss=0.5, training_score=0.7)
        metrics2 = TrainingMetrics(step=2, loss=0.3, training_score=0.8)

        self.aggregator.add_metrics(metrics1)
        self.aggregator.add_metrics(metrics2)

        summary = self.aggregator.get_training_summary()

        self.assertIn('total_epochs', summary)
        self.assertEqual(summary['total_epochs'], 2)
        self.assertIn('training_loss_mean', summary)
        self.assertAlmostEqual(summary['training_loss_mean'], 0.4, places=2)

    def test_empty_aggregator(self):
        """Test aggregator with no data."""
        best = self.aggregator.get_best_epoch("validation_score")
        self.assertIsNone(best)

        summary = self.aggregator.get_training_summary()
        self.assertEqual(summary, {})

        timeline = self.aggregator.get_lifecycle_timeline()
        self.assertEqual(timeline, [])


if __name__ == "__main__":
    unittest.main()