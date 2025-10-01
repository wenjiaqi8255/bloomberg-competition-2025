"""
Phase 4: Hyperparameter Optimization System Tests

This module tests the complete hyperparameter optimization system
including Optuna integration, search space building, and trial tracking.

Key Components Tested:
- HyperparameterOptimizer core functionality
- SearchSpaceBuilder utilities
- OptunaStudyManager integration
- TrialTracker specialized tracking
- HyperparameterOptimizationConfig validation
- End-to-end optimization workflows
"""

import sys
import os
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from datetime import datetime
import pandas as pd
import numpy as np

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import Phase 4 components
from models.training.hyperparameter_optimizer import (
    HyperparameterOptimizer, HyperparameterConfig, SearchSpace, OptimizationResult
)
from models.training.search_space_builder import (
    SearchSpaceBuilder, SearchSpacePreset
)
from models.training.optuna_integration import (
    OptunaStudyManager, OptunaConfig, quick_optimize
)
from models.training.hyperparameter_config import (
    HyperparameterOptimizationConfig, ProblemConfig, ModelConfig,
    ResourceConfig, LoggingConfig, create_default_config
)
from utils.experiment_tracking import (
    NullExperimentTracker, ExperimentConfig
)
from utils.experiment_tracking.trial_tracker import (
    TrialTracker, TrialMetadata, StudyMetadata
)


class MockModel:
    """Simple mock model for testing optimization."""

    def __init__(self, **params):
        self.params = params
        self._trained = False

    def fit(self, X, y):
        self._trained = True
        return self

    def predict(self, X):
        if not self._trained:
            raise ValueError("Model not trained")
        return np.random.normal(0, 1, len(X))

    def score(self, X, y):
        if not self._trained:
            return 0.0
        # Simulate score based on parameters
        base_score = 0.5
        if "learning_rate" in self.params:
            base_score += 0.2 * np.exp(-self.params["learning_rate"] * 5)
        if "n_estimators" in self.params:
            base_score += 0.1 * np.log(self.params["n_estimators"] / 100)
        if "max_depth" in self.params:
            base_score += 0.05 * self.params["max_depth"] / 10

        # Add noise
        noise = np.random.normal(0, 0.1)
        return np.clip(base_score + noise, 0.0, 1.0)


class TestHyperparameterConfig(unittest.TestCase):
    """Test hyperparameter optimization configuration."""

    def test_problem_config_validation(self):
        """Test problem configuration validation."""
        # Valid configuration
        config = ProblemConfig(
            problem_type="regression",
            target_metric="val_score",
            metric_direction="maximize",
            n_samples=1000,
            n_features=20
        )
        self.assertTrue(config.validate())

        # Invalid problem type
        config = ProblemConfig(problem_type="invalid")
        self.assertFalse(config.validate())

        # Invalid metric direction
        config = ProblemConfig(metric_direction="invalid")
        self.assertFalse(config.validate())

        # Classification without classes
        config = ProblemConfig(
            problem_type="classification",
            n_classes=None
        )
        self.assertFalse(config.validate())

    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Valid configuration
        config = ModelConfig(
            model_type="xgboost",
            model_family="tree_based",
            cv_folds=5,
            validation_split=0.2
        )
        self.assertTrue(config.validate())

        # Invalid model type
        config = ModelConfig(model_type="invalid")
        self.assertFalse(config.validate())

        # Invalid CV folds
        config = ModelConfig(cv_folds=1)
        self.assertFalse(config.validate())

        # Invalid validation split
        config = ModelConfig(validation_split=1.5)
        self.assertFalse(config.validate())

    def test_resource_config_validation(self):
        """Test resource configuration validation."""
        # Valid configuration
        config = ResourceConfig(
            n_jobs=4,
            max_parallel_trials=2,
            per_trial_time_limit=300
        )
        self.assertTrue(config.validate())

        # Invalid jobs
        config = ResourceConfig(n_jobs=0)
        self.assertFalse(config.validate())

        # Invalid time limit
        config = ResourceConfig(per_trial_time_limit=-1)
        self.assertFalse(config.validate())

    def test_logging_config_validation(self):
        """Test logging configuration validation."""
        # Valid configuration
        config = LoggingConfig(
            track_trials=True,
            log_level="INFO",
            tracking_backend="null"
        )
        self.assertTrue(config.validate())

        # Invalid log level
        config = LoggingConfig(log_level="INVALID")
        self.assertFalse(config.validate())

        # Invalid backend
        config = LoggingConfig(tracking_backend="invalid")
        self.assertFalse(config.validate())

    def test_comprehensive_config_validation(self):
        """Test comprehensive configuration validation."""
        config = HyperparameterOptimizationConfig(
            problem=ProblemConfig(
                problem_type="regression",
                n_samples=1000,
                n_features=10
            ),
            model=ModelConfig(
                model_type="xgboost",
                cv_folds=5
            ),
            resources=ResourceConfig(
                n_jobs=2,
                max_parallel_trials=2
            ),
            logging=LoggingConfig(
                track_trials=True,
                tracking_backend="null"
            ),
            validate_config=True
        )
        self.assertTrue(config.validate())

    def test_config_serialization(self):
        """Test configuration serialization."""
        config = create_default_config("regression", "xgboost", 100)

        # Test to_dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn("problem", config_dict)
        self.assertIn("model", config_dict)

        # Test from_dict
        restored_config = HyperparameterOptimizationConfig.from_dict(config_dict)
        self.assertEqual(restored_config.problem.problem_type, config.problem.problem_type)
        self.assertEqual(restored_config.model.model_type, config.model.model_type)

        # Test file save/load
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            config.save_to_file(temp_path)
            loaded_config = HyperparameterOptimizationConfig.load_from_file(temp_path)
            self.assertEqual(loaded_config.problem.problem_type, config.problem.problem_type)
        finally:
            os.unlink(temp_path)

    def test_factory_functions(self):
        """Test configuration factory functions."""
        # Test default config
        config = create_default_config("regression", "xgboost", 100)
        self.assertEqual(config.problem.problem_type, "regression")
        self.assertEqual(config.model.model_type, "xgboost")
        self.assertEqual(config.hyperparameter.n_trials, 100)

        # Test fast config
        config = create_fast_config("classification", "lightgbm", 50)
        self.assertEqual(config.problem.problem_type, "classification")
        self.assertEqual(config.model.model_type, "lightgbm")
        self.assertEqual(config.hyperparameter.n_trials, 50)

        # Test production config
        config = create_production_config("ranking", "random_forest", 200)
        self.assertEqual(config.problem.problem_type, "ranking")
        self.assertEqual(config.model.model_type, "random_forest")
        self.assertEqual(config.hyperparameter.n_trials, 200)
        self.assertTrue(config.logging.detailed_logging)


class TestSearchSpaceBuilder(unittest.TestCase):
    """Test search space builder functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.builder = SearchSpaceBuilder()

    def test_default_presets_loaded(self):
        """Test that default presets are loaded."""
        self.assertGreater(len(self.builder.presets), 0)
        self.assertIn("xgboost_default", self.builder.presets)
        self.assertIn("lightgbm_default", self.builder.presets)
        self.assertIn("random_forest_default", self.builder.presets)

    def test_get_preset(self):
        """Test getting presets."""
        preset = self.builder.get_preset("xgboost_default")
        self.assertIsInstance(preset, SearchSpacePreset)
        self.assertEqual(preset.model_type, "xgboost")
        self.assertGreater(len(preset.search_spaces), 0)

        # Test invalid preset
        with self.assertRaises(ValueError):
            self.builder.get_preset("invalid_preset")

    def test_list_presets(self):
        """Test listing presets."""
        all_presets = self.builder.list_presets()
        self.assertIsInstance(all_presets, list)
        self.assertGreater(len(all_presets), 0)

        # Test filtering by model type
        xgb_presets = self.builder.list_presets("xgboost")
        self.assertIn("xgboost_default", xgb_presets)

    def test_build_search_space(self):
        """Test building search spaces from presets."""
        # Build from preset
        search_spaces = self.builder.build_search_space("xgboost_default")
        self.assertIsInstance(search_spaces, dict)
        self.assertGreater(len(search_spaces), 0)

        # Check for expected parameters
        self.assertIn("n_estimators", search_spaces)
        self.assertIn("max_depth", search_spaces)
        self.assertIn("learning_rate", search_spaces)

        # Test with exclusions
        search_spaces = self.builder.build_search_space(
            "xgboost_default",
            exclude_params=["learning_rate"]
        )
        self.assertNotIn("learning_rate", search_spaces)

        # Test with custom parameters
        custom_param = SearchSpace(
            name="custom_param",
            type="int",
            low=1,
            high=10
        )
        search_spaces = self.builder.build_search_space(
            "xgboost_default",
            custom_params={"custom_param": custom_param}
        )
        self.assertIn("custom_param", search_spaces)

    def test_intelligent_search_space(self):
        """Test intelligent search space creation."""
        # Small dataset
        search_spaces = self.builder.create_intelligent_search_space(
            "xgboost", data_size=500, n_features=10
        )
        self.assertIn("n_estimators", search_spaces)
        # Should use smaller ranges for small datasets
        self.assertLessEqual(search_spaces["n_estimators"].high, 200)

        # Large dataset
        search_spaces = self.builder.create_intelligent_search_space(
            "xgboost", data_size=50000, n_features=200
        )
        self.assertIn("n_estimators", search_spaces)
        # Should use larger ranges for large datasets
        self.assertGreaterEqual(search_spaces["n_estimators"].high, 200)

        # High dimensional data
        search_spaces = self.builder.create_intelligent_search_space(
            "xgboost", data_size=1000, n_features=200
        )
        if "colsample_bytree" in search_spaces:
            # Should use smaller feature fractions for high dimensional data
            self.assertLessEqual(search_spaces["colsample_bytree"].high, 0.8)

    def test_search_space_validation(self):
        """Test search space validation."""
        # Valid search spaces
        search_spaces = {
            "param1": SearchSpace("param1", "int", low=1, high=10),
            "param2": SearchSpace("param2", "categorical", choices=["a", "b", "c"])
        }
        self.assertTrue(self.builder._validate_search_spaces(search_spaces))

        # Invalid categorical (no choices)
        search_spaces = {
            "param1": SearchSpace("param1", "categorical", choices=None)
        }
        self.assertFalse(self.builder._validate_search_spaces(search_spaces))

        # Invalid range (low >= high)
        search_spaces = {
            "param1": SearchSpace("param1", "int", low=10, high=5)
        }
        self.assertFalse(self.builder._validate_search_spaces(search_spaces))

    def test_optimize_search_space(self):
        """Test search space optimization."""
        # Create large search space
        search_spaces = {
            "param1": SearchSpace("param1", "int", low=1, high=100, step=1),
            "param2": SearchSpace("param2", "float", low=0.0, high=1.0, step=0.001),
            "param3": SearchSpace("param3", "categorical", choices=list(range(100)))
        }

        # Optimize to reduce combinations
        optimized = self.builder.optimize_search_space(search_spaces, max_total_combinations=1000)

        # Should have reduced combinations
        original_combinations = self.builder._calculate_total_combinations(search_spaces)
        optimized_combinations = self.builder._calculate_total_combinations(optimized)
        self.assertLess(optimized_combinations, original_combinations)
        self.assertLessEqual(optimized_combinations, 1000)

    def test_parameter_statistics(self):
        """Test parameter statistics generation."""
        search_spaces = self.builder.build_search_space("xgboost_default")
        stats_df = self.builder.get_parameter_statistics(search_spaces)

        self.assertIsInstance(stats_df, pd.DataFrame)
        self.assertGreater(len(stats_df), 0)
        self.assertIn("parameter", stats_df.columns)
        self.assertIn("type", stats_df.columns)


class TestHyperparameterOptimizer(unittest.TestCase):
    """Test hyperparameter optimizer core functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.tracker = NullExperimentTracker()
        self.config = HyperparameterConfig(n_trials=10, study_name="test_study")

    @patch('models.training.hyperparameter_optimizer.OPTUNA_AVAILABLE', True)
    @patch('models.training.hyperparameter_optimizer.optuna')
    def test_optimizer_initialization(self, mock_optuna):
        """Test optimizer initialization."""
        # Mock Optuna components
        mock_study = Mock()
        mock_study.return_value = mock_study
        mock_optuna.create_study.return_value = mock_study

        optimizer = HyperparameterOptimizer(
            config=self.config,
            experiment_tracker=self.tracker
        )

        self.assertEqual(optimizer.config, self.config)
        self.assertEqual(optimizer.experiment_tracker, self.tracker)
        self.assertEqual(optimizer.study_name, "test_study")

    @patch('models.training.hyperparameter_optimizer.OPTUNA_AVAILABLE', True)
    @patch('models.training.hyperparameter_optimizer.optuna')
    def test_search_space_addition(self, mock_optuna):
        """Test adding search spaces."""
        mock_study = Mock()
        mock_optuna.create_study.return_value = mock_study

        optimizer = HyperparameterOptimizer(config=self.config)

        search_spaces = {
            "param1": SearchSpace("param1", "int", low=1, high=10),
            "param2": SearchSpace("param2", "categorical", choices=["a", "b"])
        }

        optimizer.add_search_space(search_spaces)
        self.assertEqual(len(optimizer.search_space), 2)
        self.assertIn("param1", optimizer.search_space)
        self.assertIn("param2", optimizer.search_space)

    @patch('models.training.hyperparameter_optimizer.OPTUNA_AVAILABLE', True)
    @patch('models.training.hyperparameter_optimizer.optuna')
    def test_default_search_spaces_creation(self, mock_optuna):
        """Test creating default search spaces."""
        mock_study = Mock()
        mock_optuna.create_study.return_value = mock_study

        optimizer = HyperparameterOptimizer(config=self.config)

        # Test XGBoost default
        optimizer.create_default_search_spaces("xgboost")
        self.assertIn("n_estimators", optimizer.search_space)
        self.assertIn("learning_rate", optimizer.search_space)
        self.assertIn("max_depth", optimizer.search_space)

        # Test LightGBM default
        optimizer.search_space.clear()
        optimizer.create_default_search_spaces("lightgbm")
        self.assertIn("n_estimators", optimizer.search_space)
        self.assertIn("num_leaves", optimizer.search_space)
        self.assertIn("learning_rate", optimizer.search_space)

    @patch('models.training.hyperparameter_optimizer.OPTUNA_AVAILABLE', False)
    def test_optuna_unavailable(self):
        """Test handling when Optuna is unavailable."""
        with self.assertRaises(ImportError):
            HyperparameterOptimizer(config=self.config)

    def test_hyperparameter_config_validation(self):
        """Test hyperparameter configuration validation."""
        # Valid config
        config = HyperparameterConfig(
            n_trials=100,
            study_name="test",
            direction="maximize",
            metric_name="val_score"
        )
        # Should not raise exception
        self.assertIsInstance(config, HyperparameterConfig)

        # Test with various settings
        config = HyperparameterConfig(
            n_trials=50,
            sampler="random",
            pruner="hyperband",
            early_stopping=True,
            track_trials=True
        )
        self.assertEqual(config.n_trials, 50)
        self.assertEqual(config.sampler, "random")
        self.assertEqual(config.pruner, "hyperband")
        self.assertTrue(config.early_stopping)
        self.assertTrue(config.track_trials)


class TestOptunaStudyManager(unittest.TestCase):
    """Test Optuna study manager functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = OptunaConfig(study_name="test_study")

    @patch('models.training.optuna_integration.OPTUNA_AVAILABLE', True)
    @patch('models.training.optuna_integration.optuna')
    def test_study_manager_initialization(self, mock_optuna):
        """Test study manager initialization."""
        manager = OptunaStudyManager(config=self.config)

        self.assertEqual(manager.config, self.config)
        self.assertIsInstance(manager.studies, dict)

    @patch('models.training.optuna_integration.OPTUNA_AVAILABLE', True)
    @patch('models.training.optuna_integration.optuna')
    def test_sampler_creation(self, mock_optuna):
        """Test sampler creation."""
        manager = OptunaStudyManager(config=self.config)

        # Test TPE sampler
        self.config.sampler_type = "tpe"
        sampler = manager._create_sampler()
        mock_optuna.TPESampler.assert_called()

        # Test Random sampler
        self.config.sampler_type = "random"
        sampler = manager._create_sampler()
        mock_optuna.RandomSampler.assert_called()

        # Test invalid sampler
        self.config.sampler_type = "invalid"
        with self.assertRaises(ValueError):
            manager._create_sampler()

    @patch('models.training.optuna_integration.OPTUNA_AVAILABLE', True)
    @patch('models.training.optuna_integration.optuna')
    def test_pruner_creation(self, mock_optuna):
        """Test pruner creation."""
        manager = OptunaStudyManager(config=self.config)

        # Test Median pruner
        self.config.pruner_type = "median"
        pruner = manager._create_pruner()
        mock_optuna.MedianPruner.assert_called()

        # Test Hyperband pruner
        self.config.pruner_type = "hyperband"
        pruner = manager._create_pruner()
        mock_optuna.HyperbandPruner.assert_called()

        # Test Custom pruner
        self.config.pruner_type = "custom"
        pruner = manager._create_pruner()
        self.assertIsInstance(pruner, manager.__class__.__module__.CustomPruner)

        # Test no pruning
        self.config.pruner_type = "none"
        pruner = manager._create_pruner()
        self.assertIsNone(pruner)

    @patch('models.training.optuna_integration.OPTUNA_AVAILABLE', False)
    def test_optuna_unavailable(self):
        """Test handling when Optuna is unavailable."""
        with self.assertRaises(ImportError):
            OptunaStudyManager(config=self.config)

    def test_optuna_config_validation(self):
        """Test Optuna configuration validation."""
        # Valid config
        config = OptunaConfig(
            study_name="test",
            sampler_type="tpe",
            pruner_type="median",
            n_jobs=2
        )
        self.assertIsInstance(config, OptunaConfig)
        self.assertEqual(config.study_name, "test")
        self.assertEqual(config.sampler_type, "tpe")
        self.assertEqual(config.pruner_type, "median")
        self.assertEqual(config.n_jobs, 2)


class TestTrialTracker(unittest.TestCase):
    """Test trial tracker functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_tracker = NullExperimentTracker()
        self.trial_tracker = TrialTracker(
            base_tracker=self.base_tracker,
            study_name="test_study",
            optimization_config={"test": "config"}
        )

    def test_trial_tracker_initialization(self):
        """Test trial tracker initialization."""
        self.assertEqual(self.trial_tracker.study_name, "test_study")
        self.assertEqual(self.trial_tracker.base_tracker, self.base_tracker)
        self.assertEqual(self.trial_tracker.optimization_config["test"], "config")
        self.assertIsNone(self.trial_tracker.current_trial)

    def test_start_trial(self):
        """Test starting a trial."""
        parameters = {"learning_rate": 0.1, "n_estimators": 100}
        trial_config = {"model_type": "xgboost"}

        run_id = self.trial_tracker.start_trial(
            trial_number=1,
            parameters=parameters,
            trial_config=trial_config
        )

        self.assertIsNotNone(run_id)
        self.assertIsNotNone(self.trial_tracker.current_trial)
        self.assertEqual(self.trial_tracker.current_trial.trial_number, 1)
        self.assertEqual(self.trial_tracker.current_trial.parameters, parameters)
        self.assertEqual(self.trial_tracker.current_trial.status, "running")
        self.assertEqual(self.trial_tracker.study_metadata.total_trials, 1)

    def test_complete_trial(self):
        """Test completing a trial."""
        # Start trial
        parameters = {"learning_rate": 0.1}
        self.trial_tracker.start_trial(trial_number=1, parameters=parameters)

        # Complete trial
        score = 0.85
        metrics = {"val_loss": 0.2, "val_accuracy": 0.9}
        self.trial_tracker.complete_trial(
            score=score,
            metrics=metrics,
            evaluation_time=10.5
        )

        # Check results
        self.assertIsNone(self.trial_tracker.current_trial)  # Should be cleared
        self.assertEqual(len(self.trial_tracker.trials_history), 1)
        self.assertEqual(self.trial_tracker.best_score, score)
        self.assertEqual(self.trial_tracker.study_metadata.completed_trials, 1)

    def test_prune_trial(self):
        """Test pruning a trial."""
        # Start trial
        parameters = {"learning_rate": 0.1}
        self.trial_tracker.start_trial(trial_number=1, parameters=parameters)

        # Prune trial
        self.trial_tracker.prune_trial(
            step=5,
            reason="No improvement",
            intermediate_value=0.3
        )

        # Check results
        self.assertIsNone(self.trial_tracker.current_trial)
        self.assertEqual(len(self.trial_tracker.trials_history), 1)
        self.assertEqual(self.trial_tracker.trials_history[0].status, "pruned")
        self.assertEqual(self.trial_tracker.trials_history[0].pruned_step, 5)
        self.assertEqual(self.trial_tracker.study_metadata.pruned_trials, 1)

    def test_fail_trial(self):
        """Test failing a trial."""
        # Start trial
        parameters = {"learning_rate": 0.1}
        self.trial_tracker.start_trial(trial_number=1, parameters=parameters)

        # Fail trial
        self.trial_tracker.fail_trial(
            error_message="Training failed",
            exception_type="ValueError"
        )

        # Check results
        self.assertIsNone(self.trial_tracker.current_trial)
        self.assertEqual(len(self.trial_tracker.trials_history), 1)
        self.assertEqual(self.trial_tracker.trials_history[0].status, "failed")
        self.assertEqual(self.trial_tracker.trials_history[0].failure_reason, "Training failed")
        self.assertEqual(self.trial_tracker.study_metadata.failed_trials, 1)

    def test_intermediate_values(self):
        """Test logging intermediate values."""
        # Start trial
        parameters = {"learning_rate": 0.1}
        self.trial_tracker.start_trial(trial_number=1, parameters=parameters)

        # Log intermediate values
        self.trial_tracker.log_intermediate_value(step=1, value=0.5)
        self.trial_tracker.log_intermediate_value(step=2, value=0.6, metrics={"loss": 0.8})

        # Check values were recorded
        self.assertEqual(len(self.trial_tracker.current_trial.intermediate_values), 2)
        self.assertEqual(self.trial_tracker.current_trial.intermediate_values[0], (1, 0.5))
        self.assertEqual(self.trial_tracker.current_trial.intermediate_values[1], (2, 0.6))
        self.assertEqual(len(self.trial_tracker.current_trial.metrics_history), 1)

    def test_trials_dataframe(self):
        """Test getting trials as DataFrame."""
        # Add some trials
        parameters = {"learning_rate": 0.1}
        self.trial_tracker.start_trial(trial_number=1, parameters=parameters)
        self.trial_tracker.complete_trial(score=0.8)

        parameters = {"learning_rate": 0.05}
        self.trial_tracker.start_trial(trial_number=2, parameters=parameters)
        self.trial_tracker.complete_trial(score=0.9)

        # Get DataFrame
        df = self.trial_tracker.get_trials_dataframe()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("trial_number", df.columns)
        self.assertIn("score", df.columns)

    def test_study_report_generation(self):
        """Test study report generation."""
        # Add some trials
        for i in range(3):
            parameters = {"learning_rate": 0.1 * (i + 1)}
            self.trial_tracker.start_trial(trial_number=i + 1, parameters=parameters)
            self.trial_tracker.complete_trial(score=0.7 + i * 0.1)

        # Generate report
        report = self.trial_tracker.generate_study_report()

        self.assertIsInstance(report, dict)
        self.assertIn("study_summary", report)
        self.assertIn("score_statistics", report)
        self.assertIn("best_trial", report)
        self.assertIn("n_trials", report)
        self.assertEqual(report["n_trials"], 3)

    def test_parameter_importance(self):
        """Test parameter importance calculation."""
        # Add trials with different parameter effects
        param_configs = [
            {"learning_rate": 0.01, "n_estimators": 50},
            {"learning_rate": 0.05, "n_estimators": 100},
            {"learning_rate": 0.1, "n_estimators": 150},
            {"learning_rate": 0.01, "n_estimators": 200},
            {"learning_rate": 0.05, "n_estimators": 250}
        ]

        scores = [0.6, 0.7, 0.8, 0.65, 0.75]

        for i, (params, score) in enumerate(zip(param_configs, scores)):
            self.trial_tracker.start_trial(trial_number=i + 1, parameters=params)
            self.trial_tracker.complete_trial(score=score)

        # Calculate importance
        importance = self.trial_tracker.get_parameter_importance()
        self.assertIsInstance(importance, dict)
        # Should have some importance values for parameters that vary
        self.assertGreater(len(importance), 0)

    def test_study_finish(self):
        """Test finishing study."""
        # Add a trial
        parameters = {"learning_rate": 0.1}
        self.trial_tracker.start_trial(trial_number=1, parameters=parameters)
        self.trial_tracker.complete_trial(score=0.8)

        # Finish study
        self.trial_tracker.finish_study()

        self.assertIsNotNone(self.trial_tracker.study_metadata.end_time)
        self.assertEqual(self.trial_tracker.study_metadata.completed_trials, 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for Phase 4 components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('models.training.optuna_integration.OPTUNA_AVAILABLE', True)
    @patch('models.training.optuna_integration.optuna')
    def test_config_builder_integration(self, mock_optuna):
        """Test integration between config and search space builder."""
        # Create comprehensive config
        config = create_default_config("regression", "xgboost", 50)
        config.study_name = "integration_test"

        # Create search space builder
        builder = SearchSpaceBuilder()

        # Build search spaces from config
        search_spaces = builder.build_search_space("xgboost_default")
        config.search_spaces = search_spaces

        # Validate config
        self.assertTrue(config.validate())
        self.assertGreater(len(config.search_spaces), 0)

    @patch('models.training.optuna_integration.OPTUNA_AVAILABLE', True)
    @patch('models.training.optuna_integration.optuna')
    def test_tracker_integration(self, mock_optuna):
        """Test integration between tracker and optimizer components."""
        # Create trial tracker
        base_tracker = NullExperimentTracker()
        trial_tracker = TrialTracker(
            base_tracker=base_tracker,
            study_name="integration_test"
        )

        # Simulate optimization workflow
        parameters_list = [
            {"learning_rate": 0.01, "n_estimators": 50},
            {"learning_rate": 0.05, "n_estimators": 100},
            {"learning_rate": 0.1, "n_estimators": 150}
        ]
        scores = [0.6, 0.8, 0.7]

        for i, (params, score) in enumerate(zip(parameters_list, scores)):
            # Start trial
            run_id = trial_tracker.start_trial(
                trial_number=i + 1,
                parameters=params
            )
            self.assertIsNotNone(run_id)

            # Log progress
            trial_tracker.log_intermediate_value(step=1, value=score * 0.8)
            trial_tracker.log_intermediate_value(step=2, value=score * 0.9)

            # Complete trial
            trial_tracker.complete_trial(
                score=score,
                metrics={"val_loss": 1 - score},
                evaluation_time=5.0
            )

        # Check final state
        self.assertEqual(len(trial_tracker.trials_history), 3)
        self.assertEqual(trial_tracker.best_score, 0.8)
        self.assertEqual(trial_tracker.study_metadata.completed_trials, 3)

        # Generate and check report
        report = trial_tracker.generate_study_report()
        self.assertEqual(report["n_trials"], 3)
        self.assertAlmostEqual(report["study_summary"]["best_score"], 0.8)

    @patch('models.training.optuna_integration.OPTUNA_AVAILABLE', True)
    @patch('models.training.optuna_integration.optuna')
    def test_quick_optimize_integration(self, mock_optuna):
        """Test quick optimize function integration."""
        # Mock Optuna study and trial
        mock_trial = Mock()
        mock_trial.number = 0
        mock_trial.value = 0.8
        mock_trial.params = {"learning_rate": 0.1, "n_estimators": 100}

        mock_study = Mock()
        mock_study.best_trial = mock_trial
        mock_study.trials = [mock_trial]
        mock_study.best_value = 0.8
        mock_study.best_params = {"learning_rate": 0.1, "n_estimators": 100}

        mock_optuna.create_study.return_value = mock_study

        # Define search spaces
        search_spaces = {
            "learning_rate": SearchSpace("learning_rate", "float", low=0.01, high=0.3),
            "n_estimators": SearchSpace("n_estimators", "int", low=50, high=200)
        }

        # Define objective function
        def objective_function(params):
            return 0.5 + params["learning_rate"] * 2 - params["n_estimators"] / 1000

        # Run quick optimization
        result = quick_optimize(
            objective_function=objective_function,
            search_spaces=search_spaces,
            n_trials=10,
            study_name="quick_test"
        )

        # Check results
        self.assertIsInstance(result, OptimizationResult)
        self.assertEqual(result.study_name, "quick_test")
        self.assertEqual(result.n_trials, 1)  # Mock only has one trial
        self.assertIn("learning_rate", result.best_params)
        self.assertIn("n_estimators", result.best_params)

    def test_end_to_end_config_workflow(self):
        """Test end-to-end configuration workflow."""
        # Create configuration
        config = create_production_config("regression", "xgboost", 100)
        config.study_name = "e2e_test"

        # Validate
        self.assertTrue(config.validate())

        # Adjust for environment
        config.adjust_for_environment()

        # Get recommendations
        effective_trials = config.get_effective_trials()
        recommended_sampler = config.get_recommended_sampler()
        recommended_pruner = config.get_recommended_pruner()

        # Check recommendations
        self.assertGreater(effective_trials, 0)
        self.assertIn(recommended_sampler, ["tpe", "random", "cmaes", "grid"])
        self.assertIn(recommended_pruner, ["median", "hyperband", "none"])

        # Get summary
        summary = config.get_summary()
        self.assertIsInstance(summary, str)
        self.assertIn(config.study_name, summary)
        self.assertIn(config.problem.problem_type, summary)

        # Test serialization
        config_dict = config.to_dict()
        restored_config = HyperparameterOptimizationConfig.from_dict(config_dict)
        self.assertEqual(restored_config.study_name, config.study_name)

    def test_search_space_validation_integration(self):
        """Test search space validation in integration context."""
        builder = SearchSpaceBuilder()

        # Create valid search spaces
        valid_spaces = {
            "param1": SearchSpace("param1", "int", low=1, high=10),
            "param2": SearchSpace("param2", "categorical", choices=["a", "b", "c"])
        }
        self.assertTrue(builder._validate_search_spaces(valid_spaces))

        # Create invalid search spaces
        invalid_spaces = {
            "param1": SearchSpace("param1", "categorical", choices=None),  # Invalid categorical
            "param2": SearchSpace("param2", "int", low=10, high=5)  # Invalid range
        }
        self.assertFalse(builder._validate_search_spaces(invalid_spaces))

        # Check validation errors
        self.assertGreater(len(builder.validation_errors), 0)


class TestErrorHandling(unittest.TestCase):
    """Test error handling in Phase 4 components."""

    def test_config_validation_errors(self):
        """Test configuration validation errors."""
        # Invalid problem config
        with self.assertRaises(ValueError):
            config = HyperparameterOptimizationConfig(
                validate_config=True,
                problem=ProblemConfig(problem_type="invalid")
            )

        # Invalid model config
        with self.assertRaises(ValueError):
            config = HyperparameterOptimizationConfig(
                validate_config=True,
                model=ModelConfig(model_type="invalid")
            )

    def test_search_space_builder_errors(self):
        """Test search space builder error handling."""
        builder = SearchSpaceBuilder()

        # Invalid preset
        with self.assertRaises(ValueError):
            builder.get_preset("nonexistent_preset")

        # Empty search space for optimization
        empty_spaces = {}
        optimized = builder.optimize_search_space(empty_spaces)
        self.assertEqual(len(optimized), 0)

    def test_trial_tracker_errors(self):
        """Test trial tracker error handling."""
        tracker = TrialTracker(
            base_tracker=NullExperimentTracker(),
            study_name="test"
        )

        # Try to complete trial without starting one
        tracker.complete_trial(score=0.8)  # Should not raise exception
        self.assertIsNone(tracker.current_trial)

        # Try to log intermediate value without active trial
        tracker.log_intermediate_value(step=1, value=0.5)  # Should not raise exception

        # Generate report without trials
        report = tracker.generate_study_report()
        self.assertEqual(report["n_trials"], 0)
        self.assertIsNone(report["best_trial"])


if __name__ == "__main__":
    # Configure logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)

    # Run tests
    unittest.main(verbosity=2)