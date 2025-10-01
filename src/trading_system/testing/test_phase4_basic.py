"""
Phase 4: Basic Hyperparameter Optimization Tests

This module tests the core Phase 4 components with proper imports.
"""

import unittest
import sys
import os
import tempfile
import json
from datetime import datetime

# Add the src directory to Python path correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
src_dir = os.path.join(project_root, 'src')

# Add both src directory and project root to path
sys.path.insert(0, src_dir)
sys.path.insert(0, project_root)

try:
    # Try to import pandas for more comprehensive tests
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    # Try to import optuna for optimization tests
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class TestHyperparameterConfig(unittest.TestCase):
    """Test hyperparameter configuration functionality."""

    def test_problem_config_creation(self):
        """Test problem configuration creation."""
        from models.training.hyperparameter_config import ProblemConfig

        config = ProblemConfig(
            problem_type="regression",
            target_metric="val_score",
            metric_direction="maximize",
            n_samples=1000,
            n_features=20
        )

        self.assertEqual(config.problem_type, "regression")
        self.assertEqual(config.n_samples, 1000)
        self.assertEqual(config.n_features, 20)

    def test_problem_config_validation(self):
        """Test problem configuration validation."""
        from models.training.hyperparameter_config import ProblemConfig

        # Valid configuration
        config = ProblemConfig(problem_type="regression")
        self.assertTrue(config.validate())

        # Invalid problem type
        config = ProblemConfig(problem_type="invalid_type")
        self.assertFalse(config.validate())

    def test_model_config_creation(self):
        """Test model configuration creation."""
        from models.training.hyperparameter_config import ModelConfig

        config = ModelConfig(
            model_type="xgboost",
            model_family="tree_based",
            cv_folds=5
        )

        self.assertEqual(config.model_type, "xgboost")
        self.assertEqual(config.model_family, "tree_based")
        self.assertEqual(config.cv_folds, 5)

    def test_model_config_validation(self):
        """Test model configuration validation."""
        from models.training.hyperparameter_config import ModelConfig

        # Valid configuration
        config = ModelConfig(model_type="xgboost")
        self.assertTrue(config.validate())

        # Invalid model type
        config = ModelConfig(model_type="invalid_model")
        self.assertFalse(config.validate())

    def test_comprehensive_config_creation(self):
        """Test comprehensive configuration creation."""
        from models.training.hyperparameter_config import (
            HyperparameterOptimizationConfig, ProblemConfig, ModelConfig
        )

        config = HyperparameterOptimizationConfig(
            problem=ProblemConfig(problem_type="regression"),
            model=ModelConfig(model_type="xgboost"),
            study_name="test_study"
        )

        self.assertEqual(config.problem.problem_type, "regression")
        self.assertEqual(config.model.model_type, "xgboost")
        self.assertEqual(config.study_name, "test_study")

    def test_config_serialization(self):
        """Test configuration serialization."""
        from models.training.hyperparameter_config import create_default_config

        config = create_default_config("regression", "xgboost", 100)

        # Test to_dict
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertIn("problem", config_dict)

        # Test from_dict
        restored_config = HyperparameterOptimizationConfig.from_dict(config_dict)
        self.assertEqual(restored_config.problem.problem_type, config.problem.problem_type)


class TestSearchSpaceBuilder(unittest.TestCase):
    """Test search space builder functionality."""

    def test_builder_initialization(self):
        """Test search space builder initialization."""
        from models.training.search_space_builder import SearchSpaceBuilder

        builder = SearchSpaceBuilder()
        self.assertIsNotNone(builder.presets)
        self.assertGreater(len(builder.presets), 0)

    def test_preset_listing(self):
        """Test preset listing."""
        from models.training.search_space_builder import SearchSpaceBuilder

        builder = SearchSpaceBuilder()
        presets = builder.list_presets()
        self.assertIsInstance(presets, list)
        self.assertGreater(len(presets), 0)

    def test_preset_retrieval(self):
        """Test preset retrieval."""
        from models.training.search_space_builder import SearchSpaceBuilder

        builder = SearchSpaceBuilder()
        presets = builder.list_presets()

        if presets:
            preset = builder.get_preset(presets[0])
            self.assertIsNotNone(preset)
            self.assertGreater(len(preset.search_spaces), 0)

    def test_search_space_creation(self):
        """Test search space creation."""
        from models.training.search_space_builder import SearchSpaceBuilder, SearchSpace

        builder = SearchSpaceBuilder()

        # Create a simple search space
        search_spaces = {
            "param1": SearchSpace("param1", "int", low=1, high=10),
            "param2": SearchSpace("param2", "categorical", choices=["a", "b", "c"])
        }

        # Validate
        is_valid = builder._validate_search_spaces(search_spaces)
        self.assertTrue(is_valid)

        # Calculate combinations
        combinations = builder._calculate_total_combinations(search_spaces)
        self.assertGreater(combinations, 0)


class TestHyperparameterOptimizer(unittest.TestCase):
    """Test hyperparameter optimizer functionality."""

    def test_hyperparameter_config(self):
        """Test hyperparameter configuration."""
        from models.training.hyperparameter_optimizer import HyperparameterConfig

        config = HyperparameterConfig(
            n_trials=50,
            study_name="test_study",
            direction="maximize"
        )

        self.assertEqual(config.n_trials, 50)
        self.assertEqual(config.study_name, "test_study")
        self.assertEqual(config.direction, "maximize")

    def test_search_space_validation(self):
        """Test search space validation."""
        from models.training.hyperparameter_optimizer import SearchSpace

        # Valid search space
        space = SearchSpace("param1", "int", low=1, high=10)
        self.assertTrue(space.validate())

        # Invalid search space (categorical without choices)
        space = SearchSpace("param1", "categorical", choices=None)
        self.assertFalse(space.validate())

    def test_optimization_result_creation(self):
        """Test optimization result creation."""
        from models.training.hyperparameter_optimizer import OptimizationResult

        result = OptimizationResult(
            study_name="test_study",
            n_trials=10,
            best_params={"learning_rate": 0.1},
            best_score=0.8,
            best_trial_number=5,
            optimization_history=[],
            search_space={},
            optimization_time=60.0
        )

        self.assertEqual(result.study_name, "test_study")
        self.assertEqual(result.n_trials, 10)
        self.assertEqual(result.best_score, 0.8)


class TestOptunaIntegration(unittest.TestCase):
    """Test Optuna integration components."""

    def test_optuna_config_creation(self):
        """Test Optuna configuration creation."""
        from models.training.optuna_integration import OptunaConfig

        config = OptunaConfig(
            study_name="test_study",
            sampler_type="tpe",
            pruner_type="median",
            n_jobs=2
        )

        self.assertEqual(config.study_name, "test_study")
        self.assertEqual(config.sampler_type, "tpe")
        self.assertEqual(config.pruner_type, "median")
        self.assertEqual(config.n_jobs, 2)

    @unittest.skipUnless(OPTUNA_AVAILABLE, "Optuna not available")
    def test_study_manager_creation(self):
        """Test study manager creation when Optuna is available."""
        from models.training.optuna_integration import OptunaStudyManager

        config = OptunaConfig(study_name="test_study")
        manager = OptunaStudyManager(config=config)

        self.assertEqual(manager.config.study_name, "test_study")
        self.assertIsInstance(manager.studies, dict)


class TestTrialTracker(unittest.TestCase):
    """Test trial tracker functionality."""

    def test_trial_tracker_initialization(self):
        """Test trial tracker initialization."""
        from utils.experiment_tracking import NullExperimentTracker
        from utils.experiment_tracking.trial_tracker import TrialTracker

        base_tracker = NullExperimentTracker()
        trial_tracker = TrialTracker(
            base_tracker=base_tracker,
            study_name="test_study"
        )

        self.assertEqual(trial_tracker.study_name, "test_study")
        self.assertEqual(trial_tracker.base_tracker, base_tracker)
        self.assertIsNone(trial_tracker.current_trial)

    def test_trial_metadata_creation(self):
        """Test trial metadata creation."""
        from utils.experiment_tracking.trial_tracker import TrialMetadata

        metadata = TrialMetadata(
            trial_number=1,
            study_name="test_study",
            parameters={"learning_rate": 0.1}
        )

        self.assertEqual(metadata.trial_number, 1)
        self.assertEqual(metadata.study_name, "test_study")
        self.assertEqual(metadata.parameters["learning_rate"], 0.1)

    def test_study_metadata_creation(self):
        """Test study metadata creation."""
        from utils.experiment_tracking.trial_tracker import StudyMetadata

        metadata = StudyMetadata(
            study_name="test_study",
            start_time=datetime.now()
        )

        self.assertEqual(metadata.study_name, "test_study")
        self.assertIsNotNone(metadata.start_time)

    def test_trial_lifecycle(self):
        """Test trial lifecycle management."""
        from utils.experiment_tracking import NullExperimentTracker
        from utils.experiment_tracking.trial_tracker import TrialTracker

        base_tracker = NullExperimentTracker()
        trial_tracker = TrialTracker(base_tracker, "test_study")

        # Start trial
        run_id = trial_tracker.start_trial(
            trial_number=1,
            parameters={"learning_rate": 0.1}
        )
        self.assertIsNotNone(run_id)
        self.assertIsNotNone(trial_tracker.current_trial)

        # Log intermediate values
        trial_tracker.log_intermediate_value(step=1, value=0.5)
        trial_tracker.log_intermediate_value(step=2, value=0.6)
        self.assertEqual(len(trial_tracker.current_trial.intermediate_values), 2)

        # Complete trial
        trial_tracker.complete_trial(score=0.8, metrics={"val_loss": 0.2})
        self.assertIsNone(trial_tracker.current_trial)
        self.assertEqual(len(trial_tracker.trials_history), 1)
        self.assertEqual(trial_tracker.best_score, 0.8)

    def test_trial_failure_handling(self):
        """Test trial failure handling."""
        from utils.experiment_tracking import NullExperimentTracker
        from utils.experiment_tracking.trial_tracker import TrialTracker

        base_tracker = NullExperimentTracker()
        trial_tracker = TrialTracker(base_tracker, "test_study")

        # Start and fail trial
        trial_tracker.start_trial(trial_number=1, parameters={"bad": "params"})
        trial_tracker.fail_trial(error_message="Test failure")
        self.assertEqual(trial_tracker.trials_history[0].status, "failed")

    def test_trial_pruning(self):
        """Test trial pruning."""
        from utils.experiment_tracking import NullExperimentTracker
        from utils.experiment_tracking.trial_tracker import TrialTracker

        base_tracker = NullExperimentTracker()
        trial_tracker = TrialTracker(base_tracker, "test_study")

        # Start and prune trial
        trial_tracker.start_trial(trial_number=1, parameters={"learning_rate": 0.01})
        trial_tracker.prune_trial(step=2, reason="Test pruning")
        self.assertEqual(trial_tracker.trials_history[0].status, "pruned")

    def test_study_report_generation(self):
        """Test study report generation."""
        from utils.experiment_tracking import NullExperimentTracker
        from utils.experiment_tracking.trial_tracker import TrialTracker

        base_tracker = NullExperimentTracker()
        trial_tracker = TrialTracker(base_tracker, "test_study")

        # Add some trials
        for i in range(3):
            trial_tracker.start_trial(trial_number=i+1, parameters={"lr": 0.1 * (i+1)})
            trial_tracker.complete_trial(score=0.7 + i * 0.1)

        # Generate report
        report = trial_tracker.generate_study_report()
        self.assertIsInstance(report, dict)
        self.assertEqual(report["n_trials"], 3)
        self.assertIn("study_summary", report)
        self.assertIn("score_statistics", report)


class TestExperimentTracking(unittest.TestCase):
    """Test experiment tracking integration."""

    def test_null_tracker(self):
        """Test null experiment tracker."""
        from utils.experiment_tracking import NullExperimentTracker, ExperimentConfig

        tracker = NullExperimentTracker()
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            run_type="test"
        )

        # All operations should succeed without exceptions
        run_id = tracker.init_run(config)
        self.assertIsNotNone(run_id)

        tracker.log_params({"param1": "value1"})
        tracker.log_metrics({"metric1": 0.8})
        tracker.log_artifact("/tmp/test.txt", "test")
        tracker.log_alert("Test", "Test message", "info")
        tracker.finish_run()

        self.assertFalse(tracker.is_active())

    def test_experiment_config(self):
        """Test experiment configuration."""
        from utils.experiment_tracking import ExperimentConfig

        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training",
            tags=["test", "demo"],
            hyperparameters={"learning_rate": 0.01},
            metadata={"version": "1.0"}
        )

        self.assertEqual(config.project_name, "test_project")
        self.assertEqual(config.experiment_name, "test_experiment")
        self.assertEqual(config.run_type, "training")
        self.assertIn("test", config.tags)
        self.assertEqual(config.hyperparameters["learning_rate"], 0.01)


class TestIntegrationWorkflows(unittest.TestCase):
    """Test integration workflows."""

    def test_config_creation_workflow(self):
        """Test complete configuration creation workflow."""
        from models.training.hyperparameter_config import (
            create_default_config, create_fast_config, create_production_config
        )

        # Test factory functions
        default_config = create_default_config("regression", "xgboost", 100)
        fast_config = create_fast_config("classification", "lightgbm", 50)
        prod_config = create_production_config("ranking", "random_forest", 200)

        self.assertEqual(default_config.problem.problem_type, "regression")
        self.assertEqual(default_config.model.model_type, "xgboost")
        self.assertEqual(default_config.hyperparameter.n_trials, 100)

        self.assertEqual(fast_config.problem.problem_type, "classification")
        self.assertEqual(fast_config.model.model_type, "lightgbm")
        self.assertEqual(fast_config.hyperparameter.n_trials, 50)

        self.assertEqual(prod_config.problem.problem_type, "ranking")
        self.assertEqual(prod_config.model.model_type, "random_forest")
        self.assertEqual(prod_config.hyperparameter.n_trials, 200)

    def test_search_space_workflow(self):
        """Test search space building workflow."""
        from models.training.search_space_builder import SearchSpaceBuilder, SearchSpace

        builder = SearchSpaceBuilder()

        # Get preset
        presets = builder.list_presets()
        self.assertGreater(len(presets), 0)

        # Build search space
        search_spaces = builder.build_search_space(presets[0])
        self.assertGreater(len(search_spaces), 0)

        # Validate
        is_valid = builder._validate_search_spaces(search_spaces)
        self.assertTrue(is_valid)

        # Add custom parameter
        custom_param = SearchSpace("custom", "int", low=1, high=5)
        search_spaces["custom"] = custom_param

        self.assertIn("custom", search_spaces)
        self.assertEqual(len(search_spaces), len(search_spaces))

    def test_trial_tracking_workflow(self):
        """Test complete trial tracking workflow."""
        from utils.experiment_tracking import NullExperimentTracker
        from utils.experiment_tracking.trial_tracker import TrialTracker

        base_tracker = NullExperimentTracker()
        trial_tracker = TrialTracker(
            base_tracker=base_tracker,
            study_name="workflow_test",
            optimization_config={"model_type": "xgboost"}
        )

        # Simulate optimization workflow
        trial_configs = [
            {"learning_rate": 0.01, "n_estimators": 50},
            {"learning_rate": 0.05, "n_estimators": 100},
            {"learning_rate": 0.1, "n_estimators": 150}
        ]
        scores = [0.6, 0.8, 0.7]

        for i, (params, score) in enumerate(zip(trial_configs, scores)):
            # Start trial
            run_id = trial_tracker.start_trial(
                trial_number=i+1,
                parameters=params,
                trial_config={"model_type": "xgboost"}
            )

            # Log progress
            for step in range(1, 4):
                trial_tracker.log_intermediate_value(
                    step=step,
                    value=score * (0.5 + step * 0.2)
                )

            # Complete trial
            trial_tracker.complete_trial(
                score=score,
                metrics={"val_loss": 1 - score},
                evaluation_time=0.5
            )

        # Check results
        self.assertEqual(len(trial_tracker.trials_history), 3)
        self.assertEqual(trial_tracker.best_score, 0.8)
        self.assertEqual(trial_tracker.study_metadata.completed_trials, 3)

        # Generate report
        report = trial_tracker.generate_study_report()
        self.assertEqual(report["n_trials"], 3)
        self.assertAlmostEqual(report["study_summary"]["best_score"], 0.8)

        # Finish study
        trial_tracker.finish_study()
        self.assertIsNotNone(trial_tracker.study_metadata.end_time)


def main():
    """Run all tests."""
    # Configure logging
    import logging
    logging.basicConfig(level=logging.WARNING)

    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestHyperparameterConfig,
        TestSearchSpaceBuilder,
        TestHyperparameterOptimizer,
        TestOptunaIntegration,
        TestTrialTracker,
        TestExperimentTracking,
        TestIntegrationWorkflows
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)