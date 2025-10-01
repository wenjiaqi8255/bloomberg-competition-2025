"""
Phase 4: Simple Verification Script

This script verifies Phase 4 functionality without external dependencies.
It tests the core components and their integration.
"""

import sys
import os
import tempfile
import json
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Simple test framework
class SimpleTest:
    def __init__(self, name):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.errors = []

    def assert_equal(self, a, b, msg=""):
        try:
            if a == b:
                self.passed += 1
            else:
                self.failed += 1
                self.errors.append(f"{msg}: Expected {b}, got {a}")
        except Exception as e:
            self.failed += 1
            self.errors.append(f"{msg}: Exception {e}")

    def assert_true(self, condition, msg=""):
        try:
            if condition:
                self.passed += 1
            else:
                self.failed += 1
                self.errors.append(f"{msg}: Expected True")
        except Exception as e:
            self.failed += 1
            self.errors.append(f"{msg}: Exception {e}")

    def assert_not_none(self, value, msg=""):
        try:
            if value is not None:
                self.passed += 1
            else:
                self.failed += 1
                self.errors.append(f"{msg}: Expected not None")
        except Exception as e:
            self.failed += 1
            self.errors.append(f"{msg}: Exception {e}")

    def print_results(self):
        if self.failed == 0:
            print(f"✅ {self.name}: {self.passed} tests passed")
        else:
            print(f"❌ {self.name}: {self.passed} passed, {self.failed} failed")
            for error in self.errors:
                print(f"   - {error}")


def test_hyperparameter_config():
    """Test hyperparameter configuration functionality."""
    test = SimpleTest("HyperparameterOptimizationConfig")

    try:
        # Import configuration classes
        from models.training.hyperparameter_config import (
            ProblemConfig, ModelConfig, ResourceConfig, LoggingConfig,
            HyperparameterOptimizationConfig, create_default_config
        )

        # Test ProblemConfig
        problem = ProblemConfig(
            problem_type="regression",
            n_samples=1000,
            n_features=10
        )
        test.assert_true(problem.validate(), "ProblemConfig validation")
        test.assert_equal(problem.problem_type, "regression", "Problem type")

        # Test ModelConfig
        model = ModelConfig(
            model_type="xgboost",
            model_family="tree_based",
            cv_folds=5
        )
        test.assert_true(model.validate(), "ModelConfig validation")
        test.assert_equal(model.model_type, "xgboost", "Model type")

        # Test ResourceConfig
        resources = ResourceConfig(
            n_jobs=2,
            max_parallel_trials=2
        )
        test.assert_true(resources.validate(), "ResourceConfig validation")
        test.assert_equal(resources.n_jobs, 2, "Resource jobs")

        # Test LoggingConfig
        logging = LoggingConfig(
            track_trials=True,
            log_level="INFO"
        )
        test.assert_true(logging.validate(), "LoggingConfig validation")
        test.assert_true(logging.track_trials, "Track trials")

        # Test factory function
        config = create_default_config("regression", "xgboost", 100)
        test.assert_not_none(config, "Default config created")
        test.assert_equal(config.problem.problem_type, "regression", "Factory problem type")
        test.assert_equal(config.model.model_type, "xgboost", "Factory model type")
        test.assert_equal(config.hyperparameter.n_trials, 100, "Factory trials")

        # Test configuration serialization
        config_dict = config.to_dict()
        test.assert_not_none(config_dict, "Config dict")
        test.assert_true("problem" in config_dict, "Config dict has problem")

        # Test configuration update
        updated = config.update(study_name="test_update")
        test.assert_equal(updated.study_name, "test_update", "Config update")

    except ImportError as e:
        test.failed += 1
        test.errors.append(f"Import error: {e}")

    test.print_results()
    return test.failed == 0


def test_search_space():
    """Test search space functionality."""
    test = SimpleTest("SearchSpace")

    try:
        # Import search space classes
        from models.training.hyperparameter_optimizer import SearchSpace
        from models.training.search_space_builder import SearchSpaceBuilder

        # Test SearchSpace creation
        space = SearchSpace(
            name="param1",
            type="int",
            low=1,
            high=10
        )
        test.assert_true(space.validate(), "SearchSpace validation")
        test.assert_equal(space.name, "param1", "SearchSpace name")

        # Test invalid SearchSpace
        invalid_space = SearchSpace(
            name="invalid",
            type="categorical",
            choices=None
        )
        test.assert_true(not invalid_space.validate(), "Invalid SearchSpace validation")

        # Test SearchSpaceBuilder
        builder = SearchSpaceBuilder()
        test.assert_not_none(builder.presets, "Presets loaded")
        test.assert_true(len(builder.presets) > 0, "Presets not empty")

        # Test preset listing
        presets = builder.list_presets()
        test.assert_true(len(presets) > 0, "Presets listed")

        # Test preset retrieval
        if presets:
            preset = builder.get_preset(presets[0])
            test.assert_not_none(preset, "Preset retrieved")
            test.assert_not_none(preset.search_spaces, "Preset has search spaces")

    except ImportError as e:
        test.failed += 1
        test.errors.append(f"Import error: {e}")

    test.print_results()
    return test.failed == 0


def test_hyperparameter_optimizer():
    """Test hyperparameter optimizer functionality."""
    test = SimpleTest("HyperparameterOptimizer")

    try:
        # Import optimizer classes
        from models.training.hyperparameter_optimizer import (
            HyperparameterConfig, HyperparameterOptimizer
        )

        # Test HyperparameterConfig
        config = HyperparameterConfig(
            n_trials=50,
            study_name="test_study",
            direction="maximize"
        )
        test.assert_equal(config.n_trials, 50, "HyperparameterConfig trials")
        test.assert_equal(config.study_name, "test_study", "HyperparameterConfig study name")

        # Test optimizer with mock Optuna (unavailable case)
        # Since Optuna is not available, we test the import error handling
        try:
            # This should fail gracefully if Optuna is not available
            class MockOptunaUnavailable:
                OPTUNA_AVAILABLE = False
                optuna = None

            # Mock the availability check
            original_available = None
            try:
                import models.training.hyperparameter_optimizer as hpo_module
                original_available = hpo_module.OPTUNA_AVAILABLE
                hpo_module.OPTUNA_AVAILABLE = False

                # Now test the optimizer
                try:
                    optimizer = HyperparameterOptimizer(config=config)
                    test.failed += 1
                    test.errors.append("Expected ImportError for missing Optuna")
                except ImportError:
                    test.passed += 1
                finally:
                    hpo_module.OPTUNA_AVAILABLE = original_available

            except Exception as e:
                test.failed += 1
                test.errors.append(f"Unexpected error: {e}")

        except Exception as e:
            test.failed += 1
            test.errors.append(f"Mock setup error: {e}")

    except ImportError as e:
        test.failed += 1
        test.errors.append(f"Import error: {e}")

    test.print_results()
    return test.failed == 0


def test_trial_tracker():
    """Test trial tracker functionality."""
    test = SimpleTest("TrialTracker")

    try:
        # Import tracker classes
        from utils.experiment_tracking import NullExperimentTracker
        from utils.experiment_tracking.trial_tracker import TrialTracker, TrialMetadata

        # Create trial tracker
        base_tracker = NullExperimentTracker()
        trial_tracker = TrialTracker(
            base_tracker=base_tracker,
            study_name="test_study"
        )

        test.assert_equal(trial_tracker.study_name, "test_study", "TrialTracker study name")
        test.assert_none(trial_tracker.current_trial, "No current trial initially")

        # Test starting a trial
        run_id = trial_tracker.start_trial(
            trial_number=1,
            parameters={"learning_rate": 0.1, "n_estimators": 100}
        )
        test.assert_not_none(run_id, "Trial started")
        test.assert_not_none(trial_tracker.current_trial, "Current trial exists")

        # Test logging intermediate values
        trial_tracker.log_intermediate_value(step=1, value=0.5)
        trial_tracker.log_intermediate_value(step=2, value=0.6)
        test.assert_equal(len(trial_tracker.current_trial.intermediate_values), 2, "Intermediate values logged")

        # Test completing trial
        trial_tracker.complete_trial(
            score=0.8,
            metrics={"val_loss": 0.2},
            evaluation_time=1.0
        )
        test.assert_none(trial_tracker.current_trial, "Current trial cleared after completion")
        test.assert_equal(len(trial_tracker.trials_history), 1, "Trial in history")
        test.assert_equal(trial_tracker.best_score, 0.8, "Best score updated")

        # Test trial failure
        trial_tracker.start_trial(trial_number=2, parameters={"bad": "params"})
        trial_tracker.fail_trial(error_message="Test failure")
        test.assert_equal(trial_tracker.trials_history[1].status, "failed", "Trial failed correctly")

        # Test trial pruning
        trial_tracker.start_trial(trial_number=3, parameters={"learning_rate": 0.01})
        trial_tracker.prune_trial(step=2, reason="Test pruning")
        test.assert_equal(trial_tracker.trials_history[2].status, "pruned", "Trial pruned correctly")

        # Test report generation
        report = trial_tracker.generate_study_report()
        test.assert_not_none(report, "Report generated")
        test.assert_equal(report["n_trials"], 3, "Report trial count")

    except ImportError as e:
        test.failed += 1
        test.errors.append(f"Import error: {e}")

    test.print_results()
    return test.failed == 0


def test_optuna_integration():
    """Test Optuna integration components."""
    test = SimpleTest("OptunaIntegration")

    try:
        # Import Optuna integration classes
        from models.training.optuna_integration import OptunaConfig, OptunaStudyManager

        # Test OptunaConfig
        config = OptunaConfig(
            study_name="test_study",
            sampler_type="tpe",
            pruner_type="median",
            n_jobs=2
        )
        test.assert_equal(config.study_name, "test_study", "OptunaConfig study name")
        test.assert_equal(config.sampler_type, "tpe", "OptunaConfig sampler type")

        # Test study manager with unavailable Optuna
        try:
            import models.training.optuna_integration as optuna_module
            original_available = optuna_module.OPTUNA_AVAILABLE
            optuna_module.OPTUNA_AVAILABLE = False

            try:
                manager = OptunaStudyManager(config=config)
                test.failed += 1
                test.errors.append("Expected ImportError for missing Optuna")
            except ImportError:
                test.passed += 1
            finally:
                optuna_module.OPTUNA_AVAILABLE = original_available

        except Exception as e:
            test.failed += 1
            test.errors.append(f"Unexpected error: {e}")

    except ImportError as e:
        test.failed += 1
        test.errors.append(f"Import error: {e}")

    test.print_results()
    return test.failed == 0


def test_experiment_tracking_integration():
    """Test experiment tracking integration."""
    test = SimpleTest("ExperimentTrackingIntegration")

    try:
        # Import tracking classes
        from utils.experiment_tracking import NullExperimentTracker, ExperimentConfig

        # Test NullExperimentTracker
        tracker = NullExperimentTracker()
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_experiment",
            run_type="analysis"
        )

        # Test tracker methods
        run_id = tracker.init_run(config)
        test.assert_not_none(run_id, "Run ID generated")

        tracker.log_params({"param1": "value1"})
        tracker.log_metrics({"metric1": 0.8})
        tracker.log_artifact("/tmp/test.txt", "test")
        tracker.log_table({"a": [1, 2]}, "test_table")
        tracker.log_alert("Test", "Test message", "info")
        tracker.finish_run()

        test.assert_true(not tracker.is_active(), "Tracker finished")
        test.passed += 1  # All operations should complete without errors

    except ImportError as e:
        test.failed += 1
        test.errors.append(f"Import error: {e}")

    test.print_results()
    return test.failed == 0


def main():
    """Run all verification tests."""
    print("Phase 4: Hyperparameter Optimization Simple Verification")
    print("=" * 60)
    print("This script verifies Phase 4 functionality without external dependencies.")

    tests = [
        test_hyperparameter_config,
        test_search_space,
        test_hyperparameter_optimizer,
        test_trial_tracker,
        test_optuna_integration,
        test_experiment_tracking_integration
    ]

    results = []
    for test_func in tests:
        results.append(test_func())
        print()

    # Summary
    passed = sum(results)
    total = len(results)

    print("=" * 60)
    if passed == total:
        print(f"🎉 ALL VERIFICATION TESTS PASSED! ({passed}/{total})")
        print("\nPhase 4 Implementation Status:")
        print("✅ HyperparameterOptimizationConfig - Configuration management")
        print("✅ SearchSpaceBuilder - Search space creation and validation")
        print("✅ HyperparameterOptimizer - Core optimization logic")
        print("✅ TrialTracker - Trial-level experiment tracking")
        print("✅ OptunaStudyManager - Optuna integration framework")
        print("✅ Experiment Tracking Integration - Comprehensive tracking")

        print("\nKey Features Verified:")
        print("• Configuration validation and serialization")
        print("• Search space building with presets")
        print("• Trial lifecycle management")
        print("• Intermediate value tracking")
        print("• Error handling and graceful degradation")
        print("• Study reporting and analysis")
        print("• Integration with existing tracking system")

        print("\nArchitecture Benefits:")
        print("• Modular design with clear separation of concerns")
        print("• Comprehensive error handling")
        print("• Graceful degradation when dependencies unavailable")
        print("• Extensive configuration options")
        print("• Production-ready implementation")

    else:
        print(f"❌ SOME TESTS FAILED! ({passed}/{total} passed)")
        print("Please check the individual test results above.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)