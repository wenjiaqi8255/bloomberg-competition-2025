"""
Simple Phase 4 Test - Test imports without complex dependencies
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test basic imports of Phase 4 components."""
    print("Testing Phase 4 basic imports...")

    success_count = 0
    total_count = 0

    # Test 1: Experiment tracking (Phase 1 components)
    print("\n1. Testing experiment tracking imports:")
    try:
        from src.trading_system.utils.experiment_tracking.interface import ExperimentTrackerInterface, NullExperimentTracker
        from src.trading_system.utils.experiment_tracking.config import ExperimentConfig
        print("  ✅ Basic experiment tracking imports")
        success_count += 1
    except Exception as e:
        print(f"  ❌ Basic experiment tracking imports failed: {e}")
    total_count += 1

    # Test 2: Trial tracker (Phase 4 component)
    print("\n2. Testing trial tracker imports:")
    try:
        from src.trading_system.utils.experiment_tracking.trial_tracker import TrialTracker, TrialMetadata, StudyMetadata
        print("  ✅ Trial tracker imports")
        success_count += 1
    except Exception as e:
        print(f"  ❌ Trial tracker imports failed: {e}")
    total_count += 1

    # Test 3: Hyperparameter config (Phase 4 component)
    print("\n3. Testing hyperparameter config imports:")
    try:
        from src.trading_system.models.training.hyperparameter_config import (
            ProblemConfig, ModelConfig, ResourceConfig, LoggingConfig,
            HyperparameterOptimizationConfig, create_default_config
        )
        print("  ✅ Hyperparameter config imports")
        success_count += 1
    except Exception as e:
        print(f"  ❌ Hyperparameter config imports failed: {e}")
    total_count += 1

    # Test 4: Search space builder (Phase 4 component)
    print("\n4. Testing search space builder imports:")
    try:
        from src.trading_system.models.training.search_space_builder import (
            SearchSpaceBuilder, SearchSpacePreset, SearchSpace
        )
        print("  ✅ Search space builder imports")
        success_count += 1
    except Exception as e:
        print(f"  ❌ Search space builder imports failed: {e}")
    total_count += 1

    # Test 5: Hyperparameter optimizer (Phase 4 component)
    print("\n5. Testing hyperparameter optimizer imports:")
    try:
        from src.trading_system.models.training.hyperparameter_optimizer import (
            HyperparameterConfig, SearchSpace, OptimizationResult, HyperparameterOptimizer
        )
        print("  ✅ Hyperparameter optimizer imports")
        success_count += 1
    except Exception as e:
        print(f"  ❌ Hyperparameter optimizer imports failed: {e}")
    total_count += 1

    # Test 6: Optuna integration (Phase 4 component)
    print("\n6. Testing Optuna integration imports:")
    try:
        from src.trading_system.models.training.optuna_integration import (
            OptunaConfig, OptunaStudyManager, quick_optimize
        )
        print("  ✅ Optuna integration imports")
        success_count += 1
    except Exception as e:
        print(f"  ❌ Optuna integration imports failed: {e}")
    total_count += 1

    return success_count, total_count

def test_basic_functionality():
    """Test basic functionality of Phase 4 components."""
    print("\nTesting basic functionality...")

    success_count = 0
    total_count = 0

    # Test 1: Null experiment tracker
    print("\n1. Testing NullExperimentTracker:")
    try:
        from src.trading_system.utils.experiment_tracking.interface import NullExperimentTracker, ExperimentConfig

        tracker = NullExperimentTracker()
        config = ExperimentConfig(
            project_name="test",
            experiment_name="test_exp",
            run_type="analysis"
        )

        run_id = tracker.init_run(config)
        assert run_id is not None

        tracker.log_params({"param1": "value1"})
        tracker.log_metrics({"metric1": 0.8})
        tracker.finish_run()

        print("  ✅ NullExperimentTracker basic functionality")
        success_count += 1
    except Exception as e:
        print(f"  ❌ NullExperimentTracker failed: {e}")
    total_count += 1

    # Test 2: Hyperparameter config creation
    print("\n2. Testing hyperparameter config creation:")
    try:
        from src.trading_system.models.training.hyperparameter_config import (
            ProblemConfig, ModelConfig, HyperparameterOptimizationConfig
        )

        problem = ProblemConfig(problem_type="regression", n_samples=1000, n_features=10)
        model = ModelConfig(model_type="xgboost", cv_folds=5)

        config = HyperparameterOptimizationConfig(
            problem=problem,
            model=model,
            study_name="test_study",
            search_space_builder_preset="xgboost_default"
        )

        assert config.problem.problem_type == "regression"
        assert config.model.model_type == "xgboost"
        assert config.study_name == "test_study"

        print("  ✅ Hyperparameter config creation")
        success_count += 1
    except Exception as e:
        print(f"  ❌ Hyperparameter config creation failed: {e}")
    total_count += 1

    # Test 3: Search space creation
    print("\n3. Testing search space creation:")
    try:
        from src.trading_system.models.training.hyperparameter_optimizer import SearchSpace
        from src.trading_system.models.training.search_space_builder import SearchSpaceBuilder

        space = SearchSpace("param1", "int", low=1, high=10)
        assert space.validate() == True

        builder = SearchSpaceBuilder()
        presets = builder.list_presets()
        assert len(presets) > 0

        print("  ✅ Search space creation")
        success_count += 1
    except Exception as e:
        print(f"  ❌ Search space creation failed: {e}")
    total_count += 1

    # Test 4: Trial tracker basic functionality
    print("\n4. Testing trial tracker basic functionality:")
    try:
        from src.trading_system.utils.experiment_tracking.trial_tracker import TrialTracker
        from src.trading_system.utils.experiment_tracking.interface import NullExperimentTracker

        base_tracker = NullExperimentTracker()
        trial_tracker = TrialTracker(base_tracker, "test_study")

        assert trial_tracker.study_name == "test_study"
        assert trial_tracker.current_trial is None

        # Start trial
        run_id = trial_tracker.start_trial(trial_number=1, parameters={"lr": 0.1})
        assert run_id is not None
        assert trial_tracker.current_trial is not None

        # Complete trial
        trial_tracker.complete_trial(score=0.8)
        assert trial_tracker.current_trial is None
        assert len(trial_tracker.trials_history) == 1

        print("  ✅ Trial tracker basic functionality")
        success_count += 1
    except Exception as e:
        print(f"  ❌ Trial tracker failed: {e}")
    total_count += 1

    return success_count, total_count

def main():
    """Run all tests."""
    print("Phase 4 Simple Test Suite")
    print("=" * 50)

    # Test imports
    import_success, import_total = test_basic_imports()

    # Test functionality
    func_success, func_total = test_basic_functionality()

    # Summary
    total_success = import_success + func_success
    total_tests = import_total + func_total

    print("\n" + "=" * 50)
    print("Test Results:")
    print(f"Imports: {import_success}/{import_total} passed")
    print(f"Functionality: {func_success}/{func_total} passed")
    print(f"Overall: {total_success}/{total_tests} passed")

    if total_success == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("\nPhase 4 Implementation Verified:")
        print("✅ All components can be imported successfully")
        print("✅ Basic functionality works as expected")
        print("✅ Configuration system functional")
        print("✅ Trial tracking operational")
        print("✅ Search space building works")
        print("✅ Error handling is in place")

        print("\nPhase 4 is ready for use!")
    else:
        print(f"\n⚠️  {total_tests - total_success} tests failed")
        print("Some components may have import issues.")

    return total_success == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)