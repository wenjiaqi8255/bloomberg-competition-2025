"""
Demonstration of the new experiment tracking interface.

This script shows how to use the new ExperimentTrackerInterface
for different types of experiments without being tied to a specific
tracking backend.
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from trading_system.utils.experiment_tracking import (
    ExperimentTrackerInterface,
    NullExperimentTracker,
    ExperimentConfig,
    create_training_config,
    create_optimization_config,
    create_backtest_config
)


def demonstrate_interface_flexibility():
    """Demonstrate how the interface allows swapping implementations."""

    def run_experiment(tracker: ExperimentTrackerInterface, config: ExperimentConfig):
        """Run an experiment with any tracker implementation."""
        print(f"Running experiment with {type(tracker).__name__}")

        # Initialize the experiment
        run_id = tracker.init_run(config)
        print(f"  Initialized run: {run_id}")

        # Log some parameters
        tracker.log_params({"learning_rate": 0.01, "epochs": 10})
        print("  Logged parameters")

        # Simulate training with metrics
        for epoch in range(10):
            loss = 1.0 - epoch * 0.1
            accuracy = epoch * 0.1
            tracker.log_metrics({"loss": loss, "accuracy": accuracy}, step=epoch)

        print("  Logged training metrics")

        # Log an artifact (model file)
        tracker.log_artifact("/tmp/model.pkl", "trained_model", "model", "Trained model artifact")
        print("  Logged model artifact")

        # Finish the experiment
        tracker.finish_run()
        print("  Finished experiment")
        print()

    # Run the same experiment with different trackers
    print("=== Demonstrating Interface Flexibility ===\n")

    # With null tracker (no external dependencies)
    null_tracker = NullExperimentTracker()
    config1 = create_training_config("demo_project", "xgboost", {"n_estimators": 100})
    run_experiment(null_tracker, config1)

    # Try with WandB tracker if available (graceful fallback)
    try:
        from trading_system.utils.experiment_tracking.wandb_adapter import WandBExperimentTracker
        wandb_tracker = WandBExperimentTracker(project_name="demo_project")
        config2 = create_training_config("demo_project", "neural_net", {"layers": [64, 32]})
        run_experiment(wandb_tracker, config2)
    except Exception as e:
        print(f"WandB tracker not available: {e}")
        print("  This is expected in environments without WandB configured\n")


def demonstrate_configuration_system():
    """Demonstrate the configuration system."""

    print("=== Demonstrating Configuration System ===\n")

    # Create different types of experiment configurations
    training_config = create_training_config(
        project_name="ml_experiments",
        model_type="xgboost",
        hyperparameters={
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.01
        },
        tags=["xgboost", "baseline"],
        notes="Baseline XGBoost model for stock prediction"
    )

    print("Training Configuration:")
    print(f"  Project: {training_config.project_name}")
    print(f"  Experiment: {training_config.experiment_name}")
    print(f"  Run Type: {training_config.run_type}")
    print(f"  Tags: {training_config.tags}")
    print(f"  Hyperparameters: {training_config.hyperparameters}")
    print()

    # Create optimization configuration
    optimization_config = create_optimization_config(
        project_name="ml_experiments",
        model_type="xgboost",
        search_space={
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.001, 0.01, 0.1]
        },
        n_trials=50,
        tags=["optimization", "xgboost"]
    )

    print("Optimization Configuration:")
    print(f"  Project: {optimization_config.project_name}")
    print(f"  Experiment: {optimization_config.experiment_name}")
    print(f"  Run Type: {optimization_config.run_type}")
    print(f"  Search Space: {optimization_config.metadata['search_space']}")
    print(f"  Number of Trials: {optimization_config.hyperparameters['n_trials']}")
    print()

    # Create backtest configuration
    backtest_config = create_backtest_config(
        project_name="strategy_experiments",
        strategy_name="dual_momentum",
        strategy_config={
            "lookback_period": 12,
            "rebalance_frequency": "monthly",
            "risk_free_rate": 0.02
        },
        tags=["momentum", "dual_momentum"],
        group="strategy_comparison"
    )

    print("Backtest Configuration:")
    print(f"  Project: {backtest_config.project_name}")
    print(f"  Experiment: {backtest_config.experiment_name}")
    print(f"  Run Type: {backtest_config.run_type}")
    print(f"  Strategy Config: {backtest_config.hyperparameters}")
    print(f"  Group: {backtest_config.group}")
    print()


def demonstrate_context_manager():
    """Demonstrate context manager usage."""

    print("=== Demonstrating Context Manager Usage ===\n")

    tracker = NullExperimentTracker()
    config = ExperimentConfig(
        project_name="context_demo",
        experiment_name="context_test",
        run_type="training"
    )

    print("Using context manager for automatic cleanup:")
    try:
        with tracker as t:
            run_id = t.init_run(config)
            print(f"  Started run: {run_id}")

            t.log_params({"batch_size": 32})
            t.log_metrics({"loss": 0.5})

            print("  Logged some data")

            # Simulate an error condition (optional)
            # raise ValueError("Simulated error")

        print("  Context manager automatically finished the run")
    except Exception as e:
        print(f"  Error handled: {e}")

    print()


def demonstrate_child_runs():
    """Demonstrate hierarchical experiment tracking."""

    print("=== Demonstrating Child Runs (Hyperparameter Optimization) ===\n")

    tracker = NullExperimentTracker()

    # Parent optimization run
    parent_config = create_optimization_config(
        project_name="optimization_demo",
        model_type="xgboost",
        search_space={"learning_rate": [0.01, 0.1]},
        n_trials=3
    )

    with tracker as parent_tracker:
        parent_run_id = parent_tracker.init_run(parent_config)
        print(f"Started optimization run: {parent_run_id}")

        # Child trial runs
        for trial in range(3):
            child_tracker = parent_tracker.create_child_run(f"trial_{trial}")
            child_config = ExperimentConfig(
                project_name="optimization_demo",
                experiment_name=f"trial_{trial}",
                run_type="training",
                hyperparameters={"learning_rate": 0.01 * (trial + 1)}
            )

            with child_tracker:
                child_run_id = child_tracker.init_run(child_config)
                print(f"  Started child run: {child_run_id}")

                child_tracker.log_metrics({"accuracy": 0.8 + trial * 0.05})
                print(f"  Trial {trial} accuracy: {0.8 + trial * 0.05}")

        print("  All child runs completed")

    print("  Optimization run completed\n")


def demonstrate_error_handling():
    """Demonstrate graceful error handling."""

    print("=== Demonstrating Graceful Error Handling ===\n")

    tracker = NullExperimentTracker()

    # Operations on inactive tracker should not crash
    print("Operations on inactive tracker:")
    tracker.log_params({"param": "value"})  # Should not crash
    tracker.log_metrics({"metric": 1.0})   # Should not crash
    tracker.log_artifact("/tmp/file.pkl", "artifact")  # Should not crash
    print("  All operations completed without errors")

    # Context manager handles errors gracefully
    print("\nError handling in context manager:")
    try:
        with tracker as t:
            config = ExperimentConfig(
                project_name="error_demo",
                experiment_name="error_test",
                run_type="training"
            )
            t.init_run(config)
            print("  Started run")

            # Simulate an error
            raise ValueError("Simulated processing error")

    except ValueError as e:
        print(f"  Caught expected error: {e}")
        print("  Context manager cleaned up automatically")

    print()


def main():
    """Run all demonstrations."""

    print("🧪 Experiment Tracking Interface Demonstration\n")
    print("This demo shows the new experiment tracking interface that:")
    print("- Decouples experiment tracking from specific backends")
    print("- Enables dependency injection and testing")
    print("- Supports hierarchical experiments (parent/child runs)")
    print("- Provides graceful degradation when tracking is unavailable")
    print("- Maintains backward compatibility with existing WandB usage\n")

    demonstrate_interface_flexibility()
    demonstrate_configuration_system()
    demonstrate_context_manager()
    demonstrate_child_runs()
    demonstrate_error_handling()

    print("✅ Demonstration completed successfully!")
    print("\nKey benefits of the new interface:")
    print("🔄 Flexibility: Swap tracking backends without code changes")
    print("🧪 Testability: Use NullExperimentTracker for unit tests")
    print("📊 Organization: Hierarchical runs for complex experiments")
    print("🛡️ Robustness: Graceful degradation when tracking fails")
    print("🔗 Compatibility: Works with existing WandB setup")


if __name__ == "__main__":
    main()