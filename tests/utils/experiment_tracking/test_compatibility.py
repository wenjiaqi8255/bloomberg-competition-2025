"""
Tests for backward compatibility with existing WandB usage.

This module ensures that the new interface can work with existing code
without breaking changes, enabling gradual migration.
"""

import pytest
from unittest.mock import MagicMock, patch

from src.trading_system.utils.experiment_tracking import (
    ExperimentTrackerInterface,
    NullExperimentTracker,
    ExperimentConfig,
    create_training_config
)
from src.trading_system.utils.experiment_tracking.wandb_adapter import (
    WandBExperimentTracker,
    create_wandb_tracker_from_config
)


class TestBackwardCompatibility:
    """Test cases for backward compatibility with existing code."""

    def test_null_tracker_compatibility(self):
        """Test that NullExperimentTracker can replace WandBLogger in existing code."""
        tracker = NullExperimentTracker()

        # Simulate existing WandBLogger usage patterns
        tracker.init_run(ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training"
        ))

        # These should not raise errors (graceful degradation)
        tracker.log_hyperparameters = lambda params: tracker.log_params(params)
        tracker.log_hyperparameters({"learning_rate": 0.01})

        tracker.log_config = lambda config: tracker.log_params(config)
        tracker.log_config({"model_type": "xgboost"})

        tracker.log_metrics({"accuracy": 0.95, "loss": 0.1})

        # Should work without exceptions
        assert not tracker.is_active()

    @patch('src.trading_system.utils.wandb_logger.WandBLogger')
    def test_wandb_adapter_compatibility(self, mock_wandb_logger):
        """Test that WandBExperimentTracker maintains WandBLogger compatibility."""
        # Mock WandBLogger
        mock_logger_instance = MagicMock()
        mock_logger_instance.is_initialized = True
        mock_logger_instance.initialize_experiment.return_value = True
        mock_logger_instance.run = MagicMock()
        mock_wandb_logger.return_value = mock_logger_instance

        tracker = WandBExperimentTracker(project_name="test_project")

        # Test new interface methods
        config = create_training_config(
            project_name="test_project",
            model_type="xgboost",
            hyperparameters={"learning_rate": 0.01}
        )

        run_id = tracker.init_run(config)
        assert run_id is not None

        tracker.log_params({"n_estimators": 100})
        tracker.log_metrics({"accuracy": 0.95})
        tracker.log_artifact("/path/to/model.pkl", "model")

        # Test backward compatibility methods
        tracker.log_portfolio_performance(MagicMock())
        tracker.log_trades(MagicMock())
        tracker.log_dataset_info({"samples": 1000})

        # Should have called corresponding WandBLogger methods
        mock_logger_instance.initialize_experiment.assert_called_once()
        mock_logger_instance.log_hyperparameters.assert_called()
        mock_logger_instance.log_metrics.assert_called()

    @patch('src.trading_system.utils.wandb_logger.WandBLogger')
    def test_existing_strategy_runner_pattern(self, mock_wandb_logger):
        """Test that existing StrategyRunner usage patterns still work."""
        # Mock WandBLogger
        mock_logger_instance = MagicMock()
        mock_logger_instance.is_initialized = True
        mock_logger_instance.initialize_experiment.return_value = True
        mock_wandb_logger.return_value = mock_logger_instance

        # Simulate existing StrategyRunner code pattern
        tracker = WandBExperimentTracker(
            project_name='bloomberg-competition',
            tags=[],
            group=None
        )

        # Initialize experiment (existing pattern)
        tracker.init_run(ExperimentConfig(
            project_name='bloomberg-competition',
            experiment_name='test_strategy_run',
            run_type='backtest',
            tags=[]
        ))

        # Log config (existing pattern)
        config_summary = {"strategy": {"name": "dual_momentum"}}
        tracker.log_params(config_summary)

        # Log dataset info (existing pattern)
        data_stats = {"start_date": "2020-01-01", "end_date": "2023-12-31", "symbols": ["AAPL", "MSFT"]}
        tracker.log_dataset_info(data_stats)

        # Log portfolio performance (existing pattern)
        tracker.log_portfolio_performance(MagicMock(), MagicMock())

        # Log trades (existing pattern)
        tracker.log_trades(MagicMock())

        # Log risk metrics (existing pattern)
        risk_metrics = {"sharpe_ratio": 1.5, "max_drawdown": -0.15}
        tracker.log_metrics(risk_metrics)

        # Finish experiment (existing pattern)
        tracker.finish_run()

        # Should have completed without errors
        mock_logger_instance.finish_experiment.assert_called_once()

    def test_dependency_injection_pattern(self):
        """Test that trackers can be dependency injected."""
        def process_experiment(tracker: ExperimentTrackerInterface):
            """Function that processes an experiment with any tracker."""
            config = ExperimentConfig(
                project_name="test_project",
                experiment_name="injected_experiment",
                run_type="training"
            )

            run_id = tracker.init_run(config)
            tracker.log_params({"learning_rate": 0.01})
            tracker.log_metrics({"accuracy": 0.95})
            tracker.finish_run()

            return run_id

        # Should work with NullExperimentTracker
        null_tracker = NullExperimentTracker()
        null_run_id = process_experiment(null_tracker)
        assert null_run_id is not None

        # Should work with WandBExperimentTracker (mocked)
        with patch('src.trading_system.utils.wandb_logger.WandBLogger') as mock_wandb:
            mock_logger_instance = MagicMock()
            mock_logger_instance.is_initialized = True
            mock_logger_instance.initialize_experiment.return_value = True
            mock_wandb.return_value = mock_logger_instance

            wandb_tracker = WandBExperimentTracker()
            wandb_run_id = process_experiment(wandb_tracker)
            assert wandb_run_id is not None

    def test_interface_swapping(self):
        """Test that tracker implementations can be swapped without code changes."""
        class ExperimentProcessor:
            def __init__(self, tracker: ExperimentTrackerInterface):
                self.tracker = tracker

            def run_experiment(self, config: ExperimentConfig):
                """Run experiment with injected tracker."""
                run_id = self.tracker.init_run(config)

                # Simulate some experiment work
                self.tracker.log_params({"epochs": 10})
                for epoch in range(10):
                    self.tracker.log_metrics({"loss": 1.0 - epoch * 0.1}, step=epoch)

                self.tracker.log_artifact("/tmp/model.pkl", "model")
                self.tracker.finish_run()

                return run_id

        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="swap_test",
            run_type="training"
        )

        # Should work with NullExperimentTracker
        processor1 = ExperimentProcessor(NullExperimentTracker())
        run_id1 = processor1.run_experiment(config)
        assert run_id1 is not None

        # Should work with WandBExperimentTracker (mocked)
        with patch('src.trading_system.utils.wandb_logger.WandBLogger') as mock_wandb:
            mock_logger_instance = MagicMock()
            mock_logger_instance.is_initialized = True
            mock_logger_instance.initialize_experiment.return_value = True
            mock_wandb.return_value = mock_logger_instance

            processor2 = ExperimentProcessor(WandBExperimentTracker())
            run_id2 = processor2.run_experiment(config)
            assert run_id2 is not None

    @patch('src.trading_system.utils.wandb_logger.WandBLogger')
    def test_config_based_factory(self, mock_wandb_logger):
        """Test creating trackers from configuration."""
        mock_logger_instance = MagicMock()
        mock_logger_instance.is_initialized = True
        mock_logger_instance.initialize_experiment.return_value = True
        mock_wandb_logger.return_value = mock_logger_instance

        config = ExperimentConfig(
            project_name="test_project",
            experiment_name="test_experiment",
            run_type="training",
            entity="test_entity",
            tags=["tag1", "tag2"],
            group="test_group",
            hyperparameters={"learning_rate": 0.01}
        )

        tracker = create_wandb_tracker_from_config(config)
        assert isinstance(tracker, WandBExperimentTracker)
        assert tracker.wandb_logger.project_name == "test_project"
        assert tracker.wandb_logger.entity == "test_entity"

    def test_error_handling_compatibility(self):
        """Test that error handling is compatible with existing patterns."""
        tracker = NullExperimentTracker()

        # Existing code often checks if tracking is available
        if tracker.is_active():
            tracker.log_metrics({"some_metric": 1.0})

        # Should not raise errors even when inactive

    def test_context_manager_compatibility(self):
        """Test context manager usage patterns."""
        # Existing code might use context managers
        with NullExperimentTracker() as tracker:
            config = ExperimentConfig(
                project_name="test_project",
                experiment_name="context_test",
                run_type="training"
            )

            run_id = tracker.init_run(config)
            tracker.log_params({"param1": "value1"})
            tracker.log_metrics({"metric1": 1.0})

        # Should exit cleanly

    @patch('src.trading_system.utils.wandb_logger.WandBLogger')
    def test_child_run_compatibility(self, mock_wandb_logger):
        """Test child run creation for hyperparameter optimization."""
        mock_logger_instance = MagicMock()
        mock_logger_instance.is_initialized = True
        mock_logger_instance.initialize_experiment.return_value = True
        mock_wandb_logger.return_value = mock_logger_instance

        parent_tracker = WandBExperimentTracker()
        parent_config = ExperimentConfig(
            project_name="test_project",
            experiment_name="optimization_experiment",
            run_type="optimization"
        )

        parent_tracker.init_run(parent_config)

        # Create child runs (pattern used in hyperparameter optimization)
        for trial in range(3):
            child_tracker = parent_tracker.create_child_run(f"trial_{trial}")
            child_config = ExperimentConfig(
                project_name="test_project",
                experiment_name=f"trial_{trial}",
                run_type="training"
            )

            child_tracker.init_run(child_config)
            child_tracker.log_params({"learning_rate": 0.01 * (trial + 1)})
            child_tracker.log_metrics({"accuracy": 0.8 + trial * 0.05})
            child_tracker.finish_run()

        parent_tracker.finish_run()

        # Should have created parent and child runs
        assert mock_wandb_logger.call_count >= 4  # 1 parent + 3 children


class TestMigrationPath:
    """Test cases showing migration path from old to new interface."""

    def test_step_1_add_interface_without_changes(self):
        """Step 1: Add new interface without changing existing code."""
        # Existing code continues to work
        from src.trading_system.utils.wandb_logger import WandBLogger

        # New interface is available but doesn't break existing code
        from src.trading_system.utils.experiment_tracking import ExperimentTrackerInterface

        assert WandBLogger is not None
        assert ExperimentTrackerInterface is not None

    def test_step_2_introduce_adapter(self):
        """Step 2: Use adapter to bridge old and new interfaces."""
        with patch('src.trading_system.utils.wandb_logger.WandBLogger') as mock_wandb:
            mock_logger_instance = MagicMock()
            mock_logger_instance.is_initialized = True
            mock_logger_instance.initialize_experiment.return_value = True
            mock_wandb.return_value = mock_logger_instance

            # Can create new interface tracker that wraps old implementation
            new_tracker = WandBExperimentTracker()
            assert isinstance(new_tracker, ExperimentTrackerInterface)

            # Can still use old WandBLogger methods if needed
            new_tracker.log_portfolio_performance(MagicMock())

    def test_step_3_dependency_injection(self):
        """Step 3: Start using dependency injection."""
        def run_strategy_with_tracking(tracker: ExperimentTrackerInterface):
            """Strategy function that accepts any tracker."""
            config = ExperimentConfig(
                project_name="strategies",
                experiment_name="dual_momentum_test",
                run_type="backtest"
            )

            tracker.init_run(config)
            tracker.log_params({"strategy": "dual_momentum"})
            tracker.log_metrics({"sharpe": 1.5})
            tracker.finish_run()

        # Can pass in any tracker implementation
        run_strategy_with_tracking(NullExperimentTracker())

        with patch('src.trading_system.utils.wandb_logger.WandBLogger') as mock_wandb:
            mock_logger_instance = MagicMock()
            mock_logger_instance.is_initialized = True
            mock_logger_instance.initialize_experiment.return_value = True
            mock_wandb.return_value = mock_logger_instance

            run_strategy_with_tracking(WandBExperimentTracker())

    def test_step_4_full_migration(self):
        """Step 4: Fully migrate to new interface."""
        # New code uses only the new interface
        def create_experiment_runner(tracker_factory):
            """Factory function that creates experiment runners."""
            tracker = tracker_factory()

            def run_experiment(config):
                tracker.init_run(config)
                # ... experiment logic ...
                tracker.finish_run()

            return run_experiment

        # Can create different factories for different environments
        null_factory = lambda: NullExperimentTracker()

        with patch('src.trading_system.utils.wandb_logger.WandBLogger') as mock_wandb:
            mock_logger_instance = MagicMock()
            mock_logger_instance.is_initialized = True
            mock_logger_instance.initialize_experiment.return_value = True
            mock_wandb.return_value = mock_logger_instance

            wandb_factory = lambda: WandBExperimentTracker()

        # Same code works with both factories
        runner1 = create_experiment_runner(null_factory)
        runner2 = create_experiment_runner(wandb_factory)

        config = ExperimentConfig(
            project_name="test",
            experiment_name="test",
            run_type="training"
        )

        runner1(config)  # Works with null tracker
        runner2(config)  # Works with WandB tracker