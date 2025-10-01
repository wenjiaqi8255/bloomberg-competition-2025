"""
Test suite for Phase 2 enhancements to the experiment tracking system.

This module tests:
- Enhanced error handling and graceful degradation
- Visualizer component separation
- WandB adapter refactoring
- StrategyRunner integration with new interface
"""

import sys
import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.experiment_tracking import (
    ExperimentTrackerInterface,
    NullExperimentTracker,
    WandBExperimentTracker,
    ExperimentVisualizer,
    ExperimentConfig,
    create_backtest_config,
    ExperimentTrackingError
)

# Import the refactored StrategyRunner
# Note: StrategyRunner will be tested separately due to import complexities


class TestEnhancedErrorHandling(unittest.TestCase):
    """Test enhanced error handling in WandB adapter."""

    def test_wandb_adapter_offline_mode(self):
        """Test WandB adapter works correctly when WandB is unavailable."""
        # Create tracker that will fail to initialize WandB
        with patch('trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger') as mock_logger:
            mock_logger.side_effect = Exception("WandB not available")

            tracker = WandBExperimentTracker(fail_silently=True)

            # Should not have wandb_logger
            self.assertIsNone(tracker.wandb_logger)
            self.assertFalse(tracker.is_active())

            # Should still be able to log without errors
            config = create_backtest_config(
                project_name="test",
                strategy_name="test_strategy",
                strategy_config={}
            )

            # These should not raise exceptions
            run_id = tracker.init_run(config)
            self.assertIsNotNone(run_id)
            self.assertTrue(run_id.startswith("offline_run_"))

            tracker.log_params({"param": "value"})
            tracker.log_metrics({"accuracy": 0.95})
            tracker.log_artifact("/tmp/test.txt", "test")
            tracker.log_figure(None, "test_fig")
            tracker.log_table({"data": [1, 2, 3]}, "test_table")
            tracker.log_alert("Test Alert", "Test message", "info")
            tracker.finish_run()

    def test_wandb_adapter_strict_mode(self):
        """Test WandB adapter raises exceptions when fail_silently=False."""
        with patch('trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger') as mock_logger:
            mock_logger.side_effect = Exception("WandB not available")

            # Should raise ExperimentTrackingError
            with self.assertRaises(ExperimentTrackingError):
                WandBExperimentTracker(fail_silently=False)

    def test_wandb_adapter_logging_failures(self):
        """Test that logging failures are handled gracefully."""
        # Create a mock WandB logger that fails on specific operations
        mock_wandb_logger = Mock()
        mock_wandb_logger.is_initialized = True
        mock_wandb_logger.log_metrics.side_effect = Exception("Logging failed")
        mock_wandb_logger.log_hyperparameters.side_effect = Exception("Config logging failed")
        mock_wandb_logger.log_artifact.side_effect = Exception("Artifact logging failed")

        with patch('trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger', return_value=mock_wandb_logger):
            tracker = WandBExperimentTracker(fail_silently=True)

            # These should not raise exceptions
            tracker.log_metrics({"accuracy": 0.95})
            tracker.log_params({"param": "value"})
            tracker.log_artifact("/tmp/test.txt", "test")

    def test_child_run_creation_fallback(self):
        """Test child run creation falls back to NullExperimentTracker."""
        mock_wandb_logger = Mock()
        mock_wandb_logger.is_initialized = True

        # Make child run creation fail
        with patch('trading_system.utils.experiment_tracking.wandb_adapter.WandBExperimentTracker') as mock_tracker_class:
            mock_tracker_class.side_effect = Exception("Child creation failed")

            with patch('trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger', return_value=mock_wandb_logger):
                tracker = WandBExperimentTracker(fail_silently=True)

                # Child run creation should return NullExperimentTracker
                child_tracker = tracker.create_child_run("test_child")
                self.assertIsInstance(child_tracker, NullExperimentTracker)


class TestVisualizerSeparation(unittest.TestCase):
    """Test visualizer component separation and independence."""

    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = ExperimentVisualizer()

    def test_visualizer_backend_selection(self):
        """Test visualizer backend selection works correctly."""
        # Auto backend should work
        viz_auto = ExperimentVisualizer(backend="auto")
        self.assertIsNotNone(viz_auto.backend)

        # Plotly backend (if available)
        viz_plotly = ExperimentVisualizer(backend="plotly")
        self.assertTrue(viz_plotly.backend in ["plotly", "matplotlib", None])

        # Matplotlib backend (if available)
        viz_mpl = ExperimentVisualizer(backend="matplotlib")
        self.assertTrue(viz_mpl.backend in ["plotly", "matplotlib", None])

    def test_create_training_curve(self):
        """Test training curve visualization creation."""
        metrics_history = {
            "loss": [1.0, 0.8, 0.6, 0.4, 0.2],
            "accuracy": [0.6, 0.7, 0.8, 0.85, 0.9]
        }

        figure = self.visualizer.create_training_curve(metrics_history)

        if figure is not None:
            # Should have a valid figure object
            self.assertTrue(hasattr(figure, 'write_html') or hasattr(figure, 'savefig'))
        else:
            # Visualizer might be disabled due to missing libraries
            self.assertIsNone(self.visualizer.backend)

    def test_create_feature_importance(self):
        """Test feature importance visualization creation."""
        importance_data = {
            "feature_a": 0.8,
            "feature_b": 0.6,
            "feature_c": 0.4,
            "feature_d": 0.2
        }

        figure = self.visualizer.create_feature_importance(importance_data, top_n=3)

        if figure is not None:
            # Should have a valid figure object
            self.assertTrue(hasattr(figure, 'write_html') or hasattr(figure, 'savefig'))
        else:
            # Visualizer might be disabled
            self.assertIsNone(self.visualizer.backend)

    def test_create_portfolio_performance(self):
        """Test portfolio performance visualization creation."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        portfolio_values = pd.Series(
            data=100000 * (1 + np.random.randn(100).cumsum() * 0.001),
            index=dates
        )
        portfolio_df = portfolio_values.to_frame('portfolio_value')

        figure = self.visualizer.create_portfolio_performance(portfolio_df)

        if figure is not None:
            # Should have a valid figure object
            self.assertTrue(hasattr(figure, 'write_html') or hasattr(figure, 'savefig'))
        else:
            # Visualizer might be disabled
            self.assertIsNone(self.visualizer.backend)

    def test_create_drawdown_chart(self):
        """Test drawdown chart visualization creation."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        portfolio_values = pd.Series(
            data=100000 * (1 + np.random.randn(100).cumsum() * 0.001),
            index=dates
        )
        portfolio_df = portfolio_values.to_frame('portfolio_value')

        figure = self.visualizer.create_drawdown_chart(portfolio_df)

        if figure is not None:
            # Should have a valid figure object
            self.assertTrue(hasattr(figure, 'write_html') or hasattr(figure, 'savefig'))
        else:
            # Visualizer might be disabled
            self.assertIsNone(self.visualizer.backend)

    def test_figure_conversion_and_saving(self):
        """Test figure conversion to bytes and saving."""
        # Create a simple test figure if visualizer is available
        if self.visualizer.backend is not None:
            metrics_history = {"loss": [1.0, 0.8, 0.6]}
            figure = self.visualizer.create_training_curve(metrics_history)

            if figure is not None:
                # Test conversion to bytes
                image_bytes = self.visualizer.figure_to_image_bytes(figure)
                if image_bytes is not None:
                    self.assertIsInstance(image_bytes, bytes)

                # Test saving to temporary file
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    success = self.visualizer.save_figure(figure, tmp_file.name)
                    self.assertTrue(success)


class TestWandBAdapterIntegration(unittest.TestCase):
    """Test WandB adapter integration with visualizer and enhanced features."""

    def test_wandb_adapter_with_visualizer(self):
        """Test WandB adapter uses visualizer correctly."""
        mock_wandb_logger = Mock()
        mock_wandb_logger.is_initialized = True

        with patch('trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger', return_value=mock_wandb_logger):
            # Create tracker with custom visualizer
            custom_visualizer = ExperimentVisualizer()
            tracker = WandBExperimentTracker(visualizer=custom_visualizer)

            self.assertIs(tracker.visualizer, custom_visualizer)
            self.assertIsInstance(tracker.visualizer, ExperimentVisualizer)

    def test_enhanced_logging_methods(self):
        """Test enhanced logging methods in WandB adapter."""
        mock_wandb_logger = Mock()
        mock_wandb_logger.is_initialized = True

        with patch('trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger', return_value=mock_wandb_logger):
            tracker = WandBExperimentTracker()

            # Test log_training_metrics
            metrics_history = {
                "loss": [1.0, 0.8, 0.6],
                "accuracy": [0.6, 0.7, 0.8]
            }
            tracker.log_training_metrics(metrics_history, step=1)

            # Should have logged individual metrics
            mock_wandb_logger.log_metrics.assert_called()

            # Test log_feature_importance
            importance_data = {"feature_a": 0.8, "feature_b": 0.6}
            tracker.log_feature_importance(importance_data, top_n=5)

            # Should have called log_table and log_figure
            self.assertTrue(mock_wandb_logger.log_artifact.called)

    def test_portfolio_performance_logging(self):
        """Test portfolio performance logging with visualization."""
        mock_wandb_logger = Mock()
        mock_wandb_logger.is_initialized = True

        with patch('trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger', return_value=mock_wandb_logger):
            with patch.object(ExperimentVisualizer, 'create_portfolio_performance') as mock_viz:
                with patch.object(ExperimentVisualizer, 'create_drawdown_chart') as mock_drawdown:
                    mock_viz.return_value = Mock()
                    mock_drawdown.return_value = Mock()

                    tracker = WandBExperimentTracker()

                    # Create test portfolio data
                    dates = pd.date_range('2023-01-01', periods=10, freq='D')
                    portfolio_df = pd.DataFrame(
                        {"portfolio_value": [100000, 101000, 102000, 101500, 103000,
                                           104000, 103500, 105000, 106000, 107000]},
                        index=dates
                    )
                    benchmark_df = pd.DataFrame(
                        {"benchmark": [100000, 100500, 101000, 100800, 101500,
                                      102000, 101800, 102500, 103000, 103500]},
                        index=dates
                    )

                    tracker.log_portfolio_performance(portfolio_df, benchmark_df)

                    # Should have called WandB logger methods
                    mock_wandb_logger.log_portfolio_performance.assert_called_once()

                    # Should have created visualizations
                    mock_viz.assert_called_once()
                    mock_drawdown.assert_called_once()


class TestStrategyRunnerIntegration(unittest.TestCase):
    """Test StrategyRunner integration with new experiment tracking interface."""

    def test_strategy_runner_with_null_tracker(self):
        """Test StrategyRunner works with NullExperimentTracker."""
        null_tracker = NullExperimentTracker()
        runner = create_strategy_runner(
            config_path=None,
            experiment_tracker=null_tracker,
            use_wandb=False
        )

        self.assertIs(runner.experiment_tracker, null_tracker)
        self.assertIsInstance(runner.experiment_tracker, NullExperimentTracker)

    def test_strategy_runner_with_wandb_tracker(self):
        """Test StrategyRunner works with WandBExperimentTracker."""
        mock_wandb_logger = Mock()
        mock_wandb_logger.is_initialized = True

        with patch('trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger', return_value=mock_wandb_logger):
            wandb_tracker = WandBExperimentTracker(fail_silently=True)
            runner = create_strategy_runner(
                config_path=None,
                experiment_tracker=wandb_tracker
            )

            self.assertIs(runner.experiment_tracker, wandb_tracker)
            self.assertEqual(runner.wandb_logger, mock_wandb_logger)

    def test_strategy_runner_auto_wandb_fallback(self):
        """Test StrategyRunner falls back to null tracker when WandB fails."""
        with patch('trading_system.strategy_runner.WandBExperimentTracker') as mock_wandb_class:
            mock_wandb_class.side_effect = Exception("WandB unavailable")

            # Should create runner with null tracker
            runner = create_strategy_runner(use_wandb=True)
            self.assertIsInstance(runner.experiment_tracker, NullExperimentTracker)

    def test_strategy_runner_experiments_use_new_interface(self):
        """Test that strategy runner experiments use the new interface."""
        null_tracker = NullExperimentTracker()
        runner = create_strategy_runner(experiment_tracker=null_tracker)

        # Mock the initialization process
        with patch.object(runner, 'initialize') as mock_init:
            with patch.object(runner, '_fetch_data') as mock_fetch:
                with patch.object(runner, '_save_results'):
                    mock_init.return_value = None
                    mock_fetch.return_value = (
                        {"AAPL": pd.DataFrame({"Close": [100, 101, 102]})},
                        pd.DataFrame({"Close": [100, 100.5, 101]})
                    )

                    # Mock strategy and backtest engine
                    mock_strategy = Mock()
                    mock_strategy.get_name.return_value = "test_strategy"
                    mock_strategy.generate_signals.return_value = pd.DataFrame(
                        {"AAPL": [0.1, 0.1, 0.1]},
                        index=pd.date_range('2023-01-01', periods=3)
                    )
                    mock_strategy.calculate_risk_metrics.return_value = {"sharpe_ratio": 1.5}
                    runner.strategy = mock_strategy

                    mock_backtest = Mock()
                    mock_backtest.run_backtest.return_value = Mock(
                        portfolio_values=pd.Series([100, 101, 102]),
                        trades=[],
                        performance_metrics={"total_return": 0.02}
                    )
                    runner.backtest_engine = mock_backtest

                    runner.configs = {
                        'strategy': Mock(
                            strategy_type=Mock(value="test"),
                            universe=["AAPL"],
                            parameters={}
                        ),
                        'backtest': Mock(
                            start_date='2023-01-01',
                            end_date='2023-01-03',
                            benchmark_symbol='SPY'
                        )
                    }

                    # Run strategy
                    results = runner.run_strategy("test_experiment")

                    # Should have used the new tracker interface
                    self.assertIsInstance(null_tracker.run_ids, list)
                    self.assertGreater(len(null_tracker.run_ids), 0)

    def test_factory_function_variants(self):
        """Test create_strategy_runner factory function variants."""
        # With custom tracker
        custom_tracker = NullExperimentTracker()
        runner1 = create_strategy_runner(experiment_tracker=custom_tracker)
        self.assertIs(runner1.experiment_tracker, custom_tracker)

        # Without WandB
        runner2 = create_strategy_runner(use_wandb=False)
        self.assertIsInstance(runner2.experiment_tracker, NullExperimentTracker)

        # With WandB (may fall back to null tracker)
        runner3 = create_strategy_runner(use_wandb=True)
        self.assertIsInstance(runner3.experiment_tracker, ExperimentTrackerInterface)


class TestBackwardCompatibility(unittest.TestCase):
    """Test backward compatibility with existing WandBLogger usage."""

    def test_wandb_adapter_has_wandb_logger(self):
        """Test WandBExperimentTracker exposes wandb_logger for compatibility."""
        mock_wandb_logger = Mock()
        mock_wandb_logger.is_initialized = True

        with patch('trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger', return_value=mock_wandb_logger):
            tracker = WandBExperimentTracker()

            # Should have wandb_logger attribute for backward compatibility
            self.assertEqual(tracker.wandb_logger, mock_wandb_logger)

    def test_strategy_runner_maintains_wandb_logger(self):
        """Test StrategyRunner maintains wandb_logger for backward compatibility."""
        mock_wandb_logger = Mock()
        mock_wandb_logger.is_initialized = True

        with patch('trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger', return_value=mock_wandb_logger):
            runner = create_strategy_runner(use_wandb=True)

            # Should have both experiment_tracker and wandb_logger
            self.assertIsInstance(runner.experiment_tracker, WandBExperimentTracker)
            self.assertEqual(runner.wandb_logger, mock_wandb_logger)

    def test_enhanced_methods_maintain_original_signatures(self):
        """Test that enhanced methods maintain original signatures."""
        mock_wandb_logger = Mock()
        mock_wandb_logger.is_initialized = True

        with patch('trading_system.utils.experiment_tracking.wandb_adapter.WandBLogger', return_value=mock_wandb_logger):
            tracker = WandBExperimentTracker()

            # Should have the enhanced convenience methods
            self.assertTrue(hasattr(tracker, 'log_portfolio_performance'))
            self.assertTrue(hasattr(tracker, 'log_trades'))
            self.assertTrue(hasattr(tracker, 'log_dataset_info'))
            self.assertTrue(hasattr(tracker, 'log_training_metrics'))
            self.assertTrue(hasattr(tracker, 'log_feature_importance'))


if __name__ == '__main__':
    # Import numpy for test data generation
    try:
        import numpy as np
    except ImportError:
        # Create simple fallback for numpy
        class MockNumpy:
            def random(self):
                return self
            def randn(self, n):
                return [0.1] * n
            def cumsum(self, arr):
                return arr
            def __call__(self, n):
                return [0.1] * n
        np = MockNumpy()

    unittest.main(verbosity=2)