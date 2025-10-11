"""
Experiment tracking utilities for the trading system.

This package provides a unified interface for experiment tracking
across different backends (WandB, MLflow, etc.) with support for:
- Model training experiments
- Hyperparameter optimization
- Strategy backtesting
- Model monitoring
- Performance analysis

Key components:
- ExperimentTrackerInterface: Abstract interface for tracking
- ExperimentConfig: Configuration dataclasses
- WandBExperimentTracker: WandB implementation
- Visualization tools for experiment analysis
"""

from .interface import (
    ExperimentTrackerInterface,
    ExperimentTrackingError,
    NullExperimentTracker
)

from .config import (
    ExperimentConfig,
    OptimizationConfig,
    MonitoringConfig,
    create_training_config,
    create_optimization_config,
    create_backtest_config,
    create_monitoring_config,
    RUN_TYPES,
    ALERT_LEVELS
)

from .visualizer import ExperimentVisualizer
from .wandb_adapter import WandBExperimentTracker, create_wandb_tracker_from_config
from .training_interface import (
    TrainingExperimentTrackerInterface,
    TrainingMetrics,
    ModelLifecycleEvent,
    TrainingMetricsAggregator,
    TrainingExperimentConfig,
    create_training_config
)

__all__ = [
    # Interfaces
    "ExperimentTrackerInterface",
    "ExperimentTrackingError",
    "NullExperimentTracker",
    "TrainingExperimentTrackerInterface",

    # Configuration
    "ExperimentConfig",
    "OptimizationConfig",
    "MonitoringConfig",
    "TrainingExperimentConfig",

    # Training-specific data classes
    "TrainingMetrics",
    "ModelLifecycleEvent",
    "TrainingMetricsAggregator",

    # Factory functions
    "create_training_config",
    "create_optimization_config",
    "create_backtest_config",
    "create_monitoring_config",
    "create_wandb_tracker_from_config",

    # Implementations
    "WandBExperimentTracker",
    "ExperimentVisualizer",

    # Constants
    "RUN_TYPES",
    "ALERT_LEVELS"
]