"""
Simplified experiment tracking utilities for the trading system.

This package provides a minimal interface for experiment tracking
with WandB backend, following the KISS principle.

Key components:
- ExperimentTrackerInterface: Simple interface for tracking
- ExperimentConfig: Basic configuration class
- WandBExperimentTracker: WandB implementation
- NullExperimentTracker: No-op implementation for testing
"""

from .interface import (
    ExperimentTrackerInterface,
    ExperimentTrackingError,
    NullExperimentTracker
)

from .config import (
    ExperimentConfig,
    create_training_config,
    create_backtest_config
)

from .wandb_adapter import (
    WandBExperimentTracker, 
    create_wandb_tracker_from_config
)

__all__ = [
    # Interfaces
    "ExperimentTrackerInterface",
    "ExperimentTrackingError",
    "NullExperimentTracker",

    # Configuration
    "ExperimentConfig",

    # Factory functions
    "create_training_config",
    "create_backtest_config",
    "create_wandb_tracker_from_config",

    # Implementations
    "WandBExperimentTracker",
]