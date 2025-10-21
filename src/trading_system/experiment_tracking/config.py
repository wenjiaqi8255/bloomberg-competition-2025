"""
Simplified experiment configuration for the trading system.

This module defines a minimal configuration class for experiment tracking,
following the KISS principle. Only includes fields that are actually used.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """
    Simplified configuration for experiment tracking runs.
    
    Only includes fields that are actually used in the codebase.
    """

    # Basic experiment identification
    project_name: str
    experiment_name: str
    run_type: str = "training"  # Default to training

    # Experiment organization
    group: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    entity: Optional[str] = None

    # Experiment configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    # Run settings
    run_id: Optional[str] = None

    # Data and model information
    data_info: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Validate the configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate the experiment configuration."""
        if not self.project_name:
            raise ValueError("project_name is required")

        if not self.experiment_name:
            raise ValueError("experiment_name is required")

        # Validate that tags are strings
        if not all(isinstance(tag, str) for tag in self.tags):
            raise ValueError("All tags must be strings")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the configuration."""
        if tag not in self.tags:
            self.tags.append(tag)

    def add_hyperparameter(self, key: str, value: Any) -> None:
        """Add a hyperparameter to the configuration."""
        self.hyperparameters[key] = value

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the configuration."""
        self.metadata[key] = value


# Simple factory functions for common experiment types

def create_training_config(project_name: str, model_type: str,
                          hyperparameters: Dict[str, Any],
                          **kwargs) -> ExperimentConfig:
    """Create a configuration for model training experiments."""
    return ExperimentConfig(
        project_name=project_name,
        experiment_name=f"{model_type}_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        run_type="training",
        hyperparameters=hyperparameters,
        model_info={"model_type": model_type},
        **kwargs
    )


def create_backtest_config(project_name: str, strategy_name: str,
                          strategy_config: Dict[str, Any],
                          **kwargs) -> ExperimentConfig:
    """Create a configuration for strategy backtesting experiments."""
    return ExperimentConfig(
        project_name=project_name,
        experiment_name=f"{strategy_name}_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        run_type="backtest",
        hyperparameters=strategy_config,
        model_info={"strategy_name": strategy_name},
        **kwargs
    )