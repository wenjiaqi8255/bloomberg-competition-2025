"""
Experiment configuration for the trading system.

This module defines configuration dataclasses for experiment tracking,
providing a unified way to configure experiments across different
tracking backends.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Run types for different experiment phases
RUN_TYPES = [
    "training",        # Model training experiments
    "evaluation",      # Model evaluation experiments
    "optimization",    # Hyperparameter optimization experiments
    "backtest",        # Strategy backtesting experiments
    "monitoring",      # Model monitoring experiments
    "analysis",        # Data analysis experiments
    "hyperparameter_trial"  # Individual hyperparameter optimization trials
]

# Alert levels
ALERT_LEVELS = ["info", "warning", "error"]


@dataclass
class ExperimentConfig:
    """
    Configuration for experiment tracking runs.

    This class encapsulates all the information needed to initialize
    and configure an experiment run across different tracking backends.
    """

    # Basic experiment identification
    project_name: str
    experiment_name: str
    run_type: str  # Must be one of RUN_TYPES

    # Experiment organization
    group: Optional[str] = None  # For grouping related experiments
    tags: List[str] = field(default_factory=list)
    entity: Optional[str] = None  # Team/username for multi-user setups

    # Experiment configuration
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    notes: Optional[str] = None

    # Run settings
    run_id: Optional[str] = None  # For resuming existing runs
    resume: str = "allow"  # "allow", "must", "disallow"

    # Data and model information
    data_info: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        """Validate the configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """
        Validate the experiment configuration.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not self.project_name:
            raise ValueError("project_name is required")

        if not self.experiment_name:
            raise ValueError("experiment_name is required")

        if self.run_type not in RUN_TYPES:
            raise ValueError(f"run_type must be one of {RUN_TYPES}, got '{self.run_type}'")

        if self.resume not in ["allow", "must", "disallow"]:
            raise ValueError(f"resume must be one of ['allow', 'must', 'disallow'], got '{self.resume}'")

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

    @classmethod
    def from_json(cls, json_str: str) -> 'ExperimentConfig':
        """Create configuration from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml_file(cls, file_path: Union[str, Path]) -> 'ExperimentConfig':
        """Create configuration from YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to load configuration from YAML files")

        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls.from_dict(config_dict)

    def save_to_yaml(self, file_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML is required to save configuration to YAML files")

        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)

    def copy(self, **kwargs) -> 'ExperimentConfig':
        """Create a copy of the configuration with optional updates."""
        config_dict = self.to_dict()
        config_dict.update(kwargs)
        return self.__class__.from_dict(config_dict)

    def add_tag(self, tag: str) -> None:
        """Add a tag to the configuration."""
        if tag not in self.tags:
            self.tags.append(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the configuration."""
        if tag in self.tags:
            self.tags.remove(tag)

    def add_hyperparameter(self, key: str, value: Any) -> None:
        """Add a hyperparameter to the configuration."""
        self.hyperparameters[key] = value

    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the configuration."""
        self.metadata[key] = value


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization experiments."""

    # Optimization settings
    n_trials: int = 100
    timeout: Optional[int] = None  # Timeout in seconds
    direction: str = "maximize"  # "maximize" or "minimize"

    # Pruning settings
    enable_pruning: bool = True
    pruning_warmup_steps: int = 10
    pruning_interval: int = 1

    # Search space configuration
    search_space: Dict[str, Any] = field(default_factory=dict)

    # Parallel execution
    n_jobs: int = 1

    # Trial configuration
    trial_timeout: Optional[int] = None

    def __post_init__(self):
        """Validate the optimization configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate the optimization configuration."""
        if self.n_trials <= 0:
            raise ValueError("n_trials must be positive")

        if self.direction not in ["maximize", "minimize"]:
            raise ValueError(f"direction must be 'maximize' or 'minimize', got '{self.direction}'")

        if self.n_jobs <= 0:
            raise ValueError("n_jobs must be positive")

        if self.pruning_warmup_steps < 0:
            raise ValueError("pruning_warmup_steps must be non-negative")

        if self.pruning_interval <= 0:
            raise ValueError("pruning_interval must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


@dataclass
class MonitoringConfig:
    """Configuration for model monitoring experiments."""

    # Monitoring settings
    check_interval: int = 3600  # Seconds between checks
    alert_threshold: float = 0.05  # Performance degradation threshold

    # Data windows
    baseline_window: int = 252  # Trading days for baseline
    evaluation_window: int = 21  # Trading days for evaluation

    # Metrics to monitor
    metrics_to_monitor: List[str] = field(default_factory=lambda: ["ic", "sharpe", "max_drawdown"])

    # Alert settings
    alert_cooldown: int = 86400  # Seconds between similar alerts
    alert_recipients: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate the monitoring configuration."""
        self.validate()

    def validate(self) -> None:
        """Validate the monitoring configuration."""
        if self.check_interval <= 0:
            raise ValueError("check_interval must be positive")

        if not 0 <= self.alert_threshold <= 1:
            raise ValueError("alert_threshold must be between 0 and 1")

        if self.baseline_window <= 0:
            raise ValueError("baseline_window must be positive")

        if self.evaluation_window <= 0:
            raise ValueError("evaluation_window must be positive")

        if self.alert_cooldown <= 0:
            raise ValueError("alert_cooldown must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)


# Factory functions for common experiment types

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


def create_optimization_config(project_name: str, model_type: str,
                              search_space: Dict[str, Any],
                              n_trials: int = 100,
                              **kwargs) -> ExperimentConfig:
    """Create a configuration for hyperparameter optimization experiments."""
    return ExperimentConfig(
        project_name=project_name,
        experiment_name=f"{model_type}_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        run_type="optimization",
        hyperparameters={"n_trials": n_trials},
        metadata={"search_space": search_space},
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


def create_monitoring_config(project_name: str, model_id: str,
                            monitoring_config: MonitoringConfig,
                            **kwargs) -> ExperimentConfig:
    """Create a configuration for model monitoring experiments."""
    return ExperimentConfig(
        project_name=project_name,
        experiment_name=f"{model_id}_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        run_type="monitoring",
        hyperparameters=monitoring_config.to_dict(),
        model_info={"model_id": model_id},
        **kwargs
    )