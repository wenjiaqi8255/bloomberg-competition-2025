"""
Hyperparameter Optimization Configuration

This module provides comprehensive configuration classes for hyperparameter
optimization, including validation, serialization, and smart defaults.

Key Features:
- Comprehensive optimization configuration
- Configuration validation and constraints
- Environment-specific configurations
- Configuration serialization/deserialization
- Smart defaults based on problem characteristics
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import logging
import json
import time
from pathlib import Path
import os

from .hyperparameter_optimizer import HyperparameterConfig, SearchSpace
from .optuna_integration import OptunaConfig

logger = logging.getLogger(__name__)


@dataclass
class ProblemConfig:
    """Configuration for the optimization problem."""
    problem_type: str = "regression"  # "regression", "classification", "ranking"
    target_metric: str = "val_score"
    metric_direction: str = "maximize"  # "maximize", "minimize"

    # Data characteristics
    n_samples: int = 1000
    n_features: int = 20
    n_classes: Optional[int] = None  # For classification

    # Data constraints
    feature_types: Dict[str, str] = field(default_factory=dict)  # "numeric", "categorical", "text"
    missing_values: bool = False
    class_imbalance: Optional[float] = None  # Imbalance ratio for classification

    # Problem constraints
    memory_limit: Optional[str] = None  # e.g., "4GB"
    time_limit: Optional[int] = None  # Seconds per evaluation
    interpretability_required: bool = False
    online_learning: bool = False

    # Domain-specific settings
    financial_time_series: bool = False
    lookback_window: Optional[int] = None
    prediction_horizon: Optional[int] = None
    transaction_costs: bool = False

    def validate(self) -> bool:
        """Validate problem configuration."""
        errors = []

        if self.problem_type not in ["regression", "classification", "ranking"]:
            errors.append(f"Invalid problem_type: {self.problem_type}")

        if self.metric_direction not in ["maximize", "minimize"]:
            errors.append(f"Invalid metric_direction: {self.metric_direction}")

        if self.n_samples <= 0:
            errors.append(f"n_samples must be positive: {self.n_samples}")

        if self.n_features <= 0:
            errors.append(f"n_features must be positive: {self.n_features}")

        if self.problem_type == "classification" and (not self.n_classes or self.n_classes < 2):
            errors.append("n_classes must be specified and >= 2 for classification")

        if errors:
            logger.error(f"ProblemConfig validation failed: {errors}")
            return False

        return True


@dataclass
class ModelConfig:
    """Configuration for the model being optimized."""
    model_type: str = "xgboost"  # "xgboost", "lightgbm", "random_forest", "linear", "neural_network", "svm"
    model_family: str = "tree_based"  # "tree_based", "linear", "kernel", "neural"

    # Model-specific settings
    ensemble: bool = False
    ensemble_size: int = 5
    stacking: bool = False

    # Training settings
    cv_folds: int = 5
    validation_split: float = 0.2
    early_stopping: bool = True
    early_stopping_patience: int = 10

    # Feature settings
    feature_selection: bool = False
    max_features: Optional[int] = None
    feature_importance_threshold: float = 0.01

    # Regularization preferences
    prefer_regularization: bool = True
    max_regularization: float = 10.0
    min_regularization: float = 0.001

    # Complexity preferences
    prefer_simpler_models: bool = False
    max_complexity: Optional[Dict[str, Any]] = None

    def validate(self) -> bool:
        """Validate model configuration."""
        errors = []

        valid_model_types = [
            "xgboost", "lightgbm", "random_forest", "linear",
            "neural_network", "svm", "logistic_regression", "elastic_net"
        ]
        if self.model_type not in valid_model_types:
            errors.append(f"Invalid model_type: {self.model_type}")

        valid_families = ["tree_based", "linear", "kernel", "neural", "ensemble"]
        if self.model_family not in valid_families:
            errors.append(f"Invalid model_family: {self.model_family}")

        if self.cv_folds < 2:
            errors.append(f"cv_folds must be >= 2: {self.cv_folds}")

        if not 0.0 <= self.validation_split <= 1.0:
            errors.append(f"validation_split must be in [0,1]: {self.validation_split}")

        if self.ensemble_size < 1:
            errors.append(f"ensemble_size must be >= 1: {self.ensemble_size}")

        if not 0.0 <= self.feature_importance_threshold <= 1.0:
            errors.append(f"feature_importance_threshold must be in [0,1]: {self.feature_importance_threshold}")

        if errors:
            logger.error(f"ModelConfig validation failed: {errors}")
            return False

        return True


@dataclass
class ResourceConfig:
    """Configuration for computational resources."""
    # CPU settings
    n_jobs: int = 1
    max_parallel_trials: int = 1

    # Memory settings
    memory_limit: Optional[str] = None  # e.g., "8GB"
    per_trial_memory_limit: Optional[str] = None

    # Time settings
    time_limit: Optional[int] = None  # Total time in seconds
    per_trial_time_limit: Optional[int] = None  # Per trial in seconds
    timeout_check_interval: int = 30  # Seconds

    # GPU settings
    use_gpu: bool = False
    gpu_memory_limit: Optional[str] = None

    # Storage settings
    save_intermediate_results: bool = True
    artifact_retention_days: int = 30
    max_artifact_size: Optional[str] = None  # e.g., "100MB"

    def validate(self) -> bool:
        """Validate resource configuration."""
        errors = []

        if self.n_jobs < 1:
            errors.append(f"n_jobs must be >= 1: {self.n_jobs}")

        if self.max_parallel_trials < 1:
            errors.append(f"max_parallel_trials must be >= 1: {self.max_parallel_trials}")

        if self.per_trial_time_limit and self.per_trial_time_limit < 1:
            errors.append(f"per_trial_time_limit must be >= 1: {self.per_trial_time_limit}")

        if self.timeout_check_interval < 1:
            errors.append(f"timeout_check_interval must be >= 1: {self.timeout_check_interval}")

        if self.artifact_retention_days < 1:
            errors.append(f"artifact_retention_days must be >= 1: {self.artifact_retention_days}")

        if errors:
            logger.error(f"ResourceConfig validation failed: {errors}")
            return False

        return True


@dataclass
class LoggingConfig:
    """Configuration for logging and monitoring."""
    # Experiment tracking
    track_trials: bool = True
    track_intermediate_values: bool = True
    log_model_artifacts: bool = True
    log_predictions: bool = False

    # Progress reporting
    progress_report_interval: int = 10  # Trials
    detailed_logging: bool = False
    log_parameter_importance: bool = True

    # Alerts
    alert_on_failure: bool = True
    alert_on_improvement: bool = False
    alert_threshold: float = 0.01  # Relative improvement threshold

    # Storage settings
    log_level: str = "INFO"
    save_study_plots: bool = True
    save_trial_details: bool = True

    # Backend settings
    tracking_backend: str = "null"  # "wandb", "mlflow", "null"
    tracking_project: str = "hyperparameter_optimization"
    tracking_entity: Optional[str] = None

    def validate(self) -> bool:
        """Validate logging configuration."""
        errors = []

        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level not in valid_log_levels:
            errors.append(f"Invalid log_level: {self.log_level}")

        valid_backends = ["wandb", "mlflow", "null"]
        if self.tracking_backend not in valid_backends:
            errors.append(f"Invalid tracking_backend: {self.tracking_backend}")

        if self.progress_report_interval < 1:
            errors.append(f"progress_report_interval must be >= 1: {self.progress_report_interval}")

        if not 0.0 <= self.alert_threshold <= 1.0:
            errors.append(f"alert_threshold must be in [0,1]: {self.alert_threshold}")

        if errors:
            logger.error(f"LoggingConfig validation failed: {errors}")
            return False

        return True


@dataclass
class HyperparameterOptimizationConfig:
    """
    Comprehensive configuration for hyperparameter optimization.

    This class combines all sub-configurations into a single,
    validated configuration object.
    """
    # Core configurations
    problem: ProblemConfig = field(default_factory=ProblemConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    # Optimization settings
    hyperparameter: HyperparameterConfig = field(default_factory=HyperparameterConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)

    # Study settings
    study_name: str = "hyperparameter_optimization"
    study_description: str = ""
    study_tags: List[str] = field(default_factory=list)

    # Search space
    search_spaces: Dict[str, SearchSpace] = field(default_factory=dict)
    search_space_builder_preset: Optional[str] = None

    # Advanced settings
    adaptive_search_space: bool = False
    prune_search_space: bool = True
    max_search_space_combinations: int = 10000

    # Validation
    validate_config: bool = True

    def __post_init__(self):
        """Post-initialization validation and setup."""
        # Set default study name if not provided
        if not self.study_name:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.study_name = f"hpo_{self.problem.problem_type}_{self.model.model_type}_{timestamp}"

        # Set default tags
        if not self.study_tags:
            self.study_tags = [
                self.problem.problem_type,
                self.model.model_type,
                self.model.model_family
            ]

        # Validate configuration if requested
        if self.validate_config:
            if not self.validate():
                raise ValueError("Configuration validation failed")

    def validate(self) -> bool:
        """Validate the entire configuration."""
        logger.info("Validating hyperparameter optimization configuration")

        all_valid = True

        # Validate each sub-configuration
        if not self.problem.validate():
            logger.error("ProblemConfig validation failed")
            all_valid = False

        if not self.model.validate():
            logger.error("ModelConfig validation failed")
            all_valid = False

        if not self.resources.validate():
            logger.error("ResourceConfig validation failed")
            all_valid = False

        if not self.logging.validate():
            logger.error("LoggingConfig validation failed")
            all_valid = False

        # Cross-configuration validation
        validation_errors = []

        # Check resource constraints vs problem size
        if self.resources.n_jobs > self.resources.max_parallel_trials:
            validation_errors.append("n_jobs cannot exceed max_parallel_trials")

        # Check memory constraints
        if self.problem.n_samples > 100000 and not self.resources.memory_limit:
            logger.warning("Large dataset but no memory limit specified")

        # Check time constraints
        if self.hyperparameter.n_trials > 1000 and not self.resources.time_limit:
            logger.warning("Many trials but no time limit specified")

        # Check model-optimizer compatibility
        if self.model.model_type == "svm" and self.hyperparameter.n_trials > 200:
            logger.warning("SVM with many trials may be slow")

        # Check search space
        if not self.search_spaces and not self.search_space_builder_preset:
            validation_errors.append("Either search_spaces or search_space_builder_preset must be specified")

        if validation_errors:
            logger.error(f"Cross-configuration validation failed: {validation_errors}")
            all_valid = False

        if all_valid:
            logger.info("Configuration validation passed")
        else:
            logger.error("Configuration validation failed")

        return all_valid

    def get_effective_trials(self) -> int:
        """Get effective number of trials considering constraints."""
        base_trials = self.hyperparameter.n_trials

        # Adjust for time limits
        if self.resources.per_trial_time_limit and self.resources.time_limit:
            max_trials_by_time = self.resources.time_limit // self.resources.per_trial_time_limit
            base_trials = min(base_trials, max_trials_by_time)

        # Adjust for parallel execution
        if self.resources.max_parallel_trials > 1:
            # May be able to run more trials in parallel
            pass

        return max(1, base_trials)

    def get_recommended_sampler(self) -> str:
        """Get recommended sampler based on configuration."""
        if self.search_space_builder_preset:
            return "tpe"  # TPE works well with most search spaces

        if len(self.search_spaces) <= 5:
            return "grid" if len(self.search_spaces) <= 3 else "tpe"
        elif self.model.model_family == "neural":
            return "cmaes"  # CMA-ES works well for high-dimensional spaces
        else:
            return "tpe"  # Default choice

    def get_recommended_pruner(self) -> str:
        """Get recommended pruner based on configuration."""
        if not self.model.early_stopping:
            return "none"

        if self.resources.per_trial_time_limit:
            return "hyperband"  # Good for time-constrained optimization
        else:
            return "median"  # Conservative default

    def adjust_for_environment(self) -> None:
        """Adjust configuration based on environment variables."""
        # Check for environment variables
        if os.getenv("HPO_DEBUG"):
            self.logging.log_level = "DEBUG"
            self.logging.detailed_logging = True
            self.hyperparameter.n_trials = min(self.hyperparameter.n_trials, 10)

        if os.getenv("HPO_FAST"):
            self.hyperparameter.n_trials = min(self.hyperparameter.n_trials, 50)
            self.resources.max_parallel_trials = min(self.resources.max_parallel_trials, 2)

        if os.getenv("HPO_PRODUCTION"):
            self.logging.track_trials = True
            self.logging.save_study_plots = True
            self.hyperparameter.save_results = True

        # Check for available resources
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().total / (1024**3)

            if not self.resources.memory_limit:
                # Use 80% of available memory
                self.resources.memory_limit = f"{int(available_memory_gb * 0.8)}GB"

            if not self.resources.n_jobs:
                # Use number of CPU cores
                self.resources.n_jobs = psutil.cpu_count()

        except ImportError:
            logger.debug("psutil not available for resource detection")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "problem": self.problem.__dict__,
            "model": self.model.__dict__,
            "resources": self.resources.__dict__,
            "logging": self.logging.__dict__,
            "hyperparameter": self.hyperparameter.__dict__,
            "optuna": self.optuna.__dict__,
            "study_name": self.study_name,
            "study_description": self.study_description,
            "study_tags": self.study_tags,
            "search_space_builder_preset": self.search_space_builder_preset,
            "adaptive_search_space": self.adaptive_search_space,
            "prune_search_space": self.prune_search_space,
            "max_search_space_combinations": self.max_search_space_combinations
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HyperparameterOptimizationConfig':
        """Create configuration from dictionary."""
        # Handle nested configurations
        if "problem" in config_dict:
            config_dict["problem"] = ProblemConfig(**config_dict["problem"])
        if "model" in config_dict:
            config_dict["model"] = ModelConfig(**config_dict["model"])
        if "resources" in config_dict:
            config_dict["resources"] = ResourceConfig(**config_dict["resources"])
        if "logging" in config_dict:
            config_dict["logging"] = LoggingConfig(**config_dict["logging"])
        if "hyperparameter" in config_dict:
            config_dict["hyperparameter"] = HyperparameterConfig(**config_dict["hyperparameter"])
        if "optuna" in config_dict:
            config_dict["optuna"] = OptunaConfig(**config_dict["optuna"])

        return cls(**config_dict)

    def save_to_file(self, file_path: str) -> None:
        """Save configuration to file."""
        config_dict = self.to_dict()

        with open(file_path, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)

        logger.info(f"Configuration saved to {file_path}")

    @classmethod
    def load_from_file(cls, file_path: str) -> 'HyperparameterOptimizationConfig':
        """Load configuration from file."""
        with open(file_path, 'r') as f:
            config_dict = json.load(f)

        return cls.from_dict(config_dict)

    def copy(self) -> 'HyperparameterOptimizationConfig':
        """Create a copy of the configuration."""
        return self.from_dict(self.to_dict())

    def update(self, **kwargs) -> 'HyperparameterOptimizationConfig':
        """Update configuration with new values."""
        config = self.copy()

        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                logger.warning(f"Unknown configuration key: {key}")

        return config

    def get_summary(self) -> str:
        """Get configuration summary."""
        summary = f"""
Hyperparameter Optimization Configuration:
=========================================
Study: {self.study_name}
Problem: {self.problem.problem_type} ({self.problem.target_metric}, {self.problem.metric_direction})
Model: {self.model.model_type} ({self.model.model_family})
Data: {self.problem.n_samples:,} samples, {self.problem.n_features} features
Trials: {self.get_effective_trials()} (base: {self.hyperparameter.n_trials})
Sampler: {self.get_recommended_sampler()}
Pruner: {self.get_recommended_pruner()}
Resources: {self.resources.n_jobs} jobs, {self.resources.max_parallel_trials} parallel
Tracking: {self.logging.tracking_backend} backend
"""
        return summary.strip()


# Factory functions for common configurations
def create_default_config(problem_type: str = "regression",
                         model_type: str = "xgboost",
                         n_trials: int = 100) -> HyperparameterOptimizationConfig:
    """
    Create default configuration for common use cases.

    Args:
        problem_type: Type of problem
        model_type: Type of model
        n_trials: Number of trials

    Returns:
        Default configuration
    """
    config = HyperparameterOptimizationConfig(
        problem=ProblemConfig(problem_type=problem_type),
        model=ModelConfig(model_type=model_type),
        hyperparameter=HyperparameterConfig(n_trials=n_trials),
        search_space_builder_preset=f"{model_type}_default"
    )

    config.adjust_for_environment()
    return config


def create_fast_config(problem_type: str = "regression",
                      model_type: str = "xgboost",
                      n_trials: int = 50) -> HyperparameterOptimizationConfig:
    """
    Create fast configuration for quick experiments.

    Args:
        problem_type: Type of problem
        model_type: Type of model
        n_trials: Number of trials

    Returns:
        Fast configuration
    """
    config = HyperparameterOptimizationConfig(
        problem=ProblemConfig(problem_type=problem_type),
        model=ModelConfig(model_type=model_type),
        resources=ResourceConfig(
            n_jobs=2,
            max_parallel_trials=2,
            per_trial_time_limit=300  # 5 minutes
        ),
        logging=LoggingConfig(
            track_intermediate_values=False,
            detailed_logging=False
        ),
        hyperparameter=HyperparameterConfig(
            n_trials=n_trials,
            early_stopping_patience=10
        ),
        search_space_builder_preset=f"{model_type}_default"
    )

    config.adjust_for_environment()
    return config


def create_production_config(problem_type: str = "regression",
                           model_type: str = "xgboost",
                           n_trials: int = 200) -> HyperparameterOptimizationConfig:
    """
    Create production configuration with comprehensive tracking.

    Args:
        problem_type: Type of problem
        model_type: Type of model
        n_trials: Number of trials

    Returns:
        Production configuration
    """
    config = HyperparameterOptimizationConfig(
        problem=ProblemConfig(problem_type=problem_type),
        model=ModelConfig(model_type=model_type),
        resources=ResourceConfig(
            n_jobs=4,
            max_parallel_trials=4,
            save_intermediate_results=True,
            artifact_retention_days=90
        ),
        logging=LoggingConfig(
            track_trials=True,
            track_intermediate_values=True,
            log_model_artifacts=True,
            detailed_logging=True,
            save_study_plots=True,
            save_trial_details=True,
            tracking_backend="wandb"
        ),
        hyperparameter=HyperparameterConfig(
            n_trials=n_trials,
            save_results=True,
            track_trials=True
        ),
        optuna=OptunaConfig(
            save_study=True,
            generate_plots=True
        ),
        search_space_builder_preset=f"{model_type}_default"
    )

    config.adjust_for_environment()
    return config