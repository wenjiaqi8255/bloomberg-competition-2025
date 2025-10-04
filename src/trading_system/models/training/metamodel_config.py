"""
MetaModel Training Configuration

This module provides configuration classes for MetaModel training that integrate
seamlessly with the existing TrainingPipeline infrastructure.

Key Features:
- Compatible with existing TrainingConfig pattern
- Supports MetaModel-specific parameters
- Integrates with StrategyDataCollector
- Follows existing configuration patterns
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

from ..training.types import TrainingConfig


@dataclass
class MetaModelTrainingConfig:
    """
    Configuration for MetaModel training that integrates with TrainingPipeline.

    This configuration provides MetaModel-specific parameters while maintaining
    compatibility with the existing TrainingPipeline infrastructure.
    """

    # MetaModel-specific parameters
    method: str = 'ridge'  # 'equal', 'lasso', 'ridge', 'dynamic'
    alpha: float = 1.0  # Regularization strength for lasso/ridge
    positive_weights: bool = True  # Enforce positive strategy weights

    # Strategy configuration
    strategies: List[str] = field(default_factory=list)
    strategy_weights: Optional[Dict[str, float]] = None  # Fixed weights if method='equal'

    # Data collection parameters
    data_source: str = 'backtest'  # 'backtest', 'live', 'synthetic'
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    target_benchmark: Optional[str] = None  # e.g., 'SPY'

    # Training parameters (from TrainingConfig)
    use_cross_validation: bool = True
    cv_folds: int = 5
    purge_period: int = 21  # trading days
    embargo_period: int = 5  # trading days

    # Validation (from TrainingConfig)
    validation_split: float = 0.2
    early_stopping: bool = True
    early_stopping_patience: int = 10

    # Performance metrics (from TrainingConfig)
    metrics_to_compute: List[str] = field(default_factory=lambda: ['r2', 'mse', 'mae'])

    # Experiment tracking (from TrainingConfig)
    log_experiment: bool = True
    experiment_name: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

    # Feature engineering (for MetaModel, features are just strategy returns)
    feature_config: Dict[str, Any] = field(default_factory=lambda: {
        'enabled_features': [],  # No complex feature engineering needed
        'normalize_features': False,  # Don't normalize strategy returns
        'steps': [{'type': 'identity', 'name': 'passthrough'}]  # Identity transformation
    })

    # Validation settings specific to MetaModel
    min_weight: float = 0.0  # Minimum strategy weight
    max_weight: float = 1.0  # Maximum strategy weight
    weight_sum_constraint: float = 1.0  # Sum of weights should equal this

    # Performance tracking
    track_strategy_correlation: bool = True
    track_contribution_analysis: bool = True

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Validate TrainingConfig parameters
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")

        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")

        # Validate MetaModel-specific parameters
        if self.method not in ['equal', 'lasso', 'ridge', 'dynamic']:
            raise ValueError(f"Unsupported MetaModel method: {self.method}")

        if self.method in ['lasso', 'ridge'] and self.alpha <= 0:
            raise ValueError("alpha must be positive for lasso/ridge methods")

        if not 0 <= self.min_weight <= self.max_weight <= 1:
            raise ValueError("min_weight must be <= max_weight and both in [0, 1]")

        if self.weight_sum_constraint <= 0:
            raise ValueError("weight_sum_constraint must be positive")

        # Set default experiment name if not provided
        if not self.experiment_name:
            self.experiment_name = f"metamodel_{self.method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def to_training_config(self) -> TrainingConfig:
        """
        Convert to base TrainingConfig for compatibility with existing TrainingPipeline.

        Returns:
            TrainingConfig instance with shared parameters
        """
        return TrainingConfig(
            use_cross_validation=self.use_cross_validation,
            cv_folds=self.cv_folds,
            purge_period=self.purge_period,
            embargo_period=self.embargo_period,
            validation_split=self.validation_split,
            early_stopping=self.early_stopping,
            early_stopping_patience=self.early_stopping_patience,
            metrics_to_compute=self.metrics_to_compute,
            log_experiment=self.log_experiment,
            experiment_name=self.experiment_name,
            tags=self.tags
        )

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model-specific configuration for ModelFactory.

        Returns:
            Dictionary of model configuration parameters
        """
        config = {
            'method': self.method,
            'alpha': self.alpha,
            'positive_weights': self.positive_weights,
            'min_weight': self.min_weight,
            'max_weight': self.max_weight,
            'weight_sum_constraint': self.weight_sum_constraint
        }

        if self.strategy_weights:
            config['strategy_weights'] = self.strategy_weights

        return config

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MetaModelTrainingConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            MetaModelTrainingConfig instance
        """
        # Handle datetime conversion
        if 'start_date' in config_dict and isinstance(config_dict['start_date'], str):
            config_dict['start_date'] = datetime.fromisoformat(config_dict['start_date'])
        if 'end_date' in config_dict and isinstance(config_dict['end_date'], str):
            config_dict['end_date'] = datetime.fromisoformat(config_dict['end_date'])

        return cls(**config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
            else:
                result[key] = value
        return result


@dataclass
class MetaModelExperimentConfig:
    """
    High-level experiment configuration for MetaModel training.

    This provides a convenient interface for setting up complete
    MetaModel training experiments.
    """

    # Required fields (no defaults)
    name: str
    strategies: List[str]
    start_date: datetime
    end_date: datetime

    # Optional fields with defaults
    description: str = ""
    method: str = 'ridge'
    alpha: float = 1.0
    use_cross_validation: bool = True
    cv_folds: int = 5
    registry_path: str = "./models"
    model_name: Optional[str] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_metamodel_config(self) -> MetaModelTrainingConfig:
        """
        Convert to MetaModelTrainingConfig.

        Returns:
            MetaModelTrainingConfig instance
        """
        return MetaModelTrainingConfig(
            method=self.method,
            alpha=self.alpha,
            strategies=self.strategies,
            start_date=self.start_date,
            end_date=self.end_date,
            use_cross_validation=self.use_cross_validation,
            cv_folds=self.cv_folds,
            experiment_name=self.name,
            tags=self.tags
        )

    @classmethod
    def create_from_strategy_results(cls,
                                   strategies: List[str],
                                   start_date: Union[str, datetime],
                                   end_date: Union[str, datetime],
                                   method: str = 'ridge',
                                   **kwargs) -> 'MetaModelExperimentConfig':
        """
        Create experiment config from strategy backtest results.

        Args:
            strategies: List of strategy names
            start_date: Start date for data collection
            end_date: End date for data collection
            method: MetaModel method
            **kwargs: Additional configuration parameters

        Returns:
            MetaModelExperimentConfig instance
        """
        # Handle datetime conversion
        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        name = kwargs.pop('name', f"metamodel_{method}_{start_date.strftime('%Y%m%d')}")

        return cls(
            name=name,
            strategies=strategies,
            start_date=start_date,
            end_date=end_date,
            method=method,
            **kwargs
        )