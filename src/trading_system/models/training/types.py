"""
Dataclasses for model training configuration and results.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import pandas as pd
from ..base.base_model import BaseModel


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Model info
    model_type: str = "base_model"
    model_params: Dict[str, Any] = field(default_factory=dict)

    # Training parameters
    use_cross_validation: bool = True
    cv_folds: int = 5
    cv_method: str = "purged"  # Options: "purged", "expanding", "walk_forward"
    purge_period: int = 21  # trading days
    embargo_period: int = 5  # trading days

    # Validation
    validation_split: float = 0.2
    early_stopping: bool = True
    early_stopping_patience: int = 10

    # Performance metrics
    metrics_to_compute: List[str] = field(default_factory=lambda: ['r2', 'mse', 'mae'])

    # Experiment tracking
    log_experiment: bool = True
    experiment_name: str = ""
    tags: Dict[str, str] = field(default_factory=dict)
    enable_wandb: bool = False
    wandb_project: str = "trading-system"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 < self.validation_split < 1:
            raise ValueError("validation_split must be between 0 and 1")

        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")
            
        if self.cv_method not in ["purged", "expanding", "walk_forward"]:
            raise ValueError("cv_method must be one of: 'purged', 'expanding', 'walk_forward'")
            
        if self.purge_period < 0:
            raise ValueError("purge_period must be non-negative")
            
        if self.embargo_period < 0:
            raise ValueError("embargo_period must be non-negative")


@dataclass
class TrainingResult:
    """Results from model training."""
    model: BaseModel
    training_time: float
    cv_results: Optional[Dict[str, Any]] = None
    validation_metrics: Optional[Dict[str, float]] = None
    test_metrics: Optional[Dict[str, float]] = None
    training_history: List[Dict[str, float]] = field(default_factory=list)
    best_params: Optional[Dict[str, Any]] = None
    feature_pipeline: Optional[Any] = None  # Fitted feature pipeline from CV

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of training results."""
        return {
            'model_type': self.model.model_type,
            'status': self.model.status,
            'training_time': self.training_time,
            'validation_metrics': self.validation_metrics,
            'test_metrics': self.test_metrics,
            'best_params': self.best_params
        }

