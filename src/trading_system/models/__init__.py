"""
Trading System Models Package.

This package contains the core ML and statistical models used in the trading system:
- FF5 factor model implementations
- ML residual predictors
- Model validation and monitoring utilities
- Unified model interface following SOLID principles

Key Features:
- Clean architecture with single responsibility models
- Modern model interface with BaseModel abstraction
- Factory pattern for model creation
- Registry for model management
- Performance evaluation and monitoring utilities
"""

# New clean model architecture
from .base.base_model import BaseModel, ModelStatus, ModelMetadata
from .base.model_factory import ModelFactory
from .implementations.ff5_model import FF5RegressionModel
from .implementations.residual_model import ResidualPredictionModel
# from .utils.performance_evaluator import PerformanceEvaluator
# from .serving.monitor import ModelMonitor
# from .training.trainer import ModelTrainer, TrainingConfig

# Auto-register models with the factory
from . import registry

# Export main interfaces
__all__ = [
    # Core model interface
    'BaseModel',
    'ModelStatus',
    'ModelMetadata',

    # Model implementations
    'FF5RegressionModel',
    'ResidualPredictionModel',

    # Factory and registry
    'ModelFactory',

    # Training and evaluation
    # 'ModelTrainer',
    # 'TrainingConfig',
    # 'PerformanceEvaluator',

    # Monitoring
    # 'ModelMonitor',

    # Registry (for direct access)
    'registry',
]