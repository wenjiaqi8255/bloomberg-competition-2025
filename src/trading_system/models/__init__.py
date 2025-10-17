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
from .implementations.fama_macbeth_model import FamaMacBethModel

# Export main interfaces
__all__ = [
    # Core model interface
    'BaseModel',
    'ModelStatus',
    'ModelMetadata',

    # Model implementations (always available)
    'FF5RegressionModel',
    'FamaMacBethModel',

    # Factory and registry
    'ModelFactory',

    # Registry (for direct access)
    'registry',
]

# Optional ML models (require additional dependencies)
try:
    from .implementations.xgboost_model import XGBoostModel
    __all__.append('XGBoostModel')
except ImportError:
    pass

try:
    from .implementations.lstm_model import LSTMModel
    __all__.append('LSTMModel')
except ImportError:
    pass

# Auto-register models with the factory (must be after imports)
from . import registry