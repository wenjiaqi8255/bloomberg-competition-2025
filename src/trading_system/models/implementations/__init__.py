"""
Model Implementations

This package contains concrete implementations of ML models that extend
the BaseModel interface. These implementations follow SOLID principles
and focus on single responsibility.

Key Features:
- Clean architecture with single responsibility models
- Standardized interface through BaseModel abstraction
- Easy testing and validation
- No external dependencies on training/monitoring logic
"""

from .ff5_model import FF5RegressionModel
from .momentum_model import MomentumRankingModel

# Optional ML models (require additional dependencies)
__all__ = [
    'FF5RegressionModel',
    'MomentumRankingModel',
]

try:
    from .xgboost_model import XGBoostModel
    __all__.append('XGBoostModel')
except ImportError:
    pass

try:
    from .lstm_model import LSTMModel
    __all__.append('LSTMModel')
except ImportError:
    pass