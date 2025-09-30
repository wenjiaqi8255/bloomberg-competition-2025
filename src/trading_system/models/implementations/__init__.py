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
from .residual_model import ResidualPredictionModel

__all__ = [
    'FF5RegressionModel',
    'ResidualPredictionModel'
]