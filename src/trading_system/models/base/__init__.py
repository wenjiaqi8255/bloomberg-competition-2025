"""
Base Model Module

This module provides abstract base classes for all ML models in the trading system.
It defines the contract that all models must follow, ensuring consistency
and enabling polymorphic usage throughout the system.

Key Design Principles:
- Single Responsibility: Models only do prediction logic
- Open/Closed: Easy to add new model types
- Dependency Inversion: Strategies depend on abstractions, not implementations
- Interface Segregation: Small, focused interfaces
"""

from .base_model import BaseModel, ModelMetadata, ModelStatus
from .model_factory import ModelFactory, ModelTypeRegistry

__all__ = [
    'BaseModel',
    'ModelMetadata',
    'ModelStatus',
    'ModelFactory',
    'ModelTypeRegistry'
]