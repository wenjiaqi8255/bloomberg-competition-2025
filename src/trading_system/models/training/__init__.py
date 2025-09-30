"""
Model Training Infrastructure

This module provides a unified training infrastructure for all ML models
in the trading system. It separates training concerns from model logic
while providing comprehensive validation and monitoring.

Key Features:
- Unified training interface for all model types
- Built-in cross-validation with time series awareness
- Performance evaluation and comparison
- Experiment tracking integration
- Hyperparameter optimization support
"""

from .trainer import ModelTrainer, TrainingResult, TrainingConfig
from .pipeline import TrainingPipeline

__all__ = [
    'ModelTrainer',
    'TrainingResult',
    'TrainingConfig',
    'TrainingPipeline'
]