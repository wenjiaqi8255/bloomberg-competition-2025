"""
Multi-Model Experiment Module
============================

This module provides a comprehensive multi-model experiment framework that:
1. Trains multiple base models with hyperparameter optimization
2. Combines them using a metamodel with its own HPO
3. Evaluates the complete system with real backtesting

Key Components:
- MultiModelOrchestrator: Main orchestrator for the complete experiment
- ModelTrainerWithHPO: Trains individual models with HPO
- MetaModelTrainerWithHPO: Trains metamodel with HPO

Architecture:
- Uses ExperimentOrchestrator for each base model training+backtest
- Uses MetaModelPipeline for metamodel training
- All results are derived from real training, prediction, and backtesting
- No mock or simulated data anywhere
"""

from .multi_model_orchestrator import MultiModelOrchestrator
from .components.model_trainer import ModelTrainerWithHPO
from .components.metamodel_trainer import MetaModelTrainerWithHPO

__all__ = [
    'MultiModelOrchestrator',
    'ModelTrainerWithHPO', 
    'MetaModelTrainerWithHPO'
]
