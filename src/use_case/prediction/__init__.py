"""
Prediction Service Package
=========================

This package provides production-ready prediction services for generating
investment recommendations from trained models.

Key Components:
- PredictionOrchestrator: Main coordinator for prediction workflows
- PredictionResult: Data structures for prediction outputs
- PredictionResultFormatter: Output formatting utilities

Supports:
- Single model predictions (FF5, ML)
- Multi-model ensemble predictions
- Meta-model predictions
- Box-based and quantitative portfolio construction
"""

from .prediction_orchestrator import PredictionOrchestrator
from .data_types import PredictionResult, StockRecommendation
from .formatters import PredictionResultFormatter

__all__ = [
    'PredictionOrchestrator',
    'PredictionResult', 
    'StockRecommendation',
    'PredictionResultFormatter'
]
