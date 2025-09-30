"""
Data Models Module

Contains unified data structures for backtesting configuration and results.
Designed for backward compatibility while providing clean interfaces.
"""

from .results import BacktestResults

__all__ = [
    "BacktestResults"
]