"""
Trading strategy implementations.
"""

from .base_strategy import BaseStrategy
from .dual_momentum import DualMomentumStrategy

__all__ = ['BaseStrategy', 'DualMomentumStrategy']