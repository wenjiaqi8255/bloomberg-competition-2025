"""
Trading Strategies Module - Unified Architecture

All strategies now follow a consistent architecture:
    FeatureEngineeringPipeline → ModelPredictor → PositionSizer

This provides:
- Consistent training/testing methodology
- Easy component swapping
- Unified backtesting
- Clear separation of concerns
"""

# Unified Base Class
from .base_strategy import BaseStrategy

# Unified Strategies (new implementations)
from .ml_strategy import MLStrategy
from .dual_momentum import DualMomentumStrategy  
from .fama_french_5 import FamaFrench5Strategy

# Unified Factory
from .factory import StrategyFactory

__all__ = [
    # Base
    'BaseStrategy',
    
    # Strategies
    'MLStrategy',
    'DualMomentumStrategy',
    'FamaFrench5Strategy',
    
    # Factory
    'StrategyFactory',
]


# Convenience function for quick strategy creation
def create_strategy(config: dict):
    """
    Quick strategy creation from config.
    
    Example:
        strategy = create_strategy({
            'type': 'dual_momentum',
            'name': 'DM',
            'model_id': 'momentum_ranking_v1',
            'lookback_period': 252
        })
    """
    return StrategyFactory.create_from_config(config)

