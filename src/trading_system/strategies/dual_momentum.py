"""
Dual Momentum Strategy - Unified Architecture

This strategy implements the classic dual momentum approach using
the unified pipeline → model → position sizing architecture.

Dual Momentum Components:
1. Absolute Momentum: Filter for positive returns
2. Relative Momentum: Rank and select top performers
3. Equal weighting of selected assets

Architecture:
    FeatureEngineeringPipeline → MomentumRankingModel → PositionSizer

Feature Pipeline:
    - Computes momentum features for different periods
    - momentum_21d, momentum_63d, momentum_252d
    - Optional: volatility for risk-adjusted momentum

Model:
    - MomentumRankingModel (linear/rule-based)
    - TODO: Implement models/implementations/momentum_model.py
    - Can be rule-based or trainable

Position Sizing:
    - Volatility-based position sizing
    - Maximum position weight constraints

Key Insight:
    This is NOT just a "factor calculation" - it's a MODEL that can be:
    - Trained to find optimal momentum period combinations
    - Optimized for different market regimes
    - Backtested with proper train/test splits
"""

import logging
from datetime import datetime
from typing import Dict, Optional
import pandas as pd

from .base_strategy import BaseStrategy
from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..models.serving.predictor import ModelPredictor
from ..utils.position_sizer import PositionSizer

logger = logging.getLogger(__name__)


class DualMomentumStrategy(BaseStrategy):
    """
    Dual Momentum strategy using the unified architecture.
    
    The key insight is that dual momentum is essentially a MODEL:
    - It takes momentum features as input
    - It outputs signals (top N selections)
    - It can be trained to optimize momentum weights
    
    Example Usage:
        # Create feature pipeline for momentum
        from feature_engineering.models.data_types import FeatureConfig
        
        feature_config = FeatureConfig(
            enabled_features=['momentum'],
            momentum_periods=[21, 63, 252]  # Short, medium, long-term
        )
        feature_pipeline = FeatureEngineeringPipeline(feature_config)
        
        # Fit pipeline on training data
        feature_pipeline.fit(train_data)
        
        # Load momentum ranking model
        # TODO: This model needs to be implemented!
        model_predictor = ModelPredictor(model_id="momentum_ranking_v1")
        
        # The model config should specify:
        # - top_n: 5 (select top 5 performers)
        # - min_momentum: 0.0 (absolute momentum filter)
        # - mode: 'rule_based' or 'trainable'
        
        # Create position sizer
        position_sizer = PositionSizer(
            volatility_target=0.15,
            max_position_weight=0.10
        )
        
        # Create strategy
        strategy = DualMomentumStrategy(
            name="DualMomentum252",
            feature_pipeline=feature_pipeline,
            model_predictor=model_predictor,
            position_sizer=position_sizer,
            lookback_period=252  # Primary momentum period
        )
        
        # Generate signals
        signals = strategy.generate_signals(price_data, start_date, end_date)
    """
    
    def __init__(self,
                 name: str,
                 feature_pipeline: FeatureEngineeringPipeline,
                 model_predictor: ModelPredictor,
                 position_sizer: PositionSizer,
                 lookback_period: int = 252,
                 **kwargs):
        """
        Initialize Dual Momentum strategy.
        
        Args:
            name: Strategy identifier
            feature_pipeline: Fitted pipeline (should compute momentum features)
            model_predictor: Predictor with MomentumRankingModel loaded
            position_sizer: Position sizing component
            lookback_period: Primary momentum lookback period
            **kwargs: Additional parameters (top_n, min_momentum, etc.)
        """
        super().__init__(
            name=name,
            feature_pipeline=feature_pipeline,
            model_predictor=model_predictor,
            position_sizer=position_sizer,
            **kwargs
        )
        
        self.lookback_period = lookback_period
        
        logger.info(f"Initialized DualMomentumStrategy '{name}' with lookback={lookback_period}d")
    
    def get_info(self) -> Dict:
        """Get dual momentum strategy information."""
        info = super().get_info()
        info.update({
            'lookback_period': self.lookback_period,
            'strategy_type': 'dual_momentum',
            'model_complexity': 'low',
            'model_expected': 'MomentumRankingModel'
        })
        return info


# TODO: REQUIRED MODEL IMPLEMENTATION
# ------------------------------------
# File: models/implementations/momentum_model.py
# Class: MomentumRankingModel(BaseModel)
#
# This model should implement:
#
# class MomentumRankingModel(BaseModel):
#     def __init__(self, model_type="momentum_ranking", config=None):
#         # config should include:
#         # - top_n: number of assets to select
#         # - min_momentum: minimum momentum threshold
#         # - mode: 'rule_based' or 'trainable'
#         # - momentum_weights: optional weights for different periods
#         pass
#     
#     def fit(self, X, y):
#         # If mode is 'trainable':
#         #   Learn optimal weights for momentum combination
#         #   E.g., optimal_momentum = w1*mom_21d + w2*mom_63d + w3*mom_252d
#         # If mode is 'rule_based':
#         #   Just use the specified momentum period
#         pass
#     
#     def predict(self, X):
#         # Input X columns: momentum_21d, momentum_63d, momentum_252d
#         # 
#         # Steps:
#         # 1. Calculate composite momentum (weighted or single period)
#         # 2. Apply absolute momentum filter (>= min_momentum)
#         # 3. Rank remaining assets by momentum
#         # 4. Select top N
#         # 5. Return signals: 1.0 for selected, 0.0 for others
#         #    OR return momentum scores for position sizing
#         pass
#
# Register the model:
#   ModelFactory.register('momentum_ranking', MomentumRankingModel)

