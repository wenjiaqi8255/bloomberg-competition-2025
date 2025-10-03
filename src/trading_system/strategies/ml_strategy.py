"""
ML Strategy - Unified Architecture

This strategy uses complex ML models (RandomForest, XGBoost, LSTM, etc.)
to predict expected returns based on a comprehensive set of features.

Architecture:
    FeatureEngineeringPipeline → MLModelPredictor → PositionSizer

Feature Pipeline:
    - Computes comprehensive technical features
    - Momentum, volatility, technical indicators, volume, liquidity, etc.

Model:
    - Complex ML model (RandomForest, XGBoost, Neural Network, etc.)
    - TODO: Ensure ML models exist in models/implementations/
      (random_forest_model.py, xgboost_model.py, etc.)

Position Sizing:
    - Volatility-based position sizing
    - Maximum position weight constraints
"""

import logging
from datetime import datetime
from typing import Dict, Optional, List
import pandas as pd

from .base_strategy import BaseStrategy
from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..models.serving.predictor import ModelPredictor
from ..utils.position_sizer import PositionSizer

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """
    Machine Learning trading strategy using the unified architecture.
    
    This strategy demonstrates how complex ML models integrate into
    the unified pipeline framework.
    
    Example Usage:
        # Create feature pipeline for ML (comprehensive features)
        from feature_engineering.models.data_types import FeatureConfig
        
        feature_config = FeatureConfig(
            enabled_features=['momentum', 'volatility', 'technical', 'volume'],
            momentum_periods=[21, 63, 252],
            volatility_windows=[20, 60],
            include_technical=True
        )
        feature_pipeline = FeatureEngineeringPipeline(feature_config)
        
        # Fit pipeline on training data
        feature_pipeline.fit(train_data)
        
        # Load ML model
        # TODO: Ensure RandomForestModel or XGBoostModel exists
        model_predictor = ModelPredictor(model_id="random_forest_v1")
        
        # Create position sizer
        position_sizer = PositionSizer(
            volatility_target=0.15,
            max_position_weight=0.10
        )
        
        # Create strategy
        strategy = MLStrategy(
            name="MLStrategy_RF",
            feature_pipeline=feature_pipeline,
            model_predictor=model_predictor,
            position_sizer=position_sizer,
            universe=["AAPL", "TSLA", "MSFT"], # Example universe
            min_signal_strength=0.1
        )
        
        # Generate signals
        signals = strategy.generate_signals(price_data, start_date, end_date)
    """
    
    def __init__(self,
                 name: str,
                 feature_pipeline: FeatureEngineeringPipeline,
                 model_predictor: ModelPredictor,
                 position_sizer: PositionSizer,
                 universe: List[str],  # Add universe
                 min_signal_strength: float = 0.1,
                 **kwargs):
        """
        Initialize ML strategy.
        
        Args:
            name: Strategy identifier
            feature_pipeline: Fitted feature pipeline (should compute comprehensive features)
            model_predictor: Predictor with ML model loaded
            position_sizer: Position sizing component
            universe: The list of symbols this strategy trades.
            min_signal_strength: Minimum signal strength to act on
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            feature_pipeline=feature_pipeline,
            model_predictor=model_predictor,
            position_sizer=position_sizer,
            universe=universe,  # Pass to parent
            **kwargs
        )
        
        self.min_signal_strength = min_signal_strength
        
        logger.info(f"Initialized MLStrategy '{name}' with min_signal_strength={min_signal_strength}")
    
    def _get_predictions(self,
                        features: pd.DataFrame,
                        price_data: Dict[str, pd.DataFrame],
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        """
        Override to add signal strength filtering for ML predictions.
        
        ML models may output weak signals that should be filtered out.
        """
        # Get base predictions from parent
        predictions = super()._get_predictions(features, price_data, start_date, end_date)
        
        if predictions.empty:
            return predictions
        
        # Filter weak signals
        filtered_predictions = predictions.copy()
        filtered_predictions[filtered_predictions.abs() < self.min_signal_strength] = 0.0
        
        logger.debug(f"[{self.name}] Filtered weak signals (threshold={self.min_signal_strength})")
        
        return filtered_predictions
    
    def get_info(self) -> Dict:
        """Get ML strategy information."""
        info = super().get_info()
        info.update({
            'min_signal_strength': self.min_signal_strength,
            'strategy_type': 'ml_strategy',
            'model_complexity': 'high'
        })
        return info

