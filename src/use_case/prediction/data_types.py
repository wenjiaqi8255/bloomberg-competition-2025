"""
Data types for prediction service.

Defines the core data structures used throughout the prediction pipeline,
supporting both single and multi-model scenarios with detailed box information.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
from datetime import datetime

from ...trading_system.portfolio_construction.models.types import BoxKey


@dataclass
class StockRecommendation:
    """
    Individual stock recommendation with all relevant details.
    
    Contains the complete information needed for investment decisions,
    including signal strength, box classification, and risk metrics.
    """
    symbol: str
    weight: float
    signal_strength: float  # From strategy.generate_signals()
    box_classification: Optional[BoxKey]  # From BoxConstructionResult
    risk_score: float
    
    def __post_init__(self):
        """Validate recommendation data."""
        if self.weight < 0 or self.weight > 1:
            raise ValueError(f"Weight must be between 0 and 1, got {self.weight}")
        if not self.symbol or not isinstance(self.symbol, str):
            raise ValueError(f"Symbol must be non-empty string, got {self.symbol}")


@dataclass
class PredictionResult:
    """
    Complete prediction result with multi-model support and box details.
    
    This structure captures all information from the prediction pipeline,
    supporting both single models and meta-model ensembles with detailed
    portfolio construction information.
    """
    # Core outputs
    recommendations: List[StockRecommendation]
    portfolio_weights: pd.Series
    
    # Box-based details (from BoxConstructionResult if using box_based method)
    box_allocations: Optional[Dict[str, float]]  # From BoxConstructionResult.box_coverage
    stocks_by_box: Optional[Dict[str, List[str]]]  # From BoxConstructionResult.selected_stocks
    box_construction_log: Optional[List[str]]  # From BoxConstructionResult.construction_log
    
    # Model information (support multi-model)
    strategy_type: str  # 'ff5', 'ml', 'meta', etc.
    model_id: str  # Single model ID or meta-model ID
    base_model_ids: Optional[List[str]]  # For meta-models
    model_weights: Optional[Dict[str, float]]  # For meta-models
    
    # Metadata
    prediction_date: datetime
    total_positions: int
    portfolio_method: str  # 'box_based' or 'quantitative'
    
    # Risk metrics (from portfolio construction)
    expected_return: float
    expected_risk: float
    diversification_score: float
    
    def __post_init__(self):
        """Validate prediction result data."""
        if not self.recommendations:
            raise ValueError("Recommendations list cannot be empty")
        if self.total_positions <= 0:
            raise ValueError(f"Total positions must be positive, got {self.total_positions}")
        if not self.strategy_type:
            raise ValueError("Strategy type cannot be empty")
        if not self.model_id:
            raise ValueError("Model ID cannot be empty")
    
    @property
    def is_meta_model(self) -> bool:
        """Check if this is a meta-model prediction."""
        return self.strategy_type == 'meta' and self.base_model_ids is not None
    
    def top_recommendations(self, n: int = 10) -> List[StockRecommendation]:
        """Get top N recommendations by weight."""
        return sorted(self.recommendations, key=lambda x: x.weight, reverse=True)[:n]
    
    def get_recommendations_by_box(self, box_key: str) -> List[StockRecommendation]:
        """Get all recommendations for a specific box."""
        if not self.stocks_by_box or box_key not in self.stocks_by_box:
            return []
        
        box_stocks = self.stocks_by_box[box_key]
        return [rec for rec in self.recommendations if rec.symbol in box_stocks]
    
    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for reporting."""
        return {
            'strategy_type': self.strategy_type,
            'model_id': self.model_id,
            'prediction_date': self.prediction_date.isoformat(),
            'total_positions': self.total_positions,
            'portfolio_method': self.portfolio_method,
            'expected_return': self.expected_return,
            'expected_risk': self.expected_risk,
            'diversification_score': self.diversification_score,
            'is_meta_model': self.is_meta_model,
            'base_model_ids': self.base_model_ids,
            'model_weights': self.model_weights,
            'box_allocations': self.box_allocations,
            'top_5_stocks': [rec.symbol for rec in self.top_recommendations(5)],
            'top_5_weights': [rec.weight for rec in self.top_recommendations(5)]
        }
