"""
Unified Base Strategy - All Strategies Follow the Same Architecture

This is the new base class that enforces a consistent architecture across
all trading strategies:

    FeatureEngineeringPipeline → ModelPredictor → PositionSizer

Key Design Principles:
----------------------
1. **Consistency**: All strategies follow the exact same flow
2. **Separation of Concerns**: 
   - Pipeline: Feature computation
   - Model: Prediction/ranking logic
   - PositionSizer: Risk management
3. **Trainability**: All strategies can be "trained" (even rule-based ones)
4. **Composability**: Easy to swap components

The only difference between strategies is:
- Feature pipeline configuration (what features to compute)
- Model type (ML model vs linear model vs rule-based model)
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..models.serving.predictor import ModelPredictor
from .utils.portfolio_calculator import PortfolioCalculator


logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    """
    Unified base class for all trading strategies.
    
    This enforces the architecture:
        Data → FeaturePipeline → Model → RiskManager → Signals
    
    All strategies MUST follow this flow. The differences between strategies
    are only in the components they use, not in the overall structure.
    
    Example:
        # ML Strategy
        MultiStockMLStrategy(
            pipeline=FeaturePipeline(config=ml_feature_config),
            model=ModelPredictor(model_id="random_forest_v1"),
            ...
        )
        
        # Factor Strategy  
        DualMomentumStrategy(
            pipeline=FeaturePipeline(config=momentum_feature_config),
            model=ModelPredictor(model_id="momentum_ranking_v1"),
            ...
        )
    """
    
    def __init__(self,
                 name: str,
                 feature_pipeline: FeatureEngineeringPipeline,
                 model_predictor: ModelPredictor,
                 **kwargs):
        """
        Initialize unified strategy.
        
        Args:
            name: Strategy identifier
            feature_pipeline: Fitted FeatureEngineeringPipeline
            model_predictor: ModelPredictor with loaded model
            **kwargs: Additional strategy-specific parameters
        """
        self.name = name
        self.feature_pipeline = feature_pipeline
        self.model_predictor = model_predictor
        self.parameters = kwargs
        
        # Signal tracking and diagnostics
        self._last_signals = None
        self._last_price_data = None
        self._last_signal_quality = None
        self._last_position_metrics = None
        self._signal_generation_count = 0
        
        # Validate components
        self._validate_components()
        
        logger.info(f"Initialized {self.__class__.__name__} '{name}' with unified architecture")
    
    def _validate_components(self):
        """Validate that all required components are provided."""
        if not isinstance(self.feature_pipeline, FeatureEngineeringPipeline):
            raise TypeError("feature_pipeline must be FeatureEngineeringPipeline instance")
        if not isinstance(self.model_predictor, ModelPredictor):
            raise TypeError("model_predictor must be ModelPredictor instance")
    
    def generate_signals(self,
                        price_data: Dict[str, pd.DataFrame],
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        """
        Generate trading signals using the unified pipeline.
        
        This method implements the standard flow:
        1. Compute features using FeaturePipeline
        2. Get predictions from Model
        
        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            start_date: Start date for signal generation
            end_date: End date for signal generation
        
        Returns:
            DataFrame with expected returns or raw model predictions.
        """
        if not price_data:
            logger.warning("Empty price data provided")
            return pd.DataFrame()
        
        try:
            logger.info(f"[{self.name}] Generating signals from {start_date} to {end_date}")
            
            # Step 1: Compute features using pipeline
            logger.debug(f"[{self.name}] Step 1: Computing features...")
            features = self._compute_features(price_data)
            
            if features.empty:
                logger.warning(f"[{self.name}] Feature computation returned empty DataFrame")
                return pd.DataFrame()

            logger.info(f"[{self.name}] Features computed successfully: shape={features.shape}, columns={len(features.columns)}")

            # Step 2: Get model predictions
            logger.info(f"[{self.name}] Step 2: Getting model predictions...")
            predictions = self._get_predictions(features, price_data, start_date, end_date)

            logger.info(f"[{self.name}] Predictions returned: shape={predictions.shape}, empty={predictions.empty}")

            if predictions.empty:
                logger.warning(f"[{self.name}] Model predictions returned empty DataFrame")
                return pd.DataFrame()
            
            # Step 3: Evaluate signal quality (NEW)
            logger.debug(f"[{self.name}] Step 3: Evaluating signal quality...")
            self._evaluate_and_cache_signals(predictions, price_data)
            
            logger.info(f"[{self.name}] Generated signals for {len(predictions.columns)} assets")
            return predictions
            
        except Exception as e:
            logger.error(f"[{self.name}] Signal generation failed: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _compute_features(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute features using the feature pipeline.
        
        Args:
            price_data: Dictionary of price DataFrames
        
        Returns:
            DataFrame with computed features
        """
        try:
            # Prepare data in format expected by pipeline
            pipeline_data = {'price_data': price_data}

            logger.info(f"[{self.name}] 🔧 Feature pipeline transform starting...")
            logger.info(f"[{self.name}] Pipeline data keys: {list(pipeline_data.keys())}")
            logger.info(f"[{self.name}] Price data symbols: {list(price_data.keys())}")
            logger.info(f"[{self.name}] Feature pipeline fitted: {getattr(self.feature_pipeline, '_is_fitted', 'Unknown')}")

            # Transform using fitted pipeline
            features = self.feature_pipeline.transform(pipeline_data)

            logger.info(f"[{self.name}] ✅ Feature pipeline transform completed")
            logger.info(f"[{self.name}] Features shape: {features.shape}")
            logger.info(f"[{self.name}] Features columns sample: {list(features.columns[:10])}...")
            logger.debug(f"[{self.name}] Computed {len(features.columns)} features")
            return features
            
        except Exception as e:
            logger.error(f"[{self.name}] Feature computation failed: {e}")
            return pd.DataFrame()
    
    def _get_predictions(self,
                        features: pd.DataFrame,
                        price_data: Dict[str, pd.DataFrame],
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        """
        Get predictions from the model.
        
        Args:
            features: Computed features
            price_data: Original price data
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with predictions (expected returns or signals)
        """
        try:
            logger.info(f"[{self.name}] 🔍 _get_predictions started")
            logger.info(f"[{self.name}] Features shape: {features.shape}, columns: {list(features.columns[:5])}...")
            logger.info(f"[{self.name}] Price data keys: {list(price_data.keys())}")

            # Get predictions for each symbol
            predictions_dict = {}
            symbols_processed = 0

            for symbol in price_data.keys():
                logger.info(f"[{self.name}] Processing symbol {symbol}...")
                symbols_processed += 1

                # Extract symbol features
                symbol_features = self._extract_symbol_features(features, symbol)
                logger.info(f"[{self.name}] Symbol {symbol}: extracted {len(symbol_features.columns)} features")

                if symbol_features.empty:
                    logger.warning(f"[{self.name}] No features found for symbol {symbol}")
                    continue
                
                # Get prediction from model
                # Model should return expected return or signal strength
                logger.debug(f"[{self.name}] Getting prediction for {symbol} with features shape: {symbol_features.shape}")
                result = self.model_predictor.predict(
                    features=symbol_features,
                    symbol=symbol,
                    prediction_date=end_date
                )
                logger.debug(f"[{self.name}] Model prediction result for {symbol}: {result}")

                # Extract prediction value
                prediction_value = result.get('prediction', 0.0)
                predictions_dict[symbol] = prediction_value
                logger.debug(f"[{self.name}] Extracted prediction value for {symbol}: {prediction_value}")
            
            # Convert to DataFrame format
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            predictions_df = pd.DataFrame(
                index=dates,
                columns=list(predictions_dict.keys()),
                data=[list(predictions_dict.values())] * len(dates)
            )

            logger.info(f"[{self.name}] ✅ Created DataFrame: shape={predictions_df.shape}, symbols={len(predictions_dict)}, processed={symbols_processed}")
            logger.info(f"[{self.name}] 📊 Prediction dict keys: {list(predictions_dict.keys())}")
            logger.info(f"[{self.name}] 📅 Date range: {start_date} to {end_date}")
            return predictions_df

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Prediction failed: {e}")
            import traceback
            logger.error(f"[{self.name}] Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _extract_symbol_features(self, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Extract features for a specific symbol.
        
        Args:
            features: Full feature DataFrame
            symbol: Symbol to extract
        
        Returns:
            Features for the symbol
        """
        # Extract features for this symbol using MultiIndex structure
        if isinstance(features.index, pd.MultiIndex):
            # Filter by symbol in MultiIndex
            symbol_features = features.loc[symbol].copy()
            symbol_features.reset_index(drop=True, inplace=True)
            return symbol_features

        # Fallback: return all features (assume single symbol)
        return features
    
    def get_name(self) -> str:
        """Get strategy name."""
        return self.name
    
    def get_parameters(self) -> Dict:
        """Get strategy parameters."""
        return self.parameters.copy()
    
    def get_info(self) -> Dict:
        """
        Get comprehensive strategy information.
        
        Returns:
            Dictionary with strategy metadata
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'architecture': 'UnifiedPipeline',
            'components': self._get_component_info(),
            'pipeline_fitted': getattr(self.feature_pipeline, '_is_fitted', False),
            'model_info': self._get_model_info(),
            'parameters': self.parameters
        }
    
    def _get_component_info(self) -> Dict[str, str]:
        """Get information about the strategy's components."""
        components = {
            'feature_pipeline': str(type(self.feature_pipeline).__name__),
            'model_predictor': str(type(self.model_predictor).__name__),
        }
        return components

    def _get_model_info(self) -> Dict:
        """Get information about the model."""
        model_info = {}
        
        if hasattr(self.model_predictor, 'model_id'):
            model_info['model_id'] = self.model_predictor.model_id
        
        if hasattr(self.model_predictor, 'model'):
            model = self.model_predictor.model
            if hasattr(model, 'model_type'):
                model_info['model_type'] = model.model_type
        
        return model_info
    
    # ============================================================================
    # Signal Quality Evaluation & Diagnostics (NEW)
    # ============================================================================
    
    def _evaluate_and_cache_signals(self, 
                                   signals: pd.DataFrame, 
                                   price_data: Dict[str, pd.DataFrame]):
        """
        Evaluate and cache signal quality metrics.
        
        This method is called automatically after signal generation to provide
        a snapshot of the strategy's current state.
        
        Args:
            signals: Generated signals
            price_data: Price data for context
        """
        try:
            # Cache signals and data
            self._last_signals = signals.copy()
            self._last_price_data = price_data
            self._signal_generation_count += 1
            
            # Evaluate signal quality
            if not signals.empty:
                self._last_signal_quality = PortfolioCalculator.calculate_signal_quality(signals)
                self._last_position_metrics = PortfolioCalculator.calculate_position_metrics(signals)
                
                # Log key metrics
                logger.info(f"[{self.name}] Signal Quality Snapshot:")
                logger.info(f"  - Avg positions: {self._last_position_metrics.get('avg_number_of_positions', 0):.1f}")
                logger.info(f"  - Avg position weight: {self._last_position_metrics.get('avg_position_weight', 0):.3f}")
                logger.info(f"  - Signal intensity: {self._last_signal_quality.get('avg_signal_intensity', 0):.3f}")
                logger.info(f"  - Concentration risk: {PortfolioCalculator.calculate_concentration_risk(signals):.3f}")
            else:
                self._last_signal_quality = {}
                self._last_position_metrics = {}
                
        except Exception as e:
            logger.warning(f"[{self.name}] Signal evaluation failed: {e}")
    
    def evaluate_signal_quality(self, signals: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Evaluate signal quality metrics.
        
        Args:
            signals: Signals to evaluate (defaults to last generated signals)
        
        Returns:
            Dictionary with signal quality metrics including:
            - avg_signal_intensity: Average signal strength
            - max_signal_intensity: Maximum signal strength
            - avg_signal_consistency: How consistently signals are generated
            - signal_frequency: How often signals change
        """
        signals_to_eval = signals if signals is not None else self._last_signals
        
        if signals_to_eval is None or signals_to_eval.empty:
            logger.warning(f"[{self.name}] No signals available for quality evaluation")
            return {}
        
        return PortfolioCalculator.calculate_signal_quality(signals_to_eval)
    
    def analyze_positions(self, signals: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Analyze position characteristics.
        
        Args:
            signals: Signals to analyze (defaults to last generated signals)
        
        Returns:
            Dictionary with position metrics including:
            - avg_number_of_positions: Average number of positions held
            - max_number_of_positions: Maximum number of positions
            - avg_position_weight: Average weight per position
            - max_position_weight: Maximum position weight
            - avg_concentration: Average concentration (largest position)
        """
        signals_to_eval = signals if signals is not None else self._last_signals
        
        if signals_to_eval is None or signals_to_eval.empty:
            logger.warning(f"[{self.name}] No signals available for position analysis")
            return {}
        
        return PortfolioCalculator.calculate_position_metrics(signals_to_eval)
    
    def calculate_concentration_risk(self, signals: Optional[pd.DataFrame] = None) -> float:
        """
        Calculate portfolio concentration risk (HHI).
        
        Args:
            signals: Signals to evaluate (defaults to last generated signals)
        
        Returns:
            Herfindahl-Hirschman Index (0 to 1, higher = more concentrated)
        """
        signals_to_eval = signals if signals is not None else self._last_signals
        
        if signals_to_eval is None or signals_to_eval.empty:
            logger.warning(f"[{self.name}] No signals available for concentration analysis")
            return 0.0
        
        return PortfolioCalculator.calculate_concentration_risk(signals_to_eval)
    
    def __str__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', pipeline={type(self.feature_pipeline).__name__}, model={type(self.model_predictor).__name__})"

