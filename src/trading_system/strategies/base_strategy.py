"""
Unified Base Strategy - All Strategies Follow the Same Architecture

This is the new base class that enforces a consistent architecture across
all trading strategies:

    PredictionPipeline (Data + Features) → ModelPredictor → PositionSizer

Key Design Principles:
----------------------
1. **Consistency**: All strategies follow the exact same flow
2. **Separation of Concerns**: 
   - PredictionPipeline: Data acquisition & feature computation
   - ModelPredictor: Inference only
   - PositionSizer: Risk management
3. **Trainability**: All strategies can be "trained" (even rule-based ones)
4. **Composability**: Easy to swap components

The only difference between strategies is:
- Feature pipeline configuration (what features to compute)
- Model type (ML model vs linear model vs rule-based model)
- Data providers (price, factors, fundamentals, etc.)
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Any
import pandas as pd
import numpy as np

from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..models.serving.predictor import ModelPredictor
from ..models.serving.prediction_pipeline import PredictionPipeline
from ..utils.position_sizer import PositionSizer
from .utils.portfolio_calculator import PortfolioCalculator
from ..data.stock_classifier import StockClassifier
from ..allocation.box_allocator import BoxAllocator

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
        MLStrategy(
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
                 position_sizer: PositionSizer,
                 stock_classifier: Optional[StockClassifier] = None,
                 box_allocator: Optional[BoxAllocator] = None,
                 data_provider=None,
                 factor_data_provider=None,
                 **kwargs):
        """
        Initialize unified strategy.
        
        Args:
            name: Strategy identifier
            feature_pipeline: Fitted FeatureEngineeringPipeline
            model_predictor: ModelPredictor with loaded model
            position_sizer: PositionSizer for risk management
            stock_classifier: Optional StockClassifier for box-based logic
            box_allocator: Optional BoxAllocator for box-based allocation
            data_provider: Optional price data provider (for PredictionPipeline)
            factor_data_provider: Optional factor data provider (for PredictionPipeline)
            **kwargs: Additional strategy-specific parameters
        """
        self.name = name
        self.feature_pipeline = feature_pipeline
        self.model_predictor = model_predictor
        self.position_sizer = position_sizer
        self.stock_classifier = stock_classifier
        self.box_allocator = box_allocator
        self.parameters = kwargs
        
        # Data providers for prediction pipeline
        self.data_provider = data_provider
        self.factor_data_provider = factor_data_provider
        
        # Create PredictionPipeline if providers are available
        # This enables automatic data acquisition for predictions
        self.prediction_pipeline = None
        if data_provider is not None:
            self.prediction_pipeline = PredictionPipeline(
                model_predictor=model_predictor,
                feature_pipeline=feature_pipeline,
                data_provider=data_provider,
                factor_data_provider=factor_data_provider
            )
            logger.info(f"Created PredictionPipeline with data providers for {name}")
        
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
        if not isinstance(self.position_sizer, PositionSizer):
            raise TypeError("position_sizer must be PositionSizer instance")
    
    def generate_signals(self,
                        price_data: Dict[str, pd.DataFrame],
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        """
        Generate trading signals using the unified pipeline.
        
        This method implements the standard flow:
        1. Compute features using FeaturePipeline
        2. Get predictions from Model
        3. Apply risk management via PositionSizer
        
        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            start_date: Start date for signal generation
            end_date: End date for signal generation
        
        Returns:
            DataFrame with risk-adjusted portfolio weights
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
            
            # Step 2: Get model predictions
            logger.debug(f"[{self.name}] Step 2: Getting model predictions...")
            predictions = self._get_predictions(features, price_data, start_date, end_date)
            
            if predictions.empty:
                logger.warning(f"[{self.name}] Model predictions returned empty DataFrame")
                return pd.DataFrame()
            
            # Step 3: Apply allocation or risk management
            if self.box_allocator and self.stock_classifier:
                logger.debug(f"[{self.name}] Step 3a: Classifying stocks into boxes...")
                all_symbols = list(price_data.keys())
                boxes = self.stock_classifier.classify_stocks(all_symbols, price_data, as_of_date=end_date)
                
                logger.debug(f"[{self.name}] Step 3b: Applying Box-based allocation...")
                final_signals = self.box_allocator.allocate(predictions, boxes)
            else:
                logger.debug(f"[{self.name}] No box allocator or stock classifier provided")
                final_signals = predictions
            
            # Step 4: Evaluate signal quality (NEW)
            logger.debug(f"[{self.name}] Step 4: Evaluating signal quality...")
            self._evaluate_and_cache_signals(final_signals, price_data)
            
            logger.info(f"[{self.name}] Generated signals for {len(final_signals.columns)} assets")
            return final_signals
            
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
            
            # Add factor data if available and needed
            if self.factor_data_provider is not None:
                logger.debug(f"[{self.name}] Fetching factor data for feature computation")
                try:
                    # Get date range from price data
                    all_dates = []
                    for symbol_data in price_data.values():
                        if hasattr(symbol_data.index, 'tolist'):
                            all_dates.extend(symbol_data.index.tolist())
                    
                    if all_dates:
                        start_date = min(all_dates)
                        end_date = max(all_dates)
                        
                        # Fetch factor data for the same period
                        if hasattr(self.factor_data_provider, 'get_factor_data'):
                            factor_data = self.factor_data_provider.get_factor_data(
                                start_date=start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date),
                                end_date=end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
                            )
                        elif hasattr(self.factor_data_provider, 'get_factor_returns'):
                            factor_data = self.factor_data_provider.get_factor_returns(
                                start_date=start_date,
                                end_date=end_date
                            )
                        else:
                            logger.warning(f"Factor data provider has no recognized method")
                            factor_data = None
                        
                        if factor_data is not None and not factor_data.empty:
                            pipeline_data['factor_data'] = factor_data
                            logger.debug(f"[{self.name}] Added factor data: shape={factor_data.shape}")
                        else:
                            logger.warning(f"[{self.name}] No factor data returned")
                
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to fetch factor data: {e}")
            
            # Transform using fitted pipeline
            features = self.feature_pipeline.transform(pipeline_data)
            
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
        Get predictions from the model using pre-computed features.
        
        Args:
            features: Computed features
            price_data: Original price data (for symbol list)
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with predictions (expected returns or signals)
        """
        try:
            # Get predictions for each symbol
            predictions_dict = {}
            
            for symbol in price_data.keys():
                # Extract symbol features
                symbol_features = self._extract_symbol_features(features, symbol)
                
                if symbol_features.empty:
                    logger.warning(f"[{self.name}] No features for {symbol}, skipping prediction")
                    continue
                
                # Get prediction from model (simplified interface - only needs features)
                result = self.model_predictor.predict(
                    features=symbol_features,
                    symbol=symbol,
                    prediction_date=end_date
                )
                
                # Extract prediction value
                prediction_value = result.get('prediction', 0.0)
                predictions_dict[symbol] = prediction_value
            
            # Convert to DataFrame format
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            predictions_df = pd.DataFrame(
                index=dates,
                columns=list(predictions_dict.keys()),
                data=[list(predictions_dict.values())] * len(dates)
            )
            
            logger.debug(f"[{self.name}] Generated predictions for {len(predictions_dict)} symbols")
            return predictions_df
            
        except Exception as e:
            logger.error(f"[{self.name}] Prediction failed: {e}", exc_info=True)
            return pd.DataFrame()
    
    def _extract_symbol_features(self, features: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Extract features for a specific symbol, including both symbol-specific and global features.
        
        Symbol-specific features have prefixes (e.g., 'AAPL_momentum_21d')
        Global features (like FF5 factors) have no prefix (e.g., 'MKT', 'SMB')
        
        Args:
            features: Full feature DataFrame
            symbol: Symbol to extract
        
        Returns:
            Features for the symbol
        """
        # Check if features have MultiIndex structure (symbol, date)
        if isinstance(features.index, pd.MultiIndex) and 'symbol' in features.index.names:
            try:
                # Extract features for this symbol from MultiIndex
                symbol_features = features.xs(symbol, level='symbol')
                logger.debug(f"Extracted features for {symbol} from MultiIndex: shape={symbol_features.shape}")
                return symbol_features
            except KeyError:
                logger.warning(f"Symbol {symbol} not found in MultiIndex features")
                return pd.DataFrame()
        
        # Features may have symbol prefix (e.g., 'AAPL_momentum_21d')
        symbol_cols = [col for col in features.columns if col.startswith(f"{symbol}_")]
        
        # Also get columns without any symbol prefix (global features like FF5 factors)
        # These are columns that don't contain any known symbol prefix
        all_symbols = set()
        for col in features.columns:
            for s in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT', 'SPY', 'QQQ', 'IWM']:
                if col.startswith(f"{s}_"):
                    all_symbols.add(s)
                    break
        
        global_cols = [col for col in features.columns 
                      if not any(col.startswith(f"{s}_") for s in all_symbols)]
        
        if symbol_cols or global_cols:
            # Combine symbol-specific and global features
            cols_to_extract = symbol_cols + global_cols
            symbol_features = features[cols_to_extract].copy()
            
            # Remove symbol prefix from symbol-specific features
            new_columns = []
            for col in cols_to_extract:
                if col in symbol_cols:
                    new_columns.append(col.replace(f"{symbol}_", ""))
                else:
                    new_columns.append(col)  # Keep global feature names as-is
            
            symbol_features.columns = new_columns
            logger.debug(f"Extracted {len(symbol_cols)} symbol-specific + {len(global_cols)} global features for {symbol}")
            return symbol_features
        
        # If no prefix structure, return all features (assume single symbol)
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
            'position_sizer_config': {
                'volatility_target': self.position_sizer.volatility_target,
                'max_position_weight': self.position_sizer.max_position_weight
            },
            'parameters': self.parameters
        }
    
    def _get_component_info(self) -> Dict[str, str]:
        """Get information about the strategy's components."""
        components = {
            'feature_pipeline': str(type(self.feature_pipeline).__name__),
            'model_predictor': str(type(self.model_predictor).__name__),
        }
        if self.box_allocator:
            components['allocator'] = str(type(self.box_allocator).__name__)
            if self.stock_classifier:
                components['classifier'] = str(type(self.stock_classifier).__name__)
        else:
            components['position_sizer'] = str(type(self.position_sizer).__name__)
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

