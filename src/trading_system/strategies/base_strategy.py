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
from typing import Dict, Optional, Any, List
import pandas as pd
import numpy as np

from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..models.serving.predictor import ModelPredictor
from ..models.serving.prediction_pipeline import PredictionPipeline
from ..utils.position_sizer import PositionSizer
from ..utils.risk import CovarianceEstimator, SimpleCovarianceEstimator, LedoitWolfCovarianceEstimator
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
                 universe: List[str],  # Add universe parameter
                 risk_estimator: Optional[CovarianceEstimator] = None,
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
            universe: The list of symbols this strategy is allowed to trade.
            risk_estimator: Optional CovarianceEstimator for risk assessment
            stock_classifier: Optional StockClassifier for box-based logic
            box_allocator: Optional BoxAllocator for box-based allocation
            data_provider: Optional price data provider (for PredictionPipeline)
            factor_data_provider: Optional factor data provider (for PredictionPipeline)
            **kwargs: Additional strategy-specific parameters, including configs
                      like 'risk_model_config' and 'position_sizer_config'.
        """
        self.name = name
        self.universe = universe  # Store the universe
        self.feature_pipeline = feature_pipeline
        self.model_predictor = model_predictor
        
        # Configure PositionSizer from kwargs if config is provided
        position_sizer_config = kwargs.get('position_sizer_config', {})
        if position_sizer_config:
            self.position_sizer = PositionSizer(
                volatility_target=position_sizer_config.get('volatility_target', position_sizer.volatility_target),
                max_position_weight=position_sizer_config.get('max_position_weight', position_sizer.max_position_weight),
                max_leverage=position_sizer_config.get('max_leverage', position_sizer.max_leverage),
                min_position_weight=position_sizer_config.get('min_position_weight', position_sizer.min_position_weight),
                kelly_fraction=position_sizer_config.get('kelly_fraction', position_sizer.kelly_fraction)
            )
            logger.info(f"Reconfigured PositionSizer from 'position_sizer_config'.")
        else:
            self.position_sizer = position_sizer

        self.risk_estimator = risk_estimator or self._create_risk_estimator(kwargs.get('risk_model_config', {}))
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
    
    def _create_risk_estimator(self, risk_config: Dict) -> CovarianceEstimator:
        """Factory method to create a covariance estimator from config."""
        estimator_type = risk_config.get('type', 'simple').lower()
        lookback_days = risk_config.get('lookback_days', 252)
        
        if estimator_type == 'ledoit_wolf':
            logger.info(f"Initializing LedoitWolfCovarianceEstimator with lookback={lookback_days} days.")
            return LedoitWolfCovarianceEstimator(lookback_days=lookback_days)
        elif estimator_type == 'simple':
            logger.info(f"Initializing SimpleCovarianceEstimator with lookback={lookback_days} days.")
            return SimpleCovarianceEstimator(lookback_days=lookback_days)
        else:
            logger.warning(f"Unknown risk estimator type '{estimator_type}'. Defaulting to SimpleCovarianceEstimator.")
            return SimpleCovarianceEstimator(lookback_days=lookback_days)
    
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
        Main entry point for signal generation, supporting date ranges.
        This method generates signals for the entire date range by calling
        the single-date pipeline at regular rebalancing intervals.
        """
        logger.info(f"[{self.name}] Generating signals for date range {start_date.date()} to {end_date.date()}.")

        # Generate signals at regular intervals (monthly rebalancing for faster execution)
        signal_dates = pd.date_range(start=start_date, end=end_date, freq='W')

        all_signals = []

        for signal_date in signal_dates:
            logger.debug(f"[{self.name}] Generating signals for {signal_date.date()}")
            result = self.generate_signals_single_date(signal_date)
            weights_df = result.get('weights', pd.DataFrame())

            if not weights_df.empty:
                # Update the index to use the signal_date
                weights_df.index = [signal_date]
                all_signals.append(weights_df)

        if not all_signals:
            logger.warning(f"[{self.name}] No signals generated for any date in range")
            return pd.DataFrame()

        # Combine all signals
        combined_signals = pd.concat(all_signals, ignore_index=False)
        logger.info(f"[{self.name}] Generated signals for {len(combined_signals)} dates from {start_date.date()} to {end_date.date()}")

        return combined_signals

    def generate_signals_single_date(self, current_date: datetime) -> Dict:
        """
        Orchestrates the full signal generation pipeline for a single date.
        
        This new implementation follows a clear, multi-step process:
        1. Generate raw alpha signals (pure prediction).
        2. Convert alpha to expected returns.
        3. Estimate the covariance matrix for risk assessment.
        4. Apply risk adjustments to get target weights.
        5. Apply final constraints (e.g., max position size, leverage).
        
        Returns:
            A dictionary containing the final weights and diagnostic information.
        """
        from datetime import timedelta
        logger.info(f"[{self.name}] Generating signals for {current_date.date()} using full pipeline.")

        try:
            # Define lookback period for feature calculation
            lookback_start = current_date - timedelta(days=3 * 365)
            price_data = self.data_provider.get_data(start_date=lookback_start, end_date=current_date, symbols=self.universe)

            if not price_data:
                logger.error(f"[{self.name}] Could not retrieve price data for date {current_date.date()}.")
                return {'weights': pd.DataFrame()}

            # Step 1: Generate raw, z-score normalized alpha signals
            alpha_scores = self.generate_raw_alpha_signals(price_data, current_date)
            if alpha_scores.empty:
                logger.warning(f"[{self.name}] No alpha signals were generated.")
                return {'weights': pd.DataFrame()}

            # Step 2: Convert alpha scores to expected returns
            expected_returns = self.alpha_to_expected_returns(alpha_scores)

            # Step 3: Estimate covariance matrix using the configured risk estimator
            cov_matrix = self.risk_estimator.estimate(price_data, current_date)

            # Step 4: Apply risk adjustment to get initial weights
            risk_adjusted_weights = self.apply_risk_adjustment(expected_returns, cov_matrix)

            # Step 5: Apply final constraints (e.g., max position size, leverage)
            final_weights = self._apply_constraints(risk_adjusted_weights)
            
            # Step 6: Generate signal dataframe with current date
            # Use current date to ensure price data availability and stay within backtest period
            # The model prediction logic remains unchanged (predicting future returns)
            # But signal dates use current date for backtest execution

            # Find the last trading day on or before current_date
            # This ensures we have price data available for the signal date
            signal_dates = pd.date_range(end=current_date, periods=5, freq='B')
            if len(signal_dates) == 0:
                # If no business days in range, use the date directly
                signal_date = current_date
            else:
                # Use the most recent business day
                signal_date = signal_dates[-1]

            final_weights_df = pd.DataFrame(index=[signal_date], columns=final_weights.columns)
            final_weights_df.loc[signal_date] = final_weights.iloc[0]

            # Step 7: Evaluate signal quality
            if hasattr(self, '_evaluate_and_cache_signals'):
                self._evaluate_and_cache_signals(final_weights_df, price_data)

            logger.info(f"[{self.name}] Successfully generated signals for {len(final_weights.columns)} assets.")
            logger.debug(f"[{self.name}] Returning weights DataFrame with shape: {final_weights_df.shape}, index: {final_weights_df.index}")

            result = {
                'weights': final_weights_df,
                'alpha_scores': alpha_scores,
                'expected_returns': expected_returns,
                'risk_adjusted_weights': risk_adjusted_weights,
                'cov_matrix': cov_matrix,
                'metadata': {
                    'date': current_date,
                    'n_positions': (final_weights != 0).T.sum().iloc[0]
                }
            }
            logger.debug(f"[{self.name}] Result keys: {list(result.keys())}, weights empty: {result['weights'].empty}")
            return result
        except Exception as e:
            logger.error(f"[{self.name}] Signal generation failed for date {current_date.date()}: {e}", exc_info=True)
            return {'weights': pd.DataFrame()}

    def generate_raw_alpha_signals(self, price_data: Dict, date: datetime) -> pd.DataFrame:
        """
        Generates raw alpha signals from model predictions and normalizes them.
        This step is purely for prediction; no risk adjustment is applied here.
        
        Returns:
            A DataFrame with z-score normalized alpha scores for each asset.
        """
        logger.debug(f"[{self.name}] Step 1: Generating raw alpha signals...")
        
        from datetime import timedelta
        # Reuse the existing prediction logic
        predictions = self._get_forward_predictions(
            current_date=date,
            lookback_start=date - timedelta(days=3*365)
        )
        
        if predictions.empty:
            return pd.DataFrame()

        pred_series = predictions.iloc[0] # Predictions are a single-row DataFrame

        # Debug: Log prediction values
        logger.info(f"[{self.name}] Raw predictions: {pred_series.to_dict()}")
        logger.info(f"[{self.name}] Prediction stats - mean: {pred_series.mean():.6f}, std: {pred_series.std():.6f}")
        logger.info(f"[{self.name}] Prediction range - min: {pred_series.min():.6f}, max: {pred_series.max():.6f}")

        # Normalize predictions to z-scores to create alpha signals, handle no variance case
        std_dev = pred_series.std()
        if std_dev > 0:
            alpha_scores = (pred_series - pred_series.mean()) / std_dev
            logger.info(f"[{self.name}] Alpha scores (z-scores): {alpha_scores.to_dict()}")
        else:
            alpha_scores = pd.Series(0.0, index=pred_series.index) # No signal if no variance
            logger.warning(f"[{self.name}] All predictions are identical (std={std_dev:.6f}), setting alpha scores to 0")

        return pd.DataFrame([alpha_scores.fillna(0)])

    def alpha_to_expected_returns(self, alpha_scores: pd.DataFrame, scaling_factor: float = 0.02) -> pd.DataFrame:
        """
        Converts z-scored alpha signals into expected returns.

        Args:
            alpha_scores: DataFrame of z-score normalized alpha signals.
            scaling_factor: The expected return for an alpha score of 1.0 (e.g., 0.02 = 2%).
        """
        logger.debug(f"[{self.name}] Step 2: Converting alpha scores to expected returns...")

        # Ensure alpha_scores has a simple numeric index for multiplication
        if not isinstance(alpha_scores.index, (int, range, type(pd.RangeIndex))):
            # Reset index to simple numeric index if it has datetime or other index
            alpha_scores = alpha_scores.reset_index(drop=True)

        return alpha_scores * scaling_factor

    def apply_risk_adjustment(self, expected_returns: pd.DataFrame, cov_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Adjusts expected returns for risk using the covariance matrix.
        This is where portfolio construction techniques like mean-variance optimization or
        Kelly criterion are applied via the PositionSizer.
        """
        logger.debug(f"[{self.name}] Step 4: Applying risk adjustment...")
        logger.info(f"[{self.name}] Expected returns before risk adjustment: {expected_returns.iloc[0].to_dict()}")
        logger.info(f"[{self.name}] Expected returns stats - mean: {expected_returns.iloc[0].mean():.6f}, std: {expected_returns.iloc[0].std():.6f}")

        risk_adjusted = self.position_sizer.adjust_signals_with_covariance(
            raw_signals=expected_returns,
            cov_matrix=cov_matrix
        )

        logger.info(f"[{self.name}] Risk-adjusted weights: {risk_adjusted.iloc[0].to_dict()}")
        logger.info(f"[{self.name}] Risk-adjusted stats - mean: {risk_adjusted.iloc[0].mean():.6f}, std: {risk_adjusted.iloc[0].std():.6f}")

        return risk_adjusted
        
    def _apply_constraints(self, weights: pd.DataFrame) -> pd.DataFrame:
        """
        Applies final position constraints, such as max weight and leverage.
        """
        logger.debug(f"[{self.name}] Step 5: Applying final constraints...")
        constrained_weights = self.position_sizer._apply_position_constraints(weights)
        normalized_weights = self.position_sizer._normalize_weights(constrained_weights)
        return normalized_weights

    def _get_forward_predictions(self,
                               current_date: datetime,
                               lookback_start: datetime) -> pd.DataFrame:
        """
        Get forward-looking predictions from the model by delegating to the PredictionPipeline.
        This method now orchestrates the data fetching and feature engineering for a single point in time.
        """
        if not (self.data_provider and self.prediction_pipeline):
            logger.error(f"[{self.name}] Data provider or prediction pipeline is not available.")
            return pd.DataFrame()

        try:
            # Use the strategy's own configured universe of symbols.
            all_symbols = self.universe
            if not all_symbols:
                logger.warning(f"[{self.name}] No symbols defined in strategy universe.")
                return pd.DataFrame()

            # The prediction pipeline handles fetching data, computing features, and predicting.
            # It uses the full lookback window to correctly calculate features for the `current_date`.
            predictions_dict = self.prediction_pipeline.predict_for_date(
                prediction_date=current_date,
                symbols=all_symbols,
                lookback_start_date=lookback_start
            )
            
            if predictions_dict:
                # Convert the result dict {symbol: prediction_value} to the DataFrame format.
                predictions_df = pd.DataFrame([predictions_dict])
                logger.info(f"[{self.name}] Successfully generated predictions for {len(predictions_dict)} symbols.")
                return predictions_df
            else:
                logger.warning(f"[{self.name}] Prediction pipeline returned no predictions.")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"[{self.name}] Forward prediction failed: {e}", exc_info=True)
            return pd.DataFrame()

    def _apply_forward_position_sizing(self,
                                      predictions: pd.DataFrame,
                                      price_data: Dict[str, pd.DataFrame],
                                      current_date: datetime) -> pd.DataFrame:
        """
        DEPRECATED: This method is now replaced by the new pipeline in generate_signals_single_date.
        This method now redirects to the new pipeline for backward compatibility.
        """
        logger.warning(f"[{self.name}] _apply_forward_position_sizing is deprecated. "
                      f"Redirecting to generate_signals_single_date. The 'predictions' argument will be ignored.")
        result = self.generate_signals_single_date(current_date)
        return result.get('weights', pd.DataFrame())

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
            - avg_number_of_positions: Average number of positions
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

