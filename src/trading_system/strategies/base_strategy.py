"""
Unified Base Strategy - All Strategies Follow the Same Architecture

This is the new base class that enforces a consistent architecture across
all trading strategies:

    FeatureEngineeringPipeline → ModelPredictor

Key Design Principles:
----------------------
1. **Consistency**: All strategies follow the exact same flow
2. **Separation of Concerns**: 
   - Pipeline: Feature computation
   - Model: Prediction/ranking logic
3. **Trainability**: All strategies can be "trained" (even rule-based ones)
4. **Composability**: Easy to swap components

The only difference between strategies is:
- Feature pipeline configuration (what features to compute)
- Model type (ML model vs linear model vs rule-based model)
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Any, List
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
                 feature_pipeline: Optional[FeatureEngineeringPipeline] = None,
                 model_predictor: Optional[ModelPredictor] = None,
                 **kwargs):
        """
        Initialize unified strategy.

        Args:
            name: Strategy identifier
            feature_pipeline: Fitted FeatureEngineeringPipeline (can be None for MetaStrategy)
            model_predictor: ModelPredictor with loaded model (can be None for MetaStrategy)
            **kwargs: Additional strategy-specific parameters
        """
        self.name = name
        self.feature_pipeline = feature_pipeline
        self.model_predictor = model_predictor
        self.parameters = kwargs

        # Short selling control (default to long-only for safety)
        self.enable_short_selling = kwargs.get('enable_short_selling', False)

        # Normalization configuration (dual-layer architecture)
        self.enable_normalization = kwargs.get('enable_normalization', True)
        self.normalization_method = kwargs.get('normalization_method', 'minmax')  # 'zscore' or 'minmax'

        # Signal tracking and diagnostics
        self._last_signals = None
        self._last_price_data = None
        self._last_signal_quality = None
        self._last_position_metrics = None
        self._signal_generation_count = 0

        # Validate components (only if provided)
        self._validate_components()

        logger.info(f"Initialized {self.__class__.__name__} '{name}' with unified architecture")
        logger.info(f"Strategy '{name}' short selling: {'enabled' if self.enable_short_selling else 'disabled'}")
        if feature_pipeline is None:
            logger.info(f"Strategy '{name}' feature_pipeline: None (MetaStrategy mode)")
        if model_predictor is None:
            logger.info(f"Strategy '{name}' model_predictor: None (MetaStrategy mode)")
    
    def _validate_components(self):
        """Validate that all provided components are correct type."""
        # Only validate if components are provided (MetaStrategy can have None)
        if self.feature_pipeline is not None:
            if not isinstance(self.feature_pipeline, FeatureEngineeringPipeline):
                raise TypeError("feature_pipeline must be FeatureEngineeringPipeline instance or None")

        if self.model_predictor is not None:
            if not isinstance(self.model_predictor, ModelPredictor):
                raise TypeError("model_predictor must be ModelPredictor instance or None")

        # Log validation results
        if self.feature_pipeline is None and self.model_predictor is None:
            logger.info(f"[{self.name}] Both feature_pipeline and model_predictor are None (MetaStrategy mode)")
        elif self.feature_pipeline is None:
            logger.info(f"[{self.name}] feature_pipeline is None (will use model's pipeline)")
        elif self.model_predictor is None:
            logger.info(f"[{self.name}] model_predictor is None (MetaStrategy mode)")
        else:
            logger.info(f"[{self.name}] Both feature_pipeline and model_predictor are provided")
    
    def generate_signals(self,
                        pipeline_data: Dict[str, Any],
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        """
        Generate trading signals using complete pipeline data.

        ✅ REFACTORED: Following "Data preparation responsibility moves up to orchestrator" pattern.
        This strategy only consumes data, doesn't prepare it.

        Args:
            pipeline_data: Complete data prepared by orchestrator
                - 'price_data': Dict[str, DataFrame] (required) - OHLCV price data
                - 'factor_data': DataFrame (optional) - Factor data for FF5 models
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with expected returns or raw model predictions.
        """
        if not pipeline_data or 'price_data' not in pipeline_data:
            logger.error("Pipeline data missing or invalid")
            return pd.DataFrame()

        try:
            logger.info(f"[{self.name}] Generating signals from {start_date} to {end_date}")
            logger.info(f"[{self.name}] Pipeline data keys: {list(pipeline_data.keys())}")

            # Step 0: Extract and filter valid price data from pipeline_data
            logger.debug(f"[{self.name}] Step 0: Extracting and filtering price data...")
            price_data = pipeline_data['price_data']
            valid_price_data = self._filter_valid_price_data(price_data)

            if not valid_price_data:
                logger.error(f"[{self.name}] No valid price data found after filtering")
                return pd.DataFrame()

            # Update pipeline_data with filtered price data
            pipeline_data_filtered = pipeline_data.copy()
            pipeline_data_filtered['price_data'] = valid_price_data

            removed_symbols = set(price_data.keys()) - set(valid_price_data.keys())
            if removed_symbols:
                logger.info(f"[{self.name}] Excluded {len(removed_symbols)} symbols with invalid data: {sorted(removed_symbols)}")

            # Step 1: Compute features using pipeline (direct consumption)
            logger.debug(f"[{self.name}] Step 1: Computing features...")
            features = self._compute_features(pipeline_data_filtered)

            if features.empty:
                logger.warning(f"[{self.name}] Feature computation returned empty DataFrame")
                return pd.DataFrame()

            logger.info(f"[{self.name}] Features computed successfully: shape={features.shape}, columns={len(features.columns)}")

            # Step 2: Get model predictions
            logger.info(f"[{self.name}] Step 2: Getting model predictions...")
            predictions = self._get_predictions(features, valid_price_data, start_date, end_date)

            logger.info(f"[{self.name}] Predictions returned: shape={predictions.shape}, empty={predictions.empty}")

            if predictions.empty:
                logger.warning(f"[{self.name}] Model predictions returned empty DataFrame")
                return pd.DataFrame()

            # Step 3: Evaluate signal quality (NEW)
            logger.debug(f"[{self.name}] Step 3: Evaluating signal quality...")
            self._evaluate_and_cache_signals(predictions, valid_price_data)

            # Step 4: Apply short selling restrictions if configured
            logger.debug(f"[{self.name}] Step 4: Applying short selling restrictions...")
            filtered_predictions = self._apply_short_selling_restrictions(predictions)

            logger.info(f"[{self.name}] Generated signals for {len(filtered_predictions.columns)} assets")
            return filtered_predictions

        except Exception as e:
            logger.error(f"[{self.name}] Signal generation failed: {e}", exc_info=True)
            return pd.DataFrame()

    def _filter_valid_price_data(self, price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Filter price data to only include symbols with valid, non-empty data.

        Args:
            price_data: Original price data dictionary

        Returns:
            Filtered price data dictionary with only valid entries
        """
        valid_data = {}

        for symbol, data in price_data.items():
            # Check if data exists and is not empty
            if data is not None and not data.empty:
                # Check for essential columns and data quality
                if 'Close' in data.columns and not data['Close'].isna().all():
                    # Check for minimum data requirements (at least some trading days)
                    if len(data) >= 10:  # At least 10 trading days
                        valid_data[symbol] = data
                        logger.debug(f"[{self.name}] Valid data found for {symbol}: {len(data)} trading days")
                    else:
                        logger.warning(f"[{self.name}] Insufficient data for {symbol}: only {len(data)} rows")
                else:
                    logger.warning(f"[{self.name}] No valid Close prices for {symbol}")
            else:
                logger.warning(f"[{self.name}] Empty or None data for {symbol}")

        if not valid_data:
            logger.error(f"[{self.name}] No valid price data found after filtering")
        else:
            logger.info(f"[{self.name}] Filtered to {len(valid_data)} symbols with valid data")

        return valid_data
    
    def _compute_features(self, pipeline_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute features using the feature pipeline.

        Args:
            pipeline_data: Dictionary containing price_data and optionally factor_data

        Returns:
            DataFrame with computed features
        """
        try:
            logger.info(f"[{self.name}] 🔧 Feature pipeline transform starting...")
            logger.info(f"[{self.name}] Pipeline data keys: {list(pipeline_data.keys())}")
            if 'price_data' in pipeline_data:
                logger.info(f"[{self.name}] Price data symbols: {list(pipeline_data['price_data'].keys())}")
            if 'factor_data' in pipeline_data:
                logger.info(f"[{self.name}] Factor data available: {type(pipeline_data['factor_data'])}")
            logger.info(f"[{self.name}] Feature pipeline fitted: {getattr(self.feature_pipeline, '_is_fitted', 'Unknown')}")

            # Use model's feature pipeline if available, otherwise use strategy's pipeline
            feature_pipeline = self._get_feature_pipeline()
            features = feature_pipeline.transform(pipeline_data)

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
        统一预测方法 - 适用于所有模型类型
        
        ModelPredictor 自动处理：
        - Batch-capable: 单次调用预测所有股票（FF5快30倍）
        - Independent: 逐股票调用（per-stock模型必需）
        """
        try:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            symbols = list(price_data.keys())
            
            # 获取预测模式（用于日志）
            try:
                current_model = self.model_predictor.get_current_model()
                prediction_mode = current_model.prediction_mode
            except:
                prediction_mode = 'unknown'
            
            logger.info(
                f"[{self.name}] Starting predictions for {len(dates)} dates, "
                f"{len(symbols)} symbols (mode: {prediction_mode})"
            )
            
            predictions_dict = {}
            
            for i, date in enumerate(dates, 1):
                try:
                    # ====================================================
                    # 关键修复：提取当前日期的特征
                    # 如果预测日期不在features中，使用最后一个可用日期
                    # ====================================================
                    if isinstance(features.index, pd.MultiIndex):
                        logger.debug(f"Date {date}: MultiIndex features, extracting date-specific data")
                        
                        if 'date' in features.index.names:
                            # 有日期索引：提取当前日期的数据
                            try:
                                date_features = features.xs(date, level='date')
                                logger.debug(f"  Extracted {len(date_features)} rows for date {date}")
                            except KeyError:
                                # 预测日期不在features中，使用最后一个可用日期
                                logger.warning(f"  Date {date} not found in features, using last available date")
                                
                                # 获取所有可用日期
                                available_dates = features.index.get_level_values('date').unique()
                                available_dates = pd.to_datetime(available_dates).sort_values()
                                
                                # 找到小于或等于预测日期的最后一个可用日期
                                valid_dates = available_dates[available_dates <= date]
                                if len(valid_dates) > 0:
                                    last_available_date = valid_dates.max()
                                    logger.info(f"  Using last available date: {last_available_date} (requested: {date})")
                                    date_features = features.xs(last_available_date, level='date')
                                else:
                                    # 如果没有可用日期，使用最近的日期
                                    if len(available_dates) > 0:
                                        last_available_date = available_dates.max()
                                        logger.warning(f"  No dates <= {date}, using most recent date: {last_available_date}")
                                        date_features = features.xs(last_available_date, level='date')
                                    else:
                                        logger.error(f"  No available dates in features, skipping prediction")
                                        predictions_dict[date] = pd.Series(0.0, index=symbols)
                                        continue
                        else:
                            # 没有日期索引：使用全部特征（假设是单日）
                            logger.debug(f"  No 'date' level in MultiIndex, using all features")
                            date_features = features
                    else:
                        # 普通索引：假设特征已经按日期准备好
                        logger.debug(f"Date {date}: Regular index features")
                        
                        # 如果是DatetimeIndex，尝试找到可用日期
                        if isinstance(features.index, pd.DatetimeIndex):
                            if date in features.index:
                                date_features = features.loc[[date]]
                                logger.debug(f"  Filtered to single date: {date_features.shape}")
                            else:
                                # 使用最后一个可用日期
                                available_dates = features.index[features.index <= date]
                                if len(available_dates) > 0:
                                    last_available_date = available_dates.max()
                                    logger.info(f"  Date {date} not found, using last available date: {last_available_date}")
                                    date_features = features.loc[[last_available_date]]
                                else:
                                    # 使用最近的日期
                                    if len(features.index) > 0:
                                        last_available_date = features.index.max()
                                        logger.warning(f"  No dates <= {date}, using most recent date: {last_available_date}")
                                        date_features = features.loc[[last_available_date]]
                                    else:
                                        logger.error(f"  No available dates in features, skipping prediction")
                                        predictions_dict[date] = pd.Series(0.0, index=symbols)
                                        continue
                        else:
                            date_features = features
                    
                    logger.debug(f"Date {date}: final features shape = {date_features.shape}")
                    
                    # ====================================================
                    # 单次统一调用 - ModelPredictor 自动优化
                    # ====================================================
                    date_predictions = self.model_predictor.predict(
                        features=date_features,
                        symbols=symbols,
                        date=date
                    )
                    predictions_dict[date] = date_predictions
                    
                    # 进度日志
                    if i % 30 == 0 or i == len(dates):
                        logger.info(f"[{self.name}] Predicted {i}/{len(dates)} dates")
                    
                except Exception as e:
                    logger.error(f"Prediction failed for {date}: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
                    predictions_dict[date] = pd.Series(0.0, index=symbols)
            
            result = pd.DataFrame(predictions_dict).T
            logger.info(
                f"[{self.name}] Prediction complete: {result.shape[0]} dates, "
                f"{result.shape[1]} symbols"
            )
            return result
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()
    

    def _get_feature_pipeline(self):
        """
        Get the appropriate feature pipeline for predictions.

        Returns:
            Feature pipeline to use for transforming features
        """
        # For MetaStrategy, use the strategy's pipeline (should be None, handled by subclass)
        if self.model_predictor is None:
            logger.info(f"[{self.name}] No model_predictor (MetaStrategy mode), using strategy's feature pipeline")
            return self.feature_pipeline

        # Try to use the model's saved feature pipeline first
        try:
            current_model = self.model_predictor.get_current_model()
            if hasattr(current_model, 'feature_pipeline'):
                logger.info(f"[{self.name}] Using model's saved feature pipeline")
                return current_model.feature_pipeline
        except:
            pass
        
        logger.info(f"[{self.name}] Using strategy's feature pipeline (model pipeline not available)")
        return self.feature_pipeline


    def _apply_short_selling_restrictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Apply short selling restrictions to the predictions.

        Args:
            predictions: Raw predictions from the model (may include negative values)

        Returns:
            Filtered predictions with short selling restrictions applied
        """
        if predictions.empty:
            return predictions

        if not self.enable_short_selling:
            # Long-only: filter out negative predictions
            logger.info(f"[{self.name}] Applying long-only constraint (filtering negative signals)")

            # Count negative signals before filtering
            negative_count = (predictions < 0).sum().sum()
            if negative_count > 0:
                logger.info(f"[{self.name}] Filtered out {negative_count} negative signals")

            # Set negative values to 0 (no short positions)
            filtered_predictions = predictions.copy()
            filtered_predictions[filtered_predictions < 0] = 0

            # Renormalize positive signals to maintain proper weighting
            # Only renormalize if there are positive signals
            positive_mask = filtered_predictions > 0
            if positive_mask.any().any():
                # For each date, renormalize positive signals to sum to 1
                for date_idx in filtered_predictions.index:
                    date_signals = filtered_predictions.loc[date_idx]
                    positive_signals = date_signals[positive_mask.loc[date_idx]]

                    if len(positive_signals) > 0:
                        # Normalize positive signals to maintain equal relative weighting
                        # Scale them so that they sum to 1 (or less if few signals)
                        positive_sum = positive_signals.sum()
                        if positive_sum > 0:
                            # Scale to maintain relative proportions but keep within reasonable range
                            scaled_signals = positive_signals / positive_sum
                            # Ensure individual signals don't exceed 1.0
                            scaled_signals = scaled_signals.clip(0, 1.0)
                            filtered_predictions.loc[date_idx, positive_signals.index] = scaled_signals

                logger.debug(f"[{self.name}] Renormalized positive signals after filtering negatives")
            else:
                logger.warning(f"[{self.name}] No positive signals remaining after filtering")

            return filtered_predictions
        else:
            # Short selling enabled - keep all predictions as-is
            logger.debug(f"[{self.name}] Short selling enabled - keeping all signals")
            return predictions
    
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
            'feature_pipeline': str(type(self.feature_pipeline).__name__) if self.feature_pipeline is not None else 'None',
            'model_predictor': str(type(self.model_predictor).__name__) if self.model_predictor is not None else 'None',
        }
        return components

    def _get_model_info(self) -> Dict:
        """Get information about the model."""
        model_info = {}

        if self.model_predictor is None:
            model_info['model_id'] = 'None (MetaStrategy mode)'
            model_info['model_type'] = 'None (MetaStrategy mode)'
            return model_info

        if hasattr(self.model_predictor, 'model_id'):
            model_info['model_id'] = self.model_predictor.model_id

        try:
            model = self.model_predictor.get_current_model()
            if hasattr(model, 'model_type'):
                model_info['model_type'] = model.model_type
        except:
            pass

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

    # ========================================================================
    # NORMALIZATION UTILITY METHODS (DUAL-LAYER ARCHITECTURE)
    # ========================================================================

    def _cross_sectional_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-sectional Z-score normalization to signals.

        This method normalizes each row (each date) independently, ensuring that
        the relative strength of signals within each time period is preserved
        while putting all signals on a common scale.

        Mathematical Formula:
            z_score = (x - mean) / std

        Args:
            df: DataFrame with dates as index and assets as columns

        Returns:
            DataFrame with Z-score normalized values, clipped to [-3, 3] range
        """
        if df.empty:
            return df

        logger.debug(f"[{self.name}] Applying cross-sectional Z-score normalization...")
        logger.debug(f"[{self.name}] Input range: [{df.min().min():.6f}, {df.max().max():.6f}]")

        # Calculate row-wise mean and standard deviation
        row_means = df.mean(axis=1)
        row_stds = df.std(axis=1)

        # Apply Z-score normalization
        normalized = df.sub(row_means, axis=0).div(row_stds, axis=0)

        # Clip extreme values to +/- 3 standard deviations
        normalized = normalized.clip(-3, 3)

        # Fill any remaining NaN values with 0
        normalized = normalized.fillna(0)

        logger.debug(f"[{self.name}] Output range: [{normalized.min().min():.6f}, {normalized.max().max():.6f}]")
        logger.debug(f"[{self.name}] Cross-sectional normalization complete")

        return normalized

    def _min_max_normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-sectional Min-Max normalization to signals.

        This method scales each row (each date) to the range [0, 1] independently.
        Alternative to Z-score normalization when absolute ranking is preferred.

        Mathematical Formula:
            scaled = (x - min) / (max - min)

        Args:
            df: DataFrame with dates as index and assets as columns

        Returns:
            DataFrame with Min-Max normalized values in range [0, 1]
        """
        if df.empty:
            return df

        logger.debug(f"[{self.name}] Applying cross-sectional Min-Max normalization...")

        # Calculate row-wise min and max
        row_mins = df.min(axis=1)
        row_maxs = df.max(axis=1)

        # Apply Min-Max normalization
        row_ranges = row_maxs - row_mins
        # Avoid division by zero for constant rows
        row_ranges = row_ranges.replace(0, 1)

        normalized = df.sub(row_mins, axis=0).div(row_ranges, axis=0)

        # Fill any NaN values with 0.5 (neutral position)
        normalized = normalized.fillna(0.5)

        logger.debug(f"[{self.name}] Min-Max normalization complete")

        return normalized

    def _apply_normalization(self, df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Apply normalization using the specified method.

        Args:
            df: DataFrame to normalize
            method: Normalization method ('zscore' or 'minmax')

        Returns:
            Normalized DataFrame
        """
        if method == 'zscore':
            return self._cross_sectional_zscore(df)
        elif method == 'minmax':
            return self._min_max_normalize(df)
        else:
            logger.warning(f"[{self.name}] Unknown normalization method: {method}, using zscore")
            return self._cross_sectional_zscore(df)

    def __str__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"
    
    def __repr__(self):
        pipeline_name = type(self.feature_pipeline).__name__ if self.feature_pipeline else 'None'
        model_name = type(self.model_predictor).__name__ if self.model_predictor else 'None'
        return f"{self.__class__.__name__}(name='{self.name}', pipeline={pipeline_name}, model={model_name})"

