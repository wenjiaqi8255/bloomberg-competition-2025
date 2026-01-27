"""
Fama-French 5-Factor Strategy - Unified Architecture

This strategy implements the academic Fama-French 5-factor model
using the unified pipeline → model → position sizing architecture.

Fama-French 5 Factors:
1. MKT (Market): Market risk premium
2. SMB (Size): Small Minus Big market cap
3. HML (Value): High Minus Low book-to-market
4. RMW (Profitability): Robust Minus Weak profitability
5. CMA (Investment): Conservative Minus Aggressive investment

Architecture:
    FactorDataProvider → FF5RegressionModel 

Key Difference from other strategies:
    - FF5 uses FACTOR DATA directly from FF5DataProvider
    - Does NOT use technical indicators or traditional feature engineering
    - Model expects columns: ['MKT', 'SMB', 'HML', 'RMW', 'CMA']

Data Flow:
    1. FF5DataProvider provides factor data (MKT, SMB, HML, RMW, CMA, RF)
    2. For each stock, we create factor features based on historical returns
    3. FF5RegressionModel estimates factor betas and predicts expected returns

Model:
    - FF5RegressionModel (ALREADY IMPLEMENTED!)
    - ✅ models/implementations/ff5_model.py
    - Estimates factor betas via linear regression
    - Predicts expected returns from factor exposures

Position Sizing:
    - Volatility-based position sizing
    - Maximum position weight constraints

Key Insight:
    This is a LINEAR MODEL that can be TRAINED:
    - Estimates factor loadings (betas) from historical data
    - Predicts expected returns: E[R] = β_MKT*MKT + β_SMB*SMB + ...
    - Can be backtested with proper train/test methodology
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..models.serving.predictor import ModelPredictor
from ..utils.alpha_stats import compute_alpha_tstat

logger = logging.getLogger(__name__)


class FamaFrench5Strategy(BaseStrategy):
    """
    Fama-French 5-factor strategy using the unified architecture.
    
    This strategy demonstrates how academic factor models fit into
    the unified framework - they're just linear regression models!
    
    Good News: The model is ALREADY IMPLEMENTED!
    ✅ models/implementations/ff5_model.py exists and works
    
    Example Usage:
        # Create feature pipeline for FF5 factors
        from feature_engineering.models.data_types import FeatureConfig
        
        # Configure pipeline to compute the 5 factors
        # (When fundamental data unavailable, use technical proxies)
        feature_config = FeatureConfig(
            enabled_features=['momentum', 'volatility', 'volume'],
            # These will be used to construct factor proxies:
            # - MKT: market returns
            # - SMB: volatility (small cap proxy)
            # - HML: momentum reversal (value proxy)
            # - RMW: return consistency (profitability proxy)
            # - CMA: low volatility (conservative proxy)
        )
        feature_pipeline = FeatureEngineeringPipeline(feature_config)
        
        # Fit pipeline on training data
        feature_pipeline.fit(train_data)
        
        # Load FF5 model (ALREADY EXISTS!)
        model_predictor = ModelPredictor(model_id="ff5_regression_v1")
        
        # The FF5RegressionModel will:
        # 1. During training: estimate factor betas for each stock
        # 2. During prediction: calculate expected returns from factors
        
        # Create strategy
        strategy = FamaFrench5Strategy(
            name="FF5",
            feature_pipeline=feature_pipeline,
            model_predictor=model_predictor,
            lookback_days=252,
            risk_free_rate=0.02
        )
        
        # Generate signals
        signals = strategy.generate_signals(price_data, start_date, end_date)
    """
    
    def __init__(self,
                 name: str,
                 feature_pipeline: FeatureEngineeringPipeline,
                 model_predictor: ModelPredictor,
                 lookback_days: int = 252,
                 risk_free_rate: float = 0.02,
                 **kwargs):
        """
        Initialize Fama-French 5-factor strategy.

        Args:
            name: Strategy identifier
            feature_pipeline: Fitted pipeline (computes technical indicators and merges factor data)
            model_predictor: Predictor with FF5RegressionModel loaded
            lookback_days: Lookback period for factor calculation
            risk_free_rate: Risk-free rate for excess return calculation
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            feature_pipeline=feature_pipeline,
            model_predictor=model_predictor,
            **kwargs
        )

        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate
        
        # Cache for rolling t-stats computation
        self._tstats_cache: Dict[datetime, Dict[str, float]] = {}

        logger.info(f"Initialized FamaFrench5Strategy '{name}' with "
                   f"lookback={lookback_days}d, rf_rate={risk_free_rate}")
        logger.info(f"[{name}] Following SOLID principles - strategy as pure delegate")

    def _compute_features(self, pipeline_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Compute features for FF5 strategy following the new architecture.

        ✅ REFACTORED: Following "Data preparation responsibility moves up to orchestrator" pattern.
        FF5 strategy only uses pipeline_data provided by orchestrator, doesn't access providers.

        Args:
            pipeline_data: Complete data prepared by orchestrator
                - 'price_data': Dict[str, DataFrame] - OHLCV price data
                - 'factor_data': DataFrame - FF5 factor data (MKT, SMB, HML, RMW, CMA, RF)

        Returns:
            DataFrame with features computed by the FeatureEngineeringPipeline
        """
        logger.info(f"[{self.name}] 🔄 Computing features from pipeline data (new architecture)")
        logger.info(f"[{self.name}] Pipeline data keys: {list(pipeline_data.keys())}")

        # Extract price_data and factor_data from pipeline_data
        price_data = pipeline_data.get('price_data', {})
        factor_data = pipeline_data.get('factor_data')

        # Validate required data
        if not price_data:
            logger.error(f"[{self.name}] ❌ No price_data in pipeline_data")
            return pd.DataFrame()

        if factor_data is None:
            logger.error(f"[{self.name}] ❌ No factor_data in pipeline_data - FF5 strategy requires factor data!")
            return pd.DataFrame()

        # Log data info
        symbols_count = len(price_data)
        factor_shape = factor_data.shape if hasattr(factor_data, 'shape') else 'N/A'
        logger.info(f"[{self.name}] Processing {symbols_count} symbols with factor data shape: {factor_shape}")

        # Prepare pipeline input
        pipeline_input = {
            'price_data': price_data,
            'factor_data': factor_data
        }

        try:
            # Delegate to the existing feature pipeline (DRY principle)
            # The pipeline handles technical indicators, factor data merging, and time alignment
            features = self.feature_pipeline.transform(pipeline_input)

            logger.info(f"[{self.name}] ✅ Pipeline computed features: {features.shape}")
            logger.debug(f"[{self.name}] Feature columns: {list(features.columns)}")

            # CRITICAL: Check if FF5 factors are present
            expected_factors = self._expected_factor_columns()
            available_factors = [col for col in features.columns if col in expected_factors]
            missing_factors = set(expected_factors) - set(available_factors)

            if missing_factors:
                logger.error(f"[{self.name}] ❌ Missing FF5 factors: {missing_factors}")
                logger.error(f"[{self.name}] Available columns: {list(features.columns[:10])}...")
            else:
                logger.info(f"[{self.name}] ✅ All FF5 factors present: {available_factors}")

            return features

        except Exception as e:
            logger.error(f"[{self.name}] Error computing features via pipeline: {e}")
            logger.debug(f"[{self.name}] Error details:", exc_info=True)
            return pd.DataFrame()

    def _expected_factor_columns(self) -> list:
        """Get the expected factor columns for FF5 model."""
        return ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
    
    def _determine_required_factors(self, current_model) -> List[str]:
        """Determine required factors based on model type."""
        model_type = getattr(current_model, 'model_type', None)
        if model_type is None:
            # Try to infer from model_id
            model_id = getattr(current_model, 'model_id', '')
            if 'ff3' in str(model_id).lower():
                return ['MKT', 'SMB', 'HML']
            else:
                return ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
        elif 'ff3' in str(model_type).lower():
            return ['MKT', 'SMB', 'HML']
        else:
            return ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
    
    def _extract_factor_values_for_date(self, features: pd.DataFrame, date: datetime, 
                                        required_factors: List[str]) -> pd.DataFrame:
        """
        Extract factor values from features DataFrame for a specific date.
        
        Handles different feature DataFrame structures:
        - MultiIndex (date, symbol)
        - DatetimeIndex
        - Regular index with date column
        
        Args:
            features: Features DataFrame with factor columns
            date: Date to extract factors for
            required_factors: List of factor column names to extract
            
        Returns:
            DataFrame with single row containing factor values (shape: 1 x n_factors)
        """
        if features.empty:
            return pd.DataFrame()
        
        # Ensure date is datetime
        if not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)
        
        # Handle MultiIndex (date, symbol)
        if isinstance(features.index, pd.MultiIndex):
            if 'date' in features.index.names:
                try:
                    # Extract data for this date
                    date_features = features.xs(date, level='date')
                    # Take first row (all symbols should have same factor values for a given date)
                    if len(date_features) > 0:
                        first_row = date_features.iloc[[0]]
                        factor_values = first_row[required_factors]
                        return factor_values
                except KeyError:
                    # Date not found, try to use last available date
                    available_dates = features.index.get_level_values('date').unique()
                    available_dates = pd.to_datetime(available_dates).sort_values()
                    valid_dates = available_dates[available_dates <= date]
                    if len(valid_dates) > 0:
                        last_date = valid_dates.max()
                        logger.debug(f"Date {date} not found, using {last_date}")
                        date_features = features.xs(last_date, level='date')
                        if len(date_features) > 0:
                            first_row = date_features.iloc[[0]]
                            factor_values = first_row[required_factors]
                            return factor_values
                    else:
                        # Use most recent date
                        if len(available_dates) > 0:
                            last_date = available_dates.max()
                            logger.warning(f"No dates <= {date}, using most recent: {last_date}")
                            date_features = features.xs(last_date, level='date')
                            if len(date_features) > 0:
                                first_row = date_features.iloc[[0]]
                                factor_values = first_row[required_factors]
                                return factor_values
        
        # Handle DatetimeIndex
        elif isinstance(features.index, pd.DatetimeIndex):
            if date in features.index:
                date_features = features.loc[[date]]
                if len(date_features) > 0:
                    factor_values = date_features[required_factors].iloc[[0]]
                    return factor_values
            else:
                # Find closest date
                available_dates = features.index[features.index <= date]
                if len(available_dates) > 0:
                    closest_date = available_dates.max()
                    logger.debug(f"Date {date} not found, using {closest_date}")
                    date_features = features.loc[[closest_date]]
                    if len(date_features) > 0:
                        factor_values = date_features[required_factors].iloc[[0]]
                        return factor_values
                else:
                    # Use most recent date
                    if len(features.index) > 0:
                        closest_date = features.index.max()
                        logger.warning(f"No dates <= {date}, using most recent: {closest_date}")
                        date_features = features.loc[[closest_date]]
                        if len(date_features) > 0:
                            factor_values = date_features[required_factors].iloc[[0]]
                            return factor_values
        
        # Fallback: use first row if available
        if len(features) > 0:
            logger.warning(f"Could not extract factors for date {date}, using first row")
            first_row = features.iloc[[0]]
            if all(col in first_row.columns for col in required_factors):
                factor_values = first_row[required_factors]
                return factor_values
        
        logger.error(f"Failed to extract factor values for date {date}")
        return pd.DataFrame()
    
    def _apply_expected_return_significance_filter(self, expected_returns: Dict[str, float],
                                                   config: Dict[str, Any],
                                                   current_date: datetime,
                                                   pipeline_data: Dict[str, Any],
                                                   required_factors: List[str],
                                                   lookback_days: int) -> Dict[str, float]:
        """
        Apply significance filter to expected returns based on alpha t-statistics.
        
        For expected returns, we still filter based on alpha significance since
        the alpha component is what we're testing for statistical significance.
        
        Args:
            expected_returns: Dictionary of {symbol: expected_return_value}
            config: Configuration dict (same as alpha_significance)
            current_date: Current date for rolling mode
            pipeline_data: Pipeline data with price_data and factor_data
            required_factors: List of factor columns to use
            lookback_days: Lookback window for rolling computation
        
        Returns:
            Filtered expected returns dict
        """
        # Get alphas from model for t-stat computation
        current_model = self.model_predictor.get_current_model()
        if not hasattr(current_model, 'get_symbol_alphas'):
            logger.warning("Cannot compute t-stats for expected returns without alpha access")
            return expected_returns
        
        alphas = current_model.get_symbol_alphas()
        if not alphas:
            logger.warning("No alphas available for t-stat computation")
            return expected_returns
        
        # Apply rolling alpha filter to get t-stats, then apply same shrinkage to expected returns
        filtered_alphas = self._apply_rolling_alpha_filter(
            alphas.copy(),
            config,
            current_date,
            pipeline_data,
            required_factors,
            lookback_days
        )
        
        # Compute shrinkage factors based on alpha t-stats
        # We'll need to get t-stats from cache or recompute
        tstat_dict = self._tstats_cache.get(current_date, {})
        if not tstat_dict:
            logger.warning(f"No t-stats cached for {current_date}, skipping filter")
            return expected_returns
        
        threshold = float(config.get('t_threshold', 2.0))
        method = config.get('method', 'hard_threshold')
        
        # Apply same shrinkage to expected returns
        filtered_returns = expected_returns.copy()
        for symbol in list(filtered_returns.keys()):
            if symbol not in tstat_dict:
                continue
            
            t_stat = tstat_dict[symbol]
            if pd.isna(t_stat):
                filtered_returns[symbol] = 0.0
                continue
            
            # Apply shrinkage factor
            factor = self._shrinkage_factor(float(t_stat), threshold, method)
            if factor < 1.0:
                filtered_returns[symbol] *= factor
                if factor == 0.0:
                    filtered_returns[symbol] = 0.0
        
        return filtered_returns

    def generate_signals(self, pipeline_data, start_date, end_date):
        """Generate signals with pipeline data context"""
        
        # ✅ 保存pipeline_data以便_get_predictions访问
        self._current_pipeline_data = pipeline_data
        
        # 调用父类方法
        return super().generate_signals(pipeline_data, start_date, end_date)

    def _get_predictions(self, features, price_data, start_date, end_date):
        """
        Generate predictions using either alpha (intercept) or expected return (alpha + beta @ factors).
        
        Supports configurable signal source via parameters.signal_source:
        - 'alpha': Uses only the intercept term (original behavior)
        - 'expected_return': Uses full expected return E[R] = α + β @ factors (default)
        
        Supports rolling t-stats mode: computes t-stats per date using historical data to avoid look-ahead bias.
        """
        # Get signal source configuration (default to 'expected_return' per requirement 2b)
        signal_source = self.parameters.get('signal_source', 'expected_return')
        logger.info(f"[{self.name}] Using signal source: {signal_source}")
        
        current_model = self.model_predictor.get_current_model()
        
        # Get pipeline_data if available (stored in generate_signals)
        pipeline_data = getattr(self, '_current_pipeline_data', None)
        
        # Create date range DataFrame
        date_range = pd.date_range(start_date, end_date, freq='D')
        predictions_df = pd.DataFrame(index=date_range)
        
        # Get signal transformation method
        signal_method = self.parameters.get('signal_method', 'raw')
        
        # Get alpha significance config
        alpha_config = self.parameters.get('alpha_significance', {})
        rolling_tstats = alpha_config.get('rolling_tstats', False)
        
        if signal_source == 'alpha':
            # Alpha mode: use intercept term only (original behavior)
            return self._get_predictions_from_alpha(
                current_model, features, price_data, date_range, predictions_df,
                alpha_config, rolling_tstats, pipeline_data, signal_method
            )
        elif signal_source == 'expected_return':
            # Expected return mode: use full E[R] = α + β @ factors
            return self._get_predictions_from_expected_return(
                current_model, features, price_data, date_range, predictions_df,
                alpha_config, rolling_tstats, pipeline_data, signal_method
            )
        else:
            logger.warning(f"Unknown signal_source: {signal_source}, defaulting to 'expected_return'")
            return self._get_predictions_from_expected_return(
                current_model, features, price_data, date_range, predictions_df,
                alpha_config, rolling_tstats, pipeline_data, signal_method
            )
    
    def _get_predictions_from_alpha(self, current_model, features, price_data, date_range,
                                    predictions_df, alpha_config, rolling_tstats, pipeline_data, signal_method):
        """Generate predictions using alpha (intercept) only."""
        logger.info(f"[{self.name}] Using alpha (intercept) as signal source")
        
        if not hasattr(current_model, 'get_symbol_alphas'):
            logger.error("当前模型不支持 get_symbol_alphas 方法，无法获取 Alpha。")
            return pd.DataFrame()

        # Get all symbol alphas
        alphas = current_model.get_symbol_alphas()
        if not alphas:
            logger.error("未能从模型中获取任何 Alpha 值。")
            return pd.DataFrame()

        if alpha_config.get('enabled', False) and rolling_tstats and pipeline_data:
            # Rolling mode: compute t-stats per date
            logger.info("使用rolling t-stats模式：为每个日期计算历史t-stats")
            
            # Determine required factors based on model type
            required_factors = self._determine_required_factors(current_model)
            lookback_days = alpha_config.get('lookback_days', self.lookback_days)
            
            # Compute rolling t-stats for each date
            for date in date_range:
                # Get filtered alphas for this date
                filtered_alphas = self._apply_alpha_significance_filter(
                    alphas.copy(), 
                    alpha_config, 
                    current_date=date,
                    pipeline_data=pipeline_data,
                    required_factors=required_factors,
                    lookback_days=lookback_days
                )
                
                # Apply signal transformation
                transformed_signals = self._transform_alpha_to_signals(filtered_alphas, signal_method)
                
                # Store transformed signals for this date
                for symbol, signal_value in transformed_signals.items():
                    if symbol not in predictions_df.columns:
                        predictions_df[symbol] = 0.0
                    predictions_df.loc[date, symbol] = signal_value
        else:
            # CSV mode (backward compatible): filter once, apply to all dates
            if alpha_config.get('enabled', False):
                alphas = self._apply_alpha_significance_filter(alphas, alpha_config)
            
            # Apply signal transformation
            alpha_signals = self._transform_alpha_to_signals(alphas, signal_method)
            
            for symbol, signal_value in alpha_signals.items():
                predictions_df[symbol] = signal_value
            
        # Ensure DataFrame contains correct symbols
        symbols_in_data = list(price_data.keys())
        predictions_df = predictions_df.reindex(columns=symbols_in_data).fillna(0.0)

        logger.info(f"成功为 {len(predictions_df.columns)} 只股票生成了基于 Alpha 的信号 (method: {signal_method})。")
        return predictions_df
    
    def _get_predictions_from_expected_return(self, current_model, features, price_data, date_range,
                                              predictions_df, alpha_config, rolling_tstats, pipeline_data, signal_method):
        """Generate predictions using expected return E[R] = α + β @ factors."""
        logger.info(f"[{self.name}] Using expected return (alpha + beta @ factors) as signal source")
        
        symbols = list(price_data.keys())
        required_factors = self._expected_factor_columns()
        
        # Verify features contain required factors
        missing_factors = set(required_factors) - set(features.columns)
        if missing_factors:
            logger.error(f"[{self.name}] Missing required factors in features: {missing_factors}")
            return pd.DataFrame()
        
        if alpha_config.get('enabled', False) and rolling_tstats and pipeline_data:
            # Rolling mode: compute expected returns per date with t-stats filtering
            logger.info("使用rolling t-stats模式：为每个日期计算expected return和历史t-stats")
            
            lookback_days = alpha_config.get('lookback_days', self.lookback_days)
            
            for date in date_range:
                # Extract factor values for current date
                factor_values_df = self._extract_factor_values_for_date(features, date, required_factors)
                if factor_values_df.empty:
                    logger.warning(f"No factor values found for date {date}, skipping")
                    continue
                
                # Get expected returns using model.predict()
                expected_returns = self.model_predictor.predict(
                    features=factor_values_df,
                    symbols=symbols,
                    date=date
                )
                
                # Convert to dict for filtering
                expected_returns_dict = expected_returns.to_dict()
                
                # Apply significance filter if enabled
                if alpha_config.get('enabled', False):
                    filtered_returns = self._apply_expected_return_significance_filter(
                        expected_returns_dict.copy(),
                        alpha_config,
                        current_date=date,
                        pipeline_data=pipeline_data,
                        required_factors=required_factors,
                        lookback_days=lookback_days
                    )
                else:
                    filtered_returns = expected_returns_dict
                
                # Apply signal transformation
                transformed_signals = self._transform_alpha_to_signals(filtered_returns, signal_method)
                
                # Store transformed signals for this date
                for symbol, signal_value in transformed_signals.items():
                    if symbol not in predictions_df.columns:
                        predictions_df[symbol] = 0.0
                    predictions_df.loc[date, symbol] = signal_value
        else:
            # Non-rolling mode: compute expected returns once per date (or use first date's factors)
            logger.info("使用非rolling模式：为每个日期计算expected return")
            
            for date in date_range:
                # Extract factor values for current date
                factor_values_df = self._extract_factor_values_for_date(features, date, required_factors)
                if factor_values_df.empty:
                    logger.warning(f"No factor values found for date {date}, skipping")
                    continue
                
                # Get expected returns using model.predict()
                expected_returns = self.model_predictor.predict(
                    features=factor_values_df,
                    symbols=symbols,
                    date=date
                )
                
                # Convert to dict
                expected_returns_dict = expected_returns.to_dict()
                
                # Apply significance filter if enabled (CSV mode)
                if alpha_config.get('enabled', False):
                    # For expected returns, we can still use alpha filter logic
                    # since we're filtering based on alpha significance
                    filtered_returns = self._apply_alpha_significance_filter(
                        expected_returns_dict, alpha_config
                    )
                else:
                    filtered_returns = expected_returns_dict
                
                # Apply signal transformation
                transformed_signals = self._transform_alpha_to_signals(filtered_returns, signal_method)
                
                # Store transformed signals for this date
                for symbol, signal_value in transformed_signals.items():
                    if symbol not in predictions_df.columns:
                        predictions_df[symbol] = 0.0
                    predictions_df.loc[date, symbol] = signal_value
        
        # Ensure DataFrame contains correct symbols
        symbols_in_data = list(price_data.keys())
        predictions_df = predictions_df.reindex(columns=symbols_in_data).fillna(0.0)

        logger.info(f"成功为 {len(predictions_df.columns)} 只股票生成了基于 Expected Return 的信号 (method: {signal_method})。")
        return predictions_df

    def _transform_alpha_to_signals(self, alphas: Dict[str, float], method: str = 'raw') -> Dict[str, float]:
        """
        Transform alpha values to trading signals using different methods.
        
        Args:
            alphas: Dictionary of symbol -> alpha values
            method: Transformation method ('raw', 'rank', 'zscore')
                - 'raw': Use alpha values directly (original behavior)
                - 'rank': Convert to ranks (0 to 1, higher alpha = higher rank)
                - 'zscore': Normalize using Z-score (mean=0, std=1)
        
        Returns:
            Dictionary of symbol -> signal values
        """
        if not alphas:
            return {}
        
        alpha_values = np.array(list(alphas.values()))
        alpha_symbols = list(alphas.keys())
        
        if method == 'raw':
            # Original behavior: use alpha values directly
            return alphas
        
        elif method == 'rank':
            # Rank-based: Convert to percentiles (0 to 1)
            # Higher alpha = higher rank = higher signal
            try:
                from scipy.stats import rankdata
                ranks = rankdata(alpha_values, method='average')  # Average rank for ties
            except ImportError:
                # Fallback: manual ranking if scipy not available
                sorted_indices = np.argsort(alpha_values)
                ranks = np.zeros_like(alpha_values)
                for rank, idx in enumerate(sorted_indices, 1):
                    ranks[idx] = rank
            # Normalize to [0, 1]
            normalized_ranks = (ranks - 1) / (len(ranks) - 1) if len(ranks) > 1 else ranks
            signals = {symbol: float(rank) for symbol, rank in zip(alpha_symbols, normalized_ranks)}
            logger.info(f"Applied rank-based transformation: min={min(signals.values()):.4f}, max={max(signals.values()):.4f}")
            return signals
        
        elif method == 'zscore':
            # Z-score normalization: (x - mean) / std
            mean_alpha = np.mean(alpha_values)
            std_alpha = np.std(alpha_values)
            
            if std_alpha == 0:
                # All alphas are the same, return zeros
                logger.warning("All alphas are identical, Z-score normalization returns zeros")
                return {symbol: 0.0 for symbol in alpha_symbols}
            
            z_scores = (alpha_values - mean_alpha) / std_alpha
            # Optional: Apply sigmoid to map to [0, 1] range for better signal strength
            # Using tanh to map to [-1, 1], then shift to [0, 1]
            signals = {symbol: float((np.tanh(z) + 1) / 2) for symbol, z in zip(alpha_symbols, z_scores)}
            logger.info(f"Applied Z-score transformation: min={min(signals.values()):.4f}, max={max(signals.values()):.4f}")
            return signals
        
        else:
            logger.warning(f"Unknown signal method: {method}, using raw alphas")
            return alphas

    def _apply_alpha_significance_filter(self, alphas: Dict[str, float], config: Dict[str, Any], 
                                         current_date: Optional[datetime] = None,
                                         pipeline_data: Optional[Dict[str, Any]] = None,
                                         required_factors: Optional[List[str]] = None,
                                         lookback_days: Optional[int] = None) -> Dict[str, float]:
        """
        Apply significance filter to alphas based on t-statistics.
        
        Filters out or shrinks statistically insignificant alphas to prevent
        MVO from over-weighting stocks with noisy alpha estimates.
        
        Supports two modes:
        1. CSV mode (backward compatible): Reads t-stats from static CSV file
        2. Rolling mode: Computes t-stats on-the-fly using historical data up to current_date
        
        Args:
            alphas: Dictionary of {symbol: alpha_value}
            config: Configuration dict with keys:
                - enabled: bool (checked by caller)
                - t_threshold: float (default 2.0)
                - method: str ('hard_threshold', 'linear_shrinkage', 'sigmoid_shrinkage')
                - tstats_path: str (path to CSV file, used in CSV mode)
                - rolling_tstats: bool (default False, enable rolling mode)
            current_date: Current date for rolling mode (only used if rolling_tstats=True)
            pipeline_data: Pipeline data containing price_data and factor_data (only used if rolling_tstats=True)
            required_factors: List of factor columns to use (only used if rolling_tstats=True)
            lookback_days: Lookback window for rolling computation (only used if rolling_tstats=True)
        
        Returns:
            Filtered alphas dict (modified in-place for performance, but returns for clarity)
        """
        rolling_tstats = config.get('rolling_tstats', False)
        
        # Rolling mode: compute t-stats on-the-fly
        if rolling_tstats and current_date is not None and pipeline_data is not None:
            return self._apply_rolling_alpha_filter(
                alphas, config, current_date, pipeline_data, required_factors, lookback_days
            )
        
        # CSV mode (backward compatible): read from static CSV
        path = config.get('tstats_path', './alpha_tstats.csv')
        # Resolve path: if relative, make it relative to project root (where run_experiment is executed)
        if not os.path.isabs(path):
            # Try current directory first (for compatibility)
            if os.path.exists(path):
                pass  # Use as-is
            else:
                # Try project root (common case)
                project_root = Path(__file__).parent.parent.parent.parent
                root_path = project_root / path
                if root_path.exists():
                    path = str(root_path)
                else:
                    logger.warning(f"T-stat file not found at {path} or {root_path}, will try to load anyway")
        
        threshold = float(config.get('t_threshold', 2.0))
        method = config.get('method', 'hard_threshold')
        
        # Store before state for logging
        alphas_before = alphas.copy()
        mean_before = np.mean(list(alphas_before.values())) if alphas_before else 0.0
        std_before = np.std(list(alphas_before.values())) if alphas_before else 0.0
        nz_before = sum(1 for v in alphas_before.values() if v != 0.0)
        
        try:
            tstat_df = pd.read_csv(path)
            
            # Validate CSV format
            if 'symbol' not in tstat_df.columns or 't_alpha' not in tstat_df.columns:
                logger.warning(
                    f"Invalid t-stat CSV format: missing required columns. "
                    f"Expected: symbol, t_alpha. Found: {list(tstat_df.columns)}"
                )
                return alphas
            
            # Build lookup dict for O(1) access
            tstat_dict = tstat_df.set_index('symbol')['t_alpha'].to_dict()
            
            n_total = len(alphas)
            n_filtered = 0
            n_missing = 0
            
            for symbol in list(alphas.keys()):
                if symbol not in tstat_dict:
                    n_missing += 1
                    logger.debug(f"Symbol {symbol} not in t-stat CSV, keeping original alpha")
                    continue
                
                t_stat = tstat_dict[symbol]
                
                # Handle NaN
                if pd.isna(t_stat):
                    logger.debug(f"Symbol {symbol} has NaN t-stat, setting alpha=0")
                    alphas[symbol] = 0.0
                    n_filtered += 1
                    continue
                
                # Apply filtering/shrinkage
                factor = self._shrinkage_factor(float(t_stat), threshold, method)
                if factor < 1.0:
                    alphas[symbol] *= factor
                    if factor == 0.0:
                        n_filtered += 1
            
            # Log detailed metrics
            mean_after = np.mean(list(alphas.values())) if alphas else 0.0
            std_after = np.std(list(alphas.values())) if alphas else 0.0
            nz_after = sum(1 for v in alphas.values() if v != 0.0)
            
            logger.info(
                f"Alpha significance filter applied (CSV mode): "
                f"method={method}, threshold={threshold}, "
                f"zeroed/shrunk={n_filtered}/{n_total}, "
                f"missing_in_csv={n_missing}"
            )
            logger.info(
                f"Alpha distribution: "
                f"mean={mean_before:.6f}→{mean_after:.6f}, "
                f"std={std_before:.6f}→{std_after:.6f}, "
                f"non-zero={nz_before}→{nz_after}"
            )
            
        except FileNotFoundError:
            logger.warning(f"T-stat file not found: {path}. Skipping alpha significance filter.")
        except Exception as e:
            logger.error(f"Alpha significance filter failed: {e}", exc_info=True)
        
        return alphas
    
    def _apply_rolling_alpha_filter(self, alphas: Dict[str, float], config: Dict[str, Any],
                                     current_date: datetime, pipeline_data: Dict[str, Any],
                                     required_factors: Optional[List[str]] = None,
                                     lookback_days: Optional[int] = None) -> Dict[str, float]:
        """
        Apply rolling alpha significance filter using historical data up to current_date.
        
        Args:
            alphas: Dictionary of {symbol: alpha_value}
            config: Configuration dict
            current_date: Date to compute t-stats for (only uses data <= current_date)
            pipeline_data: Pipeline data with price_data and factor_data
            required_factors: List of factor columns to use
            lookback_days: Lookback window in trading days
        
        Returns:
            Filtered alphas dict
        """
        # Check cache first
        if current_date in self._tstats_cache:
            tstat_dict = self._tstats_cache[current_date]
        else:
            # Extract data from pipeline_data
            price_data = pipeline_data.get('price_data', {})
            factor_data = pipeline_data.get('factor_data')
            
            if not price_data or factor_data is None or factor_data.empty:
                logger.warning(f"Rolling t-stats: insufficient data for date {current_date}, skipping filter")
                return alphas
            
            # Default values
            if required_factors is None:
                required_factors = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
            if lookback_days is None:
                lookback_days = self.lookback_days
            
            # Convert current_date to datetime if needed
            if not isinstance(current_date, pd.Timestamp):
                current_date = pd.to_datetime(current_date)
            
            # Ensure factor_data index is datetime
            if not isinstance(factor_data.index, pd.DatetimeIndex):
                factor_data = factor_data.copy()
                factor_data.index = pd.to_datetime(factor_data.index)
            
            # Filter factor data up to current_date
            factor_historical = factor_data[factor_data.index <= current_date].copy()
            
            if factor_historical.empty:
                logger.warning(f"Rolling t-stats: no factor data up to {current_date}, skipping filter")
                return alphas
            
            # Compute t-stats for each symbol
            tstat_dict = {}
            threshold = float(config.get('t_threshold', 2.0))
            method = config.get('method', 'hard_threshold')
            
            for symbol in alphas.keys():
                if symbol not in price_data:
                    continue
                
                # Get price data for this symbol
                symbol_price_data = price_data[symbol]
                if 'Close' not in symbol_price_data.columns:
                    continue
                
                # Ensure price data index is datetime
                if not isinstance(symbol_price_data.index, pd.DatetimeIndex):
                    symbol_price_data = symbol_price_data.copy()
                    symbol_price_data.index = pd.to_datetime(symbol_price_data.index)
                
                # Filter price data up to current_date
                price_historical = symbol_price_data[symbol_price_data.index <= current_date].copy()
                
                if price_historical.empty:
                    continue
                
                # Calculate returns
                returns = price_historical['Close'].pct_change().dropna()
                
                # Use last lookback_days
                if len(returns) < lookback_days:
                    # Not enough data, skip or use all available
                    if len(returns) < 30:
                        continue
                    returns_window = returns.copy()
                else:
                    returns_window = returns.tail(lookback_days).copy()
                
                # Align factor data to returns dates
                returns_start = returns_window.index.min()
                returns_end = returns_window.index.max()
                
                # Get factor data for the returns date range
                factor_mask = (factor_historical.index >= returns_start) & (factor_historical.index <= returns_end)
                factor_window = factor_historical.loc[factor_mask].copy()
                
                if factor_window.empty:
                    continue
                
                # Handle frequency mismatch (e.g., monthly factors with daily returns)
                if len(factor_window) < len(returns_window) * 0.5:
                    # Forward fill to daily frequency
                    try:
                        factor_window = factor_window.reindex(returns_window.index, method='ffill')
                    except TypeError:
                        # For pandas 2.x
                        factor_window = factor_window.reindex(returns_window.index).ffill()
                    factor_window = factor_window.dropna()
                    returns_window = returns_window.loc[factor_window.index]
                else:
                    # Align by intersection
                    common_dates = returns_window.index.intersection(factor_window.index)
                    if len(common_dates) < 30:
                        continue
                    returns_window = returns_window.loc[common_dates]
                    factor_window = factor_window.loc[common_dates]
                
                # Calculate excess returns (stock return - risk-free rate)
                if 'RF' in factor_window.columns:
                    risk_free_rate = factor_window['RF'].loc[returns_window.index]
                    returns_window = returns_window - risk_free_rate
                
                # Ensure required factors are present
                if not all(col in factor_window.columns for col in required_factors):
                    continue
                
                # Compute t-stat using utility function
                stats = compute_alpha_tstat(returns_window, factor_window, required_factors)
                tstat_dict[symbol] = stats['t_stat']
            
            # Cache the results
            self._tstats_cache[current_date] = tstat_dict
        
        # Apply filtering/shrinkage using computed t-stats
        threshold = float(config.get('t_threshold', 2.0))
        method = config.get('method', 'hard_threshold')
        
        n_total = len(alphas)
        n_filtered = 0
        n_missing = 0
        
        for symbol in list(alphas.keys()):
            if symbol not in tstat_dict:
                n_missing += 1
                logger.debug(f"Symbol {symbol} not in rolling t-stats for {current_date}, keeping original alpha")
                continue
            
            t_stat = tstat_dict[symbol]
            
            # Handle NaN
            if pd.isna(t_stat):
                logger.debug(f"Symbol {symbol} has NaN t-stat for {current_date}, setting alpha=0")
                alphas[symbol] = 0.0
                n_filtered += 1
                continue
            
            # Apply filtering/shrinkage
            factor = self._shrinkage_factor(float(t_stat), threshold, method)
            if factor < 1.0:
                alphas[symbol] *= factor
                if factor == 0.0:
                    n_filtered += 1
        
        # Log metrics (only for first date to avoid spam)
        if len(self._tstats_cache) == 1:
            mean_after = np.mean(list(alphas.values())) if alphas else 0.0
            std_after = np.std(list(alphas.values())) if alphas else 0.0
            nz_after = sum(1 for v in alphas.values() if v != 0.0)
            
            logger.info(
                f"Rolling alpha significance filter applied for {current_date}: "
                f"method={method}, threshold={threshold}, "
                f"zeroed/shrunk={n_filtered}/{n_total}, "
                f"missing_tstats={n_missing}"
            )
        
        return alphas
    
    def _shrinkage_factor(self, t_stat: float, threshold: float, method: str) -> float:
        """
        Calculate shrinkage factor based on t-statistic.
        
        Args:
            t_stat: Alpha's t-statistic
            threshold: Significance threshold (typically 2.0)
            method: Shrinkage method ('hard_threshold', 'linear_shrinkage', 'sigmoid_shrinkage')
        
        Returns:
            Shrinkage factor (0.0 to 1.0)
        """
        abs_t = abs(t_stat)
        
        if method == 'hard_threshold':
            return 1.0 if abs_t >= threshold else 0.0
        
        elif method == 'linear_shrinkage':
            # Linear decay: t=0 → 0%, t=threshold → 100%
            return min(1.0, abs_t / threshold)
        
        elif method == 'sigmoid_shrinkage':
            # Smooth sigmoid transition around threshold
            return 1.0 / (1.0 + np.exp(-2.0 * (abs_t - threshold)))
        
        else:
            logger.warning(f"Unknown shrinkage method: {method}, using hard_threshold")
            return 1.0 if abs_t >= threshold else 0.0

    def get_info(self) -> Dict:
        """Get Fama-French strategy information."""
        info = super().get_info()
        info.update({
            'lookback_days': self.lookback_days,
            'risk_free_rate': self.risk_free_rate,
            'strategy_type': 'fama_french_5',
            'model_complexity': 'low',
            'model_expected': 'FF5RegressionModel',
            'model_status': '✅ ALREADY IMPLEMENTED',
            'factor_columns': self._expected_factor_columns(),
            'data_flow': 'FactorDataProvider → FF5RegressionModel',
            'prediction_optimization': 'FF5 batch prediction with factor sharing'
        })
        return info


# ✅ MODEL STATUS: IMPLEMENTED
# -----------------------------
# File: models/implementations/ff5_model.py
# Class: FF5RegressionModel(BaseModel)
#
# This model ALREADY EXISTS and implements:
# - fit(X, y): Estimates factor betas via linear regression
# - predict(X): Calculates expected returns from factor exposures
# - Uses sklearn.linear_model.LinearRegression or Ridge
#
# Model Specification:
#   R_stock = α + β_MKT*R_MKT + β_SMB*R_SMB + β_HML*R_HML + 
#             β_RMW*R_RMW + β_CMA*R_CMA + ε
#
# Expected Features (X):
#   - MKT: Market excess return
#   - SMB: Size factor
#   - HML: Value factor
#   - RMW: Profitability factor
#   - CMA: Investment factor
#
# Target (y):
#   - Stock excess returns (R_stock - R_f)
#
# The model is already registered and can be loaded via ModelPredictor!

