"""
Fama-French 3-Factor Strategy - Unified Architecture

This strategy mirrors the FF5 strategy but uses only the 3 classic factors:
MKT, SMB, HML. It expects factor data prepared upstream and delegates model
training/prediction to the predictor (FF3RegressionModel).
"""

import logging
import os
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

from .base_strategy import BaseStrategy
from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..models.serving.predictor import ModelPredictor
from ..utils.alpha_stats import compute_alpha_tstat

logger = logging.getLogger(__name__)


class FamaFrench3Strategy(BaseStrategy):
    """Fama-French 3-factor strategy using the unified architecture."""

    def __init__(self,
                 name: str,
                 feature_pipeline: FeatureEngineeringPipeline,
                 model_predictor: ModelPredictor,
                 lookback_days: int = 252,
                 risk_free_rate: float = 0.02,
                 **kwargs):
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

        logger.info(f"Initialized FamaFrench3Strategy '{name}' with lookback={lookback_days}d, rf_rate={risk_free_rate}")

    def _expected_factor_columns(self) -> list:
        return ['MKT', 'SMB', 'HML']

    def _compute_features(self, pipeline_data: Dict[str, Any]) -> pd.DataFrame:
        logger.info(f"[{self.name}] Computing features from pipeline data (FF3)")
        price_data = pipeline_data.get('price_data', {})
        factor_data = pipeline_data.get('factor_data')

        if not price_data:
            logger.error(f"[{self.name}] No price_data in pipeline_data")
            return pd.DataFrame()

        if factor_data is None:
            logger.error(f"[{self.name}] No factor_data in pipeline_data - FF3 strategy requires factor data!")
            return pd.DataFrame()

        pipeline_input = {
            'price_data': price_data,
            'factor_data': factor_data
        }

        try:
            features = self.feature_pipeline.transform(pipeline_input)
            expected = self._expected_factor_columns()
            missing = set(expected) - set(features.columns)
            if missing:
                logger.error(f"[{self.name}] Missing FF3 factors: {missing}")
            return features
        except Exception as e:
            logger.error(f"[{self.name}] Error computing features via pipeline: {e}")
            return pd.DataFrame()

    def generate_signals(self, pipeline_data, start_date, end_date):
        self._current_pipeline_data = pipeline_data
        return super().generate_signals(pipeline_data, start_date, end_date)

    def _get_predictions(self, features, price_data, start_date, end_date):
        """
        For simplicity and stability, use model alphas as signals across days.
        
        支持rolling t-stats模式：在每个日期使用历史数据计算t-stats，避免look-ahead bias。
        """
        current_model = self.model_predictor.get_current_model()
        if not hasattr(current_model, 'get_symbol_alphas'):
            logger.error("Current model does not support get_symbol_alphas")
            return pd.DataFrame()

        alphas = current_model.get_symbol_alphas()
        if not alphas:
            logger.error("No alphas returned from model")
            return pd.DataFrame()

        # Check if rolling t-stats is enabled
        alpha_config = self.parameters.get('alpha_significance', {})
        rolling_tstats = alpha_config.get('rolling_tstats', False)
        
        # Get pipeline_data if available (stored in generate_signals)
        pipeline_data = getattr(self, '_current_pipeline_data', None)
        
        # 创建一个符合回测时间范围的 DataFrame
        date_range = pd.date_range(start_date, end_date, freq='D')
        predictions_df = pd.DataFrame(index=date_range)

        if alpha_config.get('enabled', False) and rolling_tstats and pipeline_data:
            # Rolling mode: compute t-stats per date
            logger.info("使用rolling t-stats模式：为每个日期计算历史t-stats (FF3)")
            
            # FF3 uses 3 factors
            required_factors = ['MKT', 'SMB', 'HML']
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
                
                # Store filtered alphas for this date
                for symbol, alpha_value in filtered_alphas.items():
                    if symbol not in predictions_df.columns:
                        predictions_df[symbol] = 0.0
                    predictions_df.loc[date, symbol] = alpha_value
        else:
            # CSV mode (backward compatible): filter once, apply to all dates
            if alpha_config.get('enabled', False):
                alphas = self._apply_alpha_significance_filter(alphas, alpha_config)
            
            # 将 alpha 转换为一个 Series，这是我们的信号
            alpha_series = pd.Series(alphas)
            
            for symbol, alpha_value in alpha_series.items():
                predictions_df[symbol] = alpha_value

        symbols_in_data = list(price_data.keys())
        predictions_df = predictions_df.reindex(columns=symbols_in_data).fillna(0.0)
        
        logger.info(f"成功为 {len(predictions_df.columns)} 只股票生成了基于 Alpha 的信号。")
        return predictions_df

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
                required_factors = ['MKT', 'SMB', 'HML']
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
        info = super().get_info()
        info.update({
            'lookback_days': self.lookback_days,
            'risk_free_rate': self.risk_free_rate,
            'strategy_type': 'fama_french_3',
            'model_expected': 'FF3RegressionModel',
            'factor_columns': self._expected_factor_columns(),
            'data_flow': 'FactorDataProvider → FF3RegressionModel'
        })
        return info



