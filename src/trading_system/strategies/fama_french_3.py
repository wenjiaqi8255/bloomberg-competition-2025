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
from typing import Dict, Any, Optional
import pandas as pd

from .base_strategy import BaseStrategy
from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..models.serving.predictor import ModelPredictor

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
        """
        current_model = self.model_predictor.get_current_model()
        if not hasattr(current_model, 'get_symbol_alphas'):
            logger.error("Current model does not support get_symbol_alphas")
            return pd.DataFrame()

        alphas = current_model.get_symbol_alphas()
        if not alphas:
            logger.error("No alphas returned from model")
            return pd.DataFrame()

        # Apply alpha significance filter if enabled
        alpha_config = self.parameters.get('alpha_significance', {})
        if alpha_config.get('enabled', False):
            alphas = self._apply_alpha_significance_filter(alphas, alpha_config)

        # 将 alpha 转换为一个 Series，这是我们的信号
        alpha_series = pd.Series(alphas)
        date_range = pd.date_range(start_date, end_date, freq='D')
        predictions_df = pd.DataFrame(index=date_range)

        for symbol, alpha_value in alpha_series.items():
            predictions_df[symbol] = alpha_value

        symbols_in_data = list(price_data.keys())
        predictions_df = predictions_df.reindex(columns=symbols_in_data).fillna(0.0)
        
        logger.info(f"成功为 {len(alpha_series)} 只股票生成了基于 Alpha 的信号。")
        return predictions_df

    def _apply_alpha_significance_filter(self, alphas: Dict[str, float], config: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply significance filter to alphas based on t-statistics.
        
        Filters out or shrinks statistically insignificant alphas to prevent
        MVO from over-weighting stocks with noisy alpha estimates.
        
        Args:
            alphas: Dictionary of {symbol: alpha_value}
            config: Configuration dict with keys:
                - enabled: bool (checked by caller)
                - t_threshold: float (default 2.0)
                - method: str ('hard_threshold', 'linear_shrinkage', 'sigmoid_shrinkage')
                - tstats_path: str (path to CSV file)
        
        Returns:
            Filtered alphas dict (modified in-place for performance, but returns for clarity)
        """
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
                f"Alpha significance filter applied: "
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



