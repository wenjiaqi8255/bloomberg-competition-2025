"""
Fama-French 3-Factor Strategy - Unified Architecture

This strategy mirrors the FF5 strategy but uses only the 3 classic factors:
MKT, SMB, HML. It expects factor data prepared upstream and delegates model
training/prediction to the predictor (FF3RegressionModel).
"""

import logging
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

        alpha_series = pd.Series(alphas)
        date_range = pd.date_range(start_date, end_date, freq='D')
        predictions_df = pd.DataFrame(index=date_range)

        for symbol, alpha_value in alpha_series.items():
            predictions_df[symbol] = alpha_value

        symbols_in_data = list(price_data.keys())
        predictions_df = predictions_df.reindex(columns=symbols_in_data).fillna(0.0)
        return predictions_df

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


