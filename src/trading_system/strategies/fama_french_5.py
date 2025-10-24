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
from datetime import datetime
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..models.serving.predictor import ModelPredictor

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

    def generate_signals(self, pipeline_data, start_date, end_date):
        """Generate signals with pipeline data context"""
        
        # ✅ 保存pipeline_data以便_get_predictions访问
        self._current_pipeline_data = pipeline_data
        
        # 调用父类方法
        return super().generate_signals(pipeline_data, start_date, end_date)

    def _get_predictions(self, features, price_data, start_date, end_date):
        """
        FF5预测使用pipeline_data中的factor_data
        """
        if not hasattr(self, '_current_pipeline_data'):
            return super()._get_predictions(features, price_data, start_date, end_date)
        
        factor_data = self._current_pipeline_data.get('factor_data')
        if factor_data is None:
            logger.error("No factor_data in pipeline_data")
            return pd.DataFrame()
        
        current_model = self.model_predictor.get_current_model()
        symbols = list(price_data.keys())
        
        predictions_list = []
        
        # 对预测日期范围内的每一天
        for date in pd.date_range(start_date, end_date):
            if date not in factor_data.index:
                logger.warning(f"Date {date} not in factor_data")
                continue
            
            # 获取该日期的因子值
            date_factors = factor_data.loc[[date], ['MKT', 'SMB', 'HML', 'RMW', 'CMA']]
            
            # 预测
            preds = current_model.predict(date_factors, symbols=symbols)
            
            if isinstance(preds, np.ndarray):
                preds = pd.Series(preds, index=symbols, name=date)
            
            predictions_list.append(preds)
        
        if predictions_list:
            return pd.concat(predictions_list, axis=1).T
        else:
            return pd.DataFrame()

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

