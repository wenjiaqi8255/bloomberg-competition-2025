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
    FactorDataProvider → FF5RegressionModel → PositionSizer

Key Difference from other strategies:
    - FF5 uses FACTOR DATA directly from FF5DataProvider
    - Does NOT use technical indicators or traditional feature engineering
    - Model expects columns: ['MKT', 'SMB', 'HML', 'RMW', 'CMA']

Data Flow:
    1. FF5DataProvider provides factor data (MKT, SMB, HML, RMW, CMA, RF)
    2. For each stock, we create factor features based on historical returns
    3. FF5RegressionModel estimates factor betas and predicts expected returns
    4. PositionSizer manages risk and position sizes

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
from typing import Dict, Optional
import pandas as pd
import numpy as np

from .base_strategy import BaseStrategy
from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..models.serving.predictor import ModelPredictor
from ..utils.position_sizer import PositionSizer

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
        
        # Create position sizer
        position_sizer = PositionSizer(
            volatility_target=0.15,
            max_position_weight=0.10
        )
        
        # Create strategy
        strategy = FamaFrench5Strategy(
            name="FF5",
            feature_pipeline=feature_pipeline,
            model_predictor=model_predictor,
            position_sizer=position_sizer,
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
                 position_sizer: PositionSizer,
                 lookback_days: int = 252,
                 risk_free_rate: float = 0.02,
                 **kwargs):
        """
        Initialize Fama-French 5-factor strategy.
        
        Args:
            name: Strategy identifier
            feature_pipeline: Fitted pipeline (should compute factor proxies)
            model_predictor: Predictor with FF5RegressionModel loaded
            position_sizer: Position sizing component
            lookback_days: Lookback period for factor calculation
            risk_free_rate: Risk-free rate for excess return calculation
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            feature_pipeline=feature_pipeline,
            model_predictor=model_predictor,
            position_sizer=position_sizer,
            **kwargs
        )
        
        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate

        logger.info(f"Initialized FamaFrench5Strategy '{name}' with "
                   f"lookback={lookback_days}d, rf_rate={risk_free_rate}")

    def _compute_features(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Override feature computation for FF5 strategy.

        Instead of using technical indicators, we create factor exposure features
        based on historical returns and align them with FF5 factor data.

        Args:
            price_data: Dictionary mapping symbols to their price data

        Returns:
            DataFrame with factor features for each symbol
        """
        logger.info(f"[{self.name}] 🔄 Computing FF5 factor features")

        if not self.factor_data_provider:
            raise ValueError("FF5 strategy requires factor_data_provider")

        if not price_data:
            raise ValueError("No price data provided")

        # Get the latest date from price data
        all_dates = []
        for symbol, data in price_data.items():
            if data is not None and len(data) > 0:
                all_dates.extend(data.index.tolist())

        if not all_dates:
            raise ValueError("No valid dates found in price data")

        latest_date = max(all_dates)

        # Get factor data for the relevant period
        start_date = latest_date - pd.Timedelta(days=self.lookback_days)
        factor_data = self.factor_data_provider.get_factor_returns(start_date, latest_date)

        # Filter to only expected factor columns
        expected_factors = self._expected_factor_columns()
        if factor_data is not None and not factor_data.empty:
            factor_data = factor_data[[col for col in expected_factors if col in factor_data.columns]]

        logger.info(f"[{self.name}] Retrieved factor data: {factor_data.shape}")
        logger.info(f"[{self.name}] Factor columns: {list(factor_data.columns)}")

        all_features = []
        symbol_list = []

        for symbol, symbol_data in price_data.items():
            if symbol_data is None or len(symbol_data) < self.lookback_days:
                logger.warning(f"[{self.name}] Insufficient data for {symbol}")
                continue

            try:
                # Calculate stock returns over the lookback period
                aligned_data = symbol_data.reindex(factor_data.index, method='ffill')
                stock_returns = aligned_data['Close'].pct_change().dropna()

                if len(stock_returns) < 60:  # Need enough data points
                    logger.warning(f"[{self.name}] Not enough return data for {symbol}")
                    continue

                # Align stock returns with factor data
                aligned_returns, aligned_factors = stock_returns.align(factor_data[self._expected_factor_columns()], join='inner')

                if len(aligned_returns) < 60:
                    logger.warning(f"[{self.name}] Not enough aligned data for {symbol}")
                    continue

                # Create features DataFrame for this symbol
                features_df = pd.DataFrame(index=[symbol])

                # Add current factor values (most recent available)
                if len(aligned_factors) > 0:
                    latest_factors = aligned_factors.iloc[-1]
                    for factor in self._expected_factor_columns():
                        features_df[factor] = latest_factors[factor]
                else:
                    # Use zeros if no factor data available
                    for factor in self._expected_factor_columns():
                        features_df[factor] = 0.0

                # Add additional features based on historical data
                # Historical beta estimation (rolling window)
                if len(aligned_returns) >= 60:
                    # Calculate rolling betas for each factor
                    window_size = min(60, len(aligned_returns))

                    for factor in self._expected_factor_columns():
                        if factor in aligned_factors.columns:
                            # Simple rolling beta estimation
                            factor_returns = aligned_factors[factor]
                            stock_excess_returns = aligned_returns - self.risk_free_rate/252

                            # Calculate beta
                            if len(factor_returns) > 1 and factor_returns.std() > 0:
                                beta = (stock_excess_returns * factor_returns).rolling(window_size).cov() / factor_returns.rolling(window_size).var()
                                features_df[f'{factor}_beta'] = beta.iloc[-1] if not beta.empty else 0.0
                            else:
                                features_df[f'{factor}_beta'] = 0.0
                        else:
                            features_df[f'{factor}_beta'] = 0.0

                # Add stock-specific risk metrics
                if len(aligned_returns) >= 30:
                    # Volatility
                    volatility = aligned_returns.rolling(min(30, len(aligned_returns))).std().iloc[-1] if len(aligned_returns) > 0 else 0.0
                    features_df['stock_volatility'] = volatility

                    # Momentum
                    momentum = aligned_returns.sum()
                    features_df['momentum'] = momentum

                all_features.append(features_df)
                symbol_list.append(symbol)
                logger.debug(f"[{self.name}] Computed features for {symbol}: {list(features_df.columns)}")

            except Exception as e:
                logger.error(f"[{self.name}] Failed to compute features for {symbol}: {e}")
                continue

        if not all_features:
            raise ValueError("No features computed successfully")

        # Combine all features
        result = pd.concat(all_features, axis=0)
        logger.info(f"[{self.name}] ✅ Computed features for {len(result)} symbols")
        logger.info(f"[{self.name}] Feature columns: {list(result.columns)}")
        logger.info(f"[{self.name}] Feature sample: {result.iloc[-1].to_dict() if not result.empty else 'Empty'}")

        return result

    def _expected_factor_columns(self) -> list:
        """Get the expected factor columns for FF5 model."""
        return ['MKT', 'SMB', 'HML', 'RMW', 'CMA']

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
            'data_flow': 'FactorDataProvider → FF5RegressionModel → PositionSizer'
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

