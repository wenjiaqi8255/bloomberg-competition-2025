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
from typing import Dict, Optional, List
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
                 lookback_days: int = 252,
                 risk_free_rate: float = 0.02,
                 **kwargs):
        """
        Initialize Fama-French 5-factor strategy.

        Args:
            name: Strategy identifier
            feature_pipeline: Fitted pipeline (computes technical indicators and merges factor data)
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
            **kwargs
        )

        self.lookback_days = lookback_days
        self.risk_free_rate = risk_free_rate

        logger.info(f"Initialized FamaFrench5Strategy '{name}' with "
                   f"lookback={lookback_days}d, rf_rate={risk_free_rate}")
        logger.info(f"[{name}] Following SOLID principles - strategy as pure delegate")

    def _compute_features(self, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Compute features for FF5 strategy following SOLID principles.

        KISS + DRY Principle: Strategy is a pure delegate that coordinates components
        without reimplementing existing functionality.

        The strategy trusts:
        - FeatureEngineeringPipeline to handle technical indicators and factor data merging
        - FF5RegressionModel to handle beta estimation and prediction
        - BaseStrategy framework to handle the signal generation workflow

        Args:
            price_data: Dictionary mapping symbols to their price data

        Returns:
            DataFrame with features computed by the FeatureEngineeringPipeline
        """
        logger.info(f"[{self.name}] 🔄 Computing features via FeatureEngineeringPipeline (delegate pattern)")

        # CRITICAL FIX: FF5 strategy needs both price_data AND factor_data!
        # The factor_data should be provided by the strategy runner or orchestrator
        pipeline_input = {'price_data': price_data}

        # Try to get factor data from various sources
        factor_data = None

        # Method 1: Check if strategy has access to factor_data_provider
        if hasattr(self, 'factor_data_provider') and self.factor_data_provider:
            logger.info(f"[{self.name}] 📊 Getting factor data from factor_data_provider")
            try:
                # Get date range from price data
                all_dates = []
                for symbol_data in price_data.values():
                    if not symbol_data.empty:
                        all_dates.extend(symbol_data.index.tolist())

                if all_dates:
                    start_date = min(all_dates)
                    end_date = max(all_dates)
                    factor_data = self.factor_data_provider.get_data(start_date, end_date)
                    logger.info(f"[{self.name}] ✅ Factor data retrieved: {factor_data.shape if factor_data is not None else None}")
                else:
                    logger.warning(f"[{self.name}] No dates found in price data")
            except Exception as e:
                logger.error(f"[{self.name}] Failed to get factor data: {e}")

        # Method 2: Check if factor_data is stored as instance attribute
        elif hasattr(self, 'factor_data') and self.factor_data is not None:
            logger.info(f"[{self.name}] 📊 Using stored factor data: {self.factor_data.shape}")
            factor_data = self.factor_data

        # Method 3: Check providers dictionary
        elif hasattr(self, 'providers') and self.providers and 'factor_data_provider' in self.providers:
            logger.info(f"[{self.name}] 📊 Getting factor data from providers dictionary")
            try:
                factor_provider = self.providers['factor_data_provider']
                # Get date range from price data
                all_dates = []
                for symbol_data in price_data.values():
                    if not symbol_data.empty:
                        all_dates.extend(symbol_data.index.tolist())

                if all_dates:
                    start_date = min(all_dates)
                    end_date = max(all_dates)
                    factor_data = factor_provider.get_data(start_date, end_date)
                    logger.info(f"[{self.name}] ✅ Factor data retrieved from providers: {factor_data.shape if factor_data is not None else None}")
            except Exception as e:
                logger.error(f"[{self.name}] Failed to get factor data from providers: {e}")

        # Add factor_data to pipeline input if available
        if factor_data is not None and not factor_data.empty:
            pipeline_input['factor_data'] = factor_data
            logger.info(f"[{self.name}] 📊 Added factor data to pipeline input: {list(factor_data.columns)}")
        else:
            logger.warning(f"[{self.name}] ⚠️ No factor data available - FF5 model will fail!")

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

    def _get_predictions(self,
                     features: pd.DataFrame,
                     price_data: Dict[str, pd.DataFrame],
                     start_date: datetime,
                     end_date: datetime) -> pd.DataFrame:
        """
        FF5-optimized batch prediction using factor sharing across stocks.

        This method leverages the fact that FF5 factors are the same for all stocks
        on a given date, enabling efficient batch prediction.

        Args:
            features: Computed features with FF5 factors
            price_data: Original price data
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with predictions (expected returns) indexed by date
        """
        try:
            logger.info(f"[{self.name}] 🔍 _get_predictions started (FF5 batch mode)")
            logger.info(f"[{self.name}] Features shape: {features.shape}, columns: {list(features.columns[:10])}...")
            logger.info(f"[{self.name}] Price data keys: {list(price_data.keys())}")
            logger.info(f"[{self.name}] Date range: {start_date} to {end_date}")
            logger.info(f"[{self.name}] Model predictor type: {type(self.model_predictor)}")

            # Create date range for predictions
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            logger.info(f"[{self.name}] Prediction dates count: {len(dates)}")

            # Get list of symbols to predict
            symbols = list(price_data.keys())
            logger.info(f"[{self.name}] Symbols to predict: {len(symbols)}")

            predictions_dict = {}

            # FF5-optimized single loop over dates
            for date in dates:
                try:
                    logger.debug(f"[{self.name}] Processing date {date}...")

                    # Extract FF5 factor values for this date (shared by all symbols)
                    date_factors = self._extract_date_factors(features, date)

                    if date_factors.empty:
                        logger.warning(f"[{self.name}] No FF5 factors found for {date}")
                        # Create zero predictions for all symbols
                        date_predictions = pd.Series([0.0] * len(symbols), index=symbols)
                    else:
                        logger.debug(f"[{self.name}] Extracted FF5 factors for {date}: shape={date_factors.shape}")

                        # Batch predict all symbols for this date
                        if hasattr(self.model_predictor, 'predict_batch'):
                            # Use new batch prediction if available
                            logger.debug(f"[{self.name}] Using FF5 batch prediction for {len(symbols)} symbols")
                            date_predictions = self.model_predictor.predict_batch(
                                factors=date_factors,
                                symbols=symbols,
                                date=date
                            )
                        else:
                            # Fallback to FF5 iterative prediction
                            logger.debug(f"[{self.name}] Using FF5 iterative prediction for {len(symbols)} symbols")
                            date_predictions = self._predict_iterative(date_factors, symbols, date)

                    # Store predictions for this date
                    predictions_dict[date] = date_predictions
                    logger.debug(f"[{self.name}] Generated FF5 predictions for {date}: {date_predictions.shape}")

                except Exception as e:
                    logger.error(f"[{self.name}] Failed to get FF5 predictions for {date}: {e}")
                    # Create zero predictions for this date
                    predictions_dict[date] = pd.Series([0.0] * len(symbols), index=symbols)

            # Convert dictionary to DataFrame format (dates as index, symbols as columns)
            if predictions_dict:
                predictions_df = pd.DataFrame(predictions_dict).T  # Transpose: rows=dates, cols=symbols
                logger.info(f"[{self.name}] ✅ Created FF5 DataFrame: shape={predictions_df.shape}")
                logger.info(f"[{self.name}] 📊 Columns (symbols): {list(predictions_df.columns)}")
                logger.info(f"[{self.name}] 📅 Index (dates): {predictions_df.index[0]} to {predictions_df.index[-1]}")
            else:
                predictions_df = pd.DataFrame(index=dates)

            # Log sample predictions to verify they're not all the same
            if not predictions_df.empty:
                logger.info(f"[{self.name}] 📈 Sample FF5 predictions:")
                logger.info(f"[{self.name}]   First date ({predictions_df.index[0]}): {predictions_df.iloc[0].to_dict()}")
                if len(predictions_df) > 1:
                    logger.info(f"[{self.name}]   Last date ({predictions_df.index[-1]}): {predictions_df.iloc[-1].to_dict()}")
                logger.info(f"[{self.name}]   Prediction variance: {predictions_df.var().to_dict()}")

            return predictions_df

        except Exception as e:
            logger.error(f"[{self.name}] ❌ FF5 prediction failed: {e}")
            import traceback
            logger.error(f"[{self.name}] Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    def _extract_date_factors(self, features: pd.DataFrame, date: datetime) -> pd.DataFrame:
        """
        Extract FF5 factor values for a specific date from features DataFrame.

        This method extracts the FF5 factor values that are shared by all symbols
        for a given date. It handles different MultiIndex formats and returns
        a clean DataFrame with factor columns only.

        Args:
            features: Full feature DataFrame with MultiIndex (symbol, date) or (date, symbol)
            date: Specific date to extract FF5 factors for

        Returns:
            DataFrame with FF5 factor values only, shape (1, 5)
            Columns: ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
        """
        try:
            # Define FF5 factor columns we're looking for
            factor_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']

            # Check if we have the required FF5 factor columns
            available_factor_cols = [col for col in factor_cols if col in features.columns]
            if not available_factor_cols:
                logger.warning(f"[{self.name}] No FF5 factor columns found in features. Available: {list(features.columns)}")
                return pd.DataFrame(columns=factor_cols)

            if not isinstance(features.index, pd.MultiIndex):
                # If not MultiIndex, try to find the date in index
                if date in features.index:
                    return pd.DataFrame(
                        [features.loc[date, available_factor_cols].values],
                        columns=available_factor_cols,
                        index=[date]
                    )
                else:
                    logger.warning(f"[{self.name}] Date {date} not found in features index")
                    return pd.DataFrame(columns=available_factor_cols)

            # Handle MultiIndex structure
            index_names = features.index.names

            if 'date' in index_names and 'symbol' in index_names:
                # Panel data format (date, symbol) or (symbol, date)
                try:
                    # Extract data for this specific date
                    date_data = features.xs(date, level='date')

                    # Get FF5 factor values for any symbol (they should be the same)
                    if not date_data.empty:
                        # Take the first symbol's factor values
                        first_symbol = date_data.index[0]
                        factor_values = date_data.loc[first_symbol, available_factor_cols]

                        # Return as DataFrame
                        result = pd.DataFrame([factor_values.values],
                                                columns=available_factor_cols,
                                                index=[date])
                        logger.debug(f"[{self.name}] Extracted FF5 factors for {date}: {result.to_dict('records')[0]}")
                        return result
                    else:
                        logger.warning(f"[{self.name}] No data found for date {date}")
                        return pd.DataFrame(columns=available_factor_cols)

                except Exception as e:
                    logger.error(f"[{self.name}] Failed to extract FF5 factors for {date}: {e}")
                    return pd.DataFrame(columns=available_factor_cols)
            else:
                # Unknown index structure
                logger.warning(f"[{self.name}] Unexpected MultiIndex structure: {index_names}")
                return pd.DataFrame(columns=available_factor_cols)

        except Exception as e:
            logger.error(f"[{self.name}] Error in _extract_date_factors: {e}")
            return pd.DataFrame(columns=['MKT', 'SMB', 'HML', 'RMW', 'CMA'])

    def _predict_iterative(self, factors: pd.DataFrame, symbols: List[str], date: datetime) -> pd.Series:
        """
        FF5 fallback iterative prediction when batch prediction is not available.

        This method calls predict() for each symbol individually and combines
        the results into a Series, using FF5-specific factor formatting.

        Args:
            factors: DataFrame with FF5 factor values, shape (1, 5)
            symbols: List of symbols to predict
            date: Prediction date

        Returns:
            Series with FF5 predictions indexed by symbols
        """
        try:
            predictions = []

            for symbol in symbols:
                try:
                    # Create a copy of factors with MultiIndex for backward compatibility
                    factors_with_multiindex = factors.copy()
                    factors_with_multiindex.index = pd.MultiIndex.from_arrays(
                        [[symbol] * len(factors_with_multiindex), factors_with_multiindex.index],
                        names=['symbol', 'date']
                    )

                    # Use original predict method
                    result = self.model_predictor.predict(
                        features=factors_with_multiindex,
                        symbol=symbol,
                        prediction_date=date
                    )

                    # Extract prediction value
                    prediction_value = result.get('prediction', 0.0)
                    predictions.append(prediction_value)

                except Exception as e:
                    logger.warning(f"[{self.name}] FF5 iterative prediction failed for {symbol}: {e}")
                    predictions.append(0.0)

            return pd.Series(predictions, index=symbols, name='ff5_prediction')

        except Exception as e:
            logger.error(f"[{self.name}] FF5 iterative prediction failed: {e}")
            # Return zero predictions as fallback
            return pd.Series([0.0] * len(symbols), index=symbols, name='ff5_prediction')

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

