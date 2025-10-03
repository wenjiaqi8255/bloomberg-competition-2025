"""
Prediction Pipeline
==================

This module provides the `PredictionPipeline`, which mirrors the `TrainingPipeline`
design for production inference. It ensures that prediction follows the same clean
architecture as training:

Training Flow:
    TrainingPipeline manages data providers
    → FeatureEngineeringPipeline.fit() & transform()
    → Model.train()

Prediction Flow:
    PredictionPipeline manages data providers
    → FeatureEngineeringPipeline.transform()
    → ModelPredictor.predict()

Key Design Principles:
---------------------
1. **Single Responsibility**: PredictionPipeline handles data acquisition,
   ModelPredictor handles inference only
2. **Symmetry**: Matches TrainingPipeline's architecture
3. **Provider Management**: Centralizes all data provider dependencies
4. **Feature Consistency**: Uses the same FeatureEngineeringPipeline as training

This design fixes the architectural issue where ModelPredictor was trying to
manage data providers, violating single responsibility principle.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd

from ..base.base_model import BaseModel
from .predictor import ModelPredictor
from ...feature_engineering.pipeline import FeatureEngineeringPipeline
from ...data.base_data_provider import BaseDataProvider

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    End-to-end prediction pipeline that orchestrates:
    1. Data acquisition from providers
    2. Feature engineering
    3. Model prediction
    
    This pipeline ensures predictions use the same data flow as training.
    """
    
    def __init__(self,
                 model_predictor: ModelPredictor,
                 feature_pipeline: FeatureEngineeringPipeline,
                 data_provider: Optional[BaseDataProvider] = None,
                 factor_data_provider: Optional[BaseDataProvider] = None):
        """
        Initialize the prediction pipeline.
        
        Args:
            model_predictor: ModelPredictor instance (should NOT manage data providers)
            feature_pipeline: Fitted FeatureEngineeringPipeline from training
            data_provider: Price data provider
            factor_data_provider: Optional factor data provider (for FF5, etc.)
        """
        self.model_predictor = model_predictor
        self.feature_pipeline = feature_pipeline
        self.data_provider = data_provider
        self.factor_data_provider = factor_data_provider
        
        # Validate that feature pipeline is fitted
        if not self.feature_pipeline._is_fitted:
            logger.warning("FeatureEngineeringPipeline is not fitted. "
                         "Features may not be properly scaled.")
    
    def predict(self,
                symbols: List[str],
                prediction_date: datetime,
                price_data: Optional[Dict[str, pd.DataFrame]] = None,
                lookback_days: int = 365) -> Dict[str, Dict[str, Any]]:
        """
        Generate predictions for a list of symbols.
        
        Args:
            symbols: List of symbols to predict for
            prediction_date: Date to make predictions for
            price_data: Optional pre-fetched price data (if None, will fetch automatically)
            lookback_days: Days of historical data to fetch for feature computation
            
        Returns:
            Dictionary mapping symbols to prediction results
            
        Example:
            predictions = pipeline.predict(
                symbols=['AAPL', 'MSFT'],
                prediction_date=datetime.now()
            )
            # {'AAPL': {'prediction': 0.05, 'confidence': 0.8}, ...}
        """
        logger.info(f"PredictionPipeline: Generating predictions for {len(symbols)} symbols")
        
        try:
            # Step 1: Load data (either use provided or fetch)
            if price_data is None:
                logger.info("Step 1: Fetching price data...")
                price_data = self._fetch_price_data(symbols, prediction_date, lookback_days)
            else:
                logger.info("Step 1: Using provided price data")
            
            # Step 2: Load factor data if needed
            logger.info("Step 2: Checking for factor data requirements...")
            factor_data = self._fetch_factor_data(prediction_date, lookback_days)
            
            # Step 3: Prepare data for feature engineering
            logger.info("Step 3: Computing features...")
            feature_input_data = {'price_data': price_data}
            if factor_data is not None:
                feature_input_data['factor_data'] = factor_data
            
            # Step 4: Transform using fitted feature pipeline
            features = self.feature_pipeline.transform(feature_input_data)
            
            logger.info(f"Features computed: shape={features.shape}, columns={len(features.columns)}")
            
            # Step 5: Generate predictions for each symbol
            logger.info("Step 4: Generating model predictions...")
            predictions = {}
            
            for symbol in symbols:
                try:
                    # Extract features for this symbol
                    symbol_features = self._extract_symbol_features(features, symbol)
                    
                    if symbol_features.empty:
                        logger.warning(f"No features available for {symbol}")
                        continue
                    
                    # Get prediction from model (ModelPredictor only does inference)
                    prediction_result = self.model_predictor.predict(
                        features=symbol_features,
                        symbol=symbol,
                        prediction_date=prediction_date
                    )
                    
                    predictions[symbol] = prediction_result
                    
                except Exception as e:
                    logger.error(f"Prediction failed for {symbol}: {e}")
                    continue
            
            logger.info(f"Successfully generated predictions for {len(predictions)} symbols")
            return predictions
            
        except Exception as e:
            logger.error(f"PredictionPipeline failed: {e}")
            raise RuntimeError(f"Prediction pipeline failed: {e}")
    
    def predict_batch(self,
                     symbols: List[str],
                     prediction_dates: List[datetime],
                     price_data: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[datetime, Dict[str, Dict[str, Any]]]:
        """
        Generate predictions for multiple dates (batch prediction).
        
        Args:
            symbols: List of symbols to predict for
            prediction_dates: List of dates to make predictions for
            price_data: Optional pre-fetched price data
            
        Returns:
            Dictionary mapping dates to symbol predictions
            
        Example:
            predictions = pipeline.predict_batch(
                symbols=['AAPL', 'MSFT'],
                prediction_dates=[date1, date2, date3]
            )
            # {date1: {'AAPL': {...}, 'MSFT': {...}}, date2: {...}}
        """
        logger.info(f"PredictionPipeline: Batch prediction for {len(symbols)} symbols across {len(prediction_dates)} dates")
        
        batch_predictions = {}
        for pred_date in prediction_dates:
            try:
                predictions = self.predict(
                    symbols=symbols,
                    prediction_date=pred_date,
                    price_data=price_data
                )
                batch_predictions[pred_date] = predictions
            except Exception as e:
                logger.error(f"Batch prediction failed for date {pred_date}: {e}")
                batch_predictions[pred_date] = {}
        
        return batch_predictions
    
    def predict_for_date(self,
                         prediction_date: datetime,
                         symbols: List[str],
                         lookback_start_date: datetime) -> Dict[str, float]:
        """
        Generates predictions for a list of symbols for a single specific date.

        This is the primary method used by the backtesting engine. It orchestrates
        fetching the correct data window, computing features for that window, and
        returning a dictionary of final prediction values.

        Args:
            prediction_date: The specific date to generate predictions for.
            symbols: A list of symbols to predict.
            lookback_start_date: The start date of the historical data window needed
                                 to compute all features correctly.

        Returns:
            A dictionary mapping each symbol to its float prediction value.
            e.g., {'AAPL': 0.015, 'MSFT': -0.005}
        """
        logger.info(f"Executing predict_for_date for {prediction_date.date()} with {len(symbols)} symbols.")
        
        try:
            # Step 1: Fetch historical price data for the required lookback window.
            if self.data_provider is None:
                raise ValueError("PredictionPipeline requires a data_provider for this operation.")
            
            logger.debug(f"Fetching price data from {lookback_start_date.date()} to {prediction_date.date()}")
            price_data = self.data_provider.get_data(
                start_date=lookback_start_date,
                end_date=prediction_date,
                symbols=symbols
            )

            # Step 2: Fetch corresponding factor data if a provider is available.
            factor_data = None
            if self.factor_data_provider:
                logger.debug("Fetching factor data for the same period.")
                factor_data = self.factor_data_provider.get_data(
                    start_date=lookback_start_date,
                    end_date=prediction_date
                )

            # Step 3: Compute features using the full data window.
            pipeline_input = {'price_data': price_data}
            if factor_data is not None and not factor_data.empty:
                pipeline_input['factor_data'] = factor_data
            
            logger.debug("Transforming data through the feature pipeline...")
            all_features = self.feature_pipeline.transform(pipeline_input)

            if all_features.empty:
                logger.warning("Feature computation resulted in an empty DataFrame.")
                return {}

            # Step 4: Iterate through symbols and get the prediction for the specific date.
            predictions = {}
            for symbol in symbols:
                if symbol not in price_data or price_data[symbol].empty:
                    logger.warning(f"No price data for {symbol} on {prediction_date.date()}, skipping prediction.")
                    continue
                
                # Extract features for the current symbol.
                symbol_features = self._extract_symbol_features(all_features, symbol)
                
                if symbol_features.empty:
                    logger.warning(f"Could not extract features for {symbol} for {prediction_date.date()}.")
                    continue
                
                # IMPORTANT: Use only the latest row of features for prediction.
                latest_features = symbol_features.iloc[-1:]

                try:
                    result = self.model_predictor.predict(
                        features=latest_features,
                        symbol=symbol,
                        prediction_date=prediction_date
                    )
                    # The predictor returns a dict, we need the raw prediction value.
                    predictions[symbol] = result.get('prediction', 0.0)
                except Exception as e:
                    logger.error(f"Prediction for symbol {symbol} failed: {e}", exc_info=True)
                    predictions[symbol] = 0.0 # Assign a neutral prediction on failure

            logger.info(f"Generated predictions for {len(predictions)} symbols for {prediction_date.date()}.")
            return predictions

        except Exception as e:
            logger.error(f"An error occurred in predict_for_date: {e}", exc_info=True)
            return {}

    def _fetch_price_data(self,
                         symbols: List[str],
                         end_date: datetime,
                         lookback_days: int) -> Dict[str, pd.DataFrame]:
        """
        Fetch price data from the data provider.
        
        Args:
            symbols: List of symbols
            end_date: End date for data
            lookback_days: Number of days to look back
            
        Returns:
            Dictionary mapping symbols to price DataFrames
        """
        if self.data_provider is None:
            raise ValueError("No data provider configured for price data")
        
        start_date = end_date - timedelta(days=lookback_days)
        
        logger.info(f"Fetching price data for {len(symbols)} symbols from {start_date} to {end_date}")
        price_data = self.data_provider.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )
        
        return price_data
    
    def _fetch_factor_data(self,
                          end_date: datetime,
                          lookback_days: int) -> Optional[pd.DataFrame]:
        """
        Fetch factor data if a factor data provider is configured.
        
        Args:
            end_date: End date for data
            lookback_days: Number of days to look back
            
        Returns:
            Factor data DataFrame or None if no provider
        """
        if self.factor_data_provider is None:
            logger.debug("No factor data provider configured")
            return None
        
        start_date = end_date - timedelta(days=lookback_days)
        
        logger.info(f"Fetching factor data from {start_date} to {end_date}")
        
        # Check if provider has get_factor_data method (FF5DataProvider)
        if hasattr(self.factor_data_provider, 'get_factor_data'):
            factor_data = self.factor_data_provider.get_factor_data(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
        # Otherwise try get_factor_returns method
        elif hasattr(self.factor_data_provider, 'get_factor_returns'):
            factor_data = self.factor_data_provider.get_factor_returns(
                start_date=start_date,
                end_date=end_date
            )
        else:
            logger.warning(f"Factor data provider {type(self.factor_data_provider).__name__} "
                         "has no recognized method for fetching factor data")
            return None
        
        return factor_data
    
    def _extract_symbol_features(self,
                                features: pd.DataFrame,
                                symbol: str) -> pd.DataFrame:
        """
        Extract features for a specific symbol from the feature DataFrame.
        
        Args:
            features: Full feature DataFrame (may have MultiIndex)
            symbol: Symbol to extract features for
            
        Returns:
            Features for the specified symbol
        """
        try:
            # Check if features have MultiIndex with symbol level
            if isinstance(features.index, pd.MultiIndex):
                if 'symbol' in features.index.names:
                    # Extract by symbol level
                    symbol_features = features.xs(symbol, level='symbol')
                    logger.debug(f"Extracted features for {symbol}: shape={symbol_features.shape}")
                    return symbol_features
                else:
                    logger.warning(f"MultiIndex found but no 'symbol' level")
                    return pd.DataFrame()
            else:
                # For single-index features, return the most recent row
                # This is a fallback for simpler feature structures
                logger.debug(f"Single-index features, returning most recent")
                return features.iloc[[-1]]
                
        except KeyError:
            logger.warning(f"Symbol {symbol} not found in features")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to extract features for {symbol}: {e}")
            return pd.DataFrame()
    
    def configure_providers(self,
                          data_provider: Optional[BaseDataProvider] = None,
                          factor_data_provider: Optional[BaseDataProvider] = None) -> 'PredictionPipeline':
        """
        Configure or update data providers.
        
        Args:
            data_provider: Price data provider
            factor_data_provider: Factor data provider
            
        Returns:
            Self for method chaining
        """
        if data_provider is not None:
            self.data_provider = data_provider
            logger.info(f"Configured data provider: {type(data_provider).__name__}")
        
        if factor_data_provider is not None:
            self.factor_data_provider = factor_data_provider
            logger.info(f"Configured factor data provider: {type(factor_data_provider).__name__}")
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        return {
            'model_id': self.model_predictor.model_id,
            'feature_pipeline_fitted': self.feature_pipeline._is_fitted,
            'has_data_provider': self.data_provider is not None,
            'has_factor_provider': self.factor_data_provider is not None
        }


