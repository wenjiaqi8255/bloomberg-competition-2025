"""
Model Predictor for Production Serving

This module provides the production prediction service that wraps model loading,
prediction, and monitoring. It serves as the interface between trading strategies
and ML models, following SOLID principles.

Key Features:
- Model loading and caching
- Production prediction interface
- Integrated monitoring
- Error handling and fallback
- Performance optimization for batch predictions
- Self-contained data acquisition for production deployment
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import uuid
import threading
from contextlib import contextmanager

from ..base.model_factory import ModelFactory
from ..base.base_model import BaseModel
from ..serving.monitor import ModelMonitor, PredictionRecord, ModelHealthStatus
from ...feature_engineering.feature_engine import FeatureEngine

logger = logging.getLogger(__name__)


class ModelLoadError(Exception):
    """Exception raised when model loading fails."""
    pass


class PredictionError(Exception):
    """Exception raised when prediction fails."""
    pass


class ModelPredictor:
    """
    Production model predictor that provides prediction services.

    This class handles:
    - Loading and caching production models
    - Providing prediction interface to strategies
    - Integrating with monitoring system
    - Handling errors and fallbacks
    - Optimizing batch predictions

    Design Principles:
    - Strategies depend on this interface, not concrete models
    - Single responsibility for prediction serving
    - Integrated monitoring for all predictions
    """

    def __init__(self,
                 model_registry_path: Optional[str] = None,
                 enable_monitoring: bool = True,
                 cache_predictions: bool = True,
                 data_provider=None,
                 ff5_provider=None):
        """
        Initialize the model predictor with optional data providers.

        Args:
            model_registry_path: Path to model registry directory
            enable_monitoring: Whether to enable prediction monitoring
            cache_predictions: Whether to cache recent predictions
            data_provider: Optional data provider for price data
            ff5_provider: Optional FF5 factor data provider
        """
        self.model_registry_path = Path(model_registry_path) if model_registry_path else None
        self.enable_monitoring = enable_monitoring
        self.cache_predictions = cache_predictions

        # Model management
        self._current_model: Optional[BaseModel] = None
        self._current_model_id: Optional[str] = None
        self._model_lock = threading.RLock()

        # Monitoring (will be initialized when model is loaded)
        self._monitor = None
        self._enable_monitoring = enable_monitoring

        # Prediction cache (for recent predictions)
        self._prediction_cache: Dict[str, Any] = {}
        self._cache_lock = threading.RLock()

        # Feature engineering
        self._feature_engine = FeatureEngine()

        # Data providers for self-contained operation
        self._data_provider = data_provider
        self._ff5_provider = ff5_provider

        # Initialize default providers if none provided (for production use)
        if self._data_provider is None or self._ff5_provider is None:
            logger.info("No data providers provided - initializing default providers for self-contained operation")
            self._initialize_default_providers()

        logger.info("ModelPredictor initialized with data acquisition capabilities")

    @property
    def model_id(self) -> Optional[str]:
        """Get the current model ID."""
        return self._current_model_id

    def _initialize_default_providers(self):
        """Initialize default data providers for self-contained operation."""
        try:
            # Initialize YFinance provider for price data
            from ...data.yfinance_provider import YFinanceProvider
            self._data_provider = YFinanceProvider()

            # Initialize FF5 provider for factor data
            from ...data.ff5_provider import FF5DataProvider
            self._ff5_provider = FF5DataProvider(data_frequency="monthly")

            logger.info("Default data providers initialized successfully")
        except ImportError as e:
            logger.warning(f"Failed to initialize default providers: {e}")
            logger.warning("ModelPredictor will require external market_data for predictions")

    def predict(self,
                market_data: Optional[pd.DataFrame] = None,
                symbol: str = None,
                prediction_date: Optional[datetime] = None,
                features: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Make prediction for a single symbol.

        Enhanced to be self-contained - can fetch data automatically if market_data not provided.

        Args:
            market_data: Optional market data DataFrame (if None, will fetch automatically)
            symbol: Symbol to predict for (required if market_data is None)
            prediction_date: Date for prediction (if None, uses current date)
            features: Optional pre-computed features

        Returns:
            Dictionary with prediction results

        Raises:
            PredictionError: If prediction fails
        """
        if self._current_model is None:
            raise PredictionError("No model loaded")

        # Use current date if prediction_date not provided
        if prediction_date is None:
            prediction_date = datetime.now()

        try:
            # Prepare features if not provided
            if features is None:
                features = self._prepare_features_with_data_acquisition(
                    market_data, symbol, prediction_date
                )

            # Make prediction
            prediction = self._current_model.predict(features)

            # Handle different prediction formats
            if isinstance(prediction, np.ndarray):
                if prediction.ndim == 0:
                    prediction_value = float(prediction)
                elif len(prediction) == 1:
                    prediction_value = float(prediction[0])
                else:
                    # For multi-output, take the first one
                    prediction_value = float(prediction[0])
            else:
                prediction_value = float(prediction)

            result = {
                'symbol': symbol,
                'prediction': prediction_value,
                'prediction_date': prediction_date,
                'model_id': self._current_model_id,
                'timestamp': datetime.now()
            }

            # Log prediction for monitoring
            if self._monitor:
                prediction_record = PredictionRecord(
                    timestamp=datetime.now(),
                    model_id=self._current_model_id,
                    prediction_id=str(uuid.uuid4()),
                    features=features.iloc[-1].to_dict() if len(features) > 0 else {},
                    prediction=prediction_value,
                    metadata={
                        'symbol': symbol,
                        'prediction_date': prediction_date,
                        'feature_count': len(features.columns)
                    }
                )
                self._monitor.log_prediction(
                    features=prediction_record.features,
                    prediction=prediction_record.prediction,
                    confidence=prediction_record.confidence,
                    metadata=prediction_record.metadata
                )

            return result

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            raise PredictionError(f"Prediction failed: {e}")

    def load_model(self, model_name: str, model_path: Optional[str] = None) -> str:
        """
        Load a production model.

        Args:
            model_name: Name/type of the model to load
            model_path: Optional path to specific model instance

        Returns:
            Model ID for the loaded model

        Raises:
            ModelLoadError: If model loading fails
        """
        with self._model_lock:
            try:
                # If no specific path, try to create a new model instance
                if model_path is None:
                    model = ModelFactory.create(model_name)
                    model_id = f"{model_name}_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                else:
                    # Load from specific path
                    model_path = Path(model_path)
                    if not model_path.exists():
                        raise ModelLoadError(f"Model path not found: {model_path}")

                    # Determine model type from metadata and load
                    model = self._load_model_from_path(model_path)
                    model_id = f"{model_name}_loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                self._current_model = model
                self._current_model_id = model_id

                # Initialize monitoring for new model
                if self._enable_monitoring:
                    try:
                        from .monitor import ModelMonitor
                        self._monitor = ModelMonitor(model_id)
                        logger.info(f"Model monitoring initialized for {model_id}")
                    except ImportError as e:
                        logger.warning(f"Model monitoring not available: {e}")
                        self._monitor = None

                logger.info(f"Model loaded: {model_id} ({model_name})")
                return model_id

            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                raise ModelLoadError(f"Failed to load model {model_name}: {e}")

    def _load_model_from_path(self, path: Path) -> BaseModel:
        """
        Load a model from a directory path.

        Args:
            path: Path to model directory

        Returns:
            Loaded model instance
        """
        # Try to determine model type from metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            model_type = metadata.get('model_type')
            if model_type and model_type in ModelFactory._registry:
                # Use the specific model class to load
                model_class = ModelFactory._registry[model_type].model_class
                return model_class.load(path)

        # Fallback: try BaseModel.load
        return BaseModel.load(path)

    def get_current_model(self) -> Optional[BaseModel]:
        """
        Get the currently loaded model.

        Returns:
            Current model instance or None if no model loaded
        """
        with self._model_lock:
            return self._current_model

    def get_current_model_id(self) -> Optional[str]:
        """
        Get the ID of the currently loaded model.

        Returns:
            Current model ID or None if no model loaded
        """
        with self._model_lock:
            return self._current_model_id

    def predict(self,
                market_data: pd.DataFrame,
                symbol: str,
                prediction_date: datetime,
                features: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Make prediction for a single symbol.

        Args:
            market_data: Market data DataFrame
            symbol: Symbol to predict for
            prediction_date: Date for prediction
            features: Optional pre-computed features

        Returns:
            Dictionary with prediction results

        Raises:
            PredictionError: If prediction fails
        """
        if self._current_model is None:
            raise PredictionError("No model loaded")

        try:
            # Prepare features if not provided
            if features is None:
                features = self._prepare_features(market_data, symbol, prediction_date)

            # Make prediction
            prediction = self._current_model.predict(features)

            # Handle different prediction formats
            if isinstance(prediction, np.ndarray):
                if prediction.ndim == 0:
                    prediction_value = float(prediction)
                elif len(prediction) == 1:
                    prediction_value = float(prediction[0])
                else:
                    # For multi-output, take the first one
                    prediction_value = float(prediction[0])
            else:
                prediction_value = float(prediction)

            result = {
                'symbol': symbol,
                'prediction': prediction_value,
                'prediction_date': prediction_date,
                'model_id': self._current_model_id,
                'timestamp': datetime.now()
            }

            # Log prediction for monitoring
            if self._monitor:
                prediction_record = PredictionRecord(
                    timestamp=datetime.now(),
                    model_id=self._current_model_id,
                    prediction_id=str(uuid.uuid4()),
                    features=features.iloc[-1].to_dict() if len(features) > 0 else {},
                    prediction=prediction_value,
                    metadata={
                        'symbol': symbol,
                        'prediction_date': prediction_date.isoformat()
                    }
                )
                self._monitor.log_prediction(
                    features=prediction_record.features,
                    prediction=prediction_record.prediction,
                    confidence=prediction_record.confidence,
                    metadata=prediction_record.metadata
                )

            # Cache prediction if enabled
            if self.cache_predictions:
                self._cache_prediction(symbol, prediction_date, result)

            return result

        except Exception as e:
            logger.error(f"Prediction failed for {symbol}: {e}")
            raise PredictionError(f"Prediction failed for {symbol}: {e}")

    def predict_batch(self,
                     market_data: pd.DataFrame,
                     symbols: List[str],
                     prediction_date: datetime) -> Dict[str, Dict[str, float]]:
        """
        Make predictions for multiple symbols efficiently.

        Args:
            market_data: Market data DataFrame
            symbols: List of symbols to predict for
            prediction_date: Date for predictions

        Returns:
            Dictionary mapping symbols to prediction results
        """
        results = {}

        if self._current_model is None:
            raise PredictionError("No model loaded")

        try:
            # Prepare features for all symbols
            all_features = {}
            for symbol in symbols:
                try:
                    features = self._prepare_features(market_data, symbol, prediction_date)
                    all_features[symbol] = features
                except Exception as e:
                    logger.warning(f"Failed to prepare features for {symbol}: {e}")
                    continue

            if not all_features:
                raise PredictionError("No valid features prepared for any symbols")

            # Batch prediction
            # Combine all features into a single DataFrame
            combined_features = pd.concat(
                [features.iloc[[-1]] for features in all_features.values()],
                keys=all_features.keys(),
                names=['symbol', 'date']
            )

            # Make batch prediction
            batch_predictions = self._current_model.predict(combined_features)

            # Process results
            for i, (symbol, features) in enumerate(all_features.items()):
                if i < len(batch_predictions):
                    prediction_value = float(batch_predictions[i])

                    result = {
                        'symbol': symbol,
                        'prediction': prediction_value,
                        'prediction_date': prediction_date,
                        'model_id': self._current_model_id,
                        'timestamp': datetime.now()
                    }

                    results[symbol] = result

                    # Log prediction for monitoring
                    if self._monitor:
                        prediction_record = PredictionRecord(
                            timestamp=datetime.now(),
                            model_id=self._current_model_id,
                            prediction_id=str(uuid.uuid4()),
                            features=features.iloc[-1].to_dict(),
                            prediction=prediction_value,
                            metadata={
                                'symbol': symbol,
                                'prediction_date': prediction_date.isoformat(),
                                'batch_prediction': True
                            }
                        )
                        self._monitor.log_prediction(
                    features=prediction_record.features,
                    prediction=prediction_record.prediction,
                    confidence=prediction_record.confidence,
                    metadata=prediction_record.metadata
                )

                    # Cache prediction if enabled
                    if self.cache_predictions:
                        self._cache_prediction(symbol, prediction_date, result)

            return results

        except Exception as e:
            logger.error(f"Batch prediction failed: {e}")
            raise PredictionError(f"Batch prediction failed: {e}")

    def _prepare_features(self,
                         market_data: pd.DataFrame,
                         symbol: str,
                         prediction_date: datetime) -> pd.DataFrame:
        """
        Prepare features for prediction.

        Args:
            market_data: Market data DataFrame
            symbol: Symbol to prepare features for
            prediction_date: Date for prediction

        Returns:
            Features DataFrame
        """
        try:
            # Check if model needs factor data (FF5) or technical features
            if hasattr(self._current_model, 'model_type'):
                if self._current_model.model_type == "ff5_regression":
                    # For FF5 model, use factor columns directly
                    factor_columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
                    if all(col in market_data.columns for col in factor_columns):
                        # Return the most recent row of factor data
                        return market_data[factor_columns].iloc[[-1]]
                elif self._current_model.model_type == "residual_predictor":
                    # For residual predictor, we need both technical features AND factor data
                    return self._prepare_residual_predictor_features(market_data, symbol, prediction_date)

            # For technical feature models, use feature engineering
            # Filter data for the symbol
            if 'symbol' in market_data.columns:
                symbol_data = market_data[market_data['symbol'] == symbol]
            elif symbol in market_data.columns:
                # Symbol-based column (return data)
                symbol_data = market_data[[symbol]].copy()
                symbol_data.rename(columns={symbol: 'close'}, inplace=True)
                # Add missing OHLCV columns if needed
                for col in ['high', 'low', 'open', 'volume']:
                    if col not in symbol_data.columns:
                        symbol_data[col] = symbol_data['close']  # Simple fallback
            else:
                # Assume this is already price data for the symbol
                symbol_data = market_data.copy()

            # Sort by date
            if 'date' in symbol_data.columns:
                symbol_data = symbol_data.sort_values('date')

            # Standardize column names to match feature engineering expectations
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            symbol_data = symbol_data.rename(columns=column_mapping)

            # Ensure we have all required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in symbol_data.columns:
                    # Create missing columns from Close price as fallback
                    if col == 'Volume':
                        symbol_data[col] = 1_000_000  # Default volume
                    else:
                        symbol_data[col] = symbol_data['Close']

            # Use feature engine to compute features
            from ...feature_engineering.models.data_types import PriceData
            # PriceData is a type alias: Dict[str, pd.DataFrame] (Symbol -> OHLCV DataFrame)
            price_data = {symbol: symbol_data}

            feature_result = self._feature_engine.compute_features(price_data)
            features = feature_result.features

            # Ensure we have features for the prediction date
            if len(features) == 0:
                raise ValueError(f"No features computed for {symbol}")

            # Return the most recent features
            return features.iloc[[-1]]

        except Exception as e:
            logger.error(f"Failed to prepare features for {symbol}: {e}")
            raise ValueError(f"Failed to prepare features for {symbol}: {e}")

    def _cache_prediction(self,
                         symbol: str,
                         prediction_date: datetime,
                         result: Dict[str, Any]) -> None:
        """
        Cache a prediction result.

        Args:
            symbol: Symbol
            prediction_date: Prediction date
            result: Prediction result
        """
        with self._cache_lock:
            cache_key = f"{symbol}_{prediction_date.strftime('%Y-%m-%d')}"
            self._prediction_cache[cache_key] = result

            # Keep cache size limited (keep last 1000 predictions)
            if len(self._prediction_cache) > 1000:
                # Remove oldest entries
                sorted_keys = sorted(self._prediction_cache.keys())
                for old_key in sorted_keys[:100]:
                    del self._prediction_cache[old_key]

    def get_cached_prediction(self,
                             symbol: str,
                             prediction_date: datetime) -> Optional[Dict[str, Any]]:
        """
        Get a cached prediction if available.

        Args:
            symbol: Symbol
            prediction_date: Prediction date

        Returns:
            Cached prediction result or None
        """
        if not self.cache_predictions:
            return None

        with self._cache_lock:
            cache_key = f"{symbol}_{prediction_date.strftime('%Y-%m-%d')}"
            return self._prediction_cache.get(cache_key)

    def get_model_health(self) -> Optional[ModelHealthStatus]:
        """
        Get the health status of the current model.

        Returns:
            Model health status or None if no model loaded
        """
        if self._monitor and self._current_model_id:
            return self._monitor.get_health_status(self._current_model_id)
        return None

    def switch_model(self, model_name: str, model_path: Optional[str] = None) -> str:
        """
        Switch to a different model.

        Args:
            model_name: Name of the model to switch to
            model_path: Optional path to specific model instance

        Returns:
            New model ID
        """
        logger.info(f"Switching model from {self._current_model_id} to {model_name}")
        return self.load_model(model_name, model_path)

    def list_available_models(self) -> List[str]:
        """
        List all available model types.

        Returns:
            List of available model type names
        """
        return list(ModelFactory._registry.keys())

    def _prepare_residual_predictor_features(self,
                                           market_data: pd.DataFrame,
                                           symbol: str,
                                           prediction_date: datetime) -> pd.DataFrame:
        """
        Prepare features for residual predictor model.

        The residual predictor needs both technical features (from OHLCV data)
        and factor features (FF5 factors).

        Args:
            market_data: Market data DataFrame with OHLCV + factor columns
            symbol: Symbol to prepare features for
            prediction_date: Date for prediction

        Returns:
            Combined features DataFrame
        """
        try:
            # Filter data for the symbol
            if 'symbol' in market_data.columns:
                symbol_data = market_data[market_data['symbol'] == symbol].copy()
            else:
                # Assume this is already price data for the symbol
                symbol_data = market_data.copy()

            # Sort by date
            if 'date' in symbol_data.columns:
                symbol_data = symbol_data.sort_values('date')

            # Standardize column names to match feature engineering expectations
            column_mapping = {
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }
            symbol_data = symbol_data.rename(columns=column_mapping)

            # Ensure we have all required OHLCV columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            for col in required_columns:
                if col not in symbol_data.columns:
                    # Create missing columns from Close price as fallback
                    if col == 'Volume':
                        symbol_data[col] = 1_000_000  # Default volume
                    else:
                        symbol_data[col] = symbol_data['Close']

            # Step 1: Compute technical features using feature engine
            from ...feature_engineering.models.data_types import PriceData

            # For residual predictor, we need features for all symbols that were in training
            # Extract all symbols from market_data
            all_symbols_in_data = market_data['symbol'].unique() if 'symbol' in market_data.columns else [symbol]

            # Create price data for all symbols
            price_data = {}
            for s in all_symbols_in_data:
                s_data = market_data[market_data['symbol'] == s].copy()
                if 'date' in s_data.columns:
                    s_data = s_data.set_index('date')
                price_data[s] = s_data.sort_index()

            # Compute features
            feature_result = self._feature_engine.compute_features(price_data)
            technical_features = feature_result.features

            logger.info(f"Computed technical features shape: {technical_features.shape}")
            logger.info(f"Technical features columns sample: {list(technical_features.columns)[:10]}")

            if len(technical_features) == 0:
                raise ValueError(f"No technical features computed for symbols {all_symbols_in_data}")

            # Step 2: Get factor features from the original market data
            factor_columns = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
            factor_data = {}

            for factor_col in factor_columns:
                if factor_col in symbol_data.columns:
                    # Get the most recent factor value
                    factor_data[factor_col] = symbol_data[factor_col].iloc[-1]
                else:
                    logger.warning(f"Factor column {factor_col} not found in market data")
                    factor_data[factor_col] = 0.0  # Default value

            # Step 3: Combine technical and factor features
            # Create factor features DataFrame with matching index
            factor_features = pd.DataFrame([factor_data], index=technical_features.index)

            # Combine features
            combined_features = pd.concat([technical_features, factor_features], axis=1)

            logger.info(f"Prepared residual predictor features: {combined_features.shape}")
            logger.info(f"Technical features: {len(technical_features.columns)}")
            logger.info(f"Factor features: {len(factor_features.columns)}")
            logger.info(f"Total features: {len(combined_features.columns)}")

            # Return the most recent combined features
            return combined_features.iloc[[-1]]

        except Exception as e:
            logger.error(f"Failed to prepare residual predictor features for {symbol}: {e}")
            raise ValueError(f"Failed to prepare residual predictor features for {symbol}: {e}")

    def _prepare_features_with_data_acquisition(self,
                                                market_data: Optional[pd.DataFrame],
                                                symbol: str,
                                                prediction_date: datetime) -> pd.DataFrame:
        """
        Prepare features with automatic data acquisition if needed.

        This method makes the ModelPredictor self-contained by fetching data
        automatically when market_data is not provided.

        Args:
            market_data: Optional market data DataFrame (if None, will fetch automatically)
            symbol: Symbol to prepare features for
            prediction_date: Date for prediction

        Returns:
            Features DataFrame ready for prediction
        """
        # If market_data is provided, use existing feature preparation
        if market_data is not None:
            return self._prepare_features(market_data, symbol, prediction_date)

        # Otherwise, fetch data automatically
        if symbol is None:
            raise ValueError("Symbol is required when market_data is not provided")

        try:
            # Fetch price data
            if self._data_provider is None:
                raise PredictionError("No data provider available and no market_data provided")

            # Calculate date range for data fetch (need lookback data for features)
            end_date = prediction_date
            start_date = end_date - timedelta(days=365)  # 1 year lookback for technical indicators

            # Fetch price data for the symbol
            price_data_dict = self._data_provider.get_price_data(
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date
            )

            if symbol not in price_data_dict or price_data_dict[symbol].empty:
                raise PredictionError(f"No price data available for {symbol}")

            price_data = price_data_dict[symbol]

            # Fetch factor data if needed (for residual predictor)
            if (hasattr(self._current_model, 'model_type') and
                self._current_model.model_type == "residual_predictor"):

                if self._ff5_provider is None:
                    raise PredictionError("FF5 provider not available for residual predictor")

                factor_data = self._ff5_provider.get_factor_data(start_date, end_date)

                # Prepare market_data with both price and factor data
                market_data_auto = price_data.copy()
                market_data_auto['symbol'] = symbol

                # Add factor data if available
                if not factor_data.empty:
                    # Reindex factor data to match price data dates
                    factor_data_aligned = factor_data.reindex(market_data_auto.index, method='ffill')
                    for col in factor_data_aligned.columns:
                        market_data_auto[col] = factor_data_aligned[col]
            else:
                # For non-residual models, just use price data
                market_data_auto = price_data.copy()
                market_data_auto['symbol'] = symbol

            # Use the existing feature preparation logic
            return self._prepare_features(market_data_auto, symbol, prediction_date)

        except Exception as e:
            logger.error(f"Failed to prepare features with data acquisition for {symbol}: {e}")
            raise PredictionError(f"Failed to prepare features: {e}")

    def get_model_info(self, model_type: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a model type.

        Args:
            model_type: Model type to get info for

        Returns:
            Model information or None if not found
        """
        if model_type in ModelFactory._registry:
            registration = ModelFactory._registry[model_type]
            return {
                'type': model_type,
                'class': registration.model_class.__name__,
                'description': registration.description,
                'default_config': registration.default_config
            }
        return None