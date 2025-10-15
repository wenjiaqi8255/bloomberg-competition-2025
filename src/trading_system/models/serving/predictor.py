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
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import uuid
import threading
from contextlib import contextmanager

from ..base.model_factory import ModelFactory
from ..base.base_model import BaseModel
from ..model_persistence import ModelRegistry # Import ModelRegistry
from .monitor import ModelMonitor, PredictionRecord, ModelHealthStatus

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
                 model_id: Optional[str] = None,
                 model_path: Optional[str] = None, # Kept for potential direct path loading
                 model_instance: Optional[BaseModel] = None,
                 model_registry_path: str = "./models/"): # Standardized registry path
        """
        Initialize the model predictor with flexible model loading.

        Three ways to initialize:
        1. model_id: Create a new model instance from ModelFactory
        2. model_path: Load a trained model from disk
        3. model_instance: Directly inject a model instance

        Args:
            model_id: Model type identifier to create from ModelFactory (e.g., 'ff5_regression')
            model_path: Path to saved model directory
            model_instance: Pre-initialized model instance
            model_registry_path: Path to model registry directory (for backward compatibility)
            enable_monitoring: Whether to enable prediction monitoring
            cache_predictions: Whether to cache recent predictions

        Note:
            ModelPredictor no longer manages data providers. Use PredictionPipeline
            for end-to-end prediction with data acquisition.
        """
        self.model_registry_path = Path(model_registry_path)
        self.registry = ModelRegistry(str(self.model_registry_path)) # Initialize registry

        # Model management
        self._current_model: Optional[BaseModel] = None
        self._current_model_id: Optional[str] = None
        self._model_lock = threading.RLock()
        self._monitor = None

        # Auto-load model if provided during initialization
        if model_instance is not None:
            self._current_model = model_instance
            self._current_model_id = f"injected_{model_instance.model_type}"
            logger.info(f"ModelPredictor initialized with injected model: {self._current_model_id}")
        elif model_id is not None:
            # Main path for loading models by ID
            self.load_model(model_id)
            logger.info(f"ModelPredictor initialized by loading model_id: {model_id}")
        elif model_path is not None:
            # Fallback for loading from a direct, full path. The ID is the directory name.
            self.load_model(Path(model_path).name)
            logger.info(f"ModelPredictor initialized by loading from direct path: {model_path}")
        else:
            logger.info("ModelPredictor initialized without a model.")

    @property
    def model_id(self) -> Optional[str]:
        """Get the current model ID."""
        return self._current_model_id

    def _initialize_monitoring(self):
        """Initialize model monitoring if enabled."""
        if self._current_model_id:
            try:
                from .monitor import ModelMonitor
                self._monitor = ModelMonitor(self._current_model_id)
                logger.info(f"Model monitoring initialized for {self._current_model_id}")
            except ImportError as e:
                logger.warning(f"Model monitoring not available: {e}")
                self._monitor = None

    def predict(self,
                features: pd.DataFrame,
                symbol: str = None,
                prediction_date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Make prediction for a single symbol using pre-computed features.

        This method is now simplified to only handle inference. Data acquisition
        and feature engineering should be done by PredictionPipeline.

        Args:
            features: Pre-computed features DataFrame
            symbol: Symbol being predicted (for metadata only)
            prediction_date: Date for prediction (for metadata only)

        Returns:
            Dictionary with prediction results

        Raises:
            PredictionError: If prediction fails
        """
        if self._current_model is None:
            raise PredictionError("No model loaded")

        if features is None or features.empty:
            raise PredictionError("Features must be provided")

        # Use current date if prediction_date not provided
        if prediction_date is None:
            prediction_date = datetime.now()

        try:
            # Debug: Log model type and features before prediction
            logger.info(f"🔍 ModelPredictor.predict() starting for {symbol}")
            logger.info(f"Current model type: {getattr(self._current_model, 'model_type', 'No model_type attribute')}")
            logger.info(f"Current model ID: {self._current_model_id}")
            logger.info(f"Model class: {type(self._current_model)}")
            logger.info(f"Model trained: {getattr(self._current_model, 'is_trained', 'Unknown')}")
            logger.info(f"Features shape={features.shape}, columns={features.columns.tolist()}")
            if len(features) > 0:
                logger.info(f"Features sample: {features.iloc[-1].to_dict()}")
                logger.info(f"Features non-zero count: {(features.iloc[-1] != 0).sum()}")

            # Make prediction
            logger.info(f"Calling model.predict() for {symbol} on {prediction_date}")
            prediction = self._current_model.predict(features)
            logger.info(f"Raw prediction result: {prediction} (type: {type(prediction)})")

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
                # If an explicit model_path is provided, use its directory name as the ID
                effective_model_id = Path(model_path).name if model_path else model_name

                # The logic is now unified: we always try to load by ID.
                # If the ID exists as a directory, we load it and its artifacts.
                # If it doesn't exist, we fall back to creating a new model from the factory.
                model_dir = self.model_registry_path / effective_model_id
                
                if model_dir.is_dir():
                    logger.info(f"Found model directory for '{effective_model_id}'. Loading model and artifacts...")
                    model = self._load_model_with_artifacts(effective_model_id)
                    if model is None:
                         raise ModelLoadError(f"Could not load model from ID: {effective_model_id}")
                    model_id = effective_model_id
                else:
                    logger.info(f"'{model_name}' not found as a model ID. Assuming it's a model type and creating new instance.")
                    model = ModelFactory.create(model_name)
                    model_id = f"{model_name}_new_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                self._current_model = model
                self._current_model_id = model_id
                
                self._initialize_monitoring()

                logger.info(f"Model ready: {model_id}")
                return model_id

            except Exception as e:
                logger.error(f"Failed to load or create model '{model_name}': {e}", exc_info=True)
                raise ModelLoadError(f"Failed to load model {model_name}: {e}")

    def _load_model_with_artifacts(self, model_id: str) -> Optional[BaseModel]:
        """
        Loads a model and its artifacts, attaching the feature pipeline to the model object.
        """
        loaded = self.registry.load_model_with_artifacts(model_id)
        if loaded:
            model, artifacts = loaded
            if 'feature_pipeline' in artifacts:
                model.feature_pipeline = artifacts['feature_pipeline']
                # Mark the model as trained since it's loaded from the registry
                if hasattr(model, 'is_trained'):
                    model.is_trained = True
                logger.info(f"Attached 'feature_pipeline' artifact to loaded model '{model_id}'.")
            return model
        logger.warning(f"Loading failed. No model or artifacts found for ID '{model_id}' in registry.")
        return None

    def _load_model_from_path(self, path: Path) -> BaseModel:
        """
        [DEPRECATED] This method is replaced by _load_model_with_artifacts.
        """
        logger.warning("Using deprecated _load_model_from_path. The new loader is now ID-based.")
        return self._load_model_with_artifacts(path.name)
        
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

    def predict_batch(self,
                     factors: pd.DataFrame,
                     symbols: List[str],
                     date: datetime) -> pd.Series:
        """
        Batch prediction for multiple symbols using factor values.

        This is the new optimized method that replaces the old nested loop approach.
        Instead of predicting symbol by symbol, it predicts all symbols at once
        using the same factor values.

        Args:
            factors: DataFrame with factor values, shape (1, N_factors)
                       Columns should include ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
            symbols: List of symbols to predict
            date: Prediction date (for logging and monitoring)

        Returns:
            Series with predictions indexed by symbols

        Raises:
            PredictionError: If batch prediction fails
        """
        if self._current_model is None:
            raise PredictionError("No model loaded for batch prediction")

        if factors is None or factors.empty:
            raise PredictionError("Factors must be provided for batch prediction")

        if not symbols:
            raise PredictionError("Symbols list must be provided for batch prediction")

        try:
            logger.debug(f"[{self._current_model_id}] Batch predicting {len(symbols)} symbols for {date}")
            logger.debug(f"Factors shape: {factors.shape}, columns: {list(factors.columns)}")

            # Call the model's batch prediction method
            if hasattr(self._current_model, 'predict') and 'symbols' in self._current_model.predict.__code__.co_varnames:
                # Model supports batch prediction with symbols parameter
                logger.debug("Using model's batch prediction with symbols parameter")
                predictions = self._current_model.predict(X=factors, symbols=symbols)
            else:
                # Model doesn't support batch prediction, fall back to iterative
                logger.warning("Model doesn't support batch prediction, falling back to iterative")
                predictions = self._predict_batch_iterative(factors, symbols, date)

            # Ensure we return a Series
            if not isinstance(predictions, pd.Series):
                if isinstance(predictions, dict):
                    # Convert dict to Series
                    predictions = pd.Series(predictions)
                elif isinstance(predictions, (list, np.ndarray)):
                    predictions = pd.Series(predictions, index=symbols)
                else:
                    # Convert to Series
                    predictions = pd.Series([float(predictions)] * len(symbols), index=symbols)

            logger.debug(f"[{self._current_model_id}] Batch prediction completed: {predictions.shape}")
            logger.debug(f"[{self._current_model_id}] Sample predictions: {predictions.head(3).to_dict()}")

            # Log for monitoring if enabled
            if self._monitor:
                try:
                    prediction_record = PredictionRecord(
                        timestamp=datetime.now(),
                        model_id=self._current_model_id,
                        prediction_id=str(uuid.uuid4()),
                        features=factors.iloc[0].to_dict() if len(factors) > 0 else {},
                        prediction=float(predictions.mean()),  # Use mean as representative
                        confidence=1.0,  # Placeholder
                        metadata={
                            'symbols_count': len(symbols),
                            'factors_shape': factors.shape,
                            'date': date,
                            'batch_prediction': True
                        }
                    )
                    self._monitor.log_prediction(
                        features=prediction_record.features,
                        prediction=prediction_record.prediction,
                        confidence=prediction_record.confidence,
                        metadata=prediction_record.metadata
                    )
                except Exception as e:
                    logger.warning(f"Failed to log batch prediction: {e}")

            return predictions

        except Exception as e:
            logger.error(f"[{self._current_model_id}] Batch prediction failed: {e}")
            raise PredictionError(f"Batch prediction failed: {e}")

    def predict_batch_legacy(self,
                           features_dict: Dict[str, pd.DataFrame],
                           prediction_date: datetime) -> Dict[str, Dict[str, float]]:
        """
        Legacy batch prediction method (kept for backward compatibility).

        This method makes predictions for multiple symbols using pre-computed
        features, calling predict() for each symbol individually.

        Args:
            features_dict: Dictionary mapping symbols to their pre-computed features
            prediction_date: Date for predictions

        Returns:
            Dictionary mapping symbols to prediction results
        """
        results = {}

        if self._current_model is None:
            raise PredictionError("No model loaded")

        try:
            if not features_dict:
                raise PredictionError("No features provided")

            # Make predictions for each symbol
            for symbol, features in features_dict.items():
                try:
                    result = self.predict(
                        features=features,
                        symbol=symbol,
                        prediction_date=prediction_date
                    )
                    results[symbol] = result

                    # Cache prediction if enabled
                    if self.cache_predictions:
                        self._cache_prediction(symbol, prediction_date, result)

                except Exception as e:
                    logger.warning(f"Failed to predict for {symbol}: {e}")
                    continue

            return results

        except Exception as e:
            logger.error(f"Legacy batch prediction failed: {e}")
            raise PredictionError(f"Legacy batch prediction failed: {e}")

    def _predict_batch_iterative(self,
                                factors: pd.DataFrame,
                                symbols: List[str],
                                date: datetime) -> pd.Series:
        """
        Fallback method for batch prediction when model doesn't support native batch prediction.

        This method calls the individual predict() method for each symbol and combines
        the results into a Series. It's used as a fallback for models that don't
        support the new batch prediction interface.

        Args:
            factors: DataFrame with factor values, shape (1, N_factors)
            symbols: List of symbols to predict
            date: Prediction date (for logging and monitoring)

        Returns:
            Series with predictions indexed by symbols

        Raises:
            PredictionError: If any individual prediction fails
        """
        predictions = {}

        try:
            logger.debug(f"Using iterative batch prediction for {len(symbols)} symbols")

            for symbol in symbols:
                try:
                    # Call the individual predict method
                    result = self.predict(
                        features=factors,
                        symbol=symbol,
                        prediction_date=date
                    )

                    # Extract the prediction value
                    if isinstance(result, dict) and 'prediction' in result:
                        predictions[symbol] = result['prediction']
                    else:
                        # Fallback: try to extract numeric value
                        predictions[symbol] = float(result)

                except Exception as e:
                    logger.warning(f"Iterative prediction failed for {symbol}: {e}")
                    # Use a neutral prediction rather than failing the entire batch
                    predictions[symbol] = 0.0
                    continue

            # Convert to Series
            predictions_series = pd.Series(predictions, index=symbols)

            logger.debug(f"Iterative batch prediction completed: {predictions_series.shape}")
            logger.debug(f"Iterative batch prediction sample: {predictions_series.head(3).to_dict()}")

            return predictions_series

        except Exception as e:
            logger.error(f"Iterative batch prediction failed completely: {e}")
            raise PredictionError(f"Iterative batch prediction failed: {e}")

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