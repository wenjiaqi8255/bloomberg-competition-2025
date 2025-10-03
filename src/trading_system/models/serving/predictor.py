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
from ..serving.monitor import ModelMonitor, PredictionRecord, ModelHealthStatus

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
                 model_path: Optional[str] = None,
                 model_instance: Optional[BaseModel] = None,
                 model_registry_path: Optional[str] = None,
                 enable_monitoring: bool = True,
                 cache_predictions: bool = True):
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

        # Auto-load model if provided during initialization
        if model_instance is not None:
            # Mode 3: Direct model injection
            self._current_model = model_instance
            self._current_model_id = f"{model_instance.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"ModelPredictor initialized with injected model: {self._current_model_id}")
            self._initialize_monitoring()
        elif model_path is not None:
            # Mode 2: Load from path
            self.load_model(model_name="from_path", model_path=model_path)
            logger.info(f"ModelPredictor initialized by loading model from: {model_path}")
        elif model_id is not None:
            # Mode 1: Create from factory
            self.load_model(model_name=model_id)
            logger.info(f"ModelPredictor initialized by creating model: {model_id}")
        else:
            logger.info("ModelPredictor initialized without model (model must be loaded later)")

    @property
    def model_id(self) -> Optional[str]:
        """Get the current model ID."""
        return self._current_model_id

    def _initialize_monitoring(self):
        """Initialize model monitoring if enabled."""
        if self._enable_monitoring and self._current_model_id:
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
            logger.debug(f"Current model type: {getattr(self._current_model, 'model_type', 'No model_type attribute')}")
            logger.debug(f"Features shape={features.shape}, columns={features.columns.tolist()}")
            if len(features) > 0:
                logger.debug(f"Features sample: {features.iloc[-1].to_dict()}")

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

    def predict_batch(self,
                     features_dict: Dict[str, pd.DataFrame],
                     prediction_date: datetime) -> Dict[str, Dict[str, float]]:
        """
        Make predictions for multiple symbols efficiently using pre-computed features.

        Note: Use PredictionPipeline for end-to-end batch prediction with data acquisition.

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
            logger.error(f"Batch prediction failed: {e}")
            raise PredictionError(f"Batch prediction failed: {e}")

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