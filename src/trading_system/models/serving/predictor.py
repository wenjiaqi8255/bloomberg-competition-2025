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
                symbols: List[str],
                date: Optional[datetime] = None) -> pd.Series:
        """
        Unified batch prediction interface.
        
        Automatically optimizes based on model's batch capability:
        - Batch-capable: Predicts all symbols at once (1 model call)
        - Independent: Iterates over symbols (N model calls)
        
        Args:
            features: Feature DataFrame
            symbols: List of symbols to predict (REQUIRED)
            date: Prediction date (for logging/metadata)
            
        Returns:
            pd.Series with predictions indexed by symbols
        """
        if self._current_model is None:
            raise PredictionError("No model loaded")
            
        if not symbols:
            raise ValueError("symbols parameter is required")
    
        try:
            # Route based on batch capability
            if self._current_model.supports_batch_prediction:
                logger.debug(f"Using batch prediction for {len(symbols)} symbols")
                return self._predict_batch_mode(features, symbols, date)
            else:
                logger.debug(f"Using independent prediction for {len(symbols)} symbols")
                return self._predict_independent_mode(features, symbols, date)
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Graceful degradation
            return pd.Series(0.0, index=symbols, name='prediction')

    def _predict_batch_mode(self,
                            features: pd.DataFrame,
                            symbols: List[str],
                            date: Optional[datetime]) -> pd.Series:
        """
        Optimized batch prediction for batch-capable models.
        
        All symbols share the same inputs (e.g., factor values) → one model call.
        
        Examples:
            - FF5: All stocks use same factor values at date t
            - Cross-sectional XGBoost: Predict all stocks at date t together
        """
        logger.info(f"[Batch Mode] Predicting {len(symbols)} symbols in single call")
        
        # Extract shared inputs (e.g., factor values for FF5)
        if isinstance(features.index, pd.MultiIndex):
            # If MultiIndex, extract first symbol's features (all should be same for batch models)
            first_symbol = features.index.get_level_values('symbol')[0]
            X = features.xs(first_symbol, level='symbol').values
            logger.debug(f"Extracted shared features from symbol: {first_symbol}")
        else:
            X = features.values
        
        # Check if model supports symbols parameter
        import inspect
        sig = inspect.signature(self._current_model.predict)
        
        if 'symbols' in sig.parameters:
            # Model can filter to specific symbols (preferred)
            logger.debug(f"Model supports symbols parameter, passing: {symbols}")
            predictions = self._current_model.predict(features, symbols=symbols)
        else:
            # Model returns all trained symbols, need to filter
            logger.debug("Model does not support symbols parameter, will filter results")
            predictions = self._current_model.predict(features)
        
        # Convert to Series
        if isinstance(predictions, np.ndarray):
            if len(predictions) == len(symbols):
                result = pd.Series(predictions, index=symbols, name='prediction')
            else:
                # Length mismatch - model returned all symbols
                logger.warning(
                    f"Prediction length mismatch: got {len(predictions)}, expected {len(symbols)}. "
                    f"Model may not support symbols filtering."
                )
                # Take first N predictions (suboptimal but safe)
                result = pd.Series(predictions[:len(symbols)], index=symbols, name='prediction')
        else:
            result = pd.Series(predictions, index=symbols, name='prediction')
        
        # Log monitoring info
        if self._monitor:
            # Use existing log method if log_batch_prediction doesn't exist
            if hasattr(self._monitor, 'log_batch_prediction'):
                self._monitor.log_batch_prediction(
                    model_id=self.model_id,
                    n_symbols=len(symbols),
                    date=date,
                    prediction_mode='batch'
                )
            else:
                # Fallback to basic logging
                logger.info(f"[Monitor] Batch prediction: {len(symbols)} symbols at {date}")
        
        logger.info(f"[Batch Mode] Successfully predicted {len(result)} symbols")
        return result

    def _predict_independent_mode(self,
                                  features: pd.DataFrame,
                                  symbols: List[str],
                                  date: Optional[datetime]) -> pd.Series:
        """
        Iterative prediction for independent models.
        
        Each symbol needs independent prediction with its own model/features.
        
        Examples:
            - Per-stock XGBoost: Each stock has separate model
            - LSTM: Each stock has separate time-series model
        """
        logger.info(f"[Independent Mode] Predicting {len(symbols)} symbols iteratively")
        
        predictions = {}
        
        for i, symbol in enumerate(symbols, 1):
            try:
                # Extract symbol-specific features
                if isinstance(features.index, pd.MultiIndex):
                    symbol_features = features.xs(symbol, level='symbol')
                else:
                    # Assume features are already for this symbol
                    symbol_features = features
                
                # Convert to ndarray
                X = symbol_features.values
                
                # Predict
                pred = self._current_model.predict(X)
                
                # Extract prediction value (take last if multiple dates)
                predictions[symbol] = float(pred[-1] if len(pred) > 0 else 0.0)
                
                if i % 10 == 0:
                    logger.debug(f"Predicted {i}/{len(symbols)} symbols")
                
            except Exception as e:
                logger.warning(f"Prediction failed for {symbol}: {e}")
                predictions[symbol] = 0.0
        
        logger.info(f"[Independent Mode] Successfully predicted {len(predictions)} symbols")
        return pd.Series(predictions, name='prediction')

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
                    loaded = self.registry.load_model_with_artifacts(effective_model_id)
                    if loaded is None:
                         raise ModelLoadError(f"Could not load model from ID: {effective_model_id}")

                    model, artifacts = loaded
                    # Attach feature pipeline if available
                    if 'feature_pipeline' in artifacts:
                        model.feature_pipeline = artifacts['feature_pipeline']
                        # Mark the model as trained since it's loaded from the registry
                        if hasattr(model, 'is_trained'):
                            model.is_trained = True
                        logger.info(f"Attached 'feature_pipeline' artifact to loaded model '{effective_model_id}'.")

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
        Batch prediction - now delegates to unified predict() interface.
        
        This method is kept for backward compatibility with FF5Strategy.
        """
        return self.predict(features=factors, symbols=symbols, date=date)


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