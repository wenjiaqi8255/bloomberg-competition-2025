"""
Base Model Abstract Class

This module defines the interface that all ML models must implement.
It follows SOLID principles by separating concerns and providing
a clean contract for model implementations.

Key Features:
- Minimal interface focused on prediction logic
- Built-in metadata management
- Serialization support
- Type safety with comprehensive hints
- Performance monitoring hooks
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple
import pandas as pd
import numpy as np
import pickle
import json

logger = logging.getLogger(__name__)


class ModelStatus:
    """Model status enumeration."""
    UNTRAINED = "untrained"
    TRAINING = "training"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"

    # Class constants for backward compatibility
    TRAINED = "trained"


@dataclass
class ModelMetadata:
    """Metadata associated with a trained model."""
    model_type: str
    version: str
    trained_at: datetime = field(default_factory=datetime.now)
    training_samples: int = 0
    features: list = field(default_factory=list)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    cv_results: Optional[Dict[str, Any]] = None
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'model_type': self.model_type,
            'version': self.version,
            'trained_at': self.trained_at.isoformat(),
            'training_samples': self.training_samples,
            'features': self.features,
            'hyperparameters': self.hyperparameters,
            'performance_metrics': self.performance_metrics,
            'cv_results': self.cv_results,
            'tags': self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary."""
        data = data.copy()
        data['trained_at'] = datetime.fromisoformat(data['trained_at'])
        return cls(**data)


class BaseModel(ABC):
    """
    Abstract base class for all ML models in the trading system.

    This class defines the minimal interface that all models must implement.
    It focuses solely on prediction logic and does NOT include:
    - Training workflows (handled by trainers)
    - Data preparation (handled by feature engineers)
    - Performance evaluation (handled by evaluators)
    - Model storage (handled by registry)

    Design Principles:
    - Single Responsibility: Only handles model logic
    - Interface Segregation: Minimal, focused interface
    - Dependency Inversion: Strategies depend on this abstraction
    """

    # Class constants for compatibility
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"

    def __init__(self, model_type: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the base model.

        Args:
            model_type: String identifier for the model type
            config: Optional configuration dictionary
        """
        self.model_type = model_type
        self.config = config or {}
        self.status = "untrained"
        self.metadata = ModelMetadata(
            model_type=model_type,
            version="1.0.0",
            hyperparameters=config or {}
        )
        self._model = None  # Concrete model implementation

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'BaseModel':
        """
        Train the model on the provided data.

        This method should only handle the actual model fitting.
        Data validation, preprocessing, and performance evaluation
        should be handled by other components.

        Args:
            X: Feature DataFrame
            y: Target Series

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data is invalid
            RuntimeError: If training fails
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature DataFrame with same columns as training data

        Returns:
            Array of predictions

        Raises:
            ValueError: If model is not trained or data is invalid
        """
        pass

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make probability predictions (for classification models).

        Default implementation raises NotImplementedError.
        Override in classification models.

        Args:
            X: Feature DataFrame

        Returns:
            Array of class probabilities

        Raises:
            NotImplementedError: If model doesn't support probability prediction
        """
        raise NotImplementedError("This model doesn't support probability prediction")

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.

        Default implementation returns empty dict.
        Override in models that support feature importance.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        return {}

    def get_model_params(self) -> Dict[str, Any]:
        """
        Get model parameters.

        Returns:
            Dictionary of model parameters
        """
        if hasattr(self._model, 'get_params'):
            return self._model.get_params()
        return {}

    def set_model_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.

        Args:
            **params: Parameters to set

        Returns:
            Self for method chaining
        """
        if hasattr(self._model, 'set_params'):
            self._model.set_params(**params)
        return self

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the model to disk.

        Args:
            path: Path to save the model

        Raises:
            IOError: If save fails
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save the model
        model_path = path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self._model, f)

        # Save metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        # Save config
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'BaseModel':
        """
        Load a model from disk.

        Args:
            path: Path to the saved model

        Returns:
            Loaded model instance

        Raises:
            IOError: If load fails
            ValueError: If model is corrupted
        """
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")

        # Load metadata
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            raise ValueError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata.from_dict(metadata_dict)

        # Load config
        config_path = path / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)

        # Create model instance (this will be overridden by subclasses)
        model = cls.__new__(cls)
        model.model_type = metadata.model_type
        model.config = config
        model.status = ModelStatus.TRAINED
        model.metadata = metadata

        # Load the actual model
        model_path = path / "model.pkl"
        with open(model_path, 'rb') as f:
            model._model = pickle.load(f)

        logger.info(f"Model loaded from {path}")
        return model

    def validate_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Validate input data.

        Args:
            X: Feature DataFrame
            y: Optional target Series

        Raises:
            ValueError: If data is invalid
        """
        # Support both DataFrame and numpy array inputs for different model types
        if isinstance(X, pd.DataFrame):
            if X.empty:
                raise ValueError("X cannot be empty")
        elif isinstance(X, np.ndarray):
            if X.size == 0:
                raise ValueError("X cannot be empty")
        else:
            # For neural network models that can accept numpy arrays
            if len(X) == 0:
                raise ValueError("X cannot be empty")

        # Additional DataFrame-specific validations
        if isinstance(X, pd.DataFrame) and X.empty:
            raise ValueError("X cannot be empty")

        # DataFrame-specific feature validation
        if isinstance(X, pd.DataFrame) and self.status == ModelStatus.TRAINED and hasattr(self, '_expected_features'):
            missing_features = set(self._expected_features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

        if y is not None:
            # Support both Series and numpy array for y
            if isinstance(y, pd.Series):
                pass  # Series is valid
            elif isinstance(y, np.ndarray):
                pass  # numpy array is valid for neural networks
            else:
                if not hasattr(y, '__len__'):
                    raise ValueError("y must be a pandas Series or numpy array")

            if len(X) != len(y):
                raise ValueError("X and y must have the same length")

    def update_metadata(self, **kwargs) -> None:
        """
        Update model metadata.

        Args:
            **kwargs: Metadata fields to update
        """
        for key, value in kwargs.items():
            if hasattr(self.metadata, key):
                setattr(self.metadata, key, value)
            else:
                self.metadata.tags[key] = str(value)

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(type={self.model_type}, status={self.status})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"{self.__class__.__name__}(type={self.model_type}, "
                f"status={self.status}, version={self.metadata.version})")