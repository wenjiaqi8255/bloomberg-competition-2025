"""
Model Factory and Registry

This module provides factory methods for creating models and a registry
for managing model instances. It supports dependency injection and
enables easy swapping of model implementations.

Key Features:
- Factory pattern for model creation
- Registry for managing model versions
- Dependency injection support
- Configuration-based instantiation
"""

import logging
from pathlib import Path
from typing import Dict, Any, Type, Optional, Union
from dataclasses import dataclass

from .base_model import BaseModel, ModelMetadata

logger = logging.getLogger(__name__)


@dataclass
class ModelRegistration:
    """Model registration information."""
    model_class: Type[BaseModel]
    description: str
    default_config: Dict[str, Any]


class ModelFactory:
    """
    Factory for creating ML model instances.

    This factory handles the creation of model instances based on
    configuration and manages model registration.
    """

    _registry: Dict[str, ModelRegistration] = {}

    @classmethod
    def register(cls,
                 model_type: str,
                 model_class: Type[BaseModel],
                 description: str = "",
                 default_config: Optional[Dict[str, Any]] = None) -> None:
        """
        Register a model class with the factory.

        Args:
            model_type: String identifier for the model
            model_class: Model class (must inherit from BaseModel)
            description: Human-readable description
            default_config: Default configuration parameters
        """
        if not issubclass(model_class, BaseModel):
            raise ValueError(f"Model class {model_class} must inherit from BaseModel")

        cls._registry[model_type] = ModelRegistration(
            model_class=model_class,
            description=description,
            default_config=default_config or {}
        )

        logger.info(f"Registered model: {model_type} -> {model_class.__name__}")

    @classmethod
    def create(cls,
               model_type: str,
               config: Optional[Dict[str, Any]] = None,
               **kwargs) -> BaseModel:
        """
        Create a model instance.

        Args:
            model_type: Type of model to create
            config: Configuration dictionary
            **kwargs: Additional arguments for model constructor

        Returns:
            Model instance

        Raises:
            ValueError: If model type is not registered
        """
        if model_type not in cls._registry:
            available_models = list(cls._registry.keys())
            raise ValueError(f"Unknown model type: {model_type}. Available: {available_models}")

        registration = cls._registry[model_type]

        # Merge default config with provided config
        full_config = registration.default_config.copy()
        if config:
            full_config.update(config)

        # Merge with kwargs
        full_config.update(kwargs)

        try:
            model = registration.model_class(model_type=model_type, config=full_config)
            logger.info(f"Created model: {model_type}")
            return model
        except Exception as e:
            logger.error(f"Failed to create model {model_type}: {e}")
            raise

    @classmethod
    def list_models(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all registered models.

        Returns:
            Dictionary of model information
        """
        return {
            model_type: {
                'class': reg.model_class.__name__,
                'description': reg.description,
                'default_config': reg.default_config
            }
            for model_type, reg in cls._registry.items()
        }

    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """
        Check if a model type is registered.

        Args:
            model_type: Model type to check

        Returns:
            True if registered
        """
        return model_type in cls._registry


class ModelTypeRegistry:
    """
    Registry for managing trained model instances and versions.

    This registry handles the storage and retrieval of trained models,
    supporting version management and deployment workflows.
    """

    def __init__(self, base_path: Union[str, Path] = "./models/"):
        """
        Initialize the model registry.

        Args:
            base_path: Base path for storing models
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save_model(self,
                   model: BaseModel,
                   model_name: str,
                   version: Optional[str] = None,
                   tags: Optional[Dict[str, str]] = None) -> str:
        """
        Save a trained model to the registry.

        Args:
            model: Trained model instance
            model_name: Name for the model
            version: Optional version string
            tags: Optional tags for the model

        Returns:
            Model ID (name_version)

        Raises:
            ValueError: If model is not trained
        """
        if model.status != BaseModel.TRAINED:
            raise ValueError("Only trained models can be saved")

        if version is None:
            version = model.metadata.version

        model_id = f"{model_name}_v{version}"

        # Update metadata
        if tags:
            model.update_metadata(**tags)

        # Save model
        model_path = self.base_path / model_id
        model.save(model_path)

        logger.info(f"Saved model {model_id} to registry")
        return model_id

  
    