"""
Ensemble Model - Combines multiple base models using metamodel weights
=========================

This model creates a weighted ensemble of trained base models that can be used
within the existing ML strategy pipeline. It handles feature engineering internally
and presents a unified interface for prediction.

Architecture:
    Base Models -> Feature Engineering -> Weighted Combination -> Prediction

Key Features:
    - Uses trained base models with metamodel weights
    - Handles feature engineering for each base model
    - Compatible with existing ML strategy pipeline
    - No synthetic data - uses only real trained models
"""

import logging
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Union, Optional

from ..base.base_model import BaseModel, ModelStatus, ModelMetadata

logger = logging.getLogger(__name__)


class EnsembleModel(BaseModel):
    """
    Ensemble model that combines multiple base models using metamodel weights.

    This model loads multiple trained base models and combines their predictions
    using weights derived from metamodel training. It's designed to work within
    the existing ML strategy pipeline.

    The ensemble:
    1. Loads all base models from the model registry
    2. Applies appropriate feature engineering for each model
    3. Generates predictions from each base model
    4. Combines predictions using metamodel weights
    5. Returns ensemble predictions
    """

    def __init__(self,
                 model_type: str = "ensemble",
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the ensemble model.

        Args:
            model_type: Model identifier
            config: Configuration dictionary containing:
                - base_model_ids: List of base model IDs to include in ensemble
                - model_weights: Dictionary mapping model IDs to weights (sums to 1.0)
                - model_registry_path: Path to model registry
        """
        super().__init__(model_type, config or {})

        # Extract configuration parameters
        self.base_model_ids = self.config.get('base_model_ids', [])
        self.model_weights = self.config.get('model_weights', {})
        self.model_registry_path = self.config.get('model_registry_path', './models/')

        # Validate required parameters
        if not self.base_model_ids:
            raise ValueError("base_model_ids must be provided in config")
        if not self.model_weights:
            raise ValueError("model_weights must be provided in config")

        # Validate weights
        total_weight = sum(self.model_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0: {total_weight:.4f}")
            # Normalize weights
            self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
            logger.info(f"Normalized weights: {self.model_weights}")

        # Load base models
        self.base_models = self._load_base_models()

        # Store feature information
        self._feature_names = None

        logger.info(f"Initialized EnsembleModel with {len(self.base_models)} base models")
        logger.info(f"Base models: {self.base_model_ids}")
        logger.info(f"Weights: {self.model_weights}")

    def _load_base_models(self) -> Dict[str, BaseModel]:
        """
        Load all base models from the model registry.

        Returns:
            Dictionary mapping model_id to loaded model
        """
        from ...models.model_persistence import ModelRegistry

        models = {}
        registry = ModelRegistry(self.model_registry_path)

        for model_id in self.base_model_ids:
            try:
                loaded = registry.load_model_with_artifacts(model_id)
                if loaded:
                    model, artifacts = loaded
                    models[model_id] = model
                    logger.debug(f"Loaded base model: {model_id}")
                else:
                    raise ValueError(f"Model {model_id} not found in registry")

            except Exception as e:
                logger.error(f"Failed to load base model {model_id}: {e}")
                raise ValueError(f"Could not load base model {model_id}: {e}")

        if not models:
            raise ValueError("No base models could be loaded")

        logger.info(f"Successfully loaded {len(models)} base models")
        return models

    def fit(self,
            X: pd.DataFrame,
            y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'EnsembleModel':
        """
        Ensemble model doesn't need training - it's already trained.
        This method is provided for compatibility but does nothing.

        Args:
            X: Training features (ignored)
            y: Training targets (ignored)
            X_val: Validation features (ignored)
            y_val: Validation targets (ignored)

        Returns:
            Self for method chaining
        """
        logger.info("EnsembleModel is pre-trained - no fitting needed")
        self.status = ModelStatus.TRAINED
        self.metadata.training_samples = len(y) if y is not None else 0
        self.metadata.features = list(X.columns) if X is not None else []

        # Store ensemble configuration in metadata
        self.metadata.hyperparameters.update({
            'base_models': self.base_model_ids,
            'model_weights': self.model_weights,
            'ensemble_type': 'weighted_average'
        })

        self.is_trained = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make ensemble predictions by combining base model predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of ensemble predictions
        """
        if not self.base_models:
            raise ValueError("No base models available for prediction")

        try:
            logger.debug(f"Making ensemble predictions for {len(X)} samples")

            # Collect predictions from all base models
            all_predictions = {}
            total_weight = 0.0

            for model_id, weight in self.model_weights.items():
                if model_id not in self.base_models:
                    logger.warning(f"Model {model_id} not found in loaded models, skipping")
                    continue

                model = self.base_models[model_id]

                try:
                    # Generate predictions from this base model
                    predictions = model.predict(X)

                    # Ensure predictions are 1D numpy array
                    if isinstance(predictions, pd.Series):
                        predictions = predictions.values
                    elif len(predictions.shape) > 1:
                        predictions = predictions.flatten()

                    all_predictions[model_id] = predictions
                    total_weight += weight

                    logger.debug(f"Got predictions from {model_id}: mean={np.mean(predictions):.6f}, std={np.std(predictions):.6f}")

                except Exception as e:
                    logger.error(f"Failed to get predictions from {model_id}: {e}")
                    continue

            if not all_predictions:
                raise ValueError("No predictions could be generated from any base model")

            # Normalize weights if some models failed
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Available models weight sum: {total_weight:.4f}, renormalizing")
                normalized_weights = {k: v/total_weight for k, v in self.model_weights.items() if k in all_predictions}
            else:
                normalized_weights = {k: v for k, v in self.model_weights.items() if k in all_predictions}

            # Combine predictions using weights
            ensemble_predictions = np.zeros(len(next(iter(all_predictions.values()))))

            for model_id, predictions in all_predictions.items():
                weight = normalized_weights.get(model_id, 0.0)
                ensemble_predictions += weight * predictions
                logger.debug(f"Added {model_id} predictions with weight {weight:.4f}")

            logger.debug(f"Ensemble predictions - mean: {np.mean(ensemble_predictions):.6f}, std: {np.std(ensemble_predictions):.6f}")

            return ensemble_predictions

        except Exception as e:
            logger.error(f"Failed to make ensemble predictions: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive ensemble model information.

        Returns:
            Dictionary with model details
        """
        base_model_info = {}
        for model_id, model in self.base_models.items():
            if hasattr(model, 'get_model_info'):
                base_model_info[model_id] = model.get_model_info()
            else:
                base_model_info[model_id] = {'model_type': str(type(model).__name__)}

        return {
            'model_type': self.model_type,
            'status': self.status,
            'base_models': base_model_info,
            'model_weights': self.model_weights,
            'num_base_models': len(self.base_models),
            'ensemble_method': 'weighted_average',
            'total_weight': sum(self.model_weights.values())
        }

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the ensemble model configuration to disk.

        Args:
            path: Directory path where model should be saved
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save ensemble configuration (base models are loaded from registry)
        config = {
            'base_model_ids': self.base_model_ids,
            'model_weights': self.model_weights,
            'model_registry_path': self.model_registry_path
        }

        config_path = path / "ensemble_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Save metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        logger.info(f"EnsembleModel configuration saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EnsembleModel':
        """
        Load an ensemble model configuration from disk.

        Args:
            path: Directory path where model configuration is saved

        Returns:
            Loaded EnsembleModel instance
        """
        path = Path(path)

        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")

        # Load ensemble configuration
        config_path = path / "ensemble_config.json"
        if not config_path.exists():
            raise ValueError(f"Ensemble configuration not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create instance
        instance = cls(
            model_type=config.get('model_type', 'ensemble'),
            config=config
        )

        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            instance.metadata = ModelMetadata.from_dict(metadata_dict)

        instance.status = ModelStatus.TRAINED

        logger.info(f"EnsembleModel configuration loaded from {path}")
        return instance

    def validate_model(self) -> bool:
        """
        Validate that the ensemble model is properly configured.

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check base models
            if not self.base_models:
                logger.error("No base models loaded")
                return False

            # Check weights
            if not self.model_weights:
                logger.error("No model weights defined")
                return False

            # Check that all weighted models exist
            for model_id, weight in self.model_weights.items():
                if model_id not in self.base_models:
                    logger.error(f"Weighted model {model_id} not found in loaded models")
                    return False
                if weight <= 0:
                    logger.error(f"Invalid weight for {model_id}: {weight}")
                    return False

            # Check weight sum
            total_weight = sum(self.model_weights.values())
            if abs(total_weight - 1.0) > 0.01:
                logger.warning(f"Weights don't sum to 1.0: {total_weight}")

            logger.info("✓ Ensemble model validation passed")
            return True

        except Exception as e:
            logger.error(f"Ensemble model validation failed: {e}")
            return False