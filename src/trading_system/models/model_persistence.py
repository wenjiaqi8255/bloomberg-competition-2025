"""
Model Persistence - Storing and Retrieving Trained Models and Artifacts

This module provides a robust Model Registry for saving, loading, and managing
trained models and their associated artifacts, such as feature pipelines.
"""

import os
import json
import joblib
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

from .base.base_model import BaseModel

class ModelRegistry:
    """
    Manages the persistence of trained models and their artifacts.
    
    Each model version is stored in its own directory, containing the model,
    metadata, and any associated artifacts.
    """

    def __init__(self, storage_path: str = "./models/saved"):
        """
        Initialize the Model Registry.

        Args:
            storage_path: Root directory to store all model versions.
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save_model_with_artifacts(
        self, 
        model: BaseModel, 
        model_name: str,
        artifacts: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Saves a trained model along with its artifacts and metadata.

        Args:
            model: The trained model object to save.
            model_name: A descriptive name for the model.
            artifacts: A dictionary of objects to save alongside the model,
                       e.g., {'feature_pipeline': feature_pipeline_object}.
            tags: A dictionary of tags/metadata for this model version.

        Returns:
            The unique model ID for this version.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_id = f"{model_name}_{timestamp}"
            model_dir = self.storage_path / model_id
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save main model
            model_path = model_dir / "model.joblib"
            joblib.dump(model, model_path)

            # Save artifacts
            artifact_paths = {}
            if artifacts:
                artifacts_dir = model_dir / "artifacts"
                artifacts_dir.mkdir(exist_ok=True)
                for name, obj in artifacts.items():
                    artifact_path = artifacts_dir / f"{name}.joblib"
                    joblib.dump(obj, artifact_path)
                    artifact_paths[name] = str(artifact_path.relative_to(self.storage_path))

            # Create and save metadata
            metadata = {
                "model_id": model_id,
                "model_name": model_name,
                "model_type": getattr(model, 'model_type', 'unknown'),
                "created_at": datetime.now().isoformat(),
                "model_path": str(model_path.relative_to(self.storage_path)),
                "artifact_paths": artifact_paths,
                "tags": tags or {}
            }
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

            print(f"✅ Model and artifacts saved successfully. Model ID: {model_id}")
            return model_id

        except Exception as e:
            print(f"❌ Failed to save model {model_name}: {e}")
            raise

    def load_model_with_artifacts(self, model_id: str) -> Optional[Tuple[BaseModel, Dict[str, Any]]]:
        """
        Loads a model and its associated artifacts.

        Args:
            model_id: The ID of the model version to load.

        Returns:
            A tuple containing (loaded_model, artifacts_dictionary), or None if not found.
        """
        try:
            metadata = self.get_model_metadata(model_id)
            if not metadata:
                print(f"⚠️ Model metadata for '{model_id}' not found.")
                return None

            # Load main model
            model_path = self.storage_path / metadata["model_path"]
            model = joblib.load(model_path)

            # Load artifacts
            artifacts = {}
            for name, rel_path in metadata.get("artifact_paths", {}).items():
                artifact_path = self.storage_path / rel_path
                artifacts[name] = joblib.load(artifact_path)
            
            print(f"✅ Model and {len(artifacts)} artifacts loaded for ID: {model_id}")
            return model, artifacts

        except Exception as e:
            print(f"❌ Failed to load model or artifacts for {model_id}: {e}")
            return None

    def get_model_metadata(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves metadata for a specific model version.
        """
        metadata_path = self.storage_path / model_id / "metadata.json"
        if not metadata_path.exists():
            return None
        with open(metadata_path, 'r') as f:
            return json.load(f)

    def list_models(self) -> list:
        """
        Lists metadata for all saved model versions.
        """
        models = []
        for model_dir in self.storage_path.iterdir():
            if model_dir.is_dir():
                metadata = self.get_model_metadata(model_dir.name)
                if metadata:
                    models.append(metadata)
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return models

    # Note: The original `save_model` and `load_model` are deprecated in favor of
    # the new `..._with_artifacts` methods to enforce traceability.
    # The other methods like `delete_model` and `get_storage_info` would also
    # need to be updated to work with the new directory structure.
    # I am omitting them for now to focus on the primary goal.