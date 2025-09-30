"""
Residual Prediction Model

Clean implementation that combines FF5 factor model with ML residual prediction.
This follows the composition pattern - it contains two sub-models:
1. FF5 factor model for baseline returns
2. ML model for residual prediction

The model implements Method A from IPS: Expected Return = Factor Return + ML Residual
"""

import logging
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, Optional, Union
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

from ..base.base_model import BaseModel, ModelStatus, ModelMetadata

logger = logging.getLogger(__name__)


class ResidualPredictionModel(BaseModel):
    """
    Two-stage residual prediction model that combines FF5 factor model with ML residuals.

    Architecture:
    Stage 1: FF5 factor model predicts baseline returns from factor exposures
    Stage 2: ML model predicts residuals from technical features

    Final Prediction = FF5_Prediction + ML_Residual_Prediction

    This model composes two sub-models:
    - ff5_model: FF5RegressionModel for factor-implied returns
    - residual_model: ML model for residual prediction
    """

    def __init__(self, model_type: str = "residual_predictor", config: Optional[Dict[str, Any]] = None):
        """
        Initialize residual prediction model.

        Args:
            model_type: Model identifier
            config: Configuration dictionary with:
                - residual_model_type: 'xgboost', 'lightgbm', 'random_forest', 'ridge' (default: 'xgboost')
                - residual_params: Parameters for residual model (default: {})
                - ff5_config: Configuration for FF5 model (default: {})
        """
        super().__init__(model_type, config)

        # Configuration
        self.residual_model_type = self.config.get('residual_model_type', 'xgboost')
        self.residual_params = self.config.get('residual_params', {})
        self.ff5_config = self.config.get('ff5_config', {})

        # Initialize sub-models
        self.ff5_model = None  # Will be set via set_ff5_model or loaded
        self.residual_model = self._create_residual_model()

        # Track feature separation
        self.factor_features = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
        self.residual_features = []  # Will be set during training

        logger.info(f"Initialized ResidualPredictionModel with {self.residual_model_type} residual model")

    def _create_residual_model(self):
        """Create the residual prediction model based on configuration."""
        if self.residual_model_type == 'xgboost':
            if not XGB_AVAILABLE:
                logger.warning("XGBoost not available, falling back to RandomForest")
                return RandomForestRegressor(**self.residual_params)
            return xgb.XGBRegressor(**self.residual_params)

        elif self.residual_model_type == 'lightgbm':
            if not LGB_AVAILABLE:
                logger.warning("LightGBM not available, falling back to RandomForest")
                return RandomForestRegressor(**self.residual_params)
            return lgb.LGBMRegressor(**self.residual_params)

        elif self.residual_model_type == 'random_forest':
            return RandomForestRegressor(**self.residual_params)

        elif self.residual_model_type == 'ridge':
            return Ridge(alpha=self.residual_params.get('alpha', 1.0))

        else:
            raise ValueError(f"Unsupported residual model type: {self.residual_model_type}")

    def set_ff5_model(self, ff5_model: BaseModel):
        """
        Set the FF5 factor model.

        Args:
            ff5_model: Trained FF5 regression model
        """
        if ff5_model.status != ModelStatus.TRAINED:
            logger.warning("FF5 model is not trained. Consider training it first.")

        self.ff5_model = ff5_model
        logger.info("Set FF5 factor model")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'ResidualPredictionModel':
        """
        Fit the two-stage model.

        Args:
            X: Combined features DataFrame with both factors and technical features
            y: Target returns Series

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data validation fails or FF5 model not available
        """
        try:
            # Validate input data
            self.validate_data(X, y)

            # Check for required factor columns
            missing_factors = set(self.factor_features) - set(X.columns)
            if missing_factors:
                raise ValueError(f"Missing required factor columns: {missing_factors}")

            # Initialize FF5 model if not provided
            if self.ff5_model is None:
                from .ff5_model import FF5RegressionModel
                self.ff5_model = FF5RegressionModel(config=self.ff5_config)

            # Split features into factors and residual features
            X_factors = X[self.factor_features]
            X_residual = X.drop(columns=self.factor_features)
            self.residual_features = list(X_residual.columns)

            # Stage 1: Fit FF5 model on factor returns
            logger.info("Stage 1: Fitting FF5 factor model")
            self.ff5_model.fit(X_factors, y)

            # Stage 2: Calculate residuals and fit residual model
            logger.info("Stage 2: Calculating residuals and fitting residual model")
            factor_predictions = self.ff5_model.predict(X_factors)
            residuals = y - pd.Series(factor_predictions, index=y.index)

            # Fit residual model on technical features
            if len(X_residual) > 0:
                # Align residual data
                aligned_data = pd.concat([residuals, X_residual], axis=1).dropna()
                if len(aligned_data) == 0:
                    logger.warning("No valid data for residual model training")
                    residuals_clean = residuals
                    X_residual_clean = pd.DataFrame(index=residuals.index)
                else:
                    residuals_clean = aligned_data.iloc[:, 0]
                    X_residual_clean = aligned_data.iloc[:, 1:]

                # Fit residual model
                if len(X_residual_clean.columns) > 0:
                    self.residual_model.fit(X_residual_clean, residuals_clean)
                    logger.info(f"Fitted residual model on {len(residuals_clean)} samples")
                else:
                    logger.warning("No residual features available for training")
            else:
                logger.warning("No residual features provided")

            # Update model status and metadata
            self.status = ModelStatus.TRAINED
            self.metadata.training_samples = len(y)
            self.metadata.features = list(X.columns)

            # Store model information
            self.metadata.hyperparameters.update({
                'residual_model_type': self.residual_model_type,
                'residual_params': self.residual_params,
                'factor_features': self.factor_features,
                'residual_features': self.residual_features,
                'ff5_config': self.ff5_config
            })

            logger.info("Successfully fitted ResidualPredictionModel")
            return self

        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"Failed to fit ResidualPredictionModel: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the two-stage model.

        Args:
            X: Combined features DataFrame with both factors and technical features

        Returns:
            Array of predicted returns

        Raises:
            ValueError: If model is not trained or data is invalid
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model must be trained before making predictions")

        if self.ff5_model is None:
            raise ValueError("FF5 model is not set")

        try:
            # Validate input data
            self.validate_data(X)

            # Check for required factor columns
            missing_factors = set(self.factor_features) - set(X.columns)
            if missing_factors:
                raise ValueError(f"Missing required factor columns: {missing_factors}")

            # Split features
            X_factors = X[self.factor_features]
            X_residual = X.drop(columns=self.factor_features)

            # Stage 1: FF5 predictions
            factor_predictions = self.ff5_model.predict(X_factors)

            # Stage 2: Residual predictions (if residual features are available)
            if len(self.residual_features) > 0 and len(X_residual.columns) > 0:
                # Use only the residual features that were used during training
                X_residual_aligned = X_residual[self.residual_features]
                residual_predictions = self.residual_model.predict(X_residual_aligned)
            else:
                residual_predictions = np.zeros(len(factor_predictions))

            # Combine predictions
            final_predictions = factor_predictions + residual_predictions

            logger.debug(f"Made predictions for {len(final_predictions)} samples")
            return final_predictions

        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from both models.

        Returns:
            Dictionary combining factor betas and residual feature importance
        """
        if self.status != ModelStatus.TRAINED:
            return {}

        importance = {}

        # Get factor importance (betas) from FF5 model
        if self.ff5_model:
            factor_importance = self.ff5_model.get_feature_importance()
            for feature, score in factor_importance.items():
                importance[f"factor_{feature}"] = score

        # Get residual feature importance
        if hasattr(self.residual_model, 'feature_importances_'):
            residual_importance = self.residual_model.feature_importances_
            for feature, score in zip(self.residual_features, residual_importance):
                importance[f"residual_{feature}"] = score

        return importance

    def get_ff5_model(self) -> BaseModel:
        """Get the FF5 factor model."""
        return self.ff5_model

    def get_residual_model(self):
        """Get the residual prediction model."""
        return self.residual_model

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model details
        """
        info = {
            'model_type': self.model_type,
            'status': self.status,
            'residual_model_type': self.residual_model_type,
            'training_samples': self.metadata.training_samples,
            'factor_features': self.factor_features,
            'residual_features': self.residual_features,
            'total_features': len(self.factor_features) + len(self.residual_features)
        }

        # Add FF5 model info
        if self.ff5_model:
            info['ff5_model'] = self.ff5_model.get_model_info() if hasattr(self.ff5_model, 'get_model_info') else {}

        # Add residual model info
        if hasattr(self.residual_model, 'get_params'):
            info['residual_model_params'] = self.residual_model.get_params()

        return info

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the residual prediction model to disk.

        Args:
            path: Directory path where model should be saved
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Create model dictionary with all components
        model_dict = {
            'ff5_model': self.ff5_model,
            'residual_model': self.residual_model,
            'factor_features': self.factor_features,
            'residual_features': self.residual_features,
            'residual_model_type': self.residual_model_type
        }

        # Save the model dictionary
        model_path = path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_dict, f)

        # Save metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)

        # Save config
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

        logger.info(f"ResidualPredictionModel saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'ResidualPredictionModel':
        """
        Load a residual prediction model from disk.

        Args:
            path: Directory path where model is saved

        Returns:
            Loaded ResidualPredictionModel instance
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

        # Load the model dictionary
        model_path = path / "model.pkl"
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)

        # Create instance with proper configuration
        instance = cls(config=config)
        instance.metadata = metadata
        instance.status = ModelStatus.TRAINED

        # Restore nested models and attributes
        instance.ff5_model = model_dict['ff5_model']
        instance.residual_model = model_dict['residual_model']
        instance.factor_features = model_dict['factor_features']
        instance.residual_features = model_dict['residual_features']
        instance.residual_model_type = model_dict['residual_model_type']

        # Set _model to reference the dictionary for compatibility
        instance._model = model_dict

        logger.info(f"ResidualPredictionModel loaded from {path}")
        return instance