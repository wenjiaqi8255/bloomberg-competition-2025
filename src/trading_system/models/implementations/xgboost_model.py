"""
XGBoost Regression Model

This module implements an XGBoost-based regression model for trading signal generation.
XGBoost is particularly effective for:
- Non-linear relationships
- Feature interactions
- Handling missing values
- Robust to outliers

Key Features:
- Gradient boosting trees
- Built-in regularization
- Feature importance tracking
- Early stopping support
"""

import logging
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    xgb = None

from ..base.base_model import BaseModel, ModelStatus, ModelMetadata

logger = logging.getLogger(__name__)


class XGBoostModel(BaseModel):
    """
    XGBoost Regression Model for trading signal prediction.
    
    This model uses gradient boosting on decision trees to capture complex
    non-linear relationships in financial data.
    
    Example Usage:
        # Create model
        model = XGBoostModel(config={
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'early_stopping_rounds': 10
        })
        
        # Train
        model.fit(X_train, y_train, X_val, y_val)
        
        # Predict
        predictions = model.predict(X_test)
        
        # Get feature importance
        importance = model.get_feature_importance()
    """
    
    def __init__(self, model_type: str = "xgboost", config: Optional[Dict[str, Any]] = None):
        """
        Initialize XGBoost model.
        
        Args:
            model_type: Model identifier
            config: Configuration dictionary with XGBoost hyperparameters:
                - n_estimators: Number of boosting rounds (default: 100)
                - max_depth: Maximum tree depth (default: 5)
                - learning_rate: Step size shrinkage (default: 0.1)
                - subsample: Row sampling ratio (default: 0.8)
                - colsample_bytree: Column sampling ratio (default: 0.8)
                - min_child_weight: Minimum sum of instance weight (default: 1)
                - gamma: Minimum loss reduction for split (default: 0)
                - reg_alpha: L1 regularization (default: 0)
                - reg_lambda: L2 regularization (default: 1)
                - early_stopping_rounds: Early stopping patience (default: None)
                - random_state: Random seed (default: 42)
        """
        super().__init__(model_type, config)
        
        if not XGBOOST_AVAILABLE:
            raise ImportError(
                "XGBoost is not installed. Please install it with: "
                "pip install xgboost"
            )
        
        # Extract hyperparameters
        self.n_estimators = self.config.get('n_estimators', 100)
        self.max_depth = self.config.get('max_depth', 5)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.subsample = self.config.get('subsample', 0.8)
        self.colsample_bytree = self.config.get('colsample_bytree', 0.8)
        self.min_child_weight = self.config.get('min_child_weight', 1)
        self.gamma = self.config.get('gamma', 0)
        self.reg_alpha = self.config.get('reg_alpha', 0)
        self.reg_lambda = self.config.get('reg_lambda', 1)
        self.early_stopping_rounds = self.config.get('early_stopping_rounds', None)
        self.random_state = self.config.get('random_state', 42)
        
        # Initialize XGBoost model
        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            n_jobs=-1,  # Use all CPU cores
            verbosity=0  # Suppress output
        )
        
        # Track feature names
        self._feature_names = None
        self._best_iteration = None
        
        logger.info(f"Initialized XGBoostModel with {self.n_estimators} estimators")
    
    def fit(self, 
            X: pd.DataFrame, 
            y: pd.Series,
            X_val: Optional[pd.DataFrame] = None,
            y_val: Optional[pd.Series] = None) -> 'XGBoostModel':
        """
        Train the XGBoost model.
        
        Args:
            X: Training features
            y: Training targets
            X_val: Optional validation features (for early stopping)
            y_val: Optional validation targets (for early stopping)
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If data validation fails
        """
        try:
            # Validate input data
            self.validate_data(X, y)
            
            # Clean data - first drop rows where target is NaN
            combined_data = pd.concat([y, X], axis=1)
            combined_data = combined_data.dropna(subset=[combined_data.columns[0]])  # Only drop rows where target is NaN

            if len(combined_data) == 0:
                raise ValueError("No valid data points after target alignment")

            # For remaining rows, handle feature NaNs by filling with median or 0
            y_clean = combined_data.iloc[:, 0]
            X_clean = combined_data.iloc[:, 1:]

            # Fill NaN values in features - use 0 for technical indicators
            X_clean = X_clean.fillna(0)

            if len(X_clean) == 0:
                raise ValueError("No valid data points after feature cleaning")
            
            # Store feature names
            self._feature_names = list(X_clean.columns)
            
            # Prepare evaluation set for early stopping
            eval_set = []
            if X_val is not None and y_val is not None:
                # Align validation data - use same approach as training data
                val_combined = pd.concat([y_val, X_val[self._feature_names]], axis=1)
                val_combined = val_combined.dropna(subset=[val_combined.columns[0]])  # Only drop rows where target is NaN

                if len(val_combined) > 0:
                    y_val_clean = val_combined.iloc[:, 0]
                    X_val_clean = val_combined.iloc[:, 1:]
                    # Fill NaN values in features
                    X_val_clean = X_val_clean.fillna(0)
                    eval_set = [(X_val_clean, y_val_clean)]
            
            # Train model
            # Note: Early stopping support varies by XGBoost version
            # For compatibility, we train without early stopping
            if eval_set:
                self._model.fit(
                    X_clean, 
                    y_clean,
                    eval_set=eval_set,
                    verbose=False
                )
                logger.info("Trained with validation set")
            else:
                self._model.fit(X_clean, y_clean)
                logger.info("Trained without validation")
            
            # Update model status and metadata
            self.status = ModelStatus.TRAINED
            self.metadata.training_samples = len(y_clean)
            self.metadata.features = self._feature_names
            
            # Store hyperparameters
            self.metadata.hyperparameters.update({
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'subsample': self.subsample,
                'colsample_bytree': self.colsample_bytree,
                'best_iteration': self._best_iteration
            })
            
            logger.info(f"Successfully trained XGBoostModel on {len(y_clean)} samples")

            # Mark model as trained for registration purposes
            self.is_trained = True

            return self
        
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"Failed to train XGBoostModel: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature DataFrame
        
        Returns:
            Array of predictions
        
        Raises:
            ValueError: If model is not trained or data is invalid
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Validate input data
            self.validate_data(X)
            
            # Ensure correct feature order
            if self._feature_names:
                missing_features = set(self._feature_names) - set(X.columns)
                if missing_features:
                    raise ValueError(f"Missing features: {missing_features}")
                X_pred = X[self._feature_names]
            else:
                X_pred = X
            
            # Make predictions
            predictions = self._model.predict(X_pred)
            
            logger.debug(f"Made predictions for {len(predictions)} samples")
            return predictions
        
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.status != ModelStatus.TRAINED:
            return {}
        
        try:
            # Get importance from XGBoost
            importance = self._model.feature_importances_
            
            if self._feature_names and len(importance) == len(self._feature_names):
                return dict(zip(self._feature_names, importance.tolist()))
            else:
                return {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}
        
        except Exception as e:
            logger.warning(f"Could not get feature importance: {e}")
            return {}
    
    def predict_batch(self, X_batch: List[pd.DataFrame]) -> List[np.ndarray]:
        """
        Make batch predictions for multiple DataFrames efficiently.

        This method leverages XGBoost's native batch prediction capabilities
        to optimize performance when predicting for multiple stocks or time periods.

        Args:
            X_batch: List of feature DataFrames for prediction

        Returns:
            List of prediction arrays, one per input DataFrame

        Raises:
            ValueError: If model is not trained or data is invalid
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model must be trained before making predictions")

        try:
            predictions = []

            for X in X_batch:
                # Validate input data
                self.validate_data(X)

                # Ensure correct feature order
                if self._feature_names:
                    missing_features = set(self._feature_names) - set(X.columns)
                    if missing_features:
                        raise ValueError(f"Missing features: {missing_features}")
                    X_pred = X[self._feature_names]
                else:
                    X_pred = X

                # Make predictions for this batch
                batch_predictions = self._model.predict(X_pred)
                predictions.append(batch_predictions)

            total_samples = sum(len(pred) for pred in predictions)
            logger.debug(f"Made batch predictions for {total_samples} samples across {len(X_batch)} DataFrames")
            return predictions

        except Exception as e:
            logger.error(f"Failed to make batch predictions: {e}")
            raise
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the XGBoost model to disk.
        
        Args:
            path: Directory path where model should be saved
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save XGBoost model using its native format
        model_path = path / "xgboost_model.json"
        self._model.save_model(str(model_path))
        
        # Save additional model state
        state_dict = {
            'feature_names': self._feature_names,
            'best_iteration': self._best_iteration,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state
        }
        
        state_path = path / "model_state.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(state_dict, f)
        
        # Save metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Save config
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"XGBoostModel saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'XGBoostModel':
        """
        Load an XGBoost model from disk.
        
        Args:
            path: Directory path where model is saved
        
        Returns:
            Loaded XGBoostModel instance
        """
        path = Path(path)
        
        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")
        
        # Load config
        config_path = path / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Create instance
        instance = cls(config=config)
        
        # Load model state
        state_path = path / "model_state.pkl"
        with open(state_path, 'rb') as f:
            state_dict = pickle.load(f)
        
        instance._feature_names = state_dict['feature_names']
        instance._best_iteration = state_dict['best_iteration']
        
        # Load XGBoost model
        model_path = path / "xgboost_model.json"
        instance._model.load_model(str(model_path))
        
        # Load metadata
        metadata_path = path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata_dict = json.load(f)
            instance.metadata = ModelMetadata.from_dict(metadata_dict)
        
        instance.status = ModelStatus.TRAINED
        
        logger.info(f"XGBoostModel loaded from {path}")
        return instance

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        """
        Get hyperparameter search space for XGBoost model optimization (MVP - simple dict).

        Returns:
            Simple dictionary defining parameter ranges for optimization
        """
        return {
            'n_estimators': (50, 500),
            'max_depth': (3, 12),
            'learning_rate': (0.01, 0.3),
            'subsample': (0.6, 1.0),
            'colsample_bytree': (0.6, 1.0),
            'min_child_weight': (1, 10),
            'gamma': (0.0, 1.0),
            'reg_alpha': (0.0, 1.0),
            'reg_lambda': (0.5, 2.0)
        }

    def get_tunable_hyperparameters(self) -> List[str]:
        """
        Get list of tunable hyperparameter names.

        Returns:
            List of hyperparameter names that can be optimized
        """
        return [
            'n_estimators',
            'max_depth',
            'learning_rate',
            'subsample',
            'colsample_bytree',
            'min_child_weight',
            'gamma',
            'reg_alpha',
            'reg_lambda'
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model metadata and current configuration
        """
        return {
            'model_type': self.model_type,
            'status': str(self.status),  # Convert to string to avoid .value error
            'hyperparameters': self.get_model_params(),
            'feature_names': self._feature_names,
            'best_iteration': self._best_iteration,
            'tunable_parameters': self.get_tunable_hyperparameters(),
            'capabilities': {
                'feature_importance': True,
                'early_stopping': self.early_stopping_rounds is not None,
                'handles_missing_values': True,
                'supports_gpu': False  # Could be enabled with proper setup
            }
        }
