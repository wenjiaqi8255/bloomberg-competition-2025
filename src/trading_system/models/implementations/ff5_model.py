"""
Fama-French 5-Factor Regression Model

Clean implementation of BaseModel that focuses solely on:
- Factor beta estimation using linear regression
- Factor-implied return prediction
- Beta coefficients as feature importance

This model does NOT handle:
- Data loading (handled by DataProvider)
- Training workflow (handled by Trainer)
- Performance evaluation (handled by PerformanceEvaluator)
- Model persistence (handled by ModelRegistry)
"""

import logging
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from ..base.base_model import BaseModel, ModelStatus, ModelMetadata

logger = logging.getLogger(__name__)


class FF5RegressionModel(BaseModel):
    """
    Fama-French 5-Factor regression model.

    Estimates factor betas for individual stocks and calculates
    factor-implied expected returns using the Fama-French 5-factor model.

    Model Specification:
    R_stock = α + β_MKT * R_MKT + β_SMB * R_SMB + β_HML * R_HML +
              β_RMW * R_RMW + β_CMA * R_CMA + ε

    Where factors are:
    - MKT: Market excess return
    - SMB: Small Minus Big (size)
    - HML: High Minus Low (value)
    - RMW: Robust Minus Weak (profitability)
    - CMA: Conservative Minus Aggressive (investment)
    """

    def __init__(self, model_type: str = "ff5_regression", config: Optional[Dict[str, Any]] = None):
        """
        Initialize FF5 regression model.

        Args:
            model_type: Model identifier
            config: Configuration dictionary with:
                - regularization: 'none' or 'ridge' (default: 'none')
                - alpha: Regularization strength for ridge (default: 1.0)
                - standardize: Whether to standardize factors (default: False)
        """
        super().__init__(model_type, config)

        # Model configuration
        self.regularization = self.config.get('regularization', 'none')
        self.alpha = self.config.get('alpha', 1.0)
        self.standardize = self.config.get('standardize', False)

        # Initialize model components
        if self.regularization == 'ridge':
            # Ensure alpha is always positive
            positive_alpha = max(abs(float(self.alpha)), 1e-6)  # Ensure positive and not zero
            self._model = Ridge(alpha=positive_alpha)
        else:
            self._model = LinearRegression()

        if self.standardize:
            self._scaler = StandardScaler()
        else:
            self._scaler = None

        # Expected factor columns for validation
        self._expected_features = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']

        logger.info(f"Initialized FF5 regression model with {self.regularization} regularization")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FF5RegressionModel':
        """
        Fit the factor model to estimate betas.

        Args:
            X: Factor returns DataFrame with columns ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
            y: Stock excess returns Series aligned with X

        Returns:
            Self for method chaining

        Raises:
            ValueError: If data validation fails
        """
        try:
            # Validate input data
            self.validate_data(X, y)

            # Check for required factor columns
            missing_factors = set(self._expected_features) - set(X.columns)
            if missing_factors:
                raise ValueError(f"Missing required factor columns: {missing_factors}")

            # Align and clean data
            aligned_data = pd.concat([y, X[self._expected_features]], axis=1).dropna()
            if len(aligned_data) == 0:
                raise ValueError("No valid data points after alignment")

            y_clean = aligned_data.iloc[:, 0]
            X_clean = aligned_data.iloc[:, 1:]

            # Standardize factors if requested
            if self.standardize:
                X_clean = pd.DataFrame(
                    self._scaler.fit_transform(X_clean),
                    index=X_clean.index,
                    columns=X_clean.columns
                )

            # Fit the regression model
            self._model.fit(X_clean, y_clean)

            # Update model status and metadata
            self.status = ModelStatus.TRAINED
            self.metadata.training_samples = len(y_clean)
            self.metadata.features = list(X_clean.columns)

            # Store betas as hyperparameters for feature importance
            betas = dict(zip(X_clean.columns, self._model.coef_))
            self.metadata.hyperparameters.update({
                'betas': betas,
                'alpha': self._model.intercept_,
                'regularization': self.regularization,
                'regularization_alpha': self.alpha,
                'standardize': self.standardize
            })

            logger.info(f"Successfully fitted FF5 model on {len(y_clean)} samples")
            logger.info(f"Beta estimates: {betas}")

            # Mark model as trained for registration purposes
            self.is_trained = True

            return self

        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"Failed to fit FF5 regression model: {e}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict factor-implied returns.

        Args:
            X: Factor returns DataFrame with required factor columns

        Returns:
            Array of predicted stock returns

        Raises:
            ValueError: If model is not trained or data is invalid
        """
        logger.info(f"🔍 FF5RegressionModel.predict() starting")
        logger.info(f"Model status: {self.status}")
        logger.info(f"Expected features: {self._expected_features}")
        logger.info(f"Input columns: {list(X.columns)}")
        logger.info(f"Input shape: {X.shape}")
        logger.info(f"Input sample: {X.iloc[-1].to_dict() if not X.empty else 'Empty'}")

        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model must be trained before making predictions")

        try:
            # Validate input data
            self.validate_data(X)

            # Check for required factor columns
            missing_factors = set(self._expected_features) - set(X.columns)
            if missing_factors:
                logger.error(f"❌ Missing required factor columns: {missing_factors}")
                logger.error(f"❌ Available columns: {list(X.columns)}")
                logger.error(f"❌ Expected columns: {self._expected_features}")
                raise ValueError(f"Missing required factor columns: {missing_factors}")

            # Use only expected factor columns in correct order
            X_pred = X[self._expected_features].copy()
            logger.info(f"Using factor columns: {list(X_pred.columns)}")
            logger.info(f"Factor data sample: {X_pred.iloc[-1].to_dict() if not X_pred.empty else 'Empty'}")

            # Standardize if scaler was fitted during training
            if self.standardize and self._scaler is not None:
                logger.info(f"Applying standardization with scaler: {type(self._scaler)}")
                X_pred = pd.DataFrame(
                    self._scaler.transform(X_pred),
                    index=X_pred.index,
                    columns=X_pred.columns
                )
                logger.info(f"Standardized data sample: {X_pred.iloc[-1].to_dict() if not X_pred.empty else 'Empty'}")

            # Make predictions
            logger.info(f"Calling underlying model.predict()")
            logger.info(f"Model type: {type(self._model)}")
            predictions = self._model.predict(X_pred)
            logger.info(f"Raw model predictions: {predictions} (type: {type(predictions)}, shape: {getattr(predictions, 'shape', 'N/A')})")

            if isinstance(predictions, np.ndarray):
                logger.info(f"Prediction stats: min={predictions.min():.6f}, max={predictions.max():.6f}, mean={predictions.mean():.6f}, std={predictions.std():.6f}")
                logger.info(f"Zero predictions count: {(predictions == 0).sum()}/{len(predictions)}")

            logger.info(f"Made predictions for {len(predictions)} samples")
            return predictions

        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get factor betas as feature importance scores.

        Returns:
            Dictionary mapping factor names to their beta coefficients
        """
        if self.status != ModelStatus.TRAINED:
            return {}

        betas = self.metadata.hyperparameters.get('betas', {})
        if not betas:
            # Fallback: get coefficients directly from model
            if hasattr(self._model, 'coef_') and hasattr(self._model, 'feature_names_in_'):
                betas = dict(zip(self._model.feature_names_in_, self._model.coef_))

        return betas

    def get_factor_exposures(self) -> Dict[str, float]:
        """
        Get current factor betas (exposures).

        Returns:
            Dictionary of factor betas
        """
        return self.get_feature_importance()

    def get_alpha(self) -> float:
        """
        Get the model's alpha (intercept).

        Returns:
            Alpha value
        """
        if self.status != ModelStatus.TRAINED:
            return 0.0

        return self.metadata.hyperparameters.get('alpha', 0.0)

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model details
        """
        return {
            'model_type': self.model_type,
            'status': self.status,
            'regularization': self.regularization,
            'regularization_alpha': self.alpha,
            'standardize': self.standardize,
            'factors': self._expected_features,
            'training_samples': self.metadata.training_samples,
            'factor_betas': self.get_feature_importance(),
            'alpha': self.get_alpha()
        }

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the FF5 regression model to disk.

        Args:
            path: Directory path where model should be saved
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Create model dictionary with all components
        model_dict = {
            'model': self._model,
            'scaler': self._scaler,
            'expected_features': self._expected_features,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'standardize': self.standardize
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

        logger.info(f"FF5RegressionModel saved to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FF5RegressionModel':
        """
        Load a FF5 regression model from disk.

        Args:
            path: Directory path where model is saved

        Returns:
            Loaded FF5RegressionModel instance
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

        # Restore model components and attributes
        instance._model = model_dict['model']
        instance._scaler = model_dict['scaler']
        instance._expected_features = model_dict['expected_features']
        instance.regularization = model_dict['regularization']
        instance.alpha = model_dict['alpha']
        instance.standardize = model_dict['standardize']

        logger.info(f"FF5RegressionModel loaded from {path}")
        return instance

    def get_hyperparameter_search_space(self) -> Dict[str, Any]:
        """
        Get hyperparameter search space for FF5 model optimization (MVP - simple dict).

        Returns:
            Simple dictionary defining parameter ranges for optimization
        """
        return {
            'regularization': ['none', 'ridge'],
            'alpha': (0.01, 10.0),
            'standardize': [True, False]
        }

    def get_tunable_hyperparameters(self) -> List[str]:
        """
        Get list of tunable hyperparameter names.

        Returns:
            List of hyperparameter names that can be optimized
        """
        return [
            'regularization',
            'alpha',
            'standardize'
        ]

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.

        Returns:
            Dictionary with model metadata and current configuration
        """
        return {
            'model_type': self.model_type,
            'status': self.status.value,
            'hyperparameters': self.get_model_params(),
            'expected_features': self._expected_features,
            'tunable_parameters': self.get_tunable_hyperparameters(),
            'capabilities': {
                'factor_modeling': True,
                'beta_estimation': True,
                'regularization': self.regularization != 'none',
                'feature_standardization': self.standardize,
                'interpretable': True
            }
        }