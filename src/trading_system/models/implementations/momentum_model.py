"""
Momentum Ranking Model

This module implements a momentum-based ranking model for dual momentum strategies.
The model can operate in two modes:
1. Rule-based: Uses fixed momentum combination weights
2. Trainable: Learns optimal momentum combination weights from data

Key Features:
- Multi-period momentum combination (21d, 63d, 252d)
- Absolute momentum filtering (positive returns only)
- Relative momentum ranking (top N performers)
- Optional training to optimize weights
"""

import logging
import numpy as np
import pandas as pd
import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, Union
from sklearn.linear_model import LinearRegression

from ..base.base_model import BaseModel, ModelStatus, ModelMetadata

logger = logging.getLogger(__name__)


class MomentumRankingModel(BaseModel):
    """
    Momentum Ranking Model for Dual Momentum Strategy.
    
    This model implements momentum-based asset ranking and selection.
    It combines multiple momentum periods (short, medium, long-term) to
    generate a composite momentum score for ranking assets.
    
    Model can operate in two modes:
    - rule_based: Uses predefined weights (no training needed)
    - trainable: Learns optimal weights from historical data
    
    Example Usage:
        # Rule-based (no training needed)
        model = MomentumRankingModel(config={
            'mode': 'rule_based',
            'top_n': 5,
            'min_momentum': 0.0,
            'momentum_weights': [0.3, 0.3, 0.4]  # 21d, 63d, 252d
        })
        model.fit(X, y)  # Just sets status to TRAINED
        scores = model.predict(features)
        
        # Trainable (learns weights)
        model = MomentumRankingModel(config={
            'mode': 'trainable',
            'top_n': 5,
            'min_momentum': 0.0
        })
        model.fit(X_train, y_train)  # Learns optimal weights
        scores = model.predict(X_test)
    """
    
    def __init__(self, model_type: str = "momentum_ranking", config: Optional[Dict[str, Any]] = None):
        """
        Initialize Momentum Ranking Model.
        
        Args:
            model_type: Model identifier
            config: Configuration dictionary with:
                - mode: 'rule_based' or 'trainable' (default: 'rule_based')
                - top_n: Number of top performers to select (default: 5)
                - min_momentum: Minimum momentum threshold for filtering (default: 0.0)
                - momentum_weights: Weights for [21d, 63d, 252d] momentum (default: [0.3, 0.3, 0.4])
                - momentum_periods: List of momentum periods to use (default: [21, 63, 252])
        """
        super().__init__(model_type, config)
        
        # Model configuration
        self.mode = self.config.get('mode', 'rule_based')
        self.top_n = self.config.get('top_n', 5)
        self.min_momentum = self.config.get('min_momentum', 0.0)
        self.momentum_periods = self.config.get('momentum_periods', [21, 63, 252])
        
        # Default weights: equal weight to medium and long-term, less to short-term
        default_weights = [0.3, 0.3, 0.4][:len(self.momentum_periods)]
        momentum_weights_config = self.config.get('momentum_weights', default_weights)
        
        # Handle case where momentum_weights might be a dict (from saved config)
        if isinstance(momentum_weights_config, dict):
            # Extract weights from dict format (from metadata)
            momentum_weights_config = list(momentum_weights_config.values())
        
        self.momentum_weights = np.array(momentum_weights_config)
        
        # Normalize weights to sum to 1
        if self.momentum_weights.sum() > 0:
            self.momentum_weights = self.momentum_weights / self.momentum_weights.sum()
        else:
            self.momentum_weights = np.array(default_weights)
            self.momentum_weights = self.momentum_weights / self.momentum_weights.sum()
        
        # For trainable mode
        if self.mode == 'trainable':
            self._regression_model = LinearRegression(fit_intercept=False)
        else:
            self._regression_model = None
        
        # Expected feature columns
        self._expected_features = [f'momentum_{period}d' for period in self.momentum_periods]
        
        logger.info(f"Initialized MomentumRankingModel in {self.mode} mode")
        logger.info(f"Momentum periods: {self.momentum_periods}")
        logger.info(f"Momentum weights: {self.momentum_weights}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MomentumRankingModel':
        """
        Train the momentum model.
        
        For rule-based mode: Just validates data and sets status to TRAINED
        For trainable mode: Learns optimal momentum combination weights
        
        Args:
            X: Feature DataFrame with momentum columns (momentum_21d, momentum_63d, etc.)
            y: Target Series (forward returns)
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If data validation fails
        """
        try:
            # Validate input data
            self.validate_data(X, y)
            
            # Check for required momentum columns
            missing_features = set(self._expected_features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required momentum columns: {missing_features}")
            
            # Align and clean data
            aligned_data = pd.concat([y, X[self._expected_features]], axis=1).dropna()
            if len(aligned_data) == 0:
                raise ValueError("No valid data points after alignment")
            
            y_clean = aligned_data.iloc[:, 0]
            X_clean = aligned_data.iloc[:, 1:]
            
            if self.mode == 'trainable':
                # Learn optimal momentum weights via linear regression
                self._learn_momentum_weights(X_clean, y_clean)
            else:
                # Rule-based: no training needed, just validate
                logger.info("Rule-based mode: using predefined weights")
            
            # Update model status and metadata
            self.status = ModelStatus.TRAINED
            self.metadata.training_samples = len(y_clean)
            self.metadata.features = list(X_clean.columns)
            
            # Store weights as hyperparameters
            weight_dict = {
                f'{period}d_weight': float(weight)
                for period, weight in zip(self.momentum_periods, self.momentum_weights)
            }
            self.metadata.hyperparameters.update({
                'momentum_weights': weight_dict,
                'mode': self.mode,
                'top_n': self.top_n,
                'min_momentum': self.min_momentum
            })
            
            logger.info(f"Successfully fitted MomentumRankingModel on {len(y_clean)} samples")
            logger.info(f"Final momentum weights: {weight_dict}")
            
            return self
        
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"Failed to fit MomentumRankingModel: {e}")
            raise
    
    def _learn_momentum_weights(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Learn optimal momentum combination weights using linear regression.
        
        Args:
            X: Momentum features
            y: Forward returns
        """
        # Fit linear regression to find optimal weights
        self._regression_model.fit(X, y)
        
        # Get learned weights
        learned_weights = self._regression_model.coef_
        
        # Ensure non-negative weights and normalize
        learned_weights = np.maximum(learned_weights, 0)
        if learned_weights.sum() > 0:
            self.momentum_weights = learned_weights / learned_weights.sum()
        else:
            logger.warning("All learned weights were negative, using default weights")
        
        logger.info(f"Learned momentum weights: {self.momentum_weights}")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict momentum scores for assets.
        
        The prediction returns a composite momentum score calculated as:
        score = w1 * momentum_21d + w2 * momentum_63d + w3 * momentum_252d
        
        Higher scores indicate stronger momentum.
        
        Args:
            X: Feature DataFrame with required momentum columns
        
        Returns:
            Array of momentum scores (one per asset)
        
        Raises:
            ValueError: If model is not trained or data is invalid
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model must be trained before making predictions")
        
        try:
            # Validate input data
            self.validate_data(X)
            
            # Check for required momentum columns
            missing_features = set(self._expected_features) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required momentum columns: {missing_features}")
            
            # Use only expected momentum columns in correct order
            X_pred = X[self._expected_features].copy()
            
            # Calculate composite momentum score
            # momentum_score = w1*mom_21d + w2*mom_63d + w3*mom_252d
            momentum_scores = np.zeros(len(X_pred))
            
            for i, (period, weight) in enumerate(zip(self.momentum_periods, self.momentum_weights)):
                col = f'momentum_{period}d'
                if col in X_pred.columns:
                    momentum_scores += weight * X_pred[col].values
            
            logger.debug(f"Generated momentum scores for {len(momentum_scores)} assets")
            
            return momentum_scores
        
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def get_top_n_signals(self, X: pd.DataFrame) -> pd.Series:
        """
        Get binary signals for top N assets by momentum.
        
        This method:
        1. Calculates momentum scores
        2. Applies absolute momentum filter (min_momentum threshold)
        3. Selects top N assets by relative momentum
        4. Returns 1.0 for selected assets, 0.0 for others
        
        Args:
            X: Feature DataFrame with momentum columns
        
        Returns:
            Series with 1.0 for top N assets, 0.0 for others (index matches X)
        """
        # Get momentum scores
        momentum_scores = self.predict(X)
        
        # Create Series with scores
        scores_series = pd.Series(momentum_scores, index=X.index)
        
        # Apply absolute momentum filter
        filtered_scores = scores_series[scores_series >= self.min_momentum]
        
        # Select top N by relative momentum
        if len(filtered_scores) == 0:
            # No assets pass the filter
            return pd.Series(0.0, index=X.index)
        
        # Get top N
        top_n_indices = filtered_scores.nlargest(self.top_n).index
        
        # Create binary signals
        signals = pd.Series(0.0, index=X.index)
        signals.loc[top_n_indices] = 1.0
        
        logger.info(f"Selected {len(top_n_indices)} assets (top {self.top_n} from {len(filtered_scores)} candidates)")
        
        return signals
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get momentum weights as feature importance scores.
        
        Returns:
            Dictionary mapping momentum period to its weight
        """
        if self.status != ModelStatus.TRAINED:
            return {}
        
        importance = {
            f'momentum_{period}d': float(weight)
            for period, weight in zip(self.momentum_periods, self.momentum_weights)
        }
        
        return importance
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model details
        """
        return {
            'model_type': self.model_type,
            'status': self.status,
            'mode': self.mode,
            'top_n': self.top_n,
            'min_momentum': self.min_momentum,
            'momentum_periods': self.momentum_periods,
            'training_samples': self.metadata.training_samples,
            'momentum_weights': self.get_feature_importance()
        }
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the momentum model to disk.
        
        Args:
            path: Directory path where model should be saved
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Create model dictionary with all components
        model_dict = {
            'mode': self.mode,
            'top_n': self.top_n,
            'min_momentum': self.min_momentum,
            'momentum_periods': self.momentum_periods,
            'momentum_weights': self.momentum_weights.tolist(),
            'expected_features': self._expected_features,
            'regression_model': self._regression_model if self.mode == 'trainable' else None
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
        
        logger.info(f"MomentumRankingModel saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'MomentumRankingModel':
        """
        Load a momentum model from disk.
        
        Args:
            path: Directory path where model is saved
        
        Returns:
            Loaded MomentumRankingModel instance
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
        instance.mode = model_dict['mode']
        instance.top_n = model_dict['top_n']
        instance.min_momentum = model_dict['min_momentum']
        instance.momentum_periods = model_dict['momentum_periods']
        instance.momentum_weights = np.array(model_dict['momentum_weights'])
        instance._expected_features = model_dict['expected_features']
        instance._regression_model = model_dict.get('regression_model')
        
        logger.info(f"MomentumRankingModel loaded from {path}")
        return instance

