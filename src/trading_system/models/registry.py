"""
Model Registry Initialization

This module registers all available models with the ModelFactory.
Import this module to ensure all models are available for creation.
"""

from .base.model_factory import ModelFactory
from .implementations.ff5_model import FF5RegressionModel
from .implementations.momentum_model import MomentumRankingModel

# Try to import ML models (optional dependencies)
try:
    from .implementations.xgboost_model import XGBoostModel
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from .implementations.lstm_model import LSTMModel
    LSTM_AVAILABLE = True
except ImportError:
    LSTM_AVAILABLE = False

# Register all available models
def register_all_models():
    """Register all model implementations with the factory."""

    # Register FF5 Regression Model
    ModelFactory.register(
        model_type="ff5_regression",
        model_class=FF5RegressionModel,
        description="Fama-French 5-Factor regression model for baseline returns",
        default_config={
            "regularization": "none",
            "alpha": 1.0,
            "standardize": False
        }
    )

    # Register Momentum Ranking Model
    ModelFactory.register(
        model_type="momentum_ranking",
        model_class=MomentumRankingModel,
        description="Momentum ranking model for dual momentum strategies",
        default_config={
            "mode": "rule_based",
            "top_n": 5,
            "min_momentum": 0.0,
            "momentum_weights": [0.3, 0.3, 0.4],  # 21d, 63d, 252d
            "momentum_periods": [21, 63, 252]
        }
    )

    # Register XGBoost Model (if available)
    if XGBOOST_AVAILABLE:
        ModelFactory.register(
            model_type="xgboost",
            model_class=XGBoostModel,
            description="XGBoost gradient boosting model for non-linear relationships",
            default_config={
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "early_stopping_rounds": 10,
                "random_state": 42
            }
        )

    # Register LSTM Model (if available)
    if LSTM_AVAILABLE:
        ModelFactory.register(
            model_type="lstm",
            model_class=LSTMModel,
            description="LSTM recurrent neural network for sequential patterns",
            default_config={
                "sequence_length": 20,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.2,
                "bidirectional": False,
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "early_stopping_patience": 10
            }
        )

# Auto-register models when this module is imported
register_all_models()

# Export convenience functions
def create_ff5_model(config=None):
    """Convenience function to create FF5 model."""
    return ModelFactory.create("ff5_regression", config)

def create_momentum_model(config=None):
    """Convenience function to create Momentum Ranking model."""
    return ModelFactory.create("momentum_ranking", config)

def create_xgboost_model(config=None):
    """Convenience function to create XGBoost model."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not available. Install with: pip install xgboost")
    return ModelFactory.create("xgboost", config)

def create_lstm_model(config=None):
    """Convenience function to create LSTM model."""
    if not LSTM_AVAILABLE:
        raise ImportError("PyTorch is not available. Install with: pip install torch")
    return ModelFactory.create("lstm", config)

def list_available_models():
    """Get list of all registered model types."""
    return list(ModelFactory._registry.keys())

def get_model_info(model_type):
    """Get information about a registered model."""
    if model_type in ModelFactory._registry:
        registration = ModelFactory._registry[model_type]
        return {
            "type": model_type,
            "class": registration.model_class.__name__,
            "description": registration.description,
            "default_config": registration.default_config
        }
    return None