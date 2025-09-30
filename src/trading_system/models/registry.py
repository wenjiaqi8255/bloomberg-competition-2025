"""
Model Registry Initialization

This module registers all available models with the ModelFactory.
Import this module to ensure all models are available for creation.
"""

from .base.model_factory import ModelFactory
from .implementations.ff5_model import FF5RegressionModel
from .implementations.residual_model import ResidualPredictionModel

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

    # Register Residual Prediction Model
    ModelFactory.register(
        model_type="residual_predictor",
        model_class=ResidualPredictionModel,
        description="Two-stage model combining FF5 with ML residual prediction",
        default_config={
            "residual_model_type": "xgboost",
            "residual_params": {
                "n_estimators": 100,
                "max_depth": 3,
                "learning_rate": 0.1
            },
            "ff5_config": {
                "regularization": "ridge",
                "alpha": 1.0
            }
        }
    )

    # Additional model types can be registered here
    # Example:
    # ModelFactory.register(
    #     model_type="simple_linear",
    #     model_class=SimpleLinearModel,
    #     description="Simple linear regression model",
    #     default_config={}
    # )

# Auto-register models when this module is imported
register_all_models()

# Export convenience functions
def create_ff5_model(config=None):
    """Convenience function to create FF5 model."""
    return ModelFactory.create("ff5_regression", config)

def create_residual_predictor(config=None):
    """Convenience function to create residual predictor."""
    return ModelFactory.create("residual_predictor", config)

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