"""
Search Space Builder Utility

This module provides utilities for building and managing hyperparameter
search spaces with validation, presets, and intelligent defaults.

Key Features:
- Preset search spaces for common model types
- Search space validation and optimization
- Intelligent parameter suggestions
- Search space serialization/deserialization
- Parameter constraint handling
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from .hyperparameter_optimizer import SearchSpace, HyperparameterConfig

logger = logging.getLogger(__name__)


@dataclass
class ParameterConstraint:
    """Parameter constraint for search space."""
    constraint_type: str  # "range", "set", "conditional", "equality"
    parameters: List[str]
    condition: str
    value: Any


@dataclass
class SearchSpacePreset:
    """Preset configuration for search spaces."""
    name: str
    model_type: str
    search_spaces: Dict[str, SearchSpace]
    constraints: List[ParameterConstraint] = field(default_factory=list)
    description: str = ""
    tags: List[str] = field(default_factory=list)
    recommended_trials: int = 100


class SearchSpaceBuilder:
    """
    Utility class for building and managing hyperparameter search spaces.

    Provides predefined presets, validation, and intelligent search space
    generation for common machine learning models.
    """

    def __init__(self):
        """Initialize search space builder."""
        self.presets: Dict[str, SearchSpacePreset] = {}
        self._load_default_presets()
        self.validation_errors: List[str] = []

    def _load_default_presets(self) -> None:
        """Load default search space presets."""
        # XGBoost Presets
        self.presets["xgboost_default"] = SearchSpacePreset(
            name="xgboost_default",
            model_type="xgboost",
            search_spaces={
                "n_estimators": SearchSpace(
                    name="n_estimators",
                    type="int",
                    low=50,
                    high=500,
                    step=10
                ),
                "max_depth": SearchSpace(
                    name="max_depth",
                    type="int",
                    low=3,
                    high=12
                ),
                "learning_rate": SearchSpace(
                    name="learning_rate",
                    type="loguniform",
                    low=0.01,
                    high=0.3,
                    log=True
                ),
                "subsample": SearchSpace(
                    name="subsample",
                    type="float",
                    low=0.6,
                    high=1.0,
                    step=0.1
                ),
                "colsample_bytree": SearchSpace(
                    name="colsample_bytree",
                    type="float",
                    low=0.6,
                    high=1.0,
                    step=0.1
                ),
                "min_child_weight": SearchSpace(
                    name="min_child_weight",
                    type="int",
                    low=1,
                    high=10
                ),
                "gamma": SearchSpace(
                    name="gamma",
                    type="float",
                    low=0.0,
                    high=0.5,
                    step=0.1
                ),
                "reg_alpha": SearchSpace(
                    name="reg_alpha",
                    type="loguniform",
                    low=0.001,
                    high=10.0,
                    log=True
                ),
                "reg_lambda": SearchSpace(
                    name="reg_lambda",
                    type="loguniform",
                    low=0.001,
                    high=10.0,
                    log=True
                )
            },
            description="Default XGBoost search space with regularization",
            tags=["xgboost", "tree", "gradient_boosting"],
            recommended_trials=150
        )

        self.presets["xgboost_fast"] = SearchSpacePreset(
            name="xgboost_fast",
            model_type="xgboost",
            search_spaces={
                "n_estimators": SearchSpace(
                    name="n_estimators",
                    type="categorical",
                    choices=[100, 200, 300]
                ),
                "max_depth": SearchSpace(
                    name="max_depth",
                    type="categorical",
                    choices=[3, 6, 9]
                ),
                "learning_rate": SearchSpace(
                    name="learning_rate",
                    type="categorical",
                    choices=[0.01, 0.05, 0.1, 0.2]
                ),
                "subsample": SearchSpace(
                    name="subsample",
                    type="categorical",
                    choices=[0.8, 0.9, 1.0]
                )
            },
            description="Fast XGBoost search space with discrete parameters",
            tags=["xgboost", "fast", "tree"],
            recommended_trials=50
        )

        # LightGBM Presets
        self.presets["lightgbm_default"] = SearchSpacePreset(
            name="lightgbm_default",
            model_type="lightgbm",
            search_spaces={
                "n_estimators": SearchSpace(
                    name="n_estimators",
                    type="int",
                    low=50,
                    high=500,
                    step=10
                ),
                "max_depth": SearchSpace(
                    name="max_depth",
                    type="int",
                    low=3,
                    high=12
                ),
                "learning_rate": SearchSpace(
                    name="learning_rate",
                    type="loguniform",
                    low=0.01,
                    high=0.3,
                    log=True
                ),
                "num_leaves": SearchSpace(
                    name="num_leaves",
                    type="int",
                    low=20,
                    high=100
                ),
                "feature_fraction": SearchSpace(
                    name="feature_fraction",
                    type="float",
                    low=0.6,
                    high=1.0,
                    step=0.1
                ),
                "bagging_fraction": SearchSpace(
                    name="bagging_fraction",
                    type="float",
                    low=0.6,
                    high=1.0,
                    step=0.1
                ),
                "bagging_freq": SearchSpace(
                    name="bagging_freq",
                    type="int",
                    low=1,
                    high=10
                ),
                "min_child_samples": SearchSpace(
                    name="min_child_samples",
                    type="int",
                    low=5,
                    high=50
                ),
                "reg_alpha": SearchSpace(
                    name="reg_alpha",
                    type="loguniform",
                    low=0.001,
                    high=10.0,
                    log=True
                ),
                "reg_lambda": SearchSpace(
                    name="reg_lambda",
                    type="loguniform",
                    low=0.001,
                    high=10.0,
                    log=True
                )
            },
            description="Default LightGBM search space with regularization",
            tags=["lightgbm", "tree", "gradient_boosting"],
            recommended_trials=150
        )

        # Random Forest Presets
        self.presets["random_forest_default"] = SearchSpacePreset(
            name="random_forest_default",
            model_type="random_forest",
            search_spaces={
                "n_estimators": SearchSpace(
                    name="n_estimators",
                    type="int",
                    low=50,
                    high=500,
                    step=10
                ),
                "max_depth": SearchSpace(
                    name="max_depth",
                    type="int",
                    low=3,
                    high=20
                ),
                "min_samples_split": SearchSpace(
                    name="min_samples_split",
                    type="int",
                    low=2,
                    high=20
                ),
                "min_samples_leaf": SearchSpace(
                    name="min_samples_leaf",
                    type="int",
                    low=1,
                    high=10
                ),
                "max_features": SearchSpace(
                    name="max_features",
                    type="categorical",
                    choices=["sqrt", "log2", None]
                ),
                "bootstrap": SearchSpace(
                    name="bootstrap",
                    type="categorical",
                    choices=[True, False]
                ),
                "min_weight_fraction_leaf": SearchSpace(
                    name="min_weight_fraction_leaf",
                    type="float",
                    low=0.0,
                    high=0.5,
                    step=0.1
                )
            },
            description="Default Random Forest search space",
            tags=["random_forest", "tree", "ensemble"],
            recommended_trials=100
        )

        # Linear Models Presets
        self.presets["linear_default"] = SearchSpacePreset(
            name="linear_default",
            model_type="linear",
            search_spaces={
                "alpha": SearchSpace(
                    name="alpha",
                    type="loguniform",
                    low=0.0001,
                    high=10.0,
                    log=True
                ),
                "l1_ratio": SearchSpace(
                    name="l1_ratio",
                    type="float",
                    low=0.0,
                    high=1.0,
                    step=0.1
                ),
                "fit_intercept": SearchSpace(
                    name="fit_intercept",
                    type="categorical",
                    choices=[True, False]
                ),
                "normalize": SearchSpace(
                    name="normalize",
                    type="categorical",
                    choices=[True, False]
                )
            },
            description="Default linear models search space",
            tags=["linear", "elasticnet", "regularization"],
            recommended_trials=50
        )

        # Neural Network Presets
        self.presets["neural_network_default"] = SearchSpacePreset(
            name="neural_network_default",
            model_type="neural_network",
            search_spaces={
                "hidden_layer_size": SearchSpace(
                    name="hidden_layer_size",
                    type="categorical",
                    choices=[32, 64, 128, 256]
                ),
                "n_layers": SearchSpace(
                    name="n_layers",
                    type="categorical",
                    choices=[1, 2, 3]
                ),
                "activation": SearchSpace(
                    name="activation",
                    type="categorical",
                    choices=["relu", "tanh", "logistic"]
                ),
                "learning_rate_init": SearchSpace(
                    name="learning_rate_init",
                    type="loguniform",
                    low=0.001,
                    high=0.1,
                    log=True
                ),
                "alpha": SearchSpace(
                    name="alpha",
                    type="loguniform",
                    low=0.0001,
                    high=0.1,
                    log=True
                ),
                "batch_size": SearchSpace(
                    name="batch_size",
                    type="categorical",
                    choices=[32, 64, 128, 256]
                ),
                "max_iter": SearchSpace(
                    name="max_iter",
                    type="categorical",
                    choices=[200, 500, 1000]
                )
            },
            description="Default neural network search space",
            tags=["neural_network", "mlp", "deep_learning"],
            recommended_trials=100
        )

        # SVM Presets
        self.presets["svm_default"] = SearchSpacePreset(
            name="svm_default",
            model_type="svm",
            search_spaces={
                "C": SearchSpace(
                    name="C",
                    type="loguniform",
                    low=0.001,
                    high=1000.0,
                    log=True
                ),
                "kernel": SearchSpace(
                    name="kernel",
                    type="categorical",
                    choices=["linear", "rbf", "poly", "sigmoid"]
                ),
                "gamma": SearchSpace(
                    name="gamma",
                    type="loguniform",
                    low=0.001,
                    high=100.0,
                    log=True
                ),
                "degree": SearchSpace(
                    name="degree",
                    type="int",
                    low=2,
                    high=5
                ),
                "coef0": SearchSpace(
                    name="coef0",
                    type="float",
                    low=0.0,
                    high=1.0,
                    step=0.1
                )
            },
            description="Default SVM search space",
            tags=["svm", "kernel", "classification"],
            recommended_trials=100
        )

        logger.info(f"Loaded {len(self.presets)} default search space presets")

    def get_preset(self, preset_name: str) -> SearchSpacePreset:
        """
        Get a search space preset by name.

        Args:
            preset_name: Name of the preset

        Returns:
            SearchSpacePreset instance

        Raises:
            ValueError: If preset not found
        """
        if preset_name not in self.presets:
            available = list(self.presets.keys())
            raise ValueError(f"Preset '{preset_name}' not found. Available presets: {available}")

        return self.presets[preset_name]

    def list_presets(self, model_type: Optional[str] = None) -> List[str]:
        """
        List available presets.

        Args:
            model_type: Filter by model type

        Returns:
            List of preset names
        """
        if model_type:
            return [
                name for name, preset in self.presets.items()
                if preset.model_type == model_type
            ]
        else:
            return list(self.presets.keys())

    def build_search_space(self,
                          preset_name: str,
                          custom_params: Optional[Dict[str, SearchSpace]] = None,
                          exclude_params: Optional[List[str]] = None) -> Dict[str, SearchSpace]:
        """
        Build search space from preset with customizations.

        Args:
            preset_name: Name of the preset to use
            custom_params: Custom parameters to add or override
            exclude_params: Parameters to exclude from preset

        Returns:
            Dictionary of search spaces
        """
        preset = self.get_preset(preset_name)
        search_spaces = preset.search_spaces.copy()

        # Remove excluded parameters
        if exclude_params:
            for param in exclude_params:
                search_spaces.pop(param, None)

        # Add or override custom parameters
        if custom_params:
            search_spaces.update(custom_params)

        # Validate search spaces
        self._validate_search_spaces(search_spaces)

        return search_spaces

    def create_intelligent_search_space(self,
                                       model_type: str,
                                       data_size: int,
                                       n_features: int,
                                       problem_type: str = "regression") -> Dict[str, SearchSpace]:
        """
        Create intelligent search space based on data characteristics.

        Args:
            model_type: Type of model
            data_size: Number of training samples
            n_features: Number of features
            problem_type: Type of problem ("regression", "classification")

        Returns:
            Dictionary of search spaces
        """
        # Adjust search space based on data size
        if data_size < 1000:
            # Small dataset - use smaller models
            n_estimators_range = (50, 200)
            max_depth_range = (3, 8)
        elif data_size < 10000:
            # Medium dataset
            n_estimators_range = (100, 300)
            max_depth_range = (5, 12)
        else:
            # Large dataset - can use larger models
            n_estimators_range = (200, 500)
            max_depth_range = (8, 15)

        # Adjust based on number of features
        if n_features > 100:
            # High dimensional data
            feature_fraction_range = (0.3, 0.7)
        else:
            feature_fraction_range = (0.6, 1.0)

        # Build model-specific search space
        if model_type.lower() == "xgboost":
            search_spaces = {
                "n_estimators": SearchSpace(
                    name="n_estimators",
                    type="int",
                    low=n_estimators_range[0],
                    high=n_estimators_range[1],
                    step=10
                ),
                "max_depth": SearchSpace(
                    name="max_depth",
                    type="int",
                    low=max_depth_range[0],
                    high=max_depth_range[1]
                ),
                "learning_rate": SearchSpace(
                    name="learning_rate",
                    type="loguniform",
                    low=0.01,
                    high=0.2,
                    log=True
                ),
                "subsample": SearchSpace(
                    name="subsample",
                    type="float",
                    low=0.7,
                    high=1.0,
                    step=0.1
                ),
                "colsample_bytree": SearchSpace(
                    name="colsample_bytree",
                    type="float",
                    low=feature_fraction_range[0],
                    high=feature_fraction_range[1],
                    step=0.1
                )
            }

            if data_size > 10000:
                # Add regularization for large datasets
                search_spaces.update({
                    "reg_alpha": SearchSpace(
                        name="reg_alpha",
                        type="loguniform",
                        low=0.001,
                        high=1.0,
                        log=True
                    ),
                    "reg_lambda": SearchSpace(
                        name="reg_lambda",
                        type="loguniform",
                        low=0.001,
                        high=1.0,
                        log=True
                    )
                })

        elif model_type.lower() == "lightgbm":
            search_spaces = {
                "n_estimators": SearchSpace(
                    name="n_estimators",
                    type="int",
                    low=n_estimators_range[0],
                    high=n_estimators_range[1],
                    step=10
                ),
                "max_depth": SearchSpace(
                    name="max_depth",
                    type="int",
                    low=max_depth_range[0],
                    high=max_depth_range[1]
                ),
                "learning_rate": SearchSpace(
                    name="learning_rate",
                    type="loguniform",
                    low=0.01,
                    high=0.2,
                    log=True
                ),
                "num_leaves": SearchSpace(
                    name="num_leaves",
                    type="int",
                    low=20,
                    high=min(100, 2**max_depth_range[1])
                ),
                "feature_fraction": SearchSpace(
                    name="feature_fraction",
                    type="float",
                    low=feature_fraction_range[0],
                    high=feature_fraction_range[1],
                    step=0.1
                ),
                "bagging_fraction": SearchSpace(
                    name="bagging_fraction",
                    type="float",
                    low=0.7,
                    high=1.0,
                    step=0.1
                )
            }

        elif model_type.lower() == "random_forest":
            search_spaces = {
                "n_estimators": SearchSpace(
                    name="n_estimators",
                    type="int",
                    low=n_estimators_range[0],
                    high=n_estimators_range[1],
                    step=10
                ),
                "max_depth": SearchSpace(
                    name="max_depth",
                    type="int",
                    low=max_depth_range[0],
                    high=max_depth_range[1]
                ),
                "min_samples_split": SearchSpace(
                    name="min_samples_split",
                    type="int",
                    low=2,
                    high=min(20, data_size // 100)
                ),
                "min_samples_leaf": SearchSpace(
                    name="min_samples_leaf",
                    type="int",
                    low=1,
                    high=min(10, data_size // 1000)
                ),
                "max_features": SearchSpace(
                    name="max_features",
                    type="categorical",
                    choices=["sqrt", "log2"]
                )
            }

        else:
            # Default to simple search space
            search_spaces = {
                "n_estimators": SearchSpace(
                    name="n_estimators",
                    type="int",
                    low=50,
                    high=300,
                    step=10
                ),
                "max_depth": SearchSpace(
                    name="max_depth",
                    type="int",
                    low=3,
                    high=10
                )
            }

        logger.info(f"Created intelligent search space for {model_type} with {len(search_spaces)} parameters")
        return search_spaces

    def _validate_search_spaces(self, search_spaces: Dict[str, SearchSpace]) -> bool:
        """
        Validate search spaces.

        Args:
            search_spaces: Dictionary of search spaces to validate

        Returns:
            True if all valid, False otherwise
        """
        self.validation_errors.clear()

        for name, space in search_spaces.items():
            if not space.validate():
                self.validation_errors.append(f"Invalid search space for parameter '{name}': {space}")

        # Check for name conflicts
        names = list(search_spaces.keys())
        if len(names) != len(set(names)):
            self.validation_errors.append("Duplicate parameter names found")

        # Check for reasonable ranges
        for name, space in search_spaces.items():
            if space.type in ["int", "float", "discrete_uniform", "loguniform"]:
                if space.low >= space.high:
                    self.validation_errors.append(
                        f"Parameter '{name}': low ({space.low}) >= high ({space.high})"
                    )

                if space.type == "discrete_uniform" and space.step <= 0:
                    self.validation_errors.append(
                        f"Parameter '{name}': step must be positive for discrete_uniform"
                    )

        if self.validation_errors:
            logger.warning(f"Search space validation failed: {self.validation_errors}")
            return False
        else:
            logger.info("Search space validation passed")
            return True

    def optimize_search_space(self,
                             search_spaces: Dict[str, SearchSpace],
                             max_total_combinations: int = 10000) -> Dict[str, SearchSpace]:
        """
        Optimize search space to reduce total combinations.

        Args:
            search_spaces: Original search spaces
            max_total_combinations: Maximum allowed combinations

        Returns:
            Optimized search spaces
        """
        # Calculate current total combinations
        total_combinations = self._calculate_total_combinations(search_spaces)
        logger.info(f"Original search space has {total_combinations:,} total combinations")

        if total_combinations <= max_total_combinations:
            return search_spaces

        # Need to reduce search space
        logger.info(f"Reducing search space to under {max_total_combinations:,} combinations")

        optimized_spaces = search_spaces.copy()
        reduction_factor = max_total_combinations / total_combinations

        # Prioritize parameters to reduce
        reduction_priority = [
            "float",  # Reduce float parameters first
            "int",
            "discrete_uniform",
            "loguniform",
            "categorical"  # Reduce categorical last
        ]

        for param_type in reduction_priority:
            if total_combinations <= max_total_combinations:
                break

            # Find parameters of this type
            type_params = [(name, space) for name, space in optimized_spaces.items()
                          if space.type == param_type]

            for name, space in type_params:
                if total_combinations <= max_total_combinations:
                    break

                if param_type == "float":
                    # Reduce float precision
                    if space.step:
                        old_step = space.step
                        new_step = old_step * 2
                        space.step = new_step
                        old_combinations = (space.high - space.low) / old_step + 1
                        new_combinations = (space.high - space.low) / new_step + 1
                        total_combinations = total_combinations / old_combinations * new_combinations
                        logger.debug(f"Reduced {name} step from {old_step} to {new_step}")

                elif param_type == "int":
                    # Increase step size for int parameters
                    if space.step and space.step > 1:
                        old_step = space.step
                        new_step = min(space.step * 2, space.high - space.low)
                        space.step = new_step
                        old_combinations = (space.high - space.low) // old_step + 1
                        new_combinations = (space.high - space.low) // new_step + 1
                        total_combinations = total_combinations / old_combinations * new_combinations
                        logger.debug(f"Increased {name} step from {old_step} to {new_step}")

                elif param_type == "categorical":
                    # Reduce number of choices
                    if len(space.choices) > 2:
                        old_choices = len(space.choices)
                        new_choices = max(2, int(old_choices * reduction_factor))
                        space.choices = space.choices[:new_choices]
                        total_combinations = total_combinations / old_choices * new_choices
                        logger.debug(f"Reduced {name} choices from {old_choices} to {new_choices}")

        # Re-validate
        self._validate_search_spaces(optimized_spaces)

        final_combinations = self._calculate_total_combinations(optimized_spaces)
        logger.info(f"Optimized search space has {final_combinations:,} combinations")

        return optimized_spaces

    def _calculate_total_combinations(self, search_spaces: Dict[str, SearchSpace]) -> int:
        """Calculate total number of combinations in search space."""
        total = 1

        for space in search_spaces.values():
            if space.type == "categorical":
                total *= len(space.choices)
            elif space.type == "int":
                if space.step:
                    total *= int((space.high - space.low) / space.step) + 1
                else:
                    total *= space.high - space.low + 1
            elif space.type == "float":
                if space.step:
                    total *= int((space.high - space.low) / space.step) + 1
                else:
                    # For continuous parameters, use arbitrary discretization
                    total *= 100
            elif space.type == "discrete_uniform":
                total *= int((space.high - space.low) / space.step) + 1
            elif space.type == "loguniform":
                # For loguniform, use arbitrary discretization
                total *= 100

        return total

    def suggest_trial_count(self, search_spaces: Dict[str, SearchSpace], target_coverage: float = 0.1) -> int:
        """
        Suggest number of trials for search space.

        Args:
            search_spaces: Search space definition
            target_coverage: Target coverage of search space (0.0 to 1.0)

        Returns:
            Suggested number of trials
        """
        total_combinations = self._calculate_total_combinations(search_spaces)
        suggested_trials = int(total_combinations * target_coverage)

        # Apply reasonable bounds
        suggested_trials = max(10, min(1000, suggested_trials))

        return suggested_trials

    def export_search_space(self, search_spaces: Dict[str, SearchSpace], format: str = "json") -> str:
        """
        Export search space to string format.

        Args:
            search_spaces: Search space definition
            format: Export format ("json", "dict")

        Returns:
            Serialized search space
        """
        if format == "json":
            data = {
                name: {
                    "type": space.type,
                    "low": space.low,
                    "high": space.high,
                    "choices": space.choices,
                    "step": space.step,
                    "log": space.log
                }
                for name, space in search_spaces.items()
            }
            return json.dumps(data, indent=2)
        elif format == "dict":
            return str(search_spaces)
        else:
            raise ValueError(f"Unknown format: {format}")

    def import_search_space(self, data: str, format: str = "json") -> Dict[str, SearchSpace]:
        """
        Import search space from string format.

        Args:
            data: Serialized search space
            format: Import format ("json", "dict")

        Returns:
            Search space dictionary
        """
        if format == "json":
            loaded_data = json.loads(data)
            search_spaces = {}
            for name, params in loaded_data.items():
                search_spaces[name] = SearchSpace(
                    name=name,
                    **params
                )
            return search_spaces
        else:
            raise ValueError(f"Unknown format: {format}")

    def get_parameter_statistics(self, search_spaces: Dict[str, SearchSpace]) -> pd.DataFrame:
        """
        Get statistics about search space parameters.

        Args:
            search_spaces: Search space definition

        Returns:
            DataFrame with parameter statistics
        """
        stats = []

        for name, space in search_spaces.items():
            if space.type == "categorical":
                param_stats = {
                    "parameter": name,
                    "type": space.type,
                    "n_choices": len(space.choices),
                    "min_value": None,
                    "max_value": None,
                    "choices": str(space.choices)
                }
            elif space.type in ["int", "float", "discrete_uniform", "loguniform"]:
                param_stats = {
                    "parameter": name,
                    "type": space.type,
                    "n_choices": None,
                    "min_value": space.low,
                    "max_value": space.high,
                    "choices": None
                }
            else:
                param_stats = {
                    "parameter": name,
                    "type": space.type,
                    "n_choices": None,
                    "min_value": None,
                    "max_value": None,
                    "choices": None
                }

            stats.append(param_stats)

        return pd.DataFrame(stats)

    def save_preset(self, preset: SearchSpacePreset, filename: str) -> None:
        """
        Save a preset to file.

        Args:
            preset: Preset to save
            filename: File path
        """
        # Convert to serializable format
        preset_data = {
            "name": preset.name,
            "model_type": preset.model_type,
            "description": preset.description,
            "tags": preset.tags,
            "recommended_trials": preset.recommended_trials,
            "search_spaces": self.export_search_space(preset.search_spaces, "json")
        }

        with open(filename, 'w') as f:
            json.dump(preset_data, f, indent=2)

        logger.info(f"Saved preset '{preset.name}' to {filename}")

    def load_preset(self, filename: str) -> SearchSpacePreset:
        """
        Load a preset from file.

        Args:
            filename: File path

        Returns:
            Loaded preset
        """
        with open(filename, 'r') as f:
            preset_data = json.load(f)

        search_spaces = self.import_search_space(preset_data["search_spaces"], "json")

        preset = SearchSpacePreset(
            name=preset_data["name"],
            model_type=preset_data["model_type"],
            description=preset_data.get("description", ""),
            tags=preset_data.get("tags", []),
            recommended_trials=preset_data.get("recommended_trials", 100),
            search_spaces=search_spaces
        )

        logger.info(f"Loaded preset '{preset.name}' from {filename}")
        return preset

    def create_recommended_config(self, search_spaces: Dict[str, SearchSpace]) -> HyperparameterConfig:
        """
        Create recommended hyperparameter optimization configuration.

        Args:
            search_spaces: Search space definition

        Returns:
            Recommended HyperparameterConfig
        """
        n_trials = self.suggest_trial_count(search_spaces, target_coverage=0.1)

        return HyperparameterConfig(
            n_trials=n_trials,
            sampler="tpe",
            pruner="median",
            direction="maximize",
            metric_name="val_score",
            early_stopping=True,
            early_stopping_patience=max(20, n_trials // 5),
            track_trials=True
        )