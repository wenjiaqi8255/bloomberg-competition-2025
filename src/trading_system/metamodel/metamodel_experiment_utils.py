"""
MetaModel Experiment Utilities - Pure Functions
==============================================

This module provides pure functions for MetaModel experiment operations.
Following functional programming principles - no side effects, no state.

Design Principles:
- Pure functions only (no state, no side effects)
- Single responsibility for each function
- Reusable across different orchestrators
- Comprehensive error handling
"""

import logging
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

from src.trading_system.models.training.types import TrainingConfig
from src.trading_system.models.model_persistence import ModelRegistry
from src.trading_system.metamodel.meta_model import MetaModel


def setup_logging(verbose: bool = False) -> None:
    """
    Setup logging configuration.

    Args:
        verbose: Enable debug logging if True
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_metamodel_config_from_yaml(yaml_config: Dict[str, Any]) -> TrainingConfig:
    """
    Create TrainingConfig from YAML configuration.

    Args:
        yaml_config: Parsed YAML configuration

    Returns:
        TrainingConfig instance
    """
    metamodel_config = yaml_config.get('metamodel_training', {})

    # Parse dates
    start_date = datetime.fromisoformat(metamodel_config.get('start_date', '2022-01-01'))
    end_date = datetime.fromisoformat(metamodel_config.get('end_date', '2023-12-31'))

    return TrainingConfig(
        model_type='metamodel',
        experiment_name=metamodel_config.get('experiment_name', 'real_metamodel_experiment'),
        model_params={
            'method': metamodel_config.get('method', 'ridge'),
            'alpha': metamodel_config.get('alpha', 0.5),
            'strategies': metamodel_config.get('strategies', []),
            'data_source': metamodel_config.get('data_source', 'portfolio_files'),
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'target_benchmark': metamodel_config.get('target_benchmark', 'equal_weighted'),
            'use_cross_validation': metamodel_config.get('use_cross_validation', True),
            'cv_folds': metamodel_config.get('cv_folds', 5),
            'track_strategy_correlation': metamodel_config.get('track_strategy_correlation', True),
            'track_contribution_analysis': metamodel_config.get('track_contribution_analysis', True)
        },
        tags=yaml_config.get('experiment', {}).get('tags', {}),
        enable_wandb=True,
        wandb_project=yaml_config.get('experiment', {}).get('wandb_project', 'bloomberg-competition')
    )


def get_model_registry(config_path: str) -> ModelRegistry:
    """
    Get ModelRegistry instance from configuration.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configured ModelRegistry instance
    """
    yaml_config = load_config(config_path)
    registry_path = yaml_config.get('model_registry', {}).get('base_path', './models')
    return ModelRegistry(registry_path)


def load_trained_metamodel(model_id: str, config_path: str) -> Tuple[MetaModel, Dict[str, Any]]:
    """
    Load trained MetaModel and its artifacts.

    Args:
        model_id: Model ID of trained MetaModel
        config_path: Path to YAML configuration file

    Returns:
        Tuple of (MetaModel instance, artifacts dictionary)

    Raises:
        ValueError: If model loading fails
    """
    registry = get_model_registry(config_path)

    loaded_result = registry.load_model_with_artifacts(model_id)
    if not loaded_result:
        raise ValueError(f"Failed to load model: {model_id}")

    return loaded_result


def generate_model_name(yaml_config: Dict[str, Any], method: str) -> str:
    """
    Generate model name from configuration.

    Args:
        yaml_config: Parsed YAML configuration
        method: Training method name

    Returns:
        Generated model name
    """
    model_name_template = yaml_config.get('model_registry', {}).get(
        'model_name_template', 'real_metamodel_{method}_{date}'
    )
    return model_name_template.format(
        method=method,
        date=datetime.now().strftime('%Y%m%d_%H%M%S')
    )


def calculate_strategy_metrics(strategy_weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Calculate metrics for strategy weights.

    Args:
        strategy_weights: Dictionary of strategy weights

    Returns:
        Dictionary of calculated metrics
    """
    if not strategy_weights:
        return {
            'effective_strategies': 0,
            'max_weight': 0.0,
            'weight_concentration': 0.0,
            'diversification_score': 1.0
        }

    effective_strategies = len([w for w in strategy_weights.values() if w > 0.01])
    max_weight = max(strategy_weights.values()) if strategy_weights else 0.0
    weight_concentration = sum(w**2 for w in strategy_weights.values())
    diversification_score = 1.0 - weight_concentration

    return {
        'effective_strategies': effective_strategies,
        'max_weight': max_weight,
        'weight_concentration': weight_concentration,
        'diversification_score': diversification_score
    }


def create_recommendation_summary(strategy_weights: Dict[str, float]) -> Dict[str, Any]:
    """
    Create recommendation summary from strategy weights.

    Args:
        strategy_weights: Dictionary of strategy weights

    Returns:
        Recommendation summary dictionary
    """
    if not strategy_weights:
        return {
            'primary_strategy': None,
            'diversification_score': 1.0,
            'confidence_level': 'low'
        }

    metrics = calculate_strategy_metrics(strategy_weights)

    # Determine primary strategy
    primary_strategy = max(strategy_weights.items(), key=lambda x: x[1])[0] if strategy_weights else None

    # Determine confidence level
    if len(strategy_weights) >= 5:
        confidence_level = 'high'
    elif len(strategy_weights) >= 3:
        confidence_level = 'medium'
    else:
        confidence_level = 'low'

    return {
        'primary_strategy': primary_strategy,
        'diversification_score': metrics['diversification_score'],
        'confidence_level': confidence_level
    }


def get_top_strategies(strategy_weights: Dict[str, float], top_n: int = 5) -> list:
    """
    Get top N strategies by weight.

    Args:
        strategy_weights: Dictionary of strategy weights
        top_n: Number of top strategies to return

    Returns:
        List of (strategy_name, weight) tuples sorted by weight
    """
    if not strategy_weights:
        return []

    return sorted(
        strategy_weights.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]


def validate_model_id(model_id: str) -> bool:
    """
    Validate model ID format.

    Args:
        model_id: Model ID to validate

    Returns:
        True if valid, False otherwise
    """
    if not model_id or not isinstance(model_id, str):
        return False

    # Basic validation - should contain alphanumeric characters and underscores
    return model_id.replace('_', '').replace('-', '').isalnum()


def validate_date_string(date_string: str) -> bool:
    """
    Validate date string format (YYYY-MM-DD).

    Args:
        date_string: Date string to validate

    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.fromisoformat(date_string)
        return True
    except (ValueError, TypeError):
        return False