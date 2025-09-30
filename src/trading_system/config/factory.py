"""
Configuration Factory

Simplified factory for creating configuration objects.
Follows KISS and YAGNI principles.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Union
from datetime import datetime

from .base import BaseConfig
from .strategy import StrategyConfig
from .backtest import BacktestConfig
from .system import SystemConfig

logger = logging.getLogger(__name__)


class ConfigFactory:
    """
    Simplified factory class for creating configuration objects.

    Minimal functionality following KISS principle.
    """

    @staticmethod
    def create_backtest_config(**kwargs) -> BacktestConfig:
        """Create BacktestConfig from parameters."""
        return BacktestConfig(**kwargs)

    @staticmethod
    def create_strategy_config(**kwargs) -> StrategyConfig:
        """Create StrategyConfig from parameters."""
        return StrategyConfig(**kwargs)

    @staticmethod
    def create_system_config(**kwargs) -> SystemConfig:
        """Create SystemConfig from parameters."""
        return SystemConfig(**kwargs)

    @staticmethod
    def from_yaml(file_path: Union[str, Path]) -> Dict[str, BaseConfig]:
        """
        Load configurations from YAML file.

        Args:
            file_path: Path to YAML configuration file

        Returns:
            Dict[str, BaseConfig]: Dictionary of configuration objects
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, 'r') as f:
            config_data = yaml.safe_load(f)

        configs = {}

        for config_name, config_params in config_data.items():
            if config_name == 'backtest':
                # Convert date strings to datetime objects
                processed_params = ConfigFactory._process_backtest_params(config_params)
                configs[config_name] = ConfigFactory.create_backtest_config(**processed_params)
            elif config_name == 'strategy':
                # Convert strategy_type string to enum
                processed_params = ConfigFactory._process_strategy_params(config_params)
                configs[config_name] = ConfigFactory.create_strategy_config(**processed_params)
            elif config_name == 'system':
                # Process system parameters and ignore unknown fields
                processed_params = ConfigFactory._process_system_params(config_params)
                configs[config_name] = ConfigFactory.create_system_config(**processed_params)

        logger.info(f"Loaded {len(configs)} configurations from {file_path}")
        return configs

    @staticmethod
    def _process_backtest_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Process backtest parameters, especially date conversion."""
        from .backtest import BacktestConfig

        processed = {}
        parameters = {}

        # Get the fields that BacktestConfig actually accepts
        if hasattr(BacktestConfig, '__dataclass_fields__'):
            valid_fields = set(BacktestConfig.__dataclass_fields__.keys())

            # Also include base config fields
            if hasattr(BaseConfig, '__dataclass_fields__'):
                valid_fields.update(BaseConfig.__dataclass_fields__.keys())

            # Handle field name mapping for legacy configs
            field_mapping = {
                'transaction_cost': 'commission_rate',
                'transaction_costs': 'commission_rate',
                'slippage': 'slippage_rate'
            }

            # Separate known fields from unknown ones
            for key, value in params.items():
                if key in valid_fields or key in field_mapping:
                    # Apply field mapping if needed
                    if key in field_mapping:
                        processed[field_mapping[key]] = value
                    else:
                        processed[key] = value
                elif key == 'parameters':
                    # Existing parameters dict, merge it
                    if isinstance(value, dict):
                        parameters.update(value)
                    else:
                        logger.warning(f"Parameters field should be a dict, got {type(value)}")
                else:
                    # Put unknown fields into parameters for now
                    parameters[key] = value
                    logger.warning(f"Putting unknown backtest field in parameters: {key}")

            # Handle date conversion
            if 'start_date' in processed and isinstance(processed['start_date'], str):
                processed['start_date'] = datetime.fromisoformat(processed['start_date'])

            if 'end_date' in processed and isinstance(processed['end_date'], str):
                processed['end_date'] = datetime.fromisoformat(processed['end_date'])

        else:
            # Fallback: just copy params as-is
            processed = params.copy()

        return processed

    @staticmethod
    def _process_strategy_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Process strategy parameters, especially strategy_type conversion."""
        from .strategy import StrategyType

        processed = {}
        parameters = {}

        # Get the fields that StrategyConfig actually accepts
        if hasattr(StrategyConfig, '__dataclass_fields__'):
            valid_fields = set(StrategyConfig.__dataclass_fields__.keys())

            # Also include base config fields
            if hasattr(BaseConfig, '__dataclass_fields__'):
                valid_fields.update(BaseConfig.__dataclass_fields__.keys())

            # Handle both 'strategy_type' and 'type' fields
            strategy_type_value = None
            if 'strategy_type' in params:
                strategy_type_value = params['strategy_type']
            elif 'type' in params:
                strategy_type_value = params['type']

            if strategy_type_value and isinstance(strategy_type_value, str):
                # Map various strategy type names to enum values
                type_mapping = {
                    'DualMomentumStrategy': 'dual_momentum',
                    'MLStrategy': 'ml',
                    'FamaFrenchStrategy': 'fama_french',
                    'CoreSatelliteStrategy': 'core_satellite',
                    'SatelliteStrategy': 'satellite'
                }
                mapped_value = type_mapping.get(strategy_type_value, strategy_type_value)
                processed['strategy_type'] = StrategyType(mapped_value)
            elif 'strategy_type' in params:
                processed['strategy_type'] = params['strategy_type']

            # Handle field name mapping for legacy configs
            field_mapping = {
                'lookback_days': 'lookback_period',
                'top_n_assets': 'max_positions',
                'max_position_size': 'position_size_limit'
            }

            # Separate known fields from parameters
            for key, value in params.items():
                if key in valid_fields or key in field_mapping:
                    # Apply field mapping if needed
                    if key in field_mapping:
                        processed[field_mapping[key]] = value
                    else:
                        processed[key] = value
                elif key == 'parameters':
                    # Existing parameters dict, merge it
                    if isinstance(value, dict):
                        parameters.update(value)
                    else:
                        logger.warning(f"Parameters field should be a dict, got {type(value)}")
                elif key != 'type':  # Don't include the raw 'type' field
                    # Put everything else into parameters
                    parameters[key] = value

            # Handle universe structure
            if 'universe' in params:
                universe = params['universe']
                if isinstance(universe, dict):
                    # Flatten structured universe into a list
                    symbols = []
                    for category, items in universe.items():
                        if isinstance(items, list):
                            symbols.extend(items)
                        elif isinstance(items, str):
                            symbols.append(items)
                    processed['universe'] = symbols
                elif isinstance(universe, list):
                    processed['universe'] = universe
                # Remove from parameters since we handled it specially
                if 'universe' in parameters:
                    del parameters['universe']

            # Add parameters dict if not empty
            if parameters:
                processed['parameters'] = parameters

            # Set default universe if empty
            if not processed.get('universe'):
                processed['universe'] = ['SPY', 'QQQ', 'IWM']  # Default minimal universe

        else:
            # Fallback: just copy params as-is
            processed = params.copy()

        return processed

    @staticmethod
    def _process_system_params(params: Dict[str, Any]) -> Dict[str, Any]:
        """Process system parameters, filtering out unknown fields."""
        from .system import SystemConfig

        processed = {}

        # Get the fields that SystemConfig actually accepts
        if hasattr(SystemConfig, '__dataclass_fields__'):
            valid_fields = set(SystemConfig.__dataclass_fields__.keys())

            # Also include base config fields
            if hasattr(BaseConfig, '__dataclass_fields__'):
                valid_fields.update(BaseConfig.__dataclass_fields__.keys())

            # Filter out unknown fields
            for key, value in params.items():
                if key in valid_fields:
                    processed[key] = value
                else:
                    logger.warning(f"Ignoring unknown system config field: {key}")

            # Handle date conversion for system config
            if 'start_date' in processed and isinstance(processed['start_date'], str):
                processed['start_date'] = datetime.fromisoformat(processed['start_date'])
            if 'end_date' in processed and isinstance(processed['end_date'], str):
                processed['end_date'] = datetime.fromisoformat(processed['end_date'])
        else:
            # Fallback: just copy params as-is
            processed = params.copy()

        return processed

    @staticmethod
    def load_all_configs(yaml_path: Union[str, Path]) -> Dict[str, BaseConfig]:
        """Convenient method to load all configurations from YAML."""
        return ConfigFactory.from_yaml(yaml_path)