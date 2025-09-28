"""
Configuration loader for strategy parameters and experiment settings.
"""

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader for strategy and experiment settings.

    Features:
    - YAML configuration file loading
    - Environment variable substitution
    - Configuration validation
    - Default value handling
    """

    def __init__(self, config_path: str = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to configuration file (default: configs/strategy_config.yaml)
        """
        if config_path is None:
            # Default to configs/strategy_config.yaml
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '..', '..', '..', 'configs', 'strategy_config.yaml')

        self.config_path = config_path
        self.config = {}

    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Returns:
            Dictionary with configuration parameters
        """
        try:
            if not os.path.exists(self.config_path):
                logger.warning(f"Config file not found: {self.config_path}")
                return self._get_default_config()

            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            if config is None:
                logger.warning("Empty configuration file")
                return self._get_default_config()

            # Substitute environment variables
            config = self._substitute_env_variables(config)

            # Validate configuration
            self._validate_config(config)

            self.config = config
            logger.info(f"Configuration loaded from {self.config_path}")
            return config

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()

    def get_strategy_config(self) -> Dict[str, Any]:
        """Get strategy-specific configuration."""
        return self.config.get('strategy', {})

    def get_universe_config(self) -> Dict[str, Any]:
        """Get asset universe configuration."""
        return self.config.get('universe', {})

    def get_backtest_config(self) -> Dict[str, Any]:
        """Get backtest configuration."""
        return self.config.get('backtest', {})

    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        return self.config.get('experiment', {})

    def get_data_config(self) -> Dict[str, Any]:
        """Get data provider configuration."""
        return self.config.get('data', {})

    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance metrics configuration."""
        return self.config.get('performance', {})

    def get_risk_config(self) -> Dict[str, Any]:
        """Get risk management configuration."""
        return self.config.get('risk_management', {})

    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.config.get('logging', {})

    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration."""
        return self.config.get('output', {})

    def get_asset_universe(self, universe_type: str = "all_assets") -> list:
        """
        Get asset universe list.

        Args:
            universe_type: Type of universe ('all_assets', 'equities', 'international', etc.)

        Returns:
            List of asset tickers
        """
        universe_config = self.get_universe_config()
        return universe_config.get(universe_type, [])

    def get_strategy_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters from configuration."""
        strategy_config = self.get_strategy_config()
        return {
            'lookback_days': strategy_config.get('lookback_days', 252),
            'top_n_assets': strategy_config.get('top_n_assets', 5),
            'minimum_positive_assets': strategy_config.get('minimum_positive_assets', 3),
            'cash_ticker': strategy_config.get('cash_ticker', 'SHY'),
            'include_cash_in_universe': strategy_config.get('include_cash_in_universe', True)
        }

    def save_config(self, config: Dict[str, Any], path: str = None):
        """
        Save configuration to YAML file.

        Args:
            config: Configuration dictionary to save
            path: Path to save configuration (default: original config path)
        """
        if path is None:
            path = self.config_path

        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def _substitute_env_variables(self, config: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.

        Args:
            config: Configuration value (could be dict, list, or scalar)

        Returns:
            Configuration with environment variables substituted
        """
        if isinstance(config, dict):
            return {key: self._substitute_env_variables(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._substitute_env_variables(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            # Extract environment variable name
            env_var = config[2:-1]
            default_value = None

            # Handle default values (format: ${VAR_NAME:default_value})
            if ':' in env_var:
                env_var, default_value = env_var.split(':', 1)

            # Get environment variable or default
            value = os.getenv(env_var, default_value)
            if value is None:
                logger.warning(f"Environment variable {env_var} not found and no default provided")
                return config  # Return original string

            # Convert to appropriate type
            if value.lower() in ['true', 'false']:
                return value.lower() == 'true'
            elif value.replace('.', '').isdigit():
                return float(value) if '.' in value else int(value)
            else:
                return value
        else:
            return config

    def _validate_config(self, config: Dict[str, Any]):
        """
        Validate configuration structure and values.

        Args:
            config: Configuration dictionary to validate

        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = ['strategy', 'universe', 'backtest', 'experiment']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")

        # Validate strategy configuration
        strategy = config['strategy']
        if 'name' not in strategy:
            raise ValueError("Strategy name is required")

        # Validate backtest configuration
        backtest = config['backtest']
        if 'initial_capital' not in backtest or backtest['initial_capital'] <= 0:
            raise ValueError("Initial capital must be positive")

        if 'transaction_cost' not in backtest or backtest['transaction_cost'] < 0:
            raise ValueError("Transaction cost must be non-negative")

        # Validate universe configuration
        universe = config['universe']
        if 'all_assets' not in universe or not universe['all_assets']:
            raise ValueError("Asset universe must contain at least one asset")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if file loading fails."""
        return {
            'strategy': {
                'name': 'DualMomentum',
                'lookback_days': 252,
                'top_n_assets': 5,
                'minimum_positive_assets': 3,
                'cash_ticker': 'SHY'
            },
            'universe': {
                'all_assets': ['SPY', 'QQQ', 'IWM', 'AGG', 'TLT']
            },
            'backtest': {
                'initial_capital': 1000000,
                'transaction_cost': 0.001,
                'benchmark_symbol': 'SPY',
                'start_date': '2020-01-01',
                'end_date': '2024-12-31'
            },
            'experiment': {
                'project_name': 'bloomberg-competition',
                'tags': ['dual-momentum']
            },
            'data': {
                'provider': 'yfinance',
                'retry_attempts': 3
            }
        }

    def merge_configs(self, base_config: Dict[str, Any],
                    override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.

        Args:
            base_config: Base configuration
            override_config: Override configuration

        Returns:
            Merged configuration
        """
        def merge_dict(base: Dict, override: Dict) -> Dict:
            result = base.copy()
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result

        return merge_dict(base_config, override_config)

    def create_experiment_config(self, experiment_name: str,
                                parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create configuration for a specific experiment.

        Args:
            experiment_name: Name of the experiment
            parameters: Additional parameters to include

        Returns:
            Experiment-specific configuration
        """
        config = self.load_config().copy()

        # Update experiment configuration
        config['experiment'].update({
            'name': experiment_name,
            'timestamp': str(pd.Timestamp.now())
        })

        # Add parameters
        if parameters:
            config['experiment']['parameters'] = parameters

        return config