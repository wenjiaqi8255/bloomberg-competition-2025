"""
Configuration Loader

Pydantic-based configuration loader that replaces ConfigFactory.
Provides immediate validation and clear error messages.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Union
from pydantic import ValidationError

from .strategy import StrategyConfig
from .backtest import BacktestConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Pydantic-based configuration loader.
    
    Key features:
    - Immediate validation on load
    - Clear error messages
    - Zero conversion logic
    - Direct YAML mapping
    """
    
    def __init__(self):
        """Initialize the configuration loader."""
        self.logger = logger
    
    def load_from_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load configuration from YAML file with Pydantic validation.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            Dict[str, Any]: Dictionary of validated configuration objects
            
        Raises:
            FileNotFoundError: If configuration file not found
            ValidationError: If configuration validation fails
            ValueError: If configuration structure is invalid
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        # Load YAML
        try:
            with open(file_path, 'r') as f:
                config_data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML file: {e}") from e
        
        if not isinstance(config_data, dict):
            raise ValueError("Configuration file must contain a dictionary")
        
        self.logger.info(f"Loading configuration from {file_path}")
        
        # Validate and create configuration objects
        validated_configs = {}
        
        # Process strategy configuration
        if 'strategy' in config_data:
            try:
                strategy_data = config_data['strategy']
                validated_configs['strategy'] = StrategyConfig(**strategy_data)
                self.logger.info("✅ Strategy configuration validated")
            except ValidationError as e:
                error_msg = self._format_validation_error(e, "strategy")
                raise ValueError(f"Strategy configuration validation failed:\n{error_msg}") from e
        
        # Process backtest configuration
        if 'backtest' in config_data:
            try:
                backtest_data = config_data['backtest']
                validated_configs['backtest'] = BacktestConfig(**backtest_data)
                self.logger.info("✅ Backtest configuration validated")
            except ValidationError as e:
                error_msg = self._format_validation_error(e, "backtest")
                raise ValueError(f"Backtest configuration validation failed:\n{error_msg}") from e
        
        # Include other configuration sections as-is (for now)
        for key, value in config_data.items():
            if key not in ['strategy', 'backtest']:
                validated_configs[key] = value
                self.logger.info(f"✅ {key} configuration included (not validated)")
        
        self.logger.info(f"Successfully loaded {len(validated_configs)} configuration sections")
        return validated_configs
    
    def _format_validation_error(self, error: ValidationError, section: str) -> str:
        """
        Format Pydantic validation error into user-friendly message.
        
        Args:
            error: Pydantic ValidationError
            section: Configuration section name
            
        Returns:
            str: Formatted error message
        """
        lines = [f"Configuration validation failed for '{section}':"]
        
        for err in error.errors():
            # Build field path
            field_path = " -> ".join(str(x) for x in err['loc'])
            if field_path:
                field_path = f"{section}.{field_path}"
            else:
                field_path = section
            
            # Get error message
            msg = err['msg']
            
            # Add context
            if 'type' in err:
                expected_type = err['type']
                lines.append(f"  • {field_path}: {msg}")
                lines.append(f"    Expected type: {expected_type}")
            else:
                lines.append(f"  • {field_path}: {msg}")
            
            # Add input value if available
            if 'input' in err:
                input_value = err['input']
                if isinstance(input_value, str) and len(input_value) > 50:
                    input_value = input_value[:50] + "..."
                lines.append(f"    Input value: {input_value}")
        
        return "\n".join(lines)
    
    def validate_config_file(self, file_path: Union[str, Path]) -> bool:
        """
        Validate configuration file without loading.
        
        Args:
            file_path: Path to YAML configuration file
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            self.load_from_yaml(file_path)
            return True
        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            return False
    
    def get_config_summary(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get summary of loaded configuration.
        
        Args:
            config: Loaded configuration dictionary
            
        Returns:
            Dict[str, Any]: Configuration summary
        """
        summary = {
            'total_sections': len(config),
            'sections': list(config.keys()),
            'validated_sections': []
        }
        
        for key, value in config.items():
            if hasattr(value, 'get_summary'):
                summary['validated_sections'].append(key)
                summary[f'{key}_summary'] = value.get_summary()
        
        return summary
