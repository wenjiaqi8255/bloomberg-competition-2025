"""
Data Format Configuration

Configuration for data format conventions and standardization settings
across the trading system.

This module defines the standard data format conventions used throughout
the system, including panel data index ordering, validation settings,
and automatic fixing behavior.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class PanelDataFormatConfig:
    """Configuration for panel data format conventions."""

    # Standard index order convention
    standard_index_order: Tuple[str, str] = field(default=('date', 'symbol'))

    # Required index levels
    required_levels: set = field(default_factory=lambda: {'date', 'symbol'})

    # Validation settings
    validate_on_input: bool = True
    validate_on_output: bool = True
    strict_validation: bool = False  # If True, raises errors instead of warnings

    # Auto-fix settings
    auto_fix_index_order: bool = True
    auto_fix_missing_levels: bool = True
    auto_fix_date_format: bool = True
    auto_fix_multiindex: bool = True

    # Quality checks
    check_empty_data: bool = True
    check_duplicate_indices: bool = True
    check_nan_levels: bool = True

    # Logging level for format operations
    log_level: str = "INFO"

    # Performance settings
    enable_validation_cache: bool = False
    max_validation_warnings: int = 10  # Stop logging warnings after this many

    def __post_init__(self):
        """Validate configuration after initialization."""
        if len(self.standard_index_order) != 2:
            raise ValueError("standard_index_order must be a tuple of exactly 2 elements")

        if not self.required_levels.issuperset(set(self.standard_index_order)):
            logger.warning(f"required_levels {self.required_levels} doesn't include standard_index_order {self.standard_index_order}")

    def get_index_order(self) -> Tuple[str, str]:
        """Get the standard index order."""
        return self.standard_index_order

    def get_date_level(self) -> str:
        """Get the name of the date index level."""
        return self.standard_index_order[0]

    def get_symbol_level(self) -> str:
        """Get the name of the symbol index level."""
        return self.standard_index_order[1]


@dataclass
class DataFormatConfig:
    """Main configuration for all data format settings."""

    # Panel data configuration
    panel_data: PanelDataFormatConfig = field(default_factory=PanelDataFormatConfig)

    # General data format settings
    enforce_global_conventions: bool = True
    enable_format_logging: bool = True

    # Component-specific settings
    standardize_pipeline_output: bool = True
    standardize_model_input: bool = True
    standardize_transformer_output: bool = True

    # Error handling
    fail_on_validation_error: bool = False
    continue_on_standardization_failure: bool = True

    # Development settings
    debug_mode: bool = False
    save_format_logs: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.debug_mode:
            self.enable_format_logging = True
            self.panel_data.log_level = "DEBUG"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataFormatConfig':
        """Create configuration from dictionary."""
        panel_config = config_dict.get('panel_data', {})

        panel_data_config = PanelDataFormatConfig(
            standard_index_order=tuple(panel_config.get('standard_index_order', ('date', 'symbol'))),
            required_levels=set(panel_config.get('required_levels', {'date', 'symbol'})),
            validate_on_input=panel_config.get('validate_on_input', True),
            validate_on_output=panel_config.get('validate_on_output', True),
            strict_validation=panel_config.get('strict_validation', False),
            auto_fix_index_order=panel_config.get('auto_fix_index_order', True),
            auto_fix_missing_levels=panel_config.get('auto_fix_missing_levels', True),
            auto_fix_date_format=panel_config.get('auto_fix_date_format', True),
            auto_fix_multiindex=panel_config.get('auto_fix_multiindex', True),
            check_empty_data=panel_config.get('check_empty_data', True),
            check_duplicate_indices=panel_config.get('check_duplicate_indices', True),
            check_nan_levels=panel_config.get('check_nan_levels', True),
            log_level=panel_config.get('log_level', 'INFO'),
            enable_validation_cache=panel_config.get('enable_validation_cache', False),
            max_validation_warnings=panel_config.get('max_validation_warnings', 10)
        )

        return cls(
            panel_data=panel_data_config,
            enforce_global_conventions=config_dict.get('enforce_global_conventions', True),
            enable_format_logging=config_dict.get('enable_format_logging', True),
            standardize_pipeline_output=config_dict.get('standardize_pipeline_output', True),
            standardize_model_input=config_dict.get('standardize_model_input', True),
            standardize_transformer_output=config_dict.get('standardize_transformer_output', True),
            fail_on_validation_error=config_dict.get('fail_on_validation_error', False),
            continue_on_standardization_failure=config_dict.get('continue_on_standardization_failure', True),
            debug_mode=config_dict.get('debug_mode', False),
            save_format_logs=config_dict.get('save_format_logs', False)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'panel_data': {
                'standard_index_order': list(self.panel_data.standard_index_order),
                'required_levels': list(self.panel_data.required_levels),
                'validate_on_input': self.panel_data.validate_on_input,
                'validate_on_output': self.panel_data.validate_on_output,
                'strict_validation': self.panel_data.strict_validation,
                'auto_fix_index_order': self.panel_data.auto_fix_index_order,
                'auto_fix_missing_levels': self.panel_data.auto_fix_missing_levels,
                'auto_fix_date_format': self.panel_data.auto_fix_date_format,
                'auto_fix_multiindex': self.panel_data.auto_fix_multiindex,
                'check_empty_data': self.panel_data.check_empty_data,
                'check_duplicate_indices': self.panel_data.check_duplicate_indices,
                'check_nan_levels': self.panel_data.check_nan_levels,
                'log_level': self.panel_data.log_level,
                'enable_validation_cache': self.panel_data.enable_validation_cache,
                'max_validation_warnings': self.panel_data.max_validation_warnings
            },
            'enforce_global_conventions': self.enforce_global_conventions,
            'enable_format_logging': self.enable_format_logging,
            'standardize_pipeline_output': self.standardize_pipeline_output,
            'standardize_model_input': self.standardize_model_input,
            'standardize_transformer_output': self.standardize_transformer_output,
            'fail_on_validation_error': self.fail_on_validation_error,
            'continue_on_standardization_failure': self.continue_on_standardization_failure,
            'debug_mode': self.debug_mode,
            'save_format_logs': self.save_format_logs
        }


# Default global configuration
DEFAULT_CONFIG = DataFormatConfig()

# Development configuration (more logging, stricter validation)
DEV_CONFIG = DataFormatConfig(
    panel_data=PanelDataFormatConfig(
        validate_on_input=True,
        validate_on_output=True,
        strict_validation=False,
        log_level="DEBUG"
    ),
    debug_mode=True,
    enable_format_logging=True,
    save_format_logs=True
)

# Production configuration (optimized for performance)
PROD_CONFIG = DataFormatConfig(
    panel_data=PanelDataFormatConfig(
        validate_on_input=False,  # Skip validation for performance
        validate_on_output=False,
        strict_validation=False,
        log_level="WARNING",
        enable_validation_cache=True,
        max_validation_warnings=3
    ),
    debug_mode=False,
    enable_format_logging=False,
    save_format_logs=False,
    continue_on_standardization_failure=True
)


def get_config(env: str = "default") -> DataFormatConfig:
    """
    Get configuration for specified environment.

    Args:
        env: Environment name ('default', 'dev', 'prod')

    Returns:
        DataFormatConfig instance
    """
    configs = {
        'default': DEFAULT_CONFIG,
        'dev': DEV_CONFIG,
        'production': PROD_CONFIG,
        'prod': PROD_CONFIG
    }

    if env not in configs:
        logger.warning(f"Unknown environment '{env}', using default configuration")
        env = 'default'

    return configs[env]


def configure_from_dict(config_dict: Dict[str, Any]) -> DataFormatConfig:
    """
    Create configuration from dictionary.

    Args:
        config_dict: Configuration dictionary

    Returns:
        DataFormatConfig instance
    """
    return DataFormatConfig.from_dict(config_dict)


def configure_from_yaml(yaml_path: str) -> DataFormatConfig:
    """
    Load configuration from YAML file.

    Args:
        yaml_path: Path to YAML configuration file

    Returns:
        DataFormatConfig instance
    """
    try:
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return configure_from_dict(config_dict)
    except ImportError:
        logger.error("PyYAML not available for loading YAML configuration")
        return DEFAULT_CONFIG
    except Exception as e:
        logger.error(f"Failed to load configuration from {yaml_path}: {e}")
        return DEFAULT_CONFIG


# Environment variable configuration
def configure_from_env() -> DataFormatConfig:
    """
    Load configuration from environment variables.

    Returns:
        DataFormatConfig instance
    """
    import os

    config_dict = {}

    # Panel data settings
    if 'PANEL_STANDARD_INDEX_ORDER' in os.environ:
        order_str = os.environ['PANEL_STANDARD_INDEX_ORDER']
        config_dict['panel_data'] = config_dict.get('panel_data', {})
        config_dict['panel_data']['standard_index_order'] = tuple(order_str.split(','))

    if 'VALIDATE_ON_INPUT' in os.environ:
        config_dict['panel_data'] = config_dict.get('panel_data', {})
        config_dict['panel_data']['validate_on_input'] = os.environ['VALIDATE_ON_INPUT'].lower() == 'true'

    if 'DEBUG_MODE' in os.environ:
        config_dict['debug_mode'] = os.environ['DEBUG_MODE'].lower() == 'true'

    return configure_from_dict(config_dict) if config_dict else DEFAULT_CONFIG