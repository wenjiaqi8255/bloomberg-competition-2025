"""
Configuration Factory (DEPRECATED)

Legacy configuration factory maintained for backward compatibility.
New code should use ConfigLoader from pydantic module.

This module is deprecated and will be removed in a future version.
"""

import warnings
import logging
from pathlib import Path
from typing import Dict, Union

from .pydantic.loader import ConfigLoader
from .pydantic.base import BasePydanticConfig as BaseConfig

logger = logging.getLogger(__name__)


class ConfigFactory:
    """
    DEPRECATED: Legacy configuration factory.
    
    Use ConfigLoader from pydantic module instead.
    """

    @staticmethod
    def from_yaml(file_path: Union[str, Path]) -> Dict[str, BaseConfig]:
        """
        DEPRECATED: Load configurations from YAML file.
        
        Use ConfigLoader from pydantic module instead.
        """
        warnings.warn(
            "ConfigFactory.from_yaml() is deprecated. "
            "Use ConfigLoader.load_from_yaml() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Forward to Pydantic loader
        loader = ConfigLoader()
        return loader.load_from_yaml(file_path)

    @staticmethod
    def load_all_configs(yaml_path: Union[str, Path]) -> Dict[str, BaseConfig]:
        """DEPRECATED: Convenient method to load all configurations from YAML."""
        return ConfigFactory.from_yaml(yaml_path)