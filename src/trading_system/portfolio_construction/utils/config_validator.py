"""
Deprecated shim for portfolio config validation.

This module re-exports the authoritative `PortfolioConfigValidator` from
`trading_system.validation.config.portfolio_validator` to preserve backward
compatibility. Please update imports to use the validation path directly.
"""

import warnings
from typing import Dict, Any, List, Tuple

from ...validation.config.portfolio_validator import PortfolioConfigValidator as _AuthoritativeValidator

warnings.warn(
    "trading_system.portfolio_construction.utils.config_validator is deprecated; "
    "import from trading_system.validation.config.portfolio_validator instead.",
    DeprecationWarning,
    stacklevel=2,
)


class PortfolioConfigValidator(_AuthoritativeValidator):
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        return _AuthoritativeValidator.validate_config(config)