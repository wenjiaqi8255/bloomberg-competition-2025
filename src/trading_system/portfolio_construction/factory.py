"""
Factory for creating portfolio construction builders.

Implements the Factory pattern to create appropriate portfolio builders
based on configuration, providing a clean interface for the rest of the
system to use different construction methods.
"""

import logging
from typing import Dict, Any, Union

from src.trading_system.portfolio_construction.interface.interfaces import IPortfolioBuilder
from src.trading_system.portfolio_construction.quant_based.quantitative_builder import QuantitativePortfolioBuilder
from src.trading_system.portfolio_construction.box_based.box_based_builder import BoxBasedPortfolioBuilder
from src.trading_system.validation.config.portfolio_validator import PortfolioConfigValidator
from src.trading_system.portfolio_construction.models.exceptions import InvalidConfigError
from src.trading_system.config.pydantic.portfolio import (
    BoxBasedPortfolioConfig,
    QuantitativePortfolioConfig
)

logger = logging.getLogger(__name__)


class PortfolioBuilderFactory:
    """
    Factory for creating portfolio construction builders.

    Provides a clean interface for the system to create appropriate
    builders based on configuration while maintaining flexibility
    for future extension.
    """

    @staticmethod
    def create_builder(config: Union[Dict[str, Any], BoxBasedPortfolioConfig, QuantitativePortfolioConfig],
                      factor_data_provider=None) -> IPortfolioBuilder:
        """
        Create portfolio builder based on configuration.

        Args:
            config: Portfolio construction configuration (dict or Pydantic config)
            factor_data_provider: Optional factor data provider for factor model covariance

        Returns:
            Configured portfolio builder

        Raises:
            InvalidConfigError: If configuration is invalid
        """
        # Handle Pydantic config objects
        if isinstance(config, (BoxBasedPortfolioConfig, QuantitativePortfolioConfig)):
            config_dict = config.model_dump()
        else:
            # Handle dict config (legacy support)
            config_dict = config
            
            # Validate configuration first
            validator = PortfolioConfigValidator()
            is_valid, errors = validator.validate_config(config_dict)

            if not is_valid:
                raise InvalidConfigError(
                    f"Invalid portfolio construction configuration: {errors}",
                    config_section='portfolio_construction',
                    validation_errors=errors
                )

        try:
            # Use type checking for better type safety
            if isinstance(config, BoxBasedPortfolioConfig):
                logger.info("Creating BoxBasedPortfolioBuilder")
                return BoxBasedPortfolioBuilder(config_dict, factor_data_provider=factor_data_provider)

            elif isinstance(config, QuantitativePortfolioConfig):
                logger.info("Creating QuantitativePortfolioBuilder")
                return QuantitativePortfolioBuilder(config_dict, factor_data_provider=factor_data_provider)

            else:
                # Fallback to method-based dispatch for dict configs
                method = config_dict.get('method', 'quantitative').lower()
                
                if method == 'quantitative':
                    logger.info("Creating QuantitativePortfolioBuilder")
                    return QuantitativePortfolioBuilder(config_dict, factor_data_provider=factor_data_provider)

                elif method == 'box_based':
                    logger.info("Creating BoxBasedPortfolioBuilder")
                    return BoxBasedPortfolioBuilder(config_dict, factor_data_provider=factor_data_provider)

                else:
                    raise InvalidConfigError(
                        f"Unknown portfolio construction method: {method}",
                        config_section='method'
                    )

        except Exception as e:
            if isinstance(e, InvalidConfigError):
                raise

            logger.error(f"Failed to create portfolio builder: {e}")
            raise InvalidConfigError(
                f"Portfolio builder creation failed: {e}",
                config_section='portfolio_construction'
            )

    @staticmethod
    def get_available_methods() -> list[str]:
        """
        Get list of available portfolio construction methods.

        Returns:
            List of method names
        """
        return ['quantitative', 'box_based']

    @staticmethod
    def validate_method(method: str) -> bool:
        """
        Validate portfolio construction method name.

        Args:
            method: Method name to validate

        Returns:
            True if method is valid, False otherwise
        """
        return method.lower() in PortfolioBuilderFactory.get_available_methods()