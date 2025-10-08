"""
Factory for creating portfolio construction builders.

Implements the Factory pattern to create appropriate portfolio builders
based on configuration, providing a clean interface for the rest of the
system to use different construction methods.
"""

import logging
from typing import Dict, Any

from src.trading_system.portfolio_construction.interface.interfaces import IPortfolioBuilder
from src.trading_system.portfolio_construction.quant_based.quantitative_builder import QuantitativePortfolioBuilder
from src.trading_system.portfolio_construction.box_based.box_based_builder import BoxBasedPortfolioBuilder
from src.trading_system.portfolio_construction.utils.config_validator import PortfolioConfigValidator
from src.trading_system.portfolio_construction.models.exceptions import InvalidConfigError

logger = logging.getLogger(__name__)


class PortfolioBuilderFactory:
    """
    Factory for creating portfolio construction builders.

    Provides a clean interface for the system to create appropriate
    builders based on configuration while maintaining flexibility
    for future extension.
    """

    @staticmethod
    def create_builder(config: Dict[str, Any]) -> IPortfolioBuilder:
        """
        Create portfolio builder based on configuration.

        Args:
            config: Portfolio construction configuration

        Returns:
            Configured portfolio builder

        Raises:
            InvalidConfigError: If configuration is invalid
        """
        # Validate configuration first
        validator = PortfolioConfigValidator()
        is_valid, errors = validator.validate_config(config)

        if not is_valid:
            raise InvalidConfigError(
                f"Invalid portfolio construction configuration: {errors}",
                config_section='portfolio_construction',
                validation_errors=errors
            )

        # Determine construction method
        method = config.get('method', 'quantitative').lower()

        try:
            if method == 'quantitative':
                logger.info("Creating QuantitativePortfolioBuilder")
                return QuantitativePortfolioBuilder(config)

            elif method == 'box_based':
                logger.info("Creating BoxBasedPortfolioBuilder")
                return BoxBasedPortfolioBuilder(config)

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

    @staticmethod
    def get_method_info(method: str) -> Dict[str, Any]:
        """
        Get information about a portfolio construction method.

        Args:
            method: Method name

        Returns:
            Dictionary with method information
        """
        method = method.lower()

        if method == 'quantitative':
            return {
                'name': 'Quantitative',
                'description': 'Traditional quantitative optimization with optional box-aware sampling',
                'key_features': [
                    '7-stage optimization pipeline',
                    'Mathematical risk modeling',
                    'Optional box-aware universe sampling',
                    'Backward compatibility with existing strategies'
                ],
                'config_sections': ['optimizer', 'covariance', 'universe_size', 'box_limits'],
                'complexity': 'high'
            }

        elif method == 'box_based':
            return {
                'name': 'Box-Based',
                'description': 'Box-First methodology ensuring systematic box coverage',
                'key_features': [
                    '4-dimensional box classification',
                    'Systematic box weight allocation',
                    'Within-box stock selection',
                    'Guaranteed box diversification'
                ],
                'config_sections': ['box_weights', 'stocks_per_box', 'allocation_method'],
                'complexity': 'medium'
            }

        else:
            return {
                'name': 'Unknown',
                'description': f'Unknown method: {method}',
                'key_features': [],
                'config_sections': [],
                'complexity': 'unknown'
            }

    @staticmethod
    def create_sample_config(method: str) -> Dict[str, Any]:
        """
        Create a sample configuration for the specified method.

        Args:
            method: Portfolio construction method

        Returns:
            Sample configuration dictionary
        """
        method = method.lower()

        if method == 'quantitative':
            return {
                'method': 'quantitative',
                'universe_size': 100,
                'enable_short_selling': False,
                'optimizer': {
                    'method': 'mean_variance',
                    'risk_aversion': 1.0
                },
                'covariance': {
                    'lookback_days': 252,
                    'method': 'ledoit_wolf'
                },
                'classifier': {
                    'method': 'four_factor',
                    'cache_enabled': True
                },
                'box_limits': {
                    'size': 0.3,
                    'style': 0.3,
                    'region': 0.4
                },
                'use_box_sampling': False,
                'min_history_days': 252
            }

        elif method == 'box_based':
            return {
                'method': 'box_based',
                'stocks_per_box': 3,
                'min_stocks_per_box': 1,
                'allocation_method': 'equal',
                'allocation_config': {},
                'box_weights': {
                    'method': 'equal',
                    'dimensions': {
                        'size': ['large', 'mid', 'small'],
                        'style': ['growth', 'value'],
                        'region': ['developed', 'emerging'],
                        'sector': ['Technology', 'Financials', 'Healthcare',
                                  'Consumer Discretionary', 'Consumer Staples']
                    }
                },
                'classifier': {
                    'method': 'four_factor',
                    'cache_enabled': True
                },
                'box_selector': {
                    'type': 'signal_based'
                }
            }

        else:
            raise InvalidConfigError(f"Unknown method for sample config: {method}")

    @staticmethod
    def get_factory_info() -> Dict[str, Any]:
        """
        Get information about the factory itself.

        Returns:
            Factory information dictionary
        """
        return {
            'name': 'PortfolioBuilderFactory',
            'description': 'Factory for creating portfolio construction builders',
            'available_methods': PortfolioBuilderFactory.get_available_methods(),
            'validation_enabled': True,
            'error_handling': 'comprehensive',
            'logging': True
        }