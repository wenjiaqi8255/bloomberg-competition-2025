"""
Configuration validator for portfolio construction.

Provides validation for portfolio construction configurations to ensure
they are properly formatted and contain valid parameters before
attempting to create portfolio builders.
"""

import logging
from typing import Dict, Any, List, Tuple, Union

logger = logging.getLogger(__name__)


class PortfolioConfigValidator:
    """
    Validates portfolio construction configurations.

    Performs comprehensive validation of configuration dictionaries
    including structure validation, parameter validation, and
    business logic validation.
    """

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a portfolio construction configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Check top-level structure
        if not isinstance(config, dict):
            errors.append("Configuration must be a dictionary")
            return False, errors

        # Validate method selection
        method = config.get('method')
        if not method:
            errors.append("Missing required field: 'method'")
        elif method not in ['quantitative', 'box_based']:
            errors.append(f"Invalid method '{method}'. Must be 'quantitative' or 'box_based'")

        # Validate method-specific configurations
        if method == 'quantitative':
            errors.extend(PortfolioConfigValidator._validate_quantitative_config(config))
        elif method == 'box_based':
            errors.extend(PortfolioConfigValidator._validate_box_based_config(config))

        is_valid = len(errors) == 0
        if not is_valid:
            logger.warning(f"Configuration validation failed: {errors}")

        return is_valid, errors

    @staticmethod
    def _validate_quantitative_config(config: Dict[str, Any]) -> List[str]:
        """Validate quantitative method configuration."""
        errors = []

        # Validate universe_size
        universe_size = config.get('universe_size')
        if universe_size is None:
            errors.append("Quantitative config missing 'universe_size'")
        elif not isinstance(universe_size, int) or universe_size <= 0:
            errors.append("'universe_size' must be a positive integer")

        # Validate optimizer config
        optimizer_config = config.get('optimizer', {})
        if not isinstance(optimizer_config, dict):
            errors.append("'optimizer' must be a dictionary")
        else:
            errors.extend(PortfolioConfigValidator._validate_optimizer_config(optimizer_config))

        # Validate box_limits if present
        box_limits = config.get('box_limits')
        if box_limits is not None:
            errors.extend(PortfolioConfigValidator._validate_box_limits(box_limits))

        # Validate optional box sampling config
        if config.get('use_box_sampling', False):
            box_sampling = config.get('box_sampling', {})
            errors.extend(PortfolioConfigValidator._validate_box_sampling_config(box_sampling))

        return errors

    @staticmethod
    def _validate_box_based_config(config: Dict[str, Any]) -> List[str]:
        """Validate box-based method configuration."""
        errors = []

        # Validate box_weights config
        box_weights_config = config.get('box_weights', {})
        errors.extend(PortfolioConfigValidator._validate_box_weights_config(box_weights_config))

        # Validate stocks_per_box
        stocks_per_box = config.get('stocks_per_box')
        if stocks_per_box is None:
            errors.append("Box-based config missing 'stocks_per_box'")
        elif not isinstance(stocks_per_box, int) or stocks_per_box <= 0:
            errors.append("'stocks_per_box' must be a positive integer")

        # Validate min_stocks_per_box
        min_stocks = config.get('min_stocks_per_box')
        if min_stocks is not None:
            if not isinstance(min_stocks, int) or min_stocks <= 0:
                errors.append("'min_stocks_per_box' must be a positive integer")
            elif stocks_per_box and min_stocks > stocks_per_box:
                errors.append("'min_stocks_per_box' cannot be greater than 'stocks_per_box'")

        # Validate allocation_method
        allocation_method = config.get('allocation_method', 'equal')
        if allocation_method not in ['equal', 'signal_proportional']:
            errors.append(f"Invalid allocation_method '{allocation_method}'. Must be 'equal' or 'signal_proportional'")

        # Validate classifier config if present
        classifier_config = config.get('classifier')
        if classifier_config is not None:
            if not isinstance(classifier_config, dict):
                errors.append("'classifier' must be a dictionary")

        return errors

    @staticmethod
    def _validate_optimizer_config(config: Dict[str, Any]) -> List[str]:
        """Validate optimizer configuration."""
        errors = []

        # Validate method
        method = config.get('method')
        if not method:
            errors.append("Optimizer config missing 'method'")
        elif method not in ['mean_variance', 'equal_weight', 'top_n']:
            errors.append(f"Invalid optimizer method '{method}'. Must be 'mean_variance', 'equal_weight', or 'top_n'")

        # Validate risk_aversion for mean_variance
        if method == 'mean_variance':
            risk_aversion = config.get('risk_aversion', 2.0)
            if not isinstance(risk_aversion, (int, float)) or risk_aversion <= 0:
                errors.append("'risk_aversion' must be a positive number for mean_variance optimization")

        # Validate top_n for top_n method
        if method == 'top_n':
            top_n = config.get('top_n', 10)
            if not isinstance(top_n, int) or top_n <= 0:
                errors.append("'top_n' must be a positive integer for top_n optimization")

        return errors

    @staticmethod
    def _validate_box_limits(box_limits: Dict[str, Any]) -> List[str]:
        """Validate box limits configuration."""
        errors = []

        if not isinstance(box_limits, dict):
            errors.append("'box_limits' must be a dictionary")
            return errors

        for dimension, limits in box_limits.items():
            if not isinstance(limits, dict):
                errors.append(f"Box limits for dimension '{dimension}' must be a dictionary")
                continue

            for box_name, limit in limits.items():
                if not isinstance(limit, (int, float)):
                    errors.append(f"Box limit for '{dimension}:{box_name}' must be a number")
                elif limit < 0 or limit > 1:
                    errors.append(f"Box limit for '{dimension}:{box_name}' must be between 0 and 1")

        return errors

    @staticmethod
    def _validate_box_sampling_config(config: Dict[str, Any]) -> List[str]:
        """Validate box sampling configuration."""
        errors = []

        # Validate sampling_method
        method = config.get('sampling_method')
        if method and method not in ['full_universe', 'box_based']:
            errors.append(f"Invalid sampling_method '{method}'. Must be 'full_universe' or 'box_based'")

        # Validate stocks_per_box
        stocks_per_box = config.get('stocks_per_box')
        if stocks_per_box is not None:
            if not isinstance(stocks_per_box, int) or stocks_per_box <= 0:
                errors.append("'stocks_per_box' must be a positive integer")

        return errors

    @staticmethod
    def _validate_box_weights_config(config: Dict[str, Any]) -> List[str]:
        """Validate box weights configuration."""
        errors = []

        if not isinstance(config, dict):
            errors.append("'box_weights' must be a dictionary")
            return errors

        method = config.get('method')
        if not method:
            errors.append("Box weights config missing 'method'")
        elif method not in ['equal', 'config']:
            errors.append(f"Invalid box weights method '{method}'. Must be 'equal' or 'config'")

        if method == 'config':
            weights = config.get('weights', [])
            if not isinstance(weights, list):
                errors.append("'weights' must be a list for config method")
            else:
                total_weight = 0.0
                for i, weight_config in enumerate(weights):
                    if not isinstance(weight_config, dict):
                        errors.append(f"Weight config at index {i} must be a dictionary")
                        continue

                    # Validate box definition
                    box = weight_config.get('box')
                    if not box:
                        errors.append(f"Weight config at index {i} missing 'box'")
                    elif not isinstance(box, list) or len(box) != 4:
                        errors.append(f"Box at index {i} must be a list of 4 elements [size, style, region, sector]")

                    # Validate weight value
                    weight = weight_config.get('weight')
                    if weight is None:
                        errors.append(f"Weight config at index {i} missing 'weight'")
                    elif not isinstance(weight, (int, float)):
                        errors.append(f"Weight at index {i} must be a number")
                    elif weight < 0:
                        errors.append(f"Weight at index {i} cannot be negative")
                    else:
                        total_weight += weight

                # Check if weights sum to 1.0 (with tolerance)
                if abs(total_weight - 1.0) > 0.01:  # 1% tolerance
                    errors.append(f"Box weights must sum to 1.0. Current sum: {total_weight:.4f}")

        elif method == 'equal':
            # For equal method, validate dimensions if provided
            dimensions = config.get('dimensions', {})
            if dimensions:
                errors.extend(PortfolioConfigValidator._validate_dimensions_config(dimensions))

        return errors

    @staticmethod
    def _validate_dimensions_config(config: Dict[str, Any]) -> List[str]:
        """Validate dimensions configuration for equal weight method."""
        errors = []

        if not isinstance(config, dict):
            errors.append("'dimensions' must be a dictionary")
            return errors

        required_dimensions = ['size', 'style', 'region', 'sector']
        for dim in required_dimensions:
            values = config.get(dim)
            if values is None:
                errors.append(f"Dimensions config missing '{dim}'")
            elif not isinstance(values, list):
                errors.append(f"'{dim}' in dimensions must be a list")
            elif len(values) == 0:
                errors.append(f"'{dim}' in dimensions cannot be empty")

        return errors

    @staticmethod
    def validate_and_fix_config(config: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate config and attempt to fix common issues.

        Args:
            config: Configuration to validate and potentially fix

        Returns:
            Tuple of (is_valid, warnings, fixed_config)
        """
        is_valid, errors = PortfolioConfigValidator.validate_config(config)

        if is_valid:
            return True, [], config

        # Try to fix common issues
        fixed_config = config.copy()
        warnings = []

        # Fix default method
        if 'method' not in fixed_config:
            fixed_config['method'] = 'quantitative'
            warnings.append("Added default method: 'quantitative'")

        # Fix default quantitative config
        if fixed_config.get('method') == 'quantitative':
            quant_config = fixed_config.setdefault('quantitative', {})
            if 'universe_size' not in quant_config:
                quant_config['universe_size'] = 100
                warnings.append("Added default universe_size: 100")

        # Fix default box_based config
        if fixed_config.get('method') == 'box_based':
            box_config = fixed_config.setdefault('box_based', {})
            if 'stocks_per_box' not in box_config:
                box_config['stocks_per_box'] = 3
                warnings.append("Added default stocks_per_box: 3")
            if 'allocation_method' not in box_config:
                box_config['allocation_method'] = 'equal'
                warnings.append("Added default allocation_method: 'equal'")

            # Fix default box_weights
            if 'box_weights' not in box_config:
                box_config['box_weights'] = {'method': 'equal'}
                warnings.append("Added default box_weights method: 'equal'")

        # Re-validate after fixes
        is_valid_after_fix, remaining_errors = PortfolioConfigValidator.validate_config(fixed_config)

        return is_valid_after_fix, warnings, fixed_config