"""
Portfolio Configuration Validator
================================

Validates portfolio construction configurations.
Enhanced version of the original PortfolioConfigValidator with better error reporting.
"""

import logging
from typing import Dict, Any, List, Tuple, Union

from ..base import BaseValidator, ValidationResult, ValidationIssue, ValidationSeverity

logger = logging.getLogger(__name__)


class PortfolioConfigValidator(BaseValidator):
    """
    Validates portfolio construction configurations.
    
    Enhanced version with better error reporting and validation result structure.
    Maintains backward compatibility with the original validator.
    """
    
    def __init__(self):
        super().__init__("PortfolioConfigValidator")
    
    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate a portfolio construction configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            ValidationResult with validation outcome and detailed issues
        """
        self.log_validation_start("portfolio construction configuration")
        result = ValidationResult()
        
        # Check top-level structure
        if not isinstance(config, dict):
            result.add_error("Configuration must be a dictionary")
            return result
        
        # Validate method selection
        method = config.get('method')
        if not method:
            result.add_error("Missing required field: 'method'", 
                           suggestion="Set method to 'quantitative' or 'box_based'")
        elif method not in ['quantitative', 'box_based']:
            result.add_error(f"Invalid method '{method}'. Must be 'quantitative' or 'box_based'")
        else:
            # Validate method-specific configurations
            if method == 'quantitative':
                self._validate_quantitative_config(config, result)
            elif method == 'box_based':
                self._validate_box_based_config(config, result)
        
        self.log_validation_complete(result)
        return result
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Legacy method for backward compatibility.
        
        Provides a simple boolean-and-messages API that delegates to the
        authoritative validator implementation.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        validator = PortfolioConfigValidator()
        result = validator.validate(config)
        error_messages = [issue.message for issue in result.get_errors()]
        return result.is_valid, error_messages
    
    def _validate_quantitative_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate quantitative method configuration."""
        # Validate universe_size
        universe_size = config.get('universe_size')
        if universe_size is None:
            result.add_error("Quantitative config missing 'universe_size'", 
                           suggestion="Set universe_size to a positive integer")
        elif not isinstance(universe_size, int) or universe_size <= 0:
            result.add_error("'universe_size' must be a positive integer")
        
        # Validate optimizer config
        optimizer_config = config.get('optimizer', {})
        if not isinstance(optimizer_config, dict):
            result.add_error("'optimizer' must be a dictionary")
        else:
            self._validate_optimizer_config(optimizer_config, result)
        
        # Validate box_limits if present
        box_limits = config.get('box_limits')
        if box_limits is not None:
            self._validate_box_limits(box_limits, result)
        
        # Validate optional box sampling config
        if config.get('use_box_sampling', False):
            box_sampling = config.get('box_sampling', {})
            self._validate_box_sampling_config(box_sampling, result)
    
    def _validate_box_based_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate box-based method configuration."""
        # Validate box_weights config
        box_weights_config = config.get('box_weights', {})
        self._validate_box_weights_config(box_weights_config, result)
        
        # Validate stocks_per_box
        stocks_per_box = config.get('stocks_per_box')
        if stocks_per_box is None:
            result.add_error("Box-based config missing 'stocks_per_box'",
                           suggestion="Set stocks_per_box to a positive integer")
        elif not isinstance(stocks_per_box, int) or stocks_per_box <= 0:
            result.add_error("'stocks_per_box' must be a positive integer")
        
        # Validate min_stocks_per_box
        min_stocks = config.get('min_stocks_per_box')
        if min_stocks is not None:
            if not isinstance(min_stocks, int) or min_stocks <= 0:
                result.add_error("'min_stocks_per_box' must be a positive integer")
            elif stocks_per_box and min_stocks > stocks_per_box:
                result.add_error("'min_stocks_per_box' cannot be greater than 'stocks_per_box'")
        
        # Validate allocation_method
        allocation_method = config.get('allocation_method', 'equal')
        if allocation_method not in ['equal', 'signal_proportional']:
            result.add_error(f"Invalid allocation_method '{allocation_method}'. Must be 'equal' or 'signal_proportional'")
        
        # Validate classifier config if present
        classifier_config = config.get('classifier')
        if classifier_config is not None:
            if not isinstance(classifier_config, dict):
                result.add_error("'classifier' must be a dictionary")
    
    def _validate_optimizer_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate optimizer configuration."""
        # Validate method
        method = config.get('method')
        if not method:
            result.add_error("Optimizer config missing 'method'",
                           suggestion="Set method to 'mean_variance', 'equal_weight', or 'top_n'")
        elif method not in ['mean_variance', 'equal_weight', 'top_n']:
            result.add_error(f"Invalid optimizer method '{method}'. Must be 'mean_variance', 'equal_weight', or 'top_n'")
        
        # Validate risk_aversion for mean_variance
        if method == 'mean_variance':
            risk_aversion = config.get('risk_aversion', 2.0)
            if not isinstance(risk_aversion, (int, float)) or risk_aversion <= 0:
                result.add_error("'risk_aversion' must be a positive number for mean_variance optimization")
        
        # Validate top_n for top_n method
        if method == 'top_n':
            top_n = config.get('top_n', 10)
            if not isinstance(top_n, int) or top_n <= 0:
                result.add_error("'top_n' must be a positive integer for top_n optimization")
    
    def _validate_box_limits(self, box_limits: Dict[str, Any], result: ValidationResult):
        """Validate box limits configuration."""
        if not isinstance(box_limits, dict):
            result.add_error("'box_limits' must be a dictionary")
            return
        
        for dimension, limits in box_limits.items():
            if not isinstance(limits, dict):
                result.add_error(f"Box limits for dimension '{dimension}' must be a dictionary")
                continue
            
            for box_name, limit in limits.items():
                if not isinstance(limit, (int, float)):
                    result.add_error(f"Box limit for '{dimension}:{box_name}' must be a number")
                elif limit < 0 or limit > 1:
                    result.add_error(f"Box limit for '{dimension}:{box_name}' must be between 0 and 1")
    
    def _validate_box_sampling_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate box sampling configuration."""
        # Validate sampling_method
        method = config.get('sampling_method')
        if method and method not in ['full_universe', 'box_based']:
            result.add_error(f"Invalid sampling_method '{method}'. Must be 'full_universe' or 'box_based'")
        
        # Validate stocks_per_box
        stocks_per_box = config.get('stocks_per_box')
        if stocks_per_box is not None:
            if not isinstance(stocks_per_box, int) or stocks_per_box <= 0:
                result.add_error("'stocks_per_box' must be a positive integer")
    
    def _validate_box_weights_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate box weights configuration."""
        if not isinstance(config, dict):
            result.add_error("'box_weights' must be a dictionary")
            return
        
        method = config.get('method')
        if not method:
            result.add_error("Box weights config missing 'method'",
                           suggestion="Set method to 'equal' or 'config'")
        elif method not in ['equal', 'config']:
            result.add_error(f"Invalid box weights method '{method}'. Must be 'equal' or 'config'")
        
        if method == 'config':
            weights = config.get('weights', [])
            if not isinstance(weights, list):
                result.add_error("'weights' must be a list for config method")
            else:
                total_weight = 0.0
                for i, weight_config in enumerate(weights):
                    if not isinstance(weight_config, dict):
                        result.add_error(f"Weight config at index {i} must be a dictionary")
                        continue
                    
                    # Validate box definition
                    box = weight_config.get('box')
                    if not box:
                        result.add_error(f"Weight config at index {i} missing 'box'")
                    elif not isinstance(box, list) or len(box) != 4:
                        result.add_error(f"Box at index {i} must be a list of 4 elements [size, style, region, sector]")
                    
                    # Validate weight value
                    weight = weight_config.get('weight')
                    if weight is None:
                        result.add_error(f"Weight config at index {i} missing 'weight'")
                    elif not isinstance(weight, (int, float)):
                        result.add_error(f"Weight at index {i} must be a number")
                    elif weight < 0:
                        result.add_error(f"Weight at index {i} cannot be negative")
                    else:
                        total_weight += weight
                
                # Check if weights sum to 1.0 (with tolerance)
                if abs(total_weight - 1.0) > 0.01:  # 1% tolerance
                    result.add_error(f"Box weights must sum to 1.0. Current sum: {total_weight:.4f}")
        
        elif method == 'equal':
            # For equal method, validate dimensions if provided
            dimensions = config.get('dimensions', {})
            if dimensions:
                self._validate_dimensions_config(dimensions, result)
    
    def _validate_dimensions_config(self, config: Dict[str, Any], result: ValidationResult):
        """Validate dimensions configuration for equal weight method."""
        if not isinstance(config, dict):
            result.add_error("'dimensions' must be a dictionary")
            return
        
        required_dimensions = ['size', 'style', 'region', 'sector']
        for dim in required_dimensions:
            values = config.get(dim)
            if values is None:
                result.add_error(f"Dimensions config missing '{dim}'")
            elif not isinstance(values, list):
                result.add_error(f"'{dim}' in dimensions must be a list")
            elif len(values) == 0 and dim != 'sector':  # Allow sector to be empty for 3D boxes
                result.add_error(f"'{dim}' in dimensions cannot be empty")
            # sector can be empty - this creates 3-dimensional boxes without sector classification
    
    def validate_and_fix_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate config and attempt to fix common issues.
        
        Args:
            config: Configuration to validate and potentially fix
            
        Returns:
            Tuple of (is_valid, warnings, fixed_config)
        """
        result = self.validate(config)
        
        if result.is_valid:
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
            if 'universe_size' not in fixed_config:
                fixed_config['universe_size'] = 100
                warnings.append("Added default universe_size: 100")
        
        # Fix default box_based config
        if fixed_config.get('method') == 'box_based':
            if 'stocks_per_box' not in fixed_config:
                fixed_config['stocks_per_box'] = 3
                warnings.append("Added default stocks_per_box: 3")
            if 'allocation_method' not in fixed_config:
                fixed_config['allocation_method'] = 'equal'
                warnings.append("Added default allocation_method: 'equal'")
            
            # Fix default box_weights
            if 'box_weights' not in fixed_config:
                fixed_config['box_weights'] = {'method': 'equal'}
                warnings.append("Added default box_weights method: 'equal'")
        
        # Re-validate after fixes
        result_after_fix = self.validate(fixed_config)
        
        return result_after_fix.is_valid, warnings, fixed_config
