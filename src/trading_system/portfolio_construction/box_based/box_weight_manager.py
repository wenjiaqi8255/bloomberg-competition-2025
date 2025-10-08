"""
Box weight management for portfolio construction.

Provides flexible box weight allocation strategies with a clean interface
that allows for different weight generation methods while maintaining
consistent behavior.
"""

import logging
from typing import Dict, List, Any, Tuple

from src.trading_system.portfolio_construction.interface.interfaces import IBoxWeightProvider
from src.trading_system.portfolio_construction.models.types import BoxKey
from src.trading_system.portfolio_construction.models.exceptions import InvalidConfigError

logger = logging.getLogger(__name__)


class EqualBoxWeightProvider(IBoxWeightProvider):
    """
    Provides equal weights for all boxes.

    MVP implementation that assigns equal weight to each box.
    Simple, robust, and serves as a baseline.
    """

    def __init__(self, box_definitions: List[BoxKey]):
        """
        Initialize equal weight provider.

        Args:
            box_definitions: List of all valid box combinations
        """
        if not box_definitions:
            raise ValueError("Box definitions cannot be empty")

        self.box_definitions = box_definitions
        self._equal_weight = 1.0 / len(box_definitions)

        logger.info(f"Initialized EqualBoxWeightProvider with {len(box_definitions)} boxes, "
                   f"each receiving {self._equal_weight:.4f} weight")

    def get_box_weights(self) -> Dict[BoxKey, float]:
        """Get equal weights for all boxes."""
        return {box: self._equal_weight for box in self.box_definitions}

    def validate_weights(self) -> bool:
        """Validate equal weights."""
        weights = self.get_box_weights()
        total_weight = sum(weights.values())

        if abs(total_weight - 1.0) > 1e-6:
            logger.error(f"Equal weights sum to {total_weight:.6f}, expected 1.0")
            return False

        if any(w < 0 for w in weights.values()):
            logger.error("Equal weights contain negative values")
            return False

        return True


class ConfigurableBoxWeightProvider(IBoxWeightProvider):
    """
    Provides box weights based on configuration file.

    Allows for custom weight assignments specified in configuration,
    giving users full control over box weight distribution.
    """

    def __init__(self, box_weights_config: List[Dict[str, Any]]):
        """
        Initialize configurable weight provider.

        Args:
            box_weights_config: List of box weight configurations
        """
        self.box_weights_config = box_weights_config
        self.box_weights = self._parse_config(box_weights_config)

        logger.info(f"Initialized ConfigurableBoxWeightProvider with {len(self.box_weights)} boxes")

    def _parse_config(self, config: List[Dict[str, Any]]) -> Dict[BoxKey, float]:
        """
        Parse box weight configuration into BoxKey mapping.

        Args:
            config: List of box weight configurations

        Returns:
            Dictionary mapping BoxKey to weight
        """
        box_weights = {}

        for i, weight_config in enumerate(config):
            try:
                # Parse box definition
                box_def = weight_config.get('box')
                if not box_def:
                    raise ValueError(f"Weight config at index {i} missing 'box' field")

                if len(box_def) != 4:
                    raise ValueError(f"Box definition must have 4 elements, got {len(box_def)}")

                box_key = BoxKey(
                    size=box_def[0],
                    style=box_def[1],
                    region=box_def[2],
                    sector=box_def[3]
                )

                # Parse weight
                weight = weight_config.get('weight')
                if weight is None:
                    raise ValueError(f"Weight config at index {i} missing 'weight' field")

                if not isinstance(weight, (int, float)):
                    raise ValueError(f"Weight must be numeric, got {type(weight)}")

                if weight < 0:
                    raise ValueError(f"Weight cannot be negative: {weight}")

                box_weights[box_key] = weight

            except Exception as e:
                raise InvalidConfigError(
                    f"Failed to parse weight config at index {i}: {e}",
                    config_section='box_weights',
                    validation_errors=[str(e)]
                )

        return box_weights

    def get_box_weights(self) -> Dict[BoxKey, float]:
        """Get configured box weights."""
        return self.box_weights.copy()

    def validate_weights(self) -> bool:
        """Validate configured weights."""
        if not self.box_weights:
            logger.error("No box weights configured")
            return False

        total_weight = sum(self.box_weights.values())

        # Check if weights sum to approximately 1.0
        if abs(total_weight - 1.0) > 0.01:  # 1% tolerance
            logger.error(f"Box weights sum to {total_weight:.4f}, expected ~1.0")
            return False

        # Check for negative weights
        negative_weights = [(box, w) for box, w in self.box_weights.items() if w < 0]
        if negative_weights:
            logger.error(f"Found negative weights: {negative_weights}")
            return False

        return True


class BoxWeightManager:
    """
    Manages box weight allocation with flexible provider interface.

    Serves as the main interface for getting box weights, supporting
    different allocation strategies through the provider pattern.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize box weight manager.

        Args:
            config: Box weight configuration dictionary
        """
        self.config = config
        self.provider = self._create_provider(config)
        self._cached_weights = None

        logger.info(f"Initialized BoxWeightManager with provider: {self.provider.__class__.__name__}")

    def _create_provider(self, config: Dict[str, Any]) -> IBoxWeightProvider:
        """
        Create appropriate box weight provider based on configuration.

        Args:
            config: Box weight configuration

        Returns:
            Configured box weight provider
        """
        method = config.get('method', 'equal')

        if method == 'equal':
            return self._create_equal_provider(config)
        elif method == 'config':
            return self._create_config_provider(config)
        else:
            raise InvalidConfigError(
                f"Unknown box weight method: {method}",
                config_section='box_weights'
            )

    def _create_equal_provider(self, config: Dict[str, Any]) -> EqualBoxWeightProvider:
        """Create equal weight provider."""
        dimensions_config = config.get('dimensions', {})
        box_definitions = self._generate_all_boxes(dimensions_config)

        if not box_definitions:
            raise InvalidConfigError(
                "No box definitions generated for equal weight method",
                config_section='box_weights'
            )

        return EqualBoxWeightProvider(box_definitions)

    def _create_config_provider(self, config: Dict[str, Any]) -> ConfigurableBoxWeightProvider:
        """Create configurable weight provider."""
        weights_config = config.get('weights', [])

        if not weights_config:
            raise InvalidConfigError(
                "No weights configuration provided for config method",
                config_section='box_weights'
            )

        return ConfigurableBoxWeightProvider(weights_config)

    def _generate_all_boxes(self, dimensions_config: Dict[str, List[str]]) -> List[BoxKey]:
        """
        Generate all possible box combinations from dimensions.

        Args:
            dimensions_config: Dictionary with dimension values

        Returns:
            List of all possible BoxKey combinations
        """
        # Default dimensions if not provided
        default_dimensions = {
            'size': ['large', 'mid', 'small'],
            'style': ['growth', 'value'],
            'region': ['developed', 'emerging'],
            'sector': ['Technology', 'Financials', 'Healthcare', 'Consumer Discretionary',
                      'Consumer Staples', 'Industrials', 'Energy', 'Utilities', 'Real Estate',
                      'Materials', 'Communication Services']
        }

        # Use provided dimensions or defaults
        dimensions = {
            'size': dimensions_config.get('size', default_dimensions['size']),
            'style': dimensions_config.get('style', default_dimensions['style']),
            'region': dimensions_config.get('region', default_dimensions['region']),
            'sector': dimensions_config.get('sector', default_dimensions['sector'])
        }

        # Validate dimensions
        for dim_name, values in dimensions.items():
            if not values:
                logger.warning(f"Empty dimension '{dim_name}', using default")
                dimensions[dim_name] = default_dimensions[dim_name]

        # Generate all combinations
        import itertools
        box_definitions = []

        for size, style, region, sector in itertools.product(
            dimensions['size'], dimensions['style'],
            dimensions['region'], dimensions['sector']
        ):
            box_definitions.append(BoxKey(
                size=size, style=style, region=region, sector=sector
            ))

        logger.info(f"Generated {len(box_definitions)} box combinations from dimensions")
        return box_definitions

    def get_target_weight(self, box_key: BoxKey) -> float:
        """
        Get target weight for a specific box.

        Args:
            box_key: Box to get weight for

        Returns:
            Target weight for the box (0.0 if box not found)
        """
        if self._cached_weights is None:
            self._cached_weights = self.provider.get_box_weights()

        return self._cached_weights.get(box_key, 0.0)

    def get_target_boxes(self) -> List[BoxKey]:
        """Get list of all target boxes."""
        if self._cached_weights is None:
            self._cached_weights = self.provider.get_box_weights()

        return list(self._cached_weights.keys())

    def get_all_weights(self) -> Dict[BoxKey, float]:
        """Get all target box weights."""
        if self._cached_weights is None:
            self._cached_weights = self.provider.get_box_weights()

        return self._cached_weights.copy()

    def validate_configuration(self) -> Tuple[bool, List[str]]:
        """
        Validate the box weight configuration.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            is_valid = self.provider.validate_weights()
            if is_valid:
                return True, []
            else:
                return False, ["Box weight provider validation failed"]
        except Exception as e:
            return False, [f"Box weight validation error: {e}"]

    def get_coverage_info(self, available_boxes: List[BoxKey]) -> Dict[str, Any]:
        """
        Get information about box coverage.

        Args:
            available_boxes: List of boxes that have stocks available

        Returns:
            Dictionary with coverage statistics
        """
        target_boxes = self.get_target_boxes()
        target_weights = self.get_all_weights()

        available_set = set(available_boxes)
        target_set = set(target_boxes)

        covered_boxes = list(target_set.intersection(available_set))
        uncovered_boxes = list(target_set - available_set)

        covered_weight = sum(target_weights[box] for box in covered_boxes)
        uncovered_weight = sum(target_weights[box] for box in uncovered_boxes)

        return {
            'total_target_boxes': len(target_boxes),
            'available_boxes': len(available_boxes),
            'covered_boxes': len(covered_boxes),
            'uncovered_boxes': len(uncovered_boxes),
            'coverage_ratio': len(covered_boxes) / len(target_boxes) if target_boxes else 0,
            'covered_weight': covered_weight,
            'uncovered_weight': uncovered_weight,
            'weight_coverage': covered_weight,
            'covered_box_list': covered_boxes,
            'uncovered_box_list': uncovered_boxes
        }

    def get_manager_info(self) -> Dict[str, Any]:
        """Get information about the weight manager."""
        return {
            'provider_type': self.provider.__class__.__name__,
            'config': self.config,
            'total_boxes': len(self.get_target_boxes()),
            'total_weight': sum(self.get_all_weights().values()),
            'is_valid': self.validate_configuration()[0]
        }