"""
Strategy Configuration

Unified strategy configuration for all trading strategies.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

from .base import BaseConfig

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Supported strategy types."""
    DUAL_MOMENTUM = "dual_momentum"
    FAMA_FRENCH = "fama_french"
    FAMA_FRENCH_5 = "fama_french_5"
    ML = "ml"
    CORE_SATELLITE = "core_satellite"
    SATELLITE = "satellite"


@dataclass
class StrategyConfig(BaseConfig):
    """
    Unified strategy configuration.

    Provides common configuration for all strategy types while
    allowing strategy-specific parameters through the parameters dict.
    """

    # Strategy identification
    strategy_type: StrategyType = StrategyType.DUAL_MOMENTUM

    # Universe configuration
    universe: List[str] = field(default_factory=list)
    lookback_period: int = 252  # Default 1 year

    # Signal generation
    signal_threshold: float = 0.5  # Minimum signal strength to act
    enable_short_signals: bool = False

    # Position sizing
    allocation_method: str = "equal_weight"  # "equal_weight", "risk_parity", "volatility_scaled"
    max_positions: int = 20
    min_position_weight: float = 0.01  # 1% minimum

    # Risk management
    stop_loss_enabled: bool = True
    stop_loss_threshold: float = -0.10  # -10% stop loss
    position_size_limit: float = 0.15   # 15% max per position

    # Strategy-specific parameters (flexible)
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Feature engineering for ML strategies
    enable_features: bool = False
    feature_config: Optional[Dict[str, Any]] = None

    def _validate_config(self):
        """Validate configuration parameters."""
        if not self.universe:
            raise ValueError("Strategy universe cannot be empty")

        if self.lookback_period <= 0:
            raise ValueError("lookback_period must be positive")

        if not 0 <= self.signal_threshold <= 1:
            raise ValueError("signal_threshold must be between 0 and 1")

        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")

        if not 0 <= self.position_size_limit <= 1:
            raise ValueError("position_size_limit must be between 0 and 1")

        if not 0 <= self.min_position_weight <= self.position_size_limit:
            raise ValueError("min_position_weight must be less than position_size_limit")

        # Validate allocation method
        valid_methods = ["equal_weight", "risk_parity", "volatility_scaled"]
        if self.allocation_method not in valid_methods:
            raise ValueError(f"allocation_method must be one of {valid_methods}")

        # Validate stop loss threshold
        if self.stop_loss_enabled and not -1 <= self.stop_loss_threshold <= 0:
            raise ValueError("stop_loss_threshold must be between -100% and 0%")

        # Strategy-specific validation
        self._validate_strategy_specific()

    def _validate_strategy_specific(self):
        """Validate strategy-specific parameters."""
        if self.strategy_type == StrategyType.ML:
            self._validate_ml_params()
        elif self.strategy_type == StrategyType.DUAL_MOMENTUM:
            self._validate_momentum_params()
        elif self.strategy_type == StrategyType.CORE_SATELLITE:
            self._validate_core_satellite_params()

    def _validate_ml_params(self):
        """Validate ML strategy parameters."""
        required_params = ['model_type', 'feature_types']
        for param in required_params:
            if param not in self.parameters:
                logger.warning(f"ML strategy missing recommended parameter: {param}")

    def _validate_momentum_params(self):
        """Validate momentum strategy parameters."""
        # Momentum strategies typically need lookback periods
        if 'formation_period' not in self.parameters:
            self.parameters['formation_period'] = 252  # Default 1 year

        if 'holding_period' not in self.parameters:
            self.parameters['holding_period'] = 20  # Default 1 month

    def _validate_core_satellite_params(self):
        """Validate core+satellite strategy parameters."""
        if 'core_weight' not in self.parameters:
            self.parameters['core_weight'] = 0.7  # Default 70% core

        if 'satellite_weight' not in self.parameters:
            self.parameters['satellite_weight'] = 0.3  # Default 30% satellite

        core_weight = self.parameters['core_weight']
        satellite_weight = self.parameters['satellite_weight']

        if abs(core_weight + satellite_weight - 1.0) > 0.01:
            raise ValueError("Core and satellite weights must sum to 1.0")

    @property
    def effective_universe(self) -> List[str]:
        """Get the effective universe after filtering."""
        # Can be extended with universe filtering logic
        return self.universe

    @property
    def is_long_short(self) -> bool:
        """Check if strategy uses short positions."""
        return self.enable_short_signals

    def get_position_weight(self, universe_size: int) -> float:
        """Calculate position weight based on allocation method."""
        if self.allocation_method == "equal_weight":
            return min(1.0 / universe_size, self.position_size_limit)
        elif self.allocation_method == "risk_parity":
            # Simplified risk parity - actual implementation would consider volatilities
            return min(1.0 / universe_size, self.position_size_limit)
        elif self.allocation_method == "volatility_scaled":
            # Simplified volatility scaling - actual implementation would consider individual volatilities
            return min(1.0 / universe_size, self.position_size_limit)
        else:
            return self.position_size_limit

    def get_strategy_param(self, key: str, default: Any = None) -> Any:
        """Get strategy-specific parameter with default."""
        return self.parameters.get(key, default)

    def set_strategy_param(self, key: str, value: Any):
        """Set strategy-specific parameter."""
        self.parameters[key] = value
        logger.info(f"Set {self.strategy_type.value} parameter {key} = {value}")

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        base_summary = super().get_summary()

        base_summary.update({
            'strategy_type': self.strategy_type.value,
            'universe_size': len(self.universe),
            'lookback_period': f"{self.lookback_period} days",
            'allocation_method': self.allocation_method,
            'max_positions': self.max_positions,
            'position_size_limit': f"{self.position_size_limit:.1%}",
            'trading_mode': "long_short" if self.is_long_short else "long_only",
            'risk_management': {
                'stop_loss': f"{self.stop_loss_threshold:.1%}" if self.stop_loss_enabled else "disabled",
                'position_limit': f"{self.position_size_limit:.1%}"
            },
            'parameters_count': len(self.parameters),
            'features_enabled': self.enable_features
        })

        return base_summary

    