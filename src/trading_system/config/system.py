"""
System Configuration

Configuration for the overall trading system/orchestrator.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

from .base import BaseConfig

logger = logging.getLogger(__name__)


@dataclass
class SystemConfig(BaseConfig):
    """
    System orchestrator configuration.

    Defines how multiple strategies work together, risk limits,
    and overall system behavior.
    """

    # Basic system parameters
    core_weight: float = 0.75
    satellite_weight: float = 0.25
    max_positions: int = 20
    rebalance_frequency: int = 30  # days
    risk_budget: float = 0.15  # 15% annual volatility target
    volatility_target: float = 0.12

    # Advanced parameters
    min_correlation_threshold: float = 0.7
    max_sector_allocation: float = 0.25
    ips_compliance_required: bool = True

    # Backtest parameters (for backward compatibility)
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    initial_capital: float = 1_000_000
    transaction_costs: float = 0.001
    slippage: float = 0.0005

    # System constraints
    max_leverage: float = 1.0
    enable_short_selling: bool = False
    emergency_stop_loss: float = -0.20  # Portfolio level stop loss

    # Monitoring and reporting
    enable_monitoring: bool = True
    monitoring_frequency: str = "daily"  # "hourly", "daily", "weekly"
    report_frequency: str = "monthly"

    # Strategy orchestration
    strategy_allocation_method: str = "fixed"  # "fixed", "dynamic", "risk_parity"
    enable_strategy_rotation: bool = False
    rotation_frequency: int = 90  # days

    # Risk management overrides
    override_strategy_risk_params: bool = True
    systemwide_position_limit: float = 0.10  # 10% per position max
    systemwide_sector_limits: Dict[str, float] = field(default_factory=dict)

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate weights sum to approximately 1.0
        total_weight = self.core_weight + self.satellite_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Core ({self.core_weight}) + satellite ({self.satellite_weight}) weights must sum to 1.0")

        # Validate individual weights
        if not 0 <= self.core_weight <= 1:
            raise ValueError("core_weight must be between 0 and 1")

        if not 0 <= self.satellite_weight <= 1:
            raise ValueError("satellite_weight must be between 0 and 1")

        # Validate risk parameters
        if self.max_positions <= 0:
            raise ValueError("max_positions must be positive")

        if not 0 <= self.risk_budget <= 2:  # Allow up to 200% volatility
            raise ValueError("risk_budget must be between 0 and 2")

        if not 0 <= self.volatility_target <= 2:
            raise ValueError("volatility_target must be between 0 and 2")

        # Validate correlation threshold
        if not -1 <= self.min_correlation_threshold <= 1:
            raise ValueError("min_correlation_threshold must be between -1 and 1")

        # Validate sector allocation limit
        if not 0 <= self.max_sector_allocation <= 1:
            raise ValueError("max_sector_allocation must be between 0 and 1")

        # Validate dates
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

        # Validate financial parameters
        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        if not 0 <= self.transaction_costs <= 0.01:
            raise ValueError("transaction_costs must be reasonable")

        if not 0 <= self.slippage <= 0.01:
            raise ValueError("slippage must be reasonable")

        # Validate leverage
        if self.max_leverage < 1:
            raise ValueError("max_leverage must be at least 1.0")

        # Validate emergency stop loss
        if not -1 <= self.emergency_stop_loss <= 0:
            raise ValueError("emergency_stop_loss must be between -100% and 0%")

        # Validate systemwide position limit
        if not 0 <= self.systemwide_position_limit <= 1:
            raise ValueError("systemwide_position_limit must be between 0 and 1")

        # Validate frequencies
        valid_frequencies = ["hourly", "daily", "weekly", "monthly"]
        if self.monitoring_frequency not in valid_frequencies:
            raise ValueError(f"monitoring_frequency must be one of {valid_frequencies}")

        if self.report_frequency not in valid_frequencies:
            raise ValueError(f"report_frequency must be one of {valid_frequencies}")

        # Validate allocation method
        valid_methods = ["fixed", "dynamic", "risk_parity"]
        if self.strategy_allocation_method not in valid_methods:
            raise ValueError(f"strategy_allocation_method must be one of {valid_methods}")

    def _set_defaults(self):
        """Set default values."""
        super()._set_defaults()

        # Set default dates if not provided
        if not self.start_date:
            self.start_date = datetime(2023, 1, 1)

        if not self.end_date:
            self.end_date = datetime(2023, 12, 31)

        # Set default sector limits if none provided
        if not self.systemwide_sector_limits:
            self.systemwide_sector_limits = {
                "Technology": 0.30,
                "Healthcare": 0.20,
                "Finance": 0.20,
                "Consumer": 0.15,
                "Industrial": 0.15,
                "Other": 0.10
            }

    @property
    def is_core_satellite(self) -> bool:
        """Check if this is a core+satellite configuration."""
        return 0.2 <= self.core_weight <= 0.8 and 0.2 <= self.satellite_weight <= 0.8

    @property
    def effective_position_limit(self) -> float:
        """Get the effective position limit considering strategy and system limits."""
        return min(self.systemwide_position_limit, 1.0 / self.max_positions)

    def get_sector_limit(self, sector: str) -> float:
        """Get sector allocation limit for a specific sector."""
        return self.systemwide_sector_limits.get(sector, self.max_sector_allocation)

    def set_sector_limit(self, sector: str, limit: float):
        """Set sector allocation limit."""
        if not 0 <= limit <= 1:
            raise ValueError(f"Sector limit must be between 0 and 1, got {limit}")

        self.systemwide_sector_limits[sector] = limit
        logger.info(f"Set sector limit for {sector}: {limit:.1%}")

    def check_risk_budget_compliance(self, portfolio_volatility: float) -> bool:
        """Check if portfolio volatility complies with risk budget."""
        return portfolio_volatility <= self.risk_budget

    def check_position_size_compliance(self, position_weight: float) -> bool:
        """Check if position size complies with system limits."""
        return position_weight <= self.effective_position_limit

    def get_allocation_for_strategy(self, strategy_type: str) -> float:
        """Get allocation weight for a specific strategy type."""
        if self.strategy_allocation_method == "fixed":
            if strategy_type.lower() == "core":
                return self.core_weight
            elif strategy_type.lower() == "satellite":
                return self.satellite_weight
            else:
                return 0.0
        else:
            # Dynamic allocation would be implemented here
            logger.warning(f"Dynamic allocation not implemented for {strategy_type}")
            return 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        base_summary = super().get_summary()

        base_summary.update({
            'architecture': "Core+Satellite" if self.is_core_satellite else "Custom",
            'core_satellite_split': f"{self.core_weight:.0%}/{self.satellite_weight:.0%}",
            'risk_parameters': {
                'risk_budget': f"{self.risk_budget:.1%}",
                'volatility_target': f"{self.volatility_target:.1%}",
                'emergency_stop_loss': f"{self.emergency_stop_loss:.1%}"
            },
            'position_limits': {
                'max_positions': self.max_positions,
                'position_limit': f"{self.effective_position_limit:.1%}",
                'sector_limit': f"{self.max_sector_allocation:.1%}"
            },
            'trading_parameters': {
                'rebalance_frequency': f"{self.rebalance_frequency} days",
                'rotation_enabled': self.enable_strategy_rotation,
                'short_selling': self.enable_short_selling,
                'max_leverage': f"{self.max_leverage:.1f}x"
            },
            'monitoring': {
                'enabled': self.enable_monitoring,
                'frequency': self.monitoring_frequency,
                'ips_compliance': self.ips_compliance_required
            },
            'sectors_configured': len(self.systemwide_sector_limits)
        })

        return base_summary

    