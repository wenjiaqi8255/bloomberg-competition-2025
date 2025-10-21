"""
System Configuration (Pydantic)

System-level configuration using Pydantic for validation.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

from .base import BasePydanticConfig


class SystemConfig(BasePydanticConfig):
    """
    System orchestrator configuration using Pydantic.
    
    Provides validation for system-level parameters.
    """
    
    # Basic system parameters
    core_weight: float = Field(default=0.75, ge=0.0, le=1.0, description="Core strategy weight")
    satellite_weight: float = Field(default=0.25, ge=0.0, le=1.0, description="Satellite strategy weight")
    max_positions: int = Field(default=20, ge=1, le=1000, description="Maximum number of positions")
    rebalance_frequency: int = Field(default=30, ge=1, le=365, description="Rebalance frequency in days")
    
    # Risk parameters
    risk_budget: float = Field(default=0.15, ge=0.0, le=2.0, description="Risk budget (volatility limit)")
    volatility_target: float = Field(default=0.12, ge=0.0, le=2.0, description="Target volatility")
    
    # Advanced parameters
    min_correlation_threshold: float = Field(default=0.7, ge=-1.0, le=1.0, description="Minimum correlation threshold")
    max_sector_allocation: float = Field(default=0.25, ge=0.0, le=1.0, description="Maximum sector allocation")
    ips_compliance_required: bool = Field(default=True, description="IPS compliance required")
    
    # Backtest parameters
    start_date: Optional[datetime] = Field(default=None, description="Start date")
    end_date: Optional[datetime] = Field(default=None, description="End date")
    initial_capital: float = Field(default=1_000_000, ge=0, description="Initial capital")
    transaction_costs: float = Field(default=0.001, ge=0.0, le=0.01, description="Transaction costs")
    slippage: float = Field(default=0.0005, ge=0.0, le=0.01, description="Slippage rate")
    
    # System constraints
    max_leverage: float = Field(default=1.0, ge=1.0, le=10.0, description="Maximum leverage")
    enable_short_selling: bool = Field(default=False, description="Enable short selling")
    emergency_stop_loss: float = Field(default=-0.20, ge=-1.0, le=0.0, description="Emergency stop loss")
    
    # Strategy orchestration
    strategies: List[Dict[str, Any]] = Field(default_factory=list, description="Strategy configurations")
    strategy_allocations: Dict[str, float] = Field(default_factory=dict, description="Strategy allocations")
    max_signals_per_day: int = Field(default=100, ge=1, le=10000, description="Maximum signals per day")
    signal_conflict_resolution: str = Field(default="priority", description="Signal conflict resolution method")
    min_signal_strength: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum signal strength")
    max_position_size: float = Field(default=0.1, ge=0.0, le=1.0, description="Maximum position size")
    capacity_scaling: bool = Field(default=True, description="Enable capacity scaling")
    
    # Capital allocation
    rebalance_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Rebalance threshold")
    max_single_position_weight: float = Field(default=0.1, ge=0.0, le=1.0, description="Max single position weight")
    cash_buffer_weight: float = Field(default=0.05, ge=0.0, le=1.0, description="Cash buffer weight")
    
    # Compliance parameters
    box_exposure_limits: Dict[str, float] = Field(default_factory=dict, description="Box exposure limits")
    max_concentration_top5: float = Field(default=0.4, ge=0.0, le=1.0, description="Max concentration in top 5")
    max_concentration_top10: float = Field(default=0.6, ge=0.0, le=1.0, description="Max concentration in top 10")
    
    # Execution parameters
    max_order_size_percent: float = Field(default=0.1, ge=0.0, le=1.0, description="Max order size percentage")
    min_order_size_usd: float = Field(default=1000, ge=0, description="Minimum order size in USD")
    max_positions_per_day: int = Field(default=50, ge=1, le=1000, description="Max positions per day")
    commission_rate: float = Field(default=0.001, ge=0.0, le=0.01, description="Commission rate")
    cooling_period_hours: int = Field(default=1, ge=0, le=24, description="Cooling period in hours")
    default_order_type: str = Field(default="market", description="Default order type")
    expected_slippage_bps: float = Field(default=5.0, ge=0.0, le=100.0, description="Expected slippage in bps")
    
    # Reporting parameters
    daily_reports: bool = Field(default=True, description="Enable daily reports")
    weekly_reports: bool = Field(default=True, description="Enable weekly reports")
    monthly_reports: bool = Field(default=True, description="Enable monthly reports")
    benchmark_symbol: str = Field(default="SPY", description="Benchmark symbol")
    file_format: str = Field(default="csv", description="File format")
    output_directory: str = Field(default="./results", description="Output directory")
    
    # Portfolio construction
    portfolio_construction: Dict[str, Any] = Field(default_factory=dict, description="Portfolio construction config")
    
    # Monitoring and reporting
    enable_monitoring: bool = Field(default=True, description="Enable monitoring")
    monitoring_frequency: str = Field(default="daily", description="Monitoring frequency")
    report_frequency: str = Field(default="monthly", description="Report frequency")
    
    # Strategy orchestration
    strategy_allocation_method: str = Field(default="fixed", description="Strategy allocation method")
    enable_strategy_rotation: bool = Field(default=False, description="Enable strategy rotation")
    rotation_frequency: int = Field(default=90, ge=1, le=365, description="Rotation frequency in days")
    
    # Risk management overrides
    override_strategy_risk_params: bool = Field(default=True, description="Override strategy risk params")
    systemwide_position_limit: float = Field(default=0.10, ge=0.0, le=1.0, description="Systemwide position limit")
    systemwide_sector_limits: Dict[str, float] = Field(default_factory=dict, description="Systemwide sector limits")
    
    @model_validator(mode='after')
    def validate_weights(self):
        """Validate that weights sum to approximately 1.0."""
        total_weight = self.core_weight + self.satellite_weight
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Core ({self.core_weight}) + satellite ({self.satellite_weight}) weights must sum to 1.0")
        return self
    
    @model_validator(mode='after')
    def validate_dates(self):
        """Validate date ranges."""
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")
        return self
    
    @field_validator('signal_conflict_resolution')
    @classmethod
    def validate_signal_conflict_resolution(cls, v):
        """Validate signal conflict resolution method."""
        valid_methods = ["priority", "weighted", "latest"]
        if v not in valid_methods:
            raise ValueError(f"signal_conflict_resolution must be one of {valid_methods}")
        return v
    
    @field_validator('monitoring_frequency', 'report_frequency')
    @classmethod
    def validate_frequencies(cls, v):
        """Validate frequency fields."""
        valid_frequencies = ["hourly", "daily", "weekly", "monthly"]
        if v not in valid_frequencies:
            raise ValueError(f"Frequency must be one of {valid_frequencies}")
        return v
    
    @field_validator('strategy_allocation_method')
    @classmethod
    def validate_allocation_method(cls, v):
        """Validate allocation method."""
        valid_methods = ["fixed", "dynamic", "risk_parity"]
        if v not in valid_methods:
            raise ValueError(f"strategy_allocation_method must be one of {valid_methods}")
        return v
    
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
