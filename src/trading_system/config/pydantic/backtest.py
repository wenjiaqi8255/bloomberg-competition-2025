"""
Backtest Configuration

Pydantic configuration for backtesting parameters.
Directly maps YAML backtest section with automatic validation.
"""

from datetime import datetime
from typing import Optional, List, Literal
from pydantic import BaseModel, Field, field_validator
from .base import BasePydanticConfig


class BacktestConfig(BasePydanticConfig):
    """
    Backtest configuration.
    
    Directly maps YAML backtest section.
    """
    
    # Core parameters
    start_date: datetime = Field(description="Backtest start date")
    end_date: datetime = Field(description="Backtest end date")
    initial_capital: float = Field(gt=0, description="Initial capital amount")
    
    # Benchmark
    benchmark_symbol: Optional[str] = Field(default=None, description="Benchmark symbol")
    
    # Transaction costs
    commission_rate: float = Field(ge=0, le=0.01, default=0.001, description="Commission rate")
    spread_rate: float = Field(ge=0, le=0.01, default=0.0005, description="Spread rate")
    slippage_rate: float = Field(ge=0, le=0.01, default=0.0005, description="Slippage rate")
    short_borrow_rate: float = Field(ge=0, le=0.1, default=0.02, description="Short borrowing rate")
    
    # Trading parameters
    rebalance_frequency: Literal["daily", "weekly", "monthly", "quarterly"] = Field(
        default="monthly", 
        description="Rebalancing frequency"
    )
    position_limit: float = Field(ge=0, le=1, default=0.1, description="Maximum position weight")
    rebalance_threshold: float = Field(ge=0, le=0.1, default=0.01, description="Rebalance threshold")
    enable_short_selling: bool = Field(default=False, description="Enable short selling")
    
    # Risk management
    max_drawdown_limit: Optional[float] = Field(default=None, ge=0, le=1, description="Maximum drawdown limit")
    volatility_limit: Optional[float] = Field(default=None, ge=0, le=2, description="Volatility limit")
    
    # Risk-free rate for Sharpe ratio calculation
    risk_free_rate: float = Field(default=0.02, ge=0, le=0.1, description="Risk-free rate for Sharpe ratio")
    
    # Results saving
    save_results: bool = Field(default=True, description="Whether to save backtest results")
    output_directory: Optional[str] = Field(default=None, description="Output directory for results")
    
    # Additional parameters
    parameters: dict = Field(default_factory=dict, description="Additional backtest parameters")
    
    class Config:
        extra = "forbid"
        validate_assignment = True
    
    @field_validator('end_date')
    @classmethod
    def validate_end_date(cls, v, info):
        """Validate end_date > start_date."""
        if hasattr(info, 'data') and 'start_date' in info.data and v <= info.data['start_date']:
            raise ValueError("end_date must be after start_date")
        return v
    
    @field_validator('initial_capital')
    @classmethod
    def validate_initial_capital(cls, v):
        """Validate initial capital."""
        if v <= 0:
            raise ValueError("initial_capital must be positive")
        if v < 1000:
            raise ValueError("initial_capital seems too small (<$1000)")
        return v
    
    @field_validator('commission_rate', 'spread_rate', 'slippage_rate')
    @classmethod
    def validate_cost_rates(cls, v):
        """Validate cost rates are reasonable."""
        if v < 0:
            raise ValueError("Cost rates cannot be negative")
        if v > 0.01:
            raise ValueError("Cost rates seem unusually high (>1%)")
        return v
    
    @field_validator('short_borrow_rate')
    @classmethod
    def validate_short_borrow_rate(cls, v):
        """Validate short borrow rate is reasonable."""
        if v < 0:
            raise ValueError("Short borrow rate cannot be negative")
        if v > 0.1:
            raise ValueError("Short borrow rate seems unusually high (>10%)")
        return v
    
    @field_validator('position_limit')
    @classmethod
    def validate_position_limit(cls, v):
        """Validate position limit."""
        if v <= 0:
            raise ValueError("position_limit must be positive")
        if v > 1:
            raise ValueError("position_limit cannot exceed 100%")
        return v
    
    def get_total_cost_rate(self) -> float:
        """Calculate total transaction cost rate."""
        return self.commission_rate + self.spread_rate + self.slippage_rate
    
    def get_trading_days_per_year(self) -> int:
        """Get trading days per year based on rebalance frequency."""
        frequency_map = {
            'daily': 252,
            'weekly': 52,
            'monthly': 12,
            'quarterly': 4
        }
        return frequency_map.get(self.rebalance_frequency, 252)
    
    def get_summary(self) -> dict:
        """Get backtest configuration summary."""
        base_summary = super().get_summary()
        
        base_summary.update({
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': f"${self.initial_capital:,.0f}",
            'benchmark_symbol': self.benchmark_symbol,
            'total_cost_rate': f"{self.get_total_cost_rate():.3%}",
            'rebalance_frequency': self.rebalance_frequency,
            'position_limit': f"{self.position_limit:.1%}",
            'rebalance_threshold': f"{self.rebalance_threshold:.1%}",
            'enable_short_selling': self.enable_short_selling,
            'risk_limits': {
                'max_drawdown': f"{self.max_drawdown_limit:.1%}" if self.max_drawdown_limit else "None",
                'volatility': f"{self.volatility_limit:.1%}" if self.volatility_limit else "None"
            },
            'parameters_count': len(self.parameters)
        })
        
        return base_summary
