"""
Strategy Configuration

Pydantic configuration for trading strategies.
Directly maps YAML strategy section with automatic validation.
"""

from typing import Dict, Any, Literal, Optional, List, Union
from pydantic import BaseModel, Field, field_validator, model_validator
from .base import BasePydanticConfig
from .portfolio import (
    PortfolioConstructionConfig,
    BoxBasedPortfolioConfig,
    QuantitativePortfolioConfig,
    AllocationConfig
)


class StrategyConfig(BasePydanticConfig):
    """
    Strategy configuration.
    
    Directly maps YAML strategy section.
    """
    
    # Strategy identification
    type: Literal["ml", "fama_french_5", "fama_macbeth", "dual_momentum", "core_satellite"] = Field(
        description="Strategy type"
    )
    
    @property
    def strategy_type(self):
        """Compatibility property for strategy_type."""
        return type('StrategyType', (), {'value': self.type})()
    
    # Model configuration
    model_id: Optional[str] = Field(default=None, description="Model identifier")
    lookback_days: int = Field(ge=1, le=1000, default=252, description="Lookback period in days")
    risk_free_rate: float = Field(ge=0, le=0.1, default=0.02, description="Risk-free rate")
    
    # Portfolio construction (can be at top level or in parameters)
    portfolio_construction: Optional[Any] = Field(
        default=None, 
        description="Portfolio construction configuration"
    )
    
    # Constraints and parameters
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Strategy constraints")
    
    class Config:
        extra = "allow"  # Allow extra fields for backward compatibility
        validate_assignment = True
    
    @field_validator('lookback_days')
    @classmethod
    def validate_lookback_days(cls, v):
        """Validate lookback period."""
        if v <= 0:
            raise ValueError("lookback_days must be positive")
        if v > 1000:
            raise ValueError("lookback_days should not exceed 1000 days")
        return v
    
    @field_validator('risk_free_rate')
    @classmethod
    def validate_risk_free_rate(cls, v):
        """Validate risk-free rate."""
        if v < 0:
            raise ValueError("risk_free_rate cannot be negative")
        if v > 0.1:
            raise ValueError("risk_free_rate seems unusually high (>10%)")
        return v
    
    @model_validator(mode='before')
    @classmethod
    def extract_portfolio_construction(cls, values):
        """Extract portfolio_construction from parameters if present."""
        if isinstance(values, dict):
            # Check if portfolio_construction is in parameters
            if 'parameters' in values and isinstance(values['parameters'], dict):
                parameters = values['parameters']
                if 'portfolio_construction' in parameters:
                    # Move portfolio_construction to top level
                    values['portfolio_construction'] = parameters.pop('portfolio_construction')
        return values
    
    @model_validator(mode='after')
    def resolve_portfolio_construction_union(self):
        """Resolve portfolio construction union type based on method."""
        if self.portfolio_construction is None:
            return self
        
        if isinstance(self.portfolio_construction, dict):
            pc_data = self.portfolio_construction
            if 'method' in pc_data:
                method = pc_data['method']
                
                if method == 'box_based':
                    # Convert allocation_config dict to AllocationConfig if present
                    if 'allocation_config' in pc_data and isinstance(pc_data['allocation_config'], dict):
                        pc_data['allocation_config'] = AllocationConfig(**pc_data['allocation_config'])
                    # Create BoxBasedPortfolioConfig
                    self.portfolio_construction = BoxBasedPortfolioConfig(**pc_data)
                
                elif method == 'quantitative':
                    # For quantitative, we need to merge with parameters
                    parameters = self.parameters or {}
                    if 'universe_size' in parameters:
                        pc_data['universe_size'] = parameters['universe_size']
                    if 'optimizer' in parameters:
                        pc_data['optimizer'] = parameters['optimizer']
                    if 'covariance' in parameters:
                        pc_data['covariance'] = parameters['covariance']
                    if 'enable_short_selling' in parameters:
                        pc_data['enable_short_selling'] = parameters['enable_short_selling']
                    
                    # Remove classifier if present (not needed for quantitative)
                    if 'classifier' in pc_data:
                        del pc_data['classifier']
                    
                    # Create QuantitativePortfolioConfig
                    self.portfolio_construction = QuantitativePortfolioConfig(**pc_data)
                
                else:
                    raise ValueError(f"Unknown portfolio construction method: {method}")
        
        return self
    
    @model_validator(mode='after')
    def validate_strategy_requirements(self):
        """Validate strategy-specific requirements."""
        # FF5 strategy validation
        if self.type == 'fama_french_5':
            if self.portfolio_construction:
                # Use isinstance for type-safe validation
                if isinstance(self.portfolio_construction, BoxBasedPortfolioConfig):
                    # Box-based validation is already done in BoxBasedPortfolioConfig
                    pass
                elif isinstance(self.portfolio_construction, QuantitativePortfolioConfig):
                    # Quantitative validation is already done in QuantitativePortfolioConfig
                    pass
        
        return self
    
    def get_summary(self) -> dict:
        """Get strategy configuration summary."""
        base_summary = super().get_summary()
        
        # Get portfolio method info safely
        portfolio_method = None
        if self.portfolio_construction:
            if isinstance(self.portfolio_construction, BoxBasedPortfolioConfig):
                portfolio_method = "box_based"
            elif isinstance(self.portfolio_construction, QuantitativePortfolioConfig):
                portfolio_method = "quantitative"
            else:
                portfolio_method = getattr(self.portfolio_construction, 'method', 'unknown')
        
        base_summary.update({
            'type': self.type,
            'model_id': self.model_id,
            'lookback_days': self.lookback_days,
            'risk_free_rate': f"{self.risk_free_rate:.2%}",
            'has_portfolio_construction': self.portfolio_construction is not None,
            'portfolio_method': portfolio_method,
            'constraints_count': len(self.constraints) if self.constraints else 0,
            'parameters_count': len(self.parameters)
        })
        
        return base_summary
