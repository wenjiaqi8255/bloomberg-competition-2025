"""
Portfolio Construction Configuration

Pydantic configuration for portfolio construction methods.
Uses discriminated union to separate box_based and quantitative methods.

Available Methods
-----------------

1. Box-Based Portfolio Construction (BoxBasedPortfolioConfig)
   - Systematic diversification across investment style boxes
   - Required: stocks_per_box, box_weights, classifier
   - Use when: Want systematic style diversification
   
2. Quantitative Portfolio Construction (QuantitativePortfolioConfig)
   - Traditional optimization-based construction
   - Required: universe_size, optimizer
   - Use when: Want mean-variance or other optimization

Design Pattern
--------------
Uses Pydantic's discriminated union:
- Automatically selects correct config class based on 'method' field
- Provides type-safe access to method-specific fields
- Validates required fields at load time

Example Usage
-------------
```yaml
# Box-based method
portfolio_construction:
  method: "box_based"
  stocks_per_box: 3
  box_weights:
    method: "equal"
  classifier:
    method: "four_factor"

# Quantitative method
portfolio_construction:
  method: "quantitative"
  universe_size: 100
  optimizer:
    method: "mean_variance"
```

See Also
--------
- BoxBasedPortfolioBuilder: Implementation of box-based method
- QuantitativePortfolioBuilder: Implementation of quantitative method
"""

from typing import Dict, Any, Literal, Optional, Union, Annotated
from pydantic import BaseModel, Field, field_validator, model_validator
from .base import BasePydanticConfig


class BoxWeightsConfig(BaseModel):
    """Box weight configuration for box-based portfolio construction."""
    
    method: Literal["equal", "custom"] = Field(default="equal", description="Weight allocation method")
    dimensions: Dict[str, list] = Field(default_factory=dict, description="Box dimensions")
    custom_weights: Optional[Dict[str, float]] = Field(default=None, description="Custom box weights")
    
    class Config:
        extra = "forbid"
        validate_assignment = True


class ClassifierConfig(BaseModel):
    """Stock classifier configuration."""
    
    method: Literal["four_factor", "sector", "custom"] = Field(description="Classification method")
    cache_enabled: bool = Field(default=True, description="Enable classification caching")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Method-specific parameters")
    
    class Config:
        extra = "forbid"
        validate_assignment = True


class BoxSelectorConfig(BaseModel):
    """Box selector configuration."""
    
    type: Literal["signal_based", "random", "custom"] = Field(description="Box selection type")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Selection parameters")
    
    class Config:
        extra = "forbid"
        validate_assignment = True


class OptimizerConfig(BaseModel):
    """Optimizer configuration for quantitative portfolio construction."""
    
    method: Literal["mean_variance", "equal_weight", "top_n"] = Field(description="Optimization method")
    risk_aversion: float = Field(default=2.0, ge=0.1, le=10.0, description="Risk aversion parameter (mean_variance only)")
    top_n: int = Field(default=10, ge=1, le=100, description="Number of top stocks (top_n only)")
    max_weight: float = Field(default=0.15, ge=0.01, le=1.0, description="Maximum position weight")
    min_weight: float = Field(default=0.02, ge=0.001, le=0.5, description="Minimum position weight")
    
    # Support legacy field names from YAML
    max_position_weight: Optional[float] = Field(default=None, ge=0.01, le=1.0, description="Maximum position weight (legacy)")
    min_position_weight: Optional[float] = Field(default=None, ge=0.001, le=0.5, description="Minimum position weight (legacy)")
    
    class Config:
        extra = "allow"  # Allow extra fields for backward compatibility
        validate_assignment = True
    
    @property
    def effective_max_weight(self):
        """Get effective max weight (legacy or standard field)."""
        return self.max_position_weight if self.max_position_weight is not None else self.max_weight
    
    @property
    def effective_min_weight(self):
        """Get effective min weight (legacy or standard field)."""
        return self.min_position_weight if self.min_position_weight is not None else self.min_weight


class CovarianceConfig(BaseModel):
    """Covariance estimation configuration for quantitative portfolio construction."""
    
    lookback_days: int = Field(default=252, ge=30, le=1000, description="Lookback period for covariance estimation")
    method: Literal["ledoit_wolf", "sample", "empirical"] = Field(default="ledoit_wolf", description="Covariance estimation method")
    shrinkage: float = Field(default=0.1, ge=0.0, le=1.0, description="Shrinkage parameter for Ledoit-Wolf")
    
    class Config:
        extra = "forbid"
        validate_assignment = True


class BoxBasedPortfolioConfig(BasePydanticConfig):
    """
    Box-based portfolio construction configuration.
    
    Systematic diversification across investment style boxes.
    """
    
    method: Literal["box_based"] = Field(default="box_based", description="Portfolio construction method")
    stocks_per_box: int = Field(ge=1, description="Number of stocks per box")
    min_stocks_per_box: int = Field(ge=1, description="Minimum stocks per box")
    allocation_method: Literal["equal", "signal_proportional"] = Field(
        default="equal", 
        description="Allocation method within boxes"
    )
    
    # Box configuration
    box_weights: BoxWeightsConfig = Field(description="Box weight configuration")
    classifier: ClassifierConfig = Field(description="Stock classifier configuration")
    box_selector: Optional[BoxSelectorConfig] = Field(default=None, description="Box selector configuration")
    
    # Additional parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    
    class Config:
        extra = "forbid"
        validate_assignment = True
    
    @field_validator('min_stocks_per_box')
    @classmethod
    def validate_min_stocks_per_box(cls, v, info):
        """Validate min_stocks_per_box <= stocks_per_box."""
        if hasattr(info, 'data') and 'stocks_per_box' in info.data and v > info.data['stocks_per_box']:
            raise ValueError("min_stocks_per_box must be <= stocks_per_box")
        return v
    
    @model_validator(mode='after')
    def validate_box_based_requirements(self):
        """Validate box_based method requirements."""
        # Box-based method requires box_weights
        if not self.box_weights:
            raise ValueError("box_based method requires box_weights configuration")
        
        # Validate box_weights has dimensions
        if not self.box_weights.dimensions:
            raise ValueError("box_based method requires box_weights.dimensions")
        
        return self
    
    def get_summary(self) -> dict:
        """Get portfolio construction summary."""
        base_summary = super().get_summary()
        
        base_summary.update({
            'method': self.method,
            'stocks_per_box': self.stocks_per_box,
            'min_stocks_per_box': self.min_stocks_per_box,
            'allocation_method': self.allocation_method,
            'box_weights_method': self.box_weights.method if self.box_weights else None,
            'classifier_method': self.classifier.method if self.classifier else None,
            'parameters_count': len(self.parameters)
        })
        
        return base_summary


class QuantitativePortfolioConfig(BasePydanticConfig):
    """
    Quantitative portfolio construction configuration.
    
    Traditional optimization-based construction.
    """
    
    method: Literal["quantitative"] = Field(default="quantitative", description="Portfolio construction method")
    universe_size: int = Field(ge=10, le=1000, description="Universe size for portfolio construction")
    optimizer: OptimizerConfig = Field(description="Optimizer configuration")
    covariance: CovarianceConfig = Field(default_factory=CovarianceConfig, description="Covariance estimation configuration")
    enable_short_selling: bool = Field(default=False, description="Enable short selling")
    
    # Additional parameters
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Additional parameters")
    
    class Config:
        extra = "forbid"
        validate_assignment = True
    
    def get_summary(self) -> dict:
        """Get portfolio construction summary."""
        base_summary = super().get_summary()
        
        base_summary.update({
            'method': self.method,
            'universe_size': self.universe_size,
            'optimizer_method': self.optimizer.method if self.optimizer else None,
            'covariance_method': self.covariance.method if self.covariance else None,
            'enable_short_selling': self.enable_short_selling,
            'parameters_count': len(self.parameters)
        })
        
        return base_summary


# Union type for Portfolio Construction - using discriminated union
PortfolioConstructionConfig = Union[BoxBasedPortfolioConfig, QuantitativePortfolioConfig]
