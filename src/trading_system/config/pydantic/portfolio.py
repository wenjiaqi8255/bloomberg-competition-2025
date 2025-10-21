"""
Portfolio Construction Configuration

Pydantic configuration for portfolio construction methods.
Directly maps YAML structure with automatic validation.
"""

from typing import Dict, Any, Literal, Optional
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


class PortfolioConstructionConfig(BasePydanticConfig):
    """
    Portfolio construction configuration.
    
    Directly maps YAML portfolio_construction section.
    """
    
    # Core method
    method: Literal["quantitative", "box_based"] = Field(description="Portfolio construction method")
    
    # Box-based parameters
    stocks_per_box: int = Field(ge=1, description="Number of stocks per box")
    min_stocks_per_box: int = Field(ge=1, description="Minimum stocks per box")
    allocation_method: Literal["equal", "signal_proportional"] = Field(
        default="equal", 
        description="Allocation method within boxes"
    )
    
    # Box configuration
    box_weights: BoxWeightsConfig = Field(default_factory=BoxWeightsConfig, description="Box weight configuration")
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
        if self.method == 'box_based':
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
