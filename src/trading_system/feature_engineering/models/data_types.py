"""
Core Data Types for Feature Engineering.

This module defines all data types and interfaces for the simplified
feature engineering system.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from src.trading_system.feature_engineering.base.feature import FeatureConfig


# ============================================================================
# Core Data Types
# ============================================================================
@dataclass
class FeatureMetrics:
    """Feature performance metrics - simplified."""

    feature_name: str
    information_coefficient: float
    ic_p_value: float
    positive_ic_ratio: float
    feature_stability: float
    economic_significance: float

    # Statistical properties
    mean_value: float
    std_value: float
    skewness: float
    kurtosis: float

    # Validation results
    is_significant: bool
    is_economically_meaningful: bool
    recommendation: str  # "ACCEPT", "MARGINAL", "REJECT"


@dataclass
class FeatureResult:
    """Unified result object for feature engineering."""

    # Core data
    features: pd.DataFrame  # Combined features for all symbols
    metrics: Dict[str, FeatureMetrics]  # Feature validation metrics

    # Metadata
    config: FeatureConfig
    symbols: List[str]
    feature_names: List[str]
    accepted_features: List[str]

    # Performance summary
    total_features: int
    accepted_count: int
    acceptance_rate: float

    @property
    def summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        return {
            'total_features': self.total_features,
            'accepted_count': self.accepted_count,
            'acceptance_rate': self.acceptance_rate,
            'symbols_count': len(self.symbols),
            'average_ic': np.mean([m.information_coefficient for m in self.metrics.values()]),
            'significant_features': sum(1 for m in self.metrics.values() if m.is_significant),
            'economically_meaningful': sum(1 for m in self.metrics.values() if m.is_economically_meaningful)
        }


# ============================================================================
# Input/Output Data Types
# ============================================================================

# Input data types
PriceData = Dict[str, pd.DataFrame]  # Symbol -> OHLCV DataFrame
ForwardReturns = Dict[str, pd.Series]  # Symbol -> forward returns
FeatureData = pd.DataFrame  # Combined feature DataFrame
ValidationData = Dict[str, FeatureMetrics]  # Feature validation metrics

# ============================================================================
# Utility Functions
# ============================================================================

def create_default_config() -> FeatureConfig:
    """Create default feature configuration."""
    return FeatureConfig()


def validate_price_data(price_data: PriceData) -> bool:
    """Validate price data format and structure."""
    if not isinstance(price_data, dict):
        return False

    for symbol, df in price_data.items():
        if not isinstance(df, pd.DataFrame):
            return False
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in df.columns for col in required_columns):
            return False
        if len(df) < 60:  # Minimum data requirement
            return False

    return True


def align_data(price_data: PriceData,
               forward_returns: Optional[ForwardReturns] = None) -> tuple:
    """Align price data and forward returns by date."""
    # Find common date range across all symbols
    common_dates = None
    for symbol, df in price_data.items():
        if common_dates is None:
            common_dates = df.index
        else:
            common_dates = common_dates.intersection(df.index)

    # Filter price data to common dates
    aligned_price_data = {}
    for symbol, df in price_data.items():
        aligned_price_data[symbol] = df.loc[common_dates]

    # Align forward returns if provided
    aligned_forward_returns = None
    if forward_returns:
        aligned_forward_returns = {}
        for symbol, series in forward_returns.items():
            if symbol in aligned_price_data:
                aligned_forward_returns[symbol] = series.loc[common_dates]

    return aligned_price_data, aligned_forward_returns


# ============================================================================
# Type Aliases for Common Usage
# ============================================================================

# Commonly used types
Config = FeatureConfig
Metrics = FeatureMetrics
Result = FeatureResult