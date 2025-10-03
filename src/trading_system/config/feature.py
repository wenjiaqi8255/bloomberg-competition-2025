from dataclasses import dataclass
from typing import List
from enum import Enum


class FeatureType(Enum):
    """Feature type enumeration - simplified."""
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    TECHNICAL = "technical"
    LIQUIDITY = "liquidity"
    MEAN_REVERSION = "mean_reversion"
    TREND = "trend"
    
@dataclass
class FeatureConfig:
    """Simplified configuration for feature engineering."""

    # Time periods
    momentum_periods: List[int] = None
    volatility_windows: List[int] = None
    lookback_periods: List[int] = None

    # Feature selection
    enabled_features: List[FeatureType] = None
    include_technical: bool = True
    include_theoretical: bool = False

    # Validation parameters
    min_ic_threshold: float = 0.03
    min_significance: float = 0.05
    feature_lag: int = 1

    # Normalization
    normalize_features: bool = True
    normalization_method: str = "robust"

    # Feature selection
    max_features: int = 50

    def __post_init__(self):
        """Initialize default values."""
        if self.momentum_periods is None:
            self.momentum_periods = [21, 63, 126, 252]  # 1, 3, 6, 12 months

        if self.volatility_windows is None:
            self.volatility_windows = [20, 60]

        if self.lookback_periods is None:
            self.lookback_periods = [20, 50, 200]

        if self.enabled_features is None:
            self.enabled_features = [
                FeatureType.MOMENTUM,
                FeatureType.VOLATILITY,
                FeatureType.TECHNICAL,
                FeatureType.VOLUME
            ]
