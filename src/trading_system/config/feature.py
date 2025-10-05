from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
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

    # Steps-based configuration (for pipeline-style feature engineering)
    steps: List[Dict[str, Any]] = None

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

    # Method selection parameters (for enhanced feature engineering)
    return_methods: List[str] = field(default_factory=lambda: ["simple", "log"])
    momentum_methods: List[str] = field(default_factory=lambda: ["simple", "exponential"])
    trend_methods: List[str] = field(default_factory=lambda: ["sma", "ema", "dema"])
    volatility_methods: List[str] = field(default_factory=lambda: ["std", "parkinson", "garman_klass"])
    volume_periods: List[int] = field(default_factory=lambda: [5, 10, 20])
    volume_ratios: bool = field(default=True)
    volume_indicators: List[str] = field(default_factory=lambda: ["obv", "vwap", "ad_line"])

    # Additional parameters from templates
    return_periods: List[int] = field(default_factory=lambda: [1, 5, 10, 20])
    trend_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    feature_importance_threshold: float = field(default=0.01)
    handle_missing: str = field(default="interpolate")
    technical_indicators: List[str] = field(default_factory=lambda: ["rsi", "macd", "bollinger_bands", "stochastic", "williams_r"])
    technical_patterns: List[str] = field(default_factory=lambda: ["rsi", "macd", "bollinger_position", "stochastic"])

    # Factor model parameters (for FF5)
    factors: List[str] = field(default_factory=lambda: ["MKT", "SMB", "HML", "RMW", "CMA"])
    factor_timing: Dict[str, Any] = field(default_factory=dict)
    risk_metrics: Dict[str, Any] = field(default_factory=dict)

    # Sequence features (for LSTM)
    sequence_features: Dict[str, Any] = field(default_factory=dict)

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
