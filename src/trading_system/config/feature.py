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
    # Missing value handling strategy
    handle_missing: str = field(default="interpolate")
    missing_value_threshold: float = field(default=0.1)  # 10% threshold for warnings
    enable_missing_value_monitoring: bool = field(default=True)
    missing_value_report_path: Optional[str] = field(default=None)
    warmup_tolerance_multiplier: float = field(default=1.5)  # Allow 1.5x expected warmup
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

        # Validate missing value handling strategy
        self._validate_missing_value_config()

    def _validate_missing_value_config(self):
        """Validate missing value configuration parameters."""
        valid_strategies = ["interpolate", "forward_fill", "backward_fill", "median_fill", "mean_fill", "drop"]
        if self.handle_missing not in valid_strategies:
            raise ValueError(f"Invalid missing value strategy: {self.handle_missing}. "
                           f"Valid options: {valid_strategies}")

        if not 0 <= self.missing_value_threshold <= 1:
            raise ValueError(f"missing_value_threshold must be between 0 and 1, got {self.missing_value_threshold}")

        if self.warmup_tolerance_multiplier < 1:
            raise ValueError(f"warmup_tolerance_multiplier must be >= 1, got {self.warmup_tolerance_multiplier}")

    def get_missing_value_config(self) -> Dict[str, Any]:
        """
        Get missing value handling configuration.

        Returns:
            Dictionary with missing value handling settings
        """
        return {
            'strategy': self.handle_missing,
            'threshold': self.missing_value_threshold,
            'monitoring_enabled': self.enable_missing_value_monitoring,
            'report_path': self.missing_value_report_path,
            'warmup_tolerance': self.warmup_tolerance_multiplier
        }

    def should_log_missing_value_warning(self, missing_pct: float) -> bool:
        """
        Determine if missing value percentage should trigger a warning.

        Args:
            missing_pct: Missing value percentage (0-1)

        Returns:
            True if warning should be logged
        """
        return missing_pct > self.missing_value_threshold
