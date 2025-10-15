"""
Data Processing Configuration

Centralized configuration for data processing, feature engineering,
and caching following SOLID principles and financial industry best practices.

Design Principles:
- Single Responsibility: Only handles data processing configuration
- Open/Closed: Easy to extend with new configuration options
- Dependency Inversion: Depends on abstractions, not implementations
- DRY: Centralizes all configuration logic
- KISS: Simple, clear configuration structure
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import yaml

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for feature caching."""
    enabled: bool = True
    provider_type: str = "memory"  # memory, redis, database
    max_size: int = 1000
    ttl_seconds: int = 3600  # 1 hour
    enable_compression: bool = False
    enable_encryption: bool = False
    cleanup_threshold: float = 0.8  # Clean up when 80% full

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CacheConfig':
        """Create CacheConfig from dictionary."""
        return cls(**config_dict)


@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering pipeline."""

    # Cross-sectional features
    cross_sectional_enabled: bool = True
    cross_sectional_features: List[str] = field(default_factory=lambda: [
        'market_cap', 'book_to_market', 'size', 'value', 'momentum', 'volatility'
    ])

    # Technical indicators
    technical_enabled: bool = False
    technical_features: List[str] = field(default_factory=lambda: [
        'sma', 'ema', 'rsi', 'macd', 'bollinger_bands'
    ])

    # Feature calculation settings
    lookback_periods: Dict[str, int] = field(default_factory=lambda: {
        'momentum': 252,
        'volatility': 60,
        'ma_long': 200,
        'ma_short': 50
    })

    winsorize_percentile: float = 0.01
    normalize_features: bool = True
    normalization_method: str = "minmax"  # minmax, zscore, robust

    # Feature validation
    min_ic_threshold: float = 0.02
    min_significance: float = 0.1
    feature_lag: int = 1

    # Performance settings
    enable_vectorization: bool = True
    batch_size: int = 100
    parallel_processing: bool = True
    n_jobs: int = -1  # Use all cores

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FeatureEngineeringConfig':
        """Create FeatureEngineeringConfig from dictionary."""
        return cls(**config_dict)


@dataclass
class DataQualityConfig:
    """Configuration for data quality and validation."""

    # Missing data handling
    max_missing_ratio: float = 0.1  # Maximum 10% missing data
    missing_data_strategy: str = "forward_fill"  # forward_fill, backward_fill, interpolate, drop

    # Outlier detection
    outlier_detection: bool = True
    outlier_method: str = "iqr"  # iqr, zscore, isolation_forest
    outlier_threshold: float = 3.0

    # Data validation
    min_data_points: int = 10
    max_price_change_ratio: float = 0.5  # Max 50% daily change

    # Date handling
    date_tolerance_days: int = 7
    enable_weekend_fill: bool = True

    # Quality checks
    enable_duplicate_check: bool = True
    enable_consistency_check: bool = True
    enable_business_day_check: bool = True

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataQualityConfig':
        """Create DataQualityConfig from dictionary."""
        return cls(**config_dict)


@dataclass
class PerformanceConfig:
    """Configuration for performance monitoring and optimization."""

    # Monitoring
    enable_monitoring: bool = True
    monitor_interval_seconds: int = 60
    max_history_size: int = 1000

    # Alert thresholds
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'memory_usage_percent': 80.0,
        'feature_calculation_time_sec': 5.0,
        'cache_hit_rate_min': 0.3,
        'error_rate_max': 0.05
    })

    # Performance optimization
    enable_lazy_loading: bool = True
    enable_progressive_loading: bool = False
    memory_limit_mb: int = 2048  # 2GB

    # Logging
    enable_performance_logging: bool = True
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PerformanceConfig':
        """Create PerformanceConfig from dictionary."""
        return cls(**config_dict)


@dataclass
class DataProcessingConfig:
    """Main configuration class for data processing."""

    cache: CacheConfig = field(default_factory=CacheConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    data_quality: DataQualityConfig = field(default_factory=DataQualityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)

    # General settings
    environment: str = "development"  # development, testing, production
    debug_mode: bool = False
    log_to_file: bool = True
    log_file_path: Optional[str] = None

    # Data sources
    primary_data_source: str = "yfinance"
    backup_data_source: str = "alpha_vantage"

    # Risk management
    enable_short_selling: bool = False
    max_position_weight: float = 0.1
    max_portfolio_risk: float = 0.15

    # Experiment tracking
    enable_wandb_logging: bool = False
    wandb_project: Optional[str] = None
    experiment_tags: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'DataProcessingConfig':
        """
        Load configuration from YAML file.

        Args:
            yaml_path: Path to YAML configuration file

        Returns:
            DataProcessingConfig instance
        """
        try:
            with open(yaml_path, 'r') as file:
                config_dict = yaml.safe_load(file)

            # Parse nested configurations
            config = cls()

            if 'cache' in config_dict:
                config.cache = CacheConfig.from_dict(config_dict['cache'])

            if 'feature_engineering' in config_dict:
                config.feature_engineering = FeatureEngineeringConfig.from_dict(config_dict['feature_engineering'])

            if 'data_quality' in config_dict:
                config.data_quality = DataQualityConfig.from_dict(config_dict['data_quality'])

            if 'performance' in config_dict:
                config.performance = PerformanceConfig.from_dict(config_dict['performance'])

            # Update general settings
            for key, value in config_dict.items():
                if hasattr(config, key) and key not in ['cache', 'feature_engineering', 'data_quality', 'performance']:
                    setattr(config, key, value)

            logger.info(f"Configuration loaded from {yaml_path}")
            return config

        except Exception as e:
            logger.error(f"Error loading configuration from {yaml_path}: {e}")
            raise

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'DataProcessingConfig':
        """
        Create DataProcessingConfig from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            DataProcessingConfig instance
        """
        config = cls()

        # Parse nested configurations
        if 'cache' in config_dict:
            config.cache = CacheConfig.from_dict(config_dict['cache'])

        if 'feature_engineering' in config_dict:
            config.feature_engineering = FeatureEngineeringConfig.from_dict(config_dict['feature_engineering'])

        if 'data_quality' in config_dict:
            config.data_quality = DataQualityConfig.from_dict(config_dict['data_quality'])

        if 'performance' in config_dict:
            config.performance = PerformanceConfig.from_dict(config_dict['performance'])

        # Update general settings
        for key, value in config_dict.items():
            if hasattr(config, key) and key not in ['cache', 'feature_engineering', 'data_quality', 'performance']:
                setattr(config, key, value)

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'cache': self.cache.__dict__,
            'feature_engineering': self.feature_engineering.__dict__,
            'data_quality': self.data_quality.__dict__,
            'performance': self.performance.__dict__,
            'environment': self.environment,
            'debug_mode': self.debug_mode,
            'log_to_file': self.log_to_file,
            'log_file_path': self.log_file_path,
            'primary_data_source': self.primary_data_source,
            'backup_data_source': self.backup_data_source,
            'enable_short_selling': self.enable_short_selling,
            'max_position_weight': self.max_position_weight,
            'max_portfolio_risk': self.max_portfolio_risk,
            'enable_wandb_logging': self.enable_wandb_logging,
            'wandb_project': self.wandb_project,
            'experiment_tags': self.experiment_tags
        }

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        try:
            config_dict = self.to_dict()

            with open(yaml_path, 'w') as file:
                yaml.dump(config_dict, file, default_flow_style=False, indent=2)

            logger.info(f"Configuration saved to {yaml_path}")

        except Exception as e:
            logger.error(f"Error saving configuration to {yaml_path}: {e}")
            raise

    def validate(self) -> List[str]:
        """
        Validate configuration and return list of validation errors.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Validate cache configuration
        if self.cache.enabled and self.cache.max_size <= 0:
            errors.append("Cache max_size must be positive when cache is enabled")

        if self.cache.ttl_seconds <= 0:
            errors.append("Cache TTL must be positive")

        # Validate feature engineering
        if not self.feature_engineering.cross_sectional_features and not self.feature_engineering.technical_features:
            errors.append("At least one feature type must be enabled")

        if self.feature_engineering.winsorize_percentile < 0 or self.feature_engineering.winsorize_percentile > 0.5:
            errors.append("Winsorize percentile must be between 0 and 0.5")

        # Validate data quality
        if self.data_quality.max_missing_ratio < 0 or self.data_quality.max_missing_ratio > 1:
            errors.append("Max missing ratio must be between 0 and 1")

        if self.data_quality.min_data_points < 1:
            errors.append("Min data points must be at least 1")

        # Validate performance
        if self.performance.memory_limit_mb < 100:
            errors.append("Memory limit should be at least 100MB")

        for threshold_name, threshold_value in self.performance.alert_thresholds.items():
            if threshold_value < 0 or threshold_value > 1:
                if 'percent' in threshold_name or 'rate' in threshold_name:
                    errors.append(f"Alert threshold {threshold_name} must be between 0 and 1")

        return errors

    def get_environment_specific_config(self) -> 'DataProcessingConfig':
        """
        Get environment-specific configuration overrides.

        Returns:
            Configuration with environment-specific adjustments
        """
        config = DataProcessingConfig.from_dict(self.to_dict())

        if self.environment == "production":
            # Production settings
            config.debug_mode = False
            config.feature_engineering.enable_vectorization = True
            config.feature_engineering.parallel_processing = True
            config.performance.enable_monitoring = True
            config.performance.log_level = "WARNING"

        elif self.environment == "testing":
            # Testing settings
            config.debug_mode = True
            config.feature_engineering.batch_size = 10
            config.cache.max_size = 100
            config.performance.enable_monitoring = False
            config.performance.memory_limit_mb = 512

        else:  # development
            # Development settings (default)
            config.debug_mode = True
            config.performance.enable_monitoring = True
            config.performance.log_level = "DEBUG"

        return config

    def __str__(self):
        return f"DataProcessingConfig(env={self.environment}, cache={self.cache.enabled})"

    def __repr__(self):
        return self.__str__()


# Configuration factory functions
def create_development_config() -> DataProcessingConfig:
    """Create development configuration."""
    return DataProcessingConfig(
        environment="development",
        debug_mode=True,
        cache=CacheConfig(max_size=100),
        performance=PerformanceConfig(
            enable_monitoring=True,
            log_level="DEBUG"
        )
    )


def create_production_config() -> DataProcessingConfig:
    """Create production configuration."""
    return DataProcessingConfig(
        environment="production",
        debug_mode=False,
        cache=CacheConfig(
            max_size=5000,
            ttl_seconds=7200  # 2 hours
        ),
        performance=PerformanceConfig(
            enable_monitoring=True,
            log_level="WARNING",
            memory_limit_mb=4096  # 4GB
        ),
        feature_engineering=FeatureEngineeringConfig(
            enable_vectorization=True,
            parallel_processing=True,
            n_jobs=-1
        )
    )


def create_testing_config() -> DataProcessingConfig:
    """Create testing configuration."""
    return DataProcessingConfig(
        environment="testing",
        debug_mode=True,
        cache=CacheConfig(max_size=50),
        performance=PerformanceConfig(
            enable_monitoring=False,
            log_level="ERROR",
            memory_limit_mb=256
        ),
        feature_engineering=FeatureEngineeringConfig(
            batch_size=5,
            enable_vectorization=False,
            parallel_processing=False
        )
    )


# Configuration validator
def validate_config(config: DataProcessingConfig) -> bool:
    """
    Validate configuration and return True if valid.

    Args:
        config: Configuration to validate

    Returns:
        True if configuration is valid
    """
    errors = config.validate()

    if errors:
        logger.error("Configuration validation failed:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    else:
        logger.info("Configuration validation passed")
        return True