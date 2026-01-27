# Feature Engineering Configuration Guide

## Overview

The feature engineering system in this trading platform provides comprehensive tools for creating, validating, and managing features for quantitative trading models. This guide covers all configuration parameters, usage examples, and best practices.

## Table of Contents

1. [Basic Feature Control](#basic-feature-control)
2. [Time Period Parameters](#time-period-parameters)
3. [Method Selection](#method-selection)
4. [Technical Indicators](#technical-indicators)
5. [Feature Selection and Validation](#feature-selection-and-validation)
6. [Missing Value Handling](#missing-value-handling)
7. [Cross-Sectional Features](#cross-sectional-features)
8. [Box Features](#box-features)
9. [Data Format Configuration](#data-format-configuration)
10. [Factor Model Parameters](#factor-model-parameters)
11. [Configuration Examples](#configuration-examples)
12. [Best Practices](#best-practices)
13. [Troubleshooting](#troubleshooting)

## Basic Feature Control

### `enabled_features`
**Type**: Array of strings  
**Default**: `["momentum", "volatility", "technical", "volume"]`  
**Description**: Controls which feature types are computed.

**Available Options**:
- `momentum` - Price momentum indicators
- `volatility` - Volatility measures
- `technical` - Technical analysis indicators
- `volume` - Volume-based indicators
- `trend` - Trend following indicators
- `fama_french_factors` - Fama-French factor features

**Example**:
```yaml
enabled_features: ['momentum', 'volatility', 'technical']
```

### `include_technical`
**Type**: Boolean  
**Default**: `false`  
**Description**: Whether to include technical analysis indicators.

### `include_cross_sectional`
**Type**: Boolean  
**Default**: `true`  
**Description**: Whether to include cross-sectional features for Fama-MacBeth models.

### `include_theoretical`
**Type**: Boolean  
**Default**: `false`  
**Description**: Whether to include theoretical/academic features.

## Time Period Parameters

### `momentum_periods`
**Type**: Array of integers  
**Default**: `[21, 63, 126, 252]`  
**Description**: Lookback periods for momentum indicators (trading days).

**Common Values**:
- `21` - 1 month
- `63` - 3 months
- `126` - 6 months
- `252` - 12 months

### `volatility_windows`
**Type**: Array of integers  
**Default**: `[20, 60]`  
**Description**: Rolling windows for volatility calculations.

### `lookback_periods`
**Type**: Array of integers  
**Default**: `[20, 50, 200]`  
**Description**: General lookback periods for feature engineering.

### `return_periods`
**Type**: Array of integers  
**Default**: `[1, 5, 10, 20]`  
**Description**: Periods for return calculations.

### `trend_periods`
**Type**: Array of integers  
**Default**: `[10, 20, 50]`  
**Description**: Periods for trend indicators.

### `volume_periods`
**Type**: Array of integers  
**Default**: `[5, 10, 20]`  
**Description**: Periods for volume indicators.

## Method Selection

### `return_methods`
**Type**: Array of strings  
**Default**: `["simple", "log"]`  
**Description**: Methods for return calculations.

**Available Options**:
- `simple` - Simple returns
- `log` - Logarithmic returns

### `momentum_methods`
**Type**: Array of strings  
**Default**: `["simple", "exponential"]`  
**Description**: Methods for momentum calculations.

**Available Options**:
- `simple` - Simple momentum
- `exponential` - Exponentially weighted momentum

### `trend_methods`
**Type**: Array of strings  
**Default**: `["sma", "ema", "dema"]`  
**Description**: Methods for trend calculations.

**Available Options**:
- `sma` - Simple Moving Average
- `ema` - Exponential Moving Average
- `dema` - Double Exponential Moving Average

### `volatility_methods`
**Type**: Array of strings  
**Default**: `["std", "parkinson", "garman_klass"]`  
**Description**: Methods for volatility calculations.

**Available Options**:
- `std` - Standard deviation
- `parkinson` - Parkinson volatility estimator
- `garman_klass` - Garman-Klass volatility estimator
- `rogers_satchell` - Rogers-Satchell volatility estimator

### `volume_ratios`
**Type**: Boolean  
**Default**: `true`  
**Description**: Whether to calculate volume ratios.

### `volume_indicators`
**Type**: Array of strings  
**Default**: `["obv", "vwap", "ad_line"]`  
**Description**: Volume-based indicators to calculate.

**Available Options**:
- `obv` - On-Balance Volume
- `vwap` - Volume Weighted Average Price
- `ad_line` - Accumulation/Distribution Line

## Technical Indicators

### `technical_indicators`
**Type**: Array of strings  
**Default**: `["rsi", "macd", "bollinger_bands", "stochastic", "williams_r"]`  
**Description**: Technical indicators to calculate.

**Available Options**:
- `rsi` - Relative Strength Index
- `macd` - Moving Average Convergence Divergence
- `bollinger_bands` - Bollinger Bands
- `stochastic` - Stochastic Oscillator
- `williams_r` - Williams %R
- `adx` - Average Directional Index
- `cci` - Commodity Channel Index
- `mfi` - Money Flow Index

### `technical_patterns`
**Type**: Array of strings  
**Default**: `["rsi", "macd", "bollinger_position", "stochastic"]`  
**Description**: Technical patterns to identify.

## Feature Selection and Validation

### `max_features`
**Type**: Integer  
**Default**: `50`  
**Description**: Maximum number of features to select.

### `feature_importance_threshold`
**Type**: Number (0.0-1.0)  
**Default**: `0.01`  
**Description**: Minimum feature importance threshold for selection.

### `min_ic_threshold`
**Type**: Number (0.0-1.0)  
**Default**: `0.03`  
**Description**: Minimum Information Coefficient threshold for feature selection.

### `min_significance`
**Type**: Number (0.0-1.0)  
**Default**: `0.05`  
**Description**: Minimum significance level for feature validation.

### `feature_lag`
**Type**: Integer (0-10)  
**Default**: `1`  
**Description**: Number of periods to lag features to avoid look-ahead bias.

## Missing Value Handling

### `handle_missing`
**Type**: String  
**Default**: `"interpolate"`  
**Description**: Strategy for handling missing values.

**Available Options**:
- `forward_fill` - Forward fill missing values
- `backward_fill` - Backward fill missing values
- `drop` - Drop rows with missing values
- `interpolate` - Interpolate missing values
- `median_fill` - Fill with median values
- `mean_fill` - Fill with mean values

### `missing_value_threshold`
**Type**: Number (0.0-1.0)  
**Default**: `0.1`  
**Description**: Threshold for missing value warnings (10% default).

### `enable_missing_value_monitoring`
**Type**: Boolean  
**Default**: `true`  
**Description**: Whether to enable missing value monitoring.

### `missing_value_report_path`
**Type**: String or null  
**Default**: `null`  
**Description**: Path to save missing value reports.

### `warmup_tolerance_multiplier`
**Type**: Number (≥1.0)  
**Default**: `1.5`  
**Description**: Multiplier for warmup period tolerance.

## Cross-Sectional Features

### `cross_sectional_features`
**Type**: Array of strings  
**Default**: `["market_cap", "book_to_market", "size", "value", "momentum", "volatility"]`  
**Description**: Cross-sectional features to compute.

**Available Options**:
- `market_cap` - Market capitalization
- `book_to_market` - Book-to-market ratio
- `size` - Size factor
- `value` - Value factor
- `momentum` - Momentum factor
- `volatility` - Volatility factor
- `country_risk_premium` - Country risk premium
- `equity_risk_premium` - Equity risk premium
- `default_spread` - Default spread
- `corporate_tax_rate` - Corporate tax rate

### `cross_sectional_lookback`
**Type**: Object  
**Default**: `{"momentum": 252, "volatility": 60, "ma_long": 200, "ma_short": 50}`  
**Description**: Lookback periods for cross-sectional features.

**Properties**:
- `momentum` - Lookback period for momentum (trading days)
- `volatility` - Lookback period for volatility (trading days)
- `ma_long` - Long moving average period (trading days)
- `ma_short` - Short moving average period (trading days)

### `winsorize_percentile`
**Type**: Number (0.0-0.5)  
**Default**: `0.01`  
**Description**: Percentile for winsorization (outlier handling).

## Box Features

### `box_features`
**Type**: Object  
**Description**: Configuration for box classification features.

**Properties**:
- `enabled` - Whether to enable box classification features (default: `true`)
- `size_categories` - Whether to include size category features (default: `true`)
- `style_categories` - Whether to include style category features (default: `true`)
- `region_categories` - Whether to include region category features (default: `true`)
- `sector_categories` - Whether to include sector category features (default: `true`)
- `encoding_method` - Method for encoding categorical features (default: `"one_hot"`)
- `handle_unknown` - How to handle unknown categories (default: `"ignore"`)

**Example**:
```yaml
box_features:
  enabled: true
  size_categories: true
  style_categories: true
  region_categories: true
  sector_categories: true
  encoding_method: "one_hot"
  handle_unknown: "ignore"
```

## Data Format Configuration

### `data_format_index_order`
**Type**: Array of strings  
**Default**: `["date", "symbol"]`  
**Description**: Expected order of index levels in panel data.

### `validate_data_format`
**Type**: Boolean  
**Default**: `true`  
**Description**: Whether to validate data format consistency.

### `auto_fix_data_format`
**Type**: Boolean  
**Default**: `true`  
**Description**: Whether to automatically fix data format issues.

### `standardize_panel_output`
**Type**: Boolean  
**Default**: `true`  
**Description**: Whether to standardize panel data output format.

## Factor Model Parameters

### `factors`
**Type**: Array of strings  
**Default**: `["MKT", "SMB", "HML", "RMW", "CMA"]`  
**Description**: Factor names for factor models (FF5).

### `factor_timing`
**Type**: Object  
**Default**: `{}`  
**Description**: Timing configuration for factor models.

### `risk_metrics`
**Type**: Object  
**Default**: `{}`  
**Description**: Risk metrics configuration.

### `sequence_features`
**Type**: Object  
**Default**: `{}`  
**Description**: Configuration for sequence features (LSTM models).

## Configuration Examples

### Example 1: Basic ML Strategy
```yaml
feature_engineering:
  enabled_features: ['momentum', 'volatility', 'technical']
  momentum_periods: [21, 63, 252]
  volatility_windows: [20, 60]
  include_technical: true
  technical_indicators: ['rsi', 'macd', 'bollinger_bands']
  normalize_features: true
  normalization_method: 'robust'
  min_ic_threshold: 0.02
  max_features: 30
```

### Example 2: FF5 Factor Model
```yaml
feature_engineering:
  enabled_features: ['fama_french_factors']
  include_technical: false
  include_cross_sectional: false
  include_theoretical: false
  factors: ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
  normalize_features: false
```

### Example 3: Fama-MacBeth Cross-Sectional
```yaml
feature_engineering:
  enabled_features: ['momentum', 'volatility']
  include_cross_sectional: true
  cross_sectional_features:
    - 'market_cap'
    - 'book_to_market'
    - 'size'
    - 'value'
    - 'momentum'
    - 'volatility'
  cross_sectional_lookback:
    momentum: 252
    volatility: 60
    ma_long: 200
    ma_short: 50
  winsorize_percentile: 0.01
  normalize_features: true
  normalization_method: 'minmax'
```

### Example 4: Full Feature Set with Box Features
```yaml
feature_engineering:
  enabled_features: ['momentum', 'volatility', 'technical', 'volume', 'trend']
  momentum_periods: [21, 63, 126, 252]
  volatility_windows: [20, 60]
  trend_periods: [10, 20, 50]
  volume_periods: [5, 10, 20]
  technical_indicators: ['rsi', 'macd', 'bollinger_bands', 'stochastic', 'williams_r']
  volume_indicators: ['obv', 'vwap', 'ad_line']
  include_technical: true
  include_cross_sectional: true
  cross_sectional_features:
    - 'market_cap'
    - 'book_to_market'
    - 'size'
    - 'value'
    - 'momentum'
    - 'volatility'
  box_features:
    enabled: true
    size_categories: true
    style_categories: true
    region_categories: true
    sector_categories: true
    encoding_method: 'one_hot'
  normalize_features: true
  normalization_method: 'robust'
  min_ic_threshold: 0.03
  max_features: 50
  handle_missing: 'interpolate'
  enable_missing_value_monitoring: true
```

## Best Practices

### 1. Feature Selection
- Start with a reasonable number of features (20-50) and expand based on model performance
- Use `min_ic_threshold` to filter out low-quality features
- Consider computational cost when selecting feature types

### 2. Time Periods
- Use multiple time periods to capture different market dynamics
- Common periods: 21 (1 month), 63 (3 months), 126 (6 months), 252 (12 months)
- Balance between signal strength and noise reduction

### 3. Missing Value Handling
- Use `interpolate` for time series data
- Monitor missing value rates with `enable_missing_value_monitoring`
- Set appropriate `missing_value_threshold` for your data quality

### 4. Normalization
- Use `robust` normalization for financial data (less sensitive to outliers)
- Always normalize features before training ML models
- Consider different normalization methods for different feature types

### 5. Cross-Sectional Features
- Essential for Fama-MacBeth models
- Use appropriate lookback periods for each feature type
- Apply winsorization to handle outliers

### 6. Box Features
- Enable for ML models to capture style effects
- Use `one_hot` encoding for interpretability
- Consider computational cost with many categories

## Troubleshooting

### Common Issues

#### 1. High Missing Value Rates
**Problem**: Many features have high missing value rates  
**Solution**: 
- Check data quality and availability
- Adjust `missing_value_threshold`
- Use different `handle_missing` strategies
- Consider shorter lookback periods

#### 2. Feature Selection Issues
**Problem**: Too few features selected  
**Solution**:
- Lower `min_ic_threshold`
- Increase `max_features`
- Check feature importance thresholds
- Verify data alignment

#### 3. Memory Issues
**Problem**: Out of memory during feature computation  
**Solution**:
- Reduce `max_features`
- Use fewer time periods
- Disable unnecessary feature types
- Process data in smaller chunks

#### 4. Validation Errors
**Problem**: Schema validation fails  
**Solution**:
- Check parameter types and ranges
- Verify enum values
- Ensure required parameters are present
- Use configuration validation tools

#### 5. Performance Issues
**Problem**: Feature computation is slow  
**Solution**:
- Reduce number of features
- Use fewer time periods
- Enable caching
- Optimize data loading

### Debugging Tips

1. **Enable Logging**: Set appropriate log levels to see detailed feature computation
2. **Validate Configurations**: Use schema validation before running experiments
3. **Monitor Resources**: Track memory and CPU usage during feature computation
4. **Test Incrementally**: Start with simple configurations and add complexity gradually
5. **Check Data Quality**: Verify input data quality and alignment

### Getting Help

1. **Check Logs**: Review detailed logs for error messages
2. **Validate Schema**: Use configuration validation tools
3. **Review Examples**: Look at working configuration examples
4. **Test Parameters**: Try different parameter combinations
5. **Check Documentation**: Refer to this guide and code documentation

## Advanced Configuration

### Custom Feature Engineering
For advanced users, the system supports custom feature engineering through:
- Custom feature calculators
- Pipeline extensions
- Custom validation rules
- Advanced caching strategies

### Performance Optimization
- Use feature caching for repeated computations
- Optimize data loading and preprocessing
- Consider parallel processing for large datasets
- Monitor and tune memory usage

### Integration with Models
Different model types have specific feature requirements:
- **ML Models**: Require normalized, validated features
- **Factor Models**: Use factor data and cross-sectional features
- **LSTM Models**: Support sequence features and time series data
- **Fama-MacBeth**: Require cross-sectional features and proper alignment

This guide provides comprehensive coverage of all feature engineering configuration options. For specific use cases or advanced scenarios, refer to the code documentation and examples in the repository.

