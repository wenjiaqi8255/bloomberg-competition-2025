# Trading System Configuration Guide

## Overview

This directory contains all configuration files for the trading system. The configuration system has been reorganized to provide clear structure, comprehensive validation, and easy maintenance.

## Directory Structure

```
configs/
├── README.md                  # This file - configuration guide
├── CONFIG_REGISTRY.yaml       # Central registry of all configurations
├── schemas/                   # JSON Schema validation files
│   ├── base_schemas.json      # Common schemas
│   ├── single_experiment_schema.json
│   ├── multi_model_schema.json
│   └── prediction_schema.json
├── templates/                 # Configuration templates
│   ├── ff5_strategy_template.yaml
│   ├── metamodel_template.yaml
│   └── ...
└── [active configs]          # Currently active configuration files
```

## Configuration Types

### 1. Single Experiment Configurations
**Purpose**: Train a single model and run backtests

**Use Cases**:
- Model development and testing
- Strategy validation
- Performance analysis

**Available Configurations**:
- `ff5_box_based_experiment.yaml` - FF5 factor model with Box-First portfolio construction
- `fama_macbeth_box_based_config.yaml` - Fama-MacBeth cross-sectional model with Box-First
- `e2e_ff5_experiment.yaml` - FF5 model with traditional quantitative optimization
- `ml_strategy_config_new.yaml` - XGBoost ML model with Box-First portfolio
- `lstm_strategy_config.yaml` - LSTM neural network strategy

**Entry Point**: `src/use_case/single_experiment/run_experiment.py`

### 2. Multi-Model Configurations
**Purpose**: Train multiple models and learn optimal combination weights

**Use Cases**:
- Ensemble model development
- Model comparison and selection
- Meta-learning experiments

**Available Configurations**:
- `multi_model_experiment.yaml` - Full multi-model ensemble experiment
- `multi_model_quick_test.yaml` - Quick test version with reduced parameters

**Entry Point**: `src/use_case/multi_model_experiment/run_multi_model_experiment.py`

### 3. Prediction Configurations
**Purpose**: Use trained models for real-time predictions

**Use Cases**:
- Production predictions
- Model deployment
- Live trading signals

**Available Configurations**:
- `prediction_meta_config.yaml` - Meta-strategy predictions (combines multiple models)
- `prediction_config.yaml` - Single model FF5 predictions
- `prediction_quantitative_config.yaml` - Quantitative portfolio predictions

**Entry Point**: `src/use_case/prediction/run_prediction.py`

### 4. System Configurations
**Purpose**: Complete system-level configurations

**Use Cases**:
- Production system setup
- End-to-end optimization
- System integration

**Available Configurations**:
- `optimal_system_config.yaml` - Optimal system configuration
- `portfolio_construction_config.yaml` - Portfolio construction focused config

## Available Options

### Strategy Types
- `ml` - Machine learning strategies (XGBoost, LSTM, etc.)
- `fama_macbeth` - Fama-MacBeth cross-sectional model
- `fama_french_5` - Fama-French 5-factor model
- `ff5_regression` - FF5 regression (alias for fama_french_5)
- `meta` - Meta-strategy (combines multiple models)

### Portfolio Construction Methods
- `quantitative` - Traditional quantitative optimization
- `box_based` - Box-First methodology for systematic diversification

### Allocation Methods
- `equal` - Equal weight allocation
- `signal_proportional` - Signal strength proportional allocation

### Data Providers
- `YFinanceProvider` - Yahoo Finance data provider

### Factor Providers
- `FF5DataProvider` - Fama-French 5-factor data
- `CountryRiskProvider` - Country risk factor data

### Feature Engineering Options

For detailed feature engineering configuration, see [FEATURE_ENGINEERING_GUIDE.md](./FEATURE_ENGINEERING_GUIDE.md).

Key configuration areas:
- **Feature Types**: `momentum`, `volatility`, `technical`, `volume`, `trend`, `fama_french_factors`
- **Time Periods**: Configure lookback periods for different indicators
- **Calculation Methods**: Choose between simple, exponential, and advanced methods
- **Technical Indicators**: RSI, MACD, Bollinger Bands, Stochastic, Williams %R, etc.
- **Cross-Sectional Features**: Market cap, book-to-market, momentum, volatility proxies
- **Box Features**: One-hot encoded box classifications for ML models
- **Validation**: IC threshold, significance testing, feature importance filtering
- **Missing Value Handling**: Interpolation, forward-fill, drop strategies with monitoring

Quick example:
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
```

## How to Choose the Right Configuration

### For New Experiments
1. **Start with templates**: Use files in `templates/` directory
2. **Choose by strategy type**:
   - Factor models → `ff5_strategy_template.yaml`
   - Machine learning → `xgboost_strategy_template.yaml`
   - Multi-model → `metamodel_template.yaml`
3. **Modify parameters** as needed
4. **Validate** using the config management tool

### For Production Use
1. **Use active configurations** from the registry
2. **Check last tested date** in `CONFIG_REGISTRY.yaml`
3. **Validate** before deployment
4. **Monitor performance** and update as needed

### For Model Comparison
1. **Use multi-model configurations** for ensemble approaches
2. **Use single experiment configs** for individual model testing
3. **Compare results** using the reporting system

## Configuration Management Tools

### Command Line Tool
Use `tools/config_management.py` for configuration management:

```bash
# Validate a configuration
python tools/config_management.py validate configs/ff5_box_based_experiment.yaml

# List all available configurations
python tools/config_management.py list

# List configurations by type
python tools/config_management.py list --type single_experiment

# Generate a new template
python tools/config_management.py generate single_experiment new_config.yaml --strategy-type xgboost

# Migrate a legacy configuration
python tools/config_management.py migrate old_config.yaml new_config.yaml --type single_experiment --description "Migrated config"

# Show available options
python tools/config_management.py options

# Get information about a specific configuration
python tools/config_management.py info ff5_box_based
```

### Validation
All configurations are validated using:
1. **JSON Schema validation** - Ensures proper structure and types
2. **Business logic validation** - Checks configuration consistency
3. **Data validation** - Verifies data provider settings

## Configuration Examples

### Basic Single Experiment
```yaml
data_provider:
  type: "YFinanceProvider"
  parameters:
    max_retries: 3
    retry_delay: 1.0

training_setup:
  model:
    model_type: "fama_french_5"
    config:
      regularization: "ridge"
      alpha: 1.0
  feature_engineering:
    include_cross_sectional: true
    normalize_features: true
  parameters:
    start_date: "2024-01-01"
    end_date: "2024-12-31"
    symbols: ["AAPL", "MSFT", "GOOGL"]

strategy:
  type: "fama_french_5"
  name: "FF5_Strategy"
  parameters:
    model_id: "placeholder_model_id"
    portfolio_construction:
      method: "box_based"
      stocks_per_box: 3

backtest:
  start_date: "2025-01-01"
  end_date: "2025-12-31"
  initial_capital: 1000000
  benchmark_symbol: "SPY"
```

### Multi-Model Experiment
```yaml
experiment:
  name: "multi_model_experiment"
  output_dir: "./results/multi_model_experiment"

data_provider:
  type: "YFinanceProvider"

base_models:
  - model_type: "ff5_regression"
    hpo_trials: 10
  - model_type: "xgboost"
    hpo_trials: 10

metamodel:
  hpo_trials: 10
  methods_to_try: ["ridge", "equal"]
```

### Prediction Configuration
```yaml
prediction:
  prediction_date: "2024-01-15"

strategy:
  type: "meta"
  name: "MetaStrategy"
  base_model_ids: ["model_1", "model_2"]
  meta_weights:
    model_1: 0.6
    model_2: 0.4

data_provider:
  type: "YFinanceProvider"

universe: ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
```

## Troubleshooting

### Common Issues

1. **Configuration validation fails**
   - Check JSON syntax
   - Verify required fields are present
   - Use `python tools/config_management.py validate <config_file>` for detailed errors

2. **Model training fails**
   - Verify data provider settings
   - Check symbol list and date ranges
   - Ensure sufficient data availability

3. **Backtest fails**
   - Check strategy configuration
   - Verify model_id is correct
   - Ensure backtest date range is valid

4. **Portfolio construction fails**
   - Verify portfolio_construction method
   - Check box configuration for box_based method
   - Ensure sufficient symbols for diversification

### Getting Help

1. **Check the registry**: `CONFIG_REGISTRY.yaml` has complete information about all configurations
2. **Use validation tools**: The config management tool provides detailed error messages
3. **Review examples**: Look at existing working configurations
4. **Check logs**: Enable debug logging for detailed error information

## Migration Guide

### From Legacy Configurations
1. **Identify legacy configs**: Check `CONFIG_REGISTRY.yaml` under `archived_configs`
2. **Use migration tool**: `python tools/config_management.py migrate`
3. **Validate migrated config**: Ensure it passes validation
4. **Test thoroughly**: Run experiments to verify functionality

### Best Practices
1. **Always validate** configurations before use
2. **Use templates** for new configurations
3. **Keep configurations simple** - avoid unnecessary complexity
4. **Document changes** when modifying configurations
5. **Test thoroughly** before production use

## Advanced Features

### Custom Validation
Create custom validation rules by extending the validation framework in `src/trading_system/validation/`.

### Schema Extensions
Add new configuration options by extending the JSON schemas in `schemas/`.

### Template Customization
Create custom templates by modifying the template generation logic in `tools/config_management.py`.

## Support

For configuration-related issues:
1. Check this documentation
2. Use the configuration management tools
3. Review the validation error messages
4. Check the registry for configuration details
5. Refer to the source code for advanced customization
