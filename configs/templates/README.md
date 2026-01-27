# Configuration Templates
# ====================

This directory contains comprehensive configuration templates for different trading strategies and models in the Bloomberg competition trading system.

## Available Templates

### 1. XGBoost Strategy Template
**File**: `xgboost_strategy_template.yaml`

**Overview**: Complete configuration for XGBoost-based trading strategies with technical feature engineering and comprehensive hyperparameter optimization.

**Key Features**:
- XGBoost model with full hyperparameter search
- Technical indicators and feature engineering
- Time series cross-validation
- Integrated hyperparameter optimization with Optuna
- Production-ready backtesting configuration

**Best For**: Machine learning strategies using technical analysis features

**Usage**:
```bash
# Copy and customize
cp configs/templates/xgboost_strategy_template.yaml configs/my_xgboost_experiment.yaml

# Run the experiment
poetry run python run_experiment.py --config configs/my_xgboost_experiment.yaml
```

### 2. LSTM Strategy Template
**File**: `lstm_strategy_template.yaml`

**Overview**: Comprehensive configuration for LSTM neural network trading strategies with sequence-based prediction and time series feature engineering.

**Key Features**:
- LSTM neural network with architecture optimization
- Sequence-based feature engineering
- Time series aware cross-validation
- Advanced hyperparameter search for neural networks
- Specialized monitoring for neural network training

**Best For**: Deep learning strategies with time series patterns

**Usage**:
```bash
# Copy and customize
cp configs/templates/lstm_strategy_template.yaml configs/my_lstm_experiment.yaml

# Run the experiment
poetry run python run_experiment.py --config configs/my_lstm_experiment.yaml
```

### 3. Fama-French 5-Factor Strategy Template
**File**: `ff5_strategy_template.yaml`

**Overview**: Configuration for Fama-French 5-factor model-based trading strategies with factor exposure analysis and timing capabilities.

**Key Features**:
- FF5 factor model implementation
- Factor exposure analysis and timing
- Statistical factor model optimization
- Monthly rebalancing (matches factor data frequency)
- Comprehensive factor attribution analysis

**Best For**: Econometric strategies based on established factor models

**Usage**:
```bash
# Copy and customize
cp configs/templates/ff5_strategy_template.yaml configs/my_ff5_experiment.yaml

# Run the experiment
poetry run python run_experiment.py --config configs/my_ff5_experiment.yaml
```

### 4. MetaModel Template
**File**: `metamodel_template.yaml`

**Overview**: Configuration for MetaModel training and evaluation, combining multiple trading strategies using machine learning techniques.

**Key Features**:
- Multiple combination methods (Equal, Lasso, Ridge, Dynamic)
- Strategy data collection and validation
- Cross-validation for robust weight estimation
- Performance attribution and analysis
- System integration testing

**Best For**: Ensemble strategies combining multiple approaches

**Usage**:
```bash
# Copy and customize
cp configs/templates/metamodel_template.yaml configs/my_metamodel_experiment.yaml

# Run MetaModel training
poetry run python run_experiment.py metamodel --config configs/my_metamodel_experiment.yaml
```

## How to Use Templates

### Step 1: Copy Template
Choose the appropriate template for your strategy and copy it to the `configs/` directory:

```bash
# Example for XGBoost strategy
cp configs/templates/xgboost_strategy_template.yaml configs/my_strategy.yaml
```

### Step 2: Customize Configuration
Edit the copied configuration file to suit your needs:

**Key Sections to Customize**:
- **Experiment Metadata**: Update name, description, and tags
- **Symbols**: Modify the asset universe in `training_setup.parameters.symbols`
- **Date Ranges**: Adjust training and backtesting periods
- **Hyperparameters**: Modify search spaces and optimization parameters
- **Risk Management**: Adjust position limits, stop-loss, and drawdown limits
- **Objectives**: Change optimization objectives and performance targets

### Step 3: Validate Configuration
Run a quick validation check:

```bash
# Validate configuration syntax
python -c "import yaml; yaml.safe_load(open('configs/my_strategy.yaml'))"

# Check required parameters
poetry run python run_experiment.py --config configs/my_strategy.yaml --dry-run
```

### Step 4: Run Experiment
Execute your customized experiment:

```bash
# Run full experiment
poetry run python run_experiment.py --config configs/my_strategy.yaml

# Run with test mode (shorter time period)
poetry run python run_experiment.py --config configs/my_strategy.yaml --test-mode
```

## Customization Guidelines

### Model Selection
- **XGBoost**: Good for technical analysis, feature-rich datasets
- **LSTM**: Best for time series patterns, sequential data
- **FF5**: Ideal for factor-based investing, statistical arbitrage
- **MetaModel**: Use when combining multiple strategies

### Hyperparameter Optimization
- **Trial Count**: Start with 30-50 trials, increase to 100+ for production
- **Objectives**: Choose metrics aligned with your investment goals
- **Search Space**: Customize based on your computational budget
- **Cross-Validation**: Use at least 3 folds for robust validation

### Risk Management
- **Position Limits**: Keep individual positions under 10-15%
- **Stop-Loss**: Set appropriate thresholds (10-20%)
- **Drawdown**: Define maximum acceptable drawdown (15-25%)
- **Rebalancing**: Match frequency to your strategy's signal frequency

### Data Configuration
- **Time Periods**: Use at least 2-3 years for training
- **Asset Universe**: Start with 10-20 liquid assets
- **Frequency**: Match data frequency to strategy needs
- **Quality**: Ensure clean, complete data without gaps

## Computational Requirements

### XGBoost Strategy
- **Training**: 5-15 minutes per model
- **Optimization**: 30-120 minutes (100 trials)
- **Memory**: 2-4 GB
- **Storage**: 100-500 MB

### LSTM Strategy
- **Training**: 10-30 minutes per model
- **Optimization**: 60-180 minutes (50 trials)
- **Memory**: 4-8 GB
- **Storage**: 200-1000 MB
- **GPU**: Recommended but not required

### FF5 Strategy
- **Training**: 2-5 minutes per model
- **Optimization**: 15-45 minutes (30 trials)
- **Memory**: 1-2 GB
- **Storage**: 50-200 MB

### MetaModel
- **Training**: 5-15 minutes
- **Optimization**: 30-90 minutes (50 trials)
- **Memory**: 1-3 GB
- **Storage**: 100-500 MB

## Best Practices

### Before Running
1. **Check Data Quality**: Ensure no missing values or outliers
2. **Validate Configuration**: Syntax check all YAML files
3. **Set Environment Variables**: Configure WandB and API keys
4. **Resource Planning**: Ensure sufficient computational resources

### During Development
1. **Start Simple**: Begin with default parameters
2. **Iterative Testing**: Test components individually
3. **Monitor Progress**: Use WandB for experiment tracking
4. **Save Intermediate Results**: Cache models and features

### For Production
1. **Robust Validation**: Use cross-validation and out-of-sample testing
2. **Performance Monitoring**: Track model degradation over time
3. **Risk Controls**: Implement comprehensive risk management
4. **Documentation**: Document all parameter choices and results

## Troubleshooting

### Common Issues
- **Import Errors**: Check Python path and dependencies
- **Data Issues**: Verify symbol validity and date ranges
- **Memory Issues**: Reduce batch size or sequence length
- **Optimization Errors**: Check search space definitions

### Getting Help
1. Check the main README.md for system overview
2. Review configuration comments for parameter explanations
3. Run test scripts to validate individual components
4. Check WandB logs for detailed error messages

## Advanced Features

### Custom Models
To add custom models:
1. Implement the model interface in `src/trading_system/models/implementations/`
2. Add search space definition to `SearchSpaceBuilder`
3. Create custom configuration template
4. Register model in `ModelFactory`

### Custom Features
To add custom features:
1. Implement feature extraction in `src/trading_system/feature_engineering/`
2. Add feature configuration to template
3. Test feature importance and correlation
4. Update search space if optimizing feature parameters

### Ensemble Methods
All templates support ensemble combinations:
- **XGBoost + LSTM**: Combine technical and sequence approaches
- **Multiple Factors**: Use different factor models
- **Strategy Ensembles**: Combine multiple MetaModels
- **Time-Based Ensembles**: Different models for different market regimes

## Contributing

When creating new templates:
1. **Comprehensive Comments**: Document all parameters
2. **Example Values**: Provide reasonable default values
3. **Usage Instructions**: Include clear usage examples
4. **Requirements**: Specify computational requirements
5. **Best Practices**: Include model-specific recommendations

## Support

For questions or issues:
1. Check existing documentation and comments
2. Review test scripts for usage examples
3. Examine WandB logs for detailed information
4. Refer to the main project README for system overview