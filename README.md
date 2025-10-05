# Bloomberg Competition Trading System

A production-ready quantitative trading framework built for the Bloomberg terminal competition.

## Features

✅ **Complete Pipeline**: Data acquisition → Feature engineering → Strategy execution → Performance calculation → Experiment tracking

✅ **MetaModel Strategy Combination**: Machine learning-based combination of multiple trading strategies (Ridge, Lasso, Dynamic)

✅ **Hyperparameter Optimization**: Comprehensive optimization with Optuna for strategies, MetaModels, and ML models

✅ **Multiple Strategy Types**: Dual Momentum, ML-based, Fama-French 5-factor, and custom strategies

✅ **Team Collaboration**: Config-driven approach with unified CLI for different experiment types

✅ **Advanced Experiment Tracking**: Automatic logging to Weights & Biases with comprehensive visualizations and attribution analysis

✅ **Robust Data Pipeline**: YFinance and Fama-French data providers with retry logic, data validation, and error handling

✅ **Professional Backtesting**: Time-weighted returns, realistic transaction costs, risk metrics, and benchmark-relative performance

✅ **Model Persistence**: Save and load trained models using ModelRegistry with versioning and artifacts

## Quick Start

### 1. Environment Setup

```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### 2. Configuration

Edit `configs/strategy_config.yaml` to customize:
- Strategy parameters (lookback periods, asset selection criteria)
- Asset universe (symbols to trade)
- Backtest settings (initial capital, transaction costs)
- Experiment configuration

### 3. Run the Strategy

```bash
# Test the system
python test_pipeline.py --skip-long-test

# Run a full backtest
python run_strategy.py --experiment-name "my_experiment"

# Run in test mode (shorter time period)
python run_strategy.py --test-mode
```

### 4. Unified Experiment Runner

The system now provides a unified CLI for running different types of experiments:

```bash
# === 标准端到端实验 (训练+回测) ===
# 运行包含模型训练和策略回测的完整流程
poetry run python run_experiment.py experiment                          # 使用默认配置
poetry run python run_experiment.py experiment --config configs/ml_strategy_config_new.yaml       # XGBoost策略
poetry run python run_experiment.py experiment -c configs/e2e_ff5_experiment.yaml              # Fama-French 5因子模型
poetry run python run_experiment.py experiment -c configs/lstm_strategy_config.yaml             # LSTM策略

# === 超参数优化 ===
# 优化机器学习模型的超参数
poetry run python run_experiment.py optimize --type ml --trials 50 --config configs/ml_strategy_config_new.yaml
poetry run python run_experiment.py optimize --type ml --trials 100 --sampler tpe --metric sharpe_ratio

# === MetaModel训练 ===
# 组合多个策略的元模型
poetry run python run_experiment.py metamodel --method ridge --alpha 0.5
poetry run python run_experiment.py metamodel --config configs/metamodel_experiment_config.yaml
poetry run python run_experiment.py metamodel --method lasso --alpha 0.1 --strategies "DualMomentum,MLStrategy,FF5Strategy"
```

### 5. 具体使用场景推荐

根据你的需求，选择以下命令：

#### 场景1：只想训练+回测一个XGBoost策略
```bash
poetry run python run_experiment.py experiment -c configs/ml_strategy_config_new.yaml
```

#### 场景2：训练+超参数优化+回测 (推荐)
```bash
poetry run python run_experiment.py optimize --type ml --trials 50 --config configs/ml_strategy_config_new.yaml
```

#### 场景3：训练Fama-French 5因子模型
```bash
poetry run python run_experiment.py experiment -c configs/e2e_ff5_experiment.yaml
```

#### 场景4：训练LSTM深度学习策略
```bash
poetry run python run_experiment.py experiment -c configs/lstm_strategy_config.yaml
```

#### 场景5：组合多个策略 (高级)
```bash
poetry run python run_experiment.py metamodel --config configs/metamodel_experiment_config.yaml
```

## Architecture

```
src/trading_system/
├── data/                    # Data providers (YFinance, Fama-French, StrategyDataCollector)
├── strategies/              # Strategy implementations (Dual Momentum, ML, FF5, BaseStrategy)
├── backtest/                # Backtest engine with performance calculation
├── models/                  # Model training and serving infrastructure
│   ├── base/               # BaseModel interface and ModelRegistry
│   ├── training/           # TrainingPipeline and MetaModel training
│   └── finetune/           # Hyperparameter optimization with Optuna
├── orchestration/           # System orchestration (SystemOrchestrator, MetaModel)
├── feature_engineering/     # Technical indicators and feature pipelines
├── utils/                   # Utilities (WandB logger, risk metrics, position sizing)
└── experiment_orchestrator.py  # Unified experiment orchestration
```

## MetaModel Strategy Combination

The system includes a sophisticated MetaModel for combining multiple trading strategies:

### Features
- **Multiple Combination Methods**: Equal weighting, Lasso, Ridge regression, Dynamic optimization
- **Strategy Weight Learning**: Automatically learns optimal weights from historical performance
- **Cross-Validation**: Time series cross-validation with purge and embargo periods
- **Model Persistence**: Save and load trained MetaModels using ModelRegistry
- **Performance Attribution**: Analyze individual strategy contributions to combined performance

### Usage Examples

```bash
# Train a MetaModel with Ridge regression
poetry run python run_experiment.py metamodel --method ridge --alpha 0.5

# Train with custom strategy list and date range
poetry run python run_experiment.py metamodel \
  --method lasso \
  --alpha 0.1 \
  --strategies "DualMomentumStrategy,MLStrategy,FF5Strategy" \
  --start-date 2022-01-01 \
  --end-date 2023-12-31

# Use comprehensive YAML configuration
poetry run python run_experiment.py metamodel \
  --config configs/metamodel_experiment_config.yaml
```

### Configuration Example

```yaml
metamodel_training:
  method: "ridge"              # Combination method
  alpha: 0.5                   # Regularization strength
  strategies:                  # Strategies to combine
    - "DualMomentumStrategy"
    - "MLStrategy"
    - "FF5Strategy"
  start_date: "2022-01-01"
  end_date: "2023-12-31"
  use_cross_validation: true
  cv_folds: 5
```

## Hyperparameter Optimization

The system includes comprehensive hyperparameter optimization capabilities using Optuna with production-ready features:

### Features
- **Multiple Optimization Types**: MetaModel parameters, strategy parameters, ML model hyperparameters
- **Advanced Samplers**: TPE, Random, CMA-ES, Grid sampling algorithms
- **Pruning and Early Stopping**: Efficient search with median and hyperband pruning
- **Experiment Tracking**: Full integration with WandB for optimization trials
- **Parallel Optimization**: Multi-trial parallel execution
- **Model-Specific Search Spaces**: Pre-configured search spaces for XGBoost, LSTM, and FF5 models
- **Flexible Configuration**: YAML-based configuration with detailed parameter control

### Supported Model Types

#### XGBoost Models
**Hyperparameters Available:**
- `n_estimators`: [50, 500] - Number of trees
- `max_depth`: [3, 12] - Tree depth
- `learning_rate`: [0.01, 0.3] (log-scale) - Step size shrinkage
- `subsample`: [0.6, 1.0] - Sample ratio
- `colsample_bytree`: [0.6, 1.0] - Feature sampling ratio
- `reg_alpha`: [0.0, 1.0] - L1 regularization
- `reg_lambda`: [1.0, 5.0] - L2 regularization
- `min_child_weight`: [1, 10] - Minimum sum of instance weight

#### LSTM Models
**Hyperparameters Available:**
- `hidden_size`: [32, 64, 128, 256] - Hidden layer size
- `num_layers`: [1, 4] - Number of LSTM layers
- `dropout`: [0.1, 0.5] - Dropout rate
- `sequence_length`: [10, 20, 30, 60] - Input sequence length
- `learning_rate`: [0.001, 0.01] (log-scale) - Learning rate
- `batch_size`: [16, 32, 64] - Training batch size
- `num_epochs`: [50, 200] - Training epochs

#### Fama-French 5-Factor Models
**Hyperparameters Available:**
- `regularization`: [none, ridge] - Regularization method
- `alpha`: [0.01, 10.0] (log-scale) - Regularization strength
- `standardize`: [true, false] - Data standardization

### Usage Examples

```bash
# Optimize MetaModel parameters
poetry run python run_experiment.py optimize --type metamodel --trials 50

# Optimize strategy parameters with TPE sampler
poetry run python run_experiment.py optimize --type strategy --trials 100 --sampler tpe

# Optimize XGBoost model hyperparameters
poetry run python run_experiment.py optimize --type xgboost --trials 100 --metric sharpe_ratio

# Optimize LSTM model hyperparameters
poetry run python run_experiment.py optimize --type lstm --trials 50 --metric sortino_ratio

# Optimize FF5 model hyperparameters
poetry run python run_experiment.py optimize --type ff5 --trials 30 --metric r2

# Custom optimization with timeout and sampler
poetry run python run_experiment.py optimize \
  --type metamodel \
  --trials 100 \
  --timeout 300 \
  --sampler cmaes \
  --metric r2
```

### Configuration-Based Optimization

#### XGBoost Optimization Example
```yaml
xgboost_hyperparameter_optimization:
  enabled: true
  optimization_method: "optuna"
  n_trials: 100
  cv_folds: 5
  objective: "sharpe_ratio"
  direction: "maximize"

  sampler:
    type: "tpe"
    seed: 42

  pruner:
    type: "median"
    n_startup_trials: 5
    n_warmup_steps: 3

  search_space:
    preset: "xgboost_default"  # Use built-in preset

    # Custom parameters override preset
    custom_space:
      n_estimators:
        type: "int"
        low: 50
        high: 500
        step: 10
      max_depth:
        type: "int"
        low: 3
        high: 12
      learning_rate:
        type: "float"
        low: 0.01
        high: 0.3
        step: 0.01
        log_scale: true

  logging:
    log_optimization: true
    log_all_trials: true
    create_optimization_plot: true
```

#### LSTM Optimization Example
```yaml
lstm_hyperparameter_optimization:
  enabled: true
  optimization_method: "optuna"
  n_trials: 50
  cv_folds: 3
  objective: "sharpe_ratio"
  direction: "maximize"

  sampler:
    type: "tpe"
    seed: 42

  pruner:
    type: "median"
    n_startup_trials: 5
    n_warmup_steps: 3

  search_space:
    preset: "lstm_default"

    custom_space:
      hidden_size:
        type: "categorical"
        choices: [32, 64, 128, 256]
      num_layers:
        type: "int"
        low: 1
        high: 4
        step: 1
      dropout:
        type: "float"
        low: 0.1
        high: 0.5
        step: 0.05
      sequence_length:
        type: "categorical"
        choices: [10, 20, 30, 60]
```

#### FF5 Optimization Example
```yaml
ff5_hyperparameter_optimization:
  enabled: true
  optimization_method: "optuna"
  n_trials: 30
  cv_folds: 3
  objective: "r2"
  direction: "maximize"

  sampler:
    type: "tpe"
    seed: 42

  pruner:
    type: "median"
    n_startup_trials: 5
    n_warmup_steps: 3

  search_space:
    preset: "ff5_default"

    custom_space:
      regularization:
        type: "categorical"
        choices: ["none", "ridge"]
      alpha:
        type: "float"
        low: 0.01
        high: 10.0
        step: 0.1
        log_scale: true
      standardize:
        type: "categorical"
        choices: [true, false]
```

### Search Space Configuration

#### Parameter Types
- **`int`**: Integer parameters with `low`, `high`, and optional `step`
- **`float`**: Float parameters with `low`, `high`, optional `step` and `log_scale`
- **`categorical`**: Discrete parameters with `choices` array
- **`bool`**: Boolean parameters (true/false)

#### Preset Search Spaces
The system provides built-in presets for common use cases:

- **`xgboost_default`**: Comprehensive XGBoost parameter search (120 trials recommended)
- **`xgboost_fast`**: Reduced search space for quick testing (50 trials recommended)
- **`lstm_default`**: Full LSTM architecture search (120 trials recommended)
- **`lstm_fast`**: Simplified LSTM search (50 trials recommended)
- **`lstm_deep`**: Deep LSTM architectures (150 trials recommended)
- **`ff5_default`**: FF5 factor model optimization (30 trials recommended)

### Optimization Objectives

#### Available Metrics
- **Performance Metrics**: `sharpe_ratio`, `sortino_ratio`, `calmar_ratio`, `information_ratio`
- **Risk Metrics**: `max_drawdown`, `volatility`, `var_95`
- **Statistical Metrics**: `r2`, `mse`, `mae`, `ic` (Information Coefficient)
- **Custom Metrics**: Any metric that can be calculated from backtest results

#### Direction
- **`maximize`**: Higher values are better (e.g., Sharpe ratio)
- **`minimize`**: Lower values are better (e.g., MSE, maximum drawdown)

### Advanced Features

#### Cross-Validation Integration
- **Time Series Split**: Proper time-aware cross-validation
- **Purge Period**: Days to purge between train and test
- **Embargo Period**: Days to embargo before test period

#### Early Stopping
- **Median Pruning**: Stops trials that perform worse than median
- **Hyperband Pruning**: Asynchronous successive halving
- **Successive Halving**: Budget allocation to promising trials

#### Logging and Visualization
- **WandB Integration**: Automatic logging of all trials
- **Optimization Plots**: Interactive optimization history
- **Parameter Importance**: Most influential parameters
- **Parallel Coordinates**: Multi-dimensional parameter relationships

### Best Practices

1. **Trial Count**: Start with 30-50 trials for exploration, then 100-200 for refinement
2. **Sampler Choice**: Use TPE for most cases, Random for exploration, CMA-ES for complex spaces
3. **Metric Selection**: Choose metrics appropriate to your strategy goals
4. **Cross-Validation**: Use at least 3-fold CV, more for noisy strategies
5. **Early Stopping**: Enable pruning to save computation time
6. **Log Analysis**: Analyze optimization results to understand parameter importance

### Model-Specific Considerations

#### XGBoost
- Use `log_scale: true` for learning rate and regularization parameters
- Limit tree depth to prevent overfitting
- Consider `subsample` and `colsample_bytree` for robustness

#### LSTM
- Start with smaller architectures (1-2 layers)
- Use dropout to prevent overfitting
- Sequence length should match your data frequency
- Monitor for vanishing/exploding gradients

#### FF5
- Factor models typically have fewer tunable parameters
- Focus on regularization strength
- Standardization can significantly impact performance

## Key Components

### YFinanceProvider
- Automatic retry on API failures
- Data validation and cleaning
- Rate limiting and error handling
- Support for multiple data types

### DualMomentumStrategy
- Absolute momentum filter (positive returns only)
- Relative momentum selection (top performers)
- Equal-weight allocation
- Cash position for risk management

### StandardBacktest
- Time-weighted returns calculation
- Comprehensive risk metrics
- Transaction cost modeling
- Benchmark-relative performance

### WandBLogger
- Automatic experiment tracking
- Interactive visualizations
- Hyperparameter logging
- Team collaboration features

## Configuration Example

```yaml
strategy:
  lookback_days: 252          # 1 year momentum
  top_n_assets: 5             # Select top 5 assets
  minimum_positive_assets: 3  # Stay invested only if 3+ assets positive

universe:
  all_assets:
    - "SPY"    # S&P 500
    - "QQQ"    # Nasdaq 100
    - "IWM"    # Russell 2000
    - "AGG"    # Bonds
    - "TLT"    # Long-term Treasury

backtest:
  initial_capital: 1000000    # $1M starting capital
  transaction_cost: 0.001     # 0.1% per trade
  start_date: "2018-01-01"
  end_date: "2024-12-31"
```

## Environment Variables

Required:
```bash
WANDB_API_KEY=your_wandb_key          # From wandb.ai
ALPHA_VANTAGE_API_KEY=your_av_key     # Backup data source
```

## Testing

```bash
# Run all tests
python test_pipeline.py

# Skip long-running tests
python test_pipeline.py --skip-long-test

# Test specific components
poetry run python -c "
from trading_system.data.yfinance_provider import YFinanceProvider
provider = YFinanceProvider()
print('SPY valid:', provider.validate_symbol('SPY'))
"
```

## First Week Milestones ✅

**Day 1-3: Core Infrastructure**
- [x] Poetry project setup
- [x] YFinance data provider with error handling
- [x] Strategy interface and dual momentum implementation
- [x] Basic backtest engine
- [x] WandB integration

**Day 4-5: Pipeline Integration**
- [x] Configuration management
- [x] Strategy runner orchestration
- [x] End-to-end pipeline testing
- [x] Performance metrics calculation

**Day 6-7: Team Readiness**
- [x] Complete pipeline validation
- [x] Team-friendly configuration system
- [x] Documentation and examples
- [x] Test coverage for all components

## Team Usage

### For Non-Programmers
1. **Basic Strategy Testing**: Modify `configs/strategy_config.yaml` to change parameters
2. **Run Experiments**: Use the unified CLI for different experiment types:
   ```bash
   poetry run python run_experiment.py --config my_config.yaml
   poetry run python run_experiment.py metamodel --method ridge --alpha 0.5
   ```
3. **Hyperparameter Optimization**: Find optimal parameters automatically:
   ```bash
   poetry run python run_experiment.py optimize --type strategy --trials 50
   ```
4. **View Results**: Check results in `./results/` folder and WandB dashboard

### For Developers
1. **Strategy Development**: Implement new strategies by extending `BaseStrategy`
2. **MetaModel Development**: Create new combination methods in `MetaModel` class
3. **Data Integration**: Add new data providers implementing the same interface
4. **Optimization**: Define custom search spaces for new parameters
5. **Performance Analysis**: Extend performance metrics and attribution analysis
6. **Model Persistence**: Use ModelRegistry for saving/loading trained models

### For Quant Researchers
1. **Strategy Combination**: Use MetaModel to combine multiple strategies
2. **Parameter Optimization**: Use hyperparameter optimization for robustness testing
3. **Performance Attribution**: Analyze strategy contributions and risk factors
4. **Backtesting**: Use comprehensive backtesting with realistic transaction costs

## Performance Metrics

The system calculates:
- Total and annualized returns
- Volatility and Sharpe ratio
- Maximum drawdown and Calmar ratio
- Alpha and beta vs benchmark
- Information ratio and tracking error
- Win rate and profit factor
- Turnover and concentration risk

## Next Steps

1. **Add More Strategies**: Implement mean-reversion, value, or carry strategies
2. **Parameter Optimization**: Integrate Optuna for hyperparameter tuning
3. **Risk Management**: Add volatility targeting, stop-losses
4. **Alternative Data**: Add Alpha Vantage as backup data source
5. **Real-time Trading**: Extend for live trading capabilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your strategy extending `BaseStrategy`
4. Add tests for your implementation
5. Run the test suite
6. Submit a pull request

## License

This project is for the Bloomberg competition. Please ensure compliance with competition rules.