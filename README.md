# Bloomberg Competition Trading System

A production-ready quantitative trading framework built for the Bloomberg terminal competition.

## Features

✅ **Complete Pipeline**: Data acquisition → Strategy execution → Performance calculation → Experiment tracking

✅ **Dual Momentum Strategy**: Combines absolute and relative momentum with built-in risk management

✅ **Team Collaboration**: Config-driven approach allowing non-programmers to modify strategies

✅ **Experiment Tracking**: Automatic logging to Weights & Biases with comprehensive visualizations

✅ **Robust Data Pipeline**: YFinance provider with retry logic, data validation, and error handling

✅ **Professional Backtesting**: Time-weighted returns, risk metrics, and benchmark-relative performance

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

## Architecture

```
src/trading_system/
├── data/           # Data providers (YFinance with retry logic)
├── strategies/     # Strategy implementations (Dual Momentum + BaseStrategy)
├── backtest/       # Backtest engine with performance calculation
├── utils/          # WandB logger for experiment tracking
├── config/         # Configuration management
└── strategy_runner.py  # Main orchestrator
```

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
1. Modify `configs/strategy_config.yaml` to change parameters
2. Run `python run_strategy.py` with your experiment name
3. View results in the generated `./results/` folder and WandB dashboard

### For Developers
1. Implement new strategies by extending `BaseStrategy`
2. Add new data providers implementing the same interface
3. Extend performance metrics in the backtest engine
4. Customize visualizations in the WandB logger

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