# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a quantitative trading system for the Bloomberg competition that implements a complete pipeline from data acquisition to strategy execution with integrated backtesting and experiment tracking.

## Architecture

The system follows a modular architecture with clear separation of concerns:

### Core Components

- **Data Layer** (`src/trading_system/data/`): YFinance and Fama-French 5-factor data providers with retry logic and validation
- **Strategy Layer** (`src/trading_system/strategies/`): Multiple strategy implementations including Dual Momentum, Fama-French 5-factor, Machine Learning, and Core+Satellite approaches
- **Backtesting Engine** (`src/trading_system/backtesting/`): Unified backtesting system with realistic transaction cost modeling and performance metrics
- **Orchestration** (`src/trading_system/orchestrator/`): System orchestrator for managing complex multi-strategy systems with IPS compliance
- **Feature Engineering** (`src/trading_system/feature_engineering/`): Technical indicator calculation and feature preparation for ML strategies

### Key Design Patterns

- **Strategy Pattern**: All trading strategies inherit from `BaseStrategy` with standardized `generate_signals()` interface
- **Configuration-Driven**: All components use YAML configuration files for parameters and asset universe definitions
- **Unified Signal Format**: Trading signals use the `TradingSignal` dataclass with symbol, signal type, strength, and metadata
- **Backward Compatibility**: New backtesting engine maintains compatibility with existing strategy interfaces

## Development Commands

### Environment Setup
```bash
# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Testing
```bash
# Run all tests
python test_pipeline.py

# Skip long-running tests during development
python test_pipeline.py --skip-long-test

# Run specific strategy tests
poetry run python -m pytest src/trading_system/testing/test_core_ffml_strategy.py -v
```

### Running Strategies
```bash
# Test mode (shorter time period for development)
python run_strategy.py --test-mode

# Full backtest with custom experiment name
python run_strategy.py --experiment-name "my_experiment"

# Run specific configuration
python run_strategy.py --config configs/ml_strategy_config.yaml
```

### Code Quality
```bash
# Format code
poetry run black src/
poetry run isort src/

# Lint code
poetry run flake8 src/
```

## Configuration System

The system uses YAML configuration files in `configs/` directory:

- `strategy_config.yaml`: Main dual momentum strategy configuration
- `fama_french_config.yaml`: Fama-French 5-factor strategy settings
- `ml_strategy_config.yaml`: Machine learning strategy parameters

Key configuration sections:
- `strategy`: Strategy type and parameters
- `universe`: Asset universe definition
- `backtest`: Initial capital, dates, transaction costs
- `experiment`: WandB logging settings

## Strategy Development

### Creating New Strategies

1. Inherit from `BaseStrategy` in `src/trading_system/strategies/base_strategy.py`
2. Implement required methods:
   - `generate_signals()`: Return DataFrame with signals (symbols in columns, dates in index)
   - `calculate_risk_metrics()`: Optional risk calculations
3. Add strategy configuration to YAML file
4. Register strategy in `StrategyRunner.initialize()` method

### Signal Format

Strategies should return signals as DataFrame with:
- Index: datetime dates
- Columns: stock symbols
- Values: signal weights (positive for long, negative for short, 0 for no position)

## Data Providers

### YFinance Provider
- Retry logic with configurable attempts and delays
- Symbol validation before data fetch
- Automatic data cleaning and gap handling
- Support for different data frequencies

### Fama-French Provider
- Access to Fama-French 5-factor model data
- Integration with ML strategies for factor modeling
- Automatic data alignment and preprocessing

## Backtesting System

The system includes a unified backtesting engine with:

- **Realistic Cost Modeling**: Commission, spread, slippage, and short borrow costs
- **Portfolio Tracking**: Position-level tracking with average cost pricing
- **Risk Management**: Stop-loss, drawdown limits, position size limits
- **Performance Metrics**: Comprehensive set of risk-adjusted performance measures

### Migration Notes

The codebase has migrated from a complex multi-file backtesting architecture to a unified engine (`BacktestEngine`) while maintaining backward compatibility with existing strategies.

## Experiment Tracking

Integration with Weights & Biases for:
- Automatic experiment logging
- Performance visualization
- Hyperparameter tracking
- Team collaboration

Environment variables required:
```bash
WANDB_API_KEY=your_wandb_key
ALPHA_VANTAGE_API_KEY=your_av_key  # Backup data source
```

## Machine Learning Components

### Feature Engineering
- Technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Price-based features (returns, volatility, momentum)
- Pattern recognition features
- Feature validation and selection

### ML Strategies
- XGBoost and LightGBM models
- Residual prediction using Fama-French factors
- Cross-validation with time series splits
- Model monitoring and degradation detection

## System Orchestration

For complex multi-strategy systems:
- **Core+Satellite Architecture**: 70-80% core strategy, 20-30% satellite
- **IPS Compliance**: Investment policy statement monitoring and reporting
- **Risk Management**: Integrated risk controls across all strategies
- **Performance Attribution**: Detailed attribution analysis

## Important Implementation Details

### Time Handling
- All timestamps use Python `datetime` objects
- Data alignment handles different timezones and market holidays
- Lookback buffers automatically calculated based on strategy needs

### Error Handling
- Comprehensive logging throughout the system
- Graceful degradation for missing data
- Automatic retry with exponential backoff for API calls

### Performance Considerations
- Vectorized operations using pandas/numpy
- Efficient data structures for large datasets
- Configurable caching for repeated calculations

## Testing Strategy

- Unit tests for individual components
- Integration tests for end-to-end pipelines
- Validation tests for data quality and consistency
- Performance benchmarking against known results

## File Structure Notes

- Main entry point: `run_strategy.py`
- Test orchestrator: `test_pipeline.py`
- Results saved to `./results/` directory automatically
- Models and cached data in `./models/` directory
- Documentation in `./documentation/` directory