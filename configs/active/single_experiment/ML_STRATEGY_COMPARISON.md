# ML Strategy Configuration Comparison

## Overview
Two ML strategy configurations for controlled variable comparison:
- **Box-Based**: `ml_strategy_config_new.yaml`
- **Quantitative**: `ml_strategy_quantitative_config.yaml`

## Purpose
Compare portfolio construction methods (Box-Based vs Quantitative) while keeping all other variables constant.

## Key Differences

### Portfolio Construction Method
- **Box-Based**: Uses box classification to ensure systematic diversification across investment style boxes
- **Quantitative**: Uses traditional mean-variance optimization based on signals

### Identical Configurations (Control Variables)
✅ **Training Setup**: Same model, features, and hyperparameters  
✅ **Universe**: Same stocks (200 stocks from 12 boxes)  
✅ **Training Period**: 2022-01-01 to 2023-12-31  
✅ **Backtest Period**: 2024-07-01 to 2025-08-15  
✅ **Benchmark**: WLS index from CSV  
✅ **Risk Parameters**: Same risk_aversion (2.0), covariance method (ledoit_wolf)  
✅ **Constraints**: Same position limits, no short selling  
✅ **Transaction Costs**: Same commission and slippage rates  

## Configuration Files

### Box-Based Configuration
- File: `ml_strategy_config_new.yaml`
- Portfolio Method: `box_based`
- Selection: Box-based stock selection with mean-variance optimization within/globally
- Diversification: Systematic box coverage

### Quantitative Configuration
- File: `ml_strategy_quantitative_config.yaml`
- Portfolio Method: `quantitative`
- Selection: Signal-based selection with mean-variance optimization
- Diversification: Risk-based optimization

## Usage

### Run Box-Based Experiment
```bash
python -m src.use_case.single_experiment.run_experiment \
    --config configs/active/single_experiment/ml_strategy_config_new.yaml
```

### Run Quantitative Experiment
```bash
python -m src.use_case.single_experiment.run_experiment \
    --config configs/active/single_experiment/ml_strategy_quantitative_config.yaml
```

## Expected Results
Comparing these two configurations will reveal:
1. Performance difference between box-based and quantitative portfolio construction
2. Risk-return characteristics of each method
3. Diversification effectiveness
4. Transaction cost impact

## Notes
- Both configurations use the same XGBoost model with identical hyperparameters
- Both start from the same universe of 200 stocks
- The only difference is how stocks are selected and weighted
- This is a true controlled variable experiment
