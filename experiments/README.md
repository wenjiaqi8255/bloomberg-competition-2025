# Experiments Directory

This directory contains all analysis, comparison, experiment, and validation scripts organized by purpose.

## Directory Structure

```
experiments/
├── pipelines/           # Main entry point pipelines for core workflows
├── analysis/           # Analysis and validation scripts
├── momentum_analysis/  # Momentum-specific analysis and visualization
├── presentation/       # Presentation generation scripts
└── use_cases/          # Experiment use case runners
```

## Quick Navigation

### Pipelines (`pipelines/`)
Main entry point scripts for running core trading system workflows:
- **feature_comparison.py**: Compare different feature engineering configurations
- **ff5_experiment.py**: Run FF5 factor model training and backtesting

**Usage**: Run from repository root
```bash
python experiments/pipelines/feature_comparison.py --config configs/feature_config.yaml
python experiments/pipelines/ff5_experiment.py --config configs/ff5_box_based_experiment.yaml
```

### Analysis (`analysis/`)
Validation and analysis scripts organized by category:

#### Residual Momentum (`residual_momentum/`)
- **validate_residual_momentum.py**: Validate residual momentum strategy implementation

#### Pure Factor Baseline (`pure_factor_baseline/`)
- **pure_factor_quick_est.py**: Quick estimation of pure factor performance

#### Signal Analysis (`signal_analysis/`)
- **debug_signal_strength.py**: Debug signal generation and strength issues
- **diagnose_beta_anomaly.py**: Diagnose beta calculation anomalies
- **t2_alpha_vs_expected_return_analysis.md**: Analysis of alpha vs expected returns

#### Backtest Validation (`backtest_validation/`)
- **check_backtest_issues.py**: Identify and diagnose backtesting problems
- **check_extreme_return_days.py**: Analyze extreme return days in backtest
- **detailed_analysis.py**: Perform detailed backtest analysis

**Usage**: Most scripts can be run from repository root
```bash
python experiments/analysis/residual_momentum/validate_residual_momentum.py
```

### Momentum Analysis (`momentum_analysis/`)
Scripts for analyzing and visualizing momentum factor performance:
- **extract_momentum_importance.py**: Extract momentum importance from trained models
- **create_momentum_importance_chart.py**: Generate visualization of momentum importance

**Usage**:
```bash
python experiments/momentum_analysis/extract_momentum_importance.py --model-path models/xgboost_20251025_181301
python experiments/momentum_analysis/create_momentum_importance_chart.py
```

### Presentation (`presentation/`)
Scripts for generating competition presentation materials:
- **generate_presentation_text.py**: Generate presentation text from results

**Usage**:
```bash
python experiments/presentation/generate_presentation_text.py
```

### Use Cases (`use_cases/`)
Main experiment orchestration scripts:
- **run_single_experiment.py**: Run single model experiment with training, prediction, and backtesting
- **run_multi_model_experiment.py**: Run multi-model ensemble experiments
- **run_prediction.py**: Generate predictions using trained models
- **experiment_orchestrator.py**: Core orchestration logic

**Usage**:
```bash
# Single experiment
python experiments/use_cases/run_single_experiment.py --config configs/ml_strategy_config_new.yaml

# Multi-model experiment
python experiments/use_cases/run_multi_model_experiment.py --config configs/multi_model_config.yaml

# Prediction
python experiments/use_cases/run_prediction.py --model-path models/ff5_regression_20251027_011643 --input-data data/latest.csv --output predictions.csv
```

## Running Experiments

### Prerequisites
1. Install dependencies: `poetry install`
2. Ensure data is available in `data/` directory
3. Configure desired settings in `configs/` directory

### Common Workflow

1. **Feature Engineering**: Compare feature sets
   ```bash
   python experiments/pipelines/feature_comparison.py --config configs/feature_config.yaml
   ```

2. **Train FF5 Model**: Train and backtest FF5 strategy
   ```bash
   python experiments/pipelines/ff5_experiment.py --config configs/ff5_box_based_experiment.yaml
   ```

3. **Train ML Model**: Train XGBoost or other ML models
   ```bash
   python experiments/use_cases/run_single_experiment.py --config configs/ml_strategy_config_new.yaml
   ```

4. **Generate Predictions**: Use trained models for inference
   ```bash
   python experiments/use_cases/run_prediction.py --model-path models/ff5_regression_20251027_011643 --input-data data/latest.csv --output predictions.csv
   ```

5. **Analyze Results**: Run validation and analysis scripts
   ```bash
   python experiments/analysis/backtest_validation/check_backtest_issues.py --results results/
   ```

## Import Path Notes

All scripts in this directory are configured to run from the **repository root**. They automatically add `src/` to Python path using:

```python
repo_root = Path(__file__).parent.parent.parent  # For most scripts
sys.path.insert(0, str(repo_root / "src"))
```

This allows imports like:
```python
from trading_system.models.base.model_factory import ModelFactory
from use_case.single_experiment.experiment_orchestrator import ExperimentOrchestrator
```

## Experiment Outputs

- **Models**: Saved to `models/` directory
- **Results**: Saved to `results/` or experiment-specific directories
- **Logs**: Training logs in `wandb/` (if using Weights & Biases)
- **Cache**: Feature cache in `cache/` directory

## Related Documentation

- **Main README**: See repository root for project overview
- **Documentation**: See `docs/` for detailed methodology
- **Configuration**: See `configs/` for available configuration templates
- **Examples**: See `examples/` for demonstration scripts

## Tips for Running Experiments

1. **Start with pipelines/**: Use entry point scripts in `pipelines/` for complete workflows
2. **Check configs/**: Review and modify configuration files before running
3. **Monitor logs**: Most scripts output detailed logging information
4. **Validate results**: Use analysis scripts to check experiment outputs
5. **GPU acceleration**: Some ML experiments benefit from GPU support

## Troubleshooting

**Import Errors**: Ensure you're running from repository root
```bash
cd /path/to/bloomberg-competition
python experiments/pipelines/ff5_experiment.py --config configs/ff5_box_based_experiment.yaml
```

**Data Issues**: Verify data files exist in `data/` directory
```bash
ls -la data/
```

**Configuration Issues**: Validate YAML syntax
```bash
python -c "import yaml; yaml.safe_load(open('configs/your_config.yaml'))"
```
