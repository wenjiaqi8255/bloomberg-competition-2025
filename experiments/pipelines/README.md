# Pipeline Entry Points

This directory contains main entry point scripts for running core trading system workflows.

## Available Pipelines

### 1. Feature Comparison Pipeline
**File**: `feature_comparison.py`

**Purpose**: Compare different feature engineering configurations to identify optimal feature sets

**Usage**:
```bash
python experiments/pipelines/feature_comparison.py --config configs/feature_config.yaml
```

**What it does**:
- Computes multiple feature sets
- Compares feature performance
- Identifies best features for model training
- Outputs feature comparison results

**Output**: `feature_comparison_results/`

---

### 2. FF5 Experiment Pipeline
**File**: `ff5_experiment.py`

**Purpose**: Train Fama-French 5-factor model with complete backtesting workflow

**Usage**:
```bash
python experiments/pipelines/ff5_experiment.py --config configs/ff5_box_based_experiment.yaml
```

**What it does**:
- Trains FF5 regression model
- Runs backtest on historical data
- Generates performance metrics
- Saves trained model to `models/`

**Performance**: +40.42% return, 1.17 Sharpe ratio

**Output**:
- Trained model: `models/ff5_regression_*/`
- Backtest results: `results/`

---

## Running Pipelines

### From Repository Root
All pipelines must be run from the repository root:

```bash
cd /path/to/bloomberg-competition
python experiments/pipelines/[pipeline_name].py --config configs/[config_file].yaml
```

### Prerequisites
1. **Install dependencies**: `poetry install`
2. **Data availability**: Ensure data exists in `data/`
3. **Configuration**: Review and modify config files if needed

### Common Options

Most pipelines support:
- `--config PATH`: Specify configuration file
- `--test-mode`: Run with reduced data for testing
- `--verbose`: Enable detailed logging
- `--dry-run`: Validate configuration without running

---

## Pipeline Workflow

### Typical Pipeline Execution

1. **Feature Engineering**:
   ```bash
   python experiments/pipelines/feature_comparison.py --config configs/feature_config.yaml
   ```

2. **Model Training**:
   ```bash
   python experiments/pipelines/ff5_experiment.py --config configs/ff5_box_based_experiment.yaml
   ```

3. **Validation**:
   ```bash
   python experiments/analysis/backtest_validation/check_backtest_issues.py
   ```

---

## Configuration

### Feature Config
Location: `configs/active/feature_config.yaml`

Key settings:
- Feature sets to compare
- Time period
- Cross-validation folds
- Performance metrics

### FF5 Config
Location: `configs/active/ff5_box_based_experiment.yaml`

Key settings:
- Model hyperparameters
- Training period
- Backtesting period
- Portfolio construction method

---

## Outputs

### Feature Comparison
- **Directory**: `feature_comparison_results/`
- **Files**:
  - Feature performance metrics
  - Comparison tables
  - Visualizations

### FF5 Experiment
- **Models**: `models/ff5_regression_YYYYMMDD_HHMMSS/`
- **Results**: `results/`
- **Logs**: Console output + optional log files

---

## Troubleshooting

### Import Errors
**Problem**: `ModuleNotFoundError: No module named 'trading_system'`

**Solution**: Ensure you're running from repository root
```bash
pwd  # Should show /path/to/bloomberg-competition
```

### Data Not Found
**Problem**: `FileNotFoundError: data/prices.csv not found`

**Solution**: Download or generate required data
```bash
ls -la data/
```

### Configuration Issues
**Problem**: `ValidationError: Invalid configuration`

**Solution**: Validate YAML syntax
```bash
python -c "import yaml; yaml.safe_load(open('configs/your_config.yaml'))"
```

---

## Related Scripts

- **Experiments**: See `experiments/use_cases/` for detailed experiment scripts
- **Examples**: See `examples/` for demonstration scripts
- **Analysis**: See `experiments/analysis/` for validation scripts
