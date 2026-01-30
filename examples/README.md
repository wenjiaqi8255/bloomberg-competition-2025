# Examples Directory

This directory contains demonstration scripts showing how to use various components of the trading system.

## Directory Structure

```
examples/
├── feature_discovery/   # Exploring available features
├── portfolio/          # Portfolio construction examples
├── prediction/         # Prediction and inference examples
├── analysis/           # Analysis and validation examples
└── configuration/      # Configuration setup examples
```

## Examples by Category

### Feature Discovery (`feature_discovery/`)

#### **discover_features.py**
Explore and analyze available features in the trading system.

**Purpose**: Discover what features are available, their types, and how they're computed

**Usage**:
```bash
python examples/feature_discovery/discover_features.py
```

**Output**: Lists all available features with descriptions and usage examples

---

### Portfolio (`portfolio/`)

#### **portfolio_construction_demo.py**
Demonstrates basic portfolio construction using different methodologies.

**Purpose**: Show how to construct portfolios using various strategies

**Usage**:
```bash
python examples/portfolio/portfolio_construction_demo.py
```

**Key Features**:
- Box-First portfolio construction
- Equal-weighted portfolios
- Factor-based portfolios

#### **optimal_system_demo.py**
Demonstrates the optimal trading system configuration.

**Purpose**: Show end-to-end system usage with optimal settings

**Usage**:
```bash
python examples/portfolio/optimal_system_demo.py
```

**Key Features**:
- Complete trading system setup
- Multi-stage portfolio construction
- Performance analysis

---

### Prediction (`prediction/`)

#### **prediction_demo_single.py**
Demonstrates single model prediction workflow.

**Purpose**: Show how to generate predictions using a single trained model

**Usage**:
```bash
python examples/prediction/prediction_demo_single.py --model models/ff5_regression_20251027_011643
```

**Key Features**:
- Load trained model
- Prepare input data
- Generate predictions
- Format output

#### **prediction_demo_meta.py**
Demonstrates meta-model (ensemble) prediction workflow.

**Purpose**: Show how to use ensemble models for improved predictions

**Usage**:
```bash
python examples/prediction/prediction_demo_meta.py --meta-model models/metamodel_20251026
```

**Key Features**:
- Combine multiple model predictions
- Meta-model inference
- Ensemble performance analysis

---

### Analysis (`analysis/`)

#### **compute_alpha_tstats.py**
Compute and analyze alpha t-statistics for factors.

**Purpose**: Demonstrate statistical significance testing of factor performance

**Usage**:
```bash
python examples/analysis/compute_alpha_tstats.py --factors ff5 --period 2020-2024
```

**Key Features**:
- Fama-MacBeth regression
- T-statistic calculation
- Statistical significance analysis

#### **integration_test_example.py**
Example of integration testing for trading system components.

**Purpose**: Show how to write and run integration tests

**Usage**:
```bash
python examples/analysis/integration_test_example.py
```

**Key Features**:
- Component integration testing
- End-to-end workflow validation
- Test result reporting

---

### Configuration (`configuration/`)

#### **sector_configuration_demo.py**
Demonstrates sector-based configuration and analysis.

**Purpose**: Show how to configure and use sector-based strategies

**Usage**:
```bash
python examples/configuration/sector_configuration_demo.py
```

**Key Features**:
- Sector classification setup
- Sector-specific parameters
- Sector performance analysis

---

## Running Examples

### Prerequisites
1. Install dependencies: `poetry install`
2. Train models or download pre-trained models
3. Ensure required data is available

### Quick Start

1. **Discover Features**:
   ```bash
   python examples/feature_discovery/discover_features.py
   ```

2. **Build Portfolio**:
   ```bash
   python examples/portfolio/portfolio_construction_demo.py
   ```

3. **Generate Predictions**:
   ```bash
   python examples/prediction/prediction_demo_single.py --model models/ff5_regression_20251027_011643
   ```

4. **Analyze Performance**:
   ```bash
   python examples/analysis/compute_alpha_tstats.py
   ```

## Common Patterns

### Running from Repository Root
All examples should be run from the repository root directory:
```bash
cd /path/to/bloomberg-competition
python examples/category/example_script.py [options]
```

### Command-Line Arguments
Most examples support command-line arguments:
```bash
python examples/portfolio/portfolio_construction_demo.py --help
```

### Configuration Files
Examples use configuration files from `configs/`:
```bash
python examples/prediction/prediction_demo_single.py \
    --config configs/prediction_config.yaml \
    --model models/ff5_regression_20251027_011643
```

## Modifying Examples

### Customizing Parameters
Edit the example script or use command-line arguments to customize:
```python
# In script
config = {
    'start_date': '2020-01-01',
    'end_date': '2024-12-31',
    'rebalance_frequency': 'monthly'
}
```

### Extending Examples
Use examples as templates for your own experiments:
1. Copy the example to your workspace
2. Modify parameters and logic
3. Test thoroughly before running on full data

## Data Requirements

### Input Data
Examples typically require:
- Price data in `data/prices/`
- Factor data in `data/factors/`
- Reference data in `data/reference/`

### Model Files
Prediction examples require trained models in `models/`:
- FF5 models: `models/ff5_regression_*`
- XGBoost models: `models/xgboost_*`
- Meta-models: `models/metamodel_*`

### Configuration
All examples read configuration from `configs/`:
- Feature configs: `configs/active/feature_config.yaml`
- Model configs: `configs/active/ml_strategy_config_new.yaml`
- Portfolio configs: `configs/templates/portfolio_*.yaml`

## Output Locations

### Results
Example outputs are typically saved to:
- Predictions: `prediction_results/`
- Portfolios: `portfolios/`
- Analysis: `analysis_results/`

### Logs
Check logs for detailed execution information:
- Console output: Real-time progress
- Log files: `logs/` directory (if configured)

## Troubleshooting

### Import Errors
Ensure you're running from repository root:
```bash
pwd  # Should show /path/to/bloomberg-competition
python examples/category/example.py
```

### Missing Data
Verify data files exist:
```bash
ls -la data/
ls -la data/prices/
ls -la data/factors/
```

### Model Not Found
Check model directory:
```bash
ls -la models/
```

Train a model if needed:
```bash
python experiments/pipelines/ff5_experiment.py --config configs/ff5_box_based_experiment.yaml
```

### Configuration Issues
Validate YAML syntax:
```bash
python -c "import yaml; yaml.safe_load(open('configs/your_config.yaml'))"
```

## Next Steps

1. **Run All Examples**: Try each example to understand the system
2. **Modify Parameters**: Experiment with different configurations
3. **Build Custom Workflows**: Combine examples into custom pipelines
4. **Check Documentation**: See `docs/` for detailed methodology

## Related Documentation

- **Main README**: Project overview and quick start
- **Experiments**: See `experiments/` for full experiment scripts
- **Documentation**: See `docs/` for detailed guides
- **Configuration**: See `configs/` for available templates
