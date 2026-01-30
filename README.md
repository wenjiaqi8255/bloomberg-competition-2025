# Bloomberg Trading Competition Submission

**Competition Year**: 2025
**Status**: Complete - Ready for Review
**Repository Size**: ~150MB (cleaned for submission)

---

## Overview

This is a quantitative trading system implementing factor-based and machine learning strategies for the Bloomberg trading competition. Our primary strategy (FF5 with alpha filtering) achieved **+40.42% returns** with a **1.17 Sharpe ratio**.

### Key Results

| Strategy | Return | Sharpe Ratio | Status |
|----------|--------|--------------|--------|
| **FF5 Factor + Alpha Filtering** | **+40.42%** | **1.17** | ✅ Primary Strategy |
| XGBoost ML | -39.61% | Negative | ❌ Failed (momentum crash) |
| Pure Factor Baseline | Negative | < 0.62 | ❌ Underperformed |

### Main Innovation: Alpha T-statistic Filtering

Our core innovation is filtering Fama-French 5-factor signals by alpha t-statistics, which:
- Improved Sharpe ratio by **89%** (0.62 → 1.17)
- Reduced maximum drawdown by **8.7%**
- Focused on true stock-specific alpha (122% of signal strength)

**Key Finding**: Signal decomposition shows alpha contributes 122% while pure factor exposure (β×λ) contributes -22%. Standard multi-factor models would underperform.

---

## Quick Start

### Prerequisites
- Python 3.11+
- Dependencies: See `pyproject.toml`
- API Keys: See `.env.example` (Alpha Vantage for data, WandB for logging)

### Installation

```bash
# Install dependencies
pip install -e .

# or using poetry
poetry install

# Set up environment
cp .env.example .env
# Edit .env with your API keys
```

### Repository Structure

```
bloomberg-competition/
├── src/trading_system/         # Core trading system (15K+ LOC)
│   ├── feature_engineering/    # 70+ technical indicators
│   ├── models/                 # FF5, XGBoost, LSTM implementations
│   ├── backtesting/           # Event-driven backtest engine
│   ├── portfolio/             # Portfolio construction & optimization
│   ├── signals/               # Signal processing & alpha filtering
│   ├── data/                  # Data providers (Alpha Vantage, local)
│   └── utils/                 # Utilities, logging, validation
├── configs/                   # YAML configuration system
│   ├── active/               # Active configurations
│   ├── templates/            # Strategy templates
│   └── schemas/              # JSON schemas for validation
├── docs/                     # Technical documentation
├── 过程doc/                  # Chinese documentation (process docs)
├── tests/                    # Comprehensive test suite
├── models/                   # Trained model artifacts (39MB)
├── DEFENSE_PACKAGE.md        # Competition defense materials
└── README.md                 # This file
```

---

## Available Pipelines

All pipelines are runnable from the repository root directory.

### 1. Feature Engineering Pipeline

**Purpose**: Compute technical indicators and features for ML models

**Entry Point**: `run_feature_comparison.py`

**Usage**:
```bash
python run_feature_comparison.py --config configs/active/feature_config.yaml
```

**Output**: Feature comparison results in `feature_comparison_results/`

**Features**:
- 70+ technical indicators
- Momentum, volatility, value, quality factors
- Box classifications (style, size, sector)
- Performance comparison between feature sets

---

### 2. FF5 Strategy Training & Backtesting ⭐ PRIMARY STRATEGY

**Purpose**: Train Fama-French 5-factor model with alpha filtering and run backtest

**Entry Point**: `run_ff5_box_experiment.py`

**Usage**:
```bash
python run_ff5_box_experiment.py --config configs/ff5_box_based_experiment.yaml
```

**Output**:
- Trained model in `models/ff5_regression_YYYYMMDD_HHMMSS/`
- Backtest results in `results/`
- Training logs in WandB (if enabled)

**Performance**: +40.42% return, 1.17 Sharpe ratio

**Key Features**:
- Alpha t-statistic filtering (main innovation)
- Dynamic position sizing based on signal confidence
- Fama-French 5-factor model with box classifications
- Robust risk management

---

### 3. ML Strategy Training (XGBoost)

**Purpose**: Train XGBoost model for stock prediction

**Entry Point**: `src/use_case/single_experiment/run_experiment.py`

**Usage**:
```bash
python src/use_case/single_experiment/run_experiment.py \
    --config configs/ml_strategy_config_new.yaml
```

**Output**:
- Trained model in `models/xgboost_YYYYMMDD_HHMMSS/`
- Training logs in WandB
- Feature importance analysis

**Performance**: -39.61% return (failed due to momentum crash)

**Lessons Learned**:
- 45.8% feature importance on momentum signals
- Crashed during momentum reversal period
- See DEFENSE_PACKAGE.md for full analysis

---

### 4. Multi-Model Ensemble

**Purpose**: Combine multiple models for improved predictions

**Entry Point**: `src/use_case/multi_model_experiment/run_multi_model_experiment.py`

**Usage**:
```bash
python src/use_case/multi_model_experiment/run_multi_model_experiment.py \
    --config configs/multi_model_config.yaml
```

**Output**: Ensemble predictions and performance metrics

---

### 5. Inference/Prediction Pipeline

**Purpose**: Generate predictions using trained models

**Entry Point**: `src/use_case/prediction/run_prediction.py`

**Usage**:
```bash
python src/use_case/prediction/run_prediction.py \
    --model-path models/ff5_regression_20251027_011643 \
    --input-data data/latest.csv \
    --output predictions.csv
```

**Output**: Stock predictions in CSV format

---

## Model Artifacts

All trained models are preserved in the `models/` directory (39MB total):

### Available Models

1. **`fama_macbeth_20251014_183441/`**
   - Fama-MacBeth regression baseline
   - Pure factor model (no alpha filtering)

2. **`ff5_regression_20251013_170247/`**
   - FF5 factor model (initial version)
   - Without alpha filtering

3. **`ff5_regression_20251027_011643/`** ⭐ **PRIMARY MODEL**
   - FF5 factor model with alpha filtering
   - 40.42% return, 1.17 Sharpe ratio
   - Use this for inference and reproduction

4. **`xgboost_20251025_181301/`**
   - XGBoost ML model
   - Failed due to momentum crash
   - Preserved for analysis

### Regenerating Models

To regenerate these models, run the corresponding training pipelines (see "Available Pipelines" section above).

See `models/README.md` for detailed regeneration instructions.

---

## Documentation

### Core Documentation

- **Methodology**: `docs/methodology/`
  - Fama-French 5-factor model implementation
  - Fama-MacBeth regression methodology
  - Technical analysis fundamentals

- **Implementation**: `docs/implementation/`
  - Portfolio optimization guide
  - Backtesting engine documentation
  - Signal processing pipeline

- **API Reference**: `docs/api/`
  - Module-level documentation
  - Function signatures
  - Usage examples

### Competition Materials

- **Defense Package**: `DEFENSE_PACKAGE.md` (comprehensive defense guide)
  - Executive summary
  - Performance analysis
  - Expected Q&A
  - Research findings
  - Presentation talking points

- **Quick Reference**: `QUICK_REFERENCE.md`
  - Key metrics and numbers
  - Feature importance breakdown
  - Momentum crash analysis

- **Presentation Data**: `DEFENSE_PRESENTATION_DATA.md`
  - Extracted data for slides
  - Performance comparisons
  - Statistical significance tests

### Development Journey

- **Chinese Process Docs**: `过程doc/` (preserved per user decision)
  - Development notes in Chinese
  - Debugging sessions
  - Research iterations

---

## Configuration

All pipelines use YAML configuration files in `configs/active/`:

### Key Configuration Files

- **`feature_config.yaml`** - Feature engineering settings
  - Indicator parameters
  - Data sources
  - Output paths

- **`ff5_box_based_experiment.yaml`** - FF5 strategy configuration
  - Model hyperparameters
  - Alpha filtering thresholds
  - Portfolio constraints
  - Backtesting parameters

- **`ml_strategy_config_new.yaml`** - ML strategy configuration
  - XGBoost hyperparameters
  - Feature selection
  - Training/validation split

- **`multi_model_config.yaml`** - Ensemble configuration
  - Model combinations
  - Weighting schemes
  - Ensemble strategy

### Configuration Schema

All configurations are validated against JSON schemas in `configs/schemas/` to ensure correctness.

---

## Testing

### Test Structure

```
tests/
├── unit/              # Unit tests (individual components)
├── integration/       # Integration tests (component interaction)
└── performance/       # Performance tests (speed, memory)
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest --cov=src/trading_system tests/

# Run specific test file
pytest tests/unit/test_feature_engineering.py
```

### Test Coverage

- Feature engineering: 80%+ coverage
- Backtesting engine: 75%+ coverage
- Portfolio construction: 70%+ coverage
- Data providers: 85%+ coverage

---

## Performance Summary

### FF5 Strategy (Primary)

| Metric | Value | Improvement |
|--------|-------|-------------|
| Total Return | **+40.42%** | - |
| Annualized Return | 74.90% | - |
| Sharpe Ratio | **1.17** | +89% vs baseline |
| Maximum Drawdown | -66.88% | 8.7% better |
| Sortino Ratio | 1.26 | +76% vs baseline |
| Information Ratio | 1.00 | - |
| Beta (market) | 0.73 | Lower exposure |
| Alpha | 1.14 | Significant excess returns |
| Win Rate | 48.37% | - |
| Avg Holdings | 13 stocks | Diversified |

### XGBoost Strategy (Failed)

| Metric | Value | Issue |
|--------|-------|-------|
| Total Return | -39.61% | Momentum crash |
| Feature Importance (Momentum) | 45.8% | Over-reliance |
| Crash Period | Q4 2024 | Regime change |

**Lesson**: ML models require regime-switching capabilities to handle momentum crashes.

---

## Key Research Findings

### 1. Alpha Dominance

Signal decomposition from trained model:
```
Total Signal = Alpha (122%) + Pure Factor (-22%)
```

**Implication**: Standard multi-factor models (pure factor exposure) would underperform. Alpha filtering is essential, not optional.

### 2. Momentum Crash Risk

XGBoost failure analysis:
- 15.7% explicit momentum features
- 30.1% trend-following technical indicators
- **45.8% total momentum-dependent signals**

**Academic Support**: Daniel & Moskowitz (2016) documented momentum crashes causing 50-80% losses during market reversals.

**Implication**: ML models need regime detection or momentum diversification.

### 3. Filtering Effectiveness

Alpha t-statistic filtering:
- Sharpe ratio: 0.62 → 1.17 (+89%)
- Max drawdown: -73.27% → -66.88% (+8.7%)
- Sortino ratio: 0.72 → 1.26 (+76%)

**Implication**: Removing low-confidence signals significantly improves risk-adjusted returns.

---

## System Requirements

### Minimum Requirements
- Python 3.11+
- 8GB RAM
- 2GB free disk space (excluding models/)
- Internet connection (for Alpha Vantage API)

### Recommended Requirements
- Python 3.11+
- 16GB RAM
- 10GB free disk space (including models/)
- SSD (for faster data loading)
- Multi-core CPU (for parallel feature engineering)

### Dependencies

See `pyproject.toml` for complete dependency list. Key dependencies:
- `pandas`, `numpy` - Data manipulation
- `xgboost` - Gradient boosting
- `scikit-learn` - Machine learning utilities
- `statsmodels` - Statistical models
- `pydantic` - Configuration validation
- `alpha-vantage` - Data API
- `wandb` - Experiment tracking

---

## Troubleshooting

### Common Issues

**Issue**: `WANDB_API_KEY not found`
- **Solution**: Set up `.env` file with your API key (see `.env.example`)

**Issue**: `ModuleNotFoundError: No module named 'trading_system'`
- **Solution**: Run `pip install -e .` from repository root

**Issue**: Out of memory during feature engineering
- **Solution**: Reduce `batch_size` in config or use fewer symbols

**Issue**: Slow backtesting
- **Solution**: Enable caching in config or use smaller date range

### Getting Help

1. Check documentation in `docs/`
2. Review error messages and logs
3. Check configuration files match your environment
4. See `DEFENSE_PACKAGE.md` for methodology clarifications

---

## License

[Your license here - e.g., MIT, Apache 2.0, or "Competition Submission - Not for distribution"]

---

## Contact

**Team**: [Your Team Name]
**Email**: [Your contact email]
**Competition**: Bloomberg Trading Competition 2025

---

## Acknowledgments

- **Fama-French 5-Factor Model**: Fama & French (2015)
- **Momentum Crash Research**: Daniel & Moskowitz (2016)
- **Data Provider**: Alpha Vantage API
- **Experiment Tracking**: Weights & Biases (WandB)

---

**END OF README**
