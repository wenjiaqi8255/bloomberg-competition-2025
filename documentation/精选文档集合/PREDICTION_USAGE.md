# Prediction Configuration Usage Guide

## ML XGBoost Quantitative Prediction

### Configuration File
`prediction_ml_xgboost_quantitative.yaml`

### Model
- **Model ID**: `xgboost_20251110_010814`
- **Model Type**: XGBoost ML Model
- **Portfolio Construction**: Quantitative (mean-variance optimization)

### Usage

#### Run Prediction
```bash
# Basic usage
python -m src.use_case.prediction.run_prediction \
    --config configs/active/prediction/prediction_ml_xgboost_quantitative.yaml

# With custom output directory
python -m src.use_case.prediction.run_prediction \
    --config configs/active/prediction/prediction_ml_xgboost_quantitative.yaml \
    --output-dir ./my_prediction_results

# With verbose output
python -m src.use_case.prediction.run_prediction \
    --config configs/active/prediction/prediction_ml_xgboost_quantitative.yaml \
    --verbose

# Output in multiple formats
python -m src.use_case.prediction.run_prediction \
    --config configs/active/prediction/prediction_ml_xgboost_quantitative.yaml \
    --format all
```

### Configuration Details

#### Model Configuration
- Uses pre-trained model: `xgboost_20251110_010814`
- Feature pipeline: Loaded from model artifacts
- Lookback period: 252 days
- Signal normalization: minmax

#### Universe
- Source: CSV file (`./data/universes/complete_stock_data_converted.csv`)
- Filters: 200 stocks from 12 boxes (DM/EM, Large/Mid/Small, Growth/Value)
- Same universe as training for consistency

#### Portfolio Construction
- Method: Quantitative (mean-variance optimization)
- Risk aversion: 2.0
- Covariance method: Ledoit-Wolf
- Max position weight: 0.5
- Min position weight: 0.01
- No short selling

#### Prediction Date
- Default: `2025-11-09`
- **Important**: Update this to the latest available trading day before running prediction

### Output

Results will be saved to:
- `./prediction_results/ml_xgboost_quantitative/`
  - `prediction_result.json` - Full prediction results
  - `recommendations.csv` - Stock recommendations
  - `prediction_summary.txt` - Summary report

### Notes

1. **Prediction Date**: Make sure to update `prediction_date` to a recent trading day
2. **Model Files**: Ensure model files exist in `./models/xgboost_20251110_010814/`
3. **Data Availability**: Prediction requires historical price data for the lookback period
4. **Universe**: Uses same universe filters as training for consistency

### Troubleshooting

- **Model not found**: Check that model directory exists in `./models/`
- **No signals generated**: Check data availability for prediction date
- **Universe loading failed**: Verify CSV file exists and has required columns



