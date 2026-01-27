# Testing the New Prediction Architecture

## Overview
This document provides step-by-step instructions for testing the refactored prediction architecture that fixes the factor data flow issue.

---

## Quick Test - Run E2E Experiment

The simplest way to test the entire flow is to run the existing experiment:

```bash
cd /Users/wenjiaqi/Downloads/bloomberg-competition
python run_experiment.py
```

This will:
1. ✅ Train an FF5 model with factor data
2. ✅ Use the fitted feature pipeline for backtesting
3. ✅ Ensure factor data flows correctly during predictions
4. ✅ Generate a complete experiment report

**Expected Output**:
- Training completes successfully
- Model saved to `./models/ff5_regression_YYYYMMDD_HHMMSS_v1.0.0/`
- Backtest runs without errors
- No warnings about missing factor data
- Performance metrics displayed

---

## Detailed Testing

### Test 1: Feature Pipeline Consistency

**What to verify**: The same feature pipeline is used in training and prediction.

```python
# Check the logs for these messages:
# Training phase:
"Fitting FeatureEngineeringPipeline..."
"FeatureEngineeringPipeline fitting complete."

# Backtest phase:
"Using fitted feature pipeline from training for backtest"
"Using provided fitted FeatureEngineeringPipeline from training"
```

**Success criteria**:
- ✅ Pipeline is fitted during training
- ✅ Same pipeline instance used in backtest
- ✅ No "pipeline is not fitted" warnings

### Test 2: Factor Data Flow

**What to verify**: Factor data is available when needed.

```python
# Check logs for:
"Creating factor data provider..."
"Fetching factor data for feature computation"
"Added factor data: shape=(X, 5)"  # Should show MKT, SMB, HML, RMW, CMA
"Merging factor data with shape..."
```

**Success criteria**:
- ✅ Factor provider created
- ✅ Factor data fetched during feature computation
- ✅ Features include factor columns
- ✅ No "Missing FF5 factors" warnings

### Test 3: Data Provider Injection

**What to verify**: Data providers flow from Orchestrator → Runner → Factory → Strategy.

```python
# Check logs for provider chain:
"Creating YFinanceProvider with params..."
"Creating FF5DataProvider with params..."
"✓ Created fama_french_5 strategy 'FF5_Strategy' with data providers"
```

**Success criteria**:
- ✅ Providers created in Orchestrator
- ✅ Providers passed to StrategyRunner
- ✅ Providers passed to Strategy
- ✅ Strategy can fetch factor data automatically

### Test 4: Model Prediction

**What to verify**: ModelPredictor receives pre-computed features.

```python
# Check logs for:
"Features shape=(1, 5), columns=['MKT', 'SMB', 'HML', 'RMW', 'CMA']"
"Current model type: ff5_regression"
# Should NOT see:
"No data provider available"  # ModelPredictor shouldn't try to fetch data
```

**Success criteria**:
- ✅ Features pre-computed before prediction
- ✅ ModelPredictor only does inference
- ✅ No data fetching in ModelPredictor

---

## Debug Mode

For more detailed debugging, enable DEBUG logging:

```python
# In run_experiment.py, change:
logging.basicConfig(level=logging.DEBUG)  # Was INFO
```

This will show:
- Feature computation details
- Factor data merging process
- Exact feature shapes and columns
- Prediction inputs and outputs

---

## Common Issues and Solutions

### Issue 1: "No factor data provider configured"

**Symptom**: 
```
WARNING - No factor data provider configured
WARNING - Using zero FF5 factors for AAPL
```

**Cause**: Factor data provider not passed to Strategy

**Fix**: Ensure `ExperimentOrchestrator` includes factor_data_provider in providers dict (Line 158-159)

### Issue 2: "Provided feature_pipeline is not fitted"

**Symptom**:
```
WARNING - Provided feature_pipeline is not fitted!
```

**Cause**: Pipeline wasn't fitted during training

**Fix**: Check that `TrainingPipeline.run_pipeline()` calls `feature_pipeline.fit()`

### Issue 3: "Features must be provided"

**Symptom**:
```
PredictionError: Features must be provided
```

**Cause**: Trying to call `ModelPredictor.predict()` without features

**Fix**: Ensure features are computed before prediction in `BaseStrategy._get_predictions()`

### Issue 4: "Failed to prepare features: KeyError"

**Symptom**:
```
ERROR - Failed to prepare features for AAPL: KeyError: 'MKT'
```

**Cause**: Factor data not merged into features

**Fix**: Check that `BaseStrategy._compute_features()` fetches and merges factor data (Line 219-254)

---

## Validation Checklist

Use this checklist to verify the system is working correctly:

### Architecture
- [ ] `PredictionPipeline` exists and handles data acquisition
- [ ] `ModelPredictor` simplified (no data providers)
- [ ] `BaseStrategy` has data provider parameters
- [ ] `StrategyFactory` injects providers into Strategy
- [ ] `ExperimentOrchestrator` passes fitted pipeline to backtest

### Training Phase
- [ ] Data providers created successfully
- [ ] Feature pipeline fitted on training data
- [ ] Factor data included in training features
- [ ] Model trained with correct feature shape
- [ ] Model saved with correct model_id

### Prediction Phase
- [ ] Fitted feature pipeline reused from training
- [ ] Data providers available in Strategy
- [ ] Factor data fetched during feature computation
- [ ] Features include all required factors
- [ ] ModelPredictor receives pre-computed features
- [ ] Predictions generated successfully

### End-to-End
- [ ] Training completes without errors
- [ ] Backtest runs without errors
- [ ] No factor data warnings
- [ ] Performance metrics calculated
- [ ] Results saved correctly

---

## Performance Benchmarks

Expected performance (on standard hardware):

| Phase | Expected Time | Memory Usage |
|-------|--------------|--------------|
| Training | 10-30 seconds | ~500 MB |
| Feature Computation | 2-5 seconds | ~200 MB |
| Backtesting | 5-15 seconds | ~300 MB |
| **Total E2E** | **20-60 seconds** | **~1 GB** |

If your times are significantly different, check:
- Data size (number of symbols × date range)
- Feature complexity
- Model complexity

---

## Next Steps

After successful testing:

1. **Run multiple experiments** with different configurations
2. **Test with different model types** (not just FF5)
3. **Test with different feature configurations**
4. **Monitor memory usage** for large-scale backtests
5. **Profile performance** to identify bottlenecks

---

## Reporting Issues

If you encounter problems:

1. **Capture full logs** (with DEBUG level)
2. **Note the exact error message**
3. **Include config file** used
4. **Provide stack trace** if available
5. **Check validation checklist** above

Then refer to the architecture documentation:
- `documentation/PREDICTION_ARCHITECTURE_REFACTORING.md`
- `documentation/ML_MODEL_ARCHITECTURE_REFACTOR.md`

---

## Success Criteria

The refactoring is successful when:

✅ **All tests pass** without warnings  
✅ **Factor data flows** correctly for FF5 models  
✅ **Features are consistent** between training and prediction  
✅ **Architecture follows** Single Responsibility Principle  
✅ **Code is maintainable** and well-documented  

Congratulations! The prediction architecture refactoring is complete. 🎉


