# FF5 Training with FDR Correction - In Progress

## Configuration

**Training Config**: `configs/active/single_experiment/ff5_box_based_experiment.yaml`

### Academic Enhancements Enabled:

1. **FDR Correction**:
   - `fdr_level: 0.05` (5% False Discovery Rate)
   - `apply_fdr: true` (Benjamini-Hochberg correction enabled)

2. **Reproducibility**:
   - `random_seed: 42` (Fixed seed for reproducibility)

3. **Market Impact Modeling**:
   - `market_impact.enabled: true`
   - `gamma: 1e-6` (Permanent impact coefficient)
   - `eta: 1e-6` (Temporary impact coefficient)

## Training Status

Currently running...
- **Start Time**: 2026-01-30 19:53:55
- **Method**: FF5 Regression with Box-Based Portfolio Construction
- **Period**: 2022-01-01 to 2023-12-31 (training)
- **Symbols**: 360 stocks (from universe CSV)

## Expected FDR Output

When training completes, look for:

```
============================================================
Benjamini-Hochberg FDR Correction Results
============================================================
FDR Level (Q): 0.05
Total Features Tested: 70
Significant Features (after FDR): X
False Discovery Rate Controlled at: 5.0%

Significant Features after FDR Correction:
  [feature_1]: raw_p = 0.001, adj_p = 0.003
  [feature_2]: raw_p = 0.015, adj_p = 0.038
  ...
============================================================
```

## Model Output

Trained model will be saved to:
```
models/ff5_regression_YYYYMMDD_HHMMSS/
```

With FDR-corrected statistics in model metadata.

## Monitoring

```bash
# Check progress
tail -f /tmp/ff5_training_fdr.log

# Look for FDR output
grep -i "fdr\|benjamini\|significant" /tmp/ff5_training_fdr.log
```

---

**Next Steps After Training**:
1. Review FDR correction results
2. Compare with non-FDR model
3. Run White's Reality Check on backtest returns
4. Document impact on performance metrics
