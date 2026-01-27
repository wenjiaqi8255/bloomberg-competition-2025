# Alpha t-stat Significance Filtering (Phase 1 + Phase 2 Ready)

## Approach

Two-phase implementation: Phase 1 (quick validation) uses hard threshold with robust error handling; Phase 2 adds shrinkage methods and rolling window support. All controlled via config, defaulting to no-op for backward compatibility.

## Phase 1: Quick Validation (Immediate)

- Apply optional significance filter in `fama_french_5._get_predictions` after fetching alphas
- Support hard_threshold method (simple, immediate validation)
- Robust exception handling for missing CSV, NaN values, symbol mismatches
- Log metrics (before/after alpha distribution, filtered counts)
- External script `compute_alpha_tstats.py` generates static CSV (one-time per universe)

## Phase 2: Enhanced (Future Iteration)

- Add shrinkage methods: `linear_shrinkage` (recommended), `sigmoid_shrinkage`, keep `hard_threshold`
- Support rolling window t-stats (time-series CSV with date column)
- Enhanced logging for debugging and backtesting comparison

## Files to Change

### 1. `src/trading_system/strategies/fama_french_5.py`

- Add `_apply_alpha_significance_filter()` method:
  - Reads config from `self.config.get('alpha_significance', {})` or `kwargs`
  - Validates CSV format (symbol, t_alpha columns required)
  - Handles missing symbols, NaN t-stats gracefully
  - Applies filter/shrinkage based on method
  - Logs detailed metrics (mean, std, non-zero counts before/after)
- Modify `_get_predictions()`:
  - After `alphas = current_model.get_symbol_alphas()` (line ~270)
  - Call `alphas = self._apply_alpha_significance_filter(alphas, config)`
  - Add `_shrinkage_factor()` helper for Phase 2 methods

### 2. `examples/compute_alpha_tstats.py` (New)

- Standalone script with argparse
- Supports `--mode static` (default) and `--mode rolling` (Phase 2)
- Uses `statsmodels.OLS` for t-stat computation
- Outputs CSV with columns: `symbol, t_alpha, p_value, r_squared` (static) or `date, symbol, t_alpha` (rolling)
- Validates lookback window consistency with training config
- Includes usage instructions and validation checks

### 3. `configs/active/single_experiment/ff5_box_based_experiment.yaml`

- Add under `strategy.parameters`:
```yaml
  alpha_significance:
    enabled: true
    t_threshold: 2.0
    method: "hard_threshold"  # Phase 1: simple. Phase 2: "linear_shrinkage" or "sigmoid_shrinkage"
    tstats_path: "./alpha_tstats.csv"
```


## Implementation Details

### Exception Handling (Robust)

- Missing CSV file → log warning, skip filter (no-op)
- Invalid CSV format → log error, skip filter
- Missing symbols in CSV → log debug, keep original alpha
- NaN t-stats → log debug, set alpha=0 (conservative)
- Symbol format mismatch → attempt normalization, fallback to keep original

### Logging Metrics

```python
logger.info(
    f"Alpha significance filter: "
    f"method={method}, threshold={threshold}, "
    f"zeroed/shrunk={n_filtered}/{n_total}, "
    f"missing_in_csv={n_missing}"
)
logger.info(
    f"Alpha distribution: "
    f"mean={mean_before:.4f}→{mean_after:.4f}, "
    f"std={std_before:.4f}→{std_after:.4f}, "
    f"non-zero={nz_before}→{nz_after}"
)
```

### Shrinkage Functions (Phase 2)

- `hard_threshold`: `return 1.0 if abs_t >= threshold else 0.0`
- `linear_shrinkage`: `return min(1.0, abs_t / threshold)` (recommended)
- `sigmoid_shrinkage`: `return 1 / (1 + exp(-2*(abs_t - threshold)))`

## Why This Meets Principles

- **KISS**: Phase 1 minimal (hard threshold + CSV), Phase 2 adds complexity only when validated
- **SOLID**: Single responsibility (filter logic isolated), Open/Closed (extensible via config methods)
- **YAGNI**: No in-model changes, no training refactor, optional feature
- **DRY**: Reusable helper, no duplication, config-driven behavior
- **Robust**: Comprehensive error handling, graceful degradation, detailed logging