"""
Alpha t-statistics computation utilities.

This module provides reusable functions for computing Fama-French alpha t-statistics,
extracted from examples/compute_alpha_tstats.py for use across the codebase.
"""

import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

logger = logging.getLogger(__name__)


def compute_alpha_tstat(returns: pd.Series, factors: pd.DataFrame, required_factors: List[str] = None) -> Dict[str, Any]:
    """
    Run Fama-French regression and return alpha and its t-stat.
    
    Args:
        returns: Stock excess returns (Series with date index)
        factors: Factor returns (DataFrame with columns: MKT, SMB, HML, [RMW, CMA])
        required_factors: List of factor columns to use. Defaults to FF5 factors.
    
    Returns:
        Dictionary with 'alpha', 't_stat', 'p_value', 'r_squared', 'n_obs'
    """
    # Default to FF5 factors if not specified
    if required_factors is None:
        required_factors = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
    # Ensure both have the same index (aligned dates)
    # Normalize index types to ensure proper intersection
    returns_index = returns.index
    factors_index = factors.index
    
    # Convert to DatetimeIndex if needed (work on copies)
    if not isinstance(returns_index, pd.DatetimeIndex):
        returns_index = pd.to_datetime(returns_index)
    else:
        returns_index = returns_index.copy()
    
    if not isinstance(factors_index, pd.DatetimeIndex):
        factors_index = pd.to_datetime(factors_index)
    else:
        factors_index = factors_index.copy()
    
    # Find common dates
    common_index = returns_index.intersection(factors_index)
    if len(common_index) < 30:
        logger.debug(f"Insufficient common dates: {len(common_index)} < 30. Returns index: {len(returns_index)}, Factors index: {len(factors_index)}")
        return {
            'alpha': 0.0,
            't_stat': 0.0,
            'p_value': 1.0,
            'r_squared': 0.0,
            'n_obs': len(common_index)
        }
    
    # Align data by common dates - ensure we use the actual index from the series/dataframe
    # Reindex both to common_index to ensure perfect alignment
    returns_aligned = returns.reindex(common_index)
    factors_aligned = factors.reindex(common_index)
    
    # Drop any rows that became NaN during reindex
    valid_mask = ~(returns_aligned.isna() | factors_aligned[required_factors].isna().any(axis=1))
    returns_aligned = returns_aligned[valid_mask]
    factors_aligned = factors_aligned[valid_mask]
    
    if len(returns_aligned) < 30:
        logger.debug(f"After dropping NaN: insufficient data: {len(returns_aligned)} < 30")
        return {
            'alpha': 0.0,
            't_stat': 0.0,
            'p_value': 1.0,
            'r_squared': 0.0,
            'n_obs': len(returns_aligned)
        }
    if not all(col in factors_aligned.columns for col in required_factors):
        logger.debug(f"Missing required factors. Available: {list(factors_aligned.columns)}")
        return {
            'alpha': 0.0,
            't_stat': 0.0,
            'p_value': 1.0,
            'r_squared': 0.0,
            'n_obs': 0
        }
    
    # Convert to numpy arrays for regression
    y = returns_aligned.values
    X = factors_aligned[required_factors].values
    
    # Final check: ensure arrays are not empty
    if len(y) == 0 or X.shape[0] == 0:
        logger.debug(f"Empty arrays after alignment: y.shape={y.shape}, X.shape={X.shape}")
        return {
            'alpha': 0.0,
            't_stat': 0.0,
            'p_value': 1.0,
            'r_squared': 0.0,
            'n_obs': 0
        }
    X_with_const = add_constant(X)
    
    try:
        # Check for NaN or Inf values
        y_nan_count = np.isnan(y).sum() if len(y) > 0 else 0
        y_inf_count = np.isinf(y).sum() if len(y) > 0 else 0
        X_nan_count = np.isnan(X).sum() if X.size > 0 else 0
        X_inf_count = np.isinf(X).sum() if X.size > 0 else 0
        
        if y_nan_count > 0 or y_inf_count > 0:
            logger.debug(f"Returns contain NaN/Inf: NaN={y_nan_count}, Inf={y_inf_count}, skipping regression")
            return {
                'alpha': 0.0,
                't_stat': 0.0,
                'p_value': 1.0,
                'r_squared': 0.0,
                'n_obs': len(common_index)
            }
        if X_nan_count > 0 or X_inf_count > 0:
            logger.debug(f"Factors contain NaN/Inf: NaN={X_nan_count}, Inf={X_inf_count}, skipping regression")
            return {
                'alpha': 0.0,
                't_stat': 0.0,
                'p_value': 1.0,
                'r_squared': 0.0,
                'n_obs': len(common_index)
            }
        
        # Check if data is constant (variance = 0)
        if np.var(y) == 0:
            logger.debug(f"Returns have zero variance, skipping regression")
            return {
                'alpha': 0.0,
                't_stat': 0.0,
                'p_value': 1.0,
                'r_squared': 0.0,
                'n_obs': len(common_index)
            }
        
        model = OLS(y, X_with_const).fit()
        
        # Extract results safely
        # model.params is a pandas Series with index, but we need to access it correctly
        # The first parameter (index 0) is the constant (intercept/alpha)
        try:
            # Try accessing by name first (if it's a Series with named index)
            if hasattr(model.params, 'index') and 'const' in model.params.index:
                alpha = float(model.params['const'])
                t_stat = float(model.tvalues['const'])
                p_value = float(model.pvalues['const'])
                std_err = float(model.bse['const'])
            else:
                # Access by position (first element is the constant)
                alpha = float(model.params.iloc[0] if hasattr(model.params, 'iloc') else model.params[0])
                t_stat = float(model.tvalues.iloc[0] if hasattr(model.tvalues, 'iloc') else model.tvalues[0])
                p_value = float(model.pvalues.iloc[0] if hasattr(model.pvalues, 'iloc') else model.pvalues[0])
                std_err = float(model.bse.iloc[0] if hasattr(model.bse, 'iloc') else model.bse[0])
        except (IndexError, KeyError, AttributeError) as e:
            logger.warning(f"Failed to extract model parameters: {e}")
            logger.warning(f"Model params type: {type(model.params)}, shape: {model.params.shape if hasattr(model.params, 'shape') else 'N/A'}")
            return {
                'alpha': 0.0,
                't_stat': 0.0,
                'p_value': 1.0,
                'r_squared': 0.0,
                'n_obs': len(common_index)
            }
        
        r_squared = float(model.rsquared) if hasattr(model, 'rsquared') else 0.0
        n_obs = int(model.nobs) if hasattr(model, 'nobs') else len(common_index)
        
        # Debug: log if t_stat is suspiciously zero
        if abs(t_stat) < 1e-10:
            logger.debug(f"t_stat is zero: alpha={alpha:.8f}, std_err={std_err:.8f}, t_stat={t_stat:.8f}, "
                        f"y_mean={np.mean(y):.8f}, y_std={np.std(y):.8f}")
        
        return {
            'alpha': alpha,
            't_stat': t_stat,
            'p_value': p_value,
            'r_squared': r_squared,
            'n_obs': n_obs
        }
    except Exception as e:
        logger.warning(f"Regression failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return {
            'alpha': 0.0,
            't_stat': 0.0,
            'p_value': 1.0,
            'r_squared': 0.0,
            'n_obs': len(common_index)
        }

