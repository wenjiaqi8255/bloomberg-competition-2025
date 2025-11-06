"""
Compute t-statistics for Fama-French (FF3/FF5) alpha estimates.

This script computes alpha t-statistics by:
1. Loading a trained FF3 or FF5 model (by model_id)
2. Extracting symbols from the model's alphas
3. Reusing the training pipeline's data loading logic
4. Computing t-stats using the same factors that were used for training

This ensures consistency with training and follows DRY principles.
The script automatically detects the model type (FF3 or FF5) and uses the
appropriate factors (3 for FF3, 5 for FF5).

Usage:
    # For FF5 model:
    python examples/compute_alpha_tstats.py \
        --model-id ff5_regression_v1 \
        --output alpha_tstats.csv \
        --lookback 252

    # For FF3 model:
    python examples/compute_alpha_tstats.py \
        --model-id ff3_regression_20251106_000146 \
        --output alpha_tstats_ff3.csv \
        --lookback 252

    # If you need to override training dates:
    python examples/compute_alpha_tstats.py \
        --model-id ff5_regression_v1 \
        --config configs/active/single_experiment/ff5_box_based_experiment.yaml \
        --output alpha_tstats.csv \
        --start-date 2022-01-01 \
        --end-date 2023-12-31
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.trading_system.models.serving.predictor import ModelPredictor
from src.trading_system.models.training.training_pipeline import TrainingPipeline
from src.trading_system.feature_engineering.pipeline import FeatureEngineeringPipeline
from src.use_case.single_experiment.experiment_orchestrator import ExperimentOrchestrator
from src.trading_system.models.model_persistence import ModelRegistry

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
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


def create_data_providers_from_config(config_path: str):
    """
    Create data providers from experiment config (reusing orchestrator logic).
    
    Returns:
        Tuple of (data_provider, factor_data_provider)
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_provider_config = config.get('data_provider', {})
    factor_provider_config = config.get('factor_data_provider', {})
    
    # Reuse orchestrator's helper functions
    from src.use_case.single_experiment.experiment_orchestrator import (
        _create_data_provider, _create_factor_data_provider
    )
    
    data_provider = _create_data_provider(data_provider_config) if data_provider_config else None
    factor_data_provider = _create_factor_data_provider(factor_provider_config) if factor_provider_config else None
    
    return data_provider, factor_data_provider


def main():
    parser = argparse.ArgumentParser(
        description='Compute t-statistics for Fama-French (FF3/FF5) alpha estimates using trained model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--model-id',
        type=str,
        required=True,
        help='Trained model ID (e.g., ff5_regression_v1)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to experiment config YAML (for data provider setup). If not provided, will try to infer from model metadata.'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='alpha_tstats.csv',
        help='Output CSV file path (default: alpha_tstats.csv)'
    )
    parser.add_argument(
        '--lookback',
        type=int,
        default=252,
        help='Lookback window in trading days (default: 252). Should match training lookback.'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Override: start date for data (YYYY-MM-DD). If not provided, uses training dates from model metadata.'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='Override: end date for data (YYYY-MM-DD). If not provided, uses training dates from model metadata.'
    )
    parser.add_argument(
        '--model-registry-path',
        type=str,
        default='./models/',
        help='Path to model registry (default: ./models/)'
    )
    
    args = parser.parse_args()
    
    # Step 1: Load trained model
    logger.info(f"Loading trained model: {args.model_id}")
    predictor = ModelPredictor(
        model_id=args.model_id,
        model_registry_path=args.model_registry_path
    )
    model = predictor.get_current_model()
    
    if not model or not hasattr(model, 'get_symbol_alphas'):
        logger.error(f"Model {args.model_id} does not support get_symbol_alphas()")
        return 1
    
    # Step 1.5: Determine model type and required factors
    model_type = getattr(model, 'model_type', None)
    if model_type is None:
        # Try to infer from model_id
        if 'ff3' in args.model_id.lower():
            model_type = 'ff3_regression'
        elif 'ff5' in args.model_id.lower():
            model_type = 'ff5_regression'
        else:
            logger.warning("Could not determine model type, defaulting to FF5")
            model_type = 'ff5_regression'
    
    # Determine required factors based on model type
    if model_type == 'ff3_regression':
        required_factors = ['MKT', 'SMB', 'HML']
        logger.info("Using FF3 factors: MKT, SMB, HML")
    else:
        required_factors = ['MKT', 'SMB', 'HML', 'RMW', 'CMA']
        logger.info("Using FF5 factors: MKT, SMB, HML, RMW, CMA")
    
    # Step 2: Extract symbols from model alphas
    alphas = model.get_symbol_alphas()
    if not alphas:
        logger.error(f"Model {args.model_id} has no alphas")
        return 1
    
    symbols = list(alphas.keys())
    logger.info(f"Found {len(symbols)} symbols in model: {symbols[:10]}...")
    
    # Step 3: Determine date range (priority: args > registry tags > config > error)
    start_date = None
    end_date = None
    
    # Priority 1: Use command line arguments if provided
    if args.start_date:
        start_date = pd.to_datetime(args.start_date)
    if args.end_date:
        end_date = pd.to_datetime(args.end_date)
    
    # Priority 2: Try to get dates from model registry tags
    if not start_date or not end_date:
        registry = ModelRegistry(args.model_registry_path)
        meta = registry.get_model_metadata(args.model_id)
        if meta and 'tags' in meta:
            tags = meta.get('tags', {})
            period = tags.get('data_period')  # e.g., "2024-01-01_2025-06-30"
            if period and '_' in period:
                try:
                    ds, de = period.split('_', 1)
                    if not start_date:
                        start_date = pd.to_datetime(ds)
                    if not end_date:
                        end_date = pd.to_datetime(de)
                    logger.info(f"Using training data_period from registry tags: {start_date.date()} to {end_date.date()}")
                except Exception as e:
                    logger.warning(f"Failed to parse data_period '{period}': {e}")
    
    # Priority 3: Try to get dates from config file
    if not start_date or not end_date:
        if args.config:
            import yaml
            with open(args.config, 'r') as f:
                config = yaml.safe_load(f)
            training_params = config.get('training_setup', {}).get('parameters', {})
            if not start_date:
                start_date = pd.to_datetime(training_params.get('start_date', '2022-01-01'))
            if not end_date:
                end_date = pd.to_datetime(training_params.get('end_date', '2023-12-31'))
            logger.info(f"Using dates from config: {start_date.date()} to {end_date.date()}")
    
    # Priority 4: Error if still no dates
    if not start_date or not end_date:
        logger.error("Must provide either --config with training dates, or --start-date and --end-date, or dates must be in model registry tags")
        return 1
    
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Lookback window: {args.lookback} trading days")
    
    # Step 4: Create data providers (reuse training setup)
    if args.config:
        logger.info(f"Creating data providers from config: {args.config}")
        data_provider, factor_data_provider = create_data_providers_from_config(args.config)
    else:
        logger.warning("No config provided. Using default providers (may not match training setup).")
        from src.trading_system.data.yfinance_provider import YFinanceProvider
        from src.trading_system.data.ff5_provider import FF5DataProvider
        data_provider = YFinanceProvider(max_retries=3, retry_delay=1.0)
        factor_data_provider = FF5DataProvider(data_frequency='daily')
    
    # Step 5: Create minimal TrainingPipeline for data loading (reuse _load_data method)
    logger.info("Creating training pipeline for data loading...")
    # We need feature_pipeline for TrainingPipeline, but we only use _load_data
    # Create a minimal feature pipeline using the detected model type
    feature_pipeline = FeatureEngineeringPipeline.from_config({}, model_type=model_type)
    
    train_pipeline = TrainingPipeline(
        model_type=model_type,
        feature_pipeline=feature_pipeline,
        registry_path=args.model_registry_path
    )
    train_pipeline.configure_data(data_provider=data_provider, factor_data_provider=factor_data_provider)
    
    # Step 6: Load data using training pipeline's _load_data method
    logger.info(f"Loading data for {len(symbols)} symbols...")
    price_data, factor_data, target_data = train_pipeline._load_data(
        start_date, end_date, symbols
    )
    
    if not price_data:
        logger.error("Failed to load price data")
        return 1
    
    if not isinstance(factor_data, pd.DataFrame) or factor_data.empty:
        logger.error("Failed to load factor data")
        return 1
    
    logger.info(f"Loaded price data for {len(price_data)} symbols")
    logger.info(f"Factor data shape: {factor_data.shape}")
    
    # Step 7: Compute t-stats for each symbol
    logger.info("Computing t-statistics...")
    results = []
    # required_factors already determined in Step 1.5
    
    for symbol in symbols:
        if symbol not in price_data:
            logger.warning(f"Symbol {symbol} not in price data, skipping")
            continue
        
        # Get price returns
        prices = price_data[symbol]
        if 'Close' not in prices.columns:
            logger.warning(f"Symbol {symbol} missing 'Close' column, skipping")
            continue
        
        # Calculate raw returns
        raw_returns = prices['Close'].pct_change().dropna()
        
        # Use last lookback days (matching training window)
        if len(raw_returns) < args.lookback:
            logger.warning(f"Symbol {symbol} has insufficient data: {len(raw_returns)} < {args.lookback}")
            continue
        
        returns_window = raw_returns.tail(args.lookback).copy()
        
        # Align factor data to returns dates
        # Ensure factor_data index is datetime if it isn't already
        if not isinstance(factor_data.index, pd.DatetimeIndex):
            factor_data = factor_data.copy()
            factor_data.index = pd.to_datetime(factor_data.index)
        
        # Get factor data for the returns date range
        returns_start_date = returns_window.index.min()
        returns_end_date = returns_window.index.max()
        
        # Select factor data in the date range
        factor_mask = (factor_data.index >= returns_start_date) & (factor_data.index <= returns_end_date)
        factor_window = factor_data.loc[factor_mask].copy()
        
        if factor_window.empty:
            logger.warning(f"Symbol {symbol} has no overlapping factor data")
            continue
        
        # If factor data is monthly but returns are daily, forward fill to daily frequency
        if len(factor_window) < len(returns_window) * 0.5:
            # Reindex to daily frequency using forward fill
            # Note: method='ffill' is deprecated but still works in pandas 1.x
            # For pandas 2.x compatibility, we could use: factor_window.reindex(returns_window.index).ffill()
            try:
                factor_window = factor_window.reindex(returns_window.index, method='ffill')
            except TypeError:
                # For pandas 2.x, use ffill() method instead
                factor_window = factor_window.reindex(returns_window.index).ffill()
            # Drop rows where forward fill didn't work (NaN values at the beginning)
            factor_window = factor_window.dropna()
            # Align returns to factor_window dates after dropping NaN
            returns_window = returns_window.loc[factor_window.index]
            # Ensure it's still a DataFrame
            if not isinstance(factor_window, pd.DataFrame):
                factor_window = pd.DataFrame(factor_window)
        else:
            # Align by intersection of dates
            common_dates = returns_window.index.intersection(factor_window.index)
            if len(common_dates) < 30:
                logger.warning(f"Symbol {symbol} has insufficient aligned dates: {len(common_dates)}")
                continue
            returns_window = returns_window.loc[common_dates]
            factor_window = factor_window.loc[common_dates]
            # Ensure it's still a DataFrame
            if not isinstance(factor_window, pd.DataFrame):
                factor_window = pd.DataFrame(factor_window)
        
        # Calculate excess returns (stock return - risk-free rate)
        # Factor data should have 'RF' column for risk-free rate
        if 'RF' in factor_window.columns:
            # Convert RF from decimal to percentage if needed (RF is typically in percentage)
            # FF5 data RF is typically already in decimal form (e.g., 0.01 for 1%)
            risk_free_rate = factor_window['RF']
            # Align risk-free rate to returns dates
            risk_free_aligned = risk_free_rate.loc[returns_window.index]
            # Calculate excess returns
            returns_window = returns_window - risk_free_aligned
        else:
            logger.debug(f"Symbol {symbol}: No RF column in factor data, using raw returns")
        
        # Ensure we have the required factor columns
        if not isinstance(factor_window, pd.DataFrame):
            logger.warning(f"Symbol {symbol} factor_window is not a DataFrame")
            continue
        if not all(col in factor_window.columns for col in required_factors):
            logger.warning(f"Symbol {symbol} missing required factor columns: have {list(factor_window.columns)}, need {required_factors}")
            continue
        
        # Select only the required factor columns as DataFrame
        factor_subset = factor_window[required_factors].copy()
        
        # Ensure both have the same index type for proper alignment
        if not isinstance(returns_window.index, pd.DatetimeIndex):
            returns_window.index = pd.to_datetime(returns_window.index)
        if not isinstance(factor_subset.index, pd.DatetimeIndex):
            factor_subset.index = pd.to_datetime(factor_subset.index)
        
        # Debug: Check alignment before regression
        common_before = returns_window.index.intersection(factor_subset.index)
        if len(common_before) < 30:
            logger.warning(f"Symbol {symbol}: Only {len(common_before)} common dates after alignment. Returns: {len(returns_window)}, Factors: {len(factor_subset)}")
            continue
        
        stats = compute_alpha_tstat(returns_window, factor_subset, required_factors=required_factors)
        
        # Log warning if all stats are zero (indicating a problem)
        if stats['t_stat'] == 0.0 and stats['r_squared'] == 0.0 and stats['n_obs'] > 0:
            logger.warning(f"{symbol}: Regression returned zeros but n_obs={stats['n_obs']}. "
                         f"Returns shape: {returns_window.shape}, Factors shape: {factor_subset.shape}, "
                         f"Common dates: {len(common_before)}")
        
        results.append({
            'symbol': symbol,
            't_alpha': stats['t_stat'],
            'p_value': stats['p_value'],
            'r_squared': stats['r_squared'],
            'n_obs': stats['n_obs']
        })
        
        logger.debug(f"{symbol}: t={stats['t_stat']:.3f}, p={stats['p_value']:.4f}, R²={stats['r_squared']:.3f}, n={stats['n_obs']}")
    
    # Step 8: Save results
    if not results:
        logger.error("No results to save")
        return 1
    
    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    
    logger.info(f"Saved {len(results)} results to {args.output}")
    
    # Print summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"Total symbols processed: {len(results)}")
    logger.info(f"Mean |t-stat|: {df['t_alpha'].abs().mean():.3f}")
    logger.info(f"Std |t-stat|: {df['t_alpha'].abs().std():.3f}")
    logger.info(f"Significant (|t| >= 2.0): {(df['t_alpha'].abs() >= 2.0).sum()} ({(df['t_alpha'].abs() >= 2.0).sum() / len(df) * 100:.1f}%)")
    logger.info(f"Mean R²: {df['r_squared'].mean():.3f}")
    logger.info(f"Mean n_obs: {df['n_obs'].mean():.1f}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())