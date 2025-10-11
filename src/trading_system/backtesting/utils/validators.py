"""
Data Validation Utilities

Provides validation functions for backtesting data inputs and configurations.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


def validate_inputs(strategy_signals: Dict[datetime, List[Any]],
                   price_data: Dict[str, pd.DataFrame],
                   benchmark_data: Optional[pd.DataFrame] = None) -> bool:
    """
    Validate backtest input data.

    Args:
        strategy_signals: Dictionary mapping dates to trading signals
        price_data: Dictionary mapping symbols to price DataFrames
        benchmark_data: Optional benchmark price DataFrame

    Returns:
        True if all inputs are valid

    Raises:
        ValueError: If validation fails
    """
    # Validate strategy signals
    if not strategy_signals:
        raise ValueError("No strategy signals provided")

    if len(strategy_signals) == 0:
        raise ValueError("Strategy signals dictionary is empty")

    # Validate price data
    if not price_data:
        raise ValueError("No price data provided")

    if len(price_data) == 0:
        raise ValueError("Price data dictionary is empty")

    # Check that all symbols in signals have price data
    all_symbols = set()
    for signals in strategy_signals.values():
        for signal in signals:
            if hasattr(signal, 'symbol'):
                all_symbols.add(signal.symbol)
            elif isinstance(signal, dict) and 'symbol' in signal:
                all_symbols.add(signal['symbol'])

    missing_symbols = all_symbols - set(price_data.keys())
    if missing_symbols:
        raise ValueError(f"Missing price data for symbols: {missing_symbols}")

    # Validate price data format
    for symbol, data in price_data.items():
        validate_price_data(data, symbol)

    # Validate benchmark data if provided
    if benchmark_data is not None:
        validate_price_data(benchmark_data, "benchmark")

    logger.info(f"Input validation passed: {len(all_symbols)} symbols, "
               f"{len(strategy_signals)} signal dates")
    return True


def validate_price_data(price_data: pd.DataFrame, name: str) -> bool:
    """
    Validate price DataFrame format.

    Args:
        price_data: Price DataFrame
        name: Name of the data (for error messages)

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    if price_data is None or len(price_data) == 0:
        raise ValueError(f"{name} price data is empty")

    # Check required columns
    required_columns = ['Open', 'High', 'Low', 'Close']
    missing_cols = set(required_columns) - set(price_data.columns)
    if missing_cols:
        raise ValueError(f"Missing columns {missing_cols} in {name} price data")

    # Check for NaN values in essential columns
    essential_cols = ['Close']
    for col in essential_cols:
        if col in price_data.columns and price_data[col].isna().all():
            raise ValueError(f"All values are NaN in {col} column for {name}")

    # Check data types
    if not all(price_data[col].dtype in ['float64', 'int64', 'float32', 'int32']
              for col in required_columns):
        logger.warning(f"Non-numeric data types found in {name} price data")

    logger.debug(f"{name} price data validation passed: {len(price_data)} rows")
    return True


def validate_configuration(config: Dict[str, Any]) -> bool:
    """
    Validate backtest configuration.

    Args:
        config: Configuration dictionary

    Returns:
        True if valid

    Raises:
        ValueError: If validation fails
    """
    required_keys = ['initial_capital']
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {missing_keys}")

    # Validate capital
    if config.get('initial_capital', 0) <= 0:
        raise ValueError("initial_capital must be positive")

    # Validate dates if provided
    if 'start_date' in config and 'end_date' in config:
        start_date = config['start_date']
        end_date = config['end_date']

        if isinstance(start_date, str):
            start_date = datetime.fromisoformat(start_date)
        if isinstance(end_date, str):
            end_date = datetime.fromisoformat(end_date)

        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")

    # Validate numerical parameters
    float_params = ['commission_rate', 'spread_rate', 'slippage_rate',
                   'position_limit', 'max_drawdown_limit']
    for param in float_params:
        if param in config and config[param] is not None:
            if not isinstance(config[param], (int, float)):
                raise ValueError(f"{param} must be a number")
            if config[param] < 0:
                raise ValueError(f"{param} must be non-negative")

    logger.debug("Configuration validation passed")
    return True


def align_data_periods(strategy_signals: Dict[datetime, List[Any]],
                      price_data: Dict[str, pd.DataFrame]) -> Dict[datetime, List[Any]]:
    """
    Align strategy signals with available price data periods.

    Args:
        strategy_signals: Original strategy signals
        price_data: Price data for all symbols

    Returns:
        Aligned strategy signals
    """
    if not strategy_signals or not price_data:
        return strategy_signals

    # Get available trading days from price data
    price_dates = set()
    for symbol_data in price_data.values():
        price_dates.update(symbol_data.index)

    # Convert to datetime if needed
    normalized_dates = set()
    for date in price_dates:
        if hasattr(date, 'date'):
            normalized_dates.add(date.date())
        elif isinstance(date, str):
            normalized_dates.add(datetime.fromisoformat(date).date())
        else:
            normalized_dates.add(date)

    # Filter signals to only include dates with price data
    aligned_signals = {}
    for signal_date, signals in strategy_signals.items():
        date_to_check = signal_date.date() if hasattr(signal_date, 'date') else signal_date

        if date_to_check in normalized_dates:
            aligned_signals[signal_date] = signals
        else:
            logger.debug(f"Filtering out signal date {signal_date} - no price data available")

    filtered_count = len(strategy_signals) - len(aligned_signals)
    if filtered_count > 0:
        logger.info(f"Filtered out {filtered_count} signal dates due to missing price data")

    return aligned_signals


def clean_price_data(price_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Clean and prepare price data for backtesting.

    Args:
        price_data: Raw price data dictionary

    Returns:
        Cleaned price data dictionary
    """
    cleaned_data = {}

    for symbol, data in price_data.items():
        try:
            # Make a copy to avoid modifying original data
            cleaned = data.copy()

            # Forward fill missing values
            cleaned = cleaned.ffill()

            # Backward fill any remaining NaN at the beginning
            cleaned = cleaned.bfill()

            # Remove any remaining NaN rows
            cleaned = cleaned.dropna()

            # Ensure data is sorted by date
            cleaned = cleaned.sort_index()

            # Validate we still have data
            if len(cleaned) > 0:
                cleaned_data[symbol] = cleaned
                logger.debug(f"Cleaned {symbol}: {len(data)} -> {len(cleaned)} rows")
            else:
                logger.warning(f"All data removed during cleaning for {symbol}")

        except Exception as e:
            logger.error(f"Error cleaning price data for {symbol}: {e}")
            continue

    return cleaned_data


def estimate_data_quality(price_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """
    Estimate quality metrics for price data.

    Args:
        price_data: Price data dictionary

    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {}

    for symbol, data in price_data.items():
        if data is None or len(data) == 0:
            continue

        metrics = {
            'total_rows': len(data),
            'date_range': {
                'start': data.index[0].date() if hasattr(data.index[0], 'date') else data.index[0],
                'end': data.index[-1].date() if hasattr(data.index[-1], 'date') else data.index[-1]
            },
            'missing_values': data.isnull().sum().to_dict(),
            'duplicate_dates': data.index.duplicated().sum(),
            'data_frequency': _estimate_frequency(data.index),
            'price_changes': {
                'max_daily_change': _estimate_max_daily_change(data),
                'volatility': data['Close'].pct_change().std() * (252 ** 0.5) if len(data) > 1 else 0
            }
        }

        quality_metrics[symbol] = metrics

    return quality_metrics


def _estimate_frequency(date_index) -> str:
    """Estimate the frequency of the data."""
    if len(date_index) < 2:
        return "unknown"

    # Calculate median difference between dates
    time_diffs = date_index.to_series().diff().dropna()
    median_diff = time_diffs.median()

    # Convert to days
    if hasattr(median_diff, 'days'):
        median_days = median_diff.days
    else:
        median_days = median_diff.total_seconds() / (24 * 3600)

    if median_days <= 1.5:
        return "daily"
    elif median_days <= 8:
        return "weekly"
    elif median_days <= 35:
        return "monthly"
    else:
        return "low_frequency"


def _estimate_max_daily_change(data: pd.DataFrame) -> float:
    """Estimate maximum daily price change percentage."""
    if len(data) < 2:
        return 0.0

    daily_changes = data['Close'].pct_change().dropna()
    return daily_changes.abs().max() if len(daily_changes) > 0 else 0.0