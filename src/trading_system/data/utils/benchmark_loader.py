"""
Benchmark data loader from CSV files.

Single responsibility: Load and format benchmark price data from CSV files.
Follows KISS principle - only handles CSV loading and basic formatting.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_benchmark_from_csv(
    csv_path: str,
    start_date: datetime,
    end_date: datetime
) -> Optional[pd.DataFrame]:
    """
    Load benchmark price data from a CSV file and format it for backtesting.

    Args:
        csv_path: Path to CSV file with benchmark data
        start_date: Start date for data filtering
        end_date: End date for data filtering

    Returns:
        DataFrame with OHLCV data (Date as index), or None if file not found/invalid

    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid or missing required columns
    """
    csv_path_obj = Path(csv_path)
    
    # Check if file exists
    if not csv_path_obj.exists():
        logger.error(f"Benchmark CSV file not found: {csv_path}")
        raise FileNotFoundError(f"Benchmark CSV file not found: {csv_path}")
    
    try:
        # Try reading with Date as index first (most common format)
        df = pd.read_csv(
            csv_path,
            index_col=0,
            parse_dates=True
        )
        
        # If index is not datetime, try reading Date as a column
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.debug("Date not found as index, trying as column")
            df = pd.read_csv(csv_path, parse_dates=['Date'])
            
            if 'Date' in df.columns:
                df = df.set_index('Date')
            else:
                # Try common date column names
                for date_col in ['date', 'DATE', 'Time', 'time']:
                    if date_col in df.columns:
                        df = df.set_index(date_col)
                        break
                else:
                    raise ValueError("No date column found in CSV file")
        
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("Failed to parse date column as datetime")
        
        # Identify price column (prefer Close, fallback to Last Price or similar)
        price_column = None
        for col in ['Close', 'close', 'CLOSE', 'Last Price', 'Last Price', 'Price', 'price']:
            if col in df.columns:
                price_column = col
                break
        
        if price_column is None:
            raise ValueError("No price column found (expected: Close, Last Price, or similar)")
        
        # Ensure required OHLC columns exist (for compatibility with backtest engine)
        if 'Close' not in df.columns:
            df['Close'] = df[price_column]
        if 'Open' not in df.columns:
            df['Open'] = df['Close']
        if 'High' not in df.columns:
            df['High'] = df['Close']
        if 'Low' not in df.columns:
            df['Low'] = df['Close']
        if 'Volume' not in df.columns:
            df['Volume'] = 0
        if 'Adj Close' not in df.columns:
            df['Adj Close'] = df['Close']
        
        # Keep only OHLCV columns (standard format)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']].copy()
        
        # Remove rows with NaN in critical columns
        df = df.dropna(subset=['Close'])
        
        # Filter by date range
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        if df.empty:
            logger.warning(
                f"No benchmark data in date range {start_date.date()} to {end_date.date()}"
            )
            return None
        
        # Sort by date
        df = df.sort_index()
        
        logger.info(
            f"Loaded benchmark data from {csv_path}: {len(df)} rows "
            f"({df.index.min().date()} to {df.index.max().date()})"
        )
        
        return df
        
    except pd.errors.EmptyDataError:
        logger.error(f"Benchmark CSV file is empty: {csv_path}")
        raise ValueError(f"Benchmark CSV file is empty: {csv_path}")
    except Exception as e:
        logger.error(f"Failed to load benchmark from CSV {csv_path}: {e}")
        raise ValueError(f"Failed to load benchmark from CSV {csv_path}: {e}")

