"""
YFinance data provider with retry logic and comprehensive error handling.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

from ..types.enums import DataSource
from ..utils.validation import DataValidator, DataValidationError

logger = logging.getLogger(__name__)


class YFinanceProvider:
    """
    Robust Yahoo Finance data provider with retry logic and data validation.

    Features:
    - Automatic retry on API failures
    - Data validation and cleaning
    - Request rate limiting
    - Comprehensive error logging
    - Support for multiple data types (OHLCV, dividends, etc.)
    """

    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0,
                 request_timeout: int = 30):
        """
        Initialize the YFinance provider.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            request_timeout: Request timeout in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests

    def _wait_for_rate_limit(self):
        """Implement rate limiting to avoid API restrictions."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last_request
            logger.debug(f"Rate limiting: waiting {sleep_time:.2f}s")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def _fetch_with_retry(self, fetch_func, *args, **kwargs) -> Optional[pd.DataFrame]:
        """
        Execute fetch function with retry logic.

        Args:
            fetch_func: Function to execute (e.g., yf.download)
            *args, **kwargs: Arguments for the fetch function

        Returns:
            DataFrame with data or None if all retries fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()

                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}")

                # Add auto_adjust=False to prevent yfinance from adjusting prices automatically
                if 'auto_adjust' not in kwargs:
                    kwargs['auto_adjust'] = False
                data = fetch_func(*args, **kwargs)

                if data is not None and not data.empty:
                    return data
                else:
                    logger.warning(f"Empty data returned on attempt {attempt + 1}")
                    last_error = "Empty data returned"

            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {e}")

                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)

        logger.error(f"All {self.max_retries} attempts failed. Last error: {last_error}")
        return None

    def get_historical_data(self, symbols: Union[str, List[str]],
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime] = None,
                           period: str = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical OHLCV data for one or more symbols.

        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch (default: today)
            period: Alternative to start/end dates (e.g., '1y', '6mo', '3d')

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        if end_date is None:
            end_date = datetime.now()

        if isinstance(symbols, str):
            symbols = [symbols]

        logger.info(f"Fetching historical data for {len(symbols)} symbols "
                   f"from {start_date} to {end_date}")

        results = {}

        for symbol in symbols:
            logger.debug(f"Fetching data for {symbol}")

            try:
                if period:
                    data = self._fetch_with_retry(
                        yf.download, symbol, period=period,
                        progress=False, timeout=self.request_timeout
                    )
                else:
                    data = self._fetch_with_retry(
                        yf.download, symbol, start=start_date, end=end_date,
                        progress=False, timeout=self.request_timeout
                    )

                if data is not None and not data.empty:
                    # Data validation and cleaning
                    data = self._validate_and_clean_data(data, symbol)
                    results[symbol] = data
                    logger.info(f"Successfully fetched {len(data)} rows for {symbol}")
                else:
                    logger.warning(f"No data returned for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue

        return results

    def get_latest_price(self, symbols: Union[str, List[str]]) -> Dict[str, float]:
        """
        Get latest price for one or more symbols.

        Args:
            symbols: Single symbol or list of symbols

        Returns:
            Dictionary mapping symbols to latest prices
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        logger.info(f"Fetching latest prices for {len(symbols)} symbols")

        results = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Get today's data or most recent trading day
                data = self._fetch_with_retry(
                    ticker.history, period="1d", interval="1d",
                    timeout=self.request_timeout
                )

                if data is not None and not data.empty:
                    latest_price = data['Close'].iloc[-1]
                    results[symbol] = latest_price
                    logger.debug(f"Latest price for {symbol}: ${latest_price:.2f}")
                else:
                    logger.warning(f"No price data available for {symbol}")

            except Exception as e:
                logger.error(f"Failed to get latest price for {symbol}: {e}")
                continue

        return results

    def get_dividends(self, symbols: Union[str, List[str]],
                     start_date: Union[str, datetime] = None,
                     end_date: Union[str, datetime] = None) -> Dict[str, pd.Series]:
        """
        Fetch dividend data for symbols.

        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for dividend data
            end_date: End date for dividend data

        Returns:
            Dictionary mapping symbols to dividend Series
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        logger.info(f"Fetching dividend data for {len(symbols)} symbols")

        results = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                dividends = ticker.dividends

                if start_date or end_date:
                    dividends = self._filter_dates(dividends, start_date, end_date)

                if not dividends.empty:
                    results[symbol] = dividends
                    logger.debug(f"Found {len(dividends)} dividends for {symbol}")
                else:
                    logger.debug(f"No dividends found for {symbol}")

            except Exception as e:
                logger.error(f"Failed to fetch dividends for {symbol}: {e}")
                continue

        return results

    def _normalize_yfinance_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalize YFinance data format to standard structure.

        Args:
            data: Raw DataFrame from yfinance
            symbol: Symbol name for processing

        Returns:
            Normalized DataFrame with standard column structure
        """
        # Handle MultiIndex columns (YFinance default for single symbol)
        if isinstance(data.columns, pd.MultiIndex):
            logger.debug(f"Normalizing MultiIndex columns for {symbol}")
            # For single symbol data, flatten the MultiIndex
            data.columns = data.columns.get_level_values(0)

        # Handle multiple symbols data (different structure)
        elif data.columns.nlevels > 1:
            logger.debug(f"Normalizing multi-symbol data for {symbol}")
            # For multi-symbol data, extract the specific symbol
            if symbol in data.columns.get_level_values(1):
                # Create single symbol DataFrame
                symbol_data = pd.DataFrame()
                for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
                    if (col, symbol) in data.columns:
                        symbol_data[col] = data[(col, symbol)]
                data = symbol_data

        return data

    def _validate_and_clean_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate and clean fetched data.

        Args:
            data: Raw DataFrame from yfinance
            symbol: Symbol name for logging

        Returns:
            Cleaned and validated DataFrame

        Raises:
            DataValidationError: If data validation fails
        """
        try:
            # First normalize the data format
            data = self._normalize_yfinance_data(data, symbol)

            # Check required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in data.columns]

            if missing_columns:
                logger.warning(f"Missing columns {missing_columns} for {symbol}")
                # Add missing columns with NaN values
                for col in missing_columns:
                    data[col] = pd.NA

            # Remove rows with all NaN values
            initial_len = len(data)
            data = data.dropna(how='all')
            if len(data) < initial_len:
                logger.warning(f"Removed {initial_len - len(data)} empty rows for {symbol}")

            # Use the new data validator for comprehensive checks
            try:
                DataValidator.validate_price_data(data, symbol)
            except DataValidationError as e:
                logger.warning(f"Data validation issue for {symbol}: {e}")
                # Attempt to fix common issues

                # Check for anomalous values and fix them
                for col in ['Open', 'High', 'Low', 'Close']:
                    if col in data.columns:
                        # Negative prices are invalid
                        negative_prices = data[data[col] < 0]
                        if not negative_prices.empty:
                            logger.warning(f"Found {len(negative_prices)} negative prices in {col} for {symbol}")
                            data.loc[data[col] < 0, col] = pd.NA

                # Validate High-Low relationships
                invalid_hl = data[data['High'] < data['Low']]
                if not invalid_hl.empty:
                    logger.warning(f"Found {len(invalid_hl)} invalid High-Low pairs for {symbol}")
                    # Fix by swapping values
                    data.loc[invalid_hl.index, ['High', 'Low']] = \
                        data.loc[invalid_hl.index, ['Low', 'High']].values

                # Validate High-Low-Open-Close relationships
                invalid_ohlc = data[
                    (data['High'] < data['Open']) |
                    (data['High'] < data['Close']) |
                    (data['Low'] > data['Open']) |
                    (data['Low'] > data['Close'])
                ]
                if not invalid_ohlc.empty:
                    logger.warning(f"Found {len(invalid_ohlc)} invalid OHLC relationships for {symbol}")
                    # Fix by adjusting high/low
                    data.loc[invalid_ohlc.index, 'High'] = data.loc[invalid_ohlc.index, ['Open', 'Close', 'High']].max(axis=1)
                    data.loc[invalid_ohlc.index, 'Low'] = data.loc[invalid_ohlc.index, ['Open', 'Close', 'Low']].min(axis=1)

            # Sort by date and ensure datetime index
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)

            data = data.sort_index()

            # Add symbol column for multi-symbol datasets
            data['Symbol'] = symbol

            # Add data source metadata
            data['DataSource'] = DataSource.YFINANCE.value

            return data

        except Exception as e:
            logger.error(f"Failed to validate and clean data for {symbol}: {e}")
            raise DataValidationError(f"Data validation failed for {symbol}: {e}")

    def _filter_dates(self, data: pd.Series, start_date: Union[str, datetime] = None,
                     end_date: Union[str, datetime] = None) -> pd.Series:
        """Filter data by date range."""
        if start_date is not None:
            data = data[data.index >= pd.to_datetime(start_date)]
        if end_date is not None:
            data = data[data.index <= pd.to_datetime(end_date)]
        return data

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a symbol exists and has recent data.

        Args:
            symbol: Symbol to validate

        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            # Try to get recent data
            data = self._fetch_with_retry(
                ticker.history, period="5d", interval="1d",
                timeout=self.request_timeout
            )
            return data is not None and not data.empty
        except Exception as e:
            logger.warning(f"Symbol validation failed for {symbol}: {e}")
            return False