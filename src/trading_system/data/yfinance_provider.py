"""
YFinance data provider with retry logic and comprehensive error handling.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple

import pandas as pd
import yfinance as yf
from pandas.tseries.offsets import BDay

from ..types.enums import DataSource
from .validation import DataValidator, DataValidationError
from .base_data_provider import PriceDataProvider

logger = logging.getLogger(__name__)


class YFinanceProvider(PriceDataProvider):
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
                 request_timeout: int = 30, cache_enabled: bool = True,
                 symbols: Optional[List[str]] = None, start_date: Optional[str] = None,
                 liquidity_config: Optional[Dict[str, Any]] = None,
                 exchange_suffix_map: Optional[Dict[str, str]] = None):
        """
        Initialize the YFinance provider.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            request_timeout: Request timeout in seconds
            cache_enabled: Whether to enable caching
            symbols: List of symbols to provide data for (optional)
            start_date: Start date for historical data (optional)
            liquidity_config: Liquidity filtering configuration
        """
        super().__init__(
            max_retries=max_retries,
            retry_delay=retry_delay,
            request_timeout=request_timeout,
            cache_enabled=cache_enabled,
            rate_limit=0.5  # 500ms between requests
        )

        # Store symbols and start_date for later use
        self.symbols = symbols
        self.start_date = start_date
        self.liquidity_config = liquidity_config or {}

        # Optional mapping to override default exchange suffixes (merged at runtime)
        self._custom_exchange_suffix_map = exchange_suffix_map or {}

        # Cache successful symbol resolutions for this session
        self._symbol_resolution_cache: Dict[str, str] = {}

    @staticmethod
    def _default_exchange_suffix_map() -> Dict[str, str]:
        """Return a sensible default mapping from exchange/country codes to Yahoo suffixes.

        Notes:
        - US listings typically have no suffix. Bloomberg often uses UW/UN for NASDAQ/NYSE.
        - Some markets have multiple suffices; alternates will be tried separately.
        """
        return {
            # United States / Primary US venues
            "US": "", "UW": "", "UN": "", "USA": "",
            # United Kingdom (London Stock Exchange)
            "LN": ".L",
            # Hong Kong
            "HK": ".HK",
            # Japan (Tokyo)
            "JP": ".T", "JT": ".T",
            # Korea
            "KS": ".KS",  # KOSPI
            "KQ": ".KQ",  # KOSDAQ
            # Mainland China
            "SS": ".SS",  # Shanghai
            "SZ": ".SZ",  # Shenzhen
            # Taiwan
            "TT": ".TW",  # Bloomberg TT -> Yahoo TW (alt .TWO)
            # Australia
            "AU": ".AX",
            # Canada
            "TO": ".TO",  # TSX
            "V": ".V",    # TSXV
            "CN": ".TO",   # Some sources use CN; prefer TSX as default
            # Switzerland
            "SW": ".SW",
            "VX": ".VX",
            # Eurozone common
            "GR": ".DE",   # Germany (XETRA)
            "DE": ".DE",
            "FP": ".PA",   # France (Paris)
            "PA": ".PA",
            "AS": ".AS",   # Netherlands (Amsterdam)
            "NA": ".AS",
            "MI": ".MI",   # Italy (Milan)
            "BR": ".BR",   # Belgium (Brussels)
            "VI": ".VI",   # Austria (Vienna)
            "OL": ".OL",   # Norway (Oslo)
            "ST": ".ST",   # Sweden (Stockholm)
        }

    @staticmethod
    def _alternate_suffix_candidates(code: str) -> List[str]:
        """Return alternate Yahoo suffixes to try for a given exchange/country code."""
        code = (code or "").upper()
        if code in ("TO", "CN", "V"):
            return [".TO", ".V"]
        if code in ("SW", "VX"):
            return [".SW", ".VX"]
        if code in ("TT",):
            return [".TW", ".TWO"]
        return []

    def _merge_exchange_suffix_map(self) -> Dict[str, str]:
        mapping = self._default_exchange_suffix_map().copy()
        mapping.update(self._custom_exchange_suffix_map)
        return mapping

    def resolve_symbol(self, ticker: str, country_code: Optional[str]) -> List[str]:
        """Generate candidate Yahoo symbols from raw ticker and optional country/exchange code.

        Returns ordered candidates to try; the caller should attempt fetches in order.
        """
        if not ticker:
            return []

        raw = ticker.strip().upper()
        code = (country_code or "").strip().upper()

        # Honor cached resolution first
        cache_key = f"{raw}|{code}"
        if cache_key in self._symbol_resolution_cache:
            return [self._symbol_resolution_cache[cache_key], raw]

        mapping = self._merge_exchange_suffix_map()

        candidates: List[str] = []
        # 1) ticker + mapped suffix (if any)
        if code and code in mapping:
            candidates.append(f"{raw}{mapping[code]}")
            # 1b) alternates for this code
            for suf in self._alternate_suffix_candidates(code):
                c = f"{raw}{suf}"
                if c not in candidates:
                    candidates.append(c)
        # 2) raw ticker as fallback
        if raw not in candidates:
            candidates.append(raw)

        return candidates

    def get_data_source(self) -> DataSource:
        """Get the data source enum for this provider."""
        return DataSource.YFINANCE
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about this data provider."""
        return {
            'provider': 'Yahoo Finance',
            'data_source': DataSource.YFINANCE.value,
            'description': 'Yahoo Finance stock data provider',
            'features': [
                'Historical price data',
                'Real-time quotes',
                'Dividend data',
                'Retry logic',
                'Rate limiting',
                'Data validation'
            ],
            'rate_limit': self.rate_limit,
            'max_retries': self.max_retries,
            'cache_enabled': self.cache_enabled
        }
    
    def _fetch_raw_data(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """Fetch raw data from YFinance API."""
        # This method is called by the base class's _fetch_with_retry
        # The actual fetching logic is in the specific methods
        pass

    def get_historical_data(self, symbols: Union[str, List[str]],
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime] = None,
                           period: str = None,
                           liquidity_config: Optional[Dict[str, Any]] = None,
                           ticker_country_map: Optional[Dict[str, Optional[str]]] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical OHLCV data for one or more symbols.

        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch (default: today)
            period: Alternative to start/end dates (e.g., '1y', '6mo', '3d')
            liquidity_config: Liquidity filtering configuration (overrides instance config)

        Returns:
            Dictionary mapping successful symbols to their DataFrames
            (Failed symbols are logged but not returned)
        """
        if end_date is None:
            end_date = datetime.now()

        if isinstance(symbols, str):
            symbols = [symbols]

        logger.info(f"Fetching historical data for {len(symbols)} symbols "
                   f"from {start_date} to {end_date}")

        results = {}
        failed_symbols = []

        for symbol in symbols:
            logger.debug(f"Fetching data for {symbol}")

            # Generate resolution candidates based on optional country/exchange code
            country_code = None
            if ticker_country_map and symbol in ticker_country_map:
                country_code = ticker_country_map.get(symbol)
            candidates = self.resolve_symbol(symbol, country_code)
            if not candidates:
                candidates = [symbol]

            try:
                # Try candidates in order until one succeeds
                fetch_success = False
                last_error: Optional[Exception] = None

                for resolved in candidates:
                    # Check cache per resolved symbol
                    cache_key = self._get_cache_key(resolved, start_date, end_date, period)
                    cached_data = self._get_from_cache(cache_key)
                    if cached_data is not None:
                        results[symbol] = cached_data
                        # Cache resolution for future calls
                        self._symbol_resolution_cache[f"{symbol.strip().upper()}|{(country_code or '').strip().upper()}"] = resolved
                        logger.debug(f"Cache hit for {symbol} via {resolved}")
                        fetch_success = True
                        break

                    # Fetch data with retry logic
                    if period:
                        data = self._fetch_with_retry(
                            yf.download, resolved, period=period,
                            progress=False, timeout=self.request_timeout
                        )
                    else:
                        data = self._fetch_with_retry(
                            yf.download, resolved, start=start_date, end=end_date,
                            progress=False, timeout=self.request_timeout
                        )

                    if data is not None and not data.empty:
                        # Data validation and cleaning using base class method
                        data = self._validate_and_clean_data(data, resolved)
                        data = self.add_data_source_metadata(data)

                        # Apply liquidity filtering if configured
                        effective_liquidity_config = liquidity_config or self.liquidity_config
                        if effective_liquidity_config and effective_liquidity_config.get('enabled', False):
                            data = self.apply_liquidity_filter(data, effective_liquidity_config)

                        # Store in cache using resolved key
                        self._store_in_cache(cache_key, data)

                        # Record result under original symbol key, but log mapping
                        results[symbol] = data
                        self._symbol_resolution_cache[f"{symbol.strip().upper()}|{(country_code or '').strip().upper()}"] = resolved
                        if resolved != symbol:
                            logger.info(f"Resolved {symbol} -> {resolved}")
                        logger.info(f"Successfully fetched {len(data)} rows for {resolved}")
                        fetch_success = True
                        break
                    else:
                        logger.debug(f"No data for candidate {resolved} (from {symbol})")

                if not fetch_success:
                    logger.warning(f"No data returned for {symbol} after trying {len(candidates)} candidate(s)")
                    failed_symbols.append(symbol)

            except Exception as e:
                logger.warning(f"Data unavailable for {symbol}: {e}")
                failed_symbols.append(symbol)
                continue

        # Log summary of data availability with enhanced data quality reporting
        success_count = len(results)
        total_count = len(symbols)
        success_rate = success_count / total_count if total_count > 0 else 0

        logger.info("="*60)
        logger.info("📊 DATA QUALITY REPORT")
        logger.info("="*60)
        logger.info(f"Total requested symbols: {total_count}")
        logger.info(f"Successfully fetched: {success_count}")
        logger.info(f"Failed symbols: {len(failed_symbols)}")
        logger.info(f"Success rate: {success_rate:.1%}")

        if success_count > 0:
            logger.info(f"✅ Successfully fetched data for {success_count} symbols:")
            for symbol in sorted(results.keys()):
                data_points = len(results[symbol])
                date_range = f"{results[symbol].index.min().date()} to {results[symbol].index.max().date()}"
                logger.info(f"  • {symbol}: {data_points} data points ({date_range})")

        if failed_symbols:
            logger.warning(f"⚠️ Failed to fetch data for {len(failed_symbols)} symbols:")
            for symbol in sorted(failed_symbols):
                logger.warning(f"  • {symbol}")

        logger.info("="*60)

        # Provide context about data quality impact
        if success_rate >= 0.9:
            logger.info("✅ Excellent data quality - system should perform optimally")
        elif success_rate >= 0.8:
            logger.info("✅ Good data quality - system should perform well")
        elif success_rate >= 0.7:
            logger.warning("⚠️ Fair data quality - system performance may be impacted")
        else:
            logger.warning("⚠️ Poor data quality - system performance will be significantly impacted")

        return results  # Only return successful data, interface stays clean

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
                # Check cache first
                cache_key = self._get_cache_key(symbol, "latest_price")
                cached_data = self._get_from_cache(cache_key)
                if cached_data is not None:
                    results[symbol] = cached_data
                    continue

                ticker = yf.Ticker(symbol)
                # Get today's data or most recent trading day
                data = self._fetch_with_retry(
                    ticker.history, period="1d", interval="1d",
                    timeout=self.request_timeout
                )

                if data is not None and not data.empty:
                    latest_price = data['Close'].iloc[-1]
                    results[symbol] = latest_price
                    
                    # Store in cache (shorter cache time for latest prices)
                    self._store_in_cache(cache_key, latest_price)
                    
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

            # Use base class validation for price data
            data = self.validate_price_data(data, symbol)

            # Additional YFinance-specific validation and cleaning
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

            # Add symbol column for multi-symbol datasets
            data['Symbol'] = symbol

            return data

        except Exception as e:
            logger.error(f"Failed to validate and clean data for {symbol}: {e}")
            raise DataValidationError(f"Data validation failed for {symbol}: {e}")

    def _filter_dates(self, data: pd.Series, start_date: Union[str, datetime] = None,
                     end_date: Union[str, datetime] = None) -> pd.Series:
        """Filter data by date range."""
        # Convert Series to DataFrame for base class method
        df = data.to_frame()
        filtered_df = self.filter_by_date(df, start_date, end_date)
        return filtered_df.iloc[:, 0]  # Convert back to Series

    def get_data(self, start_date: Union[str, datetime] = None,
                 end_date: Union[str, datetime] = None,
                 symbols: Union[str, List[str]] = None,
                 liquidity_config: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        Get historical data for symbols.

        Args:
            start_date: Start date for data (uses stored start_date if None)
            end_date: End date for data (default: today)
            symbols: Symbols to get data for (uses stored symbols if None)
            liquidity_config: Liquidity filtering configuration (overrides instance config)

        Returns:
            Tuple of (successful_data_dict, failed_symbols_list)
        """
        # Use defaults from constructor if not provided
        if symbols is None:
            if self.symbols is None:
                raise ValueError("No symbols provided and no default symbols available")
            symbols = self.symbols

        if start_date is None:
            if self.start_date is None:
                raise ValueError("No start_date provided and no default start_date available")
            start_date = self.start_date

        return self.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            liquidity_config=liquidity_config
        )

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