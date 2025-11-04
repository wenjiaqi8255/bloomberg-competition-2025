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
from .cache.stock_data_cache import StockDataCache

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
                 exchange_suffix_map: Optional[Dict[str, str]] = None,
                 cache_dir: str = "./cache/stock_data",
                 enable_disk_cache: bool = True,
                 cache_version: str = "v1"):
        """
        Initialize the YFinance provider.

        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            request_timeout: Request timeout in seconds
            cache_enabled: Whether to enable in-memory caching (L1 cache)
            symbols: List of symbols to provide data for (optional)
            start_date: Start date for historical data (optional)
            liquidity_config: Liquidity filtering configuration
            cache_dir: Directory for persistent disk cache (L2 cache)
            enable_disk_cache: Whether to enable persistent disk caching
            cache_version: Cache version for compatibility checking
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

        # Initialize persistent disk cache (L2 cache)
        self.enable_disk_cache = enable_disk_cache
        self.disk_cache: Optional[StockDataCache] = None
        if enable_disk_cache:
            self.disk_cache = StockDataCache(cache_dir=cache_dir, cache_version=cache_version)
            logger.info(f"Disk cache enabled at {cache_dir}")
        else:
            logger.info("Disk cache disabled")

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
                    # Step 1: Check L1 cache (in-memory cache)
                    cache_key = self._get_cache_key(resolved, start_date, end_date, period)
                    cached_data = self._get_from_cache(cache_key)
                    if cached_data is not None:
                        results[symbol] = cached_data
                        self._symbol_resolution_cache[f"{symbol.strip().upper()}|{(country_code or '').strip().upper()}"] = resolved
                        logger.debug(f"L1 cache hit for {symbol} via {resolved}")
                        fetch_success = True
                        break

                    # Step 2: Check L2 cache (disk cache) if enabled
                    cached_data = None
                    missing_ranges = []
                    if self.enable_disk_cache and self.disk_cache is not None:
                        # Convert start_date and end_date to datetime if they're strings
                        req_start = pd.to_datetime(start_date) if isinstance(start_date, str) else start_date
                        req_end = pd.to_datetime(end_date) if isinstance(end_date, str) else end_date
                        
                        # Try to get cached data (already filtered to requested range)
                        cached_data = self.disk_cache.get(resolved, start_date=req_start, end_date=req_end)
                        
                        if cached_data is not None and not cached_data.empty:
                            # ✅ KISS: Directly check if cached_data covers the requested range
                            cached_start = cached_data.index.min()
                            cached_end = cached_data.index.max()
                            
                            # ✅ Simple check: does cached_data cover the entire requested range?
                            if cached_start <= req_start and cached_end >= req_end:
                                # All data is in cache - complete hit
                                logger.debug(f"L2 cache hit (complete) for {symbol} via {resolved}")
                                cached_data = self._validate_and_clean_data(cached_data, resolved)
                                cached_data = self.add_data_source_metadata(cached_data)
                                results[symbol] = cached_data
                                self._store_in_cache(cache_key, cached_data)
                                self._symbol_resolution_cache[f"{symbol.strip().upper()}|{(country_code or '').strip().upper()}"] = resolved
                                fetch_success = True
                                break
                            else:
                                # Partial cache hit - determine what's missing
                                missing_ranges = []
                                if cached_start > req_start:
                                    missing_ranges.append((req_start, cached_start - pd.Timedelta(days=1)))
                                if cached_end < req_end:
                                    missing_ranges.append((cached_end + pd.Timedelta(days=1), req_end))
                                
                                if missing_ranges:
                                    logger.info(f"L2 cache hit (partial) for {symbol}: need to fetch {len(missing_ranges)} missing range(s)")
                                else:
                                    # This shouldn't happen, but handle it gracefully
                                    logger.debug(f"L2 cache hit (complete) for {symbol} via {resolved}")
                                    cached_data = self._validate_and_clean_data(cached_data, resolved)
                                    cached_data = self.add_data_source_metadata(cached_data)
                                    results[symbol] = cached_data
                                    self._store_in_cache(cache_key, cached_data)
                                    self._symbol_resolution_cache[f"{symbol.strip().upper()}|{(country_code or '').strip().upper()}"] = resolved
                                    fetch_success = True
                                    break
                    
                    # Step 3: Fetch missing data from network
                    # Initialize fetched_data_list here (after Step 2, before Step 3)
                    fetched_data_list = []
                    
                    if missing_ranges:
                        # Fetch only missing ranges
                        for missing_start, missing_end in missing_ranges:
                            logger.debug(f"Fetching missing range for {symbol}: {missing_start} to {missing_end}")
                            if period:
                                # For period-based requests, fetch the full period
                                data = self._fetch_with_retry(
                                    yf.download, resolved, period=period,
                                    progress=False, timeout=self.request_timeout
                                )
                            else:
                                # yfinance uses end as an open interval; extend by +1 day and skip empty windows
                                adj_end = missing_end + pd.Timedelta(days=1)
                                if missing_start >= adj_end:
                                    logger.debug(f"Skip empty missing range for {symbol}: {missing_start} to {missing_end}")
                                    continue
                                data = self._fetch_with_retry(
                                    yf.download, resolved, start=missing_start, end=adj_end,
                                    progress=False, timeout=self.request_timeout
                                )
                            if data is not None and not data.empty:
                                fetched_data_list.append(data)
                            else:
                                logger.warning(f"Failed to fetch missing range for {symbol}: {missing_start} to {missing_end}")
                    elif not cached_data or cached_data.empty:
                        # No cache at all, fetch full range
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
                            fetched_data_list.append(data)

                    # Step 4: Combine cached and fetched data
                    if cached_data is not None and not cached_data.empty:
                        # Merge cached data with newly fetched data
                        # Normalize fetched data before merging to ensure consistent column format
                        normalized_fetched_list = []
                        if fetched_data_list:
                            for fetched_data in fetched_data_list:
                                # Normalize fetched data to ensure same column structure as cached data
                                normalized_fetched = self._normalize_yfinance_data(fetched_data, resolved)
                                normalized_fetched_list.append(normalized_fetched)
                        
                        # Ensure all data has same columns before merging
                        all_data = [cached_data]
                        all_data.extend(normalized_fetched_list if normalized_fetched_list else fetched_data_list)
                        
                        # Align columns to avoid MultiIndex issues when concatenating
                        common_columns = set(cached_data.columns)
                        for df in all_data[1:]:
                            common_columns = common_columns.intersection(set(df.columns))
                        
                        # Filter to common columns only
                        if common_columns:
                            all_data = [df[[col for col in df.columns if col in common_columns]] for df in all_data]
                        
                        data = pd.concat(all_data, axis=0)
                        # Remove duplicates (keep last occurrence, which is newer data)
                        data = data[~data.index.duplicated(keep='last')]
                        data = data.sort_index()
                        logger.debug(f"Merged cached and fetched data for {symbol}: {len(cached_data)} cached + {sum(len(d) for d in fetched_data_list)} fetched = {len(data)} total")
                    elif fetched_data_list:
                        # Only fetched data
                        data = pd.concat(fetched_data_list, axis=0) if len(fetched_data_list) > 1 else fetched_data_list[0]
                        data = data[~data.index.duplicated(keep='last')]
                        data = data.sort_index()
                    else:
                        data = None

                    if data is not None and not data.empty:
                        before_clean_rows = len(data)
                        before_clean_end = data.index.max()
                        
                        # Data validation and cleaning using base class method
                        data = self._validate_and_clean_data(data, resolved)
                        
                        after_clean_rows = len(data)
                        after_clean_end = data.index.max() if not data.empty else None
                        
                        if before_clean_rows != after_clean_rows:
                            logger.warning(f"Data rows changed during cleaning for {symbol}: {before_clean_rows} -> {after_clean_rows}")
                        if before_clean_end and after_clean_end and before_clean_end > after_clean_end:
                            logger.error(f"⚠️ CRITICAL: Data end date reduced for {symbol} during cleaning: {before_clean_end} -> {after_clean_end}")
                        
                        data = self.add_data_source_metadata(data)

                        # Apply liquidity filtering if configured
                        effective_liquidity_config = liquidity_config or self.liquidity_config
                        if effective_liquidity_config and effective_liquidity_config.get('enabled', False):
                            data = self.apply_liquidity_filter(data, effective_liquidity_config)

                        # Store in L1 cache (in-memory)
                        self._store_in_cache(cache_key, data)

                        # Store in L2 cache (disk) if enabled
                        if self.enable_disk_cache and self.disk_cache is not None:
                            try:
                                # Remove metadata columns before caching
                                data_to_cache = data.copy()
                                for col in ['DataSource', 'Provider', 'FetchTime', 'Symbol']:
                                    if col in data_to_cache.columns:
                                        data_to_cache = data_to_cache.drop(columns=[col])
                                self.disk_cache.set(resolved, data_to_cache, merge=True)
                                logger.debug(f"Stored {symbol} in L2 cache")
                            except Exception as e:
                                logger.warning(f"Failed to store {symbol} in L2 cache: {e}")

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
        elif hasattr(data.columns, 'nlevels') and data.columns.nlevels > 1:
            logger.debug(f"Normalizing multi-symbol data for {symbol}")
            # For multi-symbol data, extract the specific symbol
            if symbol in data.columns.get_level_values(1):
                # Create single symbol DataFrame
                symbol_data = pd.DataFrame()
                for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']:
                    if (col, symbol) in data.columns:
                        symbol_data[col] = data[(col, symbol)]
                data = symbol_data
            else:
                logger.warning(f"Symbol {symbol} not found in multi-symbol columns")

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

            # CLEAN DATA FIRST before validation
            # Check for anomalous values and fix them
            for col in ['Open', 'High', 'Low', 'Close']:
                if col in data.columns:
                    # Negative prices are invalid
                    negative_prices = data[data[col] < 0]
                    if not negative_prices.empty:
                        logger.warning(f"Found {len(negative_prices)} negative prices in {col} for {symbol}")
                        data.loc[data[col] < 0, col] = pd.NA

            # Fix High-Low relationships BEFORE validation
            invalid_hl = data[data['High'] < data['Low']]
            if not invalid_hl.empty:
                logger.warning(f"Found {len(invalid_hl)} invalid High-Low pairs for {symbol}")
                # Fix by swapping values
                data.loc[invalid_hl.index, ['High', 'Low']] = \
                    data.loc[invalid_hl.index, ['Low', 'High']].values

            # Fix OHLC relationships BEFORE validation
            invalid_ohlc = data[
                (data['High'] < data['Open']) |
                (data['High'] < data['Close']) |
                (data['Low'] > data['Open']) |
                (data['Low'] > data['Close'])
            ]
            if not invalid_ohlc.empty:
                logger.warning(f"Found {len(invalid_ohlc)} invalid OHLC relationships for {symbol}, fixing...")
                # Fix by adjusting high/low
                data.loc[invalid_ohlc.index, 'High'] = data.loc[invalid_ohlc.index, ['Open', 'Close', 'High']].max(axis=1)
                data.loc[invalid_ohlc.index, 'Low'] = data.loc[invalid_ohlc.index, ['Open', 'Close', 'Low']].min(axis=1)

            # Remove rows with NaN values in critical columns after cleaning
            before_drop = len(data)
            before_drop_end = data.index.max() if not data.empty else None
            
            data = data.dropna(subset=['Open', 'High', 'Low', 'Close'])
            
            after_drop = len(data)
            after_drop_end = data.index.max() if not data.empty else None
            
            if after_drop < before_drop:
                logger.warning(f"Dropped {before_drop - after_drop} rows with NaN values for {symbol}")
                if before_drop_end and after_drop_end and before_drop_end > after_drop_end:
                    logger.error(f"⚠️ CRITICAL: Data end date reduced for {symbol} from {before_drop_end} to {after_drop_end}!")

            # NOW validate the cleaned data
            # Use base class validation for price data
            self.validate_price_data(data, symbol)

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

    def clear_stock_cache(self, symbol: Optional[str] = None) -> None:
        """
        Clear stock data cache.

        Args:
            symbol: If specified, clear cache for this symbol only;
                   otherwise clear all cached stock data
        """
        # Clear L1 cache (in-memory)
        if symbol:
            # Clear cache entries for this symbol
            keys_to_remove = [
                key for key in self._cache.keys()
                if symbol.upper() in key.upper()
            ]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Cleared L1 cache for {symbol}")
        else:
            self.clear_cache()
            logger.info("Cleared all L1 cache")

        # Clear L2 cache (disk)
        if self.enable_disk_cache and self.disk_cache is not None:
            self.disk_cache.clear(symbol)
            if symbol:
                logger.info(f"Cleared L2 cache for {symbol}")
            else:
                logger.info("Cleared all L2 cache")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics for both L1 and L2 caches
        """
        stats = {
            'l1_cache': self.get_cache_info(),
            'l2_cache_enabled': self.enable_disk_cache
        }

        if self.enable_disk_cache and self.disk_cache is not None:
            stats['l2_cache'] = self.disk_cache.get_cache_stats()
        else:
            stats['l2_cache'] = None

        return stats

    def warmup_cache(
        self,
        symbols: Union[str, List[str]],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime] = None
    ) -> Dict[str, Any]:
        """
        Warm up the cache by pre-fetching data for symbols.

        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data (default: today)

        Returns:
            Dictionary with warmup results
        """
        if isinstance(symbols, str):
            symbols = [symbols]

        if end_date is None:
            end_date = datetime.now()

        logger.info(f"Warming up cache for {len(symbols)} symbols "
                   f"from {start_date} to {end_date}")

        results = self.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date
        )

        return {
            'requested_symbols': len(symbols),
            'cached_symbols': len(results),
            'success_rate': len(results) / len(symbols) if symbols else 0,
            'cached_symbol_list': list(results.keys())
        }

    def clear_invalid_cache(self) -> int:
        """
        Clear invalid cache files.

        Returns:
            Number of invalid cache files cleared
        """
        if self.enable_disk_cache and self.disk_cache is not None:
            return self.disk_cache.clear_invalid_cache()
        return 0