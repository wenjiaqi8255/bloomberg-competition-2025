"""
Abstract base class for data providers in the trading system.

This module defines the common interface and shared functionality for all data providers,
following SOLID principles:
- Single Responsibility: Each provider has a single data source responsibility
- Open/Closed: Open for extension, closed for modification
- Liskov Substitution: All providers can be used interchangeably
- Interface Segregation: Focused interfaces for different data types
- Dependency Inversion: Depends on abstractions, not concretions
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np

from ..types.enums import DataSource
from .validation import DataValidator, DataValidationError
from .filters.liquidity_filter import LiquidityFilter

logger = logging.getLogger(__name__)


class BaseDataProvider(ABC):
    """
    Abstract base class for all data providers.
    
    Provides common functionality for:
    - Data validation and cleaning
    - Error handling and retry logic
    - Caching mechanisms
    - Rate limiting
    - Logging and monitoring
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 retry_delay: float = 1.0,
                 request_timeout: int = 30,
                 cache_enabled: bool = True,
                 rate_limit: float = 0.5):
        """
        Initialize the base data provider.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries in seconds
            request_timeout: Request timeout in seconds
            cache_enabled: Whether to enable caching
            rate_limit: Minimum time between requests in seconds
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout
        self.cache_enabled = cache_enabled
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self._cache = {}
        
        logger.info(f"Initialized {self.__class__.__name__} with "
                   f"max_retries={max_retries}, cache_enabled={cache_enabled}")
    
    @abstractmethod
    def get_data_source(self) -> DataSource:
        """Get the data source enum for this provider."""
        pass
    
    @abstractmethod
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about this data provider."""
        pass
    
    @abstractmethod
    def _fetch_raw_data(self, *args, **kwargs) -> Any:
        """
        Fetch raw data from the source.
        
        This method must be implemented by subclasses to define
        how they fetch data from their specific sources.
        """
        pass
    
    def _wait_for_rate_limit(self):
        """Implement rate limiting to avoid API restrictions."""
        if self.rate_limit <= 0:
            return
            
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit:
            sleep_time = self.rate_limit - time_since_last_request
            logger.debug(f"Rate limiting: waiting {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _fetch_with_retry(self, fetch_func, *args, **kwargs) -> Optional[Any]:
        """
        Execute fetch function with retry logic.
        
        Args:
            fetch_func: Function to execute
            *args, **kwargs: Arguments for the fetch function
            
        Returns:
            Result from fetch function or None if all retries fail
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                self._wait_for_rate_limit()
                
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{self.max_retries}")
                
                result = fetch_func(*args, **kwargs)
                
                if result is not None:
                    return result
                else:
                    logger.warning(f"Empty result returned on attempt {attempt + 1}")
                    last_error = "Empty result returned"
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    sleep_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
        
        logger.error(f"All {self.max_retries} attempts failed. Last error: {last_error}")
        return None
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate a cache key for the given arguments."""
        # Create a deterministic key from arguments
        key_parts = []
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, datetime):
                key_parts.append(arg.isoformat())
            elif isinstance(arg, list):
                key_parts.append(','.join(str(x) for x in arg))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
            elif isinstance(v, datetime):
                key_parts.append(f"{k}={v.isoformat()}")
        
        return f"{self.__class__.__name__}_{'_'.join(key_parts)}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if available."""
        if not self.cache_enabled or cache_key not in self._cache:
            return None
        
        cached_data, timestamp = self._cache[cache_key]
        
        # Check if cache is still valid (24 hours by default)
        if time.time() - timestamp > 86400:  # 24 hours
            del self._cache[cache_key]
            return None
        
        logger.debug(f"Using cached data for key: {cache_key}")
        return cached_data.copy() if hasattr(cached_data, 'copy') else cached_data
    
    def _store_in_cache(self, cache_key: str, data: Any):
        """Store data in cache."""
        if not self.cache_enabled:
            return
        
        self._cache[cache_key] = (data, time.time())
        logger.debug(f"Stored data in cache with key: {cache_key}")
    
    def validate_data(self, data: pd.DataFrame, data_type: str = "general",
                     liquidity_config: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Validate and clean data using the DataValidator.

        Args:
            data: DataFrame to validate
            data_type: Type of data for validation rules
            liquidity_config: Optional liquidity filtering configuration

        Returns:
            Validated and cleaned DataFrame

        Raises:
            DataValidationError: If data validation fails
        """
        try:
            if data is None or data.empty:
                raise DataValidationError("Data is None or empty")

            # Basic validation
            if data_type == "price":
                DataValidator.validate_price_data(data, "validation")
            elif data_type == "factor":
                DataValidator.validate_factor_data(data, "validation")
            else:
                # General validation
                if len(data) == 0:
                    raise DataValidationError("Data has no rows")

                # Check for all NaN columns
                all_nan_cols = data.columns[data.isnull().all()].tolist()
                if all_nan_cols:
                    logger.warning(f"Found all-NaN columns: {all_nan_cols}")
                    data = data.drop(columns=all_nan_cols)

            # Remove rows with all NaN values
            initial_len = len(data)
            data = data.dropna(how='all')
            if len(data) < initial_len:
                logger.warning(f"Removed {initial_len - len(data)} empty rows")

            # Ensure datetime index if applicable
            if not isinstance(data.index, pd.DatetimeIndex) and 'Date' in data.columns:
                data = data.set_index('Date')
                data.index = pd.to_datetime(data.index)

            # Sort by index
            data = data.sort_index()

            # Apply liquidity filtering if configured
            if liquidity_config and liquidity_config.get('enabled', False):
                data = self.apply_liquidity_filter(data, liquidity_config)

            return data

        except Exception as e:
            logger.error(f"Data validation failed: {e}")
            raise DataValidationError(f"Data validation failed: {e}")

    def apply_liquidity_filter(self, data: pd.DataFrame,
                              liquidity_config: Dict[str, Any]) -> pd.DataFrame:
        """
        Apply liquidity filtering to data using the LiquidityFilter utility.

        This method acts as a delegate to the LiquidityFilter utility class,
        providing a clean interface for data providers to apply liquidity
        filtering without duplicating logic.

        Args:
            data: DataFrame to filter (expects 'Symbol' column or multiple symbols)
            liquidity_config: Liquidity filter configuration

        Returns:
            Filtered DataFrame with only liquid symbols
        """
        try:
            # Extract symbols from data
            if 'Symbol' in data.columns:
                # Single symbol DataFrame with Symbol column
                symbols = data['Symbol'].unique().tolist()
                price_data = {}

                for symbol in symbols:
                    symbol_data = data[data['Symbol'] == symbol].copy()
                    if 'Symbol' in symbol_data.columns:
                        symbol_data = symbol_data.drop('Symbol', axis=1)
                    price_data[symbol] = symbol_data
            else:
                # Multi-symbol DataFrame (symbols as columns)
                symbols = [col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
                price_data = {}

                for symbol in symbols:
                    if symbol in data.columns:
                        # Create standard OHLCV DataFrame for this symbol
                        symbol_data = pd.DataFrame()
                        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                            if col in data.index.names or col in data.columns:
                                # Handle different data structures
                                if col in data.columns and not data[col].dropna().empty:
                                    symbol_data[col] = data[col]
                                elif hasattr(data, 'xs'):
                                    try:
                                        symbol_data[col] = data.xs(symbol, axis=1, level=0 if hasattr(data.columns, 'levels') else 0)[col]
                                    except (KeyError, AttributeError):
                                        continue

                        if not symbol_data.empty:
                            price_data[symbol] = symbol_data

            if not price_data:
                logger.warning("No valid price data found for liquidity filtering")
                return data

            # Apply liquidity filters using the utility class
            filtered_symbols = LiquidityFilter.apply_liquidity_filters(
                symbols, price_data, liquidity_config
            )

            # Filter the original DataFrame to keep only filtered symbols
            if 'Symbol' in data.columns:
                # Single symbol format
                filtered_data = data[data['Symbol'].isin(filtered_symbols)]
            else:
                # Multi-symbol format - keep only filtered symbol columns
                symbol_columns = [col for col in data.columns if col in filtered_symbols]
                non_symbol_columns = [col for col in data.columns if col not in symbols]
                filtered_data = data[non_symbol_columns + symbol_columns]

            logger.info(f"Liquidity filtering: {len(filtered_symbols)}/{len(symbols)} symbols retained")
            return filtered_data

        except Exception as e:
            logger.error(f"Liquidity filtering failed: {e}")
            # If filtering fails, return original data
            logger.warning("Returning original data due to filtering error")
            return data

    def filter_by_date(self, data: pd.DataFrame,
                       start_date: Union[str, datetime] = None,
                       end_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        Filter data by date range.
        
        Args:
            data: DataFrame to filter
            start_date: Start date for filtering
            end_date: End date for filtering
            
        Returns:
            Filtered DataFrame
        """
        if start_date is None and end_date is None:
            return data
        
        mask = pd.Series(True, index=data.index)
        
        if start_date is not None:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            mask = mask & (data.index >= start_date)
        
        if end_date is not None:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            mask = mask & (data.index <= end_date)
        
        filtered_data = data[mask]
        
        if len(filtered_data) == 0:
            logger.warning(f"No data found for date range {start_date} to {end_date}")
        
        return filtered_data
    
    def add_data_source_metadata(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add data source metadata to DataFrame.
        
        Args:
            data: DataFrame to add metadata to
            
        Returns:
            DataFrame with metadata columns
        """
        data = data.copy()
        data['DataSource'] = self.get_data_source().value
        data['Provider'] = self.__class__.__name__
        data['FetchTime'] = datetime.now()
        return data
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about the cache."""
        return {
            'cache_enabled': self.cache_enabled,
            'cache_size': len(self._cache),
            'cache_keys': list(self._cache.keys()),
            'memory_usage': sum(
                data.nbytes if hasattr(data, 'nbytes') else 0 
                for data, _ in self._cache.values()
            )
        }
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()
        logger.info("Cache cleared")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'request_timeout': self.request_timeout,
            'rate_limit': self.rate_limit,
            'cache_info': self.get_cache_info()
        }


class PriceDataProvider(BaseDataProvider):
    """
    Abstract base class for price data providers.
    
    Extends BaseDataProvider with price-specific functionality.
    """
    
    @abstractmethod
    def get_historical_data(self, symbols: Union[str, List[str]],
                           start_date: Union[str, datetime],
                           end_date: Union[str, datetime] = None,
                           **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Get historical price data for symbols.
        
        Args:
            symbols: Single symbol or list of symbols
            start_date: Start date for data
            end_date: End date for data
            **kwargs: Additional parameters
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        pass
    
    @abstractmethod
    def get_latest_price(self, symbols: Union[str, List[str]]) -> Dict[str, float]:
        """
        Get latest price for symbols.
        
        Args:
            symbols: Single symbol or list of symbols
            
        Returns:
            Dictionary mapping symbols to latest prices
        """
        pass
    
    def validate_price_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate price data specifically."""
        return self.validate_data(data, "price")


class FactorDataProvider(BaseDataProvider):
    """
    Abstract base class for factor data providers.
    
    Extends BaseDataProvider with factor-specific functionality.
    """
    
    @abstractmethod
    def get_factor_returns(self, start_date: Union[str, datetime] = None,
                          end_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        Get factor returns data.
        
        Args:
            start_date: Start date for factor data
            end_date: End date for factor data
            
        Returns:
            DataFrame with factor returns
        """
        pass
    
    def validate_factor_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate factor data specifically."""
        return self.validate_data(data, "factor")
    
    def align_with_equity_data(self, equity_data: Dict[str, pd.DataFrame],
                              factor_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Align factor data with equity data dates.
        
        Args:
            equity_data: Dictionary of equity price DataFrames
            factor_data: Optional factor data (will fetch if not provided)
            
        Returns:
            Tuple of (aligned_factor_data, aligned_equity_data)
        """
        if factor_data is None:
            factor_data = self.get_factor_returns()
        
        # Get all equity dates
        all_equity_dates = set()
        for symbol, data in equity_data.items():
            if data is not None and len(data) > 0:
                all_equity_dates.update(data.index)
        
        all_equity_dates = sorted(all_equity_dates)
        
        # Align factor data with equity dates
        aligned_factor_data = factor_data.reindex(all_equity_dates, method='ffill')
        
        # Align equity data with factor data
        aligned_equity_data = {}
        for symbol, data in equity_data.items():
            if data is not None and len(data) > 0:
                aligned_data = data.reindex(aligned_factor_data.index, method='ffill')
                aligned_equity_data[symbol] = aligned_data
        
        return aligned_factor_data, aligned_equity_data


class ClassificationProvider(BaseDataProvider):
    """
    Abstract base class for classification providers.
    
    Extends BaseDataProvider with classification-specific functionality.
    """
    
    @abstractmethod
    def classify_items(self, items: List[str], **kwargs) -> Dict[str, Any]:
        """
        Classify items into categories.
        
        Args:
            items: List of items to classify
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with classification results
        """
        pass
    
    @abstractmethod
    def get_classification_categories(self) -> Dict[str, List[str]]:
        """
        Get available classification categories.
        
        Returns:
            Dictionary mapping category types to available values
        """
        pass
