"""
Offline Stock Metadata Provider
================================

Provides fast access to stock metadata (Market Cap, P/B ratio) from CSV files,
avoiding expensive API calls during classification and portfolio construction.

This provider implements a fallback strategy:
1. First, try to get data from CSV file
2. If not found in CSV, fall back to yfinance API
3. Cache results to minimize repeated lookups
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)


class OfflineStockMetadataProvider:
    """
    Provides stock metadata (Market Cap, P/B ratio) from CSV files.
    
    This provider loads data from a CSV file once at initialization,
    then provides fast batch lookups without repeated file I/O.
    """
    
    def __init__(self, csv_path: str, fallback_provider=None):
        """
        Initialize offline metadata provider.
        
        Args:
            csv_path: Path to CSV file containing stock metadata
            fallback_provider: Optional provider for fallback (e.g., yfinance)
        """
        self.csv_path = Path(csv_path)
        self.fallback_provider = fallback_provider
        self._data: Optional[pd.DataFrame] = None
        self._cache: Dict[str, Dict[str, float]] = {}
        
        # Column name mappings (handle different CSV formats)
        self.ticker_column = None
        self.market_cap_column = None
        self.pb_ratio_column = None
        
        self._load_data()
        logger.info(f"Initialized OfflineStockMetadataProvider with {len(self._data)} records")
    
    def _load_data(self) -> None:
        """Load data from CSV file."""
        if not self.csv_path.exists():
            logger.warning(f"CSV file not found: {self.csv_path}. Offline provider will use fallback only.")
            self._data = pd.DataFrame()
            return
        
        try:
            self._data = pd.read_csv(self.csv_path)
            logger.info(f"Loaded {len(self._data)} records from {self.csv_path}")
            
            # Identify column names (flexible matching)
            self._identify_columns()
            
        except Exception as e:
            logger.error(f"Failed to load CSV file {self.csv_path}: {e}")
            self._data = pd.DataFrame()
    
    def _identify_columns(self) -> None:
        """Identify relevant columns in the CSV file."""
        if self._data is None or self._data.empty:
            return
        
        # Create a case-insensitive mapping of column names
        columns_lower = {col.lower(): col for col in self._data.columns}
        
        # Find ticker column (case-insensitive)
        ticker_candidates = ['ticker_clean', 'ticker', 'symbol', 'ticker_original']
        for col_lower in ticker_candidates:
            if col_lower in columns_lower:
                self.ticker_column = columns_lower[col_lower]
                logger.debug(f"Identified ticker column: {self.ticker_column}")
                break
        
        # Find market cap column (case-insensitive, handle spaces)
        market_cap_candidates = ['market cap _usd_', 'market cap _usd', 'market cap', 'market_cap', 'marketcap', 'marketcap_usd']
        for col_lower in market_cap_candidates:
            if col_lower in columns_lower:
                self.market_cap_column = columns_lower[col_lower]
                logger.debug(f"Identified market cap column: {self.market_cap_column}")
                break
        
        # Find P/B ratio column (case-insensitive)
        pb_candidates = ['p_b', 'pb', 'p/b', 'price_to_book', 'pb_ratio', 'price to book']
        for col_lower in pb_candidates:
            if col_lower in columns_lower:
                self.pb_ratio_column = columns_lower[col_lower]
                logger.debug(f"Identified P/B ratio column: {self.pb_ratio_column}")
                break
        
        if not self.ticker_column:
            logger.warning("Could not identify ticker column in CSV file")
            logger.debug(f"Available columns: {list(self._data.columns)}")
        if not self.market_cap_column:
            logger.warning("Could not identify market cap column in CSV file")
            logger.debug(f"Available columns: {list(self._data.columns)}")
        if not self.pb_ratio_column:
            logger.warning("Could not identify P/B ratio column in CSV file")
            logger.debug(f"Available columns: {list(self._data.columns)}")
    
    def get_market_caps_batch(self, symbols: List[str], date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Get market caps for multiple symbols in batch.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'MSFT'])
            date: Optional date (for future time-series support, currently ignored)
            
        Returns:
            Dictionary mapping symbols to market cap values (in billions USD)
        """
        result = {}
        missing_symbols = []
        
        # Check cache first
        for symbol in symbols:
            cache_key = f"market_cap_{symbol}"
            if cache_key in self._cache:
                result[symbol] = self._cache[cache_key]['market_cap']
            else:
                missing_symbols.append(symbol)
        
        # Look up missing symbols from CSV
        if missing_symbols and self._data is not None and not self._data.empty:
            if self.ticker_column and self.market_cap_column:
                # Normalize symbols for matching (uppercase, remove spaces)
                normalized_symbols = {s.upper().strip(): s for s in missing_symbols}
                
                # Filter data for matching symbols
                for _, row in self._data.iterrows():
                    ticker = str(row[self.ticker_column]).upper().strip()
                    if ticker in normalized_symbols:
                        original_symbol = normalized_symbols[ticker]
                        market_cap = row[self.market_cap_column]
                        
                        # Convert to billions if needed (assume raw value is in USD)
                        if pd.notna(market_cap) and market_cap > 0:
                            # If value is very large (> 1e12), assume it's in raw USD, convert to billions
                            if market_cap > 1e12:
                                market_cap_billions = market_cap / 1e9
                            else:
                                # Assume already in billions
                                market_cap_billions = market_cap
                            
                            result[original_symbol] = float(market_cap_billions)
                            self._cache[f"market_cap_{original_symbol}"] = {'market_cap': market_cap_billions}
        
        # Use fallback provider for remaining missing symbols
        still_missing = [s for s in symbols if s not in result]
        if still_missing and self.fallback_provider:
            logger.debug(f"Using fallback provider for {len(still_missing)} symbols: {still_missing}")
            for symbol in still_missing:
                try:
                    # Try to get from fallback (this would need to be implemented in fallback provider)
                    # For now, we'll just log and return None
                    logger.debug(f"Fallback not yet implemented for {symbol}")
                except Exception as e:
                    logger.debug(f"Fallback failed for {symbol}: {e}")
        
        return result
    
    def get_pb_ratios_batch(self, symbols: List[str], date: Optional[datetime] = None) -> Dict[str, float]:
        """
        Get P/B ratios for multiple symbols in batch.
        
        Args:
            symbols: List of stock symbols (e.g., ['AAPL', 'MSFT'])
            date: Optional date (for future time-series support, currently ignored)
            
        Returns:
            Dictionary mapping symbols to P/B ratio values
        """
        result = {}
        missing_symbols = []
        
        # Check cache first
        for symbol in symbols:
            cache_key = f"pb_ratio_{symbol}"
            if cache_key in self._cache:
                result[symbol] = self._cache[cache_key]['pb_ratio']
            else:
                missing_symbols.append(symbol)
        
        # Look up missing symbols from CSV
        if missing_symbols and self._data is not None and not self._data.empty:
            if self.ticker_column and self.pb_ratio_column:
                # Normalize symbols for matching (uppercase, remove spaces)
                normalized_symbols = {s.upper().strip(): s for s in missing_symbols}
                
                # Filter data for matching symbols
                for _, row in self._data.iterrows():
                    ticker = str(row[self.ticker_column]).upper().strip()
                    if ticker in normalized_symbols:
                        original_symbol = normalized_symbols[ticker]
                        pb_ratio = row[self.pb_ratio_column]
                        
                        if pd.notna(pb_ratio) and pb_ratio > 0:
                            result[original_symbol] = float(pb_ratio)
                            self._cache[f"pb_ratio_{original_symbol}"] = {'pb_ratio': pb_ratio}
        
        # Use fallback provider for remaining missing symbols
        still_missing = [s for s in symbols if s not in result]
        if still_missing and self.fallback_provider:
            logger.debug(f"Using fallback provider for {len(still_missing)} symbols: {still_missing}")
        
        return result
    
    def get_metadata_batch(self, symbols: List[str], date: Optional[datetime] = None) -> Dict[str, Dict[str, float]]:
        """
        Get both market cap and P/B ratio for multiple symbols in a single call.
        
        Args:
            symbols: List of stock symbols
            date: Optional date
            
        Returns:
            Dictionary mapping symbols to dict with 'market_cap' and 'pb_ratio' keys
        """
        market_caps = self.get_market_caps_batch(symbols, date)
        pb_ratios = self.get_pb_ratios_batch(symbols, date)
        
        result = {}
        for symbol in symbols:
            result[symbol] = {
                'market_cap': market_caps.get(symbol),
                'pb_ratio': pb_ratios.get(symbol)
            }
        
        return result
    
    def clear_cache(self) -> None:
        """Clear the internal cache."""
        self._cache.clear()
        logger.debug("Cleared offline metadata cache")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cached_entries': len(self._cache),
            'csv_records': len(self._data) if self._data is not None else 0
        }

