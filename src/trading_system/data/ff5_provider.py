"""
FF5 (Fama-French 5-Factor) data provider from Kenneth French Data Library.

This module provides a robust interface to fetch Fama-French 5-factor data:
- MKT: Market excess returns
- SMB: Small Minus Big (size factor)
- HML: High Minus Low (value factor)
- RMW: Robust Minus Weak (profitability factor)
- CMA: Conservative Minus Aggressive (investment factor)
- RF: Risk-free rate

Data source: Kenneth French Data Library (Dartmouth College)
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import requests
from io import StringIO

from ..types.enums import DataSource
from ..utils.validation import DataValidator, DataValidationError
from .base_data_provider import FactorDataProvider

logger = logging.getLogger(__name__)


class FF5DataProvider(FactorDataProvider):
    """
    Fama-French 5-Factor data provider.

    Fetches factor data from Kenneth French Data Library with:
    - Automatic data cleaning and validation
    - Robust error handling
    - Date alignment capabilities
    - Multiple frequency support (daily, monthly)
    - Local caching for performance
    """

    BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/"

    def __init__(self, data_frequency: str = "monthly", cache_dir: str = None,
                 max_retries: int = 3, retry_delay: float = 1.0,
                 request_timeout: int = 30, cache_enabled: bool = True,
                 file_path: str = None):
        """
        Initialize FF5 data provider.

        Args:
            data_frequency: Data frequency ("daily" or "monthly")
            cache_dir: Directory for caching data (optional)
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            request_timeout: Request timeout in seconds
            cache_enabled: Whether to enable caching
            file_path: Path to local FF5 data file (optional)
        """
        super().__init__(
            max_retries=max_retries,
            retry_delay=retry_delay,
            request_timeout=request_timeout,
            cache_enabled=cache_enabled,
            rate_limit=1.0  # 1 second between requests for Kenneth French
        )

        self.data_frequency = data_frequency.lower()
        self.cache_dir = cache_dir
        self.file_path = file_path

        if self.data_frequency not in ["daily", "monthly"]:
            raise ValueError("data_frequency must be 'daily' or 'monthly'")

        logger.info(f"Initialized FF5 provider with {data_frequency} data")

    def get_data_source(self) -> DataSource:
        """Get the data source enum for this provider."""
        return DataSource.KENNETH_FRENCH
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about this data provider."""
        return {
            'provider': 'Kenneth French Data Library',
            'data_source': DataSource.KENNETH_FRENCH.value,
            'description': 'Fama-French 5-factor model data',
            'data_frequency': self.data_frequency,
            'factors': ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF'],
            'base_url': self.BASE_URL,
            'update_frequency': 'Daily/Monthly (depends on source)',
            'units': 'Decimal (converted from percentage)',
            'cache_enabled': self.cache_enabled,
            'cached_datasets': len(self._cache)
        }
    
    def _fetch_raw_data(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """Fetch raw data from Kenneth French Data Library."""
        # This method is called by the base class's _fetch_with_retry
        # The actual fetching logic is in the specific methods
        pass

    def get_factor_returns(self, start_date: Union[str, datetime] = None,
                           end_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        Get Fama-French 5-factor returns.

        Args:
            start_date: Start date for factor data
            end_date: End date for factor data

        Returns:
            DataFrame with columns: ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            Index: Dates
        """
        try:
            # Check cache first
            cache_key = self._get_cache_key("factor_returns", start_date, end_date, self.data_frequency)
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data

            # Determine data file based on frequency
            if self.data_frequency == "daily":
                file_url = f"{self.BASE_URL}ftp/F-F_Research_Data_5_Factors_2x3_daily_TXT.zip"
                filename = "F-F_Research_Data_5_Factors_2x3_daily.txt"
            else:
                file_url = f"{self.BASE_URL}ftp/F-F_Research_Data_5_Factors_2x3_TXT.zip"
                filename = "F-F_Research_Data_5_Factors_2x3.txt"

            # Fetch and parse data with retry logic
            raw_data = self._fetch_with_retry(
                self._fetch_and_parse_data, file_url, filename
            )

            if raw_data is None:
                raise DataValidationError("Failed to fetch FF5 data")

            # Clean and validate data using base class method
            clean_data = self.validate_factor_data(raw_data)

            # Filter by date range using base class method
            if start_date or end_date:
                clean_data = self.filter_by_date(clean_data, start_date, end_date)

            # Add data source metadata
            clean_data = self.add_data_source_metadata(clean_data)

            # Store in cache
            self._store_in_cache(cache_key, clean_data)

            logger.info(f"Retrieved {len(clean_data)} rows of {self.data_frequency} FF5 data")
            return clean_data

        except Exception as e:
            logger.error(f"Failed to get FF5 factor returns: {e}")
            raise DataValidationError(f"FF5 data fetch failed: {e}")

    def get_risk_free_rate(self, start_date: Union[str, datetime] = None,
                          end_date: Union[str, datetime] = None) -> pd.Series:
        """
        Get risk-free rate from FF5 data.

        Args:
            start_date: Start date for RF data
            end_date: End date for RF data

        Returns:
            Series with risk-free rates
        """
        factor_data = self.get_factor_returns(start_date, end_date)
        return factor_data['RF']

    def get_factor_descriptions(self) -> Dict[str, str]:
        """Get descriptions of FF5 factors."""
        return {
            'MKT': 'Market excess return (Market return - Risk-free rate)',
            'SMB': 'Small Minus Big (size factor: small cap returns - large cap returns)',
            'HML': 'High Minus Low (value factor: high B/M - low B/M)',
            'RMW': 'Robust Minus Weak (profitability factor: robust profits - weak profits)',
            'CMA': 'Conservative Minus Aggressive (investment factor: conservative investment - aggressive investment)',
            'RF': 'Risk-free rate (usually 1-month T-bill rate)'
        }

    def get_factor_statistics(self, factor_data: pd.DataFrame = None) -> Dict:
        """
        Calculate summary statistics for factors.

        Args:
            factor_data: Optional factor data (will fetch if not provided)

        Returns:
            Dictionary with factor statistics
        """
        if factor_data is None:
            factor_data = self.get_factor_returns()

        stats = {}
        for factor in factor_data.columns:
            series = factor_data[factor]
            stats[factor] = {
                'mean': series.mean(),
                'std': series.std(),
                'min': series.min(),
                'max': series.max(),
                'annualized_mean': series.mean() * (252 if self.data_frequency == "daily" else 12),
                'annualized_std': series.std() * np.sqrt(252 if self.data_frequency == "daily" else 12),
                'sharpe_ratio': (series.mean() / series.std()) * np.sqrt(252 if self.data_frequency == "daily" else 12),
                'skewness': series.skew(),
                'kurtosis': series.kurtosis()
            }

        return stats

    def align_with_equity_data(self, equity_data: Dict[str, pd.DataFrame],
                             factor_data: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Align FF5 factor data with equity data dates.

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

    def _fetch_and_parse_data(self, file_url: str, filename: str) -> pd.DataFrame:
        """Fetch and parse raw FF5 data from web."""
        try:
            import zipfile
            import io

            # Check cache first
            cache_key = f"{filename}_{self.data_frequency}"
            if cache_key in self._cache:
                logger.debug(f"Using cached data for {filename}")
                return self._cache[cache_key].copy()

            logger.info(f"Fetching FF5 data from {file_url}")

            # For Kenneth French data, we need to handle the zip file format
            response = requests.get(file_url)
            response.raise_for_status()

            # Extract zip file
            zip_file = zipfile.ZipFile(io.BytesIO(response.content))
            files = zip_file.namelist()

            if not files:
                raise ValueError("No files found in FF5 zip archive")

            # Read the content of the first file
            content = zip_file.read(files[0]).decode('utf-8', errors='ignore')
            lines = content.split('\n')

            # Find the start of the data (look for the header line)
            data_start = 0
            header_found = False
            for i, line in enumerate(lines):
                if 'Mkt-RF' in line and 'SMB' in line and 'HML' in line:
                    data_start = i + 1
                    header_found = True
                    break

            if not header_found:
                raise ValueError("Could not find FF5 data header in file")

            # Extract data lines
            data_lines = lines[data_start:]
            if not data_lines:
                raise ValueError("No valid data found in FF5 file")

            # Parse data
            data = []
            for line in data_lines:
                if line.strip():
                    # Clean up the line and split
                    clean_line = line.strip().replace(',', ' ')
                    parts = [x for x in clean_line.split() if x]

                    if len(parts) >= 6:
                        try:
                            # Parse date (format varies)
                            date_str = parts[0]
                            if self.data_frequency == "daily":
                                # Daily format: YYYYMMDD
                                date = pd.to_datetime(date_str, format='%Y%m%d')
                            else:
                                # Monthly format: YYYYMM
                                date = pd.to_datetime(date_str, format='%Y%m')

                            # Parse factor values (Mkt-RF, SMB, HML, RMW, CMA, RF)
                            factor_values = [float(x) for x in parts[1:7]]

                            row = [date] + factor_values
                            data.append(row)
                        except Exception as e:
                            logger.debug(f"Failed to parse line: {line} - {e}")
                            continue

            if not data:
                raise ValueError("No valid data rows parsed")

            # Create DataFrame
            df = pd.DataFrame(data, columns=['Date', 'MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF'])
            df.set_index('Date', inplace=True)

            # Convert percentages to decimals
            for col in df.columns:
                df[col] = df[col] / 100.0

            # Cache the data
            self._cache[cache_key] = df.copy()

            logger.info(f"Successfully parsed {len(df)} rows of FF5 data")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch and parse FF5 data: {e}")
            raise

    def _clean_factor_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate factor data."""
        try:
            # Make a copy
            data = raw_data.copy()

            # Remove any remaining non-numeric rows
            data = data.dropna()

            # Validate required columns
            required_cols = ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF']
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Check for reasonable value ranges
            for col in required_cols:
                values = data[col]
                # Factor returns should be within reasonable bounds
                if col == 'RF':  # Risk-free rate
                    mask = (values >= -0.1) & (values <= 0.1)  # -10% to +10%
                else:  # Factor returns
                    mask = (values >= -1.0) & (values <= 1.0)  # -100% to +100%

                outliers = values[~mask]
                if not outliers.empty:
                    logger.warning(f"Found {len(outliers)} outliers in {col}, filtering them out")
                    data = data[mask]

            # Sort by date
            data = data.sort_index()

            # Fill any remaining NaN values with forward fill
            data = data.ffill().bfill()

            # Use base class validation
            data = self.validate_factor_data(data)

            return data

        except Exception as e:
            logger.error(f"Failed to clean FF5 data: {e}")
            raise DataValidationError(f"Data cleaning failed: {e}")


    def get_latest_factors(self) -> pd.Series:
        """Get the latest available factor values."""
        factor_data = self.get_factor_returns()
        return factor_data.iloc[-1]

    def get_data(self, start_date: Union[str, datetime] = None,
                 end_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        Get FF5 factor data for the specified date range.

        Args:
            start_date: Start date for data (optional)
            end_date: End date for data (optional)

        Returns:
            DataFrame with factor returns data
        """
        return self.get_factor_returns(start_date=start_date, end_date=end_date)

    def get_factor_correlations(self, factor_data: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate factor return correlations."""
        if factor_data is None:
            factor_data = self.get_factor_returns()

        return factor_data.corr()

    def get_cumulative_factor_returns(self, factor_data: pd.DataFrame = None) -> pd.DataFrame:
        """Calculate cumulative factor returns."""
        if factor_data is None:
            factor_data = self.get_factor_returns()

        return (1 + factor_data).cumprod() - 1

    def get_data_info(self) -> Dict:
        """Get information about the FF5 data provider."""
        return {
            'provider': 'Kenneth French Data Library',
            'data_frequency': self.data_frequency,
            'factors': ['MKT', 'SMB', 'HML', 'RMW', 'CMA', 'RF'],
            'base_url': self.BASE_URL,
            'data_source': DataSource.KENNETH_FRENCH.value,
            'description': 'Fama-French 5-factor model data',
            'update_frequency': 'Daily/Monthly (depends on source)',
            'units': 'Decimal (converted from percentage)',
            'cache_enabled': self.cache_dir is not None,
            'cached_datasets': len(self._cache)
        }