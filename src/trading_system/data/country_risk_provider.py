"""
Country Risk Premium Data Provider

Loads country risk premium data from Excel files and provides it as factor data.
Reuses existing FactorDataProvider architecture for caching, validation, and integration.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from ..types.enums import DataSource
from .validation import DataValidationError
from .base_data_provider import FactorDataProvider

logger = logging.getLogger(__name__)


class CountryRiskProvider(FactorDataProvider):
    """
    Country risk premium data provider.
    
    Loads static country risk data from Excel and broadcasts it as time-series factor data.
    Follows DRY principles by extending FactorDataProvider and reusing existing patterns.
    """
    
    def __init__(self, 
                 excel_path: str,
                 symbol_country_map: Dict[str, str],
                 cache_enabled: bool = True,
                 **kwargs):
        """
        Initialize country risk provider.
        
        Args:
            excel_path: Path to Excel file with country risk data
            symbol_country_map: Dictionary mapping stock symbols to countries
            cache_enabled: Whether to enable caching
            **kwargs: Other FactorDataProvider parameters
        """
        super().__init__(cache_enabled=cache_enabled, **kwargs)
        
        self.excel_path = Path(excel_path)
        self.symbol_country_map = symbol_country_map
        
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {excel_path}")
            
        # Load and clean country data
        self._country_data = self._load_country_data()
        
        logger.info(f"Initialized CountryRiskProvider with {len(self._country_data)} countries")
    
    def get_data_source(self) -> DataSource:
        """Get the data source enum for this provider."""
        return DataSource.EXCEL_FILE
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about this data provider."""
        return {
            'provider': 'Country Risk Premium Excel File',
            'data_source': DataSource.EXCEL_FILE.value,
            'description': 'Country risk premium data from Excel',
            'excel_path': str(self.excel_path),
            'countries': list(self._country_data['Country'].unique()),
            'symbols_mapped': len(self.symbol_country_map),
            'cache_enabled': self.cache_enabled
        }
    
    def _fetch_raw_data(self, *args, **kwargs) -> Optional[pd.DataFrame]:
        """Fetch raw data - returns the loaded country data."""
        return self._country_data.copy()
    
    def _load_country_data(self) -> pd.DataFrame:
        """Load and clean country data from Excel."""
        try:
            logger.info(f"Loading country risk data from {self.excel_path}")
            
            # Read Excel or CSV file
            if self.excel_path.suffix.lower() == '.csv':
                df = pd.read_csv(self.excel_path)
            else:
                df = pd.read_excel(self.excel_path)
            
            # Clean column names and convert percentages
            column_mapping = {
                'Country': 'Country',
                'Rating-based Default Spread': 'default_spread',
                'Total Equity Risk Premium': 'equity_risk_premium',
                'Final ERP': 'country_risk_premium',
                'Tax Rate': 'corporate_tax_rate',
                'CRP': 'country_risk_premium_raw'  # Alternative country risk premium
            }
            df = df.rename(columns=column_mapping)

            # Convert percentage columns to decimals (one-line conversion)
            # Note: Some columns may already be decimals, others may be percentages
            percentage_columns = ['default_spread', 'equity_risk_premium', 'country_risk_premium', 'corporate_tax_rate']

            # Handle both percentage and decimal formats
            for col in percentage_columns:
                if col in df.columns:
                    # Check if values are percentages (contain %) or already decimals
                    sample_value = df[col].iloc[0] if len(df) > 0 else None
                    if sample_value and isinstance(sample_value, str) and '%' in str(sample_value):
                        # Convert percentage format
                        df[col] = df[col].astype(str).str.replace('%', '').astype(float) / 100
                    else:
                        # Already decimal format, ensure it's float
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Convert from percentage to decimal if values > 1
                    if df[col].max() > 1:
                        df[col] = df[col] / 100

            # Fill missing values with 0 for risk premium columns
            for col in percentage_columns:
                if col in df.columns:
                    df[col] = df[col].fillna(0.0)
            
            # Validate required columns
            required_cols = ['Country', 'country_risk_premium']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Remove rows with missing country names
            df = df.dropna(subset=['Country'])
            
            logger.info(f"Loaded {len(df)} countries with columns: {list(df.columns)}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to load country data: {e}")
            raise DataValidationError(f"Country data loading failed: {e}")
    
    def get_factor_returns(self, 
                          start_date: Union[str, datetime] = None,
                          end_date: Union[str, datetime] = None) -> pd.DataFrame:
        """
        Get country risk premium factor data as time series.
        
        Broadcasts static country data to daily time series for all symbols.
        """
        try:
            cache_key = self._get_cache_key("country_risk_factors", start_date, end_date)
            cached_data = self._get_from_cache(cache_key)
            if cached_data is not None:
                return cached_data
            
            # Generate daily date range
            if start_date is None:
                start_date = datetime(2020, 1, 1)
            if end_date is None:
                end_date = datetime.now()
                
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Create factor data for each symbol
            factor_data = []
            for symbol, country in self.symbol_country_map.items():
                country_row = self._country_data[self._country_data['Country'] == country]
                if country_row.empty:
                    logger.warning(f"Country '{country}' not found for symbol '{symbol}'")
                    continue
                    
                country_data = country_row.iloc[0]
                
                for date in dates:
                    factor_data.append({
                        'date': date,
                        'symbol': symbol,
                        'country_risk_premium': country_data.get('country_risk_premium', 0),
                        'equity_risk_premium': country_data.get('equity_risk_premium', 0),
                        'default_spread': country_data.get('default_spread', 0),
                        'corporate_tax_rate': country_data.get('corporate_tax_rate', 0),
                        'moodys_rating': country_data.get('moodys_rating', ''),
                    })
            
            df = pd.DataFrame(factor_data)
            if df.empty:
                logger.warning("No factor data generated")
                return pd.DataFrame()
                
            df.set_index('date', inplace=True)
            
            # Add data source metadata using base class method
            df = self.add_data_source_metadata(df)
            
            # Store in cache
            self._store_in_cache(cache_key, df)
            
            logger.info(f"Generated country risk factor data: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to get country risk factor data: {e}")
            raise DataValidationError(f"Country risk factor data failed: {e}")
    
    def get_country_risk_premium(self, country: str) -> Dict[str, float]:
        """Get risk premium data for a specific country."""
        country_data = self._country_data[self._country_data['Country'] == country]
        
        if country_data.empty:
            raise ValueError(f"Country '{country}' not found in data")
        
        row = country_data.iloc[0]
        return {
            'country_risk_premium': row.get('country_risk_premium', 0),
            'equity_risk_premium': row.get('equity_risk_premium', 0),
            'default_spread': row.get('default_spread', 0),
            'corporate_tax_rate': row.get('corporate_tax_rate', 0),
            'moodys_rating': row.get('moodys_rating', '')
        }
    
    def get_symbol_country_risk_data(self) -> Dict[str, Dict[str, float]]:
        """
        Get country risk data mapped to symbols for feature engineering.

        Returns:
            Dictionary mapping symbols to their country risk data.
            Format: {symbol: {country_risk_premium: float, equity_risk_premium: float, ...}}
        """
        symbol_risk_data = {}

        for symbol, country in self.symbol_country_map.items():
            try:
                country_data = self._country_data[self._country_data['Country'] == country]
                if not country_data.empty:
                    country_row = country_data.iloc[0]
                    symbol_risk_data[symbol] = {
                        'country_risk_premium': country_row.get('country_risk_premium', 0.0),
                        'equity_risk_premium': country_row.get('equity_risk_premium', 0.0),
                        'default_spread': country_row.get('default_spread', 0.0),
                        'corporate_tax_rate': country_row.get('corporate_tax_rate', 0.0),
                        'country': country
                    }
                    logger.debug(f"Mapped {symbol} to {country} with risk data")
                else:
                    logger.warning(f"Country '{country}' not found for symbol '{symbol}'")
                    # Use default values
                    symbol_risk_data[symbol] = {
                        'country_risk_premium': 0.0,
                        'equity_risk_premium': 0.0,
                        'default_spread': 0.0,
                        'corporate_tax_rate': 0.0,
                        'country': country
                    }
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                # Use default values
                symbol_risk_data[symbol] = {
                    'country_risk_premium': 0.0,
                    'equity_risk_premium': 0.0,
                    'default_spread': 0.0,
                    'corporate_tax_rate': 0.0,
                    'country': country
                }

        logger.info(f"Generated country risk data for {len(symbol_risk_data)} symbols")
        return symbol_risk_data

    def get_symbol_country_mapping(self) -> Dict[str, str]:
        """Get the symbol to country mapping."""
        return self.symbol_country_map.copy()
