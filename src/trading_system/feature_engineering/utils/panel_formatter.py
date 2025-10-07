"""
Panel Data Formatter

Standardized panel data format utilities to ensure consistent data structure
across all components of the trading system.

Design Principles:
- Single source of truth for panel data format conventions
- Convention over configuration for index ordering
- Automatic validation and fixing of data format issues
- Support for both DataFrame and Series formats
"""

import pandas as pd
import numpy as np
from typing import Tuple, Union, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class PanelDataFormatter:
    """
    Standardized panel data formatter that enforces consistent index ordering
    and data structure across all trading system components.

    Convention:
    - MultiIndex order: ('date', 'symbol')
    - Data types: date = datetime, symbol = str
    - Validation: Comprehensive quality checks
    """

    # Standard index order convention
    STANDARD_INDEX_ORDER = ('date', 'symbol')

    # Required levels for panel data
    REQUIRED_LEVELS = {'date', 'symbol'}

    @staticmethod
    def ensure_panel_format(
        data: Union[pd.DataFrame, pd.Series],
        index_order: Optional[Tuple[str, str]] = None,
        validate: bool = True,
        auto_fix: bool = True,
        fix_missing_levels: bool = True
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Ensure data conforms to standard panel format.

        Args:
            data: Input DataFrame or Series with MultiIndex
            index_order: Desired index order (defaults to ('date', 'symbol'))
            validate: Whether to perform validation checks
            auto_fix: Whether to automatically fix common issues
            fix_missing_levels: Whether to attempt to fix missing index levels

        Returns:
            Data in standard panel format

        Raises:
            ValueError: If data cannot be fixed and validation fails
        """
        if index_order is None:
            index_order = PanelDataFormatter.STANDARD_INDEX_ORDER

        # Handle empty data
        if len(data) == 0:
            logger.warning("Empty data provided, returning as-is")
            return data

        # Ensure MultiIndex
        if not isinstance(data.index, pd.MultiIndex):
            data = PanelDataFormatter._ensure_multiindex(data, fix_missing_levels)

        # Check and fix required levels
        data = PanelDataFormatter._ensure_required_levels(
            data, index_order, fix_missing_levels
        )

        # Fix index order if needed
        data = PanelDataFormatter._fix_index_order(data, index_order, auto_fix)

        # Convert date level to datetime if needed
        data = PanelDataFormatter._fix_date_format(data, auto_fix)

        # Validate final format
        if validate:
            PanelDataFormatter.validate_panel_format(data, index_order)

        logger.debug(f"Panel data formatted successfully. Shape: {data.shape}, "
                    f"Index order: {data.index.names}")

        return data

    @staticmethod
    def validate_panel_format(
        data: Union[pd.DataFrame, pd.Series],
        expected_order: Optional[Tuple[str, str]] = None,
        strict: bool = False
    ) -> bool:
        """
        Validate that data conforms to panel format conventions.

        Args:
            data: Data to validate
            expected_order: Expected index order
            strict: Whether to raise errors on validation failures

        Returns:
            True if valid, False otherwise

        Raises:
            ValueError: If strict=True and validation fails
        """
        if expected_order is None:
            expected_order = PanelDataFormatter.STANDARD_INDEX_ORDER

        validation_errors = []

        # Check for MultiIndex
        if not isinstance(data.index, pd.MultiIndex):
            validation_errors.append("Data must have MultiIndex")
        else:
            # Check index levels
            if set(data.index.names) != set(expected_order):
                validation_errors.append(
                    f"Index levels {data.index.names} don't match expected {expected_order}"
                )

            # Check index order
            if list(data.index.names) != list(expected_order):
                validation_errors.append(
                    f"Index order {data.index.names} doesn't match expected {expected_order}"
                )

            # Check for empty levels
            for level_name in data.index.names:
                if data.index.get_level_values(level_name).isna().any():
                    validation_errors.append(f"Index level '{level_name}' contains NaN values")

        # Check data types
        if isinstance(data.index, pd.MultiIndex):
            date_level = expected_order[0]
            symbol_level = expected_order[1]

            date_values = data.index.get_level_values(date_level)
            if not pd.api.types.is_datetime64_any_dtype(date_values):
                try:
                    pd.to_datetime(date_values)
                except (ValueError, TypeError):
                    validation_errors.append(f"Date level '{date_level}' cannot be converted to datetime")

        # Handle validation errors
        if validation_errors:
            error_msg = f"Panel data validation failed: {'; '.join(validation_errors)}"
            if strict:
                raise ValueError(error_msg)
            else:
                logger.warning(error_msg)
                return False

        logger.debug("Panel data validation passed")
        return True

    @staticmethod
    def _ensure_multiindex(
        data: Union[pd.DataFrame, pd.Series],
        fix_missing_levels: bool
    ) -> Union[pd.DataFrame, pd.Series]:
        """Convert single index to MultiIndex if needed."""
        if isinstance(data.index, pd.MultiIndex):
            return data

        logger.info("Converting single index to MultiIndex")

        # Try to infer structure from data
        if hasattr(data, 'columns') and len(data.columns) > 0:
            # DataFrame with symbols as columns
            if all(isinstance(col, str) for col in data.columns):
                # Assume symbols are columns, dates are index
                data_stack = data.stack()
                data_stack.index.names = ['date', 'symbol']
                return data_stack
            else:
                # Try to create symbol level from column names
                if fix_missing_levels:
                    logger.warning("Cannot infer symbol level from column names, keeping single index")
                    return data
        else:
            # Series - need to add symbol level
            if fix_missing_levels and hasattr(data, 'name') and data.name:
                logger.info(f"Creating symbol level from series name: {data.name}")
                data_frame = data.to_frame()
                data_frame.columns = ['value']
                data_frame.index.names = ['date']
                data_frame['symbol'] = data.name
                data_frame = data_frame.set_index('symbol', append=True)
                return data_frame['value']

        return data

    @staticmethod
    def _ensure_required_levels(
        data: Union[pd.DataFrame, pd.Series],
        index_order: Tuple[str, str],
        fix_missing_levels: bool
    ) -> Union[pd.DataFrame, pd.Series]:
        """Ensure data has required index levels."""
        if not isinstance(data.index, pd.MultiIndex):
            return data

        current_levels = set(data.index.names)
        required_levels = set(index_order)

        if current_levels == required_levels:
            return data

        if not fix_missing_levels:
            raise ValueError(f"Missing required levels: {required_levels - current_levels}")

        logger.info(f"Fixing missing levels. Current: {current_levels}, Required: {required_levels}")

        # Add missing levels with default values
        data_copy = data.copy()

        for level_name in required_levels - current_levels:
            if level_name == 'symbol':
                # Add symbol level with default value
                data_copy.index = pd.MultiIndex.from_arrays([
                    data_copy.index.get_level_values(0) if len(data_copy.index.names) > 0 else range(len(data_copy)),
                    ['UNKNOWN'] * len(data_copy)
                ], names=[data_copy.index.names[0] if data_copy.index.names else 'date', 'symbol'])
            elif level_name == 'date':
                # Add date level with default values
                data_copy.index = pd.MultiIndex.from_arrays([
                    pd.date_range('2000-01-01', periods=len(data_copy)),
                    data_copy.index.get_level_values(0) if len(data_copy.index.names) > 0 else ['UNKNOWN'] * len(data_copy)
                ], names=['date', data_copy.index.names[0] if data_copy.index.names else 'symbol'])

        return data_copy

    @staticmethod
    def _fix_index_order(
        data: Union[pd.DataFrame, pd.Series],
        index_order: Tuple[str, str],
        auto_fix: bool
    ) -> Union[pd.DataFrame, pd.Series]:
        """Fix index order if needed."""
        if not isinstance(data.index, pd.MultiIndex):
            return data

        current_order = list(data.index.names)
        desired_order = list(index_order)

        if current_order == desired_order:
            return data

        if not auto_fix:
            raise ValueError(f"Index order {current_order} doesn't match expected {desired_order}")

        logger.info(f"Reordering index from {current_order} to {desired_order}")

        # Create new index with desired order
        new_index = data.index.reorder_levels(desired_order)
        data.index = new_index

        # Sort by index for consistency
        data = data.sort_index()

        return data

    @staticmethod
    def _fix_date_format(
        data: Union[pd.DataFrame, pd.Series],
        auto_fix: bool
    ) -> Union[pd.DataFrame, pd.Series]:
        """Fix date format if needed."""
        if not isinstance(data.index, pd.MultiIndex):
            return data

        date_level_name = data.index.names[0]  # Assume first level is date

        if date_level_name not in ['date', 'Date', 'DATE']:
            return data

        date_values = data.index.get_level_values(date_level_name)

        if pd.api.types.is_datetime64_any_dtype(date_values):
            return data

        if not auto_fix:
            raise ValueError(f"Date level '{date_level_name}' is not in datetime format")

        logger.info(f"Converting date level '{date_level_name}' to datetime")

        try:
            # Convert to datetime
            converted_dates = pd.to_datetime(date_values)

            # Create new index with converted dates
            new_levels = list(data.index.levels)
            date_level_idx = list(data.index.names).index(date_level_name)
            new_levels[date_level_idx] = converted_dates

            new_index = pd.MultiIndex.from_arrays(new_levels, names=data.index.names)
            data.index = new_index

        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert dates: {e}")

        return data

    @staticmethod
    def get_cross_section(
        data: Union[pd.DataFrame, pd.Series],
        date: Union[datetime, str, pd.Timestamp],
        index_order: Optional[Tuple[str, str]] = None
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Extract cross-section for a specific date.

        Args:
            data: Panel data with MultiIndex
            date: Date to extract
            index_order: Index order (defaults to ('date', 'symbol'))

        Returns:
            Cross-section data for the specified date
        """
        # Ensure data is in standard format
        data = PanelDataFormatter.ensure_panel_format(data, index_order)

        # Convert date if needed
        if not isinstance(date, (pd.Timestamp, datetime)):
            date = pd.to_datetime(date)

        # Extract cross-section
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex to extract cross-section")

        try:
            cross_section = data.xs(date, level='date')
            logger.debug(f"Extracted cross-section for {date}: {len(cross_section)} observations")
            return cross_section
        except KeyError:
            logger.warning(f"No data found for date {date}")
            return pd.DataFrame() if isinstance(data, pd.DataFrame) else pd.Series()

    @staticmethod
    def get_time_series(
        data: Union[pd.DataFrame, pd.Series],
        symbol: str,
        index_order: Optional[Tuple[str, str]] = None
    ) -> Union[pd.DataFrame, pd.Series]:
        """
        Extract time series for a specific symbol.

        Args:
            data: Panel data with MultiIndex
            symbol: Symbol to extract
            index_order: Index order (defaults to ('date', 'symbol'))

        Returns:
            Time series data for the specified symbol
        """
        # Ensure data is in standard format
        data = PanelDataFormatter.ensure_panel_format(data, index_order)

        # Extract time series
        if not isinstance(data.index, pd.MultiIndex):
            raise ValueError("Data must have MultiIndex to extract time series")

        try:
            time_series = data.xs(symbol, level='symbol')
            logger.debug(f"Extracted time series for {symbol}: {len(time_series)} observations")
            return time_series
        except KeyError:
            logger.warning(f"No data found for symbol {symbol}")
            return pd.DataFrame() if isinstance(data, pd.DataFrame) else pd.Series()

    @staticmethod
    def get_info(data: Union[pd.DataFrame, pd.Series]) -> dict:
        """Get information about panel data structure."""
        info = {
            'shape': data.shape,
            'index_type': type(data.index).__name__,
            'has_multiindex': isinstance(data.index, pd.MultiIndex),
        }

        if isinstance(data.index, pd.MultiIndex):
            info.update({
                'index_levels': data.index.names,
                'n_dates': len(data.index.get_level_values('date').unique()) if 'date' in data.index.names else None,
                'n_symbols': len(data.index.get_level_values('symbol').unique()) if 'symbol' in data.index.names else None,
                'date_range': (
                    data.index.get_level_values('date').min(),
                    data.index.get_level_values('date').max()
                ) if 'date' in data.index.names else None,
            })

        return info