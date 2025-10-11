"""
Liquidity filter utility for stock screening.

This module provides static methods to filter stocks based on liquidity
criteria like market cap, trading volume, and price levels.
"""

import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class LiquidityFilter:
    """
    Pure utility class for liquidity filtering operations.

    All methods are static to maintain statelessness and reusability.
    This class follows the Single Responsibility Principle by focusing
    only on liquidity filtering logic.
    """

    @staticmethod
    def filter_by_market_cap(symbols: List[str],
                           market_cap_data: Dict[str, float],
                           min_market_cap: float) -> List[str]:
        """
        Filter symbols by minimum market capitalization.

        Args:
            symbols: List of stock symbols to filter
            market_cap_data: Dictionary mapping symbols to market cap values
            min_market_cap: Minimum market cap threshold

        Returns:
            Filtered list of symbols meeting market cap requirement
        """
        if min_market_cap <= 0:
            return symbols

        filtered_symbols = []
        for symbol in symbols:
            market_cap = market_cap_data.get(symbol)
            if market_cap is not None and market_cap >= min_market_cap:
                filtered_symbols.append(symbol)
            else:
                logger.debug(f"Symbol {symbol} filtered out: market cap ${market_cap:,.0f} < ${min_market_cap:,.0f}")

        logger.info(f"Market cap filter: {len(filtered_symbols)}/{len(symbols)} symbols passed")
        return filtered_symbols

    @staticmethod
    def filter_by_volume(price_data: Dict[str, pd.DataFrame],
                        min_avg_daily_volume: float,
                        lookback_days: int = 21) -> List[str]:
        """
        Filter symbols by minimum average daily trading volume.

        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            min_avg_daily_volume: Minimum average daily volume threshold
            lookback_days: Number of days to average over

        Returns:
            Filtered list of symbols meeting volume requirement
        """
        if min_avg_daily_volume <= 0:
            return list(price_data.keys())

        filtered_symbols = []

        for symbol, data in price_data.items():
            if data is None or data.empty:
                logger.debug(f"Symbol {symbol} filtered out: no price data")
                continue

            if 'Volume' not in data.columns:
                logger.debug(f"Symbol {symbol} filtered out: no volume data")
                continue

            # Calculate average volume over lookback period
            recent_volume = data['Volume'].tail(lookback_days).mean()

            if recent_volume >= min_avg_daily_volume:
                filtered_symbols.append(symbol)
            else:
                logger.debug(f"Symbol {symbol} filtered out: avg volume {recent_volume:,.0f} < {min_avg_daily_volume:,.0f}")

        logger.info(f"Volume filter: {len(filtered_symbols)}/{len(price_data)} symbols passed")
        return filtered_symbols

    @staticmethod
    def filter_by_price(price_data: Dict[str, pd.DataFrame],
                       min_price: float = 0.0,
                       max_price: float = float('inf')) -> List[str]:
        """
        Filter symbols by price range.

        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            min_price: Minimum price threshold
            max_price: Maximum price threshold

        Returns:
            Filtered list of symbols within price range
        """
        if min_price <= 0 and max_price == float('inf'):
            return list(price_data.keys())

        filtered_symbols = []

        for symbol, data in price_data.items():
            if data is None or data.empty:
                continue

            if 'Close' not in data.columns:
                continue

            latest_price = data['Close'].iloc[-1]

            if min_price <= latest_price <= max_price:
                filtered_symbols.append(symbol)
            else:
                logger.debug(f"Symbol {symbol} filtered out: price ${latest_price:.2f} not in [${min_price:.2f}, ${max_price:.2f}]")

        logger.info(f"Price filter: {len(filtered_symbols)}/{len(price_data)} symbols passed")
        return filtered_symbols

    @staticmethod
    def filter_by_data_availability(price_data: Dict[str, pd.DataFrame],
                                   min_history_days: int = 252) -> List[str]:
        """
        Filter symbols by minimum data history requirement.

        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            min_history_days: Minimum number of trading days required

        Returns:
            Filtered list of symbols with sufficient data history
        """
        if min_history_days <= 0:
            return list(price_data.keys())

        filtered_symbols = []

        for symbol, data in price_data.items():
            if data is None or data.empty:
                logger.debug(f"Symbol {symbol} filtered out: no price data")
                continue

            data_length = len(data)
            if data_length >= min_history_days:
                filtered_symbols.append(symbol)
            else:
                logger.debug(f"Symbol {symbol} filtered out: {data_length} days < {min_history_days} required")

        logger.info(f"Data availability filter: {len(filtered_symbols)}/{len(price_data)} symbols passed")
        return filtered_symbols

    @staticmethod
    def apply_liquidity_filters(symbols: List[str],
                               price_data: Dict[str, pd.DataFrame],
                               config: Dict[str, Any]) -> List[str]:
        """
        Apply all configured liquidity filters to symbols.

        This is the main entry point that orchestrates all filtering
        based on the provided configuration.

        Args:
            symbols: List of stock symbols to filter
            price_data: Dictionary mapping symbols to price DataFrames
            config: Liquidity filter configuration

        Returns:
            Filtered list of symbols passing all applicable filters
        """
        if not config.get('enabled', False):
            logger.info("Liquidity filtering disabled")
            return symbols

        logger.info(f"Applying liquidity filters to {len(symbols)} symbols")

        # Start with all symbols
        filtered_symbols = symbols.copy()

        # Apply data availability filter first
        min_history_days = config.get('min_history_days', 0)
        if min_history_days > 0:
            # Only keep symbols with sufficient data
            available_symbols = LiquidityFilter.filter_by_data_availability(
                {sym: price_data[sym] for sym in filtered_symbols if sym in price_data},
                min_history_days
            )
            filtered_symbols = [sym for sym in filtered_symbols if sym in available_symbols]

        # Apply volume filter
        min_avg_volume = config.get('min_avg_daily_volume', 0)
        if min_avg_volume > 0:
            volume_filtered = LiquidityFilter.filter_by_volume(
                {sym: price_data[sym] for sym in filtered_symbols if sym in price_data},
                min_avg_volume,
                config.get('volume_lookback_days', 21)
            )
            filtered_symbols = [sym for sym in filtered_symbols if sym in volume_filtered]

        # Apply price filter
        min_price = config.get('min_price', 0)
        max_price = config.get('max_price', float('inf'))
        if min_price > 0 or max_price < float('inf'):
            price_filtered = LiquidityFilter.filter_by_price(
                {sym: price_data[sym] for sym in filtered_symbols if sym in price_data},
                min_price,
                max_price
            )
            filtered_symbols = [sym for sym in filtered_symbols if sym in price_filtered]

        # Apply market cap filter (if market cap data is provided)
        min_market_cap = config.get('min_market_cap', 0)
        if min_market_cap > 0 and 'market_cap_data' in config:
            market_cap_data = config['market_cap_data']
            filtered_symbols = LiquidityFilter.filter_by_market_cap(
                filtered_symbols,
                market_cap_data,
                min_market_cap
            )

        logger.info(f"Liquidity filtering complete: {len(filtered_symbols)}/{len(symbols)} symbols passed")
        return filtered_symbols

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default liquidity filter configuration.

        Returns:
            Dictionary with default filter settings
        """
        return {
            'enabled': False,
            'min_market_cap': 1000000000,  # $1B
            'min_avg_daily_volume': 1000000,  # $1M
            'min_price': 5.0,  # $5
            'max_price': float('inf'),
            'min_history_days': 252,  # 1 year
            'volume_lookback_days': 21  # 1 month
        }