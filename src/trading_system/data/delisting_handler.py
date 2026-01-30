"""
Delisting Handler for Survivorship Bias Correction

This module implements survivorship bias correction by handling delisting events
and applying appropriate delisting returns to prevent optimistic backtest results.

Academic Background:
- Survivorship bias occurs when only currently-listed stocks are included in historical data
- This leads to overestimation of strategy performance because failed stocks are excluded
- Proper correction requires tracking delisting events and applying delisting returns

Methodology:
1. Track delisting events (reason, date, last price)
2. Apply CRSP-style delisting returns based on delisting reason
3. Include delisted stocks in performance calculations until delisting date
4. Apply delisting return at the end of the holding period

References:
- Shumway, T., & Warther, V. A. (1999). The delisting bias in CRSP's Nasdaq data.
  *Journal of Finance*, 54(1), 283-298.
- CRSP Delisting Returns Documentation:
  https://www.crsp.org/products/documentation/delistingreturns
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DelistingEvent:
    """
    Represents a stock delisting event.

    Attributes:
    ----------
    symbol : str
        Stock symbol
    delisting_date : datetime
        Date when stock was delisted
    delisting_reason : str
        Reason for delisting (e.g., 'merger', 'bankruptcy', 'exchange')
    last_price : float
        Last trading price before delisting
    delisting_return : Optional[float]
        Calculated delisting return
    """
    symbol: str
    delisting_date: datetime
    delisting_reason: str
    last_price: float
    delisting_return: Optional[float] = None


class DelistingHandler:
    """
    Handles delisting events and applies survivorship bias correction.

    This class manages delisting events and applies appropriate delisting returns
    to correct for survivorship bias in backtesting.

    Delisting Return Methodology (CRSP-style):
    - Performance-related delistings: -30% return
    - Exchange-related delistings: -55% return
    - Merger/Acquisition: 0% return (assume acquisition at market price)
    - Unknown reasons: -30% return (conservative)
    - Voluntary delistings: -15% return

    Example:
        >>> handler = DelistingHandler()
        >>> handler.add_delisting_event(
        ...     symbol='AAPL',
        ...     delisting_date=datetime(2020, 12, 31),
        ...     delisting_reason='bankruptcy',
        ...     last_price=100.0
        ... )
        >>> return_rate = handler.get_delisting_return('AAPL')
        >>> print(f"Delisting return: {return_rate:.2%}")
        Delisting return: -30.00%
    """

    # CRSP-style delisting return rates (conservative estimates)
    DELISTING_RETURNS = {
        'merger': 0.0,                    # Assume acquired at market price
        'acquisition': 0.0,               # Assume acquired at market price
        'bankruptcy': -0.30,              # 30% loss
        'liquidation': -0.55,             # 55% loss
        'exchange': -0.55,                # Exchange-related: 55% loss
        'voluntary': -0.15,               # Voluntary: 15% loss
        'performance': -0.30,             # Performance-related: 30% loss
        'unknown': -0.30,                 # Unknown: conservative 30% loss
        'default': -0.30,                 # Default: conservative 30% loss
    }

    def __init__(self):
        """Initialize the DelistingHandler."""
        self.delisting_events: Dict[str, DelistingEvent] = {}
        logger.info("Initialized DelistingHandler for survivorship bias correction")

    def add_delisting_event(self,
                           symbol: str,
                           delisting_date: datetime,
                           delisting_reason: str,
                           last_price: float,
                           delisting_return: Optional[float] = None) -> None:
        """
        Add a delisting event for a stock.

        Args:
        -----
        symbol : str
            Stock symbol
        delisting_date : datetime
            Date when stock was delisted
        delisting_reason : str
            Reason for delisting (e.g., 'merger', 'bankruptcy', 'exchange')
        last_price : float
            Last trading price before delisting
        delisting_return : Optional[float]
            Explicit delisting return (overrides default rates)
        """
        # Normalize delisting reason
        reason_key = delisting_reason.lower()

        # Calculate delisting return if not provided
        if delisting_return is None:
            delisting_return = self.DELISTING_RETURNS.get(
                reason_key,
                self.DELISTING_RETURNS['default']
            )

        event = DelistingEvent(
            symbol=symbol,
            delisting_date=delisting_date,
            delisting_reason=delisting_reason,
            last_price=last_price,
            delisting_return=delisting_return
        )

        self.delisting_events[symbol] = event

        logger.info(f"Added delisting event: {symbol} on {delisting_date.date()}, "
                   f"reason={delisting_reason}, return={delisting_return:.2%}")

    def get_delisting_return(self, symbol: str) -> Optional[float]:
        """
        Get the delisting return for a symbol.

        Args:
        -----
        symbol : str
            Stock symbol

        Returns:
        --------
        float or None
            Delisting return if symbol delisted, None otherwise
        """
        if symbol in self.delisting_events:
            return self.delisting_events[symbol].delisting_return
        return None

    def get_delisting_event(self, symbol: str) -> Optional[DelistingEvent]:
        """
        Get the full delisting event for a symbol.

        Args:
        -----
        symbol : str
            Stock symbol

        Returns:
        --------
        DelistingEvent or None
            Delisting event if symbol delisted, None otherwise
        """
        return self.delisting_events.get(symbol)

    def is_delisted(self, symbol: str, as_of_date: datetime) -> bool:
        """
        Check if a symbol was delisted as of a given date.

        Args:
        -----
        symbol : str
            Stock symbol
        as_of_date : datetime
            Date to check

        Returns:
        --------
        bool
            True if symbol was delisted on or before as_of_date
        """
        if symbol not in self.delisting_events:
            return False

        return self.delisting_events[symbol].delisting_date <= as_of_date

    def get_universe_at_date(self,
                             symbols: List[str],
                             as_of_date: datetime) -> List[str]:
        """
        Get list of active (non-delisted) symbols as of a given date.

        This is the core method for survivorship bias correction - it returns
        the point-in-time universe excluding stocks that have already delisted.

        Args:
        -----
        symbols : List[str]
            List of potential symbols
        as_of_date : datetime
            Date to check

        Returns:
        --------
        List[str]
            List of symbols that were active on as_of_date
        """
        active_symbols = []

        for symbol in symbols:
            # Include if not delisted, or delisted after as_of_date
            if not self.is_delisted(symbol, as_of_date):
                active_symbols.append(symbol)
            else:
                event = self.delisting_events[symbol]
                logger.debug(f"Excluding delisted symbol {symbol} "
                           f"(delisted {event.delisting_date.date()})")

        excluded_count = len(symbols) - len(active_symbols)
        if excluded_count > 0:
            logger.info(f"Survivorship bias correction: Excluded {excluded_count} "
                       f"delisted symbols as of {as_of_date.date()} "
                       f"({len(active_symbols)} active symbols remaining)")

        return active_symbols

    def adjust_return_for_delisting(self,
                                    symbol: str,
                                    period_return: float,
                                    exit_date: datetime) -> float:
        """
        Adjust portfolio return for delisting event.

        If a stock delists during the holding period, applies the delisting
        return to prevent optimistic bias.

        Args:
        -----
        symbol : str
            Stock symbol
        period_return : float
            Return for the holding period
        exit_date : datetime
            Date when position is exited

        Returns:
        --------
        float
            Adjusted return including delisting impact
        """
        if symbol not in self.delisting_events:
            return period_return

        event = self.delisting_events[symbol]

        # If delisting occurred before exit date, apply delisting return
        if event.delisting_date <= exit_date:
            logger.info(f"Applying delisting return for {symbol}: "
                       f"{event.delisting_return:.2%} (reason: {event.delisting_reason})")

            # Apply delisting return: (1 + period_return) * (1 + delisting_return) - 1
            adjusted_return = (1 + period_return) * (1 + event.delisting_return) - 1

            logger.info(f"Original return: {period_return:.2%}, "
                       f"Adjusted return: {adjusted_return:.2%}")

            return adjusted_return

        return period_return

    def get_delisting_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of delisting events.

        Returns:
        --------
        pd.DataFrame
            Summary of delisting events by reason
        """
        if not self.delisting_events:
            logger.warning("No delisting events recorded")
            return pd.DataFrame()

        # Count delistings by reason
        reason_counts = {}
        for event in self.delisting_events.values():
            reason = event.delisting_reason
            if reason not in reason_counts:
                reason_counts[reason] = {'count': 0, 'avg_return': 0.0}
            reason_counts[reason]['count'] += 1
            reason_counts[reason]['avg_return'] += event.delisting_return

        # Calculate averages
        summary = []
        for reason, stats in reason_counts.items():
            avg_return = stats['avg_return'] / stats['count']
            summary.append({
                'reason': reason,
                'count': stats['count'],
                'avg_return': avg_return,
                'avg_return_pct': f"{avg_return:.2%}"
            })

        df = pd.DataFrame(summary).sort_values('count', ascending=False)

        logger.info(f"Delisting summary: {len(self.delisting_events)} total events")
        return df

    def load_from_crsp_style_data(self, delisting_data: pd.DataFrame) -> None:
        """
        Load delisting events from CRSP-style data.

        Args:
        -----
        delisting_data : pd.DataFrame
            DataFrame with columns:
            - symbol: Stock symbol
            - dlstcd: CRSP delisting code
            - dlstdt: Delisting date
            - dlprc: Delisting price
        """
        # Map CRSP delisting codes to reasons
        # CRSP codes: https://www.crsp.org/products/documentation/delistingcodes
        crsp_code_mapping = {
            # Merger/Acquisition (codes 200-299)
            '200': 'merger',
            '201': 'merger',
            '202': 'merger',
            # Exchange-related (codes 500-599)
            '500': 'exchange',
            '501': 'exchange',
            '502': 'exchange',
            # Liquidation (codes 400-499)
            '400': 'liquidation',
            '401': 'liquidation',
            # Performance/Other (codes 100-199)
            '100': 'performance',
            '101': 'performance',
        }

        for _, row in delisting_data.iterrows():
            symbol = row['symbol']
            delisting_code = str(row.get('dlstcd', ''))
            delisting_date = pd.to_datetime(row['dlstdt'])
            last_price = row.get('dlprc', np.nan)

            # Map delisting code to reason
            delisting_reason = crsp_code_mapping.get(
                delisting_code[:3],  # First 3 digits
                'unknown'
            )

            self.add_delisting_event(
                symbol=symbol,
                delisting_date=delisting_date,
                delisting_reason=delisting_reason,
                last_price=last_price
            )

        logger.info(f"Loaded {len(delisting_data)} delisting events from CRSP-style data")


def create_mock_delisting_data(symbols: List[str],
                               delisting_rate: float = 0.02,
                               start_date: datetime = None,
                               end_date: datetime = None) -> DelistingHandler:
    """
    Create a DelistingHandler with mock delisting data for testing.

    This function simulates delisting events for backtesting when actual
    delisting data is not available. It's useful for:
    - Testing survivorship bias correction
    - Conservative estimation when delisting data is missing
    - Stress testing strategies

    Args:
    -----
    symbols : List[str]
        List of symbols to potentially delist
    delisting_rate : float
        Annual delisting rate (default: 2% = 0.02)
    start_date : datetime
        Start date for simulation
    end_date : datetime
        End date for simulation

    Returns:
    --------
    DelistingHandler
        Handler with mock delisting events
    """
    if start_date is None:
        start_date = datetime(2020, 1, 1)
    if end_date is None:
        end_date = datetime(2024, 12, 31)

    handler = DelistingHandler()

    # Calculate number of trading days
    total_days = (end_date - start_date).days
    n_symbols_to_delist = int(len(symbols) * delisting_rate * (total_days / 365))

    # Randomly select symbols and dates to delist
    np.random.seed(42)  # For reproducibility
    delisted_symbols = np.random.choice(
        symbols,
        size=min(n_symbols_to_delist, len(symbols)),
        replace=False
    )

    # Assign random delisting reasons and dates
    reasons = list(DelistingHandler.DELISTING_RETURNS.keys())
    reasons.remove('default')  # Don't use 'default' for simulation

    for symbol in delisted_symbols:
        # Random delisting date within period
        days_offset = np.random.randint(0, total_days)
        delisting_date = start_date + timedelta(days=days_offset)

        # Random delisting reason (weighted by real-world frequencies)
        # 60% mergers, 20% bankruptcies, 10% exchanges, 10% other
        reason_weights = [0.6, 0.2, 0.1, 0.05, 0.03, 0.01, 0.01]
        delisting_reason = np.random.choice(
            ['merger', 'bankruptcy', 'exchange', 'voluntary', 'performance', 'liquidation', 'unknown'],
            p=reason_weights
        )

        handler.add_delisting_event(
            symbol=symbol,
            delisting_date=delisting_date,
            delisting_reason=delisting_reason,
            last_price=100.0  # Mock price
        )

    logger.info(f"Created mock delisting data: {len(delisted_symbols)} symbols "
               f"delisted out of {len(symbols)} total ({delisting_rate:.1%} annual rate)")

    return handler
