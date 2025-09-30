"""
Transaction Cost Model - Simplified Implementation

A clean, practical transaction cost model that replaces the 580-line
complex implementation with essential, configurable cost components.
"""

import logging
import numpy as np
from typing import Dict, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Trade direction enumeration."""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"


class TransactionCostModel:
    """
    Simplified transaction cost model.

    Focuses on the most important cost components without unnecessary complexity.
    """

    def __init__(self,
                 commission_rate: float = 0.001,
                 spread_rate: float = 0.0005,
                 slippage_rate: float = 0.0002,
                 short_borrow_rate: float = 0.02):
        """
        Initialize transaction cost model.

        Args:
            commission_rate: Commission as fraction of trade value (0.1% default)
            spread_rate: Bid-ask spread cost as fraction (0.05% default)
            slippage_rate: Slippage cost as fraction (0.02% default)
            short_borrow_rate: Annual borrow rate for short positions (2% default)
        """
        self.commission_rate = commission_rate
        self.spread_rate = spread_rate
        self.slippage_rate = slippage_rate
        self.short_borrow_rate = short_borrow_rate

        logger.info(f"Initialized TransactionCostModel: commission={commission_rate:.3%}, "
                   f"spread={spread_rate:.3%}, slippage={slippage_rate:.3%}")

    def calculate_cost_breakdown(self,
                                trade_value: float,
                                direction: TradeDirection = TradeDirection.BUY,
                                market_data: Optional[Dict[str, Any]] = None,
                                holding_days: int = 1) -> Dict[str, float]:
        """
        Get detailed cost breakdown for analysis.

        Returns:
            Dictionary with individual cost components
        """
        if trade_value <= 0:
            return {
                'commission': 0.0,
                'spread': 0.0,
                'slippage': 0.0,
                'short_cost': 0.0,
                'total': 0.0,
                'cost_percentage': 0.0
            }

        # Calculate components
        commission = trade_value * self.commission_rate
        spread = trade_value * self.spread_rate
        slippage = trade_value * self.slippage_rate

        # Adjust for market conditions
        if market_data:
            volatility_multiplier = self._get_volatility_adjustment(market_data.get('volatility', 0.2))
            spread *= volatility_multiplier
            slippage *= volatility_multiplier

        # Short costs
        short_cost = 0.0
        if direction == TradeDirection.SHORT:
            daily_borrow_rate = self.short_borrow_rate / 365.25
            short_cost = trade_value * daily_borrow_rate * holding_days

        total_cost = commission + spread + slippage + short_cost

        return {
            'commission': commission,
            'spread': spread,
            'slippage': slippage,
            'short_cost': short_cost,
            'total': total_cost,
            'cost_percentage': total_cost / trade_value if trade_value > 0 else 0
        }

    def _get_volatility_adjustment(self, volatility: float) -> float:
        """
        Get volatility-based adjustment factor for spread and slippage.

        Higher volatility = wider spreads and more slippage.

        Args:
            volatility: Annualized volatility

        Returns:
            Multiplier for base costs
        """
        # Base adjustment: 1.0 for 20% volatility (typical)
        base_volatility = 0.20

        if volatility <= 0:
            return 1.0

        # Linear adjustment: higher volatility = higher multiplier
        adjustment = 1.0 + (volatility - base_volatility) * 2

        # Cap the adjustment to prevent extreme values
        return min(max(adjustment, 0.5), 3.0)
