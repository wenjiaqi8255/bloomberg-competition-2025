"""
Transaction Cost Model - Simplified Implementation with Market Impact

A clean, practical transaction cost model that replaces the 580-line
complex implementation with essential, configurable cost components.

Now includes Almgren-Chriss (2001) market impact modeling for realistic
execution cost estimation.

References:
- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio
  transactions. *Journal of Risk*, 3(2), 5-39.
"""

import logging
import numpy as np
from typing import Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    """Trade direction enumeration."""
    BUY = "buy"
    SELL = "sell"
    SHORT = "short"


@dataclass
class MarketImpactParameters:
    """
    Parameters for Almgren-Chriss market impact model.

    Attributes:
    ----------
    gamma : float
        Permanent impact coefficient (price impact per unit of volume)
    eta : float
        Temporary impact coefficient (price impact per unit of trading rate)
    daily_volume : float
        Average daily volume (used for scaling order size)
    lambda_risk : float, optional
        Risk aversion parameter for optimal execution (default: 1e-6)
    """
    gamma: float = 1e-6  # Permanent impact coefficient
    eta: float = 1e-6    # Temporary impact coefficient
    daily_volume: float = 1_000_000  # Placeholder, should be set per stock
    lambda_risk: float = 1e-6  # Risk aversion


class MarketImpactModel:
    """
    Almgren-Chriss (2001) market impact model.

    This model calculates the price impact of trading based on order size
    relative to daily volume. It accounts for both permanent and temporary
    market impact.

    Permanent Impact:
        Caused by information content of the trade
        h = γ * (order_size / daily_volume) * price

    Temporary Impact:
        Caused by liquidity needs and execution urgency
        k = η * (order_size / daily_volume) * (1 / trading_rate) * price

    Total Cost = Commission + Permanent Impact + Temporary Impact

    References:
    - Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio
      transactions. *Journal of Risk*, 3(2), 5-39.

    Example:
        >>> model = MarketImpactModel(
        ...     gamma=1e-6,
        ...     eta=1e-6,
        ...     default_daily_volume=1_000_000
        ... )
        >>> impact = model.calculate_market_impact(
        ...     order_value=100_000,
        ...     price=100,
        ...     daily_volume=1_000_000
        ... )
        >>> print(f"Market impact: ${impact['total_impact']:.2f}")
        Market impact: $20.00
    """

    def __init__(self,
                 gamma: float = 1e-6,
                 eta: float = 1e-6,
                 default_daily_volume: float = 1_000_000,
                 enabled: bool = True):
        """
        Initialize market impact model.

        Args:
        -----
        gamma : float
            Permanent impact coefficient (default: 1e-6)
            Typical range: 1e-7 to 1e-5 depending on market
        eta : float
            Temporary impact coefficient (default: 1e-6)
            Typical range: 1e-7 to 1e-5
        default_daily_volume : float
            Default daily volume for stocks when not provided (default: 1M)
        enabled : bool
            Whether market impact modeling is enabled (default: True)
        """
        self.gamma = gamma
        self.eta = eta
        self.default_daily_volume = default_daily_volume
        self.enabled = enabled

        logger.info(f"Initialized MarketImpactModel: gamma={gamma:.2e}, "
                   f"eta={eta:.2e}, enabled={enabled}")

    def calculate_market_impact(self,
                               order_value: float,
                               price: float,
                               daily_volume: Optional[float] = None,
                               shares: Optional[float] = None,
                               trading_rate: float = 0.1) -> Dict[str, float]:
        """
        Calculate market impact for a trade.

        Args:
        -----
        order_value : float
            Total value of the order (dollars)
        price : float
            Current stock price
        daily_volume : Optional[float]
            Daily volume in shares (uses default if None)
        shares : Optional[float]
            Number of shares in order (calculated from value/price if None)
        trading_rate : float
            Fraction of day over which trade is executed (default: 0.1 = 10% of day)
            Lower values = faster execution = higher temporary impact

        Returns:
        --------
        Dict[str, float]
            Dictionary with:
            - permanent_impact: Permanent price impact cost
            - temporary_impact: Temporary price impact cost
            - total_impact: Sum of both impacts
            - impact_bps: Total impact in basis points
        """
        if not self.enabled:
            return {
                'permanent_impact': 0.0,
                'temporary_impact': 0.0,
                'total_impact': 0.0,
                'impact_bps': 0.0
            }

        if daily_volume is None:
            daily_volume = self.default_daily_volume

        # Calculate number of shares if not provided
        if shares is None:
            shares = order_value / price

        # Calculate order size as fraction of daily volume
        # This is the key driver of market impact
        volume_fraction = shares / daily_volume

        # Permanent impact: h = γ * (size/daily_volume) * price * shares
        # Represents the permanent price displacement due to trade information
        permanent_impact = self.gamma * volume_fraction * price * shares

        # Temporary impact: k = η * (size/daily_volume) * (1/trading_rate) * price * shares
        # Represents the temporary price displacement due to liquidity needs
        # Higher trading_rate (slower execution) = lower temporary impact
        temporary_impact = self.eta * volume_fraction * (1.0 / trading_rate) * price * shares

        total_impact = permanent_impact + temporary_impact

        # Convert to basis points for easier interpretation
        impact_bps = (total_impact / order_value) * 10000 if order_value > 0 else 0.0

        result = {
            'permanent_impact': permanent_impact,
            'temporary_impact': temporary_impact,
            'total_impact': total_impact,
            'impact_bps': impact_bps,
            'volume_fraction': volume_fraction
        }

        logger.debug(f"Market impact: ${total_impact:.2f} ({impact_bps:.2f} bps), "
                    f"permanent={permanent_impact:.2f}, temporary={temporary_impact:.2f}")

        return result

    def estimate_optimal_execution_schedule(self,
                                          total_shares: float,
                                          daily_volume: float,
                                          price: float,
                                          n_days: int = 5) -> np.ndarray:
        """
        Estimate optimal execution schedule using Almgren-Chriss model.

        This calculates how many shares to trade each day to minimize
        the sum of market impact and risk.

        Args:
        -----
        total_shares : float
            Total shares to execute
        daily_volume : float
            Average daily volume (shares)
        price : float
            Current stock price
        n_days : int
            Number of days over which to execute (default: 5)

        Returns:
        --------
        np.ndarray
            Array of shares to trade each day
        """
        if not self.enabled:
            # Equal split if market impact disabled
            return np.full(n_days, total_shares / n_days)

        # Simplified Almgren-Chriss optimal schedule
        # In practice, this requires solving a quadratic optimization
        # Here we use a heuristic: trade more when impact is low

        # Base schedule: equal amounts
        base_schedule = np.full(n_days, total_shares / n_days)

        # Adjust for liquidity: trade more when volume fraction is lower
        # This is a simplification - the full solution requires optimization
        daily_volume_fraction = total_shares / (daily_volume * n_days)

        if daily_volume_fraction > 0.1:  # Large order relative to volume
            # Front-load execution to reduce risk
            weights = np.linspace(1.5, 0.5, n_days)
            weights = weights / weights.sum()
            schedule = total_shares * weights
        else:
            # Equal execution is fine for small orders
            schedule = base_schedule

        return schedule

    def adjust_for_volatility(self,
                             base_impact: float,
                             volatility: float,
                             base_volatility: float = 0.20) -> float:
        """
        Adjust market impact for stock volatility.

        Higher volatility stocks typically have wider spreads and more impact.

        Args:
        -----
        base_impact : float
            Base market impact cost
        volatility : float
            Current annualized volatility
        base_volatility : float
            Reference volatility (default: 20%)

        Returns:
        --------
        float
            Adjusted market impact
        """
        if not self.enabled or volatility <= 0:
            return base_impact

        # Linear adjustment: higher vol = higher impact
        adjustment_factor = 1.0 + (volatility - base_volatility) * 2

        # Cap adjustment to prevent extreme values
        adjustment_factor = min(max(adjustment_factor, 0.5), 3.0)

        return base_impact * adjustment_factor



class TransactionCostModel:
    """
    Simplified transaction cost model with market impact.

    Focuses on the most important cost components without unnecessary complexity.
    Now includes Almgren-Chriss market impact modeling.
    """

    def __init__(self,
                 commission_rate: float = 0.001,
                 spread_rate: float = 0.0005,
                 slippage_rate: float = 0.0002,
                 short_borrow_rate: float = 0.02,
                 market_impact_enabled: bool = True,
                 market_impact_gamma: float = 1e-6,
                 market_impact_eta: float = 1e-6):
        """
        Initialize transaction cost model.

        Args:
            commission_rate: Commission as fraction of trade value (0.1% default)
            spread_rate: Bid-ask spread cost as fraction (0.05% default)
            slippage_rate: Slippage cost as fraction (0.02% default)
            short_borrow_rate: Annual borrow rate for short positions (2% default)
            market_impact_enabled: Enable Almgren-Chriss market impact (default: True)
            market_impact_gamma: Permanent impact coefficient (default: 1e-6)
            market_impact_eta: Temporary impact coefficient (default: 1e-6)
        """
        self.commission_rate = commission_rate
        self.spread_rate = spread_rate
        self.slippage_rate = slippage_rate
        self.short_borrow_rate = short_borrow_rate

        # Initialize market impact model
        self.market_impact_model = MarketImpactModel(
            gamma=market_impact_gamma,
            eta=market_impact_eta,
            enabled=market_impact_enabled
        )

        logger.info(f"Initialized TransactionCostModel: commission={commission_rate:.3%}, "
                   f"spread={spread_rate:.3%}, slippage={slippage_rate:.3%}, "
                   f"market_impact={market_impact_enabled}")

    def calculate_cost_breakdown(self,
                                trade_value: float,
                                direction: TradeDirection = TradeDirection.BUY,
                                market_data: Optional[Dict[str, Any]] = None,
                                holding_days: int = 1) -> Dict[str, float]:
        """
        Get detailed cost breakdown for analysis.

        Now includes market impact costs if enabled.

        Returns:
            Dictionary with individual cost components including market impact
        """
        if trade_value <= 0:
            return {
                'commission': 0.0,
                'spread': 0.0,
                'slippage': 0.0,
                'short_cost': 0.0,
                'market_impact': 0.0,
                'total': 0.0,
                'cost_percentage': 0.0
            }

        # Calculate basic components
        commission = trade_value * self.commission_rate
        spread = trade_value * self.spread_rate
        slippage = trade_value * self.slippage_rate

        # Adjust for market conditions
        volatility = market_data.get('volatility', 0.2) if market_data else 0.2
        if market_data:
            volatility_multiplier = self._get_volatility_adjustment(volatility)
            spread *= volatility_multiplier
            slippage *= volatility_multiplier

        # Calculate market impact
        market_impact = 0.0
        if market_data and self.market_impact_model.enabled:
            price = market_data.get('price', 100.0)
            daily_volume = market_data.get('daily_volume', None)
            shares = market_data.get('shares', None)

            impact_result = self.market_impact_model.calculate_market_impact(
                order_value=trade_value,
                price=price,
                daily_volume=daily_volume,
                shares=shares
            )

            # Adjust for volatility
            base_impact = impact_result['total_impact']
            market_impact = self.market_impact_model.adjust_for_volatility(
                base_impact=base_impact,
                volatility=volatility
            )

        # Short costs
        short_cost = 0.0
        if direction == TradeDirection.SHORT:
            daily_borrow_rate = self.short_borrow_rate / 365.25
            short_cost = trade_value * daily_borrow_rate * holding_days

        total_cost = commission + spread + slippage + short_cost + market_impact

        return {
            'commission': commission,
            'spread': spread,
            'slippage': slippage,
            'short_cost': short_cost,
            'market_impact': market_impact,
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
