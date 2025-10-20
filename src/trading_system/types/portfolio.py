"""
Portfolio Types

Unified portfolio and trading types following SOLID principles.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class Position:
    """
    Unified portfolio position structure.

    Replaces both Position and PortfolioPosition from the old system.
    """
    symbol: str
    quantity: float
    average_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    weight: float  # Portfolio weight (0.0 to 1.0)

    def __post_init__(self):
        """Validate position parameters."""
        # Allow zero values for empty positions
        if self.quantity != 0 and self.average_cost <= 0:
            raise ValueError(f"Average cost must be positive for non-empty position, got {self.average_cost}")

        if self.quantity != 0 and self.current_price <= 0:
            raise ValueError(f"Current price must be positive for non-empty position, got {self.current_price}")

        # Weight should be valid, but can be 0 for empty positions
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {self.weight}")

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.quantity > 0

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.quantity < 0

    @property
    def is_empty(self) -> bool:
        """Check if position is empty."""
        return abs(self.quantity) < 1e-6

    @property
    def return_pct(self) -> float:
        """Calculate return percentage for this position."""
        if self.average_cost == 0:
            return 0.0
        return (self.current_price - self.average_cost) / self.average_cost

    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'average_cost': self.average_cost,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'unrealized_pnl': self.unrealized_pnl,
            'weight': self.weight,
            'is_long': self.is_long,
            'is_short': self.is_short,
            'return_pct': self.return_pct
        }


@dataclass
class Trade:
    """
    Unified trade execution structure.

    Single source of truth for all trade records.
    """
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    commission: float = 0.0
    trade_id: Optional[str] = None

    def __post_init__(self):
        """Validate trade parameters."""
        if self.side not in ['buy', 'sell']:
            raise ValueError(f"Trade side must be 'buy' or 'sell', got '{self.side}'")

        if self.quantity <= 0:
            raise ValueError(f"Trade quantity must be positive, got {self.quantity}")

        if self.price <= 0:
            raise ValueError(f"Trade price must be positive, got {self.price}")

        if self.commission < 0:
            raise ValueError(f"Commission cannot be negative, got {self.commission}")

    @property
    def total_cost(self) -> float:
        """Calculate total cost including commission."""
        base_cost = self.quantity * self.price
        if self.side == 'buy':
            return base_cost + self.commission
        else:
            return base_cost - self.commission

    @property
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.side == 'buy'

    @property
    def is_sell(self) -> bool:
        """Check if this is a sell trade."""
        return self.side == 'sell'

    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'commission': self.commission,
            'total_cost': self.total_cost,
            'trade_id': self.trade_id
        }


@dataclass
class PortfolioSnapshot:
    """
    Portfolio snapshot at a point in time.

    Unified structure for portfolio state tracking.
    """
    timestamp: datetime
    total_value: float
    cash_balance: float
    positions: List[Position]
    daily_return: float
    total_return: float
    drawdown: float

    def __post_init__(self):
        """Validate snapshot parameters."""
        if self.total_value < 0:
            raise ValueError(f"Total value cannot be negative, got {self.total_value}")

        if self.cash_balance < 0:
            raise ValueError(f"Cash balance cannot be negative, got {self.cash_balance}")

        # Validate positions sum to total value
        positions_value = sum(pos.market_value for pos in self.positions)
        expected_total = self.cash_balance + positions_value
        if abs(self.total_value - expected_total) > 0.01:  # Allow small rounding errors
            raise ValueError(f"Portfolio value mismatch: {self.total_value} vs {expected_total}")

    @property
    def equity_value(self) -> float:
        """Calculate total equity value."""
        return sum(pos.market_value for pos in self.positions)

    @property
    def cash_ratio(self) -> float:
        """Calculate cash as ratio of total portfolio."""
        if self.total_value == 0:
            return 0.0
        return self.cash_balance / self.total_value

    @property
    def equity_ratio(self) -> float:
        """Calculate equity as ratio of total portfolio."""
        if self.total_value == 0:
            return 0.0
        return self.equity_value / self.total_value

    @property
    def positions_count(self) -> int:
        """Get number of non-empty positions."""
        return len([pos for pos in self.positions if not pos.is_empty])

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        for pos in self.positions:
            if pos.symbol == symbol:
                return pos
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert snapshot to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_value': self.total_value,
            'cash_balance': self.cash_balance,
            'equity_value': self.equity_value,
            'cash_ratio': self.cash_ratio,
            'equity_ratio': self.equity_ratio,
            'positions_count': self.positions_count,
            'daily_return': self.daily_return,
            'total_return': self.total_return,
            'drawdown': self.drawdown,
            'positions': [pos.to_dict() for pos in self.positions]
        }


# Type aliases for backward compatibility
PositionList = List[Position]
TradeList = List[Trade]
PortfolioHistory = List[PortfolioSnapshot]