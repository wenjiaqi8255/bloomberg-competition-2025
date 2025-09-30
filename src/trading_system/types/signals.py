"""
Trading Signal Types

Unified signal type definitions following KISS principles.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class SignalType(Enum):
    """Trading signal types - simplified and comprehensive."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    NEUTRAL = "neutral"


@dataclass
class TradingSignal:
    """
    Unified trading signal structure.

    Single source of truth for all trading signals across the system.
    """
    symbol: str
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    price: float
    confidence: float = 1.0  # 0.0 to 1.0
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate signal parameters."""
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError(f"Signal strength must be between 0.0 and 1.0, got {self.strength}")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Signal confidence must be between 0.0 and 1.0, got {self.confidence}")

        if self.price <= 0:
            raise ValueError(f"Signal price must be positive, got {self.price}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'confidence': self.confidence,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingSignal':
        """Create signal from dictionary."""
        # Convert signal type string to enum
        if isinstance(data['signal_type'], str):
            data['signal_type'] = SignalType(data['signal_type'])

        # Convert timestamp string to datetime
        if isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])

        return cls(**data)

    def is_buy(self) -> bool:
        """Check if this is a buy signal."""
        return self.signal_type == SignalType.BUY

    def is_sell(self) -> bool:
        """Check if this is a sell signal."""
        return self.signal_type == SignalType.SELL

    def is_hold(self) -> bool:
        """Check if this is a hold signal."""
        return self.signal_type == SignalType.HOLD


# Type aliases for backward compatibility
SignalList = List[TradingSignal]
SignalDict = Dict[datetime, List[TradingSignal]]