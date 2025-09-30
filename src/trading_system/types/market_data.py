"""
Market Data Types

Unified market data structure definitions.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from datetime import datetime
from .enums import DataSource


@dataclass
class PriceData:
    """
    Standardized price data structure.

    Single source of truth for all price data in the system.
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None
    source: DataSource = DataSource.YFINANCE

    def __post_init__(self):
        """Validate price data parameters."""
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError(f"Price values must be positive for {self.symbol}")

        if self.high < self.low:
            raise ValueError(f"High price ({self.high}) must be >= low price ({self.low}) for {self.symbol}")

        if self.high < self.open or self.high < self.close:
            raise ValueError(f"High price ({self.high}) must be >= open ({self.open}) and close ({self.close}) for {self.symbol}")

        if self.low > self.open or self.low > self.close:
            raise ValueError(f"Low price ({self.low}) must be <= open ({self.open}) and close ({self.close}) for {self.symbol}")

        if self.volume < 0:
            raise ValueError(f"Volume cannot be negative for {self.symbol}")

    @property
    def typical_price(self) -> float:
        """Calculate typical price (H+L+C)/3."""
        return (self.high + self.low + self.close) / 3

    @property
    def median_price(self) -> float:
        """Calculate median price (H+L)/2."""
        return (self.high + self.low) / 2

    @property
    def range(self) -> float:
        """Calculate price range (H-L)."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Calculate candle body (|O-C|)."""
        return abs(self.open - self.close)

    @property
    def is_doji(self) -> bool:
        """Check if this is a doji candle (open ≈ close)."""
        return self.body / self.range < 0.1 if self.range > 0 else False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'adj_close': self.adj_close,
            'source': self.source.value,
            'typical_price': self.typical_price,
            'median_price': self.median_price,
            'range': self.range,
            'body': self.body,
            'is_doji': self.is_doji
        }


@dataclass
class FactorData:
    """
    Factor data structure for quantitative analysis.

    Supports Fama-French factors and other quantitative factors.
    """
    date: datetime
    mkt_rf: float  # Market excess return
    smb: float    # Small minus big
    hml: float    # High minus low
    rmw: float    # Robust minus weak
    cma: float    # Conservative minus aggressive
    rf: float     # Risk-free rate

    def __post_init__(self):
        """Validate factor data parameters."""
        # Check for reasonable factor ranges
        factors = [self.mkt_rf, self.smb, self.hml, self.rmw, self.cma]
        for factor in factors:
            if abs(factor) > 2.0:  # Factors rarely exceed ±200%
                raise ValueError(f"Factor value {factor} seems unreasonable")

        if not -0.1 <= self.rf <= 0.2:  # Risk-free rate typically between -10% and +20%
            raise ValueError(f"Risk-free rate {self.rf} seems unreasonable")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'date': self.date.isoformat(),
            'MKT': self.mkt_rf,
            'SMB': self.smb,
            'HML': self.hml,
            'RMW': self.rmw,
            'CMA': self.cma,
            'RF': self.rf
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactorData':
        """Create from dictionary with flexible column names."""
        # Handle different column name conventions
        mapping = {
            'mkt_rf': ['mkt_rf', 'MKT_RF', 'Mkt-RF', 'market_premium'],
            'smb': ['smb', 'SMB', 'Small_minus_Big'],
            'hml': ['hml', 'HML', 'High_minus_Low'],
            'rmw': ['rmw', 'RMW', 'Robust_minus_Weak'],
            'cma': ['cma', 'CMA', 'Conservative_minus_Aggressive'],
            'rf': ['rf', 'RF', 'risk_free_rate']
        }

        kwargs = {}
        for attr_name, possible_keys in mapping.items():
            for key in possible_keys:
                if key in data:
                    kwargs[attr_name] = data[key]
                    break

        if 'date' in data:
            if isinstance(data['date'], str):
                kwargs['date'] = datetime.fromisoformat(data['date'])
            else:
                kwargs['date'] = data['date']

        return cls(**kwargs)


# Type aliases for common data structures
PriceDict = Dict[str, PriceData]
FactorDict = Dict[datetime, FactorData]
PriceDataFrame = Dict[str, 'pd.DataFrame']  # For backward compatibility