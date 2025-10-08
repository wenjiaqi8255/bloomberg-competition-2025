"""
Data types for portfolio construction.
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime


@dataclass(frozen=True)
class BoxKey:
    """
    4-dimensional Box key for stock classification.

    Boxes are defined by:
    - size: Market capitalization category
    - style: Investment style (growth/value)
    - region: Geographic region
    - sector: Industry sector
    """
    size: str          # 'large' | 'mid' | 'small'
    style: str         # 'growth' | 'value'
    region: str        # 'developed' | 'emerging'
    sector: str        # 'Technology' | 'Financials' | 'Healthcare' | ...

    def __str__(self) -> str:
        """String representation of the box key."""
        return f"{self.size}_{self.style}_{self.region}_{self.sector}"

    def __hash__(self) -> int:
        """Hash for use as dictionary keys."""
        return hash((self.size, self.style, self.region, self.sector))

    def to_tuple(self) -> tuple:
        """Convert to tuple for serialization."""
        return (self.size, self.style, self.region, self.sector)

    @classmethod
    def from_tuple(cls, box_tuple: tuple) -> 'BoxKey':
        """Create BoxKey from tuple."""
        if len(box_tuple) != 4:
            raise ValueError(f"Box tuple must have 4 elements, got {len(box_tuple)}")
        return cls(size=box_tuple[0], style=box_tuple[1],
                  region=box_tuple[2], sector=box_tuple[3])

    @classmethod
    def from_string(cls, box_str: str) -> 'BoxKey':
        """Create BoxKey from string representation."""
        parts = box_str.split('_')
        if len(parts) != 4:
            raise ValueError(f"Box string must have 4 parts separated by '_', got {len(parts)}")
        return cls(size=parts[0], style=parts[1],
                  region=parts[2], sector=parts[3])


@dataclass
class PortfolioConstructionRequest:
    """
    Request object for portfolio construction.

    Encapsulates all data needed for portfolio construction in a single
    structured object, making the interface cleaner and more maintainable.
    """
    date: datetime
    universe: List[str]
    signals: pd.Series
    price_data: Dict[str, pd.DataFrame]
    constraints: Dict[str, Any]

    def __post_init__(self):
        """Validate request parameters."""
        if not self.universe:
            raise ValueError("Universe cannot be empty")

        if not isinstance(self.signals, pd.Series):
            raise ValueError("Signals must be a pandas Series")

        if self.signals.empty:
            raise ValueError("Signals cannot be empty")

        if not isinstance(self.price_data, dict):
            raise ValueError("Price data must be a dictionary")

        # Check that at least some stocks in universe have price data
        available_symbols = set(self.price_data.keys())
        universe_symbols = set(self.universe)
        overlap = available_symbols.intersection(universe_symbols)

        if len(overlap) < len(self.universe) * 0.5:  # At least 50% coverage
            raise ValueError(f"Insufficient price data coverage: {len(overlap)}/{len(self.universe)}")

    def get_available_symbols(self) -> List[str]:
        """Get symbols that have both signals and price data."""
        signal_symbols = set(self.signals.index)
        price_symbols = set(self.price_data.keys())
        return list(signal_symbols.intersection(price_symbols))

    def filter_by_availability(self) -> 'PortfolioConstructionRequest':
        """Return a new request filtered to only include available symbols."""
        available_symbols = self.get_available_symbols()

        filtered_signals = self.signals[self.signals.index.isin(available_symbols)]
        filtered_price_data = {symbol: self.price_data[symbol]
                              for symbol in available_symbols
                              if symbol in self.price_data}

        return PortfolioConstructionRequest(
            date=self.date,
            universe=available_symbols,
            signals=filtered_signals,
            price_data=filtered_price_data,
            constraints=self.constraints
        )


@dataclass
class BoxConstructionResult:
    """
    Result from box-based portfolio construction.

    Contains detailed information about the construction process
    for debugging and analysis.
    """
    weights: pd.Series
    box_coverage: Dict[str, float]  # box_key -> coverage ratio
    selected_stocks: Dict[str, List[str]]  # box_key -> selected stocks
    target_weights: Dict[str, float]  # box_key -> target weight
    construction_log: List[str]

    @property
    def total_coverage(self) -> float:
        """Total coverage of target box weights."""
        return sum(self.box_coverage.values())

    @property
    def number_of_boxes_used(self) -> int:
        """Number of boxes that received allocations."""
        return len([w for w in self.weights.values() if w > 0])

    @property
    def total_positions(self) -> int:
        """Total number of positions in final portfolio."""
        return len(self.weights)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to summary dictionary for reporting."""
        return {
            'total_coverage': self.total_coverage,
            'boxes_used': self.number_of_boxes_used,
            'total_positions': self.total_positions,
            'weight_sum': self.weights.sum(),
            'box_coverage': self.box_coverage,
            'construction_log': self.construction_log
        }