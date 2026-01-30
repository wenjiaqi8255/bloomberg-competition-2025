"""
Services for Box-Based Portfolio Construction
===============================================

Provides dedicated service classes for distinct responsibilities within the
box-based construction process, such as classification and stock selection.
This improves modularity and adheres to the Single Responsibility Principle.
"""

import logging
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
from collections import OrderedDict

from trading_system.portfolio_construction.models.types import BoxKey
from trading_system.portfolio_construction.utils.adapters import ClassificationAdapter
from trading_system.data.stock_classifier import StockClassifier
from trading_system.portfolio_construction.interface.interfaces import IBoxSelector

logger = logging.getLogger(__name__)


class ClassificationService:
    """Handles the classification of stocks into boxes, with caching."""

    def __init__(self, stock_classifier: StockClassifier, max_cache_size: int = 100):
        self.stock_classifier = stock_classifier
        self._cache: OrderedDict[tuple, Dict[str, BoxKey]] = OrderedDict()
        self._max_cache_size = max_cache_size

    def classify_stocks(self, universe: List[str], price_data: Dict[str, pd.DataFrame],
                        as_of_date: datetime) -> Dict[str, BoxKey]:
        """
        Classify stocks into boxes with caching support.
        """
        cache_key = (as_of_date, tuple(sorted(universe)))
        
        if cache_key in self._cache:
            logger.debug(f"Classification cache hit for date {as_of_date.date()}")
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        
        investment_boxes = self.stock_classifier.classify_stocks(
            universe, price_data, as_of_date=as_of_date
        )
        box_keys = ClassificationAdapter.convert_investment_boxes_to_box_keys(
            investment_boxes, 
            include_sector=self.stock_classifier.include_sector
        )

        if len(self._cache) >= self._max_cache_size:
            self._cache.popitem(last=False)
        self._cache[cache_key] = box_keys
        
        logger.debug(f"Classified {len(box_keys)} stocks (cache miss)")
        return box_keys


class StockSelectionService:
    """Handles the selection of stocks within each box."""

    def __init__(self, box_selector: IBoxSelector, stocks_per_box: int, min_stocks_per_box: int):
        self.box_selector = box_selector
        self.stocks_per_box = stocks_per_box
        self.min_stocks_per_box = min_stocks_per_box
        self.validate_parameters()

    def validate_parameters(self) -> None:
        """Validate the service parameters."""
        if self.stocks_per_box <= 0:
            raise ValueError("stocks_per_box must be positive")
        if self.min_stocks_per_box <= 0:
            raise ValueError("min_stocks_per_box must be positive")
        if self.min_stocks_per_box > self.stocks_per_box:
            raise ValueError("min_stocks_per_box cannot exceed stocks_per_box")

    def select_stocks_for_boxes(
        self,
        box_stocks: Dict[BoxKey, List[str]],
        signals: pd.Series
    ) -> Dict[BoxKey, List[str]]:
        """
        Iterates through boxes and selects the top stocks from each.
        """
        selected_stocks_by_box = {}
        for box_key, candidates in box_stocks.items():
            if len(candidates) < self.min_stocks_per_box:
                logger.debug(f"Skipping box {box_key}: not enough candidates.")
                continue

            selected = self.box_selector.select_stocks(
                candidates, signals, self.stocks_per_box
            )
            if selected:
                selected_stocks_by_box[box_key] = selected
        
        return selected_stocks_by_box
