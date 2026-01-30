"""
Box sampling provider for universe selection.

Implements weak coupling design by providing box-based sampling
functionality at the data provider level, allowing portfolio
construction methods to choose between full universe or box-based
sampling approaches.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd

from .base_data_provider import ClassificationProvider
from .filters.liquidity_filter import LiquidityFilter
from trading_system.portfolio_construction.models.types import BoxKey
from trading_system.portfolio_construction.utils.adapters import ClassificationAdapter

logger = logging.getLogger(__name__)


class BoxSamplingProvider(ClassificationProvider):
    """
    Data provider that implements box-based universe sampling.

    This provider implements a weak coupling design where the sampling
    logic is separated from the portfolio construction logic. It can
    sample stocks from boxes based on different strategies while
    maintaining compatibility with existing data providers.
    """

    def __init__(self, config: Dict[str, Any], stock_classifier=None):
        """
        Initialize box sampling provider.

        Args:
            config: Configuration dictionary with sampling parameters
            stock_classifier: Optional stock classifier for box classification
        """
        super().__init__(
            max_retries=config.get('max_retries', 3),
            retry_delay=config.get('retry_delay', 1.0),
            request_timeout=config.get('request_timeout', 30),
            cache_enabled=config.get('cache_enabled', True),
            rate_limit=config.get('rate_limit', 0.5)
        )

        self.config = config
        self.stock_classifier = stock_classifier
        self.box_sampler = BoxSampler(config.get('box_sampling', {}))
        self.sampling_method = config.get('sampling_method', 'full_universe')

        # Liquidity filtering configuration
        self.min_history_days = config.get('min_history_days', 252)  # 1 year
        self.min_avg_volume = config.get('min_avg_volume', None)  # Optional volume filter

        logger.info(f"Initialized BoxSamplingProvider with method: {self.sampling_method}")

    def get_data_source(self) -> str:
        """Get the data source for this provider."""
        return "BoxSamplingProvider"

    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about this provider."""
        return {
            'provider': 'Box Sampling Provider',
            'sampling_method': self.sampling_method,
            'box_sampling_config': self.box_sampler.config,
            'min_history_days': self.min_history_days,
            'min_avg_volume': self.min_avg_volume,
            'cache_enabled': self.cache_enabled
        }

    def _fetch_raw_data(self, *args, **kwargs) -> Optional[Any]:
        """Box sampling provider doesn't fetch raw data directly."""
        return None

    def classify_items(self, items: List[str], **kwargs) -> Dict[str, Any]:
        """
        Classify items (required by ClassificationProvider interface).

        Args:
            items: List of stock symbols to classify
            **kwargs: Additional classification parameters

        Returns:
            Classification results
        """
        return self.sample_universe(items, **kwargs)

    def sample_universe(self,
                       full_universe: List[str],
                       price_data: Dict[str, pd.DataFrame],
                       signals: Dict[str, float],
                       as_of_date: datetime = None) -> List[str]:
        """
        Sample universe using configured method.

        Args:
            full_universe: Complete list of available stocks
            price_data: Historical price data for all stocks
            signals: Signal strengths for stocks
            as_of_date: Date for sampling (default: today)

        Returns:
            Sampled universe list
        """
        if as_of_date is None:
            as_of_date = datetime.now()

        logger.info(f"Sampling universe from {len(full_universe)} stocks using {self.sampling_method}")

        # Step 1: Apply basic liquidity/data quality filters using LiquidityFilter utility
        liquid_stocks = self._apply_liquidity_filtering(full_universe, price_data, as_of_date)
        logger.info(f"After liquidity filter: {len(liquid_stocks)}/{len(full_universe)} stocks")

        # Step 2: Apply sampling method
        if self.sampling_method == 'full_universe':
            return liquid_stocks
        elif self.sampling_method == 'box_based':
            return self._sample_from_boxes(liquid_stocks, price_data, signals, as_of_date)
        else:
            logger.warning(f"Unknown sampling method: {self.sampling_method}, using full universe")
            return liquid_stocks

    def _apply_liquidity_filtering(self,
                                    universe: List[str],
                                    price_data: Dict[str, pd.DataFrame],
                                    as_of_date: datetime) -> List[str]:
        """
        Apply liquidity filtering using the centralized LiquidityFilter utility.

        This method delegates to the LiquidityFilter utility class, eliminating
        code duplication and ensuring consistent filtering logic across the system.

        Args:
            universe: List of stocks to filter
            price_data: Price data for all stocks
            as_of_date: Date for filtering

        Returns:
            Filtered list of liquid stocks
        """
        # Create liquidity filter configuration based on instance settings
        liquidity_config = {
            'enabled': True,
            'min_history_days': self.min_history_days,
            'min_avg_daily_volume': self.min_avg_volume,
            'volume_lookback_days': 21  # Default to 21 days for volume calculation
        }

        # Only include filters that have non-zero thresholds
        if not self.min_avg_volume:
            liquidity_config['min_avg_daily_volume'] = 0
        if not self.min_history_days:
            liquidity_config['min_history_days'] = 0

        # Apply filtering using the utility class
        filtered_symbols = LiquidityFilter.apply_liquidity_filters(
            universe, price_data, liquidity_config
        )

        return filtered_symbols

    def get_classification_categories(self) -> Dict[str, List[str]]:
        """
        Get available classification categories (required by ClassificationProvider interface).

        Returns:
            Dictionary mapping category types to available values
        """
        return {
            'sampling_method': ['full_universe', 'box_based'],
            'box_types': ['size', 'value', 'growth', 'quality'],
            'filter_types': ['liquidity', 'market_cap', 'volume', 'price']
        }

    def _sample_from_boxes(self,
                          universe: List[str],
                          price_data: Dict[str, pd.DataFrame],
                          signals: Dict[str, float],
                          as_of_date: datetime) -> List[str]:
        """
        Sample stocks using box-based approach.

        Args:
            universe: Filtered universe to sample from
            price_data: Price data for all stocks
            signals: Signal strengths for stocks
            as_of_date: Date for sampling

        Returns:
            Box-sampled universe
        """
        if not self.stock_classifier:
            logger.warning("No stock classifier provided, falling back to full universe")
            return universe

        # Classify stocks into boxes
        logger.info("Classifying stocks into boxes for sampling...")
        try:
            investment_boxes = self.stock_classifier.classify_stocks(
                universe, price_data, as_of_date=as_of_date
            )
        except Exception as e:
            logger.error(f"Box classification failed: {e}")
            return universe

        # Convert to BoxKey format
        box_keys = ClassificationAdapter.convert_investment_boxes_to_box_keys(investment_boxes)

        if not box_keys:
            logger.warning("No valid box classifications found, falling back to full universe")
            return universe

        # Sample from boxes
        try:
            sampled_stocks = self.box_sampler.sample_from_boxes(universe, box_keys, signals)
            logger.info(f"Box sampling resulted in {len(sampled_stocks)}/{len(universe)} stocks")
            return sampled_stocks
        except Exception as e:
            logger.error(f"Box sampling failed: {e}")
            return universe


class BoxSampler:
    """
    Implements box-based stock sampling logic.

    Samples stocks from each box based on configurable strategies,
    ensuring diversification across boxes while respecting signal
    strengths.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize box sampler.

        Args:
            config: Configuration for box sampling behavior
        """
        self.config = config
        self.stocks_per_box = config.get('stocks_per_box', 5)
        self.min_stocks_per_box = config.get('min_stocks_per_box', 1)
        self.sampling_strategy = config.get('sampling_strategy', 'top_signals')
        self.min_box_coverage = config.get('min_box_coverage', 0.7)  # Target coverage of boxes

    def sample_from_boxes(self,
                          universe: List[str],
                          box_keys: Dict[str, BoxKey],
                          signals: Dict[str, float]) -> List[str]:
        """
        Sample stocks from boxes.

        Args:
            universe: Available universe of stocks
            box_keys: Box classifications for stocks
            signals: Signal strengths for stocks

        Returns:
            Sampled list of stocks
        """
        # Group stocks by boxes
        box_groups = ClassificationAdapter.group_stocks_by_box(universe, box_keys)

        if not box_groups:
            logger.warning("No box groups found for sampling")
            return universe

        logger.info(f"Sampling from {len(box_groups)} boxes")

        sampled_stocks = []
        boxes_used = 0

        for box_key, stocks_in_box in box_groups.items():
            if len(stocks_in_box) < self.min_stocks_per_box:
                logger.debug(f"Skipping box {box_key}: insufficient stocks ({len(stocks_in_box)})")
                continue

            # Sample stocks from this box
            box_sample = self._sample_from_single_box(stocks_in_box, signals)
            sampled_stocks.extend(box_sample)
            boxes_used += 1

            logger.debug(f"Sampled {len(box_sample)} stocks from box {box_key}")

        coverage_ratio = boxes_used / len(box_groups) if box_groups else 0
        logger.info(f"Box sampling complete: {len(sampled_stocks)} stocks from {boxes_used}/{len(box_groups)} boxes "
                   f"(coverage: {coverage_ratio:.1%})")

        return sampled_stocks

    def _sample_from_single_box(self,
                               stocks_in_box: List[str],
                               signals: Dict[str, float]) -> List[str]:
        """
        Sample stocks from a single box.

        Args:
            stocks_in_box: Available stocks in the box
            signals: Signal strengths for stocks

        Returns:
            Sampled stocks from this box
        """
        # Filter to only include stocks with signals
        stocks_with_signals = [s for s in stocks_in_box if s in signals]

        if not stocks_with_signals:
            # If no signals, return random sample
            import random
            n = min(self.stocks_per_box, len(stocks_in_box))
            return random.sample(stocks_in_box, n)

        # Apply sampling strategy
        if self.sampling_strategy == 'top_signals':
            return self._sample_top_signals(stocks_with_signals, signals)
        elif self.sampling_strategy == 'diversified':
            return self._sample_diversified(stocks_with_signals, signals)
        else:
            logger.warning(f"Unknown sampling strategy: {self.sampling_strategy}, using top_signals")
            return self._sample_top_signals(stocks_with_signals, signals)

    def _sample_top_signals(self,
                           stocks_with_signals: List[str],
                           signals: Dict[str, float]) -> List[str]:
        """
        Sample stocks with top signal values.

        Args:
            stocks_with_signals: Stocks that have signal values
            signals: Signal strengths

        Returns:
            Top signal stocks from the box
        """
        # Sort by signal strength (descending)
        sorted_stocks = sorted(
            stocks_with_signals,
            key=lambda s: signals.get(s, 0),
            reverse=True
        )

        # Take top N
        n = min(self.stocks_per_box, len(sorted_stocks))
        return sorted_stocks[:n]

    def _sample_diversified(self,
                           stocks_with_signals: List[str],
                           signals: Dict[str, float]) -> List[str]:
        """
        Sample stocks with consideration for diversification.

        This is a placeholder for more sophisticated sampling strategies
        that consider correlations or other diversification metrics.

        Args:
            stocks_with_signals: Stocks that have signal values
            signals: Signal strengths

        Returns:
            Diversified sample from the box
        """
        # For now, fall back to top signals
        # TODO: Implement diversification logic
        return self._sample_top_signals(stocks_with_signals, signals)

    def get_sampling_stats(self) -> Dict[str, Any]:
        """
        Get sampling statistics and configuration.

        Returns:
            Dictionary with sampling configuration and stats
        """
        return {
            'stocks_per_box': self.stocks_per_box,
            'min_stocks_per_box': self.min_stocks_per_box,
            'sampling_strategy': self.sampling_strategy,
            'min_box_coverage': self.min_box_coverage,
            'config': self.config
        }