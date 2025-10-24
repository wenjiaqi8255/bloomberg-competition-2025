"""
Adapter classes for portfolio construction.

Provides adapters to convert between different data formats and
handle compatibility between existing components and the new
portfolio construction framework.
"""

import logging
from typing import Dict, List, Any

from src.trading_system.data.stock_classifier import InvestmentBox
from src.trading_system.portfolio_construction.models.types import BoxKey

logger = logging.getLogger(__name__)


class ClassificationAdapter:
    """
    Adapter for converting between different classification formats.

    Handles conversion from InvestmentBox objects to BoxKey format and
    dictionary representations, ensuring compatibility with existing
    StockClassifier and new portfolio construction components.
    """

    @staticmethod
    def convert_investment_boxes_to_box_keys(
        investment_boxes: Dict[str, InvestmentBox],
        include_sector: bool = True
    ) -> Dict[str, BoxKey]:
        """
        Convert InvestmentBox objects to BoxKey format.

        Args:
            investment_boxes: Dictionary mapping box_key to InvestmentBox objects
            include_sector: Whether to include sector information in BoxKey

        Returns:
            Dictionary mapping stock symbols to BoxKey objects
        """
        box_keys = {}

        for box_key_str, investment_box in investment_boxes.items():
            if not hasattr(investment_box, 'stocks') or not investment_box.stocks:
                logger.debug(f"Skipping empty box: {box_key_str}")
                continue

            # Create BoxKey from InvestmentBox attributes
            try:
                box_key = BoxKey(
                    size=investment_box.size.value,
                    style=investment_box.style.value,
                    region=investment_box.region.value,
                    sector=investment_box.sector if include_sector else None
                )

                # Add classification for each stock in this box
                for stock_info in investment_box.stocks:
                    if isinstance(stock_info, dict) and 'symbol' in stock_info:
                        symbol = stock_info['symbol']
                        box_keys[symbol] = box_key
                    elif hasattr(stock_info, 'symbol'):
                        symbol = stock_info.symbol
                        box_keys[symbol] = box_key
                    else:
                        logger.warning(f"Unexpected stock info format in box {box_key_str}: {type(stock_info)}")

            except AttributeError as e:
                logger.warning(f"Failed to create BoxKey from {box_key_str}: {e}")
                continue

        logger.debug(f"Converted {len(investment_boxes)} investment boxes to {len(box_keys)} stock classifications")
        return box_keys

    @staticmethod
    def convert_to_classification_dict(box_keys: Dict[str, BoxKey]) -> Dict[str, Dict[str, str]]:
        """
        Convert BoxKey objects to dictionary format.

        This format is compatible with existing code that expects
        classification dictionaries.

        Args:
            box_keys: Dictionary mapping stock symbols to BoxKey objects

        Returns:
            Dictionary mapping stock symbols to classification dictionaries
        """
        classification_dict = {}

        for symbol, box_key in box_keys.items():
            classification_dict[symbol] = {
                'size': box_key.size,
                'style': box_key.style,
                'region': box_key.region,
                'sector': box_key.sector
            }

        return classification_dict

    @staticmethod
    def extract_box_keys_from_classification_dict(
        classifications: Dict[str, Dict[str, str]]
    ) -> Dict[str, BoxKey]:
        """
        Extract BoxKey objects from classification dictionary.

        Inverse operation of convert_to_classification_dict.

        Args:
            classifications: Dictionary mapping symbols to classification dicts

        Returns:
            Dictionary mapping stock symbols to BoxKey objects
        """
        box_keys = {}

        for symbol, classification in classifications.items():
            try:
                box_key = BoxKey(
                    size=classification.get('size', 'large'),
                    style=classification.get('style', 'growth'),
                    region=classification.get('region', 'developed'),
                    sector=classification.get('sector', 'Technology')
                )
                box_keys[symbol] = box_key
            except Exception as e:
                logger.warning(f"Failed to create BoxKey for {symbol}: {e}")
                continue

        return box_keys

    @staticmethod
    def group_stocks_by_box(symbols: List[str], box_keys: Dict[str, BoxKey]) -> Dict[BoxKey, List[str]]:
        """
        Group stocks by their box classifications.

        Args:
            symbols: List of stock symbols to group
            box_keys: Dictionary mapping symbols to BoxKey objects

        Returns:
            Dictionary mapping BoxKey to list of stock symbols in that box
        """
        box_groups = {}

        for symbol in symbols:
            if symbol not in box_keys:
                logger.debug(f"No classification found for {symbol}")
                continue

            box_key = box_keys[symbol]
            if box_key not in box_groups:
                box_groups[box_key] = []
            box_groups[box_key].append(symbol)

        logger.debug(f"Grouped {len(symbols)} stocks into {len(box_groups)} boxes")
        return box_groups

    @staticmethod
    def validate_classification_coverage(
        symbols: List[str],
        box_keys: Dict[str, BoxKey],
        min_coverage_ratio: float = 0.8
    ) -> Dict[str, Any]:
        """
        Validate classification coverage for a list of symbols.

        Args:
            symbols: List of symbols to check coverage for
            box_keys: Dictionary mapping symbols to BoxKey objects
            min_coverage_ratio: Minimum required coverage ratio

        Returns:
            Dictionary with coverage statistics and any issues
        """
        total_symbols = len(symbols)
        classified_symbols = len([s for s in symbols if s in box_keys])
        coverage_ratio = classified_symbols / total_symbols if total_symbols > 0 else 0

        unclassified_symbols = [s for s in symbols if s not in box_keys]

        result = {
            'total_symbols': total_symbols,
            'classified_symbols': classified_symbols,
            'coverage_ratio': coverage_ratio,
            'min_coverage_ratio': min_coverage_ratio,
            'unclassified_symbols': unclassified_symbols,
            'is_adequate_coverage': coverage_ratio >= min_coverage_ratio,
            'issues': []
        }

        if not result['is_adequate_coverage']:
            result['issues'].append(
                f"Insufficient classification coverage: {coverage_ratio:.2%} < {min_coverage_ratio:.2%}"
            )

        if unclassified_symbols:
            result['issues'].append(
                f"{len(unclassified_symbols)} symbols remain unclassified: {unclassified_symbols[:10]}..."
            )

        return result


class SignalAdapter:
    """
    Adapter for signal format conversion and processing.

    Handles conversion between different signal formats and provides
    utility methods for signal processing in portfolio construction.
    """

    @staticmethod
    def combine_strategy_signals(strategy_signals: Dict[str, Any], date) -> Dict[str, float]:
        """
        Combine signals from multiple strategies into a single signal dictionary.

        Args:
            strategy_signals: Dictionary mapping strategy names to signal data
            date: Date for signal combination

        Returns:
            Dictionary mapping stock symbols to combined signal values
        """
        combined_signals = {}

        for strategy_name, signals in strategy_signals.items():
            if not signals:
                logger.debug(f"No signals from strategy '{strategy_name}'")
                continue

            # Handle different signal formats
            if hasattr(signals, '__iter__') and not isinstance(signals, dict):
                # Assume it's a list of TradingSignal objects
                for signal in signals:
                    if hasattr(signal, 'symbol') and hasattr(signal, 'strength'):
                        combined_signals[signal.symbol] = signal.strength
            elif isinstance(signals, dict):
                combined_signals.update(signals)

        logger.debug(f"Combined signals from {len(strategy_signals)} strategies: {len(combined_signals)} symbols")
        return combined_signals

    @staticmethod
    def filter_signals_by_universe(signals: Dict[str, float], universe: List[str]) -> Dict[str, float]:
        """
        Filter signals to only include symbols in the universe.

        Args:
            signals: Dictionary of signals for all symbols
            universe: List of symbols to keep

        Returns:
            Filtered signals dictionary
        """
        filtered_signals = {
            symbol: signal for symbol, signal in signals.items()
            if symbol in universe
        }

        logger.debug(f"Filtered signals: {len(signals)} -> {len(filtered_signals)} symbols")
        return filtered_signals

    @staticmethod
    def normalize_signals(signals: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize signals to a consistent range.

        Args:
            signals: Raw signals dictionary

        Returns:
            Normalized signals dictionary
        """
        if not signals:
            return {}

        signal_values = list(signals.values())
        min_signal = min(signal_values)
        max_signal = max(signal_values)

        if max_signal == min_signal:
            # All signals are the same, return neutral signals
            return {symbol: 0.0 for symbol in signals}

        # Normalize to [-1, 1] range
        normalized = {}
        for symbol, signal in signals.items():
            normalized[symbol] = 2.0 * (signal - min_signal) / (max_signal - min_signal) - 1.0

        return normalized

    @staticmethod
    def validate_signals(signals: Dict[str, float]) -> Dict[str, Any]:
        """
        Validate signal dictionary for common issues.

        Args:
            signals: Signals dictionary to validate

        Returns:
            Dictionary with validation results and any issues
        """
        result = {
            'total_symbols': len(signals),
            'positive_signals': len([s for s in signals.values() if s > 0]),
            'negative_signals': len([s for s in signals.values() if s < 0]),
            'zero_signals': len([s for s in signals.values() if s == 0]),
            'signal_range': {'min': None, 'max': None},
            'issues': []
        }

        if signals:
            signal_values = list(signals.values())
            result['signal_range']['min'] = min(signal_values)
            result['signal_range']['max'] = max(signal_values)

            # Check for extreme values
            extreme_threshold = 10.0  # Arbitrary threshold for extreme signals
            extreme_signals = [s for s in signal_values if abs(s) > extreme_threshold]
            if extreme_signals:
                result['issues'].append(
                    f"Found {len(extreme_signals)} extreme signals > {extreme_threshold}"
                )

            # Check for NaN or infinite values
            invalid_signals = [(sym, val) for sym, val in signals.items()
                             if not isinstance(val, (int, float)) or not (float('-inf') < val < float('inf'))]
            if invalid_signals:
                result['issues'].append(
                    f"Found {len(invalid_signals)} invalid signal values: {invalid_signals[:5]}..."
                )

        return result