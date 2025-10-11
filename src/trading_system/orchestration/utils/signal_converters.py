"""
Signal Conversion Utilities
==========================

Pure functions for converting between different signal representations.
Extracted from SystemOrchestrator to eliminate duplication and improve reusability.
"""

import logging
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd

from ...types.signals import TradingSignal

logger = logging.getLogger(__name__)


class SignalConverters:
    """
    Pure function utilities for signal conversion.
    
    These methods are stateless and can be used across different components
    without side effects, following the Single Responsibility Principle.
    """
    
    @staticmethod
    def convert_signals_to_dataframes(strategy_signals: Dict[str, List[TradingSignal]], 
                                    date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Convert TradingSignal objects to DataFrames for MetaModel compatibility.

        Args:
            strategy_signals: Dictionary mapping strategy names to lists of TradingSignal objects
            date: Current date for signal generation

        Returns:
            Dictionary mapping strategy names to DataFrames with expected returns
        """
        converted_signals = {}

        for strategy_name, signals in strategy_signals.items():
            if not signals:
                logger.warning(f"No signals from strategy '{strategy_name}'")
                continue

            # Create a DataFrame with signal values
            signals_data = {}
            for signal in signals:
                if hasattr(signal, 'symbol') and hasattr(signal, 'strength'):
                    signals_data[signal.symbol] = signal.strength

            if signals_data:
                # Create DataFrame with single row for the current date
                signal_df = pd.DataFrame([signals_data], index=[date])
                converted_signals[strategy_name] = signal_df
                logger.debug(f"Converted {len(signals_data)} signals from '{strategy_name}' to DataFrame")
            else:
                logger.warning(f"No valid signal data found for strategy '{strategy_name}'")

        return converted_signals

    @staticmethod
    def convert_investment_boxes_to_dict(investment_boxes: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """
        Convert InvestmentBox objects to dictionary format for PortfolioOptimizer compatibility.

        Note: StockClassifier returns Dict[box_key, InvestmentBox], but we need
        Dict[symbol, classification_dict]. We need to extract symbols from InvestmentBox.stocks.

        Args:
            investment_boxes: Dictionary mapping box_keys to InvestmentBox objects

        Returns:
            Dictionary mapping stock symbols to classification dictionaries
        """
        classifications_dict = {}

        for box_key, investment_box in investment_boxes.items():
            if hasattr(investment_box, 'stocks') and investment_box.stocks:
                # Extract classification info from the InvestmentBox
                if hasattr(investment_box, 'size') and hasattr(investment_box, 'style') and hasattr(investment_box, 'region') and hasattr(investment_box, 'sector'):
                    classification_info = {
                        'size': investment_box.size.value if hasattr(investment_box.size, 'value') else str(investment_box.size),
                        'style': investment_box.style.value if hasattr(investment_box.style, 'value') else str(investment_box.style),
                        'region': investment_box.region.value if hasattr(investment_box.region, 'value') else str(investment_box.region),
                        'sector': investment_box.sector.name if hasattr(investment_box.sector, 'name') else str(investment_box.sector)
                    }
                else:
                    # Fallback classification
                    classification_info = {
                        'size': 'large',
                        'style': 'growth',
                        'region': 'developed',
                        'sector': 'Unknown'
                    }

                # Add classification info for each stock in this box
                for stock_info in investment_box.stocks:
                    if isinstance(stock_info, dict) and 'symbol' in stock_info:
                        symbol = stock_info['symbol']
                        classifications_dict[symbol] = classification_info
                        logger.debug(f"Classified {symbol} as {classification_info}")
                    elif hasattr(stock_info, 'symbol'):
                        symbol = stock_info.symbol
                        classifications_dict[symbol] = classification_info
                        logger.debug(f"Classified {symbol} as {classification_info}")
            else:
                # Handle unexpected format - box_key might be the symbol directly
                logger.warning(f"Unexpected investment box format for {box_key}: {type(investment_box)}")
                # Try to extract symbol from box_key as fallback
                if '_' not in box_key:  # Likely a symbol, not a box_key
                    classifications_dict[box_key] = {
                        'size': 'large',
                        'style': 'growth',
                        'region': 'developed',
                        'sector': 'Unknown'
                    }

        logger.debug(f"Converted {len(classifications_dict)} stocks to dictionary format")
        return classifications_dict

    @staticmethod
    def extract_signal_strengths(signals: List[TradingSignal]) -> Dict[str, float]:
        """
        Extract signal strengths as a dictionary mapping symbols to strengths.
        
        Args:
            signals: List of TradingSignal objects
            
        Returns:
            Dictionary mapping symbols to signal strengths
        """
        signal_dict = {}
        for signal in signals:
            if hasattr(signal, 'symbol') and hasattr(signal, 'strength'):
                signal_dict[signal.symbol] = signal.strength
        return signal_dict

    @staticmethod
    def filter_signals_by_strength(signals: List[TradingSignal], 
                                 min_strength: float = 0.01) -> List[TradingSignal]:
        """
        Filter signals by minimum strength threshold.
        
        Args:
            signals: List of TradingSignal objects
            min_strength: Minimum signal strength threshold
            
        Returns:
            Filtered list of signals
        """
        return [signal for signal in signals if signal.strength >= min_strength]

    @staticmethod
    def sort_signals_by_strength(signals: List[TradingSignal], 
                               descending: bool = True) -> List[TradingSignal]:
        """
        Sort signals by strength.
        
        Args:
            signals: List of TradingSignal objects
            descending: Whether to sort in descending order (highest first)
            
        Returns:
            Sorted list of signals
        """
        return sorted(signals, key=lambda s: s.strength, reverse=descending)
