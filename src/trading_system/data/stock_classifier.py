"""
Stock classifier for IPS box-based allocation system.

This module classifies stocks into investment boxes based on:
- Size: Large/Mid/Small (market capitalization)
- Style: Value/Growth (momentum & volatility proxies)
- Region: Developed/Emerging markets
- Sector: GICS sector classification

Since Bloomberg data is unavailable, we use yfinance data and
technical proxies for fundamental metrics.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import yfinance as yf

from ..types.enums import DataSource
from .validation import DataValidationError
from .yfinance_provider import YFinanceProvider
from .base_data_provider import ClassificationProvider

logger = logging.getLogger(__name__)


class SizeCategory(Enum):
    """Size classification categories."""
    LARGE = "large"
    MID = "mid"
    SMALL = "small"


class StyleCategory(Enum):
    """Style classification categories."""
    VALUE = "value"
    GROWTH = "growth"


class RegionCategory(Enum):
    """Region classification categories."""
    DEVELOPED = "developed"
    EMERGING = "emerging"


class SectorCategory(Enum):
    """Sector classification categories."""
    TECHNOLOGY = "Technology"
    HEALTHCARE = "Healthcare"
    FINANCIALS = "Financials"
    CONSUMER_DISCRETIONARY = "Consumer Discretionary"
    CONSUMER_STAPLES = "Consumer Staples"
    INDUSTRIALS = "Industrials"
    ENERGY = "Energy"
    UTILITIES = "Utilities"
    REAL_ESTATE = "Real Estate"
    MATERIALS = "Materials"
    COMMUNICATION_SERVICES = "Communication Services"


class InvestmentBox:
    """Represents an investment box combining Size×Style×Region×Sector."""

    def __init__(self, size: SizeCategory, style: StyleCategory,
                 region: RegionCategory, sector: str):
        self.size = size
        self.style = style
        self.region = region
        self.sector = sector
        self.stocks = []
        self.key = f"{size.value}_{style.value}_{region.value}_{sector}"

    def add_stock(self, symbol: str, market_cap: float, score: float = 0):
        """Add a stock to this investment box."""
        self.stocks.append({
            'symbol': symbol,
            'market_cap': market_cap,
            'score': score
        })

    def get_top_stocks(self, n: int = 5) -> List[str]:
        """Get top N stocks by score."""
        sorted_stocks = sorted(self.stocks, key=lambda x: x['score'], reverse=True)
        return [stock['symbol'] for stock in sorted_stocks[:n]]

    def __str__(self):
        return f"InvestmentBox({self.key})"

    def __repr__(self):
        return f"InvestmentBox(size={self.size}, style={self.style}, region={self.region}, sector={self.sector})"


class StockClassifier(ClassificationProvider):
    """
    Stock classifier for IPS box-based allocation system.

    Classifies stocks into Size×Style×Region×Sector boxes using available data.
    Uses technical proxies when fundamental data is unavailable.
    """

    def __init__(self, config: Dict[str, Any],
                 yfinance_provider: YFinanceProvider = None,
                 max_retries: int = 3, retry_delay: float = 1.0,
                 request_timeout: int = 30, cache_enabled: bool = True):
        """
        Initialize stock classifier.

        Args:
            config: Configuration dictionary for classification rules.
            yfinance_provider: YFinance provider instance
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
            request_timeout: Request timeout in seconds
            cache_enabled: Whether to enable caching
        """
        super().__init__(
            max_retries=max_retries,
            retry_delay=retry_delay,
            request_timeout=request_timeout,
            cache_enabled=cache_enabled,
            rate_limit=0.5  # 500ms between requests
        )
        
        self.yfinance_provider = yfinance_provider or YFinanceProvider()
        self.market_cap_cache = {}
        self.sector_cache = {}

        # Load configuration for classification rules
        self._load_config(config)

    def _load_config(self, config: Dict[str, Any]):
        """Load classification rules from config."""
        logger.info("Loading StockClassifier configuration...")

        # Size config with defaults
        size_config = config.get('size_config', {})
        self.size_thresholds = {
            SizeCategory.LARGE: size_config.get('thresholds', {}).get('large', 10.0),
            SizeCategory.MID: size_config.get('thresholds', {}).get('mid', 2.0),
            SizeCategory.SMALL: 0.0
        }
        
        # Region config with defaults
        region_config = config.get('region_config', {})
        default_developed_markets = {
            'US', 'CA', 'GB', 'DE', 'FR', 'IT', 'ES', 'NL', 'CH', 'SE', 'NO', 'DK',
            'AU', 'NZ', 'JP', 'SG', 'HK', 'TW', 'KR'
        }
        self.developed_markets = set(region_config.get('developed_markets', default_developed_markets))

        # Sector config with defaults
        sector_config = config.get('sector_config', {})
        default_sector_mappings = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA', 'ADBE', 'INTC'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'T', 'BMY', 'AMGN'],
            'Financials': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BRK-B', 'BLK'],
            'Consumer Discretionary': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'LOW', 'TGT'],
            'Consumer Staples': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'MDLZ', 'CL', 'GIS'],
            'Industrials': ['CAT', 'GE', 'HON', 'UNP', 'BA', 'MMM', 'DE', 'UPS'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PSX', 'VLO', 'MPC'],
            'Utilities': ['NEE', 'D', 'SO', 'DUK', 'AEP', 'EXC', 'SRE', 'PEG'],
            'Real Estate': ['SPG', 'AMT', 'PLD', 'EQIX', 'PSA', 'CCI', 'DLR', 'O'],
            'Materials': ['LIN', 'BHP', 'RIO', 'DD', 'APD', 'ECL', 'FCX', 'NUE'],
            'Communication Services': ['GOOGL', 'META', 'T', 'VZ', 'DIS', 'NFLX', 'CMCSA', 'TMUS']
        }
        self.sector_mappings = sector_config.get('mappings', default_sector_mappings)
        
        logger.info(f"Size thresholds loaded: LARGE > ${self.size_thresholds[SizeCategory.LARGE]}B")
    
    def get_data_source(self) -> DataSource:
        """Get the data source enum for this provider."""
        return DataSource.YFINANCE  # Uses YFinance for data
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about this data provider."""
        return {
            'provider': 'Stock Classifier',
            'data_source': DataSource.YFINANCE.value,
            'description': 'Stock classification system for IPS box-based allocation',
            'classification_dimensions': ['Size', 'Style', 'Region', 'Sector'],
            'size_categories': [cat.value for cat in SizeCategory],
            'style_categories': [cat.value for cat in StyleCategory],
            'region_categories': [cat.value for cat in RegionCategory],
            'sector_categories': [cat.value for cat in SectorCategory],
            'classification_method': 'Technical proxies for fundamental factors',
            'cache_enabled': self.cache_enabled
        }
    
    def _fetch_raw_data(self, *args, **kwargs) -> Optional[Any]:
        """Fetch raw data from YFinance API."""
        # This method is called by the base class's _fetch_with_retry
        # The actual fetching logic is in the specific methods
        pass

    def classify_items(self, items: List[str], **kwargs) -> Dict[str, Any]:
        """
        Classify items into categories.
        
        Args:
            items: List of stock symbols to classify
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with classification results
        """
        return self.classify_stocks(items, **kwargs)
    
    def get_classification_categories(self) -> Dict[str, List[str]]:
        """
        Get available classification categories.
        
        Returns:
            Dictionary mapping category types to available values
        """
        return {
            'size': [cat.value for cat in SizeCategory],
            'style': [cat.value for cat in StyleCategory],
            'region': [cat.value for cat in RegionCategory],
            'sector': [cat.value for cat in SectorCategory]
        }

    def classify_stocks(self, symbols: List[str],
                       price_data: Dict[str, pd.DataFrame] = None,
                       as_of_date: datetime = None) -> Dict[str, InvestmentBox]:
        """
        Classify stocks into investment boxes.

        Args:
            symbols: List of stock symbols to classify
            price_data: Optional price data dictionary
            as_of_date: Date for classification (default: today)

        Returns:
            Dictionary mapping box keys to InvestmentBox objects
        """
        if as_of_date is None:
            as_of_date = datetime.now()

        logger.info(f"Classifying {len(symbols)} stocks as of {as_of_date}")

        # Get price data if not provided
        if price_data is None:
            price_data = self.yfinance_provider.get_historical_data(
                symbols,
                start_date=as_of_date - timedelta(days=365),
                end_date=as_of_date
            )

        boxes = {}

        for symbol in symbols:
            try:
                if symbol not in price_data or price_data[symbol] is None:
                    logger.debug(f"No price data for {symbol}")
                    continue

                # Classify the stock
                classification = self._classify_single_stock(
                    symbol, price_data[symbol], as_of_date
                )

                if classification:
                    box_key = classification['box_key']
                    if box_key not in boxes:
                        boxes[box_key] = InvestmentBox(
                            size=classification['size'],
                            style=classification['style'],
                            region=classification['region'],
                            sector=classification['sector']
                        )

                    # Add stock to box with classification score
                    boxes[box_key].add_stock(
                        symbol=symbol,
                        market_cap=classification['market_cap'],
                        score=classification['score']
                    )

            except Exception as e:
                logger.warning(f"Failed to classify {symbol}: {e}")
                continue

        logger.info(f"Created {len(boxes)} investment boxes")
        return boxes

    def _classify_single_stock(self, symbol: str, price_data: pd.DataFrame,
                              as_of_date: datetime) -> Optional[Dict]:
        """
        Classify a single stock into an investment box.

        Args:
            symbol: Stock symbol
            price_data: Price data for the stock
            as_of_date: Classification date

        Returns:
            Dictionary with classification results or None if classification fails
        """
        try:
            # Get market data as of classification date
            data_up_to_date = price_data[price_data.index <= as_of_date]
            if len(data_up_to_date) < 20:  # Need minimum data
                # Try to fetch more data if current data is insufficient
                logger.debug(f"Insufficient data for {symbol} ({len(data_up_to_date)} < 20), fetching more...")
                try:
                    more_data = self.yfinance_provider.get_historical_data(
                        [symbol],
                        start_date=as_of_date - timedelta(days=365),
                        end_date=as_of_date
                    )
                    if symbol in more_data and more_data[symbol] is not None:
                        price_data = more_data[symbol]
                        data_up_to_date = price_data[price_data.index <= as_of_date]
                        logger.debug(f"Fetched {len(data_up_to_date)} rows for {symbol}")
                    else:
                        logger.warning(f"Could not fetch additional data for {symbol}")
                        return None
                except Exception as e:
                    logger.warning(f"Failed to fetch more data for {symbol}: {e}")
                    return None

                # Still insufficient after retry
                if len(data_up_to_date) < 20:
                    logger.warning(f"Still insufficient data for {symbol} after retry: {len(data_up_to_date)} < 20")
                    return None

            # 1. Classify by Size (market capitalization)
            size_category, market_cap = self._classify_by_size(symbol, data_up_to_date)

            # 2. Classify by Style (value vs growth using fundamental data + technical proxies)
            style_category, style_score = self._classify_by_style(symbol, data_up_to_date)

            # 3. Classify by Region
            region_category = self._classify_by_region(symbol)

            # 4. Classify by Sector
            sector = self._classify_by_sector(symbol)

            # Calculate overall classification score
            overall_score = self._calculate_classification_score(
                market_cap, style_score, data_up_to_date
            )

            return {
                'symbol': symbol,
                'size': size_category,
                'style': style_category,
                'region': region_category,
                'sector': sector,
                'market_cap': market_cap,
                'score': overall_score,
                'box_key': f"{size_category.value}_{style_category.value}_{region_category.value}_{sector}"
            }

        except Exception as e:
            logger.debug(f"Classification failed for {symbol}: {e}")
            return None

    def _classify_by_size(self, symbol: str, price_data: pd.DataFrame) -> Tuple[SizeCategory, float]:
        """Classify stock by size using market capitalization."""
        try:
            # Get market cap
            market_cap = self._get_market_cap(symbol, price_data)

            if market_cap >= self.size_thresholds[SizeCategory.LARGE]:
                return SizeCategory.LARGE, market_cap
            elif market_cap >= self.size_thresholds[SizeCategory.MID]:
                return SizeCategory.MID, market_cap
            else:
                return SizeCategory.SMALL, market_cap

        except Exception as e:
            logger.debug(f"Size classification failed for {symbol}: {e}")
            return SizeCategory.MID, 5.0  # Default to mid-cap

    def _classify_by_style(self, symbol: str, price_data: pd.DataFrame) -> Tuple[StyleCategory, float]:
        """
        Classify stock by style using fundamental data when available,
        supplemented with technical proxies for value/growth.

        Fundamental indicators:
        - P/E ratio: Low PE = value, high PE = growth
        - P/B ratio: Low PB = value, high PB = growth

        Technical proxies (when fundamental data unavailable):
        - Momentum as value proxy (contrarian indicator)
        - Volatility as growth proxy (growth stocks tend to be more volatile)
        """
        try:
            # 1. Try to get fundamental data first
            fundamental_score = self._get_fundamental_style_score(symbol)

            if fundamental_score is not None:
                # Use fundamental data for classification
                if fundamental_score < 0.3:
                    return StyleCategory.VALUE, fundamental_score
                elif fundamental_score > 0.7:
                    return StyleCategory.GROWTH, fundamental_score
                else:
                    # Mixed style - use technical indicators as tiebreaker
                    technical_score = self._get_technical_style_score(price_data)
                    combined_score = (fundamental_score + technical_score) / 2
                    return StyleCategory.GROWTH if combined_score > 0.5 else StyleCategory.VALUE, combined_score

            # Fallback to technical indicators
            technical_score = self._get_technical_style_score(price_data)
            return StyleCategory.GROWTH if technical_score > 0.5 else StyleCategory.VALUE, technical_score

        except Exception as e:
            logger.debug(f"Style classification failed for {symbol}: {e}")
            return StyleCategory.GROWTH, 0.5  # Default to growth

    def _get_fundamental_style_score(self, symbol: str) -> Optional[float]:
        """
        Get style score based on fundamental indicators.

        Returns:
            Float between 0 (value) and 1 (growth), or None if data unavailable
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            pe_ratio = info.get('trailingPE')
            pb_ratio = info.get('priceToBook')
            book_value = info.get('bookValue')

            if pe_ratio is None and pb_ratio is None:
                return None

            # Calculate style scores (0 = value, 1 = growth)
            pe_score = 0
            pb_score = 0

            if pe_ratio is not None:
                # PE ratio: < 15 = value, > 25 = growth
                if pe_ratio < 15:
                    pe_score = 0.0
                elif pe_ratio > 25:
                    pe_score = 1.0
                else:
                    pe_score = (pe_ratio - 15) / 10  # Linear interpolation

            if pb_ratio is not None:
                # PB ratio: < 1.5 = value, > 3.5 = growth
                if pb_ratio < 1.5:
                    pb_score = 0.0
                elif pb_ratio > 3.5:
                    pb_score = 1.0
                else:
                    pb_score = (pb_ratio - 1.5) / 2.0  # Linear interpolation

            # Combine scores (average available indicators)
            if pe_score > 0 and pb_score > 0:
                return (pe_score + pb_score) / 2
            elif pe_score > 0:
                return pe_score
            elif pb_score > 0:
                return pb_score
            else:
                return None

        except Exception as e:
            logger.debug(f"Fundamental style score calculation failed for {symbol}: {e}")
            return None

    def _get_technical_style_score(self, price_data: pd.DataFrame) -> float:
        """
        Get style score based on technical indicators (fallback method).

        Returns:
            Float between 0 (value) and 1 (growth)
        """
        try:
            returns = price_data['Close'].pct_change().dropna()

            if len(returns) < 20:
                return 0.5  # Default

            # 1. Momentum indicators (value proxy - low momentum = value)
            short_term_momentum = returns.tail(21).mean()  # 1 month

            # Value score: negative momentum (contrarian)
            momentum_score = max(0, min(1, -short_term_momentum * 50 + 0.5))  # Normalize to 0-1

            # 2. Volatility (growth proxy - high volatility = growth)
            volatility = returns.std() * np.sqrt(252)
            volatility_score = max(0, min(1, volatility / 0.4))  # Normalize by 40% vol

            # 3. Price momentum relative to long-term trend
            current_price = price_data['Close'].iloc[-1]
            sma_200 = price_data['Close'].rolling(200).mean().iloc[-1]
            price_trend_score = max(0, min(1, (sma_200 - current_price) / sma_200 + 0.5))  # Normalize

            # Combine scores
            technical_score = (momentum_score + volatility_score + price_trend_score) / 3
            return technical_score

        except Exception as e:
            logger.debug(f"Style classification failed: {e}")
            return StyleCategory.GROWTH, 0.5

    def _classify_by_region(self, symbol: str) -> RegionCategory:
        """Classify stock by region using ticker suffix."""
        try:
            # Extract country code from symbol (common patterns)
            symbol_upper = symbol.upper()

            # Check for direct country suffixes
            if '.' in symbol_upper:
                country_code = symbol_upper.split('.')[-1]
                if country_code in self.developed_markets:
                    return RegionCategory.DEVELOPED
                else:
                    return RegionCategory.EMERGING

            # Check for common patterns
            if any(symbol_upper.endswith(suffix) for suffix in ['.L', '.TO', '.DE', '.PA']):
                return RegionCategory.DEVELOPED
            elif any(symbol_upper.endswith(suffix) for suffix in ['.MX', '.BA', '.SA', '.JK']):
                return RegionCategory.EMERGING

            # Default to developed for US stocks
            return RegionCategory.DEVELOPED

        except Exception as e:
            logger.debug(f"Region classification failed for {symbol}: {e}")
            return RegionCategory.DEVELOPED

    def _classify_by_sector(self, symbol: str) -> str:
        """Classify stock by sector using mappings."""
        try:
            symbol_upper = symbol.upper()

            # Check direct mappings
            for sector, sector_symbols in self.sector_mappings.items():
                if symbol_upper in sector_symbols:
                    return sector

            # Try to get sector from yfinance
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if 'sector' in info and info['sector']:
                    return info['sector']
            except:
                pass

            # Fallback classification based on symbol patterns
            if any(x in symbol_upper for x in ['TECH', 'SOFT', 'COMP']):
                return 'Technology'
            elif any(x in symbol_upper for x in ['PHARMA', 'BIO', 'HEALTH']):
                return 'Healthcare'
            elif any(x in symbol_upper for x in ['BANK', 'FIN', 'INSUR']):
                return 'Financials'
            elif any(x in symbol_upper for x in ['ENERGY', 'OIL', 'GAS']):
                return 'Energy'
            elif any(x in symbol_upper for x in ['RETAIL', 'CONSUMER']):
                return 'Consumer Discretionary'
            else:
                return 'Industrials'  # Default

        except Exception as e:
            logger.debug(f"Sector classification failed for {symbol}: {e}")
            return 'Industrials'

    def _get_market_cap(self, symbol: str, price_data: pd.DataFrame) -> float:
        """Get market capitalization using available data."""
        try:
            # Check cache first
            if symbol in self.market_cap_cache:
                return self.market_cap_cache[symbol]

            # Try to get from yfinance
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                if 'marketCap' in info and info['marketCap']:
                    market_cap = info['marketCap'] / 1e9  # Convert to billions
                    self.market_cap_cache[symbol] = market_cap
                    return market_cap
            except:
                pass

            # Estimate from price data and typical share counts
            if len(price_data) > 0:
                current_price = price_data['Close'].iloc[-1]

                # Estimate shares outstanding based on typical patterns
                # This is a rough approximation
                if symbol.upper() in ['AAPL', 'MSFT', 'GOOGL', 'AMZN']:
                    shares_billions = 15.0  # Large tech
                elif symbol.upper() in ['JPM', 'BAC', 'WFC']:
                    shares_billions = 10.0  # Large banks
                elif symbol.upper().startswith('SPY') or symbol.upper().startswith('QQQ'):
                    shares_billions = 0.1  # ETF shares
                else:
                    shares_billions = 1.0  # Default assumption

                market_cap = current_price * shares_billions / 1e9  # Convert to billions
                self.market_cap_cache[symbol] = market_cap
                return market_cap

            # Default estimate
            return 5.0  # $5B default

        except Exception as e:
            logger.debug(f"Market cap estimation failed for {symbol}: {e}")
            return 5.0  # Default to mid-cap range

    def _calculate_classification_score(self, market_cap: float, style_score: float,
                                      price_data: pd.DataFrame) -> float:
        """Calculate overall classification score."""
        try:
            # Market cap score (larger = higher score, but capped)
            cap_score = min(1.0, np.log10(market_cap + 1) / 3.0)

            # Volatility score (moderate volatility preferred)
            returns = price_data['Close'].pct_change().dropna()
            if len(returns) >= 20:
                volatility = returns.std() * np.sqrt(252)
                # Ideal volatility around 20-30%
                vol_score = 1.0 - abs(volatility - 0.25) / 0.25
                vol_score = max(0, min(1, vol_score))
            else:
                vol_score = 0.5

            # Liquidity score (higher volume = better)
            volume_score = 0.5  # Default
            if 'Volume' in price_data.columns and len(price_data) >= 20:
                recent_volume = price_data['Volume'].tail(20).mean()
                # Score based on volume magnitude (very rough estimate)
                volume_score = min(1.0, recent_volume / 1e6)  # Normalize by 1M shares

            # Combine scores
            overall_score = (cap_score * 0.3 + style_score * 0.4 +
                           vol_score * 0.2 + volume_score * 0.1)

            return max(0, min(1, overall_score))

        except Exception as e:
            logger.debug(f"Score calculation failed: {e}")
            return 0.5

    def get_box_summary(self, boxes: Dict[str, InvestmentBox]) -> Dict:
        """Get summary statistics of investment boxes."""
        summary = {
            'total_boxes': len(boxes),
            'total_stocks': sum(len(box.stocks) for box in boxes.values()),
            'boxes_by_size': {},
            'boxes_by_style': {},
            'boxes_by_region': {},
            'sectors_covered': set(),
            'largest_boxes': [],
            'smallest_boxes': []
        }

        for box in boxes.values():
            # Count by size
            size_key = box.size.value
            summary['boxes_by_size'][size_key] = summary['boxes_by_size'].get(size_key, 0) + 1

            # Count by style
            style_key = box.style.value
            summary['boxes_by_style'][style_key] = summary['boxes_by_style'].get(style_key, 0) + 1

            # Count by region
            region_key = box.region.value
            summary['boxes_by_region'][region_key] = summary['boxes_by_region'].get(region_key, 0) + 1

            # Track sectors
            summary['sectors_covered'].add(box.sector)

        # Find largest and smallest boxes
        box_sizes = [(key, len(box.stocks)) for key, box in boxes.items()]
        summary['largest_boxes'] = sorted(box_sizes, key=lambda x: x[1], reverse=True)[:5]
        summary['smallest_boxes'] = sorted(box_sizes, key=lambda x: x[1])[:5]

        summary['sectors_covered'] = len(summary['sectors_covered'])

        return summary

    def optimize_box_structure(self, boxes: Dict[str, InvestmentBox],
                             min_stocks_per_box: int = 2,
                             max_boxes: int = 30) -> Dict[str, InvestmentBox]:
        """
        Optimize box structure by merging small boxes.

        Args:
            boxes: Original investment boxes
            min_stocks_per_box: Minimum stocks required per box
            max_boxes: Maximum number of boxes allowed

        Returns:
            Optimized boxes dictionary
        """
        if len(boxes) <= max_boxes:
            return boxes

        # Sort boxes by stock count (ascending)
        sorted_boxes = sorted(boxes.items(), key=lambda x: len(x[1].stocks))

        optimized_boxes = {}
        merged_count = 0

        for box_key, box in sorted_boxes:
            if len(box.stocks) >= min_stocks_per_box and len(optimized_boxes) < max_boxes:
                optimized_boxes[box_key] = box
            elif len(optimized_boxes) < max_boxes:
                # Merge with nearest box
                self._merge_box(box, optimized_boxes)
                merged_count += 1

        logger.info(f"Optimized boxes: {len(boxes)} -> {len(optimized_boxes)} "
                   f"(merged {merged_count} boxes)")
        return optimized_boxes

    def _merge_box(self, box: InvestmentBox, target_boxes: Dict[str, InvestmentBox]):
        """Merge a box into the most similar existing box."""
        # Find similar box (same size, style, region if possible)
        best_match = None
        best_similarity = 0

        for target_key, target_box in target_boxes.items():
            similarity = 0
            if box.size == target_box.size:
                similarity += 3
            if box.style == target_box.style:
                similarity += 2
            if box.region == target_box.region:
                similarity += 1

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = target_key

        if best_match:
            # Merge stocks into the best match
            target_boxes[best_match].stocks.extend(box.stocks)
            logger.debug(f"Merged {box.key} into {best_match}")
        else:
            # Add as new box if space allows
            if len(target_boxes) < 30:
                target_boxes[box.key] = box

    def classify_stock(self, symbol: str, price_data: pd.DataFrame = None,
                   as_of_date: datetime = None) -> Dict:
        """
        Classify a single stock (wrapper for _classify_single_stock).

        Args:
            symbol: Stock symbol to classify
            price_data: Optional price data for the stock
            as_of_date: Date for classification

        Returns:
            Dictionary with classification results
        """
        if price_data is None:
            # Get price data if not provided
            if as_of_date is None:
                as_of_date = datetime.now()

            price_data = self.yfinance_provider.get_historical_data(
                [symbol],
                start_date=as_of_date - timedelta(days=365),
                end_date=as_of_date
            )

        if symbol not in price_data or price_data[symbol] is None:
            raise ValueError(f"No price data available for {symbol}")

        return self._classify_single_stock(symbol, price_data[symbol], as_of_date or datetime.now())

    def get_classification_info(self) -> Dict:
        """Get information about the classification system."""
        return {
            'size_categories': [cat.value for cat in SizeCategory],
            'style_categories': [cat.value for cat in StyleCategory],
            'region_categories': [cat.value for cat in RegionCategory],
            'sector_categories': [cat.value for cat in SectorCategory],
            'sectors': list(self.sector_mappings.keys()),
            'size_thresholds': {cat.value: threshold for cat, threshold in self.size_thresholds.items()},
            'developed_markets': self.developed_markets,
            'classification_method': 'Technical proxies for fundamental factors',
            'data_sources': ['yfinance', 'technical indicators']
        }