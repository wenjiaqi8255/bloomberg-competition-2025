"""
Box Feature Provider for generating box classification dummy variables.

This module provides functionality to generate box classification features
that can be used in machine learning models to capture the effects of
different investment boxes on stock returns.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass

from ..data.stock_classifier import (
    StockClassifier, SizeCategory, StyleCategory,
    RegionCategory, SectorCategory, InvestmentBox
)
from ..portfolio_construction.models.types import BoxKey
from ..portfolio_construction.utils.adapters import ClassificationAdapter

logger = logging.getLogger(__name__)


@dataclass
class BoxFeatureConfig:
    """Configuration for box feature generation."""
    enabled: bool = True
    size_categories: bool = True
    style_categories: bool = True
    region_categories: bool = True
    sector_categories: bool = True
    encoding_method: str = "one_hot"  # "one_hot" or "target_encoding"
    handle_unknown: str = "ignore"  # "ignore" or "fallback"

    # Stock classifier configuration
    stock_classifier_config: Optional[Dict[str, Any]] = None


class BoxFeatureProvider:
    """
    Provider for generating box classification dummy variables.

    This class leverages the existing StockClassifier to generate
    one-hot encoded features for different box classifications,
    allowing ML models to capture box-specific effects.
    
    Now supports fit/transform pattern to prevent data leakage:
    - fit(): Learn box classifications from training data
    - transform(): Use learned classifications for backtesting
    """

    def __init__(self, config: BoxFeatureConfig, stock_classifier: Optional[StockClassifier] = None):
        """
        Initialize box feature provider.

        Args:
            config: Configuration for box feature generation
            stock_classifier: Optional stock classifier instance
        """
        self.config = config
        self.stock_classifier = stock_classifier
        self._box_categories_cache = None
        self._classification_cache = {}
        
        # Fit/Transform state management
        self._is_fitted = False
        self._fixed_classifications = {}  # Store training-period classifications

        # Initialize stock classifier if not provided
        if self.stock_classifier is None and config.enabled:
            classifier_config = config.stock_classifier_config or {}
            self.stock_classifier = StockClassifier(classifier_config)

        logger.info(f"Initialized BoxFeatureProvider with {self._get_enabled_categories()}")

    def fit(self, price_data: Dict[str, pd.DataFrame], train_end_date: datetime,
            symbols: Optional[List[str]] = None):
        """
        Fit the box feature provider on training data.
        
        This method learns box classifications at the end of the training period
        and stores them for use during backtesting, preventing data leakage.
        
        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            train_end_date: End date of training period (classification date)
            symbols: Optional list of symbols to classify (uses all if None)
        """
        if not self.config.enabled:
            logger.info("Box features are disabled, skipping fit")
            self._is_fitted = True
            return
        
        if symbols is None:
            symbols = list(price_data.keys())
        
        logger.info(f"Fitting BoxFeatureProvider on {len(symbols)} symbols as of {train_end_date}")
        
        # Learn box classifications at training end date
        self._fixed_classifications = self._classify_stocks(symbols, price_data, train_end_date)
        
        if not self._fixed_classifications:
            logger.warning("No box classifications learned during fitting")
            return
        
        self._is_fitted = True
        logger.info(f"BoxFeatureProvider fitted successfully on {len(self._fixed_classifications)} symbols")
        
        # Log some examples of learned classifications
        for symbol, classification in list(self._fixed_classifications.items())[:3]:
            logger.debug(f"Learned classification for {symbol}: {classification}")
    
    def transform(self, price_data: Dict[str, pd.DataFrame],
                  symbols: Optional[List[str]] = None,
                  as_of_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Transform data using learned box classifications.
        
        This method uses the classifications learned during fitting,
        preventing data leakage during backtesting.
        
        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            symbols: Optional list of symbols to process (uses all if None)
            as_of_date: Ignored for transform (uses learned classifications)
            
        Returns:
            DataFrame with box classification features using learned classifications
        """
        if not self.config.enabled:
            logger.info("Box features are disabled, returning empty DataFrame")
            return pd.DataFrame()
        
        if not self._is_fitted:
            raise ValueError("Must fit before transform")
        
        if symbols is None:
            symbols = list(price_data.keys())
        
        logger.debug(f"Transforming box features for {len(symbols)} symbols using learned classifications")
        
        # Generate features using learned classifications
        all_features = []
        
        for symbol in symbols:
            if symbol not in self._fixed_classifications:
                logger.debug(f"No learned classification for {symbol}, skipping")
                continue
            
            symbol_features = self._generate_symbol_features(
                symbol, self._fixed_classifications[symbol], price_data.get(symbol, pd.DataFrame())
            )
            if symbol_features is not None and not symbol_features.empty:
                all_features.append(symbol_features)
        
        # Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=0)
            combined_features.sort_index(inplace=True)
            logger.debug(f"Transformed box features: {combined_features.shape}")
            return combined_features
        else:
            logger.warning("No box features generated during transform")
            return pd.DataFrame()

    def _get_enabled_categories(self) -> Dict[str, bool]:
        """Get enabled feature categories."""
        return {
            'size': self.config.size_categories,
            'style': self.config.style_categories,
            'region': self.config.region_categories,
            'sector': self.config.sector_categories
        }

    def generate_box_features(self,
                            price_data: Dict[str, pd.DataFrame],
                            symbols: Optional[List[str]] = None,
                            as_of_date: Optional[datetime] = None,
                            use_transform: bool = False) -> pd.DataFrame:
        """
        Generate box classification features for all symbols.

        Args:
            price_data: Dictionary mapping symbols to price DataFrames
            symbols: Optional list of symbols to process (uses all if None)
            as_of_date: Date for classification (default: latest date in data)
            use_transform: If True, use transform method (for backtesting)

        Returns:
            DataFrame with MultiIndex (date, symbol) and box feature columns
        """
        if not self.config.enabled:
            logger.info("Box features are disabled, returning empty DataFrame")
            return pd.DataFrame()

        if use_transform and self._is_fitted:
            # Use transform method for backtesting (prevents data leakage)
            return self.transform(price_data, symbols, as_of_date)

        # Original method for training (with data leakage warning)
        if symbols is None:
            symbols = list(price_data.keys())

        if as_of_date is None:
            # Use the latest common date across all symbols
            all_dates = []
            for symbol in symbols:
                if symbol in price_data and price_data[symbol] is not None:
                    all_dates.extend(price_data[symbol].index.tolist())
            if all_dates:
                as_of_date = max(all_dates)
            else:
                as_of_date = datetime.now()

        logger.info(f"Generating box features for {len(symbols)} symbols as of {as_of_date}")

        # Step 1: Classify stocks into boxes
        box_classifications = self._classify_stocks(symbols, price_data, as_of_date)

        if not box_classifications:
            logger.warning("No box classifications generated, returning empty DataFrame")
            return pd.DataFrame()

        # Step 2: Generate features for each symbol
        all_features = []

        for symbol in symbols:
            if symbol not in box_classifications:
                logger.debug(f"No classification for {symbol}, skipping")
                continue

            symbol_features = self._generate_symbol_features(
                symbol, box_classifications[symbol], price_data[symbol]
            )
            if symbol_features is not None and not symbol_features.empty:
                all_features.append(symbol_features)

        # Step 3: Combine all features
        if all_features:
            combined_features = pd.concat(all_features, axis=0)
            combined_features.sort_index(inplace=True)
            logger.info(f"Generated box features: {combined_features.shape}")
            return combined_features
        else:
            logger.warning("No box features generated for any symbols")
            return pd.DataFrame()

    def _classify_stocks(self,
                        symbols: List[str],
                        price_data: Dict[str, pd.DataFrame],
                        as_of_date: datetime) -> Dict[str, Dict[str, Any]]:
        """
        Classify stocks into box categories.

        Args:
            symbols: List of symbols to classify
            price_data: Price data for all symbols
            as_of_date: Classification date

        Returns:
            Dictionary mapping symbols to classification dictionaries
        """
        if self.stock_classifier is None:
            logger.error("No stock classifier available")
            return {}

        # Check cache first
        cache_key = f"classification_{as_of_date.strftime('%Y%m%d')}_{len(symbols)}"
        if cache_key in self._classification_cache:
            logger.debug("Using cached classification results")
            return self._classification_cache[cache_key]

        try:
            # Use existing stock classifier
            investment_boxes = self.stock_classifier.classify_stocks(
                symbols, price_data, as_of_date=as_of_date
            )

            # Convert to symbol-level classifications
            classifications = {}
            for box_key, investment_box in investment_boxes.items():
                for stock_info in investment_box.stocks:
                    if isinstance(stock_info, dict):
                        symbol = stock_info['symbol']
                    elif hasattr(stock_info, 'symbol'):
                        symbol = stock_info.symbol
                    else:
                        continue

                    classifications[symbol] = {
                        'size': investment_box.size.value,
                        'style': investment_box.style.value,
                        'region': investment_box.region.value,
                        'sector': investment_box.sector,
                        'box_key': box_key
                    }

            # Cache results
            self._classification_cache[cache_key] = classifications
            logger.debug(f"Classified {len(classifications)} stocks into boxes")

            return classifications

        except Exception as e:
            logger.error(f"Stock classification failed: {e}")
            return {}

    def _generate_symbol_features(self,
                                 symbol: str,
                                 classification: Dict[str, Any],
                                 price_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate box features for a single symbol.

        Args:
            symbol: Stock symbol
            classification: Box classification for the symbol
            price_data: Price data for the symbol

        Returns:
            DataFrame with MultiIndex (date, symbol) and box feature columns
        """
        try:
            if price_data is None or price_data.empty:
                logger.debug(f"No price data for {symbol}")
                return None

            # Create feature DataFrame with same index as price data
            features = pd.DataFrame(index=price_data.index)

            # Generate dummy variables for each enabled category
            if self.config.size_categories:
                size_dummies = self._create_size_dummies(classification['size'])
                for col, val in size_dummies.items():
                    features[col] = val

            if self.config.style_categories:
                style_dummies = self._create_style_dummies(classification['style'])
                for col, val in style_dummies.items():
                    features[col] = val

            if self.config.region_categories:
                region_dummies = self._create_region_dummies(classification['region'])
                for col, val in region_dummies.items():
                    features[col] = val

            if self.config.sector_categories:
                sector_dummies = self._create_sector_dummies(classification['sector'])
                for col, val in sector_dummies.items():
                    features[col] = val

            # Create MultiIndex (date, symbol)
            symbol_multiindex = pd.MultiIndex.from_arrays([
                [symbol] * len(features.index),
                features.index
            ], names=['symbol', 'date'])

            features.index = symbol_multiindex
            features = features.swaplevel()  # Change to (date, symbol)

            return features

        except Exception as e:
            logger.error(f"Failed to generate features for {symbol}: {e}")
            return None

    def _create_size_dummies(self, size_category: str) -> Dict[str, int]:
        """Create dummy variables for size categories."""
        dummies = {}

        # All possible size categories
        all_sizes = [cat.value for cat in SizeCategory]

        for size in all_sizes:
            col_name = f"box_size_{size}"
            dummies[col_name] = 1 if size == size_category else 0

        return dummies

    def _create_style_dummies(self, style_category: str) -> Dict[str, int]:
        """Create dummy variables for style categories."""
        dummies = {}

        # All possible style categories
        all_styles = [cat.value for cat in StyleCategory]

        for style in all_styles:
            col_name = f"box_style_{style}"
            dummies[col_name] = 1 if style == style_category else 0

        return dummies

    def _create_region_dummies(self, region_category: str) -> Dict[str, int]:
        """Create dummy variables for region categories."""
        dummies = {}

        # All possible region categories
        all_regions = [cat.value for cat in RegionCategory]

        for region in all_regions:
            col_name = f"box_region_{region}"
            dummies[col_name] = 1 if region == region_category else 0

        return dummies

    def _create_sector_dummies(self, sector_category: str) -> Dict[str, int]:
        """Create dummy variables for sector categories."""
        dummies = {}

        # All possible sector categories
        all_sectors = [cat.value for cat in SectorCategory]

        for sector in all_sectors:
            col_name = f"box_sector_{sector.lower().replace(' ', '_')}"
            dummies[col_name] = 1 if sector == sector_category else 0

        return dummies

    def get_box_categories(self) -> Dict[str, List[str]]:
        """
        Get all possible box categories for feature generation.

        Returns:
            Dictionary mapping category types to list of possible values
        """
        if self._box_categories_cache is not None:
            return self._box_categories_cache

        categories = {}

        if self.config.size_categories:
            categories['size'] = [cat.value for cat in SizeCategory]

        if self.config.style_categories:
            categories['style'] = [cat.value for cat in StyleCategory]

        if self.config.region_categories:
            categories['region'] = [cat.value for cat in RegionCategory]

        if self.config.sector_categories:
            categories['sector'] = [cat.value for cat in SectorCategory]

        self._box_categories_cache = categories
        return categories

    def get_feature_names(self) -> List[str]:
        """
        Get all feature column names that will be generated.

        Returns:
            List of feature column names
        """
        feature_names = []
        categories = self.get_box_categories()

        for category, values in categories.items():
            for value in values:
                if category == 'sector':
                    # Special handling for sector names
                    col_name = f"box_{category}_{value.lower().replace(' ', '_')}"
                else:
                    col_name = f"box_{category}_{value}"
                feature_names.append(col_name)

        return feature_names

    def validate_config(self) -> Dict[str, Any]:
        """
        Validate the box feature configuration.

        Returns:
            Dictionary with validation results
        """
        issues = []

        if not self.config.enabled:
            return {"valid": True, "issues": [], "message": "Box features are disabled"}

        # Check encoding method
        if self.config.encoding_method not in ["one_hot", "target_encoding"]:
            issues.append(f"Invalid encoding method: {self.config.encoding_method}")

        # Check handle unknown method
        if self.config.handle_unknown not in ["ignore", "fallback"]:
            issues.append(f"Invalid handle_unknown method: {self.config.handle_unknown}")

        # Check if at least one category is enabled
        enabled_categories = sum([
            self.config.size_categories,
            self.config.style_categories,
            self.config.region_categories,
            self.config.sector_categories
        ])

        if enabled_categories == 0:
            issues.append("At least one box category must be enabled")

        # Check stock classifier availability
        if self.stock_classifier is None:
            issues.append("No stock classifier available")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "enabled_categories": self._get_enabled_categories(),
            "feature_count": len(self.get_feature_names())
        }

    def clear_cache(self):
        """Clear internal caches."""
        self._box_categories_cache = None
        self._classification_cache.clear()
        logger.info("Box feature provider cache cleared")

    def get_provider_info(self) -> Dict[str, Any]:
        """
        Get information about the box feature provider.

        Returns:
            Dictionary with provider information
        """
        validation_result = self.validate_config()

        return {
            "provider": "Box Feature Provider",
            "description": "Generates box classification dummy variables for ML models",
            "config": {
                "enabled": self.config.enabled,
                "size_categories": self.config.size_categories,
                "style_categories": self.config.style_categories,
                "region_categories": self.config.region_categories,
                "sector_categories": self.config.sector_categories,
                "encoding_method": self.config.encoding_method,
                "handle_unknown": self.config.handle_unknown
            },
            "validation": validation_result,
            "feature_names": self.get_feature_names() if self.config.enabled else [],
            "cache_info": {
                "classification_cache_size": len(self._classification_cache),
                "categories_cached": self._box_categories_cache is not None
            }
        }