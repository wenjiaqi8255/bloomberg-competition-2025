"""
Cross-Sectional Feature Calculator

This module implements cross-sectional feature calculation for panel data analysis,
particularly for Fama-MacBeth regression and other cross-sectional studies.

Key Features:
- Market capitalization proxies (price × volume based)
- Book-to-Market proxies (technical indicator based)
- Momentum-based value proxies
- Size-based proxies
- Cross-sectional rankings and normalizations

Design Principles:
- Single Responsibility: Only calculates cross-sectional features
- Open/Closed: Easy to extend with new feature calculations
- Dependency Inversion: Works with any price data format
"""

import logging
from typing import Dict, Optional, List, Any
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class CrossSectionalFeatureCalculator:
    """
    Calculator for cross-sectional stock features.
    
    This class computes features that vary across stocks at a given point in time,
    which are essential for cross-sectional regression (e.g., Fama-MacBeth).
    
    Features Computed:
    1. Market Cap Proxy: price × volume (relative to market)
    2. Book-to-Market Proxy: momentum reversal + volatility
    3. Size Factor: log of market cap proxy
    4. Value Factor: based on price-to-MA ratios
    5. Cross-sectional rankings and z-scores
    
    Example:
        calculator = CrossSectionalFeatureCalculator()
        
        # Training phase: fit to learn statistics
        calculator.fit(price_data, train_dates)
        
        # Backtest phase: transform using learned statistics
        features = calculator.transform(price_data, target_date)
        
        # Result: DataFrame with columns ['symbol', 'market_cap', 'book_to_market', ...]
    """
    
    def __init__(self,
                 lookback_periods: Optional[Dict[str, int]] = None,
                 winsorize_percentile: float = 0.01,
                 **kwargs):
        """
        Initialize the cross-sectional feature calculator.
        
        Args:
            lookback_periods: Dictionary specifying lookback windows for various features
                             Default: {'momentum': 252, 'volatility': 60, 'ma': 200}
            winsorize_percentile: Percentile for winsorization (default: 0.01 = 1%)
        """
        self.lookback_periods = lookback_periods or {
            'momentum': 252,
            'volatility': 60,
            'ma_long': 200,
            'ma_short': 50
        }
        self.winsorize_percentile = winsorize_percentile
        
        # Fit/Transform state management
        self._is_fitted = False
        self.ranking_stats = {}  # Store training statistics for ranking/zscore

        # Feature caching (Dependency Inversion Principle)
        self._cache_provider = kwargs.get('cache_provider', None)
        self._cache_enabled = self._cache_provider is not None

        # Cache statistics tracking (always available)
        self._cache_hits = 0
        self._cache_misses = 0

        # Fallback in-memory cache if no provider specified (KISS principle)
        if not self._cache_enabled:
            self._fallback_cache = {}
            logger.info("No cache provider specified - using fallback in-memory cache")

        logger.info(f"Initialized CrossSectionalFeatureCalculator with lookback_periods={self.lookback_periods}")
        logger.info(f"Feature caching enabled: {self._cache_enabled}")
    
    def fit(self, price_data: Dict[str, pd.DataFrame], dates: List[datetime], 
            feature_names: Optional[List[str]] = None,
            country_risk_data: Optional[Dict[str, Dict[str, float]]] = None):
        """
        Fit the calculator on training data to learn ranking statistics.
        
        This method computes features for all training dates and learns the
        distribution statistics needed for cross-sectional ranking and z-score
        calculations during backtesting.
        
        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            dates: List of training dates
            feature_names: Optional list of features to calculate
            country_risk_data: Optional country risk data
        """
        logger.info(f"Fitting CrossSectionalFeatureCalculator on {len(dates)} training dates")
        
        if not dates:
            raise ValueError("Training dates cannot be empty")
        
        # Collect all features across training dates
        all_features = []
        for date in dates:
            logger.debug(f"Fitting on date: {date}")
            features = self.calculate_cross_sectional_features(
                price_data, date, feature_names, country_risk_data
            )
            if not features.empty:
                all_features.append(features)
        
        if not all_features:
            raise ValueError("No features computed during fitting")
        
        # Combine all training features
        combined_features = pd.concat(all_features, ignore_index=True)
        logger.info(f"Combined training features shape: {combined_features.shape}")
        
        # Learn ranking statistics for numeric features
        numeric_columns = combined_features.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col not in ['date']:  # Skip non-feature columns
                self.ranking_stats[col] = {
                    'mean': combined_features[col].mean(),
                    'std': combined_features[col].std(),
                    'percentiles': combined_features[col].quantile([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
                }
                logger.debug(f"Learned stats for {col}: mean={self.ranking_stats[col]['mean']:.4f}, "
                           f"std={self.ranking_stats[col]['std']:.4f}")
        
        self._is_fitted = True
        logger.info(f"CrossSectionalFeatureCalculator fitted successfully on {len(numeric_columns)} features")
    
    def transform(self, price_data: Dict[str, pd.DataFrame], target_date: datetime,
                  feature_names: Optional[List[str]] = None,
                  country_risk_data: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        """
        Transform data using learned statistics (for backtesting).
        
        This method computes features for a single date using the statistics
        learned during the fit phase, preventing data leakage.
        
        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            target_date: Single date for feature calculation
            feature_names: Optional list of features to calculate
            country_risk_data: Optional country risk data
            
        Returns:
            DataFrame with features and learned-based rankings/zscores
        """
        if not self._is_fitted:
            raise ValueError("Must fit before transform")
        
        logger.debug(f"Transforming data for date: {target_date}")
        
        # Calculate raw features for the target date
        features_df = self.calculate_cross_sectional_features(
            price_data, target_date, feature_names, country_risk_data
        )
        
        if features_df.empty:
            return features_df
        
        # Apply learned statistics for ranking and z-score
        features_df = self._apply_learned_transformations(features_df)
        
        return features_df
    
    def calculate_cross_sectional_features(
        self,
        price_data: Dict[str, pd.DataFrame],
        date: datetime,
        feature_names: Optional[List[str]] = None,
        country_risk_data: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """
        Calculate cross-sectional features for all stocks at a given date.

        This method uses vectorized operations for improved performance when processing
        multiple stocks. It falls back to individual processing for stocks with
        insufficient data or special cases.

        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            date: Date for which to calculate cross-sectional features
            feature_names: Optional list of features to calculate (default: all)
            country_risk_data: Optional dictionary mapping symbols to country risk data

        Returns:
            DataFrame with cross-sectional features for all stocks
        """
        logger.debug(f"Calculating cross-sectional features for {len(price_data)} stocks at {date}")

        if feature_names is None:
            feature_names = ['market_cap', 'book_to_market', 'size', 'value', 'momentum', 'volatility']
            if country_risk_data:
                country_features = ['country_risk_premium', 'equity_risk_premium', 'default_spread', 'corporate_tax_rate']
                feature_names.extend(country_features)

        # Step 1: Prepare data using vectorized operations
        prepared_data = self._prepare_vectorized_data(price_data, date, feature_names)

        if not prepared_data:
            logger.warning(f"No prepared data available for {date}")
            return pd.DataFrame()

        # Step 2: Vectorized feature calculation for majority of stocks
        vectorized_features = self._calculate_vectorized_features(prepared_data, date, feature_names)

        # Step 3: Individual processing for stocks that need special handling
        individual_features = self._process_special_cases(price_data, date, feature_names,
                                                       prepared_data.keys(), country_risk_data)

        # Step 4: Combine results
        all_features = vectorized_features + individual_features

        if not all_features:
            logger.warning(f"No features calculated for date {date}")
            return pd.DataFrame()

        # Create DataFrame
        features_df = pd.DataFrame(all_features)
        logger.debug(f"Created features DataFrame with shape: {features_df.shape}")

        # Step 5: Add cross-sectional rankings and z-scores (vectorized)
        features_df = self._add_cross_sectional_transformations(features_df)

        # Step 6: Winsorize extreme values (vectorized)
        features_df = self._winsorize_features(features_df)

        logger.debug(f"Calculated cross-sectional features for {len(features_df)} stocks at {date}")

        return features_df
    
    def _calculate_market_cap_proxy(self, data: pd.DataFrame, date: datetime) -> float:
        """
        Calculate market capitalization proxy.
        
        Proxy formula: Price × Average Volume (20-day)
        This approximates market cap when actual shares outstanding is unavailable.
        
        Args:
            data: Historical OHLCV data
            date: Date for calculation
        
        Returns:
            Market cap proxy value
        """
        try:
            # Get data up to date
            data_up_to_date = data[data.index <= date]
            
            # Current price
            current_price = data_up_to_date['Close'].iloc[-1]
            
            # Average volume over last 20 days
            avg_volume = data_up_to_date['Volume'].iloc[-20:].mean()
            
            # Market cap proxy
            market_cap_proxy = current_price * avg_volume
            
            return market_cap_proxy
            
        except Exception as e:
            logger.debug(f"Error calculating market cap proxy: {e}")
            return np.nan
    
    def _calculate_book_to_market_proxy(self, data: pd.DataFrame, date: datetime) -> float:
        """
        Calculate Book-to-Market ratio proxy.
        
        Proxy approach:
        - High B/M (value stocks) tend to have:
          * Lower price relative to long-term average
          * Negative recent momentum
          * Higher dividend yields (approximated by stability)
        
        Formula: (MA_200 / Current_Price) × (1 - Momentum_12M)
        
        Args:
            data: Historical OHLCV data
            date: Date for calculation
        
        Returns:
            Book-to-market proxy value
        """
        try:
            data_up_to_date = data[data.index <= date]
            
            # Current price
            current_price = data_up_to_date['Close'].iloc[-1]
            
            # Long-term moving average (200-day)
            ma_200 = data_up_to_date['Close'].iloc[-self.lookback_periods['ma_long']:].mean()
            
            # Price-to-MA ratio (inverse for B/M proxy)
            price_to_ma = ma_200 / current_price if current_price > 0 else np.nan
            
            # Momentum (negative momentum increases B/M proxy)
            momentum = self._calculate_momentum(data, date, self.lookback_periods['momentum'])
            momentum_adjustment = 1 - momentum if not np.isnan(momentum) else 1.0
            
            # Combine
            bm_proxy = price_to_ma * momentum_adjustment
            
            return bm_proxy
            
        except Exception as e:
            logger.debug(f"Error calculating B/M proxy: {e}")
            return np.nan
    
    def _calculate_size_factor(self, data: pd.DataFrame, date: datetime) -> float:
        """
        Calculate size factor (SMB proxy).
        
        Size = log(Market Cap Proxy)
        Smaller values indicate smaller companies (higher SMB loading).
        
        Args:
            data: Historical OHLCV data
            date: Date for calculation
        
        Returns:
            Size factor value
        """
        try:
            market_cap = self._calculate_market_cap_proxy(data, date)
            
            if np.isnan(market_cap) or market_cap <= 0:
                return np.nan
            
            # Log transformation for normality
            size_factor = np.log(market_cap)
            
            return size_factor
            
        except Exception as e:
            logger.debug(f"Error calculating size factor: {e}")
            return np.nan
    
    def _calculate_value_factor(self, data: pd.DataFrame, date: datetime) -> float:
        """
        Calculate value factor (HML proxy).
        
        Value indicator based on:
        - Price relative to 200-day MA
        - Price relative to 52-week high
        - Mean reversion tendency
        
        Args:
            data: Historical OHLCV data
            date: Date for calculation
        
        Returns:
            Value factor value (higher = more value-like)
        """
        try:
            data_up_to_date = data[data.index <= date]
            
            current_price = data_up_to_date['Close'].iloc[-1]
            
            # Price relative to 200-day MA
            ma_200 = data_up_to_date['Close'].iloc[-self.lookback_periods['ma_long']:].mean()
            price_to_ma = current_price / ma_200 if ma_200 > 0 else np.nan
            
            # Price relative to 52-week high
            high_52w = data_up_to_date['High'].iloc[-252:].max() if len(data_up_to_date) >= 252 else data_up_to_date['High'].max()
            price_to_high = current_price / high_52w if high_52w > 0 else np.nan
            
            # Value factor: lower ratios = more value-like
            # Invert so higher values = more value
            value_factor = 2.0 - (price_to_ma + price_to_high) / 2 if not np.isnan(price_to_ma) and not np.isnan(price_to_high) else np.nan
            
            return value_factor
            
        except Exception as e:
            logger.debug(f"Error calculating value factor: {e}")
            return np.nan
    
    def _calculate_momentum(self, data: pd.DataFrame, date: datetime, window: int) -> float:
        """
        Calculate momentum over specified window.
        
        Momentum = (Price_t - Price_{t-window}) / Price_{t-window}
        
        Args:
            data: Historical OHLCV data
            date: Date for calculation
            window: Lookback window in days
        
        Returns:
            Momentum value
        """
        try:
            data_up_to_date = data[data.index <= date]
            
            if len(data_up_to_date) < window + 1:
                return np.nan
            
            current_price = data_up_to_date['Close'].iloc[-1]
            past_price = data_up_to_date['Close'].iloc[-window-1]
            
            momentum = (current_price - past_price) / past_price if past_price > 0 else np.nan
            
            return momentum
            
        except Exception as e:
            logger.debug(f"Error calculating momentum: {e}")
            return np.nan
    
    def _calculate_volatility(self, data: pd.DataFrame, date: datetime, window: int) -> float:
        """
        Calculate realized volatility over specified window.
        
        Volatility = std(daily_returns) × sqrt(252)  [annualized]
        
        Args:
            data: Historical OHLCV data
            date: Date for calculation
            window: Lookback window in days
        
        Returns:
            Annualized volatility
        """
        try:
            data_up_to_date = data[data.index <= date]
            
            if len(data_up_to_date) < window + 1:
                return np.nan
            
            # Calculate returns
            prices = data_up_to_date['Close'].iloc[-window-1:]
            returns = prices.pct_change().dropna()
            
            # Annualized volatility
            volatility = returns.std() * np.sqrt(252)
            
            return volatility
            
        except Exception as e:
            logger.debug(f"Error calculating volatility: {e}")
            return np.nan
    
    def _add_cross_sectional_transformations(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add cross-sectional rankings and z-scores using current data.
        
        This method is used during training to compute rankings based on
        the current cross-section of stocks.
        
        Args:
            features_df: DataFrame with raw features
        
        Returns:
            DataFrame with additional rank and z-score columns
        """
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Skip columns that already have rank/zscore suffixes to avoid duplication
            if col in features_df.columns and not col.endswith('_rank') and not col.endswith('_zscore'):
                # Cross-sectional rank (percentile) - use current data
                rank_col = f"{col}_rank"
                features_df[rank_col] = features_df[col].rank(pct=True)
                
                # Cross-sectional z-score - use current data
                zscore_col = f"{col}_zscore"
                mean = features_df[col].mean()
                std = features_df[col].std()
                
                if std > 0:
                    features_df[zscore_col] = (features_df[col] - mean) / std
                else:
                    features_df[zscore_col] = 0.0
        
        return features_df
    
    def _apply_learned_transformations(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply learned statistics for ranking and z-score (for backtesting).
        
        This method uses statistics learned during training to compute
        rankings and z-scores, preventing data leakage.
        
        Args:
            features_df: DataFrame with raw features
        
        Returns:
            DataFrame with additional rank and z-score columns using learned stats
        """
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # Skip columns that already have rank/zscore suffixes to avoid duplication
            if col in features_df.columns and not col.endswith('_rank') and not col.endswith('_zscore'):
                if col in self.ranking_stats:
                    stats = self.ranking_stats[col]
                    
                    # Cross-sectional rank using learned percentiles
                    rank_col = f"{col}_rank"
                    features_df[rank_col] = features_df[col].apply(
                        lambda x: self._compute_rank_from_percentiles(x, stats['percentiles'])
                    )
                    
                    # Cross-sectional z-score using learned mean/std
                    zscore_col = f"{col}_zscore"
                    if stats['std'] > 0:
                        features_df[zscore_col] = (features_df[col] - stats['mean']) / stats['std']
                    else:
                        features_df[zscore_col] = 0.0
                else:
                    # Fallback: use current data if no learned stats available
                    rank_col = f"{col}_rank"
                    features_df[rank_col] = features_df[col].rank(pct=True)
                    
                    zscore_col = f"{col}_zscore"
                    mean = features_df[col].mean()
                    std = features_df[col].std()
                    if std > 0:
                        features_df[zscore_col] = (features_df[col] - mean) / std
                    else:
                        features_df[zscore_col] = 0.0
        
        return features_df
    
    def _compute_rank_from_percentiles(self, value: float, percentiles: pd.Series) -> float:
        """
        Compute rank based on learned percentiles.
        
        Args:
            value: Value to rank
            percentiles: Learned percentiles from training data
            
        Returns:
            Rank value between 0 and 1
        """
        if pd.isna(value):
            return 0.5  # Middle rank for NaN values
        
        # Find which percentile bucket the value falls into
        for i, p in enumerate([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
            if value <= percentiles[p]:
                # Linear interpolation within the bucket
                if i == 0:
                    return p * (value / percentiles[p]) if percentiles[p] > 0 else 0.0
                else:
                    prev_p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9][i-1]
                    prev_percentile = percentiles[prev_p]
                    bucket_size = percentiles[p] - prev_percentile
                    if bucket_size > 0:
                        position_in_bucket = (value - prev_percentile) / bucket_size
                        return prev_p + position_in_bucket * (p - prev_p)
                    else:
                        return prev_p
        
        # Value is above 90th percentile
        return 1.0
    
    def _winsorize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Winsorize extreme values to reduce impact of outliers.
        
        Replaces values beyond percentile thresholds with threshold values.
        
        Args:
            features_df: DataFrame with features
        
        Returns:
            DataFrame with winsorized features
        """
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if not col.endswith('_rank') and not col.endswith('_zscore'):
                # Calculate percentiles
                lower = features_df[col].quantile(self.winsorize_percentile)
                upper = features_df[col].quantile(1 - self.winsorize_percentile)
                
                # Winsorize
                features_df[col] = features_df[col].clip(lower=lower, upper=upper)
        
        return features_df
    
    def calculate_panel_features(
        self,
        price_data: Dict[str, pd.DataFrame],
        dates: List[datetime],
        feature_names: Optional[List[str]] = None,
        country_risk_data: Optional[Dict[str, Dict[str, float]]] = None,
        use_transform: bool = False
    ) -> pd.DataFrame:
        """
        Calculate cross-sectional features for multiple dates (panel data).
        
        This is useful for preparing data for Fama-MacBeth regression,
        which requires cross-sectional data at each time period.
        
        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            dates: List of dates for which to calculate features
            feature_names: Optional list of features to calculate
            country_risk_data: Optional country risk data
            use_transform: If True, use transform method (for backtesting)
        
        Returns:
            DataFrame with MultiIndex (date, symbol) containing all features
        """
        all_panels = []
        
        for date in dates:
            if use_transform and self._is_fitted:
                # Use transform method for backtesting (prevents data leakage)
                cross_section = self.transform(
                    price_data, date, feature_names, country_risk_data
                )
            else:
                # Use calculate_cross_sectional_features for training
                cross_section = self.calculate_cross_sectional_features(
                    price_data, date, feature_names, country_risk_data
                )
            
            if not cross_section.empty:
                all_panels.append(cross_section)
        
        if not all_panels:
            logger.warning("No panel features calculated")
            return pd.DataFrame()
        
        # Concatenate all cross-sections
        panel_df = pd.concat(all_panels, ignore_index=True)
        
        # Set MultiIndex (date, symbol)
        panel_df = panel_df.set_index(['date', 'symbol'])
        
        logger.info(f"Calculated panel features for {len(dates)} dates, "
                   f"{len(panel_df.index.get_level_values('symbol').unique())} symbols")
        
        return panel_df

    # ============================================================================
    # FEATURE CACHING MECHANISM
    # ============================================================================

    
    def _get_from_cache(self, symbols: list, date: datetime, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """
        Retrieve cross-sectional features from cache using the adapter.

        Args:
            symbols: List of stock symbols
            date: Target date
            feature_names: List of feature names

        Returns:
            Cached features DataFrame or None if not found
        """
        if not self._cache_enabled:
            return self._get_from_fallback_cache(symbols, date, feature_names)

        try:
            # Use the adapter to retrieve cached cross-sectional features
            cached_features = self._cache_provider.get_cross_sectional_features(
                symbols=symbols,
                date=date,
                feature_names=feature_names,
                lookback_periods=self.lookback_periods
            )

            if cached_features is not None:
                logger.debug(f"Cache HIT for {len(symbols)} symbols at {date}")
                return cached_features
            else:
                logger.debug(f"Cache MISS for {len(symbols)} symbols at {date}")
                return None

        except Exception as e:
            logger.error(f"Cache retrieval error: {e}")
            return self._get_from_fallback_cache(symbols, date, feature_names)

    def _store_in_cache(self, features: pd.DataFrame, symbols: list, date: datetime, feature_names: List[str]):
        """
        Store cross-sectional features in cache using the adapter.

        Args:
            features: Features DataFrame to cache
            symbols: List of stock symbols
            date: Target date
            feature_names: List of feature names
        """
        if not self._cache_enabled:
            self._store_in_fallback_cache(features, symbols, date, feature_names)
            return

        try:
            # Use the adapter to store cross-sectional features
            self._cache_provider.set_cross_sectional_features(
                features_df=features,
                symbols=symbols,
                date=date,
                feature_names=feature_names,
                lookback_periods=self.lookback_periods
            )

            logger.debug(f"Cached features for {len(symbols)} symbols at {date}")

        except Exception as e:
            logger.error(f"Cache storage error: {e}")
            self._store_in_fallback_cache(features, symbols, date, feature_names)

    def _get_from_fallback_cache(self, symbols: list, date: datetime, feature_names: List[str]) -> Optional[pd.DataFrame]:
        """Fallback in-memory cache implementation."""
        cache_key = self._generate_fallback_cache_key(symbols, date, feature_names)

        if cache_key in self._fallback_cache:
            self._cache_hits += 1
            logger.debug(f"Fallback cache HIT for key {cache_key}")
            return self._fallback_cache[cache_key].copy()
        else:
            self._cache_misses += 1
            logger.debug(f"Fallback cache MISS for key {cache_key}")
            return None

    def _store_in_fallback_cache(self, features: pd.DataFrame, symbols: list, date: datetime, feature_names: List[str]):
        """Fallback in-memory cache storage."""
        cache_key = self._generate_fallback_cache_key(symbols, date, feature_names)

        # Simple FIFO cleanup when cache gets too large
        max_fallback_size = 100
        if len(self._fallback_cache) >= max_fallback_size:
            # Remove oldest entries
            keys_to_remove = list(self._fallback_cache.keys())[:len(self._fallback_cache) // 2]
            for key in keys_to_remove:
                del self._fallback_cache[key]
            logger.debug(f"Fallback cache cleanup: removed {len(keys_to_remove)} entries")

        self._fallback_cache[cache_key] = features.copy()
        logger.debug(f"Stored in fallback cache: {cache_key}")

    def _generate_fallback_cache_key(self, symbols: list, date: datetime, feature_names: List[str]) -> str:
        """Generate cache key for fallback cache."""
        key_data = {
            'symbols': tuple(sorted(symbols)),
            'date': date.isoformat(),
            'features': tuple(sorted(feature_names)),
            'lookback_periods': tuple(sorted(self.lookback_periods.items()))
        }
        return str(hash(str(key_data)))

    def clear_cache(self):
        """Clear the feature cache."""
        if self._cache_enabled and hasattr(self._cache_provider, 'clear'):
            try:
                self._cache_provider.clear()
                logger.info("Provider cache cleared")
            except Exception as e:
                logger.error(f"Error clearing provider cache: {e}")

        # Clear fallback cache
        if hasattr(self, '_fallback_cache'):
            self._fallback_cache.clear()

        # Reset stats (always reset regardless of cache type)
        self._cache_hits = 0
        self._cache_misses = 0

        logger.info("Feature cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.

        Returns:
            Dictionary with cache statistics
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0

        stats = {
            'cache_enabled': self._cache_enabled,
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

        # Add provider-specific stats if available
        if self._cache_enabled and hasattr(self._cache_provider, 'get_cache_stats'):
            try:
                provider_stats = self._cache_provider.get_cache_stats()
                stats['provider_stats'] = provider_stats
            except Exception as e:
                logger.error(f"Error getting provider stats: {e}")

        # Add fallback cache stats
        if hasattr(self, '_fallback_cache'):
            stats['fallback_cache_size'] = len(self._fallback_cache)

        return stats

    def calculate_cross_sectional_features_cached(
        self,
        price_data: Dict[str, pd.DataFrame],
        date: datetime,
        feature_names: Optional[List[str]] = None,
        country_risk_data: Optional[Dict[str, Dict[str, float]]] = None
    ) -> pd.DataFrame:
        """
        Calculate cross-sectional features with intelligent caching support.

        This method wraps the main feature calculation with a caching layer that
        follows the existing cache provider interface, avoiding redundant computations
        for identical inputs while maintaining SOLID principles.

        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            date: Date for which to calculate cross-sectional features
            feature_names: Optional list of features to calculate (default: all)
            country_risk_data: Optional dictionary mapping symbols to country risk data

        Returns:
            DataFrame with cross-sectional features for all stocks
        """
        if feature_names is None:
            feature_names = ['market_cap', 'book_to_market', 'size', 'value', 'momentum', 'volatility']
            if country_risk_data:
                country_features = ['country_risk_premium', 'equity_risk_premium', 'default_spread', 'corporate_tax_rate']
                feature_names.extend(country_features)

        # Extract symbols from price data
        symbols = list(price_data.keys())

        # Try to get from cache first (using adapter interface)
        cached_features = self._get_from_cache(symbols, date, feature_names)
        if cached_features is not None:
            logger.debug(f"Using cached features for {len(cached_features)} symbols")
            return cached_features

        # Cache miss - compute features
        logger.debug(f"Computing features for {len(symbols)} symbols (cache miss)")
        features = self.calculate_cross_sectional_features(
            price_data, date, feature_names, country_risk_data
        )

        # Store in cache if computation was successful
        if not features.empty:
            self._store_in_cache(features, symbols, date, feature_names)

        return features

    def _prepare_vectorized_data(self, price_data: Dict[str, pd.DataFrame], date: datetime,
                                feature_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Prepare data for vectorized feature calculation with more lenient requirements.

        Filters and prepares data for stocks that have sufficient history and are available
        on the target date, enabling vectorized processing.

        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            date: Target date for feature calculation
            feature_names: List of features to be calculated

        Returns:
            Dictionary mapping symbols to prepared DataFrames with sufficient history
        """
        prepared_data = {}

        # Calculate maximum required lookback period
        max_required = 0
        for feature_name in feature_names:
            if feature_name in self.lookback_periods:
                max_required = max(max_required, self.lookback_periods[feature_name])

        # More lenient requirements: accept at least 50% of required data
        min_required = max(30, max_required // 2)  # At least 30 days or 50% of required

        for symbol, data in price_data.items():
            try:
                # Check if date is available (same logic as original)
                date_to_use = self._find_available_date(data, date)
                if date_to_use is None:
                    continue

                # Get historical data up to this date
                historical_data = data[data.index <= date_to_use].copy()

                # Check if we have sufficient history (more lenient)
                if len(historical_data) >= min_required:
                    prepared_data[symbol] = historical_data
                else:
                    logger.debug(f"Insufficient history for {symbol}: {len(historical_data)} < {min_required}")

            except Exception as e:
                logger.debug(f"Error preparing data for {symbol}: {e}")
                continue

        if len(prepared_data) == 0:
            logger.warning(f"No symbols met minimum data requirements for {date} "
                         f"(required {min_required} days, max {max_required} days)")

        logger.debug(f"Prepared {len(prepared_data)} symbols for vectorized processing "
                    f"(min required {min_required} days, max {max_required} days history)")

        return prepared_data

    def _find_available_date(self, data: pd.DataFrame, target_date: datetime) -> Optional[datetime]:
        """
        Find the best available date for calculation with more flexible tolerance.

        Args:
            data: Price data DataFrame
            target_date: Target date

        Returns:
            Best available date or None if no suitable date found
        """
        # First try exact match
        if target_date in data.index:
            return target_date

        # Try to find the nearest available date (more flexible)
        available_dates = data.index[data.index <= target_date]
        if len(available_dates) > 0:
            date_to_use = available_dates[-1]
            days_diff = (target_date - date_to_use).days

            # More lenient tolerance for historical data
            max_tolerance_days = 30  # Increased from 7 to 30 days
            if days_diff <= max_tolerance_days:
                return date_to_use

        # Try to find any date after target (forward looking)
        future_dates = data.index[data.index > target_date]
        if len(future_dates) > 0:
            date_to_use = future_dates[0]
            days_diff = (date_to_use - target_date).days

            max_future_tolerance_days = 14  # Increased from 3 to 14 days
            if days_diff <= max_future_tolerance_days:
                return date_to_use

        # Last resort: use the most recent date available
        if len(data.index) > 0:
            most_recent = data.index[-1]
            logger.debug(f"Using most recent available date {most_recent} for target {target_date}")
            return most_recent

        return None

    def _calculate_vectorized_features(self, prepared_data: Dict[str, pd.DataFrame],
                                     date: datetime, feature_names: List[str]) -> List[Dict]:
        """
        Calculate features using vectorized operations for all prepared stocks.

        This method processes all stocks simultaneously using pandas vectorized operations,
        which is significantly faster than individual stock processing.

        Args:
            prepared_data: Dictionary of prepared DataFrames with sufficient history
            date: Target date for calculation
            feature_names: List of features to calculate

        Returns:
            List of feature dictionaries for all successfully processed stocks
        """
        if not prepared_data:
            return []

        # Create aligned data matrices for vectorized computation
        symbols = list(prepared_data.keys())
        all_features = []

        try:
            # Step 1: Extract current values (vectorized)
            current_values = self._extract_current_values_vectorized(prepared_data, date)
            if current_values.empty:
                return []

            # Step 2: Calculate momentum features (vectorized)
            momentum_features = self._calculate_momentum_vectorized(prepared_data, date)

            # Step 3: Calculate volatility features (vectorized)
            volatility_features = self._calculate_volatility_vectorized(prepared_data, date)

            # Step 4: Calculate moving averages (vectorized)
            ma_features = self._calculate_moving_averages_vectorized(prepared_data, date)

            # Step 5: Combine all features
            for symbol in symbols:
                if symbol in current_values.index:
                    symbol_features = {'symbol': symbol, 'date': date}

                    # Market cap proxy
                    if 'market_cap' in feature_names:
                        close_price = current_values.loc[symbol, 'Close']
                        volume_20d = current_values.loc[symbol, 'Volume_20d']
                        symbol_features['market_cap_proxy'] = close_price * volume_20d

                    # Size factor (log of market cap)
                    if 'size' in feature_names and 'market_cap_proxy' in symbol_features:
                        market_cap = symbol_features['market_cap_proxy']
                        if not np.isnan(market_cap) and market_cap > 0:
                            symbol_features['size_factor'] = np.log(market_cap)
                        else:
                            symbol_features['size_factor'] = np.nan

                    # Momentum
                    if 'momentum' in feature_names and symbol in momentum_features.index:
                        symbol_features['momentum_12m'] = momentum_features.loc[symbol]

                    # Volatility
                    if 'volatility' in feature_names and symbol in volatility_features.index:
                        symbol_features['volatility_60d'] = volatility_features.loc[symbol]

                    # Book-to-market proxy
                    if 'book_to_market' in feature_names and symbol in ma_features.index:
                        symbol_features['book_to_market_proxy'] = self._calculate_bm_proxy_from_features(
                            current_values.loc[symbol, 'Close'],
                            ma_features.loc[symbol, 'MA_200'],
                            momentum_features.loc[symbol] if symbol in momentum_features.index else np.nan
                        )

                    # Value factor
                    if 'value' in feature_names and symbol in ma_features.index:
                        symbol_features['value_factor'] = self._calculate_value_factor_from_features(
                            current_values.loc[symbol, 'Close'],
                            ma_features.loc[symbol, 'MA_200'],
                            ma_features.loc[symbol, 'High_52w']
                        )

                    all_features.append(symbol_features)

        except Exception as e:
            logger.error(f"Error in vectorized feature calculation: {e}")
            # Fall back to individual processing for all stocks
            return []

        logger.debug(f"Vectorized calculation processed {len(all_features)} stocks")
        return all_features

    def _extract_current_values_vectorized(self, prepared_data: Dict[str, pd.DataFrame],
                                         date: datetime) -> pd.DataFrame:
        """
        Extract current values for all stocks using vectorized operations.

        Args:
            prepared_data: Dictionary of prepared DataFrames
            date: Target date

        Returns:
            DataFrame with current values for all stocks
        """
        current_data = []

        for symbol, data in prepared_data.items():
            try:
                # Get the most recent data point
                current_row = data.iloc[-1]

                # Calculate 20-day average volume
                volume_20d = data['Volume'].iloc[-20:].mean() if len(data) >= 20 else data['Volume'].iloc[-1]

                # Create row with current values
                row_data = {
                    'Close': current_row['Close'],
                    'Volume_20d': volume_20d,
                    'High': current_row['High'],
                    'Low': current_row['Low']
                }

                current_data.append(row_data)

            except Exception as e:
                logger.debug(f"Error extracting current values for {symbol}: {e}")
                continue

        if not current_data:
            return pd.DataFrame()

        # Create DataFrame with symbol index
        symbols = list(prepared_data.keys())[:len(current_data)]  # Ensure alignment
        df = pd.DataFrame(current_data, index=symbols)

        return df

    def _calculate_momentum_vectorized(self, prepared_data: Dict[str, pd.DataFrame],
                                     date: datetime) -> pd.Series:
        """
        Calculate momentum for all stocks using vectorized operations.

        Args:
            prepared_data: Dictionary of prepared DataFrames
            date: Target date

        Returns:
            Series with momentum values indexed by symbol
        """
        momentum_data = {}
        lookback = self.lookback_periods.get('momentum', 252)

        for symbol, data in prepared_data.items():
            try:
                if len(data) >= lookback + 1:
                    current_price = data['Close'].iloc[-1]
                    past_price = data['Close'].iloc[-lookback-1]
                    momentum = (current_price - past_price) / past_price if past_price > 0 else np.nan
                    momentum_data[symbol] = momentum
            except Exception as e:
                logger.debug(f"Error calculating momentum for {symbol}: {e}")
                momentum_data[symbol] = np.nan

        return pd.Series(momentum_data)

    def _calculate_volatility_vectorized(self, prepared_data: Dict[str, pd.DataFrame],
                                       date: datetime) -> pd.Series:
        """
        Calculate volatility for all stocks using vectorized operations.

        Args:
            prepared_data: Dictionary of prepared DataFrames
            date: Target date

        Returns:
            Series with volatility values indexed by symbol
        """
        volatility_data = {}
        lookback = self.lookback_periods.get('volatility', 60)

        for symbol, data in prepared_data.items():
            try:
                if len(data) >= lookback + 1:
                    prices = data['Close'].iloc[-lookback-1:]
                    returns = prices.pct_change().dropna()
                    volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else np.nan
                    volatility_data[symbol] = volatility
            except Exception as e:
                logger.debug(f"Error calculating volatility for {symbol}: {e}")
                volatility_data[symbol] = np.nan

        return pd.Series(volatility_data)

    def _calculate_moving_averages_vectorized(self, prepared_data: Dict[str, pd.DataFrame],
                                            date: datetime) -> pd.DataFrame:
        """
        Calculate moving averages for all stocks using vectorized operations.

        Args:
            prepared_data: Dictionary of prepared DataFrames
            date: Target date

        Returns:
            DataFrame with MA values indexed by symbol
        """
        ma_data = {}
        ma_long = self.lookback_periods.get('ma_long', 200)

        for symbol, data in prepared_data.items():
            try:
                ma_row = {}

                # 200-day moving average
                if len(data) >= ma_long:
                    ma_row['MA_200'] = data['Close'].iloc[-ma_long:].mean()
                else:
                    ma_row['MA_200'] = np.nan

                # 52-week high
                if len(data) >= 252:
                    ma_row['High_52w'] = data['High'].iloc[-252:].max()
                else:
                    ma_row['High_52w'] = data['High'].max()

                ma_data[symbol] = ma_row

            except Exception as e:
                logger.debug(f"Error calculating moving averages for {symbol}: {e}")
                ma_data[symbol] = {'MA_200': np.nan, 'High_52w': np.nan}

        if not ma_data:
            return pd.DataFrame()

        return pd.DataFrame.from_dict(ma_data, orient='index')

    def _calculate_bm_proxy_from_features(self, close_price: float, ma_200: float,
                                        momentum: float) -> float:
        """
        Calculate book-to-market proxy from pre-calculated features.

        Args:
            close_price: Current closing price
            ma_200: 200-day moving average
            momentum: 12-month momentum

        Returns:
            Book-to-market proxy value
        """
        try:
            if ma_200 <= 0 or close_price <= 0:
                return np.nan

            # Price-to-MA ratio (inverse for B/M proxy)
            price_to_ma = ma_200 / close_price

            # Momentum adjustment
            momentum_adjustment = 1 - momentum if not np.isnan(momentum) else 1.0

            # Combine
            bm_proxy = price_to_ma * momentum_adjustment

            return bm_proxy

        except Exception as e:
            logger.debug(f"Error calculating B/M proxy from features: {e}")
            return np.nan

    def _calculate_value_factor_from_features(self, close_price: float, ma_200: float,
                                            high_52w: float) -> float:
        """
        Calculate value factor from pre-calculated features.

        Args:
            close_price: Current closing price
            ma_200: 200-day moving average
            high_52w: 52-week high

        Returns:
            Value factor value
        """
        try:
            if ma_200 <= 0 or high_52w <= 0 or close_price <= 0:
                return np.nan

            # Price relative to 200-day MA
            price_to_ma = close_price / ma_200

            # Price relative to 52-week high
            price_to_high = close_price / high_52w

            # Value factor: lower ratios = more value-like
            # Invert so higher values = more value
            value_factor = 2.0 - (price_to_ma + price_to_high) / 2

            return value_factor

        except Exception as e:
            logger.debug(f"Error calculating value factor from features: {e}")
            return np.nan

    def _process_special_cases(self, price_data: Dict[str, pd.DataFrame], date: datetime,
                              feature_names: List[str], vectorized_symbols: set,
                              country_risk_data: Optional[Dict[str, Dict[str, float]]] = None) -> List[Dict]:
        """
        Process stocks that couldn't be handled by vectorized calculation.

        This method handles stocks with insufficient data, missing values, or other
        special cases that require individual processing.

        Args:
            price_data: Original price data dictionary
            date: Target date
            feature_names: List of features to calculate
            vectorized_symbols: Set of symbols already processed vectorially
            country_risk_data: Optional country risk data

        Returns:
            List of feature dictionaries for specially processed stocks
        """
        special_features = []

        # Find symbols that need individual processing
        remaining_symbols = set(price_data.keys()) - vectorized_symbols

        if not remaining_symbols:
            return special_features

        logger.debug(f"Processing {len(remaining_symbols)} special cases individually")

        for symbol in remaining_symbols:
            try:
                # Use original individual processing logic
                symbol_features = self._process_individual_symbol(
                    price_data[symbol], symbol, date, feature_names, country_risk_data
                )

                if symbol_features:
                    special_features.append(symbol_features)

            except Exception as e:
                logger.debug(f"Error processing special case {symbol}: {e}")
                continue

        logger.debug(f"Special case processing completed for {len(special_features)} stocks")
        return special_features

    def _process_individual_symbol(self, data: pd.DataFrame, symbol: str, date: datetime,
                                 feature_names: List[str],
                                 country_risk_data: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[Dict]:
        """
        Process a single symbol using the original individual logic.

        Args:
            data: Price data for the symbol
            symbol: Symbol name
            date: Target date
            feature_names: Features to calculate
            country_risk_data: Optional country risk data

        Returns:
            Feature dictionary or None if processing failed
        """
        try:
            # Find available date
            date_to_use = self._find_available_date(data, date)
            if date_to_use is None:
                return None

            # Get historical data
            historical_data = data[data.index <= date_to_use].copy()

            # Calculate required periods
            required_periods = {}
            for feature_name in feature_names:
                if feature_name in self.lookback_periods:
                    required_periods[feature_name] = self.lookback_periods[feature_name]

            max_required = max(required_periods.values()) if required_periods else 0

            if len(historical_data) < max_required:
                return None

            # Calculate features
            symbol_features = {'symbol': symbol, 'date': date_to_use}

            for feature_name in feature_names:
                try:
                    if feature_name == 'market_cap':
                        symbol_features['market_cap_proxy'] = self._calculate_market_cap_proxy(
                            historical_data, date_to_use
                        )
                    elif feature_name == 'book_to_market':
                        symbol_features['book_to_market_proxy'] = self._calculate_book_to_market_proxy(
                            historical_data, date_to_use
                        )
                    elif feature_name == 'size':
                        symbol_features['size_factor'] = self._calculate_size_factor(
                            historical_data, date_to_use
                        )
                    elif feature_name == 'value':
                        symbol_features['value_factor'] = self._calculate_value_factor(
                            historical_data, date_to_use
                        )
                    elif feature_name == 'momentum':
                        symbol_features['momentum_12m'] = self._calculate_momentum(
                            historical_data, date_to_use, self.lookback_periods['momentum']
                        )
                    elif feature_name == 'volatility':
                        symbol_features['volatility_60d'] = self._calculate_volatility(
                            historical_data, date_to_use, self.lookback_periods['volatility']
                        )
                    elif feature_name in ['country_risk_premium', 'equity_risk_premium', 'default_spread', 'corporate_tax_rate']:
                        # Handle country risk features
                        if country_risk_data and symbol in country_risk_data:
                            country_data = country_risk_data[symbol]
                            symbol_features[feature_name] = country_data.get(feature_name, np.nan)
                        else:
                            symbol_features[feature_name] = np.nan

                except Exception as e:
                    logger.debug(f"Error calculating {feature_name} for {symbol}: {e}")
                    symbol_features[feature_name] = np.nan

            return symbol_features

        except Exception as e:
            logger.debug(f"Error processing individual symbol {symbol}: {e}")
            return None


