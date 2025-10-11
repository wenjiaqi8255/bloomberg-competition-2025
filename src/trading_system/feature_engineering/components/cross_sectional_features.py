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
from typing import Dict, Optional, List
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
                 winsorize_percentile: float = 0.01):
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
        
        logger.info(f"Initialized CrossSectionalFeatureCalculator with lookback_periods={self.lookback_periods}")
    
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

        Args:
            price_data: Dictionary mapping symbols to OHLCV DataFrames
            date: Date for which to calculate cross-sectional features
            feature_names: Optional list of features to calculate (default: all)
            country_risk_data: Optional dictionary mapping symbols to country risk data

        Returns:
            DataFrame with columns:
                - symbol: Stock symbol
                - market_cap_proxy: Market capitalization proxy
                - book_to_market_proxy: Book-to-market proxy
                - size_factor: Log of market cap (SMB factor proxy)
                - value_factor: Value indicator (HML factor proxy)
                - momentum_12m: 12-month momentum
                - volatility_60d: 60-day volatility
                - country_risk_premium: Country risk premium (if country_risk_data provided)
                - equity_risk_premium: Equity risk premium (if country_risk_data provided)
                - default_spread: Default spread (if country_risk_data provided)
                - corporate_tax_rate: Corporate tax rate (if country_risk_data provided)
                - [cross-sectional ranks and z-scores]

        Raises:
            ValueError: If date is not available in price data
        """
        # Enhanced logging for debugging
        logger.info(f"=== CROSS-SECTIONAL FEATURES DEBUG ===")
        logger.info(f"Target date: {date}")
        logger.info(f"Requested feature names: {feature_names}")
        logger.info(f"Available symbols: {list(price_data.keys())}")
        logger.info(f"Lookback periods: {self.lookback_periods}")

        if feature_names is None:
            feature_names = ['market_cap', 'book_to_market', 'size', 'value', 'momentum', 'volatility']
            # Add country risk features if country risk data is provided
            if country_risk_data:
                country_features = ['country_risk_premium', 'equity_risk_premium', 'default_spread', 'corporate_tax_rate']
                feature_names.extend(country_features)
            logger.info(f"Using default feature names: {feature_names}")

        all_features = []
        successful_symbols = 0
        
        for symbol, data in price_data.items():
            logger.debug(f"Processing symbol: {symbol}")
            logger.debug(f"  Data shape: {data.shape}")
            logger.debug(f"  Date range: {data.index.min()} to {data.index.max()}")

            try:
                # Check if date is available - improved date matching
                date_to_use = None

                # First try exact match
                if date in data.index:
                    date_to_use = date
                    logger.debug(f"  Using exact date: {date_to_use}")
                else:
                    # Try to find the nearest available date (more flexible)
                    available_dates = data.index[data.index <= date]
                    if len(available_dates) > 0:
                        date_to_use = available_dates[-1]
                        days_diff = (date - date_to_use).days
                        logger.debug(f"  Using nearest date: {date_to_use} ({days_diff} days before target)")

                        # Skip if the date is too far from target (configurable tolerance)
                        max_tolerance_days = 7  # Allow up to 1 week tolerance
                        if days_diff > max_tolerance_days:
                            logger.warning(f"Date {date} too far from nearest {date_to_use} "
                                         f"({days_diff} days > {max_tolerance_days}), skipping {symbol}")
                            continue
                    else:
                        # Try to find any date after target (forward looking)
                        future_dates = data.index[data.index > date]
                        if len(future_dates) > 0:
                            date_to_use = future_dates[0]
                            days_diff = (date_to_use - date).days
                            logger.debug(f"  Using future date: {date_to_use} ({days_diff} days after target)")

                            # Smaller tolerance for future dates
                            max_future_tolerance_days = 3
                            if days_diff > max_future_tolerance_days:
                                logger.warning(f"Date {date} too far from future {date_to_use} "
                                             f"({days_diff} days > {max_future_tolerance_days}), skipping {symbol}")
                                continue
                        else:
                            logger.warning(f"No available dates near {date} for {symbol}, skipping")
                            continue

                # Get historical data up to this date
                historical_data = data[data.index <= date_to_use].copy()

                # Calculate dynamic requirements based on requested features
                required_periods = {}
                for feature_name in feature_names:
                    if feature_name in self.lookback_periods:
                        required_periods[feature_name] = self.lookback_periods[feature_name]

                min_required = min(required_periods.values()) if required_periods else 0
                max_required = max(required_periods.values()) if required_periods else 0

                available_history = len(historical_data)
                logger.debug(f"  Historical data: {available_history} days available")
                logger.debug(f"  Feature requirements: {required_periods}")
                logger.debug(f"  Min required: {min_required}, Max required: {max_required}")

                # More flexible history check: allow processing if at least one feature can be calculated
                if available_history < min_required:
                    logger.warning(f"Insufficient history for {symbol} at {date}: "
                                 f"{available_history} < {min_required}, skipping")
                    continue
                elif available_history < max_required:
                    # Filter features that can be calculated with available history
                    viable_features = []
                    for feature_name in feature_names:
                        if feature_name in self.lookback_periods:
                            if available_history >= self.lookback_periods[feature_name]:
                                viable_features.append(feature_name)
                        else:
                            # Features without specific requirements are assumed viable
                            viable_features.append(feature_name)

                    if not viable_features:
                        logger.warning(f"No viable features for {symbol} at {date} with {available_history} days, skipping")
                        continue

                    logger.info(f"Processing {symbol} with viable features only: {viable_features}")
                    # Filter feature_names to only viable ones
                    feature_names = viable_features
                
                # Calculate features for this symbol
                symbol_features = {'symbol': symbol, 'date': date_to_use}
                calculated_features = []

                for feature_name in feature_names:
                    logger.debug(f"  Calculating feature: {feature_name}")

                    try:
                        if feature_name == 'market_cap':
                            symbol_features['market_cap_proxy'] = self._calculate_market_cap_proxy(
                                historical_data, date_to_use
                            )
                            calculated_features.append('market_cap_proxy')

                        elif feature_name == 'book_to_market':
                            symbol_features['book_to_market_proxy'] = self._calculate_book_to_market_proxy(
                                historical_data, date_to_use
                            )
                            calculated_features.append('book_to_market_proxy')

                        elif feature_name == 'size':
                            symbol_features['size_factor'] = self._calculate_size_factor(
                                historical_data, date_to_use
                            )
                            calculated_features.append('size_factor')

                        elif feature_name == 'value':
                            symbol_features['value_factor'] = self._calculate_value_factor(
                                historical_data, date_to_use
                            )
                            calculated_features.append('value_factor')

                        elif feature_name == 'momentum':
                            symbol_features['momentum_12m'] = self._calculate_momentum(
                                historical_data, date_to_use, self.lookback_periods['momentum']
                            )
                            calculated_features.append('momentum_12m')

                        elif feature_name == 'volatility':
                            symbol_features['volatility_60d'] = self._calculate_volatility(
                                historical_data, date_to_use, self.lookback_periods['volatility']
                            )
                            calculated_features.append('volatility_60d')

                        elif feature_name in ['country_risk_premium', 'equity_risk_premium', 'default_spread', 'corporate_tax_rate']:
                            # Handle static country risk features
                            if country_risk_data and symbol in country_risk_data:
                                country_data = country_risk_data[symbol]
                                if feature_name in country_data:
                                    symbol_features[feature_name] = country_data[feature_name]
                                    calculated_features.append(feature_name)
                                else:
                                    logger.debug(f"  Country risk feature {feature_name} not found for {symbol}")
                                    symbol_features[feature_name] = np.nan
                                    calculated_features.append(feature_name)
                            else:
                                logger.debug(f"  No country risk data available for {symbol}")
                                symbol_features[feature_name] = np.nan
                                calculated_features.append(feature_name)

                        else:
                            logger.warning(f"  Unknown feature name: {feature_name}")

                    except Exception as e:
                        logger.error(f"  Error calculating {feature_name} for {symbol}: {e}")
                        continue

                logger.debug(f"  Successfully calculated {len(calculated_features)} features: {calculated_features}")
                all_features.append(symbol_features)
                successful_symbols += 1
                
            except Exception as e:
                logger.warning(f"Error calculating features for {symbol} at {date}: {e}")
                continue
        
        logger.info(f"Cross-sectional calculation summary for {date}:")
        logger.info(f"  Successfully processed symbols: {successful_symbols}/{len(price_data)}")
        logger.info(f"  Total feature records: {len(all_features)}")

        if not all_features:
            logger.error(f"No features calculated for date {date}")
            logger.error("=== CROSS-SECTIONAL FEATURES DEBUG END ===")
            return pd.DataFrame()

        # Create DataFrame
        features_df = pd.DataFrame(all_features)
        logger.info(f"Created features DataFrame with shape: {features_df.shape}")
        logger.info(f"Feature columns: {list(features_df.columns)}")
        logger.info("=== CROSS-SECTIONAL FEATURES DEBUG END ===")
        
        # Add cross-sectional rankings and z-scores
        features_df = self._add_cross_sectional_transformations(features_df)
        
        # Winsorize extreme values
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


