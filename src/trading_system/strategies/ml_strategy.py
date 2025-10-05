"""
Multi-Stock ML Strategy - Machine Learning strategy for models trained on multiple stocks

This strategy uses ML models (RandomForest, XGBoost, LSTM, etc.) trained on multiple stocks
to predict expected returns based on a comprehensive set of features.

Architecture:
    FeatureEngineeringPipeline → MLModelPredictor → PositionSizer

Key Features:
    - Automatically fetches data for all model training stocks
    - Ensures feature compatibility during inference
    - Signal strength filtering for robust predictions
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
import yfinance as yf

from .base_strategy import BaseStrategy
from ..feature_engineering.pipeline import FeatureEngineeringPipeline
from ..models.serving.predictor import ModelPredictor
from ..data.yfinance_provider import YFinanceProvider

logger = logging.getLogger(__name__)


class MLStrategy(BaseStrategy):
    """
    Enhanced ML strategy for models trained on multiple stocks.

    This strategy handles the common scenario where a model was trained on a basket
    of stocks (e.g., 10 stocks) but during inference we want to trade only a subset
    (e.g., 3-5 stocks). The model still expects features for all training stocks.

    Key Features:
    - Automatically detects model's training stocks from metadata
    - Fetches price data for all training stocks
    - Computes features for all stocks to ensure model compatibility
    - Only generates signals for the target trading universe
    """

    def __init__(self,
                 name: str,
                 feature_pipeline: FeatureEngineeringPipeline,
                 model_predictor: ModelPredictor,
                 universe: List[str],  # Trading universe (subset of model stocks)
                 model_training_stocks: Optional[List[str]] = None,  # Auto-detected if None
                 min_signal_strength: float = 0.1,
                 **kwargs):
        """
        Initialize multi-stock ML strategy.

        Args:
            name: Strategy identifier
            feature_pipeline: Fitted feature pipeline
            model_predictor: Predictor with ML model loaded
            universe: The list of symbols this strategy actually trades
            model_training_stocks: All stocks the model was trained on (auto-detected if None)
            min_signal_strength: Minimum signal strength to act on
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            feature_pipeline=feature_pipeline,
            model_predictor=model_predictor,
            **kwargs
        )

        # Store universe explicitly
        self.universe = universe
        self.min_signal_strength = min_signal_strength

        # Auto-detect model training stocks if not provided
        if model_training_stocks is None:
            self.model_training_stocks = self._detect_model_training_stocks()
            # If detection fails, provide a default based on the model we know about
            if not self.model_training_stocks:
                logger.warning("Model training stock detection failed, using default stocks")
                self.model_training_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']
        else:
            self.model_training_stocks = model_training_stocks

        logger.info(f"MultiStockMLStrategy '{name}' initialized:")
        logger.info(f"  - Trading universe: {universe}")
        logger.info(f"  - Model training stocks: {self.model_training_stocks}")
        logger.info(f"  - Min signal strength: {min_signal_strength}")
        logger.info(f"  - Will fetch data for {len(self.model_training_stocks)} stocks to ensure model compatibility")

    def _detect_model_training_stocks(self) -> List[str]:
        """
        Detect which stocks the model was trained on by examining model metadata.

        Returns:
            List of stock symbols the model was trained on
        """
        try:
            # Try to read the metadata.json file directly from the model directory
            if hasattr(self.model_predictor, 'model_id'):
                model_id = self.model_predictor.model_id
                metadata_path = f'./models/{model_id}/metadata.json'

                try:
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    logger.debug(f"Loaded metadata from {metadata_path}")
                    logger.debug(f"Metadata keys: {list(metadata.keys())}")

                    # Check different possible locations for symbols
                    symbols = None

                    # Check in tags.symbols (from the metadata structure we saw)
                    if 'tags' in metadata and 'symbols' in metadata['tags']:
                        symbols = metadata['tags']['symbols']

                    # Check in metadata directly
                    elif 'symbols' in metadata:
                        symbols = metadata['symbols']

                    # Check in model_metadata
                    elif 'model_metadata' in metadata and 'tags' in metadata['model_metadata']:
                        symbols = metadata['model_metadata']['tags'].get('symbols')

                    if symbols and isinstance(symbols, str):
                        # Parse comma-separated string: "AAPL,MSFT,GOOGL,..."
                        result = [s.strip() for s in symbols.split(',')]
                        logger.info(f"Auto-detected model training stocks from metadata file: {result}")
                        return result

                except FileNotFoundError:
                    logger.warning(f"Metadata file not found at {metadata_path}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to parse metadata file: {e}")

            # Fallback: Try to get model metadata from model_predictor
            if hasattr(self.model_predictor, 'model') and hasattr(self.model_predictor.model, 'metadata'):
                metadata = self.model_predictor.model.metadata
                logger.debug(f"Model metadata structure: {list(metadata.keys()) if isinstance(metadata, dict) else type(metadata)}")

                # Check different possible locations for symbols
                symbols = None

                # Check in metadata directly
                if isinstance(metadata, dict):
                    symbols = metadata.get('symbols')

                    # Check in tags.symbols
                    if symbols is None and 'tags' in metadata:
                        symbols = metadata['tags'].get('symbols')

                if symbols and isinstance(symbols, str):
                    # Parse comma-separated string: "AAPL,MSFT,GOOGL,..."
                    result = [s.strip() for s in symbols.split(',')]
                    logger.info(f"Auto-detected model training stocks from model: {result}")
                    return result

            logger.warning("Could not find symbols in model metadata")
            return []

        except Exception as e:
            logger.warning(f"Could not auto-detect model training stocks: {e}")
            return []

    def _fetch_additional_stock_data(self,
                                   price_data: Dict[str, pd.DataFrame],
                                   start_date: datetime,
                                   end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Fetch price data for additional stocks that the model expects but aren't
        in the provided price_data.

        Args:
            price_data: Original price data dictionary
            start_date: Start date for data fetching
            end_date: End date for data fetching

        Returns:
            Enhanced price data with all model training stocks
        """
        enhanced_price_data = price_data.copy()

        # Find stocks that model needs but aren't provided
        provided_stocks = set(price_data.keys())
        needed_stocks = set(self.model_training_stocks)
        missing_stocks = needed_stocks - provided_stocks

        if not missing_stocks:
            logger.debug(f"[{self.name}] All model training stocks already provided")
            return enhanced_price_data

        logger.info(f"[{self.name}] Fetching data for {len(missing_stocks)} missing model training stocks: {sorted(missing_stocks)}")

        # Initialize data provider
        data_provider = YFinanceProvider()

        # Fetch data for missing stocks
        for symbol in missing_stocks:
            try:
                stock_data = data_provider.get_historical_data(
                    symbols=symbol,
                    start_date=start_date,
                    end_date=end_date
                )
                # get_historical_data returns a dict, get the DataFrame for this symbol
                if isinstance(stock_data, dict) and symbol in stock_data:
                    stock_data = stock_data[symbol]

                if not stock_data.empty:
                    enhanced_price_data[symbol] = stock_data
                    logger.info(f"[{self.name}] ✅ Fetched {len(stock_data)} rows for {symbol}")
                else:
                    logger.warning(f"[{self.name}] ⚠️ No data available for {symbol}")

            except Exception as e:
                logger.error(f"[{self.name}] ❌ Failed to fetch data for {symbol}: {e}")

        # Verify we got data for most training stocks
        final_stocks = set(enhanced_price_data.keys())
        still_missing = needed_stocks - final_stocks

        if still_missing:
            logger.warning(f"[{self.name}] Could not fetch data for {len(still_missing)} stocks: {sorted(still_missing)}")
        else:
            logger.info(f"[{self.name}] ✅ Successfully fetched data for all {len(needed_stocks)} model training stocks")

        return enhanced_price_data

    def _get_predictions(self,
                        features: pd.DataFrame,
                        price_data: Dict[str, pd.DataFrame],
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        """
        Override to add signal strength filtering and normalization for ML predictions.

        ML models may output weak signals that should be filtered out or extreme values
        that need to be normalized.
        """
        # Get base predictions from parent
        predictions = super()._get_predictions(features, price_data, start_date, end_date)

        if predictions.empty:
            return predictions

        logger.info(f"[{self.name}] 📊 Raw predictions statistics:")
        logger.info(f"[{self.name}]   Shape: {predictions.shape}")
        logger.info(f"[{self.name}]   Mean: {predictions.mean().mean():.6f}")
        logger.info(f"[{self.name}]   Std: {predictions.std().mean():.6f}")
        logger.info(f"[{self.name}]   Min: {predictions.min().min():.6f}")
        logger.info(f"[{self.name}]   Max: {predictions.max().max():.6f}")

        # Apply signal strength filtering
        filtered_predictions = predictions.copy()

        # Filter weak signals
        weak_signal_mask = filtered_predictions.abs() < self.min_signal_strength
        num_weak_signals = weak_signal_mask.sum().sum()
        filtered_predictions[weak_signal_mask] = 0.0

        logger.info(f"[{self.name}] 🎯 Filtered {num_weak_signals} weak signals (threshold={self.min_signal_strength})")

        # Apply outlier detection and normalization
        normalized_predictions = self._normalize_predictions(filtered_predictions)

        logger.info(f"[{self.name}] 📈 Normalized predictions statistics:")
        logger.info(f"[{self.name}]   Mean: {normalized_predictions.mean().mean():.6f}")
        logger.info(f"[{self.name}]   Std: {normalized_predictions.std().mean():.6f}")
        logger.info(f"[{self.name}]   Min: {normalized_predictions.min().min():.6f}")
        logger.info(f"[{self.name}]   Max: {normalized_predictions.max().max():.6f}")

        return normalized_predictions

    def _normalize_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize predictions to handle extreme values and ensure reasonable bounds.

        Args:
            predictions: Raw predictions DataFrame

        Returns:
            Normalized predictions DataFrame
        """
        normalized = predictions.copy()

        # Handle extreme values using quantile-based clipping
        for column in normalized.columns:
            series = normalized[column]

            # Calculate quantiles for this symbol
            q_low = series.quantile(0.01)  # 1st percentile
            q_high = series.quantile(0.99)  # 99th percentile

            # Clip extreme values
            clipped_series = series.clip(lower=q_low, upper=q_high)
            normalized[column] = clipped_series

            if series.min() != clipped_series.min() or series.max() != clipped_series.max():
                logger.debug(f"[{self.name}] Clipped {column} from [{series.min():.6f}, {series.max():.6f}] to [{clipped_series.min():.6f}, {clipped_series.max():.6f}]")

        # Apply additional normalization to ensure reasonable scale
        # Use robust scaling based on median and MAD (median absolute deviation)
        all_values = normalized.values.flatten()
        non_zero_values = all_values[all_values != 0]

        if len(non_zero_values) > 0:
            median_val = np.median(non_zero_values)
            mad_val = np.median(np.abs(non_zero_values - median_val))

            if mad_val > 0:
                # Robust scaling: (x - median) / MAD
                for column in normalized.columns:
                    mask = normalized[column] != 0
                    normalized.loc[mask, column] = (normalized.loc[mask, column] - median_val) / mad_val

                logger.debug(f"[{self.name}] Applied robust scaling (median={median_val:.6f}, MAD={mad_val:.6f})")

                # Rescale to reasonable range for trading signals (e.g., -0.3 to 0.3)
                max_abs_val = normalized.abs().max().max()
                if max_abs_val > 0.3:
                    scale_factor = 0.3 / max_abs_val
                    normalized = normalized * scale_factor
                    logger.debug(f"[{self.name}] Rescaled by factor {scale_factor:.6f} to keep signals in [-0.3, 0.3] range")

        return normalized

    def generate_signals(self,
                        price_data: Dict[str, pd.DataFrame],
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        """
        Generate trading signals using multi-stock enhanced approach.

        This method extends the base implementation to ensure the model receives
        features for all stocks it was trained on, even if we only trade a subset.

        Args:
            price_data: Original price data (may only contain trading universe)
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with trading signals for the trading universe only
        """
        try:
            logger.info(f"[{self.name}] Starting multi-stock signal generation...")

            # Step 1: Ensure we have data for all model training stocks
            enhanced_price_data = self._fetch_additional_stock_data(price_data, start_date, end_date)

            # Step 2: Compute features for all stocks (this is what the model expects)
            features = self._compute_features(enhanced_price_data)

            if features.empty:
                logger.error(f"[{self.name}] Feature computation failed")
                return pd.DataFrame()

            # Step 3: Get predictions from model (model sees all training stocks)
            predictions = self._get_predictions(features, enhanced_price_data, start_date, end_date)

            if predictions.empty:
                logger.error(f"[{self.name}] No predictions generated")
                return pd.DataFrame()

            # Step 4: Filter predictions to only include our trading universe
            if self.universe:
                available_universe = [s for s in self.universe if s in predictions.columns]
                if not available_universe:
                    logger.error(f"[{self.name}] None of the trading universe stocks found in predictions")
                    return pd.DataFrame()

                filtered_predictions = predictions[available_universe]
                logger.info(f"[{self.name}] Filtered to {len(available_universe)} trading universe stocks: {available_universe}")
            else:
                filtered_predictions = predictions

            # Step 5: Apply signal strength filtering
            if hasattr(self, 'min_signal_strength'):
                filtered_predictions = filtered_predictions.copy()
                filtered_predictions[filtered_predictions.abs() < self.min_signal_strength] = 0.0
                logger.debug(f"[{self.name}] Applied signal strength filtering (threshold={self.min_signal_strength})")

            # Step 6: Apply position sizing
            signals = self._apply_position_sizing(filtered_predictions, enhanced_price_data)

            # Step 7: Validate signal diversity (NEW)
            if not signals.empty:
                self.log_signal_validation(signals)

            logger.info(f"[{self.name}] ✅ Generated signals for {len(signals.columns)} assets")
            return signals

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Multi-stock signal generation failed: {e}", exc_info=True)
            return pd.DataFrame()

    def _apply_position_sizing(self, predictions: pd.DataFrame, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Apply position sizing to predictions based on relative prediction strength.

        Args:
            predictions: DataFrame with raw model predictions (dates x symbols)
            price_data: Dictionary with price data for each symbol

        Returns:
            DataFrame with position-sized signals (dates x symbols)
        """
        if predictions.empty:
            return predictions

        signals = predictions.copy()

        # Log input statistics
        logger.info(f"[{self.name}] 📊 Input predictions statistics:")
        logger.info(f"[{self.name}]   Shape: {signals.shape}")
        logger.info(f"[{self.name}]   Mean: {signals.mean().mean():.6f}")
        logger.info(f"[{self.name}]   Std: {signals.std().mean():.6f}")
        logger.info(f"[{self.name}]   Min: {signals.min().min():.6f}")
        logger.info(f"[{self.name}]   Max: {signals.max().max():.6f}")

        # Normalize signals based on relative prediction strength for each date
        for date_idx in signals.index:
            date_signals = signals.loc[date_idx].copy()

            # Separate long and short signals
            long_signals = date_signals[date_signals > 0]
            short_signals = date_signals[date_signals < 0]

            # Apply prediction-strength-based weighting for long positions
            if not long_signals.empty:
                if len(long_signals) == 1:
                    # Single long position, give it full weight
                    signals.loc[date_idx, long_signals.index] = 1.0
                else:
                    # Multiple long positions, weight by prediction strength
                    long_sum = long_signals.sum()
                    if long_sum > 0:
                        long_weights = long_signals / long_sum
                        signals.loc[date_idx, long_signals.index] = long_weights
                    else:
                        # Fallback to equal weights if all predictions are zero
                        equal_weight = 1.0 / len(long_signals)
                        signals.loc[date_idx, long_signals.index] = equal_weight

            # Apply prediction-strength-based weighting for short positions
            if not short_signals.empty:
                if len(short_signals) == 1:
                    # Single short position, give it full negative weight
                    signals.loc[date_idx, short_signals.index] = -1.0
                else:
                    # Multiple short positions, weight by prediction strength (absolute values)
                    short_abs = short_signals.abs()
                    short_sum = short_abs.sum()
                    if short_sum > 0:
                        short_weights = -short_abs / short_sum
                        signals.loc[date_idx, short_signals.index] = short_weights
                    else:
                        # Fallback to equal weights if all predictions are zero
                        equal_weight = -1.0 / len(short_signals)
                        signals.loc[date_idx, short_signals.index] = equal_weight

        # Apply max position size limit if configured
        if hasattr(self, 'max_position_size') and self.max_position_size < 1.0:
            signals = signals.clip(lower=-self.max_position_size, upper=self.max_position_size)
            # Renormalize to maintain sum constraint
            for date_idx in signals.index:
                date_signals = signals.loc[date_idx]
                total_long = date_signals[date_signals > 0].sum()
                total_short = date_signals[date_signals < 0].sum()

                if total_long > 0:
                    long_mask = date_signals > 0
                    signals.loc[date_idx, long_mask] = date_signals[long_mask] / total_long

                if total_short < 0:
                    short_mask = date_signals < 0
                    signals.loc[date_idx, short_mask] = date_signals[short_mask] / abs(total_short)

        # Log output statistics
        logger.info(f"[{self.name}] 📈 Output signals statistics:")
        logger.info(f"[{self.name}]   Mean: {signals.mean().mean():.6f}")
        logger.info(f"[{self.name}]   Std: {signals.std().mean():.6f}")
        logger.info(f"[{self.name}]   Min: {signals.min().min():.6f}")
        logger.info(f"[{self.name}]   Max: {signals.max().max():.6f}")

        # Log sample date to verify diversity
        if not signals.empty:
            sample_date = signals.index[0]
            sample_signals = signals.loc[sample_date]
            logger.info(f"[{self.name}] 📊 Sample signals for {sample_date}:")
            for symbol, signal in sample_signals.items():
                if abs(signal) > 0.01:  # Only log non-trivial signals
                    logger.info(f"[{self.name}]   {symbol}: {signal:.6f}")

        logger.debug(f"[{self.name}] Applied prediction-strength-based position sizing to {signals.shape[0]} dates")
        return signals

    def get_info(self) -> Dict:
        """Get multi-stock ML strategy information."""
        info = super().get_info()
        info.update({
            'strategy_type': 'ml_strategy',
            'model_training_stocks': self.model_training_stocks,
            'trading_universe': self.universe,
            'min_signal_strength': self.min_signal_strength,
            'model_complexity': 'high',
            'feature_compatibility': 'multi_stock_enhanced'
        })
        return info

    def validate_signal_diversity(self, signals: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate that signals are diverse and not all equal.

        This method helps verify that the fixes for signal equalization are working.

        Args:
            signals: Generated signals DataFrame

        Returns:
            Dictionary with validation metrics
        """
        if signals.empty:
            return {'error': 'No signals to validate'}

        validation_results = {}

        # Check 1: Signal variance over time
        time_variance = signals.var(axis=0)
        validation_results['signal_variance_by_symbol'] = time_variance.to_dict()
        validation_results['avg_time_variance'] = time_variance.mean()

        # Check 2: Cross-sectional variance at each date
        cross_sectional_variance = signals.var(axis=1)
        validation_results['cross_sectional_variance_stats'] = {
            'mean': cross_sectional_variance.mean(),
            'std': cross_sectional_variance.std(),
            'min': cross_sectional_variance.min(),
            'max': cross_sectional_variance.max()
        }

        # Check 3: Unique signal values
        unique_values_per_symbol = signals.nunique()
        validation_results['unique_values_per_symbol'] = unique_values_per_symbol.to_dict()
        validation_results['avg_unique_values'] = unique_values_per_symbol.mean()

        # Check 4: Signal correlation matrix
        if not signals.empty and signals.shape[1] > 1:
            correlation_matrix = signals.corr()
            avg_correlation = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean()
            validation_results['avg_pairwise_correlation'] = avg_correlation
            validation_results['max_correlation'] = correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max()

        # Check 5: Position distribution
        long_positions = (signals > 0).sum().sum()
        short_positions = (signals < 0).sum().sum()
        total_positions = signals.shape[0] * signals.shape[1]
        validation_results['position_distribution'] = {
            'long_positions': long_positions,
            'short_positions': short_positions,
            'neutral_positions': total_positions - long_positions - short_positions,
            'long_percentage': long_positions / total_positions * 100,
            'short_percentage': short_positions / total_positions * 100
        }

        # Check 6: Signal range analysis
        signal_ranges = {}
        for symbol in signals.columns:
            symbol_signals = signals[symbol]
            signal_ranges[symbol] = {
                'min': symbol_signals.min(),
                'max': symbol_signals.max(),
                'range': symbol_signals.max() - symbol_signals.min(),
                'std': symbol_signals.std()
            }
        validation_results['signal_ranges'] = signal_ranges

        # Overall validation summary
        validation_results['validation_summary'] = {
            'signals_are_diverse': validation_results['avg_time_variance'] > 1e-6,
            'time_variance_ok': validation_results['avg_time_variance'] > 1e-6,
            'cross_sectional_variance_ok': validation_results['cross_sectional_variance_stats']['mean'] > 1e-6,
            'unique_values_ok': validation_results['avg_unique_values'] > 1,
            'signals_not_static': not (signals.std().max() < 1e-6)
        }

        return validation_results

    def log_signal_validation(self, signals: pd.DataFrame):
        """
        Log validation results to help with debugging.

        Args:
            signals: Generated signals DataFrame
        """
        logger.info(f"[{self.name}] 🔍 Validating signal diversity...")

        validation_results = self.validate_signal_diversity(signals)

        if 'error' in validation_results:
            logger.error(f"[{self.name}] ❌ Validation failed: {validation_results['error']}")
            return

        # Log key metrics
        logger.info(f"[{self.name}] 📊 Signal Validation Results:")
        logger.info(f"[{self.name}]   Average time variance: {validation_results['avg_time_variance']:.8f}")
        logger.info(f"[{self.name}]   Cross-sectional variance: {validation_results['cross_sectional_variance_stats']['mean']:.8f}")
        logger.info(f"[{self.name}]   Average unique values per symbol: {validation_results['avg_unique_values']:.1f}")

        if 'avg_pairwise_correlation' in validation_results:
            logger.info(f"[{self.name}]   Average pairwise correlation: {validation_results['avg_pairwise_correlation']:.4f}")

        pos_dist = validation_results['position_distribution']
        logger.info(f"[{self.name}]   Position distribution: {pos_dist['long_percentage']:.1f}% long, {pos_dist['short_percentage']:.1f}% short")

        # Log validation summary
        summary = validation_results['validation_summary']
        logger.info(f"[{self.name}] ✅ Validation summary:")
        for key, value in summary.items():
            status = "✅" if value else "❌"
            logger.info(f"[{self.name}]   {status} {key}: {value}")

        # Sample signal patterns
        if not signals.empty:
            logger.info(f"[{self.name}] 📈 Sample signal patterns:")
            sample_dates = signals.index[:min(3, len(signals.index))]
            for date in sample_dates:
                date_signals = signals.loc[date]
                non_zero = date_signals[abs(date_signals) > 0.01]
                if not non_zero.empty:
                    logger.info(f"[{self.name}]   {date.date()}: {non_zero.to_dict()}")