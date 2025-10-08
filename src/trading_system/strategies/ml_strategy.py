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
        Override to add basic signal strength filtering for ML predictions.

        SOLID Principle: Strategy only provides expected returns, not position sizing.
        Weight adjustment is delegated to portfolio optimizer layer.
        """
        # Get base predictions from parent
        predictions = super()._get_predictions(features, price_data, start_date, end_date)

        if predictions.empty:
            return predictions

        logger.info(f"[{self.name}] 📊 Raw predictions (expected returns) statistics:")
        logger.info(f"[{self.name}]   Shape: {predictions.shape}")
        logger.info(f"[{self.name}]   Mean: {predictions.mean().mean():.6f}")
        logger.info(f"[{self.name}]   Std: {predictions.std().mean():.6f}")
        logger.info(f"[{self.name}]   Min: {predictions.min().min():.6f}")
        logger.info(f"[{self.name}]   Max: {predictions.max().max():.6f}")

        # Apply strategy-layer normalization (Layer 1 of dual-layer architecture)
        if self.enable_normalization:
            predictions = self._apply_normalization(predictions, self.normalization_method)
            logger.info(f"[{self.name}] 🔄 Applied strategy-layer normalization (method: {self.normalization_method})")
            logger.info(f"[{self.name}]   Normalized range: [{predictions.min().min():.6f}, {predictions.max().max():.6f}]")
        else:
            logger.info(f"[{self.name}] ⚪ Strategy-layer normalization disabled")

        # Apply minimal signal strength filtering - only filter extremely weak signals
        # Strategy layer's role: identify strong vs weak signals, not weight allocation
        filtered_predictions = predictions.copy()

        # Only filter out extremely weak signals (keep most information for optimizer)
        # Use different threshold based on whether signals are normalized
        if self.enable_normalization and self.normalization_method == 'zscore':
            # For Z-score normalized signals, use a reasonable threshold
            weak_signal_threshold = 0.1  # ~0.1 sigma is quite weak
        else:
            # For non-normalized or MinMax signals, use absolute threshold
            weak_signal_threshold = 0.001

        weak_signal_mask = filtered_predictions.abs() < weak_signal_threshold
        num_weak_signals = weak_signal_mask.sum().sum()
        filtered_predictions[weak_signal_mask] = 0.0

        if num_weak_signals > 0:
            logger.info(f"[{self.name}] 🎯 Filtered {num_weak_signals} extremely weak signals (threshold={weak_signal_threshold})")

        logger.info(f"[{self.name}] 📈 Expected returns for portfolio optimizer:")
        logger.info(f"[{self.name}]   Mean: {filtered_predictions.mean().mean():.6f}")
        logger.info(f"[{self.name}]   Std: {filtered_predictions.std().mean():.6f}")
        logger.info(f"[{self.name}]   Non-zero signals: {(filtered_predictions != 0).sum().sum()}")

        return filtered_predictions

    # NOTE: Strategy-layer normalization re-enabled for dual-layer architecture
    # Layer 1: Strategy-level normalization (ensures consistent scale within each strategy)
    # Layer 2: MetaModel-level normalization (ensures fair combination across strategies)
    # SOLID Principle: Each layer has clear responsibility for normalization

    def generate_signals(self,
                        price_data: Dict[str, pd.DataFrame],
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        """
        Generate expected returns using multi-stock enhanced approach.

        SOLID Principle: Strategy is a pure delegate that provides expected returns.
        Portfolio optimization and weight adjustment are handled by separate layers.

        Args:
            price_data: Original price data (may only contain trading universe)
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with expected returns for the trading universe only
        """
        try:
            logger.info(f"[{self.name}] 🔄 Generating expected returns (delegate pattern)...")

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

                expected_returns = predictions[available_universe]
                logger.info(f"[{self.name}] ✅ Filtered to {len(available_universe)} trading universe stocks: {available_universe}")
            else:
                expected_returns = predictions

            # Step 5: Apply minimal signal strength filtering (keep most information for optimizer)
            if hasattr(self, 'min_signal_strength') and self.min_signal_strength > 0:
                # Use higher threshold to avoid removing valuable information for optimizer
                threshold = min(self.min_signal_strength, 0.001)
                expected_returns = expected_returns.copy()
                weak_signals = expected_returns.abs() < threshold
                expected_returns[weak_signals] = 0.0
                logger.debug(f"[{self.name}] Applied minimal filtering (threshold={threshold})")

            # Step 6: Validate expected returns diversity (for debugging)
            if not expected_returns.empty:
                self._log_expected_returns_validation(expected_returns)

            logger.info(f"[{self.name}] ✅ Expected returns ready for portfolio optimizer:")
            logger.info(f"[{self.name}]   Assets: {len(expected_returns.columns)}")
            logger.info(f"[{self.name}]   Periods: {len(expected_returns.index)}")
            logger.info(f"[{self.name}]   Non-zero signals: {(expected_returns != 0).sum().sum()}")

            return expected_returns

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Expected returns generation failed: {e}", exc_info=True)
            return pd.DataFrame()

      # REMOVED: _apply_position_sizing method
    # SOLID Principle: Position sizing and weight allocation are portfolio optimizer's responsibility
    # Strategy layer only provides expected returns; optimizer handles risk-adjusted weight allocation

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

    def _log_expected_returns_validation(self, expected_returns: pd.DataFrame):
        """
        Log validation results for expected returns to help with debugging.

        SOLID Principle: Simple validation without weight adjustment or normalization.

        Args:
            expected_returns: Generated expected returns DataFrame
        """
        logger.info(f"[{self.name}] 🔍 Validating expected returns diversity...")

        if expected_returns.empty:
            logger.error(f"[{self.name}] ❌ No expected returns to validate")
            return

        # Basic validation metrics
        logger.info(f"[{self.name}] 📊 Expected Returns Validation:")
        logger.info(f"[{self.name}]   Shape: {expected_returns.shape}")
        logger.info(f"[{self.name}]   Mean: {expected_returns.mean().mean():.6f}")
        logger.info(f"[{self.name}]   Std: {expected_returns.std().mean():.6f}")
        logger.info(f"[{self.name}]   Min: {expected_returns.min().min():.6f}")
        logger.info(f"[{self.name}]   Max: {expected_returns.max().max():.6f}")

        # Check signal diversity (important for portfolio optimizer)
        time_variance = expected_returns.var(axis=0)
        logger.info(f"[{self.name}]   Average time variance: {time_variance.mean():.8f}")

        # Check for static signals (bad)
        static_signals = (time_variance < 1e-8).sum()
        if static_signals > 0:
            logger.warning(f"[{self.name}] ⚠️ Found {static_signals} static signals")
        else:
            logger.info(f"[{self.name}] ✅ No static signals found")

        # Check cross-sectional variance
        cross_sectional_variance = expected_returns.var(axis=1)
        logger.info(f"[{self.name}]   Cross-sectional variance: {cross_sectional_variance.mean():.8f}")

        # Log sample expected returns for verification
        if not expected_returns.empty:
            sample_date = expected_returns.index[0]
            sample_returns = expected_returns.loc[sample_date]
            non_zero = sample_returns[abs(sample_returns) > 0.001]
            logger.info(f"[{self.name}] 📈 Sample expected returns for {sample_date.date()}:")
            for symbol, ret in non_zero.items():
                logger.info(f"[{self.name}]   {symbol}: {ret:.6f}")

        logger.info(f"[{self.name}] ✅ Expected returns validation complete")