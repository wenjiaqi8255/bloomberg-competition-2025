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
                 available_universe: Optional[List[str]] = None,  # Actually available stocks
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
            available_universe: Actually available stocks (overrides auto-detection if provided)
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

        # Set minimum stocks requirement
        self.minimum_stocks = kwargs.get('minimum_stocks', 10)  # Default minimum 10 stocks

        # Use available universe if provided, otherwise auto-detect model training stocks
        if available_universe is not None:
            self.model_training_stocks = available_universe
            logger.info(f"📊 STRATEGY INITIALIZATION:")
            logger.info(f"  Using provided available universe: {available_universe}")
        elif model_training_stocks is not None:
            self.model_training_stocks = model_training_stocks
            logger.info(f"📊 STRATEGY INITIALIZATION:")
            logger.info(f"  Using provided model training stocks: {model_training_stocks}")
        else:
            self.model_training_stocks = self._detect_model_training_stocks()
            # If detection fails, provide a default based on the model we know about
            if not self.model_training_stocks:
                logger.warning("Model training stock detection failed, using default stocks")
                self.model_training_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'WMT']

        logger.info("="*60)
        logger.info(f"🤖 MULTI-STOCK ML STRATEGY INITIALIZED: {name}")
        logger.info("="*60)
        logger.info(f"📊 Configuration Summary:")
        logger.info(f"  • Trading universe: {len(universe)} stocks: {universe}")
        logger.info(f"  • Model training stocks: {len(self.model_training_stocks)} stocks")
        logger.info(f"  • Minimum stocks required: {self.minimum_stocks}")
        logger.info(f"  • Min signal strength: {min_signal_strength}")
        logger.info(f"  • Data will be fetched for {len(self.model_training_stocks)} stocks to ensure model compatibility")

        # Show which training stocks are also in trading universe
        trading_overlap = set(universe) & set(self.model_training_stocks)
        non_training_stocks = set(universe) - set(self.model_training_stocks)

        logger.info(f"📈 Trading Strategy Details:")
        logger.info(f"  • Stocks in both training and trading: {len(trading_overlap)}: {sorted(trading_overlap)}")
        if non_training_stocks:
            logger.info(f"  • Trading stocks not in training: {len(non_training_stocks)}: {sorted(non_training_stocks)}")

        # Validate minimum stocks requirement
        logger.info(f"🔍 Validating minimum stocks requirement...")
        validation_result = self._validate_minimum_stocks()

        if validation_result:
            logger.info(f"✅ Strategy validation passed - ready for trading")
        else:
            logger.error(f"❌ Strategy validation failed - cannot operate reliably")

        logger.info("="*60)

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

                    # Priority 1: Check for actual_training_stocks (new field)
                    if 'actual_training_stocks' in metadata:
                        symbols = metadata['actual_training_stocks']
                        if symbols and isinstance(symbols, list):
                            logger.info(f"Using actual training stocks from metadata: {symbols}")
                            return symbols
                        elif symbols and isinstance(symbols, str):
                            result = [s.strip() for s in symbols.split(',')]
                            logger.info(f"Parsed actual training stocks from metadata: {result}")
                            return result

                    # Priority 2: Check in tags.symbols (from the metadata structure we saw)
                    if 'tags' in metadata and 'symbols' in metadata['tags']:
                        symbols = metadata['tags']['symbols']

                    # Priority 3: Check in metadata directly
                    elif 'symbols' in metadata:
                        symbols = metadata['symbols']

                    # Priority 4: Check in model_metadata
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

    def _validate_minimum_stocks(self) -> bool:
        """
        Validate that the strategy has enough training stocks to operate properly.

        Returns:
            True if sufficient stocks are available, False otherwise
        """
        try:
            available_count = len(self.model_training_stocks)
            required_count = self.minimum_stocks

            if available_count < required_count:
                logger.error(f"[{self.name}] Insufficient training stocks: {available_count} < {required_count}")
                logger.error(f"[{self.name}] Strategy cannot operate reliably with less than {required_count} stocks")
                return False

            logger.info(f"[{self.name}] ✅ Minimum stocks requirement met: {available_count} >= {required_count}")
            return True

        except Exception as e:
            logger.error(f"[{self.name}] Failed to validate minimum stocks requirement: {e}")
            return False

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
        provided_stocks = set(enhanced_price_data.keys())
        needed_stocks = set(self.model_training_stocks)
        missing_stocks = needed_stocks - provided_stocks

        if not missing_stocks:
            logger.debug(f"[{self.name}] All model training stocks already provided")
            return enhanced_price_data

        logger.info(f"[{self.name}] Fetching data for {len(missing_stocks)} missing model training stocks: {sorted(missing_stocks)}")

        # Initialize data provider
        data_provider = YFinanceProvider()

        # Fetch data for missing stocks
        if missing_stocks:
            logger.info(f"[{self.name}] Attempting to fetch data for {len(missing_stocks)} missing stocks")
            stock_data_dict = data_provider.get_historical_data(
                symbols=list(missing_stocks),
                start_date=start_date,
                end_date=end_date
            )

            # Add fetched data (Data Provider already handles failures internally)
            for symbol, data in stock_data_dict.items():
                if not data.empty:
                    enhanced_price_data[symbol] = data
                    logger.info(f"[{self.name}] ✅ Fetched {len(data)} rows for {symbol}")
                else:
                    logger.warning(f"[{self.name}] ⚠️ Empty data for {symbol}")

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
        Optimized ML prediction using batch processing for better performance.

        This method leverages XGBoost's batch prediction capabilities to optimize
        performance when predicting for multiple stocks across multiple time periods.

        SOLID Principle: Strategy only provides expected returns, not position sizing.
        Weight adjustment is delegated to portfolio optimizer layer.
        """
        try:
            logger.info(f"[{self.name}] 🚀 Starting optimized batch ML predictions...")
            logger.info(f"[{self.name}] Features shape: {features.shape}")
            logger.info(f"[{self.name}] Model training stocks: {len(self.model_training_stocks)}")

            # Check if model supports batch prediction
            try:
                model = self.model_predictor.get_current_model()
                if model and hasattr(model, 'predict_batch'):
                    logger.info(f"[{self.name}] ✅ Using optimized batch prediction")
                    predictions = self._get_predictions_batch_optimized(features, price_data, start_date, end_date)
                else:
                    logger.info(f"[{self.name}] ⚠️ Model doesn't support batch prediction, using standard method")
                    predictions = super()._get_predictions(features, price_data, start_date, end_date)
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to check batch prediction capability: {e}, using standard method")
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
            # For ML strategies, always use min-max normalization to ensure [0,1] range for TradingSignal compatibility
            if self.enable_normalization:
                normalization_method = 'minmax'  # Force min-max for ML strategies
                predictions = self._apply_normalization(predictions, normalization_method)
                logger.info(f"[{self.name}] 🔄 Applied strategy-layer normalization (method: {normalization_method})")
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

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Batch prediction failed, falling back to standard method: {e}")
            # Fallback to parent method
            return super()._get_predictions(features, price_data, start_date, end_date)

    def _get_predictions_batch_optimized(self,
                                         features: pd.DataFrame,
                                         price_data: Dict[str, pd.DataFrame],
                                         start_date: datetime,
                                         end_date: datetime) -> pd.DataFrame:
        """
        Optimized batch prediction implementation for XGBoost models.

        This method groups predictions by symbol and uses batch prediction
        to reduce overhead and improve performance.

        Args:
            features: Computed features DataFrame
            price_data: Original price data
            start_date: Start date for predictions
            end_date: End date for predictions

        Returns:
            DataFrame with predictions indexed by date
        """
        try:
            logger.info(f"[{self.name}] 🚀 Batch-optimized prediction starting...")

            # Create date range for predictions
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            logger.info(f"[{self.name}] Processing {len(dates)} dates for {len(self.model_training_stocks)} symbols")

            # Group features by symbol for batch processing
            symbol_features_dict = {}
            for symbol in self.model_training_stocks:
                try:
                    # Extract all features for this symbol across all dates
                    symbol_features = self._extract_symbol_features(features, symbol)
                    if not symbol_features.empty:
                        symbol_features_dict[symbol] = symbol_features
                        logger.debug(f"[{self.name}] Extracted {len(symbol_features)} feature rows for {symbol}")
                    else:
                        logger.warning(f"[{self.name}] No features found for {symbol}")
                except Exception as e:
                    logger.warning(f"[{self.name}] Failed to extract features for {symbol}: {e}")

            if not symbol_features_dict:
                logger.error(f"[{self.name}] No valid features found for any symbols")
                return pd.DataFrame()

            logger.info(f"[{self.name}] Prepared features for {len(symbol_features_dict)} symbols")

            # Create batch predictions for each date
            predictions_dict = {}
            model = self.model_predictor.model

            for date in dates:
                try:
                    # Prepare batch of features for all symbols on this date
                    date_features_batch = []
                    date_symbols = []

                    for symbol, symbol_features in symbol_features_dict.items():
                        # Try to get features for this specific date
                        date_features = self._extract_symbol_features(features, symbol, date)
                        if not date_features.empty:
                            date_features_batch.append(date_features.iloc[0])  # Get the single row
                            date_symbols.append(symbol)

                    if date_features_batch:
                        # Convert batch to DataFrame for prediction
                        batch_df = pd.DataFrame(date_features_batch, index=date_symbols)
                        logger.debug(f"[{self.name}] Batch prediction for {date}: {batch_df.shape}")

                        # Use model's batch prediction
                        if hasattr(model, 'predict_batch'):
                            # For XGBoost with explicit batch method
                            batch_predictions = model.predict_batch([batch_df])[0]
                        else:
                            # Fallback to standard predict
                            batch_predictions = model.predict(batch_df)

                        # Create Series with predictions indexed by symbols
                        date_predictions = pd.Series(batch_predictions, index=date_symbols, name='prediction')
                        predictions_dict[date] = date_predictions

                        logger.debug(f"[{self.name}] Generated {len(date_predictions)} predictions for {date}")
                    else:
                        logger.debug(f"[{self.name}] No features available for {date}")
                        # Create zero predictions for available symbols
                        zero_predictions = pd.Series([0.0] * len(self.model_training_stocks),
                                                   index=self.model_training_stocks, name='prediction')
                        predictions_dict[date] = zero_predictions

                except Exception as e:
                    logger.error(f"[{self.name}] Batch prediction failed for {date}: {e}")
                    # Create zero predictions as fallback
                    zero_predictions = pd.Series([0.0] * len(self.model_training_stocks),
                                               index=self.model_training_stocks, name='prediction')
                    predictions_dict[date] = zero_predictions

            # Convert dictionary to DataFrame format (dates as index, symbols as columns)
            if predictions_dict:
                predictions_df = pd.DataFrame(predictions_dict).T  # Transpose: rows=dates, cols=symbols
                logger.info(f"[{self.name}] ✅ Batch prediction completed: shape={predictions_df.shape}")
                logger.info(f"[{self.name}] 📊 Columns (symbols): {list(predictions_df.columns)}")
                logger.info(f"[{self.name}] 📅 Index (dates): {predictions_df.index[0]} to {predictions_df.index[-1]}")
                return predictions_df
            else:
                logger.error(f"[{self.name}] No predictions generated")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Batch prediction optimization failed: {e}")
            import traceback
            logger.error(f"[{self.name}] Traceback: {traceback.format_exc()}")
            return pd.DataFrame()

    # NOTE: Strategy-layer normalization re-enabled for dual-layer architecture
    # Layer 1: Strategy-level normalization (ensures consistent scale within each strategy)
    # Layer 2: MetaModel-level normalization (ensures fair combination across strategies)
    # SOLID Principle: Each layer has clear responsibility for normalization

    def generate_signals(self,
                        pipeline_data: Dict[str, Any],
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        """
        Generate expected returns using multi-stock enhanced approach.

        ✅ REFACTORED: Following "Data preparation responsibility moves up to orchestrator" pattern.
        ML strategy only uses pipeline_data provided by orchestrator, doesn't fetch its own data.

        Args:
            pipeline_data: Complete data prepared by orchestrator
                - 'price_data': Dict[str, DataFrame] - OHLCV price data for trading universe
                - 'factor_data': DataFrame - Factor data if needed
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with expected returns for the trading universe only
        """
        try:
            logger.info(f"[{self.name}] 🔄 Generating expected returns from pipeline data (new architecture)")
            logger.info(f"[{self.name}] Pipeline data keys: {list(pipeline_data.keys())}")

            # Step 1: Extract price_data from pipeline_data
            price_data = pipeline_data.get('price_data', {})
            if not price_data:
                logger.error(f"[{self.name}] ❌ No price_data in pipeline_data")
                return pd.DataFrame()

            logger.info(f"[{self.name}] Processing {len(price_data)} symbols from trading universe")

            # Step 2: Ensure we have data for all model training stocks
            enhanced_price_data = self._fetch_additional_stock_data(price_data, start_date, end_date)

            # Step 3: Compute features for all stocks (this is what the model expects)
            # Update pipeline_data with enhanced price data for feature computation
            enhanced_pipeline_data = pipeline_data.copy()
            enhanced_pipeline_data['price_data'] = enhanced_price_data

            features = self._compute_features(enhanced_pipeline_data)

            if features.empty:
                logger.error(f"[{self.name}] Feature computation failed")
                return pd.DataFrame()

            # Step 4: Get predictions from model (model sees all training stocks)
            predictions = self._get_predictions(features, enhanced_price_data, start_date, end_date)

            if predictions.empty:
                logger.error(f"[{self.name}] No predictions generated")
                return pd.DataFrame()

            # Step 5: Filter predictions to only include our trading universe
            if self.universe:
                available_universe = [s for s in self.universe if s in predictions.columns]
                if not available_universe:
                    logger.error(f"[{self.name}] None of the trading universe stocks found in predictions")
                    return pd.DataFrame()

                expected_returns = predictions[available_universe]
                logger.info(f"[{self.name}] ✅ Filtered to {len(available_universe)} trading universe stocks: {available_universe}")
            else:
                expected_returns = predictions

            # Step 6: Apply minimal signal strength filtering (keep most information for optimizer)
            if hasattr(self, 'min_signal_strength') and self.min_signal_strength > 0:
                # Use higher threshold to avoid removing valuable information for optimizer
                threshold = min(self.min_signal_strength, 0.001)
                expected_returns = expected_returns.copy()
                weak_signals = expected_returns.abs() < threshold
                expected_returns[weak_signals] = 0.0
                logger.debug(f"[{self.name}] Applied minimal filtering (threshold={threshold})")

            # Step 7: Validate expected returns diversity (for debugging)
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