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
        Override to add signal strength filtering for ML predictions.

        ML models may output weak signals that should be filtered out.
        """
        # Get base predictions from parent
        predictions = super()._get_predictions(features, price_data, start_date, end_date)

        if predictions.empty:
            return predictions

        # Filter weak signals
        filtered_predictions = predictions.copy()
        filtered_predictions[filtered_predictions.abs() < self.min_signal_strength] = 0.0

        logger.debug(f"[{self.name}] Filtered weak signals (threshold={self.min_signal_strength})")

        return filtered_predictions

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

            logger.info(f"[{self.name}] ✅ Generated signals for {len(signals.columns)} assets")
            return signals

        except Exception as e:
            logger.error(f"[{self.name}] ❌ Multi-stock signal generation failed: {e}", exc_info=True)
            return pd.DataFrame()

    def _apply_position_sizing(self, predictions: pd.DataFrame, price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Apply position sizing to predictions.

        Args:
            predictions: DataFrame with raw model predictions (dates x symbols)
            price_data: Dictionary with price data for each symbol

        Returns:
            DataFrame with position-sized signals (dates x symbols)
        """
        if predictions.empty:
            return predictions

        # Apply equal weight position sizing for now
        # This can be enhanced later with more sophisticated position sizing methods
        signals = predictions.copy()

        # Normalize signals to sum to 1 (or 0 for neutral positions)
        for date_idx in signals.index:
            date_signals = signals.loc[date_idx]

            # Separate long and short signals
            long_signals = date_signals[date_signals > 0]
            short_signals = date_signals[date_signals < 0]

            # Equal weight among long positions
            if not long_signals.empty:
                long_weight = 1.0 / len(long_signals) if len(long_signals) > 0 else 0
                signals.loc[date_idx, long_signals.index] = long_weight

            # Equal weight among short positions
            if not short_signals.empty:
                short_weight = -1.0 / len(short_signals) if len(short_signals) > 0 else 0
                signals.loc[date_idx, short_signals.index] = short_weight

        # Apply max position size limit if configured
        if hasattr(self, 'max_position_size') and self.max_position_size < 1.0:
            signals = signals.clip(lower=-self.max_position_size, upper=self.max_position_size)

        logger.debug(f"[{self.name}] Applied position sizing to {signals.shape[0]} dates")
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