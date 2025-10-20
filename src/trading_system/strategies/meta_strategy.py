"""
Meta Strategy
============

A strategy that combines multiple base strategies using meta-model weights.
This follows the same pattern as other strategies for consistency.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)


class MetaStrategy(BaseStrategy):
    """
    Meta strategy that combines multiple base strategies using learned weights.

    This strategy manages multiple base strategies and combines their signals
    using weights learned by a MetaModel during training.

    The strategy:
    1. Creates base strategies from model IDs
    2. Generates signals from each base strategy (with proper feature engineering)
    3. Combines signals using meta-model weights
    4. Returns weighted signals as the final strategy output
    """

    def __init__(self,
                 name: str,
                 # Meta strategy configuration (from YAML)
                 base_model_ids: List[str],
                 meta_weights: Dict[str, float],

                 # Standard BaseStrategy parameters
                 feature_pipeline=None,  # MetaStrategy doesn't need its own
                 model_predictor=None,   # MetaStrategy doesn't need its own
                 universe: Optional[List[str]] = None,

                 # Infrastructure providers (critical for base strategies)
                 data_provider=None,
                 factor_data_provider=None,

                 # Configuration
                 model_registry_path: str = "./models/",
                 **kwargs):
        """
        Initialize the meta strategy from configuration.

        Args:
            name: Strategy identifier
            base_model_ids: List of base model IDs to combine
            meta_weights: Dictionary mapping model IDs to weights (sums to 1.0)
            feature_pipeline: Not used for MetaStrategy (can be None)
            model_predictor: Not used for MetaStrategy (can be None)
            universe: Trading universe (list of symbols)
            data_provider: Market data provider for base strategies
            factor_data_provider: Factor data provider for FF5 strategies
            model_registry_path: Path to model registry
            **kwargs: Additional parameters
        """
        super().__init__(
            name=name,
            feature_pipeline=feature_pipeline,
            model_predictor=model_predictor,
            universe=universe,
            **kwargs
        )

        # MetaStrategy-specific attributes
        self.base_model_ids = base_model_ids
        self.meta_weights = meta_weights
        self.universe = universe  # Explicitly store universe

        # ✅ FIXED: MetaStrategy holds providers for base strategy creation only
        # Providers are used for creating base strategies, not for runtime data fetching
        self._data_provider = data_provider
        self._factor_data_provider = factor_data_provider
        self.model_registry_path = model_registry_path

        # Validate configuration
        self._validate_config()

        # Lazy initialization of base strategies
        self._base_strategies = None

        logger.info(f"MetaStrategy initialized: {name}")
        logger.info(f"  Base models: {len(base_model_ids)} - {base_model_ids}")
        logger.info(f"  Meta weights: {meta_weights}")
        logger.info(f"  Universe: {len(universe) if universe else 0} symbols")
        logger.info(f"  Has data_provider: {data_provider is not None}")
        logger.info(f"  Has factor_data_provider: {factor_data_provider is not None}")

    def _validate_config(self):
        """Validate meta strategy configuration."""
        if not self.base_model_ids:
            raise ValueError("MetaStrategy requires base_model_ids")

        if not self.meta_weights:
            raise ValueError("MetaStrategy requires meta_weights")

        # Check that weights match model IDs
        model_ids = set(self.base_model_ids)
        weight_keys = set(self.meta_weights.keys())

        if model_ids != weight_keys:
            missing_weights = model_ids - weight_keys
            extra_weights = weight_keys - model_ids

            error_msg = f"Model IDs and weights don't match:"
            if missing_weights:
                error_msg += f" missing weights for {missing_weights}"
            if extra_weights:
                error_msg += f" extra weights for {extra_weights}"

            raise ValueError(error_msg)

        # Check weights sum to 1.0 (with tolerance)
        total_weight = sum(self.meta_weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0: {total_weight:.4f}, normalizing")
            self.meta_weights = {k: v/total_weight for k, v in self.meta_weights.items()}
            logger.info(f"Normalized weights: {self.meta_weights}")

        # ✅ FIXED: MetaStrategy holds providers for base strategy creation
        # Providers are used to create base strategies, not for runtime data fetching
        if self._data_provider is None:
            logger.warning("MetaStrategy _data_provider is None - base strategies may not initialize correctly")
        if self._factor_data_provider is None:
            logger.info("MetaStrategy _factor_data_provider is None - OK if no FF5 strategies")

        logger.info("✅ MetaStrategy configuration validation passed")

    def _initialize_base_strategies(self):
        """
        Lazy initialization of base strategies.

        This method creates actual strategy instances from model IDs,
        ensuring each strategy has proper data providers and feature engineering.
        """
        if self._base_strategies is not None:
            return  # Already initialized

        logger.info(f"Initializing {len(self.base_model_ids)} base strategies...")

        from .factory import StrategyFactory

        self._base_strategies = {}

        for model_id in self.base_model_ids:
            try:
                # Infer strategy type from model ID
                strategy_type = self._infer_strategy_type(model_id)
                logger.info(f"Creating {strategy_type} strategy for model {model_id}")

                # Create base strategy configuration
                base_config = {
                    'type': strategy_type,
                    'name': f'{strategy_type}_{model_id}',
                    'model_id': model_id,
                    'model_registry_path': self.model_registry_path,
                    'use_fitted_pipeline': True,  # Use saved pipeline from training
                    'universe': self.universe,
                    # Add minimal required parameters
                    'min_signal_strength': 0.00001,
                    'enable_normalization': True,
                    'normalization_method': 'minmax',
                    'enable_short_selling': False
                }

                # ✅ FIXED: Create base strategy with proper providers
                # MetaStrategy provides providers for base strategy creation only
                base_strategy = StrategyFactory.create_from_config(
                    config=base_config,
                    providers={
                        'data_provider': self._data_provider,
                        'factor_data_provider': self._factor_data_provider
                    }
                )

                self._base_strategies[model_id] = base_strategy
                logger.info(f"✅ Created base strategy: {base_strategy.__class__.__name__}")

            except Exception as e:
                logger.error(f"❌ Failed to create base strategy for {model_id}: {e}")
                raise ValueError(f"Could not create base strategy for {model_id}: {e}")

        logger.info(f"✅ Successfully initialized {len(self._base_strategies)} base strategies")

    def _infer_strategy_type(self, model_id: str) -> str:
        """
        Infer strategy type from model ID.

        Args:
            model_id: Model identifier (e.g., 'ff5_regression_20241015')

        Returns:
            Strategy type string
        """
        model_id_lower = model_id.lower()

        if 'ff5' in model_id_lower or 'fama_french' in model_id_lower:
            return 'fama_french_5'
        elif 'fama_macbeth' in model_id_lower:
            return 'fama_macbeth'
        elif 'xgboost' in model_id_lower or 'xgb' in model_id_lower:
            return 'ml'
        elif 'lstm' in model_id_lower or 'rnn' in model_id_lower:
            return 'ml'
        elif 'ridge' in model_id_lower or 'linear' in model_id_lower:
            return 'ml'
        else:
            # Default to ML strategy for unknown models
            logger.warning(f"Unknown model type for {model_id}, defaulting to 'ml'")
            return 'ml'

    def generate_signals(self,
                        pipeline_data: Dict[str, Any],
                        start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        """
        Generate trading signals by combining base strategy signals.

        Args:
            pipeline_data: Complete data prepared by orchestrator
                - 'price_data': Dict[str, DataFrame] (required) - OHLCV price data
                - 'factor_data': DataFrame (optional) - Factor data for FF5 models
            start_date: Start date for signal generation
            end_date: End date for signal generation

        Returns:
            DataFrame with signals (dates × symbols)
        """
        logger.info(f"[{self.name}] 🔄 Generating meta signals from {start_date} to {end_date}")
        logger.info(f"[{self.name}] Pipeline data keys: {list(pipeline_data.keys())}")

        # Initialize base strategies on first call
        self._initialize_base_strategies()

        # Collect signals from all base strategies
        all_signals = {}
        successful_strategies = 0

        for model_id, base_strategy in self._base_strategies.items():
            try:
                logger.info(f"[{self.name}] Getting signals from {model_id} ({base_strategy.__class__.__name__})")

                # ✅ MetaStrategy only passes through pipeline_data
                # Base strategies handle their own feature engineering
                signals = base_strategy.generate_signals(
                    pipeline_data=pipeline_data,  # ✅ Only pass pipeline_data
                    start_date=start_date,
                    end_date=end_date
                )

                if signals is not None and not signals.empty:
                    all_signals[model_id] = signals
                    successful_strategies += 1

                    # Log signal statistics
                    non_zero_signals = (signals != 0).sum().sum()
                    logger.info(f"[{self.name}] ✅ {model_id}: {signals.shape}, non-zero: {non_zero_signals}")
                else:
                    logger.warning(f"[{self.name}] ⚠️ {model_id} returned empty signals")
                    # Create zero signals with correct shape
                    if self.universe:
                        zero_signals = pd.DataFrame(
                            0.0,
                            index=pd.date_range(start_date, end_date),
                            columns=self.universe
                        )
                        all_signals[model_id] = zero_signals

            except Exception as e:
                logger.error(f"[{self.name}] ❌ Failed to get signals from {model_id}: {e}")
                # Create zero signals as fallback
                if self.universe:
                    zero_signals = pd.DataFrame(
                        0.0,
                        index=pd.date_range(start_date, end_date),
                        columns=self.universe
                    )
                    all_signals[model_id] = zero_signals

        logger.info(f"[{self.name}] Generated signals from {successful_strategies}/{len(self._base_strategies)} strategies")

        if not all_signals:
            raise RuntimeError("No signals generated from any base strategy")

        # Combine signals using meta weights
        combined_signals = self._combine_signals(all_signals)

        logger.info(f"[{self.name}] ✅ Combined signals: {combined_signals.shape}")
        logger.info(f"[{self.name}] Non-zero signals: {(combined_signals != 0).sum().sum()}")

        return combined_signals

    def _combine_signals(self, all_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine signals from base strategies using meta weights.

        Args:
            all_signals: Dictionary mapping model_id to signals DataFrame

        Returns:
            Combined signals DataFrame
        """
        logger.info(f"[{self.name}] 🔄 Combining signals using meta weights")

        # Get reference index and columns from first valid signals
        reference_signals = next(iter(all_signals.values()))
        combined = pd.DataFrame(0.0, index=reference_signals.index, columns=reference_signals.columns)

        total_weight = 0.0

        for model_id, signals in all_signals.items():
            weight = self.meta_weights.get(model_id, 0.0)
            if weight == 0.0:
                logger.warning(f"[{self.name}] Zero weight for {model_id}, skipping")
                continue

            # Align signals with combined DataFrame
            aligned_signals = signals.reindex(index=combined.index, columns=combined.columns, fill_value=0.0)

            # Add weighted signals
            combined += weight * aligned_signals
            total_weight += weight

            logger.debug(f"[{self.name}] Added {model_id} signals with weight {weight:.4f}")

        # Normalize if total weight is not 1.0
        if total_weight > 0 and abs(total_weight - 1.0) > 0.01:
            logger.warning(f"[{self.name}] Total weight: {total_weight:.4f}, normalizing")
            combined = combined / total_weight

        # Log combination statistics
        non_zero_count = (combined != 0).sum().sum()
        logger.info(f"[{self.name}] ✅ Combined signals statistics:")
        logger.info(f"[{self.name}]   Shape: {combined.shape}")
        logger.info(f"[{self.name}]   Non-zero signals: {non_zero_count}")
        logger.info(f"[{self.name}]   Mean: {combined.mean().mean():.6f}")
        logger.info(f"[{self.name}]   Std: {combined.std().mean():.6f}")

        return combined

    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Get the current strategy weights.

        Returns:
            Dictionary mapping model_id to weight
        """
        return self.meta_weights.copy()

    def get_base_model_ids(self) -> List[str]:
        """
        Get the list of base model IDs.

        Returns:
            List of base model IDs
        """
        return self.base_model_ids.copy()

    def validate_strategy(self) -> bool:
        """
        Validate that the meta strategy is properly configured.

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check base model IDs
            if not self.base_model_ids:
                logger.error("No base model IDs found")
                return False

            # Check meta weights
            if not self.meta_weights:
                logger.error("No meta weights found")
                return False

            # Check that weights match model IDs
            model_ids = set(self.base_model_ids)
            weight_keys = set(self.meta_weights.keys())

            if model_ids != weight_keys:
                logger.error(f"Model IDs and weights don't match: {model_ids} vs {weight_keys}")
                return False

            # ✅ FIXED: MetaStrategy holds providers for base strategy creation
            # Providers are used during initialization, not during signal generation
            logger.debug("MetaStrategy providers are available for base strategy creation")

            # Check base strategies if initialized
            if self._base_strategies is not None:
                if len(self._base_strategies) != len(self.base_model_ids):
                    logger.error(
                        f"Mismatch between base strategies ({len(self._base_strategies)}) "
                        f"and model IDs ({len(self.base_model_ids)})"
                    )
                    return False

                # Check that all weighted models have loaded strategies
                for model_id in self.meta_weights.keys():
                    if model_id not in self._base_strategies:
                        logger.error(f"No strategy loaded for model: {model_id}")
                        return False

            logger.info("✅ Meta strategy validation passed")
            return True

        except Exception as e:
            logger.error(f"Meta strategy validation failed: {e}")
            return False

    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Get information about this meta strategy.

        Returns:
            Dictionary with strategy information
        """
        return {
            'name': self.name,
            'type': 'MetaStrategy',
            'num_base_strategies': len(self.base_model_ids),
            'base_model_ids': self.base_model_ids,
            'strategy_weights': self.get_strategy_weights(),
            'universe_size': len(self.universe) if self.universe else 0,
            'strategies_initialized': self._base_strategies is not None,
            'is_valid': self.validate_strategy()
        }





