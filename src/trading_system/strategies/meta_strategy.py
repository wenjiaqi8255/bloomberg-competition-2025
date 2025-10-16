"""
Meta Strategy
============

A strategy that combines multiple base strategies using a meta-model.
This allows the meta-model to be treated as a regular strategy for backtesting.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

from .base_strategy import BaseStrategy
from ..metamodel.meta_model import MetaModel
from ..models.training.training_pipeline import TrainingPipeline

logger = logging.getLogger(__name__)


class MetaStrategy(BaseStrategy):
    """
    Meta strategy that combines multiple base strategies using a meta-model.
    
    This strategy acts as a wrapper around a meta-model, allowing it to be
    used in the same way as any other strategy for backtesting purposes.
    
    The strategy:
    1. Loads all base models
    2. Generates signals from each base model
    3. Uses the meta-model to combine these signals
    4. Returns the combined signal as the final strategy output
    """

    def __init__(self, 
                 meta_model: MetaModel,
                 base_strategy_ids: List[str],
                 model_registry_path: str = "./models/",
                 name: str = "MetaStrategy"):
        """
        Initialize the meta strategy.
        
        Args:
            meta_model: The trained meta-model for combining strategies
            base_strategy_ids: List of base strategy/model IDs to combine
            model_registry_path: Path to the model registry
            name: Name for this strategy
        """
        super().__init__(name=name)
        
        self.meta_model = meta_model
        self.base_strategy_ids = base_strategy_ids
        self.model_registry_path = model_registry_path
        
        # Load base models
        self.base_models = self._load_base_models()
        
        logger.info(f"MetaStrategy initialized with {len(self.base_models)} base models")
        logger.info(f"Base strategy IDs: {self.base_strategy_ids}")

    def _load_base_models(self) -> Dict[str, Any]:
        """
        Load all base models from the registry.
        
        Returns:
            Dictionary mapping strategy_id to loaded model
        """
        models = {}
        
        for strategy_id in self.base_strategy_ids:
            try:
                model = TrainingPipeline.load_model(
                    self.model_registry_path, 
                    strategy_id
                )
                models[strategy_id] = model
                logger.debug(f"Loaded base model: {strategy_id}")
                
            except Exception as e:
                logger.error(f"Failed to load base model {strategy_id}: {e}")
                raise
        
        return models

    def generate_signals(self, 
                        date: pd.Timestamp, 
                        data: pd.DataFrame) -> pd.Series:
        """
        Generate trading signals by combining base strategy signals.
        
        Args:
            date: Current date for signal generation
            data: Market data for signal generation
            
        Returns:
            Combined signals (symbol -> signal strength)
        """
        logger.debug(f"Generating meta signals for {date}")
        
        # 1. Collect signals from all base models
        base_signals = {}
        
        for strategy_id, model in self.base_models.items():
            try:
                # Generate signals from this base model
                signals = self._generate_base_signals(model, data)
                base_signals[strategy_id] = signals
                
                logger.debug(f"Generated {len(signals)} signals from {strategy_id}")
                
            except Exception as e:
                logger.error(f"Failed to generate signals from {strategy_id}: {e}")
                # Continue with other models, use zero signals for failed model
                base_signals[strategy_id] = pd.Series(0.0, index=data.index)
        
        # 2. Convert to DataFrame (symbols × strategies)
        signals_df = pd.DataFrame(base_signals)
        
        # 3. Use meta-model to combine signals
        combined_signals = self._combine_signals_with_meta_model(signals_df)
        
        logger.debug(f"Generated {len(combined_signals)} combined signals")
        return combined_signals

    def _generate_base_signals(self, model: Any, data: pd.DataFrame) -> pd.Series:
        """
        Generate signals from a single base model.
        
        Args:
            model: The base model
            data: Market data
            
        Returns:
            Signals from the base model
        """
        # This is a simplified implementation
        # In practice, you would need to handle different model types
        # and ensure proper feature engineering
        
        try:
            # For models with predict method
            if hasattr(model, 'predict'):
                signals = model.predict(data)
                if isinstance(signals, np.ndarray):
                    signals = pd.Series(signals, index=data.index)
                return signals
            
            # For models with other prediction methods
            elif hasattr(model, 'generate_signals'):
                return model.generate_signals(data)
            
            else:
                logger.warning("Model does not have predict or generate_signals method")
                return pd.Series(0.0, index=data.index)
                
        except Exception as e:
            logger.error(f"Error generating signals from model: {e}")
            return pd.Series(0.0, index=data.index)

    def _combine_signals_with_meta_model(self, signals_df: pd.DataFrame) -> pd.Series:
        """
        Combine base strategy signals using the meta-model.
        
        Args:
            signals_df: DataFrame with base strategy signals (symbols × strategies)
            
        Returns:
            Combined signals
        """
        # Get strategy weights from meta-model
        weights = self.meta_model.strategy_weights
        
        # Initialize combined signals
        combined_signals = pd.Series(0.0, index=signals_df.index)
        
        # Apply weights to each strategy
        for strategy_id, weight in weights.items():
            if strategy_id in signals_df.columns:
                combined_signals += weight * signals_df[strategy_id]
            else:
                logger.warning(f"Strategy {strategy_id} not found in signals, skipping")
        
        # Normalize if needed (optional)
        if combined_signals.abs().max() > 1.0:
            combined_signals = combined_signals / combined_signals.abs().max()
        
        return combined_signals

    def update_meta_model(self, new_meta_model: MetaModel):
        """
        Update the meta-model (for online learning scenarios).
        
        Args:
            new_meta_model: New meta-model to use
        """
        logger.info("Updating meta-model")
        self.meta_model = new_meta_model
        
        # Update base strategy IDs if they changed
        new_strategy_ids = list(new_meta_model.strategy_weights.keys())
        if new_strategy_ids != self.base_strategy_ids:
            self.base_strategy_ids = new_strategy_ids
            self.base_models = self._load_base_models()
            logger.info("Reloaded base models due to strategy ID changes")

    def get_strategy_weights(self) -> Dict[str, float]:
        """
        Get the current strategy weights from the meta-model.
        
        Returns:
            Dictionary mapping strategy_id to weight
        """
        return self.meta_model.strategy_weights.copy()

    def get_base_strategy_ids(self) -> List[str]:
        """
        Get the list of base strategy IDs.
        
        Returns:
            List of base strategy IDs
        """
        return self.base_strategy_ids.copy()

    def validate_strategy(self) -> bool:
        """
        Validate that the meta strategy is properly configured.
        
        Returns:
            True if valid, False otherwise
        """
        try:
            # Check meta-model
            if self.meta_model is None:
                logger.error("Meta-model is None")
                return False
            
            # Check strategy weights
            weights = self.meta_model.strategy_weights
            if not weights:
                logger.error("No strategy weights found")
                return False
            
            # Check base models
            if len(self.base_models) != len(self.base_strategy_ids):
                logger.error(
                    f"Mismatch between base models ({len(self.base_models)}) "
                    f"and strategy IDs ({len(self.base_strategy_ids)})"
                )
                return False
            
            # Check that all weighted strategies have loaded models
            for strategy_id in weights.keys():
                if strategy_id not in self.base_models:
                    logger.error(f"No model loaded for strategy: {strategy_id}")
                    return False
            
            logger.info("✓ Meta strategy validation passed")
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
            'num_base_strategies': len(self.base_strategy_ids),
            'base_strategy_ids': self.base_strategy_ids,
            'strategy_weights': self.get_strategy_weights(),
            'meta_model_method': getattr(self.meta_model, 'method', 'unknown'),
            'is_valid': self.validate_strategy()
        }




