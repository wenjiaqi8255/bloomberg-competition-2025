"""
Meta-Model for Strategy Combination

This module provides a MetaModel class responsible for combining signals
from multiple trading strategies into a single, unified signal.

The key responsibilities of the MetaModel are:
- To learn the optimal weights for each strategy.
- To combine strategy signals based on these weights.
- To support different methods for learning and combination, such as
  equal weighting, Lasso/Ridge regression, or dynamic weighting schemes.

Now inherits from BaseModel to leverage existing model infrastructure.
"""

import logging
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from ..models.base.base_model import BaseModel

logger = logging.getLogger(__name__)

class MetaModel(BaseModel):
    """
    The MetaModel combines signals from multiple strategies into a single,
    diversified portfolio signal. It can learn the optimal allocation
    to each strategy over time.

    Now inherits from BaseModel to leverage existing model infrastructure
    including ModelRegistry, persistence, and training pipeline integration.
    """

    def __init__(self, method: str = 'equal', **kwargs: Any):
        """
        Initializes the MetaModel.

        Args:
            method: The method for combining strategies. Supported methods:
                    'equal', 'lasso', 'ridge', 'dynamic'.
            **kwargs: Additional parameters for the chosen method.
        """
        if method not in ['equal', 'lasso', 'ridge', 'dynamic']:
            raise ValueError(f"Unsupported MetaModel method: {method}")

        # Initialize BaseModel with proper model_type and config
        config = {
            'method': method,
            **kwargs
        }
        super().__init__(model_type=f"metamodel_{method}", config=config)

        self.method = method
        self.strategy_weights: Dict[str, float] = {}
        self.model = None  # For regression-based models (sklearn models)
        self.params = kwargs

        logger.info(f"MetaModel initialized with method '{self.method}'.")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MetaModel':
        """
        Implement BaseModel's fit method for strategy weight learning.

        Args:
            X: DataFrame of strategy returns (strategies as columns)
            y: Target portfolio returns

        Returns:
            Self for method chaining
        """
        # Convert DataFrame to the format expected by existing logic
        strategy_returns = {col: X[col] for col in X.columns}
        target_returns = y

        # Use existing fit logic
        self._fit_strategy_weights(strategy_returns, target_returns)

        # Update BaseModel metadata
        self.status = "trained"
        self.metadata.hyperparameters.update(self.params)
        self.metadata.features = list(strategy_returns.keys())

        return self

    def _fit_strategy_weights(self,
                             strategy_returns: Dict[str, pd.Series],
                             target_returns: pd.Series):
        """
        Trains the meta-model to learn the optimal weights for each strategy.

        Args:
            strategy_returns: A dictionary where keys are strategy names and
                              values are pd.Series of historical returns for
                              each strategy.
            target_returns: A pd.Series of the target portfolio returns
                            (e.g., benchmark or actual historical returns).
        """
        logger.info(f"Fitting MetaModel using '{self.method}' method...")

        if self.method == 'equal':
            n_strategies = len(strategy_returns)
            if n_strategies > 0:
                weight = 1.0 / n_strategies
                self.strategy_weights = {name: weight for name in strategy_returns.keys()}
            logger.info(f"Assigned equal weight ({weight:.2%}) to {n_strategies} strategies.")

        elif self.method in ['lasso', 'ridge']:
            from sklearn.linear_model import Lasso, Ridge

            X = pd.DataFrame(strategy_returns)
            y = target_returns

            # Align data
            X, y = X.align(y, join='inner', axis=0)

            if X.empty:
                logger.error("No aligned data for MetaModel fitting. Aborting.")
                return

            if self.method == 'lasso':
                alpha = self.params.get('alpha', 0.01)
                self.model = Lasso(alpha=alpha, positive=True) # Ensure weights are positive
            else: # ridge
                alpha = self.params.get('alpha', 1.0)
                self.model = Ridge(alpha=alpha, positive=True)

            self.model.fit(X, y)

            # Normalize coefficients to sum to 1
            raw_coef = self.model.coef_
            coef_sum = raw_coef.sum()
            if coef_sum > 0:
                normalized_coef = raw_coef / coef_sum
            else:
                logger.warning("Sum of coefficients is zero or negative. Cannot normalize.")
                normalized_coef = raw_coef

            self.strategy_weights = dict(zip(X.columns, normalized_coef))
            logger.info(f"Fitted {self.method} model. Learned weights: {self.strategy_weights}")

        else:
            logger.warning(f"Fit method not implemented for '{self.method}'. Using equal weights as fallback.")
            self._fit_strategy_weights({'equal': pd.Series()}, pd.Series()) # Fallback to equal

    def combine(self, strategy_signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combines signals from multiple strategies using the learned weights.

        Args:
            strategy_signals: A dictionary where keys are strategy names and
                              values are DataFrames of signals (expected returns
                              or weights) from each strategy.

        Returns:
            A single DataFrame representing the combined signal.
        """
        if not self.strategy_weights:
            logger.warning("MetaModel has no strategy weights. Using equal weighting as a default.")
            n_strategies = len(strategy_signals)
            if n_strategies == 0:
                return pd.DataFrame()
            default_weight = 1.0 / n_strategies
            self.strategy_weights = {name: default_weight for name in strategy_signals.keys()}

        logger.info(f"Combining {len(strategy_signals)} signals using weights: {self.strategy_weights}")

        combined_signal = pd.DataFrame()
        
        for strategy_name, signals_df in strategy_signals.items():
            weight = self.strategy_weights.get(strategy_name, 0)
            if weight == 0:
                logger.debug(f"Skipping strategy '{strategy_name}' with zero weight.")
                continue

            # Ensure signals_df is a DataFrame
            if isinstance(signals_df, pd.Series):
                signals_df = signals_df.to_frame().T
            
            if signals_df.empty:
                logger.warning(f"Signal DataFrame for '{strategy_name}' is empty.")
                continue

            weighted_signals = signals_df * weight

            if combined_signal.empty:
                combined_signal = weighted_signals
            else:
                # Use .add for robust alignment of columns and indices
                combined_signal = combined_signal.add(weighted_signals, fill_value=0)
        
        if combined_signal.empty:
            logger.warning("Combined signal is empty after processing all strategies.")
        else:
            logger.info(f"Successfully combined signals into a single DataFrame with shape {combined_signal.shape}")

        return combined_signal

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Implement BaseModel's predict method for portfolio signal generation.

        Args:
            X: DataFrame of strategy signals. Can be either:
               - Single index: strategies as columns, dates as index
               - MultiIndex: (symbol, date) as index, features as columns

        Returns:
            Array of combined portfolio signals
        """
        if not self.strategy_weights:
            logger.warning("MetaModel has no strategy weights. Using equal weighting.")

        # Handle MultiIndex format (from TrainingPipeline)
        if isinstance(X.index, pd.MultiIndex):
            return self._predict_multiindex(X)
        # Handle single index format (direct strategy returns)
        else:
            return self._predict_single_index(X)

    def _predict_multiindex(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using MultiIndex format data from TrainingPipeline.

        Args:
            X: MultiIndex DataFrame with (symbol, date) index

        Returns:
            Array of combined portfolio signals
        """
        # Extract unique symbols (strategies) from the MultiIndex
        symbols = X.index.get_level_values('symbol').unique()

        if not self.strategy_weights:
            # Use equal weighting if no weights available
            n_strategies = len(symbols)
            if n_strategies == 0:
                return np.array([])
            default_weight = 1.0 / n_strategies
            self.strategy_weights = {symbol: default_weight for symbol in symbols}

        # Collect signals for each strategy
        strategy_signals = {}
        for symbol in symbols:
            # Get data for this strategy
            strategy_data = X.loc[symbol]

            # For MetaModel, we expect a feature column representing strategy returns
            # Try to find the most appropriate feature column
            if 'strategy_return' in strategy_data.columns:
                signal_values = strategy_data['strategy_return']
            elif len(strategy_data.columns) == 1:
                # If only one column, use it directly
                signal_values = strategy_data.iloc[:, 0]
            else:
                # Use the first column as fallback
                signal_values = strategy_data.iloc[:, 0]
                logger.warning(f"Using first column '{signal_values.name}' for strategy '{symbol}'")

            # Create DataFrame in expected format for combine method
            strategy_df = signal_values.to_frame().T
            strategy_signals[symbol] = strategy_df

        # Use existing combine method
        combined_signal = self.combine(strategy_signals)

        # Return combined values
        if not combined_signal.empty:
            return combined_signal.iloc[0].values
        else:
            return np.array([])

    def _predict_single_index(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using single index format (strategies as columns).

        Args:
            X: DataFrame with strategies as columns

        Returns:
            Array of combined portfolio signals
        """
        if not self.strategy_weights:
            n_strategies = len(X.columns)
            if n_strategies == 0:
                return np.array([])
            default_weight = 1.0 / n_strategies
            self.strategy_weights = {col: default_weight for col in X.columns}

        # Convert DataFrame to strategy_signals format for existing combine method
        strategy_signals = {}
        for strategy in X.columns:
            # Create a single-row DataFrame for each strategy
            strategy_df = pd.DataFrame({strategy: X[strategy]})
            strategy_signals[strategy] = strategy_df

        # Use existing combine method
        combined_signal = self.combine(strategy_signals)

        # Return the first row (since we're predicting for a single time point)
        if not combined_signal.empty:
            return combined_signal.iloc[0].values
        else:
            return np.array([])

    # Compatibility method for existing code that uses old interface
    def fit_legacy(self,
                   strategy_returns: Dict[str, pd.Series],
                   target_returns: pd.Series):
        """
        Legacy fit method for backward compatibility.
        New code should use fit(X, y) method.
        """
        X = pd.DataFrame(strategy_returns)
        self.fit(X, target_returns)

    # Note: Persistence methods are removed since BaseModel and ModelRegistry provide this functionality
    # Use BaseModel's serialization or ModelRegistry for saving/loading models


