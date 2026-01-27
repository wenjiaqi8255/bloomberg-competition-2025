"""
Portfolio Weight Utilities
==========================

Provides centralized, pure functions for common portfolio weight operations
such as normalization, validation, and constraint application. This ensures
consistent logic and adherence to the DRY principle.
"""

import logging
import pandas as pd
from typing import Dict

logger = logging.getLogger(__name__)

class WeightUtils:
    """A utility class for stateless operations on portfolio weights."""

    TOLERANCE = 1e-6

    @staticmethod
    def normalize_weights(weights: pd.Series) -> pd.Series:
        """
        Normalize a series of portfolio weights to sum to 1.0.

        Negative weights (shorts) are preserved, and the portfolio is
        normalized based on the sum of absolute weights (gross exposure).
        If the sum is zero, it returns a zero-weight series.

        Args:
            weights: A pandas Series of portfolio weights.

        Returns:
            A new pandas Series with weights normalized to sum to 1.0.
        """
        if weights.empty:
            return weights

        total_weight = weights.sum()
        if abs(total_weight) < WeightUtils.TOLERANCE:
            logger.warning("Total portfolio weight is near zero; cannot normalize.")
            return pd.Series(0.0, index=weights.index)
        
        normalized = weights / total_weight
        
        # Final check for numerical stability
        if abs(normalized.sum() - 1.0) > WeightUtils.TOLERANCE:
            logger.warning(f"Re-normalizing due to numerical precision issue. Initial sum: {normalized.sum()}")
            normalized = normalized / normalized.sum()
            
        return normalized

    @staticmethod
    def validate_weights(weights: pd.Series, max_leverage: float = 1.0) -> bool:
        """
        Validate portfolio weights against common sanity checks.

        Args:
            weights: A pandas Series of portfolio weights.
            max_leverage: The maximum allowed leverage (sum of absolute weights).

        Returns:
            True if weights are valid, False otherwise.
        """
        if not isinstance(weights, pd.Series):
            logger.error("Weights must be a pandas Series for validation.")
            return False

        # Check for NaN or Inf values
        if not pd.to_numeric(weights, errors='coerce').notna().all():
            logger.error(f"Weights contain NaN or non-numeric values: {weights[weights.isna()]}")
            return False
            
        # Check leverage
        gross_exposure = weights.abs().sum()
        if gross_exposure > max_leverage + WeightUtils.TOLERANCE:
            logger.error(f"Leverage constraint violated. Gross exposure: {gross_exposure:.4f} > {max_leverage}")
            return False
            
        # Check sum if not leveraged
        if max_leverage == 1.0 and abs(weights.sum() - 1.0) > WeightUtils.TOLERANCE:
            logger.error(f"Fully-invested portfolio weights do not sum to 1.0. Sum: {weights.sum():.4f}")
            return False
            
        return True


