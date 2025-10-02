"""
Position Sizing and Risk Management Utilities

This module provides reusable components for applying pre-trade risk management
to trading signals. It decouples the logic of position sizing from the core
strategy's signal generation logic.

Key Responsibilities:
- Volatility targeting
- Position size constraints
- Weight normalization
- Leverage control
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class PositionSizer:
    """
    Position sizing and risk management component.
    
    This class applies pre-trade risk controls to raw trading signals,
    converting signal strengths into actual position sizes based on:
    - Volatility targeting
    - Maximum position weight constraints
    - Total portfolio leverage limits
    
    Example:
        position_sizer = PositionSizer(
            volatility_target=0.15,
            max_position_weight=0.10
        )
        
        adjusted_signals = position_sizer.adjust_signals(
            raw_signals, 
            asset_volatilities
        )
    """
    
    def __init__(self,
                 volatility_target: float = 0.15,
                 max_position_weight: float = 0.10,
                 max_leverage: float = 1.0,
                 min_position_weight: float = 0.01):
        """
        Initialize position sizer.
        
        Args:
            volatility_target: Target portfolio volatility (e.g., 0.15 = 15%)
            max_position_weight: Maximum weight for any single position
            max_leverage: Maximum total portfolio leverage
            min_position_weight: Minimum position weight (positions below this are zeroed)
        """
        self.volatility_target = volatility_target
        self.max_position_weight = max_position_weight
        self.max_leverage = max_leverage
        self.min_position_weight = min_position_weight
        
        logger.debug(f"Initialized PositionSizer: vol_target={volatility_target}, "
                    f"max_weight={max_position_weight}")
    
    def adjust_signals(self,
                      raw_signals: pd.DataFrame,
                      asset_volatilities: pd.Series,
                      target_volatility: Optional[float] = None) -> pd.DataFrame:
        """
        Apply position sizing and risk management to raw signals.
        
        Args:
            raw_signals: DataFrame with raw signal strengths (-1 to 1)
                        Index: dates, Columns: symbols
            asset_volatilities: Series mapping symbols to their volatilities
            target_volatility: Optional override for volatility target
        
        Returns:
            DataFrame with risk-adjusted position weights
        """
        if raw_signals.empty:
            return raw_signals
        
        target_vol = target_volatility or self.volatility_target
        adjusted_signals = raw_signals.copy()
        
        # Apply volatility scaling to each position
        for symbol in adjusted_signals.columns:
            if symbol in asset_volatilities.index:
                asset_vol = asset_volatilities[symbol]
                if asset_vol > 0:
                    # Scale position by inverse volatility
                    vol_scalar = target_vol / asset_vol
                    adjusted_signals[symbol] = adjusted_signals[symbol] * vol_scalar
        
        # Apply position constraints
        adjusted_signals = self._apply_position_constraints(adjusted_signals)
        
        # Normalize weights to ensure proper portfolio allocation
        adjusted_signals = self._normalize_weights(adjusted_signals)
        
        return adjusted_signals
    
    def _apply_position_constraints(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Apply maximum and minimum position weight constraints.
        
        Args:
            signals: Signal DataFrame
        
        Returns:
            Constrained signals
        """
        constrained = signals.copy()
        
        # Apply maximum position weight
        constrained = constrained.clip(
            lower=-self.max_position_weight,
            upper=self.max_position_weight
        )
        
        # Zero out positions below minimum threshold
        mask = constrained.abs() < self.min_position_weight
        constrained[mask] = 0.0
        
        return constrained
    
    def _normalize_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize position weights to sum to max_leverage.
        
        Args:
            signals: Signal DataFrame
        
        Returns:
            Normalized signals
        """
        normalized = signals.copy()
        
        for idx in signals.index:
            row = signals.loc[idx]
            total_abs_weight = row.abs().sum()
            
            if total_abs_weight > self.max_leverage:
                # Scale down to respect leverage constraint
                scale_factor = self.max_leverage / total_abs_weight
                normalized.loc[idx] = row * scale_factor
            elif total_abs_weight > 0:
                # Optionally normalize to fully utilize leverage
                # (comment out if you want to allow under-leveraged portfolios)
                # scale_factor = self.max_leverage / total_abs_weight
                # normalized.loc[idx] = row * scale_factor
                pass
        
        return normalized
    
    def calculate_position_sizes(self,
                                signals: pd.DataFrame,
                                portfolio_value: float,
                                prices: pd.Series) -> pd.DataFrame:
        """
        Convert position weights to actual position sizes (shares).
        
        Args:
            signals: DataFrame with position weights
            portfolio_value: Total portfolio value
            prices: Current prices for each symbol
        
        Returns:
            DataFrame with position sizes in shares
        """
        position_sizes = signals.copy()
        
        for symbol in signals.columns:
            if symbol in prices.index and prices[symbol] > 0:
                # Convert weight to dollar amount
                dollar_amount = signals[symbol] * portfolio_value
                # Convert to shares
                position_sizes[symbol] = dollar_amount / prices[symbol]
            else:
                position_sizes[symbol] = 0
        
        return position_sizes
    
    def get_risk_metrics(self, signals: pd.DataFrame, 
                        asset_volatilities: pd.Series) -> dict:
        """
        Calculate risk metrics for current positions.
        
        Args:
            signals: Current position weights
            asset_volatilities: Asset volatilities
        
        Returns:
            Dictionary with risk metrics
        """
        if signals.empty:
            return {}
        
        # Get the latest positions
        latest_positions = signals.iloc[-1]
        
        # Calculate portfolio volatility (simplified - assumes uncorrelated assets)
        position_variances = []
        for symbol, weight in latest_positions.items():
            if symbol in asset_volatilities.index:
                var = (weight * asset_volatilities[symbol]) ** 2
                position_variances.append(var)
        
        portfolio_vol = np.sqrt(sum(position_variances)) if position_variances else 0
        
        # Other metrics
        total_leverage = latest_positions.abs().sum()
        max_position = latest_positions.abs().max()
        num_positions = (latest_positions.abs() > self.min_position_weight).sum()
        
        return {
            'portfolio_volatility': portfolio_vol,
            'total_leverage': total_leverage,
            'max_position_weight': max_position,
            'num_positions': num_positions,
            'volatility_target': self.volatility_target,
            'max_allowed_leverage': self.max_leverage
        }
    
    def validate_signals(self, signals: pd.DataFrame) -> tuple[bool, str]:
        """
        Validate that signals comply with risk constraints.
        
        Args:
            signals: Signal DataFrame to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if signals.empty:
            return False, "Empty signals"
        
        # Check signal range
        if (signals.abs() > 1.0).any().any():
            return False, "Signals exceed [-1, 1] range before position sizing"
        
        # Check for NaN or inf values
        if signals.isna().any().any():
            return False, "Signals contain NaN values"
        
        if np.isinf(signals.values).any():
            return False, "Signals contain infinite values"
        
        return True, "Valid"
    
    def __repr__(self):
        return (f"PositionSizer(volatility_target={self.volatility_target}, "
                f"max_position_weight={self.max_position_weight}, "
                f"max_leverage={self.max_leverage})")

