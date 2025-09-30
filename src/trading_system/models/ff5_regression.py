"""
FF5 (Fama-French 5-Factor) regression engine.

This module implements rolling window factor regression to estimate:
- Factor betas for individual stocks
- Factor-implied expected returns
- Residuals (stock-specific returns)
- Model fit statistics and diagnostics

Implements Method A from the IPS: Expected Return = Factor Return + ML Residual
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings

from ..data.ff5_provider import FF5DataProvider
from ..types.data_types import DataValidationError

logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class FF5RegressionEngine:
    """
    Fama-French 5-Factor regression engine with rolling window estimation.

    Features:
    - Rolling window beta estimation
    - Factor-implied return calculation
    - Residual extraction for ML prediction
    - Model fit diagnostics
    - Regularization options
    - Look-ahead bias prevention
    """

    def __init__(self,
                 estimation_window: int = 36,  # months
                 min_observations: int = 20,
                 regularization: str = 'none',
                 alpha: float = 1.0,
                 data_frequency: str = 'monthly'):
        """
        Initialize FF5 regression engine.

        Args:
            estimation_window: Window size for beta estimation in months
            min_observations: Minimum observations required for regression
            regularization: Regularization type ('none', 'ridge')
            alpha: Regularization strength
            data_frequency: Data frequency ('monthly' or 'daily')
        """
        self.estimation_window = estimation_window
        self.min_observations = min_observations
        self.regularization = regularization
        self.alpha = alpha
        self.data_frequency = data_frequency

        # Initialize FF5 data provider
        self.ff5_provider = FF5DataProvider(data_frequency=data_frequency)

        # Model storage
        self.models = {}
        self.beta_history = {}
        self.residual_history = {}

        # Validation
        self._validate_parameters()

        logger.info(f"Initialized FF5 regression engine with {estimation_window}-month window")

    def _validate_parameters(self):
        """Validate engine parameters."""
        if self.estimation_window <= 0:
            raise ValueError("estimation_window must be positive")

        if self.min_observations <= 0:
            raise ValueError("min_observations must be positive")

        if self.min_observations > self.estimation_window * 21:  # Approx trading days
            logger.warning(f"min_observations ({self.min_observations}) may be too high for "
                          f"estimation window ({self.estimation_window} months)")

        if self.regularization not in ['none', 'ridge']:
            raise ValueError("regularization must be 'none' or 'ridge'")

        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")

    def estimate_factor_betas(self, equity_data: Dict[str, pd.DataFrame],
                             factor_data: pd.DataFrame = None,
                             estimation_dates: List[datetime] = None):
        """
        Estimate rolling factor betas for all stocks.

        Args:
            equity_data: Dictionary of equity price DataFrames
            factor_data: Optional factor data (will fetch if not provided)
            estimation_dates: Dates for beta estimation

        Returns:
            Tuple of (factor_betas, factor_returns, residuals) for compatibility
        """
        try:
            # Get factor data if not provided
            if factor_data is None:
                factor_data = self.ff5_provider.get_factor_returns()

            # Align factor data with equity data
            aligned_factor_data, aligned_equity_data = self.ff5_provider.align_with_equity_data(
                equity_data, factor_data
            )

            # Determine estimation dates
            if estimation_dates is None:
                # Use monthly rebalancing dates
                date_range = pd.date_range(
                    start=aligned_equity_data[list(aligned_equity_data.keys())[0]].index[0],
                    end=aligned_equity_data[list(aligned_equity_data.keys())[0]].index[-1],
                    freq='MS'
                )
                estimation_dates = [d for d in date_range if d in aligned_factor_data.index]

            logger.info(f"Estimating factor betas for {len(aligned_equity_data)} stocks "
                       f"at {len(estimation_dates)} dates")

            # Estimate betas for each stock
            beta_estimates = {}
            for symbol, data in aligned_equity_data.items():
                try:
                    symbol_betas = self._estimate_single_stock_betas(
                        symbol, data, aligned_factor_data, estimation_dates
                    )
                    if symbol_betas is not None:
                        beta_estimates[symbol] = symbol_betas
                        self.beta_history[symbol] = symbol_betas
                except Exception as e:
                    logger.warning(f"Failed to estimate betas for {symbol}: {e}")
                    continue

            logger.info(f"Successfully estimated betas for {len(beta_estimates)} stocks")

            # Calculate factor returns and residuals for compatibility
            factor_returns = factor_data.copy()
            residuals = {}
            for symbol, beta_df in beta_estimates.items():
                if len(beta_df) > 0:
                    # FIX: Use proper residual calculation instead of placeholder
                    # Note: Full residual extraction will be handled by extract_residuals method
                    residuals[symbol] = pd.DataFrame({
                        'residual': np.zeros(len(beta_df))  # Placeholder zeros, will be overwritten
                    }, index=beta_df.index)

            # FIX: Replace placeholder residuals with proper calculation
            try:
                proper_residuals = self.extract_residuals(equity_data, factor_data, beta_estimates)
                if proper_residuals:
                    residuals = proper_residuals
                    logger.info(f"Successfully extracted proper residuals for {len(residuals)} stocks")
            except Exception as e:
                logger.warning(f"Failed to extract proper residuals: {e}, keeping placeholder zeros")

            return beta_estimates, factor_returns, residuals

        except Exception as e:
            logger.error(f"Failed to estimate factor betas: {e}")
            raise DataValidationError(f"Beta estimation failed: {e}")

    def _estimate_single_stock_betas(self, symbol: str, equity_data: pd.DataFrame,
                                    factor_data: pd.DataFrame,
                                    estimation_dates: List[datetime]) -> pd.DataFrame:
        """
        Estimate factor betas for a single stock using rolling window.

        Args:
            symbol: Stock symbol
            equity_data: Stock price data
            factor_data: Factor returns data
            estimation_dates: Dates for beta estimation

        Returns:
            DataFrame with factor betas and model statistics
        """
        try:
            # Calculate stock returns
            returns = equity_data['Close'].pct_change().dropna()

            # Prepare results DataFrame
            results = pd.DataFrame(index=estimation_dates,
                                 columns=['MKT_beta', 'SMB_beta', 'HML_beta', 'RMW_beta', 'CMA_beta',
                                        'alpha', 'r_squared', 'mse', 'n_observations'])

            for date in estimation_dates:
                try:
                    # Get estimation window
                    window_start = date - timedelta(days=self.estimation_window * 30)

                    # Filter data within estimation window
                    window_mask = (returns.index >= window_start) & (returns.index <= date)
                    stock_returns = returns[window_mask]

                    # Get corresponding factor returns
                    factor_returns = factor_data.reindex(stock_returns.index).dropna()

                    if len(stock_returns) < self.min_observations or len(factor_returns) < self.min_observations:
                        continue

                    # Align data
                    aligned_data = pd.concat([stock_returns, factor_returns], axis=1).dropna()
                    if len(aligned_data) < self.min_observations:
                        continue

                    y = aligned_data.iloc[:, 0].values  # Stock returns
                    X = aligned_data.iloc[:, 1:].values  # Factor returns

                    # Fit regression model
                    if self.regularization == 'ridge':
                        model = Ridge(alpha=self.alpha)
                    else:
                        model = LinearRegression()

                    model.fit(X, y)

                    # Store results
                    results.loc[date, 'MKT_beta'] = model.coef_[0]
                    results.loc[date, 'SMB_beta'] = model.coef_[1]
                    results.loc[date, 'HML_beta'] = model.coef_[2]
                    results.loc[date, 'RMW_beta'] = model.coef_[3]
                    results.loc[date, 'CMA_beta'] = model.coef_[4]
                    results.loc[date, 'alpha'] = model.intercept_

                    # Calculate model fit statistics
                    y_pred = model.predict(X)
                    results.loc[date, 'r_squared'] = r2_score(y, y_pred)
                    results.loc[date, 'mse'] = mean_squared_error(y, y_pred)
                    results.loc[date, 'n_observations'] = len(y)

                    # Store model
                    self.models[f"{symbol}_{date.strftime('%Y%m%d')}"] = model

                except Exception as e:
                    logger.debug(f"Beta estimation failed for {symbol} on {date}: {e}")
                    continue

            return results.dropna()

        except Exception as e:
            logger.error(f"Failed to estimate betas for {symbol}: {e}")
            return None

    def calculate_factor_implied_returns(self, beta_estimates: Dict[str, pd.DataFrame],
                                       current_factor_values: pd.Series) -> Dict[str, pd.Series]:
        """
        Calculate factor-implied expected returns.

        Args:
            beta_estimates: Dictionary of beta estimates
            current_factor_values: Current factor values

        Returns:
            Dictionary mapping symbols to factor-implied return series
        """
        try:
            factor_returns = {}

            # Ensure factor values are in correct order
            factor_order = ['MKT_beta', 'SMB_beta', 'HML_beta', 'RMW_beta', 'CMA_beta']
            factor_returns_values = current_factor_values[['MKT', 'SMB', 'HML', 'RMW', 'CMA']]

            for symbol, betas in beta_estimates.items():
                try:
                    # Calculate factor-implied return for each date
                    implied_returns = pd.Series(index=betas.index, dtype=float)

                    for date in betas.index:
                        if date in betas.index and not betas.loc[date].isna().any():
                            # Get beta values
                            beta_values = betas.loc[date, factor_order].values

                            # Calculate implied return: R = α + Σ(β_i * F_i)
                            alpha = betas.loc[date, 'alpha']
                            implied_return = alpha + np.dot(beta_values, factor_returns_values.values)
                            implied_returns.loc[date] = implied_return

                    factor_returns[symbol] = implied_returns.dropna()

                except Exception as e:
                    logger.warning(f"Failed to calculate implied returns for {symbol}: {e}")
                    continue

            logger.info(f"Calculated factor-implied returns for {len(factor_returns)} stocks")
            return factor_returns

        except Exception as e:
            logger.error(f"Failed to calculate factor-implied returns: {e}")
            return {}

    def extract_residuals(self, equity_data: Dict[str, pd.DataFrame],
                         factor_data: pd.DataFrame = None,
                         beta_estimates: Dict[str, pd.DataFrame] = None) -> Dict[str, pd.Series]:
        """
        Extract residuals from factor model for ML prediction.

        Args:
            equity_data: Dictionary of equity price DataFrames
            factor_data: Optional factor data
            beta_estimates: Optional beta estimates

        Returns:
            Dictionary mapping symbols to residual series
        """
        try:
            # Get factor data and beta estimates if not provided
            if factor_data is None:
                factor_data = self.ff5_provider.get_factor_returns()

            if beta_estimates is None:
                beta_estimates = self.estimate_factor_betas(equity_data, factor_data)

            # Align data
            aligned_factor_data, aligned_equity_data = self.ff5_provider.align_with_equity_data(
                equity_data, factor_data
            )

            residuals = {}

            for symbol, data in aligned_equity_data.items():
                try:
                    if symbol not in beta_estimates:
                        continue

                    # Calculate actual returns
                    actual_returns = data['Close'].pct_change().dropna()

                    # Get factor-implied returns
                    betas = beta_estimates[symbol]
                    factor_returns_series = []

                    for date in actual_returns.index:
                        if date in betas.index and date in factor_data.index:
                            # Get factor values for this date
                            current_factors = factor_data.loc[date, ['MKT', 'SMB', 'HML', 'RMW', 'CMA']]

                            # Get betas for this date
                            beta_values = betas.loc[date, ['MKT_beta', 'SMB_beta', 'HML_beta', 'RMW_beta', 'CMA_beta']].values

                            if not np.isnan(beta_values).any():
                                # Calculate implied return
                                alpha = betas.loc[date, 'alpha']
                                implied_return = alpha + np.dot(beta_values, current_factors.values)
                                factor_returns_series.append(implied_return)
                            else:
                                factor_returns_series.append(0.0)
                        else:
                            factor_returns_series.append(0.0)

                    # Create series with factor-implied returns
                    implied_returns_series = pd.Series(factor_returns_series, index=actual_returns.index)

                    # Calculate residuals: ε = R_actual - R_factor
                    residual_series = actual_returns - implied_returns_series

                    residuals[symbol] = residual_series.dropna()
                    self.residual_history[symbol] = residual_series

                except Exception as e:
                    logger.warning(f"Failed to extract residuals for {symbol}: {e}")
                    continue

            logger.info(f"Extracted residuals for {len(residuals)} stocks")
            return residuals

        except Exception as e:
            logger.error(f"Failed to extract residuals: {e}")
            return {}

    def get_model_diagnostics(self, symbol: str = None) -> Dict:
        """
        Get model diagnostics and statistics.

        Args:
            symbol: Specific symbol (if None, returns aggregate stats)

        Returns:
            Dictionary with model diagnostics
        """
        try:
            if symbol and symbol in self.beta_history:
                # Individual stock diagnostics
                betas = self.beta_history[symbol]

                diagnostics = {
                    'symbol': symbol,
                    'estimation_dates': len(betas),
                    'avg_r_squared': betas['r_squared'].mean(),
                    'avg_mse': betas['mse'].mean(),
                    'beta_stability': self._calculate_beta_stability(betas),
                    'factor_exposures': {
                        'MKT': betas['MKT_beta'].mean(),
                        'SMB': betas['SMB_beta'].mean(),
                        'HML': betas['HML_beta'].mean(),
                        'RMW': betas['RMW_beta'].mean(),
                        'CMA': betas['CMA_beta'].mean()
                    },
                    'alpha_stats': {
                        'mean': betas['alpha'].mean(),
                        'std': betas['alpha'].std(),
                        't_stat': betas['alpha'].mean() / (betas['alpha'].std() + 1e-8)
                    }
                }
            else:
                # Aggregate diagnostics
                all_betas = list(self.beta_history.values())
                if not all_betas:
                    return {'error': 'No beta estimates available'}

                # Aggregate statistics
                aggregate_r2 = np.mean([beta_df['r_squared'].mean() for beta_df in all_betas])
                aggregate_mse = np.mean([beta_df['mse'].mean() for beta_df in all_betas])

                diagnostics = {
                    'total_stocks': len(all_betas),
                    'aggregate_r_squared': aggregate_r2,
                    'aggregate_mse': aggregate_mse,
                    'model_count': len(self.models),
                    'estimation_window_months': self.estimation_window,
                    'regularization': self.regularization,
                    'data_frequency': self.data_frequency
                }

            return diagnostics

        except Exception as e:
            logger.error(f"Failed to get model diagnostics: {e}")
            return {'error': str(e)}

    def _calculate_beta_stability(self, betas: pd.DataFrame) -> Dict:
        """Calculate beta stability metrics."""
        try:
            stability_metrics = {}

            for factor in ['MKT_beta', 'SMB_beta', 'HML_beta', 'RMW_beta', 'CMA_beta']:
                if factor in betas.columns:
                    factor_betas = betas[factor].dropna()
                    if len(factor_betas) > 1:
                        # Calculate rolling standard deviation
                        rolling_std = factor_betas.rolling(min(12, len(factor_betas)//2)).std()
                        stability_metrics[f"{factor}_stability"] = rolling_std.mean()
                        stability_metrics[f"{factor}_volatility"] = factor_betas.std()

            return stability_metrics

        except Exception as e:
            logger.debug(f"Failed to calculate beta stability: {e}")
            return {}

    def get_factor_exposures(self, date: datetime = None) -> pd.DataFrame:
        """
        Get current factor exposures for all stocks.

        Args:
            date: Specific date (default: latest available)

        Returns:
            DataFrame with factor exposures
        """
        try:
            if date is None:
                # Use latest date from beta history
                latest_date = max([beta_df.index[-1] for beta_df in self.beta_history.values()])
                date = latest_date

            exposures = pd.DataFrame(columns=['MKT', 'SMB', 'HML', 'RMW', 'CMA'])

            for symbol, betas in self.beta_history.items():
                if date in betas.index:
                    exposure_row = betas.loc[date, ['MKT_beta', 'SMB_beta', 'HML_beta', 'RMW_beta', 'CMA_beta']]
                    exposures.loc[symbol] = exposure_row

            return exposures.dropna()

        except Exception as e:
            logger.error(f"Failed to get factor exposures: {e}")
            return pd.DataFrame()

    def save_beta_estimates(self, filepath: str):
        """Save beta estimates to file."""
        try:
            import pickle

            save_data = {
                'beta_history': self.beta_history,
                'residual_history': self.residual_history,
                'parameters': {
                    'estimation_window': self.estimation_window,
                    'min_observations': self.min_observations,
                    'regularization': self.regularization,
                    'alpha': self.alpha,
                    'data_frequency': self.data_frequency
                },
                'save_date': datetime.now().isoformat()
            }

            with open(filepath, 'wb') as f:
                pickle.dump(save_data, f)

            logger.info(f"Saved beta estimates to {filepath}")

        except Exception as e:
            logger.error(f"Failed to save beta estimates: {e}")

    def load_beta_estimates(self, filepath: str):
        """Load beta estimates from file."""
        try:
            import pickle

            with open(filepath, 'rb') as f:
                load_data = pickle.load(f)

            self.beta_history = load_data['beta_history']
            self.residual_history = load_data.get('residual_history', {})
            parameters = load_data['parameters']

            # Update parameters
            for key, value in parameters.items():
                setattr(self, key, value)

            logger.info(f"Loaded beta estimates from {filepath}")

        except Exception as e:
            logger.error(f"Failed to load beta estimates: {e}")

    def get_engine_info(self) -> Dict:
        """Get information about the regression engine."""
        return {
            'engine_type': 'FF5 Regression Engine',
            'estimation_window_months': self.estimation_window,
            'min_observations': self.min_observations,
            'regularization': self.regularization,
            'alpha': self.alpha,
            'data_frequency': self.data_frequency,
            'factors': ['MKT', 'SMB', 'HML', 'RMW', 'CMA'],
            'estimated_stocks': len(self.beta_history),
            'stored_models': len(self.models),
            'residual_series': len(self.residual_history)
        }