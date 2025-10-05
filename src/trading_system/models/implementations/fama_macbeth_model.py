"""
Fama-MacBeth Two-Step Regression Model

This module implements the Fama-MacBeth (1973) methodology for cross-sectional
asset pricing. Unlike time-series regression, this model estimates risk premia
by running cross-sectional regressions at each time period.

Methodology:
Step 1: For each time period t, run cross-sectional regression:
        R_it = γ_0t + γ_1t * Feature1_it + γ_2t * Feature2_it + ... + ε_it
        where i indexes stocks and t indexes time

Step 2: Calculate average coefficients across time:
        γ_avg = (1/T) * Σ_t γ_t

Step 3: Use average coefficients for prediction:
        E[R_i] = γ_avg_0 + γ_avg_1 * Feature1_i + γ_avg_2 * Feature2_i + ...

Key Features:
- Handles panel data with MultiIndex(date, symbol)
- Computes time-series of cross-sectional regression coefficients
- Provides statistical significance tests (Fama-MacBeth standard errors)
- Compatible with unified BaseModel interface

References:
- Fama, E. F., & MacBeth, J. D. (1973). Risk, return, and equilibrium:
  Empirical tests. Journal of Political Economy, 81(3), 607-636.
"""

import logging
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import json
import pickle
from sklearn.linear_model import LinearRegression, Ridge
from scipy import stats

from ..base.base_model import BaseModel, ModelStatus, ModelMetadata

logger = logging.getLogger(__name__)


class FamaMacBethModel(BaseModel):
    """
    Fama-MacBeth cross-sectional regression model.
    
    This model estimates factor risk premia by running cross-sectional
    regressions at each time period and averaging the coefficients.
    
    Model Specification:
        Step 1 (Cross-sectional regression at time t):
            R_it = γ_0t + Σ_k(γ_kt * X_kit) + ε_it
        
        Step 2 (Time-series average):
            γ_k = (1/T) * Σ_t γ_kt
        
        Step 3 (Prediction):
            E[R_i] = γ_0 + Σ_k(γ_k * X_ki)
    
    Statistical Properties:
        - Fama-MacBeth standard errors account for time-series correlation
        - t-statistics test if γ_k is significantly different from zero
        - Handles unbalanced panels (different stocks at different times)
    
    Example:
        # Prepare panel data
        features_panel = pd.DataFrame(...)  # MultiIndex(date, symbol)
        target_panel = pd.Series(...)       # MultiIndex(date, symbol)
        
        # Train model
        model = FamaMacBethModel(config={'regularization': 'none'})
        model.fit(features_panel, target_panel)
        
        # Get average coefficients
        print(model.get_average_coefficients())
        
        # Get t-statistics for significance
        print(model.get_coefficient_statistics())
        
        # Predict
        predictions = model.predict(new_features_panel)
    """
    
    def __init__(self, model_type: str = "fama_macbeth", config: Optional[Dict[str, Any]] = None):
        """
        Initialize Fama-MacBeth model.
        
        Args:
            model_type: Model identifier
            config: Configuration dictionary with:
                - regularization: 'none' or 'ridge' (default: 'none')
                - alpha: Regularization strength for ridge (default: 1.0)
                - min_cross_section_size: Minimum stocks per cross-section (default: 3)
                - newey_west_lags: Lags for Newey-West correction (default: None, auto-detect)
        """
        super().__init__(model_type, config)
        
        # Model configuration
        self.regularization = self.config.get('regularization', 'none')
        self.alpha = self.config.get('alpha', 1.0)
        self.min_cross_section_size = self.config.get('min_cross_section_size', 3)
        self.newey_west_lags = self.config.get('newey_west_lags', None)
        
        # Model state (will be populated during training)
        self.gamma_history: List[Dict[str, Any]] = []  # Time series of γ_t
        self.gamma_mean: Optional[Dict[str, float]] = None  # Average γ
        self.gamma_std: Optional[Dict[str, float]] = None  # Std dev of γ
        self.gamma_tstat: Optional[Dict[str, float]] = None  # t-statistics
        self.gamma_pvalue: Optional[Dict[str, float]] = None  # p-values
        self.feature_names: List[str] = []
        self.dates_used: List[pd.Timestamp] = []
        
        logger.info(f"Initialized FamaMacBethModel with regularization={self.regularization}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'FamaMacBethModel':
        """
        Fit the Fama-MacBeth model.
        
        Step 1: Run cross-sectional regression at each date
        Step 2: Calculate time-series average of coefficients
        Step 3: Compute standard errors and t-statistics
        
        Args:
            X: Panel data with MultiIndex(date, symbol), columns = features
            y: Target returns with MultiIndex(date, symbol)
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If data validation fails or insufficient cross-sections
        """
        try:
            # Validate input data
            self._validate_panel_data(X, y)
            
            # Ensure data is aligned
            aligned_data = pd.concat([y.rename('target'), X], axis=1).dropna()
            if len(aligned_data) == 0:
                raise ValueError("No valid data points after alignment and NaN removal")
            
            y_clean = aligned_data['target']
            X_clean = aligned_data.drop('target', axis=1)
            
            # Store feature names
            self.feature_names = list(X_clean.columns)
            
            # Get unique dates
            dates = X_clean.index.get_level_values('date').unique().sort_values()
            self.dates_used = dates.tolist()
            
            logger.info(f"Running Fama-MacBeth regression on {len(dates)} dates")
            
            # Step 1: Cross-sectional regressions
            gamma_history = []
            successful_dates = []
            
            for date in dates:
                try:
                    # Extract cross-section for this date
                    X_cross = X_clean.xs(date, level='date')
                    y_cross = y_clean.xs(date, level='date')
                    
                    # Check minimum size requirement
                    if len(X_cross) < self.min_cross_section_size:
                        logger.debug(f"Skipping {date}: only {len(X_cross)} observations")
                        continue
                    
                    # Run cross-sectional regression
                    if self.regularization == 'ridge':
                        model = Ridge(alpha=self.alpha)
                    else:
                        model = LinearRegression()
                    
                    model.fit(X_cross, y_cross)
                    
                    # Store coefficients for this date
                    gamma_t = {
                        'date': date,
                        'intercept': model.intercept_,
                        'coefs': dict(zip(self.feature_names, model.coef_)),
                        'n_obs': len(X_cross),
                        'r_squared': model.score(X_cross, y_cross)
                    }
                    
                    gamma_history.append(gamma_t)
                    successful_dates.append(date)
                    
                    logger.debug(f"Date {date}: R² = {gamma_t['r_squared']:.4f}, "
                               f"n = {gamma_t['n_obs']}")
                    
                except Exception as e:
                    logger.warning(f"Error in cross-sectional regression for {date}: {e}")
                    continue
            
            if len(gamma_history) == 0:
                raise ValueError("No successful cross-sectional regressions")
            
            logger.info(f"Completed {len(gamma_history)} cross-sectional regressions")
            
            # Store gamma history
            self.gamma_history = gamma_history
            
            # Step 2: Calculate time-series statistics
            self._calculate_coefficient_statistics()
            
            # Update model status and metadata
            self.status = ModelStatus.TRAINED
            self.metadata.training_samples = len(y_clean)
            self.metadata.features = self.feature_names
            self.metadata.hyperparameters.update({
                'gamma_mean': self.gamma_mean,
                'gamma_std': self.gamma_std,
                'gamma_tstat': self.gamma_tstat,
                'gamma_pvalue': self.gamma_pvalue,
                'n_cross_sections': len(gamma_history),
                'n_dates': len(successful_dates),
                'regularization': self.regularization,
                'alpha': self.alpha
            })
            
            # Log results
            self._log_estimation_results()
            
            self.is_trained = True
            return self
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"Failed to fit Fama-MacBeth model: {e}")
            raise
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict expected returns using average coefficients.
        
        Prediction formula:
            E[R_i] = γ_avg_0 + Σ_k(γ_avg_k * X_ki)
        
        Args:
            X: Feature DataFrame (can be panel or cross-sectional)
        
        Returns:
            Array of predicted returns
        
        Raises:
            ValueError: If model is not trained
        """
        if self.status != ModelStatus.TRAINED:
            raise ValueError("Model must be trained before making predictions")
        
        if self.gamma_mean is None:
            raise ValueError("Model coefficients not available")
        
        try:
            # Validate features
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")
            
            # Use only expected features in correct order
            X_pred = X[self.feature_names].copy()
            
            # Extract coefficient values
            intercept = self.gamma_mean['intercept']
            coefs = np.array([self.gamma_mean['coefs'][f] for f in self.feature_names])
            
            # Make predictions: γ_0 + X @ γ
            predictions = intercept + X_pred.values @ coefs
            
            logger.debug(f"Made predictions for {len(predictions)} observations")
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to make predictions: {e}")
            raise
    
    def _validate_panel_data(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Validate that input data is in panel format.
        
        Args:
            X: Features DataFrame
            y: Target Series
        
        Raises:
            ValueError: If data is not in expected format
        """
        # Check MultiIndex
        if not isinstance(X.index, pd.MultiIndex):
            raise ValueError("X must have MultiIndex for panel data")
        
        if not isinstance(y.index, pd.MultiIndex):
            raise ValueError("y must have MultiIndex for panel data")
        
        # Check index names
        x_names = X.index.names
        y_names = y.index.names
        
        if len(x_names) != 2 or len(y_names) != 2:
            raise ValueError("MultiIndex must have exactly 2 levels")
        
        # Check for date and symbol levels
        required_levels = {'date', 'symbol'}
        if set(x_names) != required_levels or set(y_names) != required_levels:
            raise ValueError(f"MultiIndex must have levels: {required_levels}")
        
        # Check alignment
        if not X.index.equals(y.index):
            logger.warning("X and y indices are not identical, will align during processing")
        
        # Check for sufficient data
        dates = X.index.get_level_values('date').unique()
        if len(dates) < 2:
            raise ValueError(f"Need at least 2 time periods, got {len(dates)}")
        
        logger.debug(f"Panel data validation passed: {len(dates)} dates, "
                    f"{len(X.index.get_level_values('symbol').unique())} symbols")
    
    def _calculate_coefficient_statistics(self) -> None:
        """
        Calculate time-series statistics for coefficients.
        
        Computes:
        - Mean (average γ)
        - Standard deviation
        - t-statistics
        - p-values
        
        Uses Fama-MacBeth standard errors (time-series std of cross-sectional coefficients).
        """
        if not self.gamma_history:
            raise ValueError("No gamma history available")
        
        T = len(self.gamma_history)
        
        # Extract time series of coefficients
        intercepts = [g['intercept'] for g in self.gamma_history]
        coef_series = {f: [] for f in self.feature_names}
        
        for gamma_t in self.gamma_history:
            for feature in self.feature_names:
                coef_series[feature].append(gamma_t['coefs'][feature])
        
        # Calculate statistics
        # Mean
        self.gamma_mean = {
            'intercept': np.mean(intercepts),
            'coefs': {f: np.mean(coef_series[f]) for f in self.feature_names}
        }
        
        # Standard deviation
        self.gamma_std = {
            'intercept': np.std(intercepts, ddof=1),
            'coefs': {f: np.std(coef_series[f], ddof=1) for f in self.feature_names}
        }
        
        # t-statistics: t = mean / (std / sqrt(T))
        self.gamma_tstat = {
            'intercept': self.gamma_mean['intercept'] / (self.gamma_std['intercept'] / np.sqrt(T)),
            'coefs': {
                f: self.gamma_mean['coefs'][f] / (self.gamma_std['coefs'][f] / np.sqrt(T))
                for f in self.feature_names
            }
        }
        
        # p-values (two-tailed test)
        df = T - 1  # degrees of freedom
        self.gamma_pvalue = {
            'intercept': 2 * (1 - stats.t.cdf(abs(self.gamma_tstat['intercept']), df)),
            'coefs': {
                f: 2 * (1 - stats.t.cdf(abs(self.gamma_tstat['coefs'][f]), df))
                for f in self.feature_names
            }
        }
        
        logger.debug(f"Calculated coefficient statistics based on {T} time periods")
    
    def _log_estimation_results(self) -> None:
        """Log Fama-MacBeth estimation results."""
        logger.info("=" * 60)
        logger.info("Fama-MacBeth Estimation Results")
        logger.info("=" * 60)
        logger.info(f"Number of time periods: {len(self.gamma_history)}")
        logger.info(f"Number of features: {len(self.feature_names)}")
        logger.info("")
        logger.info("Average Coefficients (Risk Premia):")
        logger.info(f"  Intercept: {self.gamma_mean['intercept']:.6f} "
                   f"(t = {self.gamma_tstat['intercept']:.2f}, "
                   f"p = {self.gamma_pvalue['intercept']:.4f})")
        
        for feature in self.feature_names:
            coef = self.gamma_mean['coefs'][feature]
            tstat = self.gamma_tstat['coefs'][feature]
            pval = self.gamma_pvalue['coefs'][feature]
            significance = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            
            logger.info(f"  {feature:20s}: {coef:10.6f} "
                       f"(t = {tstat:6.2f}, p = {pval:.4f}) {significance}")
        
        logger.info("")
        logger.info("Significance levels: *** p<0.01, ** p<0.05, * p<0.1")
        logger.info("=" * 60)
    
    def get_average_coefficients(self) -> Dict[str, Any]:
        """
        Get average coefficients (risk premia).
        
        Returns:
            Dictionary with intercept and feature coefficients
        """
        if self.gamma_mean is None:
            return {}
        return self.gamma_mean
    
    def get_coefficient_statistics(self) -> pd.DataFrame:
        """
        Get comprehensive statistics for all coefficients.
        
        Returns:
            DataFrame with columns: coefficient, std_error, t_stat, p_value, significance
        """
        if self.gamma_mean is None:
            return pd.DataFrame()
        
        rows = []
        
        # Intercept
        rows.append({
            'feature': 'Intercept',
            'coefficient': self.gamma_mean['intercept'],
            'std_error': self.gamma_std['intercept'] / np.sqrt(len(self.gamma_history)),
            't_stat': self.gamma_tstat['intercept'],
            'p_value': self.gamma_pvalue['intercept'],
            'significant': self.gamma_pvalue['intercept'] < 0.05
        })
        
        # Feature coefficients
        for feature in self.feature_names:
            rows.append({
                'feature': feature,
                'coefficient': self.gamma_mean['coefs'][feature],
                'std_error': self.gamma_std['coefs'][feature] / np.sqrt(len(self.gamma_history)),
                't_stat': self.gamma_tstat['coefs'][feature],
                'p_value': self.gamma_pvalue['coefs'][feature],
                'significant': self.gamma_pvalue['coefs'][feature] < 0.05
            })
        
        return pd.DataFrame(rows)
    
    def get_gamma_time_series(self) -> pd.DataFrame:
        """
        Get time series of cross-sectional regression coefficients.
        
        Returns:
            DataFrame with date index and coefficient columns
        """
        if not self.gamma_history:
            return pd.DataFrame()
        
        rows = []
        for gamma_t in self.gamma_history:
            row = {'date': gamma_t['date'], 'intercept': gamma_t['intercept']}
            row.update(gamma_t['coefs'])
            row['n_obs'] = gamma_t['n_obs']
            row['r_squared'] = gamma_t['r_squared']
            rows.append(row)
        
        df = pd.DataFrame(rows).set_index('date')
        return df
    
    def save(self, path: Union[str, Path]) -> None:
        """
        Save the Fama-MacBeth model to disk.
        
        Args:
            path: Directory path where model should be saved
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Create model dictionary
        model_dict = {
            'gamma_history': self.gamma_history,
            'gamma_mean': self.gamma_mean,
            'gamma_std': self.gamma_std,
            'gamma_tstat': self.gamma_tstat,
            'gamma_pvalue': self.gamma_pvalue,
            'feature_names': self.feature_names,
            'dates_used': [d.isoformat() for d in self.dates_used],
            'regularization': self.regularization,
            'alpha': self.alpha,
            'min_cross_section_size': self.min_cross_section_size
        }
        
        # Save model dictionary
        model_path = path / "model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model_dict, f)
        
        # Save metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata.to_dict(), f, indent=2)
        
        # Save config
        config_path = path / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"FamaMacBethModel saved to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'FamaMacBethModel':
        """
        Load a Fama-MacBeth model from disk.
        
        Args:
            path: Directory path where model is saved
        
        Returns:
            Loaded FamaMacBethModel instance
        """
        path = Path(path)
        
        if not path.exists():
            raise ValueError(f"Model path does not exist: {path}")
        
        # Load metadata
        metadata_path = path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata.from_dict(metadata_dict)
        
        # Load config
        config_path = path / "config.json"
        config = {}
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Load model dictionary
        model_path = path / "model.pkl"
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        # Create instance
        instance = cls(config=config)
        instance.metadata = metadata
        instance.status = ModelStatus.TRAINED
        
        # Restore model state
        instance.gamma_history = model_dict['gamma_history']
        instance.gamma_mean = model_dict['gamma_mean']
        instance.gamma_std = model_dict['gamma_std']
        instance.gamma_tstat = model_dict['gamma_tstat']
        instance.gamma_pvalue = model_dict['gamma_pvalue']
        instance.feature_names = model_dict['feature_names']
        instance.dates_used = [pd.Timestamp(d) for d in model_dict['dates_used']]
        instance.regularization = model_dict['regularization']
        instance.alpha = model_dict['alpha']
        instance.min_cross_section_size = model_dict['min_cross_section_size']
        
        instance.is_trained = True
        
        logger.info(f"FamaMacBethModel loaded from {path}")
        return instance
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance as absolute value of coefficients.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.gamma_mean is None:
            return {}
        
        return {f: abs(self.gamma_mean['coefs'][f]) for f in self.feature_names}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model metadata and statistics
        """
        info = {
            'model_type': self.model_type,
            'status': self.status.value if hasattr(self.status, 'value') else self.status,
            'n_cross_sections': len(self.gamma_history) if self.gamma_history else 0,
            'n_features': len(self.feature_names),
            'features': self.feature_names,
            'regularization': self.regularization,
            'methodology': 'Fama-MacBeth (1973) Two-Step Regression'
        }
        
        if self.gamma_mean:
            info['average_coefficients'] = self.gamma_mean
            info['significant_features'] = [
                f for f in self.feature_names
                if self.gamma_pvalue['coefs'][f] < 0.05
            ]
        
        return info


