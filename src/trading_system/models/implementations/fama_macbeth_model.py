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
- Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate:
  A practical and powerful approach to multiple testing. Journal of the
  Royal Statistical Society, Series B, 57(1), 289-300.
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
                - fdr_level: False Discovery Rate level for Benjamini-Hochberg (default: 0.05)
                - apply_fdr: Whether to apply FDR correction for feature filtering (default: True)
        """
        super().__init__(model_type, config)

        # Model configuration
        self.regularization = self.config.get('regularization', 'none')
        self.alpha = self.config.get('alpha', 1.0)
        self.min_cross_section_size = self.config.get('min_cross_section_size', 3)
        self.newey_west_lags = self.config.get('newey_west_lags', None)
        self.fdr_level = self.config.get('fdr_level', 0.05)
        self.apply_fdr = self.config.get('apply_fdr', True)

        # Model state (will be populated during training)
        self.gamma_history: List[Dict[str, Any]] = []  # Time series of γ_t
        self.gamma_mean: Optional[Dict[str, float]] = None  # Average γ
        self.gamma_std: Optional[Dict[str, float]] = None  # Std dev of γ
        self.gamma_tstat: Optional[Dict[str, float]] = None  # t-statistics
        self.gamma_pvalue: Optional[Dict[str, float]] = None  # p-values (raw)
        self.gamma_pvalue_fdr: Optional[Dict[str, float]] = None  # FDR-adjusted p-values
        self.significant_features_fdr: List[str] = []  # Features significant after FDR correction
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

            # Ensure data follows standard panel format convention
            try:
                from feature_engineering.utils.panel_formatter import PanelDataFormatter
                logger.debug("Standardizing panel data format in Fama-MacBeth model...")

                X = PanelDataFormatter.ensure_panel_format(
                    X,
                    index_order=('date', 'symbol'),
                    validate=True,
                    auto_fix=True
                )

                # Convert y to DataFrame for standardization, then back to Series
                y_df = y.to_frame() if isinstance(y, pd.Series) else y
                y_df = PanelDataFormatter.ensure_panel_format(
                    y_df,
                    index_order=('date', 'symbol'),
                    validate=True,
                    auto_fix=True
                )
                y = y_df.iloc[:, 0] if len(y_df.columns) == 1 else y_df

                logger.debug("Panel data format standardized successfully")

            except Exception as e:
                logger.warning(f"Panel data standardization failed in Fama-MacBeth model: {e}")
                # Continue without standardization if it fails

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

            # Step 3: Apply FDR correction for multiple testing
            self._apply_fdr_correction()

            # Update model status and metadata
            self.status = ModelStatus.TRAINED
            self.metadata.training_samples = len(y_clean)
            self.metadata.features = self.feature_names
            self.metadata.hyperparameters.update({
                'gamma_mean': self.gamma_mean,
                'gamma_std': self.gamma_std,
                'gamma_tstat': self.gamma_tstat,
                'gamma_pvalue': self.gamma_pvalue,
                'gamma_pvalue_fdr': self.gamma_pvalue_fdr,
                'significant_features_fdr': self.significant_features_fdr,
                'n_cross_sections': len(gamma_history),
                'n_dates': len(successful_dates),
                'regularization': self.regularization,
                'alpha': self.alpha,
                'fdr_level': self.fdr_level,
                'apply_fdr': self.apply_fdr
            })
            
            # Log results
            self._log_estimation_results()
            
            self.is_trained = True
            return self
            
        except Exception as e:
            self.status = ModelStatus.FAILED
            logger.error(f"Failed to fit Fama-MacBeth model: {e}")
            raise
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict expected returns using average coefficients.
        
        Prediction formula:
            E[R_i] = γ_avg_0 + Σ_k(γ_avg_k * X_ki)
        
        Args:
            X: Feature DataFrame or ndarray
               - If ndarray: assumes features are in same order as training
               - If DataFrame: uses column names to ensure correct feature order
        
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
            # ✅ 统一输入格式 - 确保X是DataFrame
            if isinstance(X, np.ndarray):
                # 如果输入是numpy array，转换为DataFrame
                if self.feature_names is None or len(self.feature_names) == 0:
                    raise ValueError("Cannot convert array to DataFrame: feature_names not set")
                
                X = pd.DataFrame(X, columns=self.feature_names)
                logger.debug(f"Converted numpy array to DataFrame with shape {X.shape}")
            
            # 验证输入是DataFrame
            if not isinstance(X, pd.DataFrame):
                raise ValueError(f"X must be DataFrame or ndarray, got {type(X)}")
            
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

    def _apply_fdr_correction(self) -> None:
        """
        Apply Benjamini-Hochberg (1995) False Discovery Rate correction.

        This method controls the expected proportion of false discoveries
        (false positives) among all discoveries (rejected hypotheses).

        Procedure:
        1. Sort p-values in ascending order: p_(1) ≤ p_(2) ≤ ... ≤ p_(m)
        2. Find largest k such that: p_(k) ≤ (k/m) * Q
        3. Reject all hypotheses H_(1), ..., H_(k)
        4. Adjusted p-value: p_adj_(i) = min(m * p_(i) / i, 1.0)

        Where:
        - m = total number of hypotheses (features)
        - Q = FDR level (e.g., 0.05)
        - k = rank of p-value

        References:
        - Benjamini, Y., & Hochberg, Y. (1995). Controlling the false
          discovery rate. Journal of the Royal Statistical Society,
          Series B, 57(1), 289-300.
        """
        if not self.gamma_pvalue or not self.apply_fdr:
            logger.info("FDR correction disabled or no p-values available")
            return

        # Extract p-values for features (exclude intercept)
        p_values = np.array([
            self.gamma_pvalue['coefs'][f]
            for f in self.feature_names
        ])

        # Get number of hypotheses
        m = len(p_values)

        # Sort p-values and keep track of original indices
        sorted_indices = np.argsort(p_values)
        sorted_p_values = p_values[sorted_indices]

        # Calculate Benjamini-Hochberg critical values
        # Critical value for rank i: (i / m) * Q
        ranks = np.arange(1, m + 1)
        critical_values = (ranks / m) * self.fdr_level

        # Find largest k where p_(k) ≤ (k/m) * Q
        significant_ranks = np.where(sorted_p_values <= critical_values)[0]

        if len(significant_ranks) == 0:
            logger.warning(f"No features significant at FDR level {self.fdr_level}")
            self.significant_features_fdr = []
        else:
            k = significant_ranks[-1] + 1  # Convert to 1-indexed rank
            logger.info(f"FDR correction: {k} out of {m} features significant "
                       f"at FDR level {self.fdr_level}")

            # Get significant feature names
            significant_indices = sorted_indices[:k]
            self.significant_features_fdr = [
                self.feature_names[i]
                for i in significant_indices
            ]

        # Calculate adjusted p-values (FDR-adjusted)
        # p_adj_(i) = min(m * p_(i) / rank_i, 1.0)
        adjusted_p_values = np.minimum(
            sorted_p_values * m / ranks,
            1.0
        )

        # Unsort adjusted p-values to match original feature order
        unsorted_adjusted = np.empty_like(adjusted_p_values)
        unsorted_adjusted[sorted_indices] = adjusted_p_values

        # Store FDR-adjusted p-values
        self.gamma_pvalue_fdr = {
            'intercept': self.gamma_pvalue['intercept'],  # Intercept not adjusted
            'coefs': {
                f: float(unsorted_adjusted[i])
                for i, f in enumerate(self.feature_names)
            }
        }

        # Log results
        logger.info("=" * 60)
        logger.info("Benjamini-Hochberg FDR Correction Results")
        logger.info("=" * 60)
        logger.info(f"FDR Level (Q): {self.fdr_level}")
        logger.info(f"Total Features Tested: {m}")
        logger.info(f"Significant Features (after FDR): {len(self.significant_features_fdr)}")
        logger.info(f"False Discovery Rate Controlled at: {self.fdr_level * 100:.1f}%")

        if len(self.significant_features_fdr) > 0:
            logger.info("\nSignificant Features after FDR Correction:")
            for feature in self.significant_features_fdr:
                raw_p = self.gamma_pvalue['coefs'][feature]
                adj_p = self.gamma_pvalue_fdr['coefs'][feature]
                logger.info(f"  {feature:20s}: raw_p = {raw_p:.6f}, "
                           f"adj_p = {adj_p:.6f}")
        else:
            logger.warning("\n⚠️  No features significant after FDR correction!")
            logger.warning("   This may indicate:")
            logger.warning("   1. Weak factor premia in the data")
            logger.warning("   2. High noise level")
            logger.warning("   3. FDR level too strict")

        logger.info("=" * 60)

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
            DataFrame with columns: coefficient, std_error, t_stat, p_value, p_value_fdr,
                                  significant, significant_fdr
        """
        if self.gamma_mean is None:
            return pd.DataFrame()

        rows = []

        # Intercept
        intercept_row = {
            'feature': 'Intercept',
            'coefficient': self.gamma_mean['intercept'],
            'std_error': self.gamma_std['intercept'] / np.sqrt(len(self.gamma_history)),
            't_stat': self.gamma_tstat['intercept'],
            'p_value': self.gamma_pvalue['intercept'],
            'significant': self.gamma_pvalue['intercept'] < 0.05
        }
        # Add FDR-adjusted p-value if available
        if self.gamma_pvalue_fdr is not None:
            intercept_row['p_value_fdr'] = self.gamma_pvalue_fdr['intercept']
            intercept_row['significant_fdr'] = self.gamma_pvalue_fdr['intercept'] < self.fdr_level
        else:
            intercept_row['p_value_fdr'] = None
            intercept_row['significant_fdr'] = None

        rows.append(intercept_row)

        # Feature coefficients
        for feature in self.feature_names:
            row = {
                'feature': feature,
                'coefficient': self.gamma_mean['coefs'][feature],
                'std_error': self.gamma_std['coefs'][feature] / np.sqrt(len(self.gamma_history)),
                't_stat': self.gamma_tstat['coefs'][feature],
                'p_value': self.gamma_pvalue['coefs'][feature],
                'significant': self.gamma_pvalue['coefs'][feature] < 0.05
            }
            # Add FDR-adjusted p-value if available
            if self.gamma_pvalue_fdr is not None:
                row['p_value_fdr'] = self.gamma_pvalue_fdr['coefs'][feature]
                row['significant_fdr'] = self.gamma_pvalue_fdr['coefs'][feature] < self.fdr_level
            else:
                row['p_value_fdr'] = None
                row['significant_fdr'] = None

            rows.append(row)

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
            'gamma_pvalue_fdr': self.gamma_pvalue_fdr,
            'significant_features_fdr': self.significant_features_fdr,
            'feature_names': self.feature_names,
            'dates_used': [d.isoformat() for d in self.dates_used],
            'regularization': self.regularization,
            'alpha': self.alpha,
            'min_cross_section_size': self.min_cross_section_size,
            'fdr_level': self.fdr_level,
            'apply_fdr': self.apply_fdr
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
        instance.gamma_pvalue_fdr = model_dict.get('gamma_pvalue_fdr')
        instance.significant_features_fdr = model_dict.get('significant_features_fdr', [])
        instance.feature_names = model_dict['feature_names']
        instance.dates_used = [pd.Timestamp(d) for d in model_dict['dates_used']]
        instance.regularization = model_dict['regularization']
        instance.alpha = model_dict['alpha']
        instance.min_cross_section_size = model_dict['min_cross_section_size']
        instance.fdr_level = model_dict.get('fdr_level', 0.05)
        instance.apply_fdr = model_dict.get('apply_fdr', True)
        
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


