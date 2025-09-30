"""
Feature Validation Module.

This module provides unified feature validation capabilities including
Information Coefficient analysis, statistical significance testing,
and economic significance evaluation.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from scipy import stats

from ..models.data_types import FeatureMetrics, FeatureData, ForwardReturns

logger = logging.getLogger(__name__)


class FeatureValidator:
    """
    Unified feature validator with IC analysis.

    This validator evaluates features using Information Coefficient (IC)
    methodology commonly used in quantitative finance research.
    """

    def __init__(self, min_ic_threshold: float = 0.03, min_significance: float = 0.05):
        """
        Initialize validator.

        Args:
            min_ic_threshold: Minimum absolute IC threshold for acceptance
            min_significance: Maximum p-value for statistical significance
        """
        self.min_ic_threshold = min_ic_threshold
        self.min_significance = min_significance

        logger.debug(f"Initialized FeatureValidator: IC_threshold={min_ic_threshold}, "
                    f"significance={min_significance}")

    def validate_features(self, features: FeatureData, forward_returns: ForwardReturns) -> Dict[str, FeatureMetrics]:
        """
        Validate all features using Information Coefficient analysis.

        Args:
            features: Feature DataFrame with symbol-prefixed columns
            forward_returns: Dictionary of forward returns by symbol

        Returns:
            Dictionary of feature metrics by feature name
        """
        logger.info(f"Validating {len(features.columns)} features")

        metrics = {}
        feature_columns = self._extract_feature_columns(features, forward_returns)

        for feature_name in feature_columns:
            try:
                feature_metrics = self._validate_single_feature(feature_name, features, forward_returns)
                metrics[feature_name] = feature_metrics

                if feature_metrics.recommendation == "ACCEPT":
                    logger.debug(f"✓ Accepted {feature_name}: IC={feature_metrics.information_coefficient:.3f}")
                else:
                    logger.debug(f"✗ Rejected {feature_name}: IC={feature_metrics.information_coefficient:.3f}")

            except Exception as e:
                logger.warning(f"Failed to validate feature {feature_name}: {e}")
                # Create default metrics for failed validation
                metrics[feature_name] = self._create_default_metrics(feature_name)

        accepted_count = sum(1 for m in metrics.values() if m.recommendation == "ACCEPT")
        logger.info(f"Validation completed: {accepted_count}/{len(metrics)} features accepted")

        return metrics

    def _extract_feature_columns(self, features: FeatureData, forward_returns: ForwardReturns) -> List[str]:
        """Extract valid feature columns that have corresponding forward returns."""
        valid_columns = []

        for column in features.columns:
            # Extract symbol from column name (format: "symbol_featurename")
            if '_' in column:
                symbol = column.split('_', 1)[0]
                if symbol in forward_returns:
                    valid_columns.append(column)

        return valid_columns

    def _validate_single_feature(self, feature_name: str, features: FeatureData,
                               forward_returns: ForwardReturns) -> FeatureMetrics:
        """Validate a single feature."""
        # Extract symbol from feature name
        symbol = feature_name.split('_', 1)[0]
        feature_values = features[feature_name]
        return_values = forward_returns[symbol]

        # Align dates and remove NaN
        aligned_feature, aligned_returns = self._align_series(feature_values, return_values)

        if len(aligned_feature) < 50:  # Minimum observations for validation
            return self._create_default_metrics(feature_name)

        # Calculate validation metrics
        return self._calculate_feature_metrics(feature_name, aligned_feature, aligned_returns)

    def _align_series(self, feature_series: pd.Series, return_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Align feature and return series by date and remove NaN values."""
        # Find common dates
        common_dates = feature_series.index.intersection(return_series.index)

        if len(common_dates) < 10:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        # Align and clean data
        aligned_feature = feature_series.loc[common_dates].dropna()
        aligned_returns = return_series.loc[common_dates].dropna()

        # Further align to remove any remaining NaN
        valid_dates = aligned_feature.index.intersection(aligned_returns.index)
        if len(valid_dates) < 10:
            return pd.Series(dtype=float), pd.Series(dtype=float)

        return aligned_feature.loc[valid_dates], aligned_returns.loc[valid_dates]

    def _calculate_feature_metrics(self, feature_name: str,
                                 feature_values: pd.Series, return_values: pd.Series) -> FeatureMetrics:
        """Calculate comprehensive feature validation metrics."""
        # Convert to numpy arrays
        feat_array = feature_values.values
        ret_array = return_values.values

        # Remove any remaining NaN values
        valid_mask = ~(np.isnan(feat_array) | np.isnan(ret_array))
        feat_clean = feat_array[valid_mask]
        ret_clean = ret_array[valid_mask]

        if len(feat_clean) < 50:
            return self._create_default_metrics(feature_name)

        # Information Coefficient (Pearson correlation)
        ic, ic_p_value = stats.pearsonr(feat_clean, ret_clean)
        if np.isnan(ic):
            ic = 0.0
            ic_p_value = 1.0

        # Calculate t-statistic for IC
        n = len(feat_clean)
        if abs(ic) < 1 and n > 2:
            ic_t_stat = ic * np.sqrt(n - 2) / np.sqrt(1 - ic**2)
        else:
            ic_t_stat = 0.0

        # Rank IC (Spearman correlation)
        try:
            rank_ic, _ = stats.spearmanr(feat_clean, ret_clean)
            if np.isnan(rank_ic):
                rank_ic = 0.0
        except:
            rank_ic = 0.0

        # Positive IC ratio (correlation with positive returns only)
        try:
            positive_returns = np.maximum(ret_clean, 0)
            positive_ic = np.corrcoef(feat_clean, positive_returns)[0, 1]
            if np.isnan(positive_ic):
                positive_ic = 0.0
        except:
            positive_ic = 0.0

        # Decile correlation
        decile_corr = self._calculate_decile_correlation(feat_clean, ret_clean)

        # Feature stability (rolling IC stability)
        feature_stability = self._calculate_feature_stability(feat_clean, ret_clean)

        # Economic significance (hedge portfolio return)
        economic_significance = self._calculate_economic_significance(feat_clean, ret_clean)

        # Statistical properties
        mean_val = np.mean(feat_clean)
        std_val = np.std(feat_clean)
        skewness = stats.skew(feat_clean)
        kurtosis = stats.kurtosis(feat_clean)

        # Validation flags
        is_significant = ic_p_value < self.min_significance and abs(ic) > self.min_ic_threshold
        is_economically_meaningful = abs(economic_significance) > 0.02  # 2% annualized threshold
        is_stable = feature_stability > 0.5  # 50% stability threshold

        # Recommendation
        if is_significant and is_economically_meaningful and is_stable:
            recommendation = "ACCEPT"
        elif abs(ic) > 0.02:
            recommendation = "MARGINAL"
        else:
            recommendation = "REJECT"

        return FeatureMetrics(
            feature_name=feature_name,
            information_coefficient=ic,
            ic_p_value=ic_p_value,
            positive_ic_ratio=positive_ic,
            feature_stability=feature_stability,
            economic_significance=economic_significance,
            mean_value=mean_val,
            std_value=std_val,
            skewness=skewness,
            kurtosis=kurtosis,
            is_significant=is_significant,
            is_economically_meaningful=is_economically_meaningful,
            recommendation=recommendation
        )

    def _calculate_decile_correlation(self, features: np.ndarray, returns: np.ndarray) -> float:
        """Calculate decile correlation between feature quantiles and returns."""
        try:
            # Create deciles based on feature values
            feature_series = pd.Series(features)
            return_series = pd.Series(returns)

            # Handle duplicate values in qcut
            deciles = pd.qcut(feature_series, 10, labels=False, duplicates='drop')

            # Calculate average return by decile
            decile_returns = pd.DataFrame({'decile': deciles, 'return': return_series}).groupby('decile')['return'].mean()

            # Correlation between decile rank and return
            if len(decile_returns) >= 5:
                return np.corrcoef(range(len(decile_returns)), decile_returns.values)[0, 1]
            else:
                return 0.0
        except Exception as e:
            logger.debug(f"Decile correlation calculation failed: {e}")
            return 0.0

    def _calculate_feature_stability(self, features: np.ndarray, returns: np.ndarray) -> float:
        """Calculate feature stability using rolling IC analysis."""
        try:
            n_obs = min(len(features), len(returns))
            if n_obs < 126:  # Need at least 6 months of daily data
                return 0.0

            # Calculate rolling IC using 63-day windows (quarterly)
            rolling_ics = []
            window_size = 63
            step_size = 21  # Move window forward by 3 weeks

            for i in range(window_size, n_obs, step_size):
                feat_window = features[i-window_size:i]
                ret_window = returns[i-window_size:i]

                if len(feat_window) > 10:
                    ic, _ = stats.pearsonr(feat_window, ret_window)
                    if not np.isnan(ic):
                        rolling_ics.append(ic)

            if len(rolling_ics) > 2:
                # Stability = 1 - coefficient of variation of IC
                ic_mean = np.mean(rolling_ics)
                ic_std = np.std(rolling_ics)
                cv = ic_std / abs(ic_mean) if ic_mean != 0 else float('inf')
                stability = max(0, 1 - cv)
                return stability
            else:
                return 0.0
        except Exception as e:
            logger.debug(f"Feature stability calculation failed: {e}")
            return 0.0

    def _calculate_economic_significance(self, features: np.ndarray, returns: np.ndarray) -> float:
        """Calculate economic significance using hedge portfolio approach."""
        try:
            feature_series = pd.Series(features)
            return_series = pd.Series(returns)

            # Remove NaN values
            valid_mask = ~(feature_series.isna() | return_series.isna())
            feat_clean = feature_series[valid_mask]
            ret_clean = return_series[valid_mask]

            if len(feat_clean) < 50:
                return 0.0

            # Create hedge portfolios: long top 30%, short bottom 30%
            top_30_threshold = feat_clean.quantile(0.7)
            bottom_30_threshold = feat_clean.quantile(0.3)

            long_returns = ret_clean[feat_clean >= top_30_threshold]
            short_returns = ret_clean[feat_clean <= bottom_30_threshold]

            if len(long_returns) > 0 and len(short_returns) > 0:
                long_mean = long_returns.mean()
                short_mean = short_returns.mean()
                hedge_return = long_mean - short_mean

                # Annualize assuming daily data
                annual_hedge_return = hedge_return * 252
                return annual_hedge_return
            else:
                return 0.0
        except Exception as e:
            logger.debug(f"Economic significance calculation failed: {e}")
            return 0.0

    def _create_default_metrics(self, feature_name: str) -> FeatureMetrics:
        """Create default metrics for failed validation."""
        return FeatureMetrics(
            feature_name=feature_name,
            information_coefficient=0.0,
            ic_p_value=1.0,
            positive_ic_ratio=0.0,
            feature_stability=0.0,
            economic_significance=0.0,
            mean_value=0.0,
            std_value=0.0,
            skewness=0.0,
            kurtosis=0.0,
            is_significant=False,
            is_economically_meaningful=False,
            recommendation="REJECT"
        )

    def get_validation_summary(self, metrics: Dict[str, FeatureMetrics]) -> pd.DataFrame:
        """Get summary DataFrame of validation results."""
        summary_data = []

        for feature_name, metric in metrics.items():
            summary_data.append({
                'feature_name': feature_name,
                'ic': metric.information_coefficient,
                'ic_p_value': metric.ic_p_value,
                'positive_ic_ratio': metric.positive_ic_ratio,
                'feature_stability': metric.feature_stability,
                'economic_significance': metric.economic_significance,
                'recommendation': metric.recommendation,
                'is_significant': metric.is_significant,
                'is_economically_meaningful': metric.is_economically_meaningful
            })

        return pd.DataFrame(summary_data).sort_values('ic', key=abs, ascending=False)

    def filter_accepted_features(self, features: FeatureData,
                               metrics: Dict[str, FeatureMetrics]) -> FeatureData:
        """Filter features to only include accepted ones."""
        accepted_columns = []

        for column in features.columns:
            if column in metrics:
                metric = metrics[column]
                if metric.recommendation in ["ACCEPT", "MARGINAL"]:
                    accepted_columns.append(column)

        return features[accepted_columns].copy()


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_features(features: FeatureData, forward_returns: ForwardReturns,
                     min_ic_threshold: float = 0.03, min_significance: float = 0.05) -> Dict[str, FeatureMetrics]:
    """
    Convenience function for feature validation.

    Args:
        features: Feature DataFrame
        forward_returns: Forward returns by symbol
        min_ic_threshold: Minimum IC threshold
        min_significance: Maximum p-value

    Returns:
        Dictionary of feature metrics
    """
    validator = FeatureValidator(min_ic_threshold, min_significance)
    return validator.validate_features(features, forward_returns)