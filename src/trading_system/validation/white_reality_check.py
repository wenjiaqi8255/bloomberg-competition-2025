"""
White's Reality Check for Data Snooping Bias

This module implements White's Reality Check (2000) to test whether a strategy's
performance is statistically significant after accounting for data snooping bias.

Academic Background:
- Data snooping occurs when multiple strategies are tested and the best one is
  selected without proper adjustment
- This leads to overestimation of true performance (selection bias)
- White's Reality Check tests if the best strategy outperforms a benchmark

Methodology:
1. Generate N random strategies via bootstrap resampling
2. Calculate performance metric (e.g., Sharpe ratio) for each
3. Compute empirical distribution of performance under null hypothesis
4. Calculate p-value: P(performance_random ≥ performance_actual)
5. Reject null if p-value < significance level

Null Hypothesis (H0):
- The best strategy's performance is due to luck (data snooping)
- Any strategy could have performed as well by chance

Alternative Hypothesis (H1):
- The best strategy's performance is statistically significant
- The strategy has true predictive power

References:
- White, H. (2000). A reality check for data snooping. *Econometrica*, 68(5),
  1097-1126.
- Hansen, P. R. (2005). A test for superior predictive ability. *Journal of
  Econometrics*, 124(1), 29-51.
"""

import logging
from typing import Dict, List, Optional, Callable, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RealityCheckResult:
    """
    Results from White's Reality Check test.

    Attributes:
    ----------
    actual_performance : float
        Actual strategy performance (e.g., Sharpe ratio)
    null_mean : float
        Mean performance under null hypothesis (random strategies)
    null_std : float
        Standard deviation under null hypothesis
    p_value : float
        Probability of observing actual_performance under H0
    is_significant : bool
        Whether result is statistically significant
    n_bootstrap : int
        Number of bootstrap iterations performed
    test_statistic : float
        Test statistic (typically actual_performance)
    critical_value : float
        Critical value at significance level
    percentile : float
        Percentile of actual_performance in null distribution
    """
    actual_performance: float
    null_mean: float
    null_std: float
    p_value: float
    is_significant: bool
    n_bootstrap: int
    test_statistic: float
    critical_value: float
    percentile: float


class WhiteRealityCheck:
    """
    White's Reality Check for data snooping bias.

    This class implements White's (2000) Reality Check to determine whether
    a trading strategy's performance is statistically significant after
    accounting for multiple testing/data snooping.

    Example:
        >>> # Prepare strategy returns
        >>> strategy_returns = pd.Series([...])  # Daily returns
        >>>
        >>> # Run reality check
        >>> wrc = WhiteRealityCheck(n_bootstrap=1000, significance_level=0.05)
        >>> result = wrc.test(strategy_returns, metric='sharpe')
        >>>
        >>> print(f"P-value: {result.p_value:.4f}")
        >>> print(f"Significant: {result.is_significant}")
        >>>
        >>> # Visualize results
        >>> wrc.plot_null_distribution()
    """

    def __init__(self,
                 n_bootstrap: int = 1000,
                 significance_level: float = 0.05,
                 random_seed: Optional[int] = 42):
        """
        Initialize White's Reality Check.

        Args:
        -----
        n_bootstrap : int
            Number of bootstrap iterations for null distribution (default: 1000)
            Higher values = more accurate p-values but slower computation
        significance_level : float
            Significance level for hypothesis testing (default: 0.05)
        random_seed : Optional[int]
            Random seed for reproducibility (default: 42)
        """
        self.n_bootstrap = n_bootstrap
        self.significance_level = significance_level
        self.random_seed = random_seed

        self.null_distribution: Optional[np.ndarray] = None
        self.actual_returns: Optional[pd.Series] = None

        logger.info(f"Initialized WhiteRealityCheck: n_bootstrap={n_bootstrap}, "
                   f"significance_level={significance_level}")

    def test(self,
             strategy_returns: pd.Series,
             benchmark_returns: Optional[pd.Series] = None,
             metric: str = 'sharpe',
             metric_params: Optional[Dict] = None) -> RealityCheckResult:
        """
        Perform White's Reality Check test.

        Args:
        -----
        strategy_returns : pd.Series
            Strategy returns (daily or other frequency)
        benchmark_returns : Optional[pd.Series]
            Benchmark returns for comparison (e.g., S&P 500)
        metric : str
            Performance metric to test:
            - 'sharpe': Sharpe ratio (default)
            - 'sortino': Sortino ratio
            - 'calmar': Calmar ratio
            - 'total_return': Total return
            - 'alpha': Alpha (requires benchmark)
            - 'information_ratio': Information ratio (requires benchmark)
        metric_params : Optional[Dict]
            Additional parameters for metric calculation

        Returns:
        --------
        RealityCheckResult
            Test results with p-value and significance determination
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        self.actual_returns = strategy_returns

        # Calculate actual performance metric
        actual_performance = self._calculate_metric(
            strategy_returns,
            benchmark_returns,
            metric,
            metric_params
        )

        logger.info(f"Actual strategy {metric}: {actual_performance:.4f}")

        # Generate null distribution via bootstrap
        logger.info(f"Generating null distribution with {self.n_bootstrap} "
                   f"bootstrap iterations...")
        null_distribution = self._generate_null_distribution(
            strategy_returns,
            benchmark_returns,
            metric,
            metric_params
        )

        self.null_distribution = null_distribution

        # Calculate test statistics
        null_mean = np.mean(null_distribution)
        null_std = np.std(null_distribution, ddof=1)

        # Calculate p-value
        # P(performance_random ≥ performance_actual)
        p_value = np.mean(null_distribution >= actual_performance)

        # Determine significance
        is_significant = p_value < self.significance_level

        # Calculate critical value (1 - significance_level percentile)
        critical_value = np.percentile(
            null_distribution,
            (1 - self.significance_level) * 100
        )

        # Calculate percentile of actual performance in null distribution
        percentile = (np.sum(null_distribution <= actual_performance) /
                      len(null_distribution) * 100)

        # Create result
        result = RealityCheckResult(
            actual_performance=actual_performance,
            null_mean=null_mean,
            null_std=null_std,
            p_value=p_value,
            is_significant=is_significant,
            n_bootstrap=self.n_bootstrap,
            test_statistic=actual_performance,
            critical_value=critical_value,
            percentile=percentile
        )

        # Log results
        self._log_results(result, metric)

        return result

    def _generate_null_distribution(self,
                                    strategy_returns: pd.Series,
                                    benchmark_returns: Optional[pd.Series],
                                    metric: str,
                                    metric_params: Optional[Dict]) -> np.ndarray:
        """
        Generate null distribution via bootstrap resampling.

        Under the null hypothesis, the strategy has no predictive power, so
        we can randomly shuffle returns to simulate random strategies.

        Returns:
        --------
        np.ndarray
            Array of performance metrics from bootstrap iterations
        """
        null_performance = []

        returns_array = strategy_returns.values
        n_obs = len(returns_array)

        for i in range(self.n_bootstrap):
            # Bootstrap: randomly resample returns with replacement
            # This preserves the return distribution but breaks any patterns
            if i % 100 == 0:
                logger.debug(f"Bootstrap iteration {i}/{self.n_bootstrap}")

            # Method 1: Random permutation (breaks time-series dependence)
            shuffled_indices = np.random.permutation(n_obs)
            shuffled_returns = pd.Series(
                returns_array[shuffled_indices],
                index=strategy_returns.index
            )

            # Calculate performance metric for shuffled returns
            perf = self._calculate_metric(
                shuffled_returns,
                benchmark_returns,
                metric,
                metric_params
            )

            null_performance.append(perf)

        return np.array(null_performance)

    def _calculate_metric(self,
                         returns: pd.Series,
                         benchmark_returns: Optional[pd.Series],
                         metric: str,
                         metric_params: Optional[Dict]) -> float:
        """
        Calculate performance metric.

        Args:
        -----
        returns : pd.Series
            Strategy returns
        benchmark_returns : Optional[pd.Series]
            Benchmark returns (required for some metrics)
        metric : str
            Metric to calculate
        metric_params : Optional[Dict]
            Additional parameters

        Returns:
        --------
        float
            Calculated metric value
        """
        if metric_params is None:
            metric_params = {}

        if metric == 'sharpe':
            # Calculate Sharpe ratio
            risk_free_rate = metric_params.get('risk_free_rate', 0.02)
            periods_per_year = metric_params.get('periods_per_year', 252)

            excess_returns = returns - risk_free_rate / periods_per_year
            sharpe = (excess_returns.mean() * periods_per_year /
                     (excess_returns.std() * np.sqrt(periods_per_year)))
            return sharpe

        elif metric == 'sortino':
            # Calculate Sortino ratio
            risk_free_rate = metric_params.get('risk_free_rate', 0.02)
            periods_per_year = metric_params.get('periods_per_year', 252)

            excess_returns = returns - risk_free_rate / periods_per_year
            downside_returns = excess_returns[excess_returns < 0]
            downside_std = downside_returns.std() * np.sqrt(periods_per_year)

            if downside_std == 0:
                return np.inf

            sortino = (excess_returns.mean() * periods_per_year) / downside_std
            return sortino

        elif metric == 'total_return':
            # Total cumulative return
            return (1 + returns).prod() - 1

        elif metric == 'alpha':
            # Calculate Jensen's alpha (requires benchmark)
            if benchmark_returns is None:
                raise ValueError("Benchmark returns required for alpha calculation")

            # Simple alpha calculation (for more precision, use regression)
            strategy_return = (1 + returns).prod() - 1
            benchmark_return = (1 + benchmark_returns).prod() - 1
            risk_free_rate = metric_params.get('risk_free_rate', 0.02)

            # Alpha = strategy_return - [rf + beta * (benchmark_return - rf)]
            # Simplified: assume beta = 1
            alpha = strategy_return - (risk_free_rate + (benchmark_return - risk_free_rate))
            return alpha

        elif metric == 'information_ratio':
            # Information ratio (requires benchmark)
            if benchmark_returns is None:
                raise ValueError("Benchmark returns required for information ratio")

            excess_returns = returns - benchmark_returns
            tracking_error = excess_returns.std() * np.sqrt(252)

            if tracking_error == 0:
                return np.inf

            ir = (excess_returns.mean() * 252) / tracking_error
            return ir

        else:
            raise ValueError(f"Unknown metric: {metric}")

    def _log_results(self, result: RealityCheckResult, metric: str) -> None:
        """Log test results."""
        logger.info("=" * 70)
        logger.info("White's Reality Check Results")
        logger.info("=" * 70)
        logger.info(f"Performance Metric: {metric}")
        logger.info(f"Significance Level: {self.significance_level}")
        logger.info(f"Bootstrap Iterations: {result.n_bootstrap}")
        logger.info("")
        logger.info(f"Actual Strategy {metric}: {result.actual_performance:.4f}")
        logger.info(f"Null Distribution Mean: {result.null_mean:.4f}")
        logger.info(f"Null Distribution Std: {result.null_std:.4f}")
        logger.info(f"Critical Value ({(1-self.significance_level)*100:.0f}th percentile): "
                   f"{result.critical_value:.4f}")
        logger.info(f"")
        logger.info(f"P-value: {result.p_value:.4f}")
        logger.info(f"Percentile in Null Distribution: {result.percentile:.1f}th")
        logger.info("")

        if result.is_significant:
            logger.info("✅ RESULT: STRATEGY PERFORMANCE IS STATISTICALLY SIGNIFICANT")
            logger.info(f"   P-value ({result.p_value:.4f}) < significance level "
                       f"({self.significance_level})")
            logger.info("   Reject null hypothesis - strategy has true predictive power")
        else:
            logger.warning("⚠️  RESULT: STRATEGY PERFORMANCE IS NOT SIGNIFICANT")
            logger.warning(f"   P-value ({result.p_value:.4f}) ≥ significance level "
                        f"({self.significance_level})")
            logger.warning("   Cannot reject null hypothesis - performance may be due to chance")

        logger.info("")
        logger.info("Interpretation:")
        logger.info("- P-value < 0.05: Strong evidence against data snooping")
        logger.info("- P-value ≥ 0.05: Performance may be due to luck/data mining")
        logger.info("=" * 70)

    def get_summary_dataframe(self) -> pd.DataFrame:
        """
        Get summary of test results as a DataFrame.

        Returns:
        --------
        pd.DataFrame
            Summary statistics
        """
        if self.null_distribution is None:
            raise ValueError("No test results available. Run test() first.")

        return pd.DataFrame({
            'statistic': [
                'actual_performance',
                'null_mean',
                'null_std',
                'null_min',
                'null_max',
                'null_median',
                'p_value',
                'critical_value',
                'percentile'
            ],
            'value': [
                self.actual_returns.mean() if self.actual_returns is not None else np.nan,
                np.mean(self.null_distribution),
                np.std(self.null_distribution, ddof=1),
                np.min(self.null_distribution),
                np.max(self.null_distribution),
                np.median(self.null_distribution),
                np.mean(self.null_distribution >= (self.actual_returns.mean()
                  if self.actual_returns is not None else 0)),
                np.percentile(self.null_distribution, (1 - self.significance_level) * 100),
                (np.sum(self.null_distribution <= (self.actual_returns.mean()
                  if self.actual_returns is not None else 0)) /
                 len(self.null_distribution) * 100)
            ]
        })


def perform_reality_check(strategy_returns: pd.Series,
                         benchmark_returns: Optional[pd.Series] = None,
                         metric: str = 'sharpe',
                         n_bootstrap: int = 1000,
                         significance_level: float = 0.05,
                         random_seed: int = 42) -> RealityCheckResult:
    """
    Convenience function to perform White's Reality Check.

    Args:
    -----
    strategy_returns : pd.Series
        Strategy returns
    benchmark_returns : Optional[pd.Series]
        Benchmark returns
    metric : str
        Performance metric to test
    n_bootstrap : int
        Number of bootstrap iterations
    significance_level : float
        Significance level
    random_seed : int
        Random seed

    Returns:
    --------
    RealityCheckResult
        Test results

    Example:
        >>> returns = pd.Series([0.01, -0.005, 0.02, ...])
        >>> result = perform_reality_check(returns, n_bootstrap=5000)
        >>> print(f"P-value: {result.p_value:.4f}")
    """
    wrc = WhiteRealityCheck(
        n_bootstrap=n_bootstrap,
        significance_level=significance_level,
        random_seed=random_seed
    )

    return wrc.test(strategy_returns, benchmark_returns, metric)
