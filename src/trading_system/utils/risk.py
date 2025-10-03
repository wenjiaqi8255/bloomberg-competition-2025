"""
Risk Metrics Utilities

Consolidated risk calculation methods extracted from multiple locations
to eliminate duplication and provide consistent risk metrics across the system.

Replaces:
- backtesting/risk_management.py (partial)
- strategy-specific _calculate_risk_metrics methods
- individual strategy risk calculations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from ..types import PortfolioSnapshot as Portfolio, Position

logger = logging.getLogger(__name__)


@dataclass
class RiskMetrics:
    """Container for comprehensive risk metrics."""
    var_95: float = 0.0
    var_99: float = 0.0
    expected_shortfall_95: float = 0.0
    expected_shortfall_99: float = 0.0
    beta_to_market: float = 1.0
    tracking_error: float = 0.0
    correlation_risk: float = 0.0
    concentration_risk: float = 0.0
    volatility: float = 0.0
    max_drawdown: float = 0.0
    downside_deviation: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    value_at_risk: float = 0.0
    conditional_var: float = 0.0
    overall_risk_score: float = 0.0


class RiskCalculator:
    """
    Unified risk metrics calculator.

    Provides static methods for calculating various risk metrics
    consistently across the entire trading system.
    """

    @staticmethod
    def portfolio_beta(portfolio_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate portfolio beta relative to market.

        Args:
            portfolio_returns: Portfolio return series
            market_returns: Market return series

        Returns:
            Portfolio beta
        """
        if len(portfolio_returns) < 2 or len(market_returns) < 2:
            return 1.0

        # Align returns
        common_index = portfolio_returns.index.intersection(market_returns.index)
        if len(common_index) < 2:
            return 1.0

        portfolio_aligned = portfolio_returns.loc[common_index]
        market_aligned = market_returns.loc[common_index]

        covariance = np.cov(portfolio_aligned, market_aligned)[0, 1]
        market_variance = np.var(market_aligned)

        return covariance / market_variance if market_variance > 0 else 1.0

    @staticmethod
    def tracking_error(portfolio_returns: pd.Series, benchmark_returns: pd.Series,
                      periods_per_year: int = 252) -> float:
        """
        Calculate tracking error relative to benchmark.

        Args:
            portfolio_returns: Portfolio return series
            benchmark_returns: Benchmark return series
            periods_per_year: Number of periods per year

        Returns:
            Annualized tracking error
        """
        if len(portfolio_returns) < 2 or len(benchmark_returns) < 2:
            return 0.0

        # Align returns
        common_index = portfolio_returns.index.intersection(benchmark_returns.index)
        if len(common_index) < 2:
            return 0.0

        portfolio_aligned = portfolio_returns.loc[common_index]
        benchmark_aligned = benchmark_returns.loc[common_index]

        active_returns = portfolio_aligned - benchmark_aligned
        return active_returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def concentration_risk(portfolio: Portfolio, method: str = 'herfindahl') -> float:
        """
        Calculate portfolio concentration risk.

        Args:
            portfolio: Portfolio snapshot
            method: Calculation method ('herfindahl', 'max_weight')

        Returns:
            Concentration risk metric
        """
        if not portfolio.positions:
            return 0.0

        weights = [abs(pos.weight) for pos in portfolio.positions if pos.weight != 0]

        if not weights:
            return 0.0

        if method == 'herfindahl':
            # Herfindahl-Hirschman Index
            return sum(w**2 for w in weights)
        elif method == 'max_weight':
            return max(weights)
        else:
            raise ValueError("Method must be 'herfindahl' or 'max_weight'")

    @staticmethod
    def correlation_risk(returns_matrix: pd.DataFrame) -> float:
        """
        Calculate average correlation risk in portfolio.

        Args:
            returns_matrix: DataFrame with returns for all holdings

        Returns:
            Average absolute correlation
        """
        if returns_matrix.empty or returns_matrix.shape[1] < 2:
            return 0.0

        correlation_matrix = returns_matrix.corr()

        # Get upper triangle without diagonal
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
        correlations = correlation_matrix.values[mask]

        # Return mean absolute correlation
        return np.mean(np.abs(correlations)) if len(correlations) > 0 else 0.0

    @staticmethod
    def downside_deviation(returns: pd.Series, target_return: float = 0.0,
                          periods_per_year: int = 252) -> float:
        """
        Calculate downside deviation.

        Args:
            returns: Return series
            target_return: Target/threshold return (default 0)
            periods_per_year: Number of periods per year

        Returns:
            Annualized downside deviation
        """
        if len(returns) < 2:
            return 0.0

        downside_returns = returns[returns < target_return] - target_return

        if len(downside_returns) == 0:
            return 0.0

        return downside_returns.std() * np.sqrt(periods_per_year)

    @staticmethod
    def expected_shortfall(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Expected Shortfall (Conditional Value at Risk).

        Args:
            returns: Return series
            confidence: Confidence level

        Returns:
            Expected shortfall at specified confidence level
        """
        if len(returns) < 2:
            return 0.0

        var_threshold = np.percentile(returns, (1 - confidence) * 100)
        tail_returns = returns[returns <= var_threshold]

        return tail_returns.mean() if len(tail_returns) > 0 else 0.0

    @staticmethod
    def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk (VaR).

        Args:
            returns: Return series
            confidence: Confidence level

        Returns:
            VaR at specified confidence level
        """
        if len(returns) < 2:
            return 0.0

        return np.percentile(returns, (1 - confidence) * 100)

    @staticmethod
    def skewness(returns: pd.Series) -> float:
        """
        Calculate return skewness.

        Args:
            returns: Return series

        Returns:
            Skewness of returns
        """
        if len(returns) < 3:
            return 0.0

        return returns.skew()

    @staticmethod
    def kurtosis(returns: pd.Series) -> float:
        """
        Calculate return kurtosis.

        Args:
            returns: Return series

        Returns:
            Kurtosis of returns
        """
        if len(returns) < 4:
            return 0.0

        return returns.kurtosis()

    @staticmethod
    def drawdown_risk(returns: pd.Series) -> Dict[str, float]:
        """
        Calculate drawdown-related risk metrics.

        Args:
            returns: Return series

        Returns:
            Dictionary with drawdown metrics
        """
        if len(returns) < 2:
            return {'max_drawdown': 0.0, 'avg_drawdown': 0.0, 'drawdown_duration': 0.0}

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # Maximum drawdown
        max_dd = drawdown.min()

        # Average drawdown (excluding zero values)
        negative_dd = drawdown[drawdown < 0]
        avg_dd = negative_dd.mean() if len(negative_dd) > 0 else 0.0

        # Maximum drawdown duration
        drawdown_periods = []
        in_drawdown = False
        current_duration = 0

        for dd in drawdown:
            if dd < 0:
                if not in_drawdown:
                    in_drawdown = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_drawdown:
                    drawdown_periods.append(current_duration)
                    in_drawdown = False
                    current_duration = 0

        # Handle case where series ends in drawdown
        if in_drawdown:
            drawdown_periods.append(current_duration)

        max_duration = max(drawdown_periods) if drawdown_periods else 0

        return {
            'max_drawdown': max_dd,
            'avg_drawdown': avg_dd,
            'max_duration': max_duration,
            'drawdown_frequency': len(negative_dd) / len(returns)
        }

    @staticmethod
    def liquidity_risk(volume_data: pd.DataFrame, position_sizes: Dict[str, float],
                       trading_days: int = 20) -> Dict[str, float]:
        """
        Calculate liquidity risk based on volume and position sizes.

        Args:
            volume_data: DataFrame with trading volumes
            position_sizes: Dictionary of position sizes
            trading_days: Number of days for average volume calculation

        Returns:
            Dictionary with liquidity risk metrics
        """
        if volume_data.empty:
            return {'avg_liquidity_ratio': 0.0, 'max_liquidity_ratio': 0.0, 'illiquid_positions': 0}

        liquidity_ratios = []
        illiquid_count = 0

        for symbol, position_size in position_sizes.items():
            if symbol in volume_data.columns:
                avg_volume = volume_data[symbol].tail(trading_days).mean()
                if avg_volume > 0:
                    # Daily position size as percentage of average volume
                    daily_exposure = position_size * 1000000  # Assuming position_size is in millions
                    liquidity_ratio = daily_exposure / avg_volume
                    liquidity_ratios.append(liquidity_ratio)

                    # Consider illiquid if position > 10% of average daily volume
                    if liquidity_ratio > 0.1:
                        illiquid_count += 1

        if not liquidity_ratios:
            return {'avg_liquidity_ratio': 0.0, 'max_liquidity_ratio': 0.0, 'illiquid_positions': 0}

        return {
            'avg_liquidity_ratio': np.mean(liquidity_ratios),
            'max_liquidity_ratio': np.max(liquidity_ratios),
            'illiquid_positions': illiquid_count,
            'liquidity_concentration': np.std(liquidity_ratios)
        }

    @staticmethod
    def calculate_comprehensive_risk(portfolio: Portfolio,
                                   returns_matrix: pd.DataFrame,
                                   benchmark_returns: Optional[pd.Series] = None) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a portfolio.

        Args:
            portfolio: Portfolio snapshot
            returns_matrix: Historical returns for holdings
            benchmark_returns: Optional benchmark returns

        Returns:
            RiskMetrics object with all calculated metrics
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = RiskCalculator._calculate_portfolio_returns(portfolio, returns_matrix)

            if portfolio_returns.empty:
                logger.warning("Unable to calculate portfolio returns")
                return RiskMetrics()

            # Basic risk metrics
            volatility = portfolio_returns.std() * np.sqrt(252)
            var_95 = RiskCalculator.value_at_risk(portfolio_returns, 0.95)
            var_99 = RiskCalculator.value_at_risk(portfolio_returns, 0.99)
            es_95 = RiskCalculator.expected_shortfall(portfolio_returns, 0.95)
            es_99 = RiskCalculator.expected_shortfall(portfolio_returns, 0.99)

            # Drawdown metrics
            drawdown_metrics = RiskCalculator.drawdown_risk(portfolio_returns)
            max_drawdown = drawdown_metrics['max_drawdown']

            # Statistical measures
            skewness = RiskCalculator.skewness(portfolio_returns)
            kurtosis = RiskCalculator.kurtosis(portfolio_returns)
            downside_deviation = RiskCalculator.downside_deviation(portfolio_returns)

            # Portfolio-specific risks
            concentration_risk = RiskCalculator.concentration_risk(portfolio)
            correlation_risk = RiskCalculator.correlation_risk(returns_matrix)

            # Market-related risks (if benchmark provided)
            beta_to_market = 1.0
            tracking_error = 0.0

            if benchmark_returns is not None:
                beta_to_market = RiskCalculator.portfolio_beta(portfolio_returns, benchmark_returns)
                tracking_error = RiskCalculator.tracking_error(portfolio_returns, benchmark_returns)

            # Overall risk score (0-1 scale)
            overall_risk_score = RiskCalculator._calculate_overall_risk_score(
                var_99, max_drawdown, concentration_risk, correlation_risk, volatility
            )

            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                expected_shortfall_95=es_95,
                expected_shortfall_99=es_99,
                beta_to_market=beta_to_market,
                tracking_error=tracking_error,
                correlation_risk=correlation_risk,
                concentration_risk=concentration_risk,
                volatility=volatility,
                max_drawdown=max_drawdown,
                downside_deviation=downside_deviation,
                skewness=skewness,
                kurtosis=kurtosis,
                value_at_risk=var_99,  # Alias for var_99
                conditional_var=es_99,  # Alias for expected_shortfall_99
                overall_risk_score=overall_risk_score
            )

        except Exception as e:
            logger.error(f"Error calculating comprehensive risk metrics: {e}")
            return RiskMetrics()

    @staticmethod
    def _calculate_portfolio_returns(portfolio: Portfolio, returns_matrix: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns from position weights and individual returns."""
        if not portfolio.positions or returns_matrix.empty:
            return pd.Series(dtype=float)

        # Get weights for positions that have return data
        weights = {}
        for pos in portfolio.positions:
            if pos.symbol in returns_matrix.columns and pos.weight != 0:
                weights[pos.symbol] = pos.weight

        if not weights:
            return pd.Series(dtype=float)

        # Align returns matrix with portfolio positions
        available_returns = returns_matrix[list(weights.keys())]
        weights_array = np.array(list(weights.values()))

        # Calculate weighted returns
        portfolio_returns = (available_returns * weights_array).sum(axis=1)

        return portfolio_returns.dropna()

    @staticmethod
    def _calculate_overall_risk_score(var_99: float, max_drawdown: float,
                                    concentration_risk: float, correlation_risk: float,
                                    volatility: float) -> float:
        """
        Calculate overall risk score (0-1 scale).

        Combines multiple risk measures into a single score.
        """
        # Normalize individual risk measures to 0-1 scale
        var_score = min(abs(var_99) / 0.05, 1.0)  # 5% daily VaR as maximum
        dd_score = min(abs(max_drawdown) / 0.20, 1.0)  # 20% drawdown as maximum
        conc_score = min(concentration_risk, 1.0)
        corr_score = min(correlation_risk, 1.0)
        vol_score = min(volatility / 0.30, 1.0)  # 30% annual volatility as maximum

        # Weighted average (adjust weights as needed)
        weights = [0.25, 0.25, 0.2, 0.15, 0.15]  # VaR, DD, Concentration, Correlation, Volatility
        risk_measures = [var_score, dd_score, conc_score, corr_score, vol_score]

        overall_score = sum(w * r for w, r in zip(weights, risk_measures))
        return min(overall_score, 1.0)

    @staticmethod
    def check_risk_limits(risk_metrics: RiskMetrics, limits: Dict[str, float]) -> Dict[str, bool]:
        """
        Check if portfolio violates any risk limits.

        Args:
            risk_metrics: Calculated risk metrics
            limits: Dictionary of risk limits

        Returns:
            Dictionary indicating which limits are breached
        """
        results = {}

        # Check each limit if it's specified
        if 'max_var_99' in limits:
            results['var_99_limit'] = abs(risk_metrics.var_99) <= limits['max_var_99']

        if 'max_drawdown' in limits:
            results['drawdown_limit'] = abs(risk_metrics.max_drawdown) <= limits['max_drawdown']

        if 'max_volatility' in limits:
            results['volatility_limit'] = risk_metrics.volatility <= limits['max_volatility']

        if 'max_concentration' in limits:
            results['concentration_limit'] = risk_metrics.concentration_risk <= limits['max_concentration']

        if 'max_correlation' in limits:
            results['correlation_limit'] = risk_metrics.correlation_risk <= limits['max_correlation']

        if 'max_tracking_error' in limits:
            results['tracking_error_limit'] = risk_metrics.tracking_error <= limits['max_tracking_error']

        if 'max_overall_risk' in limits:
            results['overall_risk_limit'] = risk_metrics.overall_risk_score <= limits['max_overall_risk']

        return results

    @staticmethod
    def generate_risk_summary(risk_metrics: RiskMetrics) -> Dict[str, str]:
        """
        Generate formatted risk summary for reporting.

        Args:
            risk_metrics: RiskMetrics object

        Returns:
            Dictionary with formatted risk metrics
        """
        return {
            'VaR 95%': f"{risk_metrics.var_95:.2%}",
            'VaR 99%': f"{risk_metrics.var_99:.2%}",
            'Expected Shortfall 95%': f"{risk_metrics.expected_shortfall_95:.2%}",
            'Expected Shortfall 99%': f"{risk_metrics.expected_shortfall_99:.2%}",
            'Volatility': f"{risk_metrics.volatility:.2%}",
            'Max Drawdown': f"{risk_metrics.max_drawdown:.2%}",
            'Downside Deviation': f"{risk_metrics.downside_deviation:.2%}",
            'Beta to Market': f"{risk_metrics.beta_to_market:.3f}",
            'Tracking Error': f"{risk_metrics.tracking_error:.2%}",
            'Concentration Risk': f"{risk_metrics.concentration_risk:.3f}",
            'Correlation Risk': f"{risk_metrics.correlation_risk:.3f}",
            'Skewness': f"{risk_metrics.skewness:.3f}",
            'Kurtosis': f"{risk_metrics.kurtosis:.3f}",
            'Overall Risk Score': f"{risk_metrics.overall_risk_score:.3f}"
        }

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from sklearn.covariance import LedoitWolf

class CovarianceEstimator(ABC):
    """
    Abstract base class for covariance matrix estimators.
    This provides a unified interface for different covariance estimation methods,
    allowing strategies to easily switch between them.
    """
    
    @abstractmethod
    def estimate(self, price_data: Dict[str, pd.DataFrame], date: datetime) -> pd.DataFrame:
        """
        Estimate the annualized covariance matrix.
        
        Args:
            price_data: Dictionary of historical price data for multiple symbols.
            date: The date as of which to perform the estimation.
            
        Returns:
            A pandas DataFrame representing the annualized N x N covariance matrix.
        """
        pass

    def _get_returns_matrix(self, price_data: Dict, date: datetime, lookback_days: int) -> pd.DataFrame:
        """Helper to build a returns matrix from price data."""
        returns_dict = {}
        end_date = date
        start_date = end_date - timedelta(days=lookback_days + 5) # buffer for non-trading days

        for symbol, data in price_data.items():
            # Filter data up to the estimation date and get the required lookback period
            recent_data = data[(data.index <= end_date) & (data.index >= start_date)].tail(lookback_days)
            if not recent_data.empty:
                returns_dict[symbol] = recent_data['Close'].pct_change().dropna()
        
        return pd.DataFrame(returns_dict)


class SimpleCovarianceEstimator(CovarianceEstimator):
    """
    Calculates the sample covariance matrix from historical returns.
    Serves as a baseline covariance estimation method.
    """
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
    
    def estimate(self, price_data: Dict, date: datetime) -> pd.DataFrame:
        returns_df = self._get_returns_matrix(price_data, date, self.lookback_days)
        
        if returns_df.empty or len(returns_df) < 2:
            return pd.DataFrame()
            
        # Sample covariance matrix, annualized
        cov_matrix = returns_df.cov() * 252
        return cov_matrix


class LedoitWolfCovarianceEstimator(CovarianceEstimator):
    """
    Estimates covariance using Ledoit-Wolf shrinkage.
    This method is robust to estimation errors, especially when the number of assets
    is large relative to the number of observations.
    """
    
    def __init__(self, lookback_days: int = 252):
        self.lookback_days = lookback_days
        
    def estimate(self, price_data: Dict, date: datetime) -> pd.DataFrame:
        returns_df = self._get_returns_matrix(price_data, date, self.lookback_days)
        
        if returns_df.empty or len(returns_df) < 2:
            return pd.DataFrame()
        
        # Apply Ledoit-Wolf shrinkage
        lw = LedoitWolf()
        # The fit method expects observations in rows, features in columns
        shrunk_cov_daily = lw.fit(returns_df.dropna()).covariance_
        
        # Annualize and return as a DataFrame
        shrunk_cov_annualized = shrunk_cov_daily * 252
        return pd.DataFrame(shrunk_cov_annualized, index=returns_df.columns, columns=returns_df.columns)