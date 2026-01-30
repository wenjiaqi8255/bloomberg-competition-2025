"""
Portfolio Optimizer
===================

This module provides the `PortfolioOptimizer` class, which is responsible for
constructing the optimal portfolio by solving a constrained optimization problem.

The primary goal is to maximize the Sharpe Ratio (or a similar utility function)
subject to various constraints, including:
- Sum of weights constraint (e.g., fully invested)
- Individual asset weight constraints (e.g., max 15% in any single stock)
- Box constraints (e.g., max 30% allocation to the 'Technology' sector)

This component is central to the new risk management framework, replacing the
heuristic-based `BoxAllocator` with a formal, mathematical approach.
"""

import logging
from typing import Dict, List, Any, Literal
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from .constraint_applier import ConstraintApplier
from trading_system.portfolio_construction.utils.weight_utils import WeightUtils

logger = logging.getLogger(__name__)

# Type alias for optimization methods
OptimizationMethod = Literal["mean_variance", "equal_weight", "top_n"]


class PortfolioOptimizer:
    """
    Solves for the optimal portfolio weights using various methods.
    
    Supports multiple allocation strategies:
    - mean_variance: Traditional mean-variance optimization (Markowitz)
    - equal_weight: Simple 1/N equal weighting across all assets
    - top_n: Equal weighting across top N assets by expected return
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PortfolioOptimizer.

        Args:
            config: Configuration dictionary with parameters:
                - method: Optimization method ('mean_variance', 'equal_weight', 'top_n')
                - risk_aversion: Risk aversion parameter for mean-variance (default: 2.0)
                - top_n: Number of top assets for 'top_n' method (default: 10)
                - enable_short_selling: Allow negative weights (short positions) (default: False)
        """
        self.method: OptimizationMethod = config.get('method', 'mean_variance')
        self.risk_aversion = config.get('risk_aversion', 2.0)
        self.top_n = config.get('top_n', 10)
        self.enable_short_selling = config.get('enable_short_selling', False)
        self.max_position_weight = config.get('max_position_weight', None)
        self.config = config

        # Validate method
        valid_methods = ['mean_variance', 'equal_weight', 'top_n']
        if self.method not in valid_methods:
            logger.warning(
                f"Invalid method '{self.method}'. Using 'mean_variance'. "
                f"Valid methods: {valid_methods}"
            )
            self.method = 'mean_variance'

        logger.info(f"PortfolioOptimizer initialized with method='{self.method}'")

    def optimize(self,
                 expected_returns: pd.Series,
                 cov_matrix: pd.DataFrame,
                 constraints: List[Dict[str, Any]]) -> pd.Series:
        """
        Find the optimal portfolio weights using the configured method.

        Args:
            expected_returns: A Series of expected returns for each asset.
            cov_matrix: The covariance matrix of asset returns.
            constraints: A list of constraint dictionaries.

        Returns:
            A Series containing the optimal weights for each asset.
        """
        # Align inputs to ensure consistent ordering
        common_symbols = expected_returns.index.intersection(cov_matrix.index)
        expected_returns = expected_returns.loc[common_symbols]
        cov_matrix = cov_matrix.loc[common_symbols, common_symbols]

        if expected_returns.empty:
            logger.warning("Cannot optimize with empty expected returns.")
            return pd.Series(dtype=float)

        # Dispatch to appropriate method
        if self.method == 'mean_variance':
            return self._optimize_mean_variance(expected_returns, cov_matrix, constraints)
        elif self.method == 'equal_weight':
            return self._optimize_equal_weight(expected_returns, cov_matrix, constraints)
        elif self.method == 'top_n':
            return self._optimize_top_n(expected_returns, cov_matrix, constraints)
        else:
            logger.error(f"Unknown method '{self.method}', falling back to mean-variance")
            return self._optimize_mean_variance(expected_returns, cov_matrix, constraints)

    def _optimize_mean_variance(self,
                                expected_returns: pd.Series,
                                cov_matrix: pd.DataFrame,
                                constraints: List[Dict[str, Any]]) -> pd.Series:
        """
        Performs classic mean-variance optimization (Markowitz).
        
        Maximizes: E[R] - (λ/2) * Variance
        """
        num_assets = len(expected_returns)
        initial_weights = np.ones(num_assets) / num_assets

        # Objective function: Minimize the negative utility
        def objective_function(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = weights.T @ cov_matrix @ weights
            return -(portfolio_return - 0.5 * self.risk_aversion * portfolio_variance)

        # Build constraints using the new applier
        applier = ConstraintApplier(self.config, num_assets, expected_returns.index)
        scipy_constraints = applier.get_linear_constraints(constraints)
        bounds = applier.get_bounds()

        # Perform optimization
        result = minimize(
            fun=objective_function,
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints,
            options={'disp': False, 'maxiter': 1000}
        )

        if result.success:
            optimal_weights = pd.Series(result.x, index=expected_returns.index)
            logger.debug("Mean-variance optimization successful.")
            # Normalize to handle potential floating point inaccuracies from solver
            return WeightUtils.normalize_weights(optimal_weights)
        else:
            logger.warning(f"Mean-variance optimization failed: {result.message}. Returning empty weights.")
            return pd.Series(dtype=float)
    
    def _optimize_equal_weight(self,
                               expected_returns: pd.Series,
                               cov_matrix: pd.DataFrame,
                               constraints: List[Dict[str, Any]]) -> pd.Series:
        """
        Finds the portfolio that is closest to equal-weight while satisfying all constraints.
        This is a quadratic programming problem that minimizes tracking error from a 1/N portfolio.
        """
        num_assets = len(expected_returns)
        equal_weights = np.ones(num_assets) / num_assets

        # Objective: Minimize (w - w_equal)^T * I * (w - w_equal), which simplifies to sum((w_i - 1/N)^2)
        # This finds the solution that is closest to the equal weight vector.
        def objective_function(weights):
            return np.sum((weights - equal_weights)**2)

        # Build constraints using the new applier
        applier = ConstraintApplier(self.config, num_assets, expected_returns.index)
        scipy_constraints = applier.get_linear_constraints(constraints)
        
        # Equal weight is inherently a long-only strategy
        bounds_config = self.config.copy()
        bounds_config['enable_short_selling'] = False
        long_only_applier = ConstraintApplier(bounds_config, num_assets, expected_returns.index)
        bounds = long_only_applier.get_bounds()

        # Perform optimization
        result = minimize(
            fun=objective_function,
            x0=equal_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints,
            options={'disp': False, 'maxiter': 1000}
        )

        if result.success:
            optimal_weights = pd.Series(result.x, index=expected_returns.index)
            logger.debug("Constrained equal-weight optimization successful.")
            return WeightUtils.normalize_weights(optimal_weights)
        else:
            logger.warning(f"Constrained equal-weight optimization failed: {result.message}. Returning empty weights.")
            return pd.Series(dtype=float)
    
    def _optimize_top_n(self,
                       expected_returns: pd.Series,
                       cov_matrix: pd.DataFrame,
                       constraints: List[Dict[str, Any]]) -> pd.Series:
        """
        Selects top N assets by signal and finds the portfolio closest to equal-weight
        among them, subject to constraints.
        """
        n = min(self.top_n, len(expected_returns))
        top_assets = expected_returns.nlargest(n).index
        num_assets = len(expected_returns)

        # Target is 1/n for top assets, 0 for others
        target_weights = pd.Series(0.0, index=expected_returns.index)
        target_weights.loc[top_assets] = 1.0 / n

        # Objective: Minimize deviation from this target vector
        def objective_function(weights):
            return np.sum((weights - target_weights.values)**2)

        # Build constraints using the new applier
        applier = ConstraintApplier(self.config, num_assets, expected_returns.index)
        scipy_constraints = applier.get_linear_constraints(constraints)

        # Top-N is inherently a long-only strategy
        bounds_config = self.config.copy()
        bounds_config['enable_short_selling'] = False
        long_only_applier = ConstraintApplier(bounds_config, num_assets, expected_returns.index)
        bounds = long_only_applier.get_bounds()

        # Perform optimization
        result = minimize(
            fun=objective_function,
            x0=target_weights.values,
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints,
            options={'disp': False, 'maxiter': 1000}
        )

        if result.success:
            optimal_weights = pd.Series(result.x, index=expected_returns.index)
            logger.debug(f"Constrained top-{n} optimization successful.")
            return WeightUtils.normalize_weights(optimal_weights)
        else:
            logger.warning(f"Constrained top-{n} optimization failed: {result.message}. Returning empty weights.")
            return pd.Series(dtype=float)
            
    @staticmethod
    def build_box_constraints(
        classifications: Dict[str, Dict[str, str]],
        box_limits: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Builds a list of box constraint dictionaries for the optimizer.
        
        Delegates to the ConstraintApplier.
        """
        return ConstraintApplier.build_box_constraints_from_config(classifications, box_limits)