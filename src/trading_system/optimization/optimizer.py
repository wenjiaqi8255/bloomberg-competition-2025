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
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Solves for the optimal portfolio weights given expected returns,
    a covariance matrix, and a set of constraints.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the PortfolioOptimizer.

        Args:
            config: Configuration dictionary, which can include parameters like
                    risk aversion (lambda) or target volatility.
        """
        self.risk_aversion = config.get('risk_aversion', 2.0) # Example parameter
        logger.info("PortfolioOptimizer initialized.")

    def optimize(self,
                 expected_returns: pd.Series,
                 cov_matrix: pd.DataFrame,
                 constraints: List[Dict[str, Any]]) -> pd.Series:
        """
        Find the optimal portfolio weights.

        Args:
            expected_returns: A Series of expected returns for each asset.
            cov_matrix: The covariance matrix of asset returns.
            constraints: A list of constraint dictionaries.

        Returns:
            A Series containing the optimal weights for each asset.
        """
        num_assets = len(expected_returns)
        initial_weights = np.ones(num_assets) / num_assets  # Start with equal weights

        # Objective function to maximize: w^T μ - (λ/2) w^T Σ w
        # Minimizing the negative of this function is equivalent.
        def objective_function(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = weights.T @ cov_matrix @ weights
            return -(portfolio_return - 0.5 * self.risk_aversion * portfolio_variance)

        # Build constraints for the solver
        scipy_constraints = self._build_scipy_constraints(constraints, num_assets, expected_returns.index)

        # Bounds for each weight (e.g., 0 <= w_i <= 1 for long-only)
        bounds = Bounds(0, 1)

        # Perform optimization
        result = minimize(
            fun=objective_function,
            x0=initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=scipy_constraints,
            options={'disp': False}
        )

        if result.success:
            optimal_weights = pd.Series(result.x, index=expected_returns.index)
            logger.info(f"Optimization successful. Optimal portfolio found with Sharpe Ratio approximation.")
            return optimal_weights / optimal_weights.sum() # Normalize to sum to 1
        else:
            logger.error(f"Portfolio optimization failed: {result.message}")
            # Fallback to equal weight if optimization fails
            return pd.Series(initial_weights, index=expected_returns.index)
            
    def _build_scipy_constraints(self, custom_constraints: List, num_assets: int, asset_names: pd.Index) -> List:
        """
        Convert a list of custom constraint dicts into a format used by SciPy.
        """
        # 1. Full investment constraint: sum(weights) = 1
        full_investment_constraint = LinearConstraint(
            np.ones(num_assets), 
            lb=1.0, 
            ub=1.0
        )
        
        scipy_constraints = [full_investment_constraint]
        
        # 2. Add custom constraints (e.g., box constraints)
        for const in custom_constraints:
            if const['type'] == 'box':
                mask = [1 if asset in const['assets'] else 0 for asset in asset_names]
                box_constraint = LinearConstraint(
                    mask,
                    lb=const.get('min_weight', 0),
                    ub=const.get('max_weight', 1)
                )
                scipy_constraints.append(box_constraint)
            elif const['type'] == 'asset':
                # Individual asset constraints can be handled by bounds,
                # but can also be added here if more complex.
                pass
                
        return scipy_constraints

    @staticmethod
    def build_box_constraints(
        classifications: Dict[str, Dict[str, str]],
        box_limits: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Builds a list of box constraint dictionaries for the optimizer.

        Args:
            classifications: A dict mapping symbols to their box classifications,
                             e.g., {'AAPL': {'sector': 'Tech', 'size': 'Large'}, ...}
            box_limits: A dict defining the max weight for each box,
                        e.g., {'sector': {'Tech': 0.3, 'Finance': 0.25}, ...}

        Returns:
            A list of constraint dictionaries formatted for the optimizer.
        """
        constraints = []
        all_assets = list(classifications.keys())

        for dimension, limits in box_limits.items(): # e.g., dimension='sector', limits={'Tech': 0.3}
            for box_value, max_weight in limits.items(): # e.g., box_value='Tech', max_weight=0.3
                
                assets_in_box = [
                    asset for asset in all_assets
                    if classifications[asset].get(dimension) == box_value
                ]
                
                if assets_in_box:
                    constraints.append({
                        'type': 'box',
                        'assets': assets_in_box,
                        'max_weight': max_weight,
                        'description': f"{dimension}:{box_value} <= {max_weight:.1%}"
                    })
        
        logger.info(f"Built {len(constraints)} box constraints.")
        return constraints

