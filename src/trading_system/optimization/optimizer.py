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
        """
        self.method: OptimizationMethod = config.get('method', 'mean_variance')
        self.risk_aversion = config.get('risk_aversion', 2.0)
        self.top_n = config.get('top_n', 10)
        
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
        # Dispatch to appropriate method
        if self.method == 'mean_variance':
            return self._optimize_mean_variance(expected_returns, cov_matrix, constraints)
        elif self.method == 'equal_weight':
            return self._optimize_equal_weight(expected_returns, constraints)
        elif self.method == 'top_n':
            return self._optimize_top_n(expected_returns, constraints)
        else:
            # Fallback (should not reach here due to validation in __init__)
            logger.error(f"Unknown method '{self.method}', falling back to equal weight")
            return self._optimize_equal_weight(expected_returns, constraints)

    def _optimize_mean_variance(self,
                                expected_returns: pd.Series,
                                cov_matrix: pd.DataFrame,
                                constraints: List[Dict[str, Any]]) -> pd.Series:
        """
        Traditional mean-variance optimization (Markowitz).
        
        Maximizes: E[R] - (λ/2) * Variance
        where λ is the risk aversion parameter.
        """
        num_assets = len(expected_returns)
        initial_weights = np.ones(num_assets) / num_assets

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
            logger.info(f"Mean-variance optimization successful")
            return optimal_weights / optimal_weights.sum()
        else:
            logger.error(f"Mean-variance optimization failed: {result.message}")
            # Fallback to equal weight if optimization fails
            return pd.Series(initial_weights, index=expected_returns.index)
    
    def _optimize_equal_weight(self,
                               expected_returns: pd.Series,
                               constraints: List[Dict[str, Any]]) -> pd.Series:
        """
        Simple 1/N equal weighting strategy.
        
        This is a naive yet robust approach that:
        - Avoids estimation error in mean-variance optimization
        - Provides maximum diversification
        - Has been shown to outperform mean-variance in many practical settings
        
        Reference: DeMiguel et al. (2009) "Optimal Versus Naive Diversification"
        """
        num_assets = len(expected_returns)
        equal_weights = np.ones(num_assets) / num_assets
        weights = pd.Series(equal_weights, index=expected_returns.index)
        
        # Apply box constraints if any (e.g., sector limits)
        weights = self._apply_constraints_to_weights(weights, constraints)
        
        logger.info(f"Equal weight optimization: {num_assets} assets, weight={1/num_assets:.4f} each")
        return weights / weights.sum()
    
    def _optimize_top_n(self,
                       expected_returns: pd.Series,
                       constraints: List[Dict[str, Any]]) -> pd.Series:
        """
        Top-N equal weighting strategy.
        
        Selects the N assets with highest expected returns and allocates
        equal weight (1/N) to each. This combines:
        - Signal selectivity (only invest in best opportunities)
        - Simplicity (equal weight among selected assets)
        - Risk control (diversification across N assets)
        
        This approach is commonly used by practitioners who want to
        capture alpha signals while avoiding over-concentration.
        """
        # Select top N assets by expected return
        n = min(self.top_n, len(expected_returns))
        top_assets = expected_returns.nlargest(n).index
        
        # Allocate equal weight to top N assets
        weights = pd.Series(0.0, index=expected_returns.index)
        weights[top_assets] = 1.0 / n
        
        # Apply box constraints if any
        weights = self._apply_constraints_to_weights(weights, constraints)
        
        logger.info(
            f"Top-{n} optimization: Selected {n} assets from {len(expected_returns)}, "
            f"weight={1/n:.4f} each"
        )
        return weights / weights.sum()
    
    def _apply_constraints_to_weights(self,
                                     weights: pd.Series,
                                     constraints: List[Dict[str, Any]]) -> pd.Series:
        """
        Apply box constraints to weights by proportionally scaling down
        over-allocated sectors/boxes and redistributing to other assets.
        
        This is a simplified constraint enforcement for equal_weight and top_n methods.
        For strict constraint enforcement, use mean_variance method.
        """
        adjusted_weights = weights.copy()
        
        # Iterate through constraints and adjust if violated
        for const in constraints:
            if const['type'] == 'box':
                assets_in_box = const['assets']
                max_weight = const['max_weight']
                
                # Calculate current allocation to this box
                box_mask = adjusted_weights.index.isin(assets_in_box)
                box_allocation = adjusted_weights[box_mask].sum()
                
                # If over-allocated, scale down and redistribute
                if box_allocation > max_weight:
                    # Scale down box assets to meet constraint
                    scale_factor = max_weight / box_allocation
                    excess_weight = box_allocation - max_weight
                    
                    # Reduce weights in the constrained box
                    adjusted_weights[box_mask] *= scale_factor
                    
                    # Redistribute excess to assets outside the box
                    non_box_mask = ~box_mask
                    non_box_weights = adjusted_weights[non_box_mask]
                    
                    if non_box_weights.sum() > 0:
                        # Proportionally increase non-box assets
                        redistribution_factor = 1 + (excess_weight / non_box_weights.sum())
                        adjusted_weights[non_box_mask] *= redistribution_factor
                    
                    logger.info(
                        f"Adjusted box constraint '{const.get('description', 'N/A')}': "
                        f"{box_allocation:.2%} → {max_weight:.2%}, "
                        f"redistributed {excess_weight:.2%} to other assets"
                    )
        
        return adjusted_weights
            
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