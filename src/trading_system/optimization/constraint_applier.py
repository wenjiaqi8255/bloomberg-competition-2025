"""
Optimization Constraint Applier
===============================

Provides a dedicated service for applying and managing optimization constraints.
This class encapsulates the logic for translating a high-level configuration
into the specific formats required by numerical solvers (e.g., SciPy).
"""

import logging
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, Bounds

logger = logging.getLogger(__name__)

class ConstraintApplier:
    """Applies constraints to the portfolio optimization problem."""

    def __init__(self, config: Dict[str, Any], num_assets: int, asset_names: pd.Index):
        """
        Initialize the ConstraintApplier.

        Args:
            config: Optimizer configuration.
            num_assets: The number of assets in the portfolio.
            asset_names: The index of asset names.
        """
        self.config = config
        self.num_assets = num_assets
        self.asset_names = asset_names

    def get_bounds(self) -> Bounds:
        """
        Builds weight bounds for individual assets.
        """
        enable_short_selling = self.config.get('enable_short_selling', False)
        max_position_weight = self.config.get('max_position_weight')

        if enable_short_selling:
            upper = max_position_weight if max_position_weight is not None else 1.0
            lower = -upper
        else:
            upper = max_position_weight if max_position_weight is not None else 1.0
            lower = 0.0
        
        logger.debug(f"Applying bounds: [{lower}, {upper}]")
        return Bounds(lower, upper)

    def get_linear_constraints(self, custom_constraints: List[Dict[str, Any]]) -> List[LinearConstraint]:
        """
        Builds linear constraints, including the full investment constraint and custom ones.
        """
        # 1. Full investment constraint: sum(weights) = 1
        full_investment_constraint = LinearConstraint(np.ones(self.num_assets), lb=1.0, ub=1.0)
        scipy_constraints = [full_investment_constraint]
        
        # 2. Add custom constraints (e.g., box constraints)
        for const in custom_constraints:
            if const.get('type') == 'box':
                mask = self.asset_names.isin(const.get('assets', [])).astype(int)
                box_constraint = LinearConstraint(
                    mask,
                    lb=const.get('min_weight', 0),
                    ub=const.get('max_weight', 1)
                )
                scipy_constraints.append(box_constraint)
        
        logger.debug(f"Applying {len(scipy_constraints)} linear constraints.")
        return scipy_constraints

    @staticmethod
    def build_box_constraints_from_config(
        classifications: Dict[str, Dict[str, str]],
        box_limits: Dict[str, Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """
        Builds a list of box constraint dictionaries for the optimizer.
        """
        constraints = []
        all_assets = list(classifications.keys())

        for dimension, limits in box_limits.items():
            for box_value, max_weight in limits.items():
                assets_in_box = [
                    asset for asset in all_assets
                    if classifications.get(asset, {}).get(dimension) == box_value
                ]
                
                if assets_in_box:
                    constraints.append({
                        'type': 'box',
                        'assets': assets_in_box,
                        'max_weight': max_weight,
                        'description': f"{dimension}:{box_value} <= {max_weight:.1%}"
                    })
        
        logger.info(f"Built {len(constraints)} box constraints from configuration.")
        return constraints
