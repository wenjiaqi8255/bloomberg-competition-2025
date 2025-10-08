"""
Interfaces for portfolio construction strategies.

This module defines the abstract interfaces that all portfolio construction
strategies must implement, ensuring a consistent contract while allowing
different implementation approaches.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import pandas as pd

from src.trading_system.portfolio_construction.models.types import PortfolioConstructionRequest, BoxConstructionResult, BoxKey


class IPortfolioBuilder(ABC):
    """
    Abstract interface for portfolio construction strategies.

    This is the only abstract interface in the portfolio construction
    system, following the principle of minimal abstraction.
    Different construction methods (quantitative, box-based, etc.)
    implement this interface with their own internal logic.

    The interface follows a simple contract:
    Input: Market data, signals, and constraints
    Output: Portfolio weights
    """

    @abstractmethod
    def build_portfolio(self, request: PortfolioConstructionRequest) -> pd.Series:
        """
        Build a portfolio from the given request.

        Args:
            request: Portfolio construction request containing all necessary data

        Returns:
            Series of portfolio weights indexed by stock symbols

        Raises:
            PortfolioConstructionError: If construction fails
        """
        pass

    @abstractmethod
    def get_method_name(self) -> str:
        """
        Get the name of the construction method.

        Returns:
            Human-readable method name for logging and reporting
        """
        pass

    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate the configuration for this builder.

        Args:
            config: Configuration dictionary

        Returns:
            True if configuration is valid, False otherwise

        Raises:
            InvalidConfigError: If configuration is invalid
        """
        pass

    def build_portfolio_with_result(self, request: PortfolioConstructionRequest) -> BoxConstructionResult:
        """
        Build portfolio with detailed construction information.

        Default implementation calls build_portfolio and creates a basic result.
        Subclasses can override for more detailed construction tracking.

        Args:
            request: Portfolio construction request

        Returns:
            BoxConstructionResult with weights and basic information
        """
        weights = self.build_portfolio(request)

        # Create basic result
        return BoxConstructionResult(
            weights=weights,
            box_coverage={},  # Empty for non-box methods
            selected_stocks={},
            target_weights={},
            construction_log=[f"Built portfolio using {self.get_method_name()}"]
        )

    def get_construction_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the construction method.

        Returns:
            Dictionary with method information
        """
        return {
            'method_name': self.get_method_name(),
            'method_type': self.__class__.__name__,
            'description': self.__class__.__doc__ or "No description available"
        }


class IBoxWeightProvider(ABC):
    """
    Interface for box weight allocation strategies.

    This interface allows for different approaches to determining
    target weights for each box (equal, config-based, benchmark-based, etc.).
    """

    @abstractmethod
    def get_box_weights(self) -> Dict[BoxKey, float]:
        """
        Get target weights for all boxes.

        Returns:
            Dictionary mapping BoxKey to target weight (should sum to 1.0)
        """
        pass

    @abstractmethod
    def validate_weights(self) -> bool:
        """
        Validate that weights are properly configured.

        Returns:
            True if weights are valid (sum to ~1.0, non-negative, etc.)
        """
        pass


class IWithinBoxAllocator(ABC):
    """
    Interface for allocating weights within a box.

    Different strategies can be used for allocating the box's total
    weight among the selected stocks (equal weight, signal-proportional, etc.).
    """

    @abstractmethod
    def allocate(self, stocks: List[str], total_weight: float,
                signals: pd.Series) -> Dict[str, float]:
        """
        Allocate weights within a box.

        Args:
            stocks: List of selected stocks in the box
            total_weight: Total weight allocated to this box
            signals: Signal strengths for all stocks

        Returns:
            Dictionary mapping stock symbols to weights within the box
        """
        pass


class IBoxSelector(ABC):
    """
    Interface for selecting stocks within a box.

    Different strategies can be used for selecting which stocks
    from a box to include in the portfolio.
    """

    @abstractmethod
    def select_stocks(self, candidate_stocks: List[str], signals: pd.Series,
                     n_stocks: int) -> List[str]:
        """
        Select stocks from candidates within a box.

        Args:
            candidate_stocks: All stocks available in the box
            signals: Signal strengths for all stocks
            n_stocks: Number of stocks to select

        Returns:
            List of selected stock symbols
        """
        pass