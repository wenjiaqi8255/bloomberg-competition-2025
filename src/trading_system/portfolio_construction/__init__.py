"""
Portfolio Construction Module
============================

This module provides portfolio construction strategies with a focus on Box-First
methodology while maintaining backward compatibility with traditional quantitative
approaches.

Key Components:
- IPortfolioBuilder: Unified interface for portfolio construction
- BoxBasedPortfolioBuilder: Box-First methodology implementation
- QuantitativePortfolioBuilder: Traditional optimization wrapper
- PortfolioBuilderFactory: Factory for creating builders from config
"""

from src.trading_system.portfolio_construction.interface.interfaces import IPortfolioBuilder, PortfolioConstructionRequest
from src.trading_system.portfolio_construction.models.types import BoxKey, BoxConstructionResult
from .factory import PortfolioBuilderFactory

__all__ = [
    'IPortfolioBuilder',
    'PortfolioConstructionRequest',
    'BoxKey',
    'BoxConstructionResult',
    'PortfolioBuilderFactory'
]