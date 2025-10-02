"""
Orchestration Module - Multi-Strategy System Coordination
=========================================================

This module provides the SystemOrchestrator and related components for
coordinating multiple trading strategies in a production environment.
"""

from .system_orchestrator import SystemOrchestrator, SystemResult

__all__ = [
    'SystemOrchestrator',
    'SystemResult',
]

