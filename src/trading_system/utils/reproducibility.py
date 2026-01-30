"""
Reproducibility Utilities

This module provides utilities for ensuring reproducible results across
different runs of the trading system.

Key Features:
- Random seed management for numpy, python, and tensorflow/keras
- Deterministic operations where possible
- Logging of random seed usage for audit trails

Academic Standards:
- All random operations should use fixed seeds for reproducibility
- Seeds should be logged in experiment metadata
- Different seeds should be used for different experimental runs

References:
- The Pragmatic Programmer: "Reproducibility is key to scientific progress"
"""

import logging
import random
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """
    Manages random seeds and reproducibility settings.

    This class ensures that all stochastic operations in the trading system
    use fixed random seeds for reproducibility.

    Example:
        >>> manager = ReproducibilityManager(seed=42)
        >>> manager.set_all_seeds()
        >>> # Run deterministic operations
        >>> model.train()
        >>> # Get seed used for logging
        >>> print(f"Used seed: {manager.seed}")
    """

    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize reproducibility manager.

        Args:
        -----
        seed : Optional[int]
            Random seed to use. If None, operations will be truly random.
            Default is 42 for reproducibility.
        """
        self.seed = seed
        self._original_state = None

        logger.info(f"Initialized ReproducibilityManager with seed={seed}")

    def set_all_seeds(self) -> int:
        """
        Set random seeds for all relevant libraries.

        This ensures reproducibility across:
        - Python's random module
        - NumPy operations
        - TensorFlow/Keras (if available)
        - Any other stochastic components

        Returns:
        --------
        int
            The seed that was set
        """
        if self.seed is None:
            logger.warning("Random seed is None - operations will NOT be reproducible")
            return 0

        # Set Python random seed
        random.seed(self.seed)

        # Set NumPy seed
        np.random.seed(self.seed)

        # Set TensorFlow seed if available
        try:
            import tensorflow as tf
            tf.random.set_seed(self.seed)
            logger.debug("Set TensorFlow random seed")
        except ImportError:
            pass

        # Set PyTorch seed if available
        try:
            import torch
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)
            logger.debug("Set PyTorch random seed")
        except ImportError:
            pass

        logger.info(f"Set all random seeds to {self.seed}")
        return self.seed

    def save_random_state(self):
        """
        Save current random state for later restoration.

        Useful when you need to perform non-deterministic operations
        temporarily and then restore determinism.
        """
        self._original_state = {
            'python': random.getstate(),
            'numpy': np.random.get_state()
        }
        logger.debug("Saved random state")

    def restore_random_state(self):
        """Restore previously saved random state."""
        if self._original_state is None:
            logger.warning("No saved random state to restore")
            return

        random.setstate(self._original_state['python'])
        np.random.set_state(self._original_state['numpy'])
        logger.debug("Restored random state")

    def __enter__(self):
        """Context manager entry - set seeds."""
        self.save_random_state()
        self.set_all_seeds()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - restore state."""
        self.restore_random_state()


def set_global_seed(seed: int = 42) -> int:
    """
    Convenience function to set global random seed.

    Args:
    -----
    seed : int
        Random seed to use

    Returns:
    --------
    int
        The seed that was set

    Example:
        >>> from trading_system.utils.reproducibility import set_global_seed
        >>> set_global_seed(42)
        >>> np.random.rand()  # Always produces same result
    """
    manager = ReproducibilityManager(seed=seed)
    return manager.set_all_seeds()


def get_seed_from_config(config: dict) -> Optional[int]:
    """
    Extract random seed from configuration dictionary.

    Args:
    -----
    config : dict
        Configuration dictionary that may contain 'random_seed' key

    Returns:
    --------
    Optional[int]
        Random seed if found, None otherwise
    """
    # Check experiment section
    if 'experiment' in config and 'random_seed' in config['experiment']:
        return config['experiment']['random_seed']

    # Check top-level config
    if 'random_seed' in config:
        return config['random_seed']

    # Return default seed
    logger.info("No random_seed in config, using default seed (42)")
    return 42


def setup_reproducibility_from_config(config: dict) -> ReproducibilityManager:
    """
    Setup reproducibility manager from configuration.

    This is the main entry point for ensuring reproducibility in experiments.

    Args:
    -----
    config : dict
        Experiment configuration

    Returns:
    --------
    ReproducibilityManager
        Configured reproducibility manager

    Example:
        >>> config = load_config('experiment_config.yaml')
        >>> repro_manager = setup_reproducibility_from_config(config)
        >>> # Now all operations are reproducible
    """
    seed = get_seed_from_config(config)
    manager = ReproducibilityManager(seed=seed)
    manager.set_all_seeds()

    # Log seed for audit trail
    logger.info(f"Reproducibility setup complete with seed={seed}")
    logger.info(f"To reproduce this run, use random_seed={seed} in config")

    return manager


def log_reproducibility_info(metadata: dict, seed: Optional[int]) -> None:
    """
    Log reproducibility information to metadata dictionary.

    This ensures that the random seed is saved with model metadata
    for later reproduction.

    Args:
    -----
    metadata : dict
        Metadata dictionary to update
    seed : Optional[int]
        Random seed used
    """
    metadata['reproducibility'] = {
        'random_seed': seed,
        'numpy_seed_set': seed is not None,
        'reproducible': seed is not None
    }

    logger.debug(f"Logged reproducibility info to metadata: seed={seed}")
