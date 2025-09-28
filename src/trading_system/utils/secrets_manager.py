"""
Secrets management for trading system with environment variable support.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ApiConfig:
    """Configuration class for API credentials."""
    wandb_api_key: Optional[str] = None
    alpha_vantage_api_key: Optional[str] = None
    # Add other secrets as needed
    other_secrets: Optional[Dict[str, str]] = None


class SecretsManager:
    """
    Centralized secrets management with multiple source support.

    Features:
    - Environment variable loading
    - .env file support
    - Secure credential handling
    - Configuration validation
    """

    def __init__(self, env_file: str = ".env"):
        """
        Initialize secrets manager.

        Args:
            env_file: Path to .env file (relative to project root)
        """
        self.env_file = Path(env_file)
        self._config = ApiConfig()
        self._load_secrets()

    def _load_secrets(self):
        """Load secrets from all available sources."""
        try:
            # Load from .env file if it exists
            if self.env_file.exists():
                self._load_from_env_file()

            # Load from environment variables (these override .env)
            self._load_from_environment()

            # Validate required secrets
            self._validate_secrets()

            logger.info("Secrets loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load secrets: {e}")
            raise

    def _load_from_env_file(self):
        """Load secrets from .env file."""
        try:
            with open(self.env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key] = value
                        logger.debug(f"Loaded {key} from .env file")

        except Exception as e:
            logger.warning(f"Failed to load .env file: {e}")

    def _load_from_environment(self):
        """Load secrets from environment variables."""
        # WandB API key
        self._config.wandb_api_key = os.getenv('WANDB_API_KEY')

        # Alpha Vantage API key
        self._config.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY')

        logger.debug("Environment variables loaded")

    def _validate_secrets(self):
        """Validate that required secrets are available."""
        missing_secrets = []

        if not self._config.wandb_api_key:
            missing_secrets.append('WANDB_API_KEY')

        if not self._config.alpha_vantage_api_key:
            missing_secrets.append('ALPHA_VANTAGE_API_KEY')

        if missing_secrets:
            logger.warning(f"Missing secrets: {', '.join(missing_secrets)}")

    def get_wandb_api_key(self) -> Optional[str]:
        """Get WandB API key."""
        return self._config.wandb_api_key

    def get_alpha_vantage_api_key(self) -> Optional[str]:
        """Get Alpha Vantage API key."""
        return self._config.alpha_vantage_api_key

    def get_config(self) -> ApiConfig:
        """Get complete configuration object."""
        return self._config

    def setup_wandb_environment(self):
        """Setup WandB environment variables."""
        if self._config.wandb_api_key:
            os.environ['WANDB_API_KEY'] = self._config.wandb_api_key
            logger.info("WandB API key configured in environment")
            return True
        return False

    def __str__(self) -> str:
        """String representation for logging."""
        status = []
        if self._config.wandb_api_key:
            status.append("WandB: ✓")
        else:
            status.append("WandB: ✗")

        if self._config.alpha_vantage_api_key:
            status.append("AlphaVantage: ✓")
        else:
            status.append("AlphaVantage: ✗")

        return f"SecretsManager({', '.join(status)})"