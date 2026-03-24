"""
Configuration Module
====================

Centralizes all project-wide settings in one place. This avoids magic numbers
scattered across the codebase and makes it easy to adjust behavior.

All settings can be overridden via environment variables, making the project
deployable across different environments (local, Docker, CI, cloud) without
code changes.

Settings:
    RANDOM_SEED: Controls reproducibility for all random operations (data
        generation, model training, cross-validation splits). Set to 42 by
        convention — any integer works, but changing it will produce different
        results across the entire pipeline.
        Override: CHURN_RANDOM_SEED=123

    API_HOST / API_PORT: Network binding for the FastAPI server. 0.0.0.0
        listens on all interfaces (needed for Docker); change to 127.0.0.1
        for local-only access.
        Override: CHURN_API_HOST=127.0.0.1  CHURN_API_PORT=9000

    LOG_LEVEL: Logging verbosity. DEBUG for development, WARNING for production.
        Override: CHURN_LOG_LEVEL=DEBUG

    BASE_DIR: Project root directory, resolved from this file's location.
        Used for relative path resolution (e.g., model save directory).

    MODELS_DIR: Default directory for saved model artifacts.

    DEFAULT_N_SAMPLES: Default number of synthetic samples for data generation.
        Override: CHURN_DEFAULT_SAMPLES=5000

    DEFAULT_TEST_SIZE: Default train/test split ratio.
        Override: CHURN_DEFAULT_TEST_SIZE=0.25
"""

import logging
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Reproducibility seed — used by numpy, sklearn, and data generation.
# Changing this value will produce different synthetic data and model results.
RANDOM_SEED = int(os.environ.get("CHURN_RANDOM_SEED", "42"))

# API server configuration
API_HOST = os.environ.get("CHURN_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("CHURN_API_PORT", "8011"))

# Logging configuration
LOG_LEVEL = os.environ.get("CHURN_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Default data generation parameters
DEFAULT_N_SAMPLES = int(os.environ.get("CHURN_DEFAULT_SAMPLES", "1000"))
DEFAULT_TEST_SIZE = float(os.environ.get("CHURN_DEFAULT_TEST_SIZE", "0.2"))
