"""
Configuration Module
====================

Centralizes all project-wide settings in one place. This avoids magic numbers
scattered across the codebase and makes it easy to adjust behavior.

Settings:
    RANDOM_SEED: Controls reproducibility for all random operations (data
        generation, model training, cross-validation splits). Set to 42 by
        convention — any integer works, but changing it will produce different
        results across the entire pipeline.

    API_HOST / API_PORT: Network binding for the FastAPI server. 0.0.0.0
        listens on all interfaces (needed for Docker); change to 127.0.0.1
        for local-only access.

    BASE_DIR: Project root directory, resolved from this file's location.
        Used for relative path resolution (e.g., model save directory).

    MODELS_DIR: Default directory for saved model artifacts.
"""

import logging
from pathlib import Path

# Configure logging with timestamps and module names for debugging.
# In production, you might use structured logging (JSON format) instead.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# Reproducibility seed — used by numpy, sklearn, and data generation.
# Changing this value will produce different synthetic data and model results.
RANDOM_SEED = 42

# API server configuration
API_HOST = "0.0.0.0"
API_PORT = 8011
