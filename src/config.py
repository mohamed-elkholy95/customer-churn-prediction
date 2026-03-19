"""Configuration."""
import logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent.parent
RANDOM_SEED = 42
API_HOST = "0.0.0.0"
API_PORT = 8011
