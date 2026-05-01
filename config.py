"""
Configuration and logging setup for the project.
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Find .env file relative to this config.py file
BASE_DIR = Path(__file__).resolve().parent
env_path = BASE_DIR / '.env'
load_dotenv(dotenv_path=env_path)

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-lite:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it"
]

def setup_logging():
    """
    Sets up the logging configuration with a clean format:
    timestamp, level, and message.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

setup_logging()
logger = logging.getLogger(__name__)
