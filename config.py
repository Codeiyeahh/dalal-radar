"""
Configuration and logging setup for the project.
"""
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

APIFY_TOKEN = os.getenv("APIFY_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-lite:free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

FALLBACK_MODELS = [
    "google/gemma-4-31b-it:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat:free",
    "qwen/qwen3-8b:free"
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
