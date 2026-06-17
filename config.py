import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# API Keys
BAND_API_KEY = os.getenv("BAND_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
AIML_API_KEY = os.getenv("AIML_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# API Endpoints
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
AIML_BASE_URL = "https://api.aimlapi.com/v1"

# Setup global logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)