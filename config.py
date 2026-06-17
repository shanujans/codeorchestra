import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# API Keys
BAND_API_KEY = os.getenv("BAND_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
AIML_API_KEY = os.getenv("AIML_API_KEY", "")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# API Endpoints
HF_BASE_URL = "https://api-inference.huggingface.co/v1/"
AIML_BASE_URL = "https://api.aimlapi.com/v1"

# Setup global logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)