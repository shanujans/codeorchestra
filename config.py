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

# Set to "true" in .env to run the FINAL QAOA circuit sample on real IBM
# Quantum hardware (requires `pip install qiskit-ibm-runtime` and a saved
# IBM account/token). The COBYLA parameter-tuning loop always runs locally
# on AerSimulator regardless of this flag -- real-hardware queue times make
# dozens of iterative calls impractical. If the connection fails for any
# reason, optimizer.py falls back to local simulation automatically.
USE_IBM_HARDWARE = os.getenv("USE_IBM_HARDWARE", "false").lower() == "true"

# API Endpoints
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
AIML_BASE_URL = "https://api.aimlapi.com/v1"

# Setup global logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)