# Research Assistant Configuration

import os
from pathlib import Path

# API Keys (Set these as environment variables)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")  # Optional: newsapi.org (100 req/day free)

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MEMORY_DIR = DATA_DIR / "memory"
LOGS_DIR = DATA_DIR / "logs"
REPORTS_DIR = DATA_DIR / "reports"

# Create directories
for dir_path in [DATA_DIR, MEMORY_DIR, LOGS_DIR, REPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Agent Settings
MAX_RETRIES = 3
MAX_REPLANS = 2
REQUEST_TIMEOUT = 30
MAX_SEARCH_RESULTS = 5

# Memory Settings
SHORT_TERM_SIZE = 10
VECTOR_DB_PATH = MEMORY_DIR / "vector_store.pkl"

# Model Settings
GEMINI_MODEL = "gemini-pro"
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 2048

# Tool Settings
WIKIPEDIA_LANG = "en"
NEWS_SOURCES = ["general", "technology", "science"]