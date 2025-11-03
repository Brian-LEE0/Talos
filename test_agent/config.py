"""
Configuration settings for the application
"""

# Database configuration
DATABASE_URL = "postgresql://localhost:5432/mydb"
DATABASE_USER = "admin"
DATABASE_PASSWORD = "secret123"

# Processing configuration
MAX_RETRIES = 3
TIMEOUT = 30  # seconds
BATCH_SIZE = 1000

# Feature flags
ENABLE_CACHING = True
ENABLE_LOGGING = True
DEBUG_MODE = False

# Paths
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
LOG_DIR = "logs"
