"""
Configuration settings for Football Analysis API
Configured to use C drive for data storage
"""

import os
from pathlib import Path

class Settings:
    # API Configuration
    API_TITLE = "Football Analysis API"
    API_VERSION = "1.0.0"
    API_HOST = "0.0.0.0"
    API_PORT = 8000
    
    # C Drive Data Paths (Heavy data storage)
    DATA_ROOT = Path("C:/FootballData")
    UPLOAD_DIR = DATA_ROOT / "uploads"
    ANALYSIS_DIR = DATA_ROOT / "analysis"
    MODELS_DIR = DATA_ROOT / "models"
    TEMP_DIR = DATA_ROOT / "temp"
    DATASET_DIR = DATA_ROOT / "dataset"
    
    # D Drive Code Paths (Lightweight)
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    AI_ENGINE_PATH = PROJECT_ROOT / "3dsp_utils"
    
    # Database Configuration
    DATABASE_URL = "sqlite:///C:/FootballData/football_app.db"
    
    # Security
    SECRET_KEY = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    
    # File Upload Settings
    MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
    ALLOWED_VIDEO_EXTENSIONS = [".mp4", ".avi", ".mov", ".mkv"]
    
    # AI Processing Settings
    MAX_PROCESSING_TIME = 300  # 5 minutes
    DEFAULT_ANALYSIS_FRAMES = 20
    
    def __init__(self):
        # Create directories if they don't exist
        self.create_directories()
    
    def create_directories(self):
        """Create necessary directories on C drive"""
        directories = [
            self.DATA_ROOT,
            self.UPLOAD_DIR,
            self.ANALYSIS_DIR,
            self.MODELS_DIR,
            self.TEMP_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Directory ready: {directory}")

# Global settings instance
settings = Settings()

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    settings.DATABASE_URL = os.getenv("DATABASE_URL", settings.DATABASE_URL)
    settings.SECRET_KEY = os.getenv("SECRET_KEY", settings.SECRET_KEY)