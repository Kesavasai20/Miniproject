"""
FloatChat Configuration Management
Centralized configuration using Pydantic Settings
"""

import os
from pathlib import Path
from typing import Optional
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Project paths
    BASE_DIR: Path = Path(__file__).parent
    DATA_DIR: Path = Field(default=Path("./data"))
    RAW_DATA_DIR: Path = Field(default=Path("./data/raw"))
    PROCESSED_DATA_DIR: Path = Field(default=Path("./data/processed"))
    LOGS_DIR: Path = Field(default=Path("./logs"))
    
    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "floatchat"
    POSTGRES_USER: str = "floatchat"
    POSTGRES_PASSWORD: str = "floatchat_secret_2024"
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # Ollama LLM
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "mistral"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    
    # Vector Database
    CHROMA_PERSIST_DIR: Path = Field(default=Path("./data/chroma"))
    FAISS_INDEX_PATH: Path = Field(default=Path("./data/faiss"))
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    
    # ARGO Data Sources
    INCOIS_FTP: str = "ftp://ftp.incois.gov.in/argo"
    IFREMER_FTP: str = "ftp://ftp.ifremer.fr/ifremer/argo"
    IFREMER_GDAC: str = "https://data-argo.ifremer.fr"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_DEBUG: bool = True
    API_TITLE: str = "FloatChat API"
    API_VERSION: str = "1.0.0"
    
    # Streamlit
    STREAMLIT_PORT: int = 8501
    STREAMLIT_THEME: str = "dark"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[Path] = Field(default=Path("./logs/floatchat.log"))
    
    # Voice Recognition
    WHISPER_MODEL: str = "base"
    ENABLE_VOICE: bool = True
    
    # Feature Flags
    ENABLE_3D_GLOBE: bool = True
    ENABLE_ANOMALY_DETECTION: bool = True
    ENABLE_VOICE_COMMANDS: bool = True
    ENABLE_MULTI_LANGUAGE: bool = True
    ENABLE_REAL_TIME_ALERTS: bool = True
    ENABLE_COLLABORATIVE_ANNOTATIONS: bool = True
    
    # Alert Configuration
    ENABLE_EMAIL_ALERTS: bool = False
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    ALERT_EMAIL: Optional[str] = None
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.DATA_DIR,
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.LOGS_DIR,
            self.CHROMA_PERSIST_DIR,
            self.FAISS_INDEX_PATH.parent if self.FAISS_INDEX_PATH else None
        ]
        for dir_path in dirs:
            if dir_path:
                Path(dir_path).mkdir(parents=True, exist_ok=True)


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    settings = Settings()
    settings.ensure_directories()
    return settings


# Convenience instance
settings = get_settings()
