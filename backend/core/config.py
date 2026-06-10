"""
Core Configuration — Pydantic Settings
"""
import os
from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    APP_NAME: str = "ParkVoice AI"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    SECRET_KEY: str = "dev-secret-change-in-production"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./parkvoice.db"

    # Storage
    UPLOAD_DIR: str = "./uploads"
    MODEL_DIR: str = "./models"
    MAX_UPLOAD_SIZE_MB: int = 50
    ALLOWED_AUDIO_FORMATS: List[str] = ["wav", "mp3", "m4a", "ogg", "flac"]

    # Model paths
    ENSEMBLE_MODEL_PATH: str = "./models/ensemble_model.pkl"
    SEVERITY_MODEL_PATH: str = "./models/severity_model.pkl"
    CNN_MODEL_PATH: str = "./models/cnn_model.pt"
    TEMPORAL_MODEL_PATH: str = "./models/temporal_progression.pt"
    ONNX_MODEL_PATH: str = "./models/ensemble_quantized.onnx"

    # Audio Feature Extraction
    SAMPLE_RATE: int = 22050
    N_MFCC: int = 40
    N_MELS: int = 128
    HOP_LENGTH: int = 512
    N_FFT: int = 2048

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    # Compute Device
    DEVICE: str = "auto"  # auto | cpu | cuda

    @property
    def effective_device(self) -> str:
        if self.DEVICE == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.DEVICE


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
