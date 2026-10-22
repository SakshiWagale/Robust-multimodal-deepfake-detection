"""
config.py
---------
Centralized configuration management for the Flask application.
"""

import os

# Absolute path to project directory
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class BaseConfig:
    """Base configuration shared across all environments."""

    APP_NAME = "Multimodal Deepfake Detection System"

    # Flask Secret Key
    SECRET_KEY = os.environ.get(
        "SECRET_KEY",
        "change-this-secret-key-in-production"
    )

    # ==========================
    # Upload Configuration
    # ==========================
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50 MB

    ALLOWED_EXTENSIONS = {
        "jpg",
        "jpeg",
        "png",
        "webp",
        "mp4",
        "avi",
        "mov",
        "wav",
        "mp3"
    }

    # ==========================
    # Flask Server
    # ==========================
    HOST = "127.0.0.1"
    PORT = 5000

    DEBUG = False
    TESTING = False


class DevelopmentConfig(BaseConfig):
    """Development configuration."""
    DEBUG = True


class TestingConfig(BaseConfig):
    """Testing configuration."""
    DEBUG = True
    TESTING = True


class ProductionConfig(BaseConfig):
    """Production configuration."""
    DEBUG = False


_CONFIG_MAP = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(config_name="development"):
    """
    Return configuration class based on environment.
    """
    return _CONFIG_MAP.get(config_name.lower(), DevelopmentConfig)