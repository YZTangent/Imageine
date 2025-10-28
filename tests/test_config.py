"""Test configuration system."""
import pytest
from src.utils.config import Config


def test_config_loads():
    """Test that config loads successfully."""
    config = Config("config/default.yaml")
    assert config is not None


def test_api_config():
    """Test API configuration."""
    config = Config("config/default.yaml")
    assert config.api.host == "0.0.0.0"
    assert config.api.port == 8000


def test_models_config():
    """Test models configuration."""
    config = Config("config/default.yaml")
    assert config.models.device in ["cuda", "cpu"]
    assert config.models.dtype in ["float16", "float32"]


def test_generation_config():
    """Test generation configuration."""
    config = Config("config/default.yaml")
    assert config.generation.default_steps > 0
    assert config.generation.default_guidance_scale > 0
