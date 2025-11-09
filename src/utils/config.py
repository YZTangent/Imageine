"""Configuration management for Imageine."""
import yaml
from pathlib import Path
from typing import Dict, Any
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Model configuration."""
    base_model: str
    device: str = "cuda"
    dtype: str = "float16"
    cache_dir: str = "./models_cache"


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    max_request_size_mb: int = 50


class GenerationConfig(BaseModel):
    """Generation default configuration."""
    default_steps: int = 25
    default_guidance_scale: float = 7.5
    scheduler: str = "DPMSolverMultistep"


class LimitsConfig(BaseModel):
    """Request limits configuration."""
    max_resolution: int = 2048
    min_resolution: int = 256
    request_timeout_seconds: int = 120


class LoggingConfig(BaseModel):
    """Pipeline logging configuration."""
    enabled: bool = True
    output_dir: str = "./pipeline_outputs"
    save_intermediates: bool = True


class Config:
    """Main configuration class."""

    def __init__(self, config_path: str = "config/default.yaml"):
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_file, 'r') as f:
            self._config = yaml.safe_load(f)

    @property
    def api(self) -> APIConfig:
        """Get API configuration."""
        return APIConfig(**self._config['api'])

    @property
    def models(self) -> ModelConfig:
        """Get models configuration."""
        return ModelConfig(**self._config['models'])

    @property
    def generation(self) -> GenerationConfig:
        """Get generation configuration."""
        return GenerationConfig(**self._config['generation'])

    @property
    def limits(self) -> LimitsConfig:
        """Get limits configuration."""
        return LimitsConfig(**self._config['limits'])

    @property
    def logging(self) -> LoggingConfig:
        """Get logging configuration."""
        return LoggingConfig(**self._config.get('logging', {}))

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._config.get(key, default)
