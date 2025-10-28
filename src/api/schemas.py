"""API request/response schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List


class GenerationConfig(BaseModel):
    """Configuration for image generation."""

    num_inference_steps: int = Field(
        default=25,
        ge=1,
        le=100,
        description="Number of diffusion steps (higher = better quality, slower)"
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=0,
        le=20,
        description="How closely to follow prompt (7.5 is recommended)"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility (None = random)"
    )
    # Phase 2: Advanced options
    force_controlnet: bool = Field(
        default=False,
        description="Force use of complex pipeline with ControlNet"
    )
    control_types: Optional[List[str]] = Field(
        default=None,
        description="Specific control types to use (pose, depth, canny)"
    )
    controlnet_conditioning_scale: float = Field(
        default=1.0,
        ge=0,
        le=2.0,
        description="ControlNet influence strength (0-2)"
    )
    ip_adapter_scale: float = Field(
        default=0.6,
        ge=0,
        le=1.0,
        description="IP-Adapter influence for reference images (0-1)"
    )


class GenerateRequest(BaseModel):
    """Request for image generation."""

    base_image: str = Field(
        ...,
        description="Base64 encoded image or URL"
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Text prompt for editing (e.g., 'blue car', 'wearing red shirt')"
    )
    reference_image: Optional[str] = Field(
        default=None,
        description="Base64 encoded reference image (for IP-Adapter, Phase 2)"
    )
    negative_prompt: str = Field(
        default="deformed, distorted, low quality, blurry, bad anatomy",
        description="What to avoid in generation"
    )
    config: GenerationConfig = Field(
        default_factory=GenerationConfig,
        description="Generation configuration"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "base_image": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
                "prompt": "blue car",
                "negative_prompt": "deformed, distorted, low quality",
                "config": {
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "seed": 42
                }
            }
        }


class GenerationMetadata(BaseModel):
    """Metadata about generation."""

    input_resolution: str
    output_resolution: str
    seed_used: Optional[int]


class GenerationResult(BaseModel):
    """Result of successful generation."""

    image: str = Field(description="Base64 encoded result image")
    pipeline_used: str = Field(description="Pipeline type (simple or complex)")
    models_used: List[str] = Field(description="List of models used")
    processing_time_ms: int = Field(description="Total processing time in milliseconds")
    metadata: GenerationMetadata


class ErrorDetail(BaseModel):
    """Error details."""

    code: str = Field(description="Error code")
    message: str = Field(description="Human-readable error message")
    details: Optional[dict] = Field(default=None, description="Additional error details")


class GenerateResponse(BaseModel):
    """Response for image generation."""

    success: bool
    result: Optional[GenerationResult] = None
    error: Optional[ErrorDetail] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(description="Service status")
    models_loaded: List[str] = Field(description="List of loaded models")
    gpu_available: bool = Field(description="Whether GPU is available")
    gpu_memory_free_mb: Optional[int] = Field(
        default=None,
        description="Free GPU memory in MB"
    )
