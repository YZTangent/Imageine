"""API route handlers."""
from fastapi import APIRouter, HTTPException
from PIL import Image
import base64
import io
import time
import logging
from .schemas import (
    GenerateRequest,
    GenerateResponse,
    GenerationResult,
    GenerationMetadata,
    ErrorDetail,
    HealthResponse
)
from ..core.simple_pipeline import SimplePipeline
from ..core.complex_pipeline import ComplexPipeline
from ..core.router import TaskRouter
from ..models.loader import ModelManager
import torch

logger = logging.getLogger(__name__)
router = APIRouter()

# Global instances (initialized on startup)
model_manager: ModelManager = None
simple_pipeline: SimplePipeline = None
complex_pipeline: ComplexPipeline = None
task_router: TaskRouter = None


def init_models(config):
    """
    Initialize models (called from main.py on startup).

    Args:
        config: Configuration object
    """
    global model_manager, simple_pipeline, complex_pipeline, task_router

    logger.info("Initializing models...")
    model_manager = ModelManager(config)
    simple_pipeline = SimplePipeline(model_manager, config)
    complex_pipeline = ComplexPipeline(model_manager, config)
    task_router = TaskRouter(config)

    # Warm up models
    logger.info("Loading and warming up models...")
    model_manager.load_sd_inpainting()
    logger.info("✓ Models initialized")


def decode_image(image_str: str) -> Image.Image:
    """
    Decode base64 image string to PIL Image.

    Args:
        image_str: Base64 encoded image (with or without data URL prefix)

    Returns:
        PIL Image

    Raises:
        ValueError: If image cannot be decoded
    """
    try:
        # Remove data URL prefix if present
        if image_str.startswith('data:image'):
            image_str = image_str.split(',', 1)[1]

        # Decode base64
        image_data = base64.b64decode(image_str)

        # Open as PIL Image
        image = Image.open(io.BytesIO(image_data))

        return image

    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")


def encode_image(image: Image.Image, format: str = "PNG") -> str:
    """
    Encode PIL Image to base64 string.

    Args:
        image: PIL Image
        format: Image format (PNG, JPEG, etc.)

    Returns:
        Base64 encoded image string
    """
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode()


@router.post("/api/v1/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate edited image.

    This endpoint performs image editing based on a text prompt using AI models.
    It intelligently routes between simple and complex pipelines based on the task.

    Args:
        request: Generation request with image and parameters

    Returns:
        Generated image and metadata
    """
    try:
        start_time = time.time()

        # Decode input images
        try:
            base_image = decode_image(request.base_image)
        except ValueError as e:
            return GenerateResponse(
                success=False,
                error=ErrorDetail(
                    code="INVALID_INPUT",
                    message=f"Invalid base image: {str(e)}"
                )
            )

        reference_image = None
        if request.reference_image:
            try:
                reference_image = decode_image(request.reference_image)
            except ValueError as e:
                return GenerateResponse(
                    success=False,
                    error=ErrorDetail(
                        code="INVALID_INPUT",
                        message=f"Invalid reference image: {str(e)}"
                    )
                )

        logger.info(f"📝 Request received: '{request.prompt}'")
        logger.info(f"   Input size: {base_image.size}")
        if reference_image:
            logger.info(f"   Reference image: {reference_image.size}")

        # Route to appropriate pipeline
        if request.config.control_types is not None:
            # User specified control types
            pipeline_type = "complex"
            control_types = request.config.control_types
            logger.info(f"   Using user-specified controls: {control_types}")
        else:
            # Intelligent routing
            pipeline_type, control_types = task_router.route(
                prompt=request.prompt,
                base_image=base_image,
                reference_image=reference_image,
                force_complex=request.config.force_controlnet
            )

        # Generate
        try:
            if pipeline_type == "complex":
                logger.info(f"   Pipeline: Complex (controls: {control_types})")

                result_image = complex_pipeline.generate(
                    base_image=base_image,
                    prompt=request.prompt,
                    reference_image=reference_image,
                    control_types=control_types,
                    negative_prompt=request.negative_prompt,
                    num_inference_steps=request.config.num_inference_steps,
                    guidance_scale=request.config.guidance_scale,
                    controlnet_conditioning_scale=request.config.controlnet_conditioning_scale,
                    ip_adapter_scale=request.config.ip_adapter_scale,
                    seed=request.config.seed
                )

                models_used = ["sd-inpainting"]
                if control_types:
                    models_used.extend([f"controlnet-{ct}" for ct in control_types])
                if reference_image:
                    models_used.append("ip-adapter")

            else:
                logger.info(f"   Pipeline: Simple")

                result_image = simple_pipeline.generate(
                    base_image=base_image,
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt,
                    num_inference_steps=request.config.num_inference_steps,
                    guidance_scale=request.config.guidance_scale,
                    seed=request.config.seed
                )

                models_used = ["sd-inpainting"]

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return GenerateResponse(
                    success=False,
                    error=ErrorDetail(
                        code="OUT_OF_MEMORY",
                        message="GPU out of memory. Try reducing image size or number of steps.",
                        details={"error": str(e)}
                    )
                )
            raise

        processing_time = (time.time() - start_time) * 1000

        # Encode result
        result_b64 = encode_image(result_image)

        logger.info(f"✅ Generation complete in {processing_time:.0f}ms")

        return GenerateResponse(
            success=True,
            result=GenerationResult(
                image=result_b64,
                pipeline_used=pipeline_type,
                models_used=models_used,
                processing_time_ms=int(processing_time),
                metadata=GenerationMetadata(
                    input_resolution=f"{base_image.width}x{base_image.height}",
                    output_resolution=f"{result_image.width}x{result_image.height}",
                    seed_used=request.config.seed
                )
            )
        )

    except Exception as e:
        logger.error(f"❌ Generation failed: {e}", exc_info=True)
        return GenerateResponse(
            success=False,
            error=ErrorDetail(
                code="MODEL_ERROR",
                message=f"Generation failed: {str(e)}",
                details={"type": type(e).__name__}
            )
        )


@router.get("/api/v1/health", response_model=HealthResponse)
async def health():
    """
    Health check endpoint.

    Returns:
        Service health status and resource information
    """
    gpu_available = torch.cuda.is_available()
    gpu_memory = None

    if gpu_available:
        try:
            gpu_memory = torch.cuda.mem_get_info()[0] // (1024 * 1024)  # Free memory in MB
        except Exception:
            gpu_memory = None

    models_loaded = []
    if model_manager and model_manager._sd_pipe is not None:
        models_loaded.append("sd-inpainting")

    return HealthResponse(
        status="healthy" if models_loaded else "starting",
        models_loaded=models_loaded,
        gpu_available=gpu_available,
        gpu_memory_free_mb=gpu_memory
    )


@router.get("/")
async def root():
    """
    Root endpoint.

    Returns:
        Service information
    """
    return {
        "name": "Imageine API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }
