"""Model loading and management."""
import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionControlNetInpaintPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler
)
from typing import Optional, List
import logging
from .grounding_dino import GroundingDINODetector
from .extractors import PoseExtractor, DepthExtractor
from .ip_adapter import IPAdapterLoader

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and caching of ML models."""

    def __init__(self, config):
        """
        Initialize model manager.

        Args:
            config: Configuration object
        """
        self.config = config

        # Auto-detect device
        device_config = config.models.device
        if device_config == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
                logger.info("Auto-detected device: CUDA (NVIDIA GPU)")
            elif torch.backends.mps.is_available():
                self.device = "mps"
                logger.info("Auto-detected device: MPS (Apple Silicon)")
            else:
                self.device = "cpu"
                logger.info("Auto-detected device: CPU (no GPU acceleration)")
        else:
            self.device = device_config
            logger.info(f"Using configured device: {self.device}")

        self.dtype = torch.float16 if config.models.dtype == "float16" else torch.float32
        self.cache_dir = config.models.cache_dir

        # Model instances
        self._sd_pipe: Optional[StableDiffusionInpaintPipeline] = None
        self._controlnet_pipe: Optional[StableDiffusionControlNetInpaintPipeline] = None
        self._controlnet_pose: Optional[ControlNetModel] = None
        self._controlnet_depth: Optional[ControlNetModel] = None

        # Phase 2 components
        self._grounding_dino: Optional[GroundingDINODetector] = None
        self._pose_extractor: Optional[PoseExtractor] = None
        self._depth_extractor: Optional[DepthExtractor] = None
        self._ip_adapter: Optional[IPAdapterLoader] = None

        logger.info(f"ModelManager initialized (device={self.device}, dtype={self.dtype})")

    def load_sd_inpainting(self) -> StableDiffusionInpaintPipeline:
        """
        Load Stable Diffusion Inpainting pipeline.

        Returns:
            Loaded pipeline
        """
        if self._sd_pipe is not None:
            logger.info("SD Inpainting already loaded, returning cached instance")
            return self._sd_pipe

        logger.info("Loading SD Inpainting pipeline...")

        try:
            self._sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.config.models.base_model,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir
            )

            # Set scheduler
            self._sd_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self._sd_pipe.scheduler.config
            )
            logger.info("✓ Scheduler set to DPMSolverMultistep")

            # Move to device
            self._sd_pipe = self._sd_pipe.to(self.device)
            logger.info(f"✓ Pipeline moved to {self.device}")

            # Enable optimizations
            if hasattr(self._sd_pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    self._sd_pipe.enable_xformers_memory_efficient_attention()
                    logger.info("✓ xformers memory-efficient attention enabled")
                except Exception as e:
                    logger.warning(f"xformers not available: {e}")
                    logger.info("Continuing without xformers (slightly slower)")

            # Enable attention slicing for memory efficiency
            self._sd_pipe.enable_attention_slicing()
            logger.info("✓ Attention slicing enabled")

            logger.info("✅ SD Inpainting pipeline loaded successfully")
            return self._sd_pipe

        except Exception as e:
            logger.error(f"Failed to load SD Inpainting: {e}")
            raise

    @property
    def sd_pipe(self) -> StableDiffusionInpaintPipeline:
        """
        Get SD pipeline, loading if necessary.

        Returns:
            SD Inpainting pipeline
        """
        if self._sd_pipe is None:
            self.load_sd_inpainting()
        return self._sd_pipe

    def load_controlnet(
        self,
        control_type: str = "pose"
    ) -> ControlNetModel:
        """
        Load ControlNet model.

        Args:
            control_type: Type of control (pose, depth, canny, etc.)

        Returns:
            ControlNet model
        """
        if control_type == "pose" and self._controlnet_pose is not None:
            return self._controlnet_pose
        elif control_type == "depth" and self._controlnet_depth is not None:
            return self._controlnet_depth

        logger.info(f"Loading ControlNet ({control_type})...")

        try:
            if control_type == "pose":
                model_id = "lllyasviel/control_v11p_sd15_openpose"
            elif control_type == "depth":
                model_id = "lllyasviel/control_v11f1p_sd15_depth"
            else:
                raise ValueError(f"Unknown control type: {control_type}")

            controlnet = ControlNetModel.from_pretrained(
                model_id,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir
            )
            controlnet = controlnet.to(self.device)

            # Cache the model
            if control_type == "pose":
                self._controlnet_pose = controlnet
            elif control_type == "depth":
                self._controlnet_depth = controlnet

            logger.info(f"✓ ControlNet ({control_type}) loaded successfully")
            return controlnet

        except Exception as e:
            logger.error(f"Failed to load ControlNet ({control_type}): {e}")
            raise

    def load_controlnet_pipeline(
        self,
        control_types: List[str]
    ) -> StableDiffusionControlNetInpaintPipeline:
        """
        Load ControlNet inpainting pipeline.

        Args:
            control_types: List of control types to use

        Returns:
            ControlNet pipeline
        """
        logger.info(f"Loading ControlNet pipeline with {control_types}...")

        try:
            # Load ControlNet models
            controlnets = [self.load_controlnet(ct) for ct in control_types]

            # Create pipeline with ControlNet
            if len(controlnets) == 1:
                controlnet = controlnets[0]
            else:
                controlnet = controlnets

            pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
                self.config.models.base_model,
                controlnet=controlnet,
                torch_dtype=self.dtype,
                cache_dir=self.cache_dir
            )

            # Set scheduler
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )

            # Move to device
            pipe = pipe.to(self.device)

            # Enable optimizations
            if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                    logger.info("✓ xformers enabled for ControlNet pipeline")
                except Exception:
                    pass

            pipe.enable_attention_slicing()

            self._controlnet_pipe = pipe
            logger.info("✓ ControlNet pipeline loaded successfully")
            return pipe

        except Exception as e:
            logger.error(f"Failed to load ControlNet pipeline: {e}")
            raise

    @property
    def grounding_dino(self) -> GroundingDINODetector:
        """Get GroundingDINO detector, loading if necessary."""
        if self._grounding_dino is None:
            self._grounding_dino = GroundingDINODetector(device=self.device)
        return self._grounding_dino

    @property
    def pose_extractor(self) -> PoseExtractor:
        """Get pose extractor, loading if necessary."""
        if self._pose_extractor is None:
            self._pose_extractor = PoseExtractor(device=self.device)
        return self._pose_extractor

    @property
    def depth_extractor(self) -> DepthExtractor:
        """Get depth extractor, loading if necessary."""
        if self._depth_extractor is None:
            self._depth_extractor = DepthExtractor(device=self.device)
        return self._depth_extractor

    @property
    def ip_adapter(self) -> IPAdapterLoader:
        """Get IP-Adapter, loading if necessary."""
        if self._ip_adapter is None:
            self._ip_adapter = IPAdapterLoader(
                device=self.device,
                dtype=self.dtype,
                cache_dir=self.cache_dir
            )
        return self._ip_adapter

    def unload_all(self):
        """Unload all models to free memory."""
        if self._sd_pipe is not None:
            logger.info("Unloading SD Inpainting pipeline...")
            del self._sd_pipe
            self._sd_pipe = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA cache cleared")

        logger.info("✓ All models unloaded")

    def get_memory_usage(self) -> dict:
        """
        Get current GPU memory usage.

        Returns:
            Dictionary with memory stats
        """
        if not torch.cuda.is_available():
            return {"gpu_available": False}

        return {
            "gpu_available": True,
            "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            "free_mb": (torch.cuda.mem_get_info()[0]) / (1024 * 1024),
            "total_mb": (torch.cuda.mem_get_info()[1]) / (1024 * 1024),
        }
