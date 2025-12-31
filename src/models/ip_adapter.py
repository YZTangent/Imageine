"""IP-Adapter for reference image conditioning."""
import torch
from PIL import Image
from typing import Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class IPAdapterLoader:
    """Loader for IP-Adapter models."""

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        cache_dir: str = "./models_cache"
    ):
        """
        Initialize IP-Adapter loader.

        Args:
            device: Device to run on
            dtype: Data type for model
            cache_dir: Directory to cache models
        """
        self.device = device
        self.dtype = dtype
        self.cache_dir = cache_dir
        self._adapter = None

        logger.info(f"IPAdapterLoader initialized (device={device})")

    def load_adapter(self, base_model: str = "sd15"):
        """
        Load IP-Adapter model.

        Args:
            base_model: Base model type (sd15, sdxl, etc.)
        """
        if self._adapter is not None:
            return self._adapter

        logger.info(f"Loading IP-Adapter for {base_model}...")

        # Try multiple approaches for IP-Adapter

        # Approach 1: Try loading IP-Adapter from diffusers (modern approach)
        try:
            from diffusers import AutoPipelineForInpainting
            from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

            logger.info("Attempting to load CLIP vision model for IP-Adapter...")

            # Load CLIP image encoder for IP-Adapter
            self._image_encoder = CLIPVisionModelWithProjection.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                cache_dir=self.cache_dir,
                torch_dtype=self.dtype
            ).to(self.device)

            self._feature_extractor = CLIPImageProcessor.from_pretrained(
                "h94/IP-Adapter",
                subfolder="models/image_encoder",
                cache_dir=self.cache_dir
            )

            self._adapter = {
                "image_encoder": self._image_encoder,
                "feature_extractor": self._feature_extractor,
                "type": "clip_vision"
            }

            logger.info("✓ IP-Adapter (CLIP vision) loaded successfully")
            return self._adapter

        except Exception as e:
            logger.debug(f"CLIP vision approach failed: {e}")

        # Approach 2: Try simplified image conditioning using CLIP embeddings
        try:
            from transformers import CLIPProcessor, CLIPModel

            logger.info("Attempting simplified CLIP-based conditioning...")

            self._clip_model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=self.cache_dir,
                torch_dtype=self.dtype
            ).to(self.device)

            self._clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir=self.cache_dir
            )

            self._adapter = {
                "clip_model": self._clip_model,
                "clip_processor": self._clip_processor,
                "type": "clip_embeddings"
            }

            logger.info("✓ CLIP-based conditioning loaded successfully")
            return self._adapter

        except Exception as e:
            logger.debug(f"CLIP embeddings approach failed: {e}")

        # Fallback: No IP-Adapter
        logger.warning("IP-Adapter not available, reference images will not be used")
        logger.info("Generation will proceed without reference image conditioning")
        self._adapter = None
        return self._adapter

    def prepare_reference_image(
        self,
        reference_image: Image.Image,
        target_size: tuple = (224, 224)
    ) -> torch.Tensor:
        """
        Prepare reference image for IP-Adapter.

        Args:
            reference_image: Reference PIL image
            target_size: Target size for CLIP encoder

        Returns:
            Processed image tensor
        """
        # Resize and normalize
        image = reference_image.resize(target_size, Image.LANCZOS)

        # Convert to tensor
        image_array = torch.from_numpy(
            np.array(image).transpose(2, 0, 1)
        ).float() / 255.0

        # Normalize for CLIP
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

        image_tensor = (image_array - mean) / std

        return image_tensor.unsqueeze(0).to(self.device, dtype=self.dtype)

    def extract_image_embeddings(
        self,
        reference_image: Image.Image
    ) -> Optional[torch.Tensor]:
        """
        Extract embeddings from reference image.

        Args:
            reference_image: Reference PIL image

        Returns:
            Image embeddings tensor or None
        """
        if self._adapter is None:
            return None

        try:
            adapter_type = self._adapter.get("type")

            if adapter_type == "clip_vision":
                # Use CLIP vision encoder
                image_encoder = self._adapter["image_encoder"]
                feature_extractor = self._adapter["feature_extractor"]

                inputs = feature_extractor(images=reference_image, return_tensors="pt")
                inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

                with torch.no_grad():
                    image_embeds = image_encoder(**inputs).image_embeds

                logger.debug(f"Extracted CLIP vision embeddings: {image_embeds.shape}")
                return image_embeds

            elif adapter_type == "clip_embeddings":
                # Use CLIP model
                clip_model = self._adapter["clip_model"]
                clip_processor = self._adapter["clip_processor"]

                inputs = clip_processor(images=reference_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "text"}

                with torch.no_grad():
                    image_features = clip_model.get_image_features(**inputs)

                logger.debug(f"Extracted CLIP embeddings: {image_features.shape}")
                return image_features

        except Exception as e:
            logger.error(f"Failed to extract image embeddings: {e}")
            return None

    def apply_adapter(
        self,
        pipe,
        reference_image: Optional[Image.Image],
        scale: float = 0.6
    ):
        """
        Apply IP-Adapter to pipeline.

        Args:
            pipe: Diffusion pipeline
            reference_image: Reference image (optional)
            scale: Adapter influence scale (0-1)

        Returns:
            Modified pipeline with IP-Adapter conditioning
        """
        if reference_image is None:
            logger.debug("No reference image provided, skipping IP-Adapter")
            return pipe

        # Load adapter if not loaded
        if self._adapter is None:
            self.load_adapter()

        if self._adapter is None:
            logger.debug("IP-Adapter not available, skipping")
            return pipe

        try:
            # Extract embeddings from reference image
            image_embeds = self.extract_image_embeddings(reference_image)

            if image_embeds is None:
                logger.warning("Failed to extract image embeddings")
                return pipe

            # Try to use load_ip_adapter method if available (newer diffusers)
            if hasattr(pipe, 'load_ip_adapter'):
                try:
                    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
                    pipe.set_ip_adapter_scale(scale)
                    logger.info(f"✓ IP-Adapter loaded and applied with scale={scale}")
                    return pipe
                except Exception as e:
                    logger.debug(f"load_ip_adapter method failed: {e}")

            # Alternative: Store embeddings for manual injection
            # Note: This requires modifying the pipeline's forward pass,
            # which is complex and version-specific
            pipe._ip_adapter_image_embeds = image_embeds
            pipe._ip_adapter_scale = scale

            logger.info(f"IP-Adapter embeddings prepared (scale={scale})")
            logger.warning("Note: Full IP-Adapter integration requires custom pipeline modifications")
            logger.info("Consider using text prompts that describe the reference image for best results")

        except Exception as e:
            logger.warning(f"Failed to apply IP-Adapter: {e}")
            logger.info("Continuing without reference image conditioning")

        return pipe
