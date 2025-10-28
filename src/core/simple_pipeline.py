"""Simple pipeline implementation (fast path)."""
from PIL import Image
from typing import Optional
import torch
from ..models.loader import ModelManager
from ..utils.image_processing import (
    preprocess_image,
    postprocess_image,
    create_mask_from_prompt,
    apply_mask_blur
)
import logging

logger = logging.getLogger(__name__)


class SimplePipeline:
    """Simple inpainting pipeline without ControlNet."""

    def __init__(self, model_manager: ModelManager, config):
        """
        Initialize simple pipeline.

        Args:
            model_manager: ModelManager instance
            config: Configuration object
        """
        self.model_manager = model_manager
        self.config = config
        logger.info("SimplePipeline initialized")

    def generate(
        self,
        base_image: Image.Image,
        prompt: str,
        negative_prompt: str = "deformed, distorted, low quality, blurry, bad anatomy",
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate edited image using simple inpainting.

        Args:
            base_image: Input image to edit
            prompt: Text description of desired edit
            negative_prompt: What to avoid in generation
            num_inference_steps: Number of diffusion steps (higher = better quality, slower)
            guidance_scale: How closely to follow prompt (7.5 is good default)
            seed: Random seed for reproducibility (None = random)

        Returns:
            Edited image (same size as input)
        """
        logger.info("=" * 60)
        logger.info("Starting simple pipeline generation")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Steps: {num_inference_steps}, Guidance: {guidance_scale}, Seed: {seed}")
        logger.info("=" * 60)

        # 1. Preprocess
        logger.info("Step 1/5: Preprocessing image...")
        processed_image, original_size = preprocess_image(base_image, target_size=(512, 512))
        logger.info(f"  Original size: {original_size}")
        logger.info(f"  Processed size: {processed_image.size}")

        # 2. Create mask
        logger.info("Step 2/5: Creating mask...")
        mask = create_mask_from_prompt(processed_image, prompt)
        mask = apply_mask_blur(mask, blur_radius=21)
        logger.info(f"  Mask size: {mask.size}")

        # 3. Set seed
        if seed is not None:
            generator = torch.Generator(device=self.model_manager.device).manual_seed(seed)
            logger.info(f"  Using seed: {seed}")
        else:
            generator = None
            logger.info("  Using random seed")

        # 4. Run inpainting
        logger.info("Step 3/5: Running SD inpainting...")
        logger.info(f"  Loading pipeline on {self.model_manager.device}...")

        pipe = self.model_manager.sd_pipe

        logger.info("  Generating image...")
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=processed_image,
            mask_image=mask,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        ).images[0]

        logger.info("  ✓ Inpainting complete")

        # 5. Postprocess
        logger.info("Step 4/5: Postprocessing...")
        final_image = postprocess_image(result, original_size)
        logger.info(f"  Final size: {final_image.size}")

        logger.info("Step 5/5: Done!")
        logger.info("=" * 60)
        logger.info("✅ Generation complete")
        logger.info("=" * 60)

        return final_image

    def warmup(self):
        """
        Warmup pipeline by running a test generation.
        This compiles and caches operations for faster subsequent runs.
        """
        logger.info("Warming up pipeline...")

        # Create a small test image
        test_image = Image.new('RGB', (512, 512), color=(128, 128, 128))

        # Run a quick generation
        self.generate(
            base_image=test_image,
            prompt="test image",
            num_inference_steps=1,
            seed=42
        )

        logger.info("✓ Pipeline warmup complete")
