"""Simple pipeline implementation (fast path)."""
from PIL import Image
from typing import Optional
import torch
import time
from ..models.loader import ModelManager
from ..utils.image_processing import (
    preprocess_image,
    postprocess_image,
    create_mask_from_prompt,
    apply_mask_blur
)
from ..utils.pipeline_logger import PipelineLogger
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

        # Initialize pipeline logger
        log_config = config.logging
        self.pipeline_logger = PipelineLogger(
            output_dir=log_config.output_dir,
            enabled=log_config.enabled and log_config.save_intermediates
        )

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

        # Start logging session
        start_time = time.time()
        self.pipeline_logger.start_session()

        # 1. Preprocess
        logger.info("Step 1/5: Preprocessing image...")
        self.pipeline_logger.save_image(base_image, "00_input_original", step=0)

        processed_image, original_size = preprocess_image(base_image, target_size=(512, 512))
        logger.info(f"  Original size: {original_size}")
        logger.info(f"  Processed size: {processed_image.size}")

        self.pipeline_logger.save_image(processed_image, "01_preprocessed", step=1)

        # 2. Create mask
        logger.info("Step 2/5: Creating mask...")
        mask = create_mask_from_prompt(processed_image, prompt)
        self.pipeline_logger.save_image(mask, "02_mask_raw", step=2)

        mask = apply_mask_blur(mask, blur_radius=21)
        logger.info(f"  Mask size: {mask.size}")
        self.pipeline_logger.save_image(mask, "03_mask_blurred", step=3)

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
        self.pipeline_logger.save_image(result, "04_generated_raw", step=4)

        # 5. Postprocess
        logger.info("Step 4/5: Postprocessing...")
        final_image = postprocess_image(result, original_size)
        logger.info(f"  Final size: {final_image.size}")
        self.pipeline_logger.save_image(final_image, "05_final_output", step=5)

        # Save metadata
        end_time = time.time()
        metadata = {
            "pipeline": "simple",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "seed": seed if seed is not None else "random",
            "original_size": list(original_size),
            "processed_size": list(processed_image.size),
            "processing_time_seconds": round(end_time - start_time, 2)
        }
        self.pipeline_logger.save_metadata(metadata)

        logger.info("Step 5/5: Done!")
        logger.info("=" * 60)
        logger.info("✅ Generation complete")
        logger.info("=" * 60)

        # End logging session
        self.pipeline_logger.end_session()

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
