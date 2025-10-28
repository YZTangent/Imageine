"""Complex pipeline with ControlNet support."""
from PIL import Image
from typing import Optional, List
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


class ComplexPipeline:
    """Complex inpainting pipeline with ControlNet support."""

    def __init__(self, model_manager: ModelManager, config):
        """
        Initialize complex pipeline.

        Args:
            model_manager: ModelManager instance
            config: Configuration object
        """
        self.model_manager = model_manager
        self.config = config
        logger.info("ComplexPipeline initialized")

    def generate(
        self,
        base_image: Image.Image,
        prompt: str,
        reference_image: Optional[Image.Image] = None,
        control_types: Optional[List[str]] = None,
        negative_prompt: str = "deformed, distorted, low quality, blurry, bad anatomy",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        controlnet_conditioning_scale: float = 1.0,
        ip_adapter_scale: float = 0.6,
        seed: Optional[int] = None
    ) -> Image.Image:
        """
        Generate edited image using complex pipeline with ControlNet.

        Args:
            base_image: Input image to edit
            prompt: Text description of desired edit
            reference_image: Optional reference image for IP-Adapter
            control_types: List of control types to use (pose, depth, etc.)
            negative_prompt: What to avoid in generation
            num_inference_steps: Number of diffusion steps
            guidance_scale: How closely to follow prompt
            controlnet_conditioning_scale: ControlNet influence (0-1)
            ip_adapter_scale: IP-Adapter influence (0-1)
            seed: Random seed for reproducibility

        Returns:
            Edited image (same size as input)
        """
        logger.info("=" * 60)
        logger.info("Starting complex pipeline generation")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Control types: {control_types}")
        logger.info(f"Has reference image: {reference_image is not None}")
        logger.info("=" * 60)

        if control_types is None:
            control_types = []

        # 1. Preprocess
        logger.info("Step 1: Preprocessing images...")
        processed_image, original_size = preprocess_image(base_image, target_size=(512, 512))
        logger.info(f"  Processed size: {processed_image.size}")

        # 2. Create mask
        logger.info("Step 2: Creating mask...")
        mask = create_mask_from_prompt(
            processed_image,
            prompt,
            detector=self.model_manager.grounding_dino
        )
        mask = apply_mask_blur(mask, blur_radius=21)

        # 3. Extract control images
        control_images = []
        if control_types:
            logger.info(f"Step 3: Extracting control images ({len(control_types)})...")

            for control_type in control_types:
                if control_type == "pose":
                    logger.info("  Extracting pose...")
                    pose_image = self.model_manager.pose_extractor.extract(processed_image)
                    if pose_image:
                        control_images.append(pose_image)
                        logger.info("  ✓ Pose extracted")
                    else:
                        logger.warning("  ⚠ Pose extraction failed, skipping")

                elif control_type == "depth":
                    logger.info("  Extracting depth...")
                    depth_image = self.model_manager.depth_extractor.extract(processed_image)
                    if depth_image:
                        control_images.append(depth_image)
                        logger.info("  ✓ Depth extracted")
                    else:
                        logger.warning("  ⚠ Depth extraction failed, skipping")

            logger.info(f"  Total control images: {len(control_images)}")
        else:
            logger.info("Step 3: No control types specified, skipping")

        # 4. Set seed
        if seed is not None:
            generator = torch.Generator(device=self.model_manager.device).manual_seed(seed)
            logger.info(f"Step 4: Using seed: {seed}")
        else:
            generator = None
            logger.info("Step 4: Using random seed")

        # 5. Choose pipeline and generate
        if control_images:
            logger.info("Step 5: Running ControlNet inpainting...")
            pipe = self.model_manager.load_controlnet_pipeline(control_types)

            # Prepare controlnet conditioning scale
            if len(control_images) == 1:
                cn_scale = controlnet_conditioning_scale
            else:
                cn_scale = [controlnet_conditioning_scale] * len(control_images)

            logger.info("  Generating with ControlNet...")
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=processed_image,
                mask_image=mask,
                control_image=control_images if len(control_images) > 1 else control_images[0],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                controlnet_conditioning_scale=cn_scale,
                generator=generator
            ).images[0]

        else:
            logger.info("Step 5: Running standard inpainting...")
            pipe = self.model_manager.sd_pipe

            # Apply IP-Adapter if reference image provided
            if reference_image:
                logger.info("  Applying IP-Adapter...")
                pipe = self.model_manager.ip_adapter.apply_adapter(
                    pipe,
                    reference_image,
                    scale=ip_adapter_scale
                )

            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=processed_image,
                mask_image=mask,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]

        logger.info("  ✓ Generation complete")

        # 6. Postprocess
        logger.info("Step 6: Postprocessing...")
        final_image = postprocess_image(result, original_size)
        logger.info(f"  Final size: {final_image.size}")

        logger.info("=" * 60)
        logger.info("✅ Complex pipeline complete")
        logger.info("=" * 60)

        return final_image
