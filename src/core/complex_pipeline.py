"""Complex pipeline with ControlNet support."""
from PIL import Image
from typing import Optional, List
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

        # Initialize pipeline logger
        log_config = config.logging
        self.pipeline_logger = PipelineLogger(
            output_dir=log_config.output_dir,
            enabled=log_config.enabled and log_config.save_intermediates
        )

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

        # Start logging session
        start_time = time.time()
        self.pipeline_logger.start_session()

        if control_types is None:
            control_types = []

        # 1. Preprocess
        logger.info("Step 1: Preprocessing images...")
        self.pipeline_logger.save_image(base_image, "00_input_original", step=0)

        if reference_image is not None:
            self.pipeline_logger.save_image(reference_image, "00_reference_image", step=0)

        processed_image, original_size = preprocess_image(base_image, target_size=(512, 512))
        logger.info(f"  Processed size: {processed_image.size}")
        self.pipeline_logger.save_image(processed_image, "01_preprocessed", step=1)

        # 2. Create mask
        logger.info("Step 2: Creating mask...")
        mask = create_mask_from_prompt(
            processed_image,
            prompt,
            detector=self.model_manager.grounding_dino
        )
        self.pipeline_logger.save_image(mask, "02_mask_raw", step=2)

        mask = apply_mask_blur(mask, blur_radius=21)
        self.pipeline_logger.save_image(mask, "03_mask_blurred", step=3)

        # 3. Extract control images
        control_images = []
        control_step = 4
        if control_types:
            logger.info(f"Step 3: Extracting control images ({len(control_types)})...")

            for control_type in control_types:
                if control_type == "pose":
                    logger.info("  Extracting pose...")
                    pose_image = self.model_manager.pose_extractor.extract(processed_image)
                    if pose_image:
                        control_images.append(pose_image)
                        self.pipeline_logger.save_image(pose_image, f"04_control_pose", step=control_step)
                        control_step += 1
                        logger.info("  ✓ Pose extracted")
                    else:
                        logger.warning("  ⚠ Pose extraction failed, skipping")

                elif control_type == "depth":
                    logger.info("  Extracting depth...")
                    depth_image = self.model_manager.depth_extractor.extract(processed_image)
                    if depth_image:
                        control_images.append(depth_image)
                        self.pipeline_logger.save_image(depth_image, f"04_control_depth", step=control_step)
                        control_step += 1
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
        self.pipeline_logger.save_image(result, "05_generated_raw", step=10)

        # 6. Postprocess
        logger.info("Step 6: Postprocessing...")
        final_image = postprocess_image(result, original_size)
        logger.info(f"  Final size: {final_image.size}")
        self.pipeline_logger.save_image(final_image, "06_final_output", step=11)

        # Save metadata
        end_time = time.time()
        metadata = {
            "pipeline": "complex",
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "controlnet_conditioning_scale": controlnet_conditioning_scale,
            "ip_adapter_scale": ip_adapter_scale,
            "seed": seed if seed is not None else "random",
            "control_types": control_types,
            "has_reference_image": reference_image is not None,
            "original_size": list(original_size),
            "processed_size": list(processed_image.size),
            "processing_time_seconds": round(end_time - start_time, 2)
        }
        self.pipeline_logger.save_metadata(metadata)

        logger.info("=" * 60)
        logger.info("✅ Complex pipeline complete")
        logger.info("=" * 60)

        # End logging session
        self.pipeline_logger.end_session()

        return final_image
