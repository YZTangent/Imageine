"""Structure extractors for ControlNet (pose, depth, etc.)."""
import torch
import numpy as np
from PIL import Image
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PoseExtractor:
    """Extract pose keypoints from images."""

    def __init__(self, device: str = "cuda"):
        """
        Initialize pose extractor.

        Args:
            device: Device to run on
        """
        self.device = device
        self._model = None

        logger.info(f"PoseExtractor initialized (device={device})")

    def load_model(self):
        """Load pose estimation model."""
        if self._model is not None:
            return

        logger.info("Loading pose estimation model...")

        # Try multiple approaches for pose detection

        # Approach 1: Try DWPose (newer, better)
        try:
            from controlnet_aux import DWposeDetector

            logger.info("Attempting to load DWPose...")
            self._model = DWposeDetector.from_pretrained("lllyasviel/ControlNet")
            logger.info("✓ Pose model loaded (DWPose)")
            return

        except ImportError:
            logger.debug("DWPose not available in controlnet_aux")
        except Exception as e:
            logger.debug(f"DWPose loading failed: {e}")

        # Approach 2: Try OpenPose (classic)
        try:
            from controlnet_aux import OpenposeDetector

            logger.info("Attempting to load OpenPose...")
            self._model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            logger.info("✓ Pose model loaded (OpenPose)")
            return

        except ImportError:
            logger.debug("OpenPose not available in controlnet_aux")
        except Exception as e:
            logger.debug(f"OpenPose loading failed: {e}")

        # Approach 3: Try alternative package
        try:
            from controlnet_aux import OpenposeDetector as Openpose

            logger.info("Attempting alternative OpenPose loading...")
            self._model = Openpose.from_pretrained("lllyasviel/Annotators")
            logger.info("✓ Pose model loaded (OpenPose - alternative)")
            return

        except Exception as e:
            logger.debug(f"Alternative OpenPose failed: {e}")

        # All approaches failed
        logger.error("Failed to load any pose estimation model")
        logger.warning("Install controlnet_aux: pip install controlnet-aux")
        self._model = None

    def extract(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Extract pose from image.

        Args:
            image: Input PIL image

        Returns:
            Pose visualization image or None if extraction fails
        """
        self.load_model()

        if self._model is None:
            logger.warning("Pose model not available, returning None")
            return None

        try:
            # Extract pose
            pose_image = self._model(image)
            logger.debug("Pose extracted successfully")
            return pose_image

        except Exception as e:
            logger.error(f"Pose extraction failed: {e}")
            return None


class DepthExtractor:
    """Extract depth maps from images."""

    def __init__(self, device: str = "cuda", model_type: str = "depth-anything"):
        """
        Initialize depth extractor.

        Args:
            device: Device to run on
            model_type: Type of depth model (depth-anything, midas, etc.)
        """
        self.device = device
        self.model_type = model_type
        self._model = None

        logger.info(f"DepthExtractor initialized (device={device}, type={model_type})")

    def load_model(self):
        """Load depth estimation model."""
        if self._model is not None:
            return

        logger.info(f"Loading depth model ({self.model_type})...")

        # Try multiple approaches for depth estimation

        # Approach 1: Depth-Anything (preferred for quality)
        if self.model_type == "depth-anything":
            try:
                from transformers import pipeline

                logger.info("Attempting to load Depth-Anything-V2...")
                self._model = pipeline(
                    "depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("✓ Depth-Anything-V2 model loaded")
                return

            except Exception as e:
                logger.debug(f"Depth-Anything-V2 failed: {e}")

                # Try V1 as fallback
                try:
                    logger.info("Trying Depth-Anything V1...")
                    self._model = pipeline(
                        "depth-estimation",
                        model="LiheYoung/depth-anything-small-hf",
                        device=0 if self.device == "cuda" else -1
                    )
                    logger.info("✓ Depth-Anything V1 model loaded")
                    return
                except Exception as e2:
                    logger.debug(f"Depth-Anything V1 failed: {e2}")

        # Approach 2: MiDaS (alternative)
        if self.model_type == "midas" or self.model_type == "depth-anything":
            try:
                from controlnet_aux import MidasDetector

                logger.info("Attempting to load MiDaS...")
                self._model = MidasDetector.from_pretrained("lllyasviel/ControlNet")
                self.model_type = "midas"  # Update type
                logger.info("✓ MiDaS model loaded")
                return

            except ImportError:
                logger.debug("controlnet_aux not available for MiDaS")
            except Exception as e:
                logger.debug(f"MiDaS loading failed: {e}")

        # Approach 3: Try Intel's DPT model
        try:
            from transformers import pipeline

            logger.info("Attempting to load DPT depth model...")
            self._model = pipeline(
                "depth-estimation",
                model="Intel/dpt-large",
                device=0 if self.device == "cuda" else -1
            )
            self.model_type = "dpt"
            logger.info("✓ DPT depth model loaded")
            return

        except Exception as e:
            logger.debug(f"DPT model failed: {e}")

        # All approaches failed
        logger.error(f"Failed to load any depth estimation model")
        logger.warning("Try: pip install transformers[torch] controlnet-aux")
        self._model = None

    def extract(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Extract depth map from image.

        Args:
            image: Input PIL image

        Returns:
            Depth map as PIL image or None if extraction fails
        """
        self.load_model()

        if self._model is None:
            logger.warning("Depth model not available, returning None")
            return None

        try:
            if self.model_type in ["depth-anything", "dpt"]:
                # Pipeline-based depth models return a dict with 'depth' or 'predicted_depth'
                result = self._model(image)

                # Try to get depth map from result
                if isinstance(result, dict):
                    depth_map = result.get("depth") or result.get("predicted_depth")
                elif isinstance(result, Image.Image):
                    depth_map = result
                else:
                    logger.error(f"Unexpected result type: {type(result)}")
                    return None

                logger.debug(f"Depth extracted successfully ({self.model_type})")
                return depth_map

            elif self.model_type == "midas":
                # MiDaS from controlnet_aux
                depth_map = self._model(image)
                logger.debug("Depth extracted successfully (MiDaS)")
                return depth_map

            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return None

        except Exception as e:
            logger.error(f"Depth extraction failed: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None


class CannyExtractor:
    """Extract Canny edges from images."""

    def __init__(self, low_threshold: int = 100, high_threshold: int = 200):
        """
        Initialize Canny edge detector.

        Args:
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        logger.info("CannyExtractor initialized")

    def extract(self, image: Image.Image) -> Image.Image:
        """
        Extract Canny edges from image.

        Args:
            image: Input PIL image

        Returns:
            Edge map as PIL image
        """
        try:
            import cv2

            # Convert to numpy
            image_np = np.array(image)

            # Convert to grayscale
            if len(image_np.shape) == 3:
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_np

            # Apply Canny edge detection
            edges = cv2.Canny(gray, self.low_threshold, self.high_threshold)

            # Convert back to PIL
            edge_image = Image.fromarray(edges)

            logger.debug("Canny edges extracted successfully")
            return edge_image

        except Exception as e:
            logger.error(f"Canny extraction failed: {e}")
            # Return white image as fallback
            return Image.new('L', image.size, 255)
