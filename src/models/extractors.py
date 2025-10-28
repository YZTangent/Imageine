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

        try:
            from controlnet_aux import OpenposeDetector

            self._model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            logger.info("✓ Pose model loaded (OpenPose)")

        except ImportError:
            logger.warning("controlnet_aux not available, trying DWPose...")

            try:
                # Try alternative pose detector
                from controlnet_aux import DWposeDetector

                self._model = DWposeDetector.from_pretrained("lllyasviel/ControlNet")
                logger.info("✓ Pose model loaded (DWPose)")

            except Exception as e:
                logger.error(f"Failed to load pose model: {e}")
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

        try:
            if self.model_type == "depth-anything":
                from transformers import pipeline

                self._model = pipeline(
                    "depth-estimation",
                    model="depth-anything/Depth-Anything-V2-Small-hf",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("✓ Depth-Anything model loaded")

            elif self.model_type == "midas":
                from controlnet_aux import MidasDetector

                self._model = MidasDetector.from_pretrained("lllyasviel/ControlNet")
                logger.info("✓ MiDaS model loaded")

            else:
                raise ValueError(f"Unknown depth model type: {self.model_type}")

        except ImportError as e:
            logger.error(f"Required library not available: {e}")
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load depth model: {e}")
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
            if self.model_type == "depth-anything":
                # Depth-Anything returns a dict with 'depth' and 'predicted_depth'
                result = self._model(image)
                depth_map = result["depth"]
                logger.debug("Depth extracted successfully (Depth-Anything)")
                return depth_map

            elif self.model_type == "midas":
                # MiDaS from controlnet_aux
                depth_map = self._model(image)
                logger.debug("Depth extracted successfully (MiDaS)")
                return depth_map

        except Exception as e:
            logger.error(f"Depth extraction failed: {e}")
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
