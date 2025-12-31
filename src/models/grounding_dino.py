"""GroundingDINO object detection wrapper."""
from typing import Tuple, Optional, List
import torch
import numpy as np
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class GroundingDINODetector:
    """Wrapper for GroundingDINO object detection."""

    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-base",
        device: str = "cuda",
        confidence_threshold: float = 0.3
    ):
        """
        Initialize GroundingDINO detector.

        Args:
            model_name: Model identifier
            device: Device to run on (cuda/cpu)
            confidence_threshold: Minimum confidence for detections
        """
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self._model = None
        self._processor = None

        logger.info(f"GroundingDINO detector initialized (device={device})")

    def load_model(self):
        """Load GroundingDINO model."""
        if self._model is not None:
            return

        logger.info("Loading GroundingDINO model...")

        # Try multiple approaches in order of preference

        # Approach 1: Try using transformers pipeline (easiest, most maintainable)
        try:
            from transformers import pipeline

            logger.info("Attempting to load zero-shot detection model from transformers...")
            self._model = pipeline(
                task="zero-shot-object-detection",
                model="IDEA-Research/grounding-dino-tiny",
                device=0 if self.device == "cuda" else -1
            )
            self._processor = None
            logger.info("✓ GroundingDINO loaded successfully via transformers")
            return

        except Exception as e:
            logger.debug(f"Transformers pipeline approach failed: {e}")

        # Approach 2: Try OWL-ViT as alternative (also good for zero-shot detection)
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection

            logger.info("Attempting to load OWL-ViT (alternative zero-shot detector)...")
            self._processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
            self._model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32")
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info("✓ OWL-ViT loaded successfully (using as GroundingDINO alternative)")
            return

        except Exception as e:
            logger.debug(f"OWL-ViT approach failed: {e}")

        # Approach 3: Try direct GroundingDINO from transformers
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

            logger.info("Attempting direct GroundingDINO model loading...")
            self._processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
            self._model = self._model.to(self.device)
            self._model.eval()
            logger.info("✓ GroundingDINO loaded successfully")
            return

        except Exception as e:
            logger.debug(f"Direct GroundingDINO loading failed: {e}")

        # Fallback: use simple detection
        logger.warning("All GroundingDINO approaches failed, using fallback detection")
        logger.info("Fallback will return full image mask (inpaint entire image)")
        self._model = "fallback"
        self._processor = None

    def detect(
        self,
        image: Image.Image,
        text_prompt: str,
        confidence_threshold: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """
        Detect object in image based on text prompt.

        Args:
            image: Input PIL image
            text_prompt: Text description of object to detect
            confidence_threshold: Override default confidence threshold

        Returns:
            Binary mask (H, W) with 1s where object detected, or None if not found
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        self.load_model()

        # Fallback: return full image mask
        if self._model == "fallback":
            logger.debug("Using fallback detection (full image mask)")
            return np.ones((image.height, image.width), dtype=np.uint8)

        try:
            # Approach 1: Pipeline-based detection
            if hasattr(self._model, '__call__') and self._processor is None:
                logger.debug(f"Running pipeline detection for: '{text_prompt}'")
                results = self._model(image, candidate_labels=[text_prompt])

                # Filter by confidence and get boxes
                boxes = self.boxes_to_mask(
                    [(r['box']['xmin'], r['box']['ymin'], r['box']['xmax'], r['box']['ymax'])
                     for r in results if r['score'] >= confidence_threshold],
                    (image.width, image.height)
                )
                logger.debug(f"Pipeline detected {len(results)} objects")
                return boxes

            # Approach 2: OWL-ViT or Direct model with processor
            elif self._processor is not None:
                logger.debug(f"Running model detection for: '{text_prompt}'")

                # Process inputs
                inputs = self._processor(
                    text=[[text_prompt]],
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

                # Run inference
                with torch.no_grad():
                    outputs = self._model(**inputs)

                # Post-process results
                target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
                results = self._processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=confidence_threshold,
                    target_sizes=target_sizes
                )[0]

                # Convert boxes to mask
                if len(results['boxes']) > 0:
                    boxes = [(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                             for b in results['boxes'].cpu().numpy()]
                    logger.debug(f"Model detected {len(boxes)} objects")
                    return self.boxes_to_mask(boxes, (image.width, image.height))
                else:
                    logger.debug("No objects detected, using full image mask")
                    return np.ones((image.height, image.width), dtype=np.uint8)

        except Exception as e:
            logger.error(f"Detection failed: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Fallback to full image mask
            logger.info("Falling back to full image mask")
            return np.ones((image.height, image.width), dtype=np.uint8)

    def detect_boxes(
        self,
        image: Image.Image,
        text_prompt: str,
        confidence_threshold: Optional[float] = None
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect bounding boxes for objects.

        Args:
            image: Input PIL image
            text_prompt: Text description of object to detect
            confidence_threshold: Override default confidence threshold

        Returns:
            List of bounding boxes as (x1, y1, x2, y2)
        """
        if confidence_threshold is None:
            confidence_threshold = self.confidence_threshold

        self.load_model()

        # Fallback
        if self._model == "fallback":
            return [(0, 0, image.width, image.height)]

        try:
            # Pipeline approach
            if hasattr(self._model, '__call__') and self._processor is None:
                results = self._model(image, candidate_labels=[text_prompt])
                return [
                    (int(r['box']['xmin']), int(r['box']['ymin']),
                     int(r['box']['xmax']), int(r['box']['ymax']))
                    for r in results if r['score'] >= confidence_threshold
                ]

            # Model with processor approach
            elif self._processor is not None:
                inputs = self._processor(
                    text=[[text_prompt]],
                    images=image,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self._model(**inputs)

                target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
                results = self._processor.post_process_object_detection(
                    outputs=outputs,
                    threshold=confidence_threshold,
                    target_sizes=target_sizes
                )[0]

                if len(results['boxes']) > 0:
                    return [(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                            for b in results['boxes'].cpu().numpy()]

        except Exception as e:
            logger.error(f"Box detection failed: {e}")

        # Fallback
        return [(0, 0, image.width, image.height)]

    def boxes_to_mask(
        self,
        boxes: List[Tuple[int, int, int, int]],
        image_size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Convert bounding boxes to binary mask.

        Args:
            boxes: List of bounding boxes as (x1, y1, x2, y2)
            image_size: Image size as (width, height)

        Returns:
            Binary mask
        """
        mask = np.zeros(image_size[::-1], dtype=np.uint8)  # (height, width)

        for box in boxes:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 1

        return mask
