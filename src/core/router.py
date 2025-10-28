"""Task analyzer and router for intelligent pipeline selection."""
import re
from typing import Tuple, List, Optional
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class TaskRouter:
    """Analyzes tasks and routes to appropriate pipeline."""

    # Keywords that trigger complex pipeline
    COMPLEX_KEYWORDS = {
        "person": ["try on", "wearing", "person", "human", "body", "pose", "outfit", "clothes", "shirt", "pants", "dress", "jacket"],
        "3d_object": ["car", "vehicle", "automobile", "furniture", "chair", "table", "sofa", "room"],
        "structure": ["perspective", "3d", "depth", "spatial"]
    }

    # Keywords that suggest simple pipeline
    SIMPLE_KEYWORDS = ["color", "style", "texture", "pattern", "gradient", "tone", "hue"]

    def __init__(self, config=None):
        """
        Initialize task router.

        Args:
            config: Optional configuration object
        """
        self.config = config
        self.confidence_threshold = 0.7
        logger.info("TaskRouter initialized")

    def analyze_prompt(self, prompt: str) -> dict:
        """
        Analyze prompt for complexity indicators.

        Args:
            prompt: Text prompt

        Returns:
            Dictionary with analysis results
        """
        prompt_lower = prompt.lower()

        # Check for complex keywords
        person_score = 0
        object_3d_score = 0
        structure_score = 0
        simple_score = 0

        for keyword in self.COMPLEX_KEYWORDS["person"]:
            if keyword in prompt_lower:
                person_score += 1

        for keyword in self.COMPLEX_KEYWORDS["3d_object"]:
            if keyword in prompt_lower:
                object_3d_score += 1

        for keyword in self.COMPLEX_KEYWORDS["structure"]:
            if keyword in prompt_lower:
                structure_score += 1

        for keyword in self.SIMPLE_KEYWORDS:
            if keyword in prompt_lower:
                simple_score += 1

        return {
            "person_score": person_score,
            "object_3d_score": object_3d_score,
            "structure_score": structure_score,
            "simple_score": simple_score,
            "prompt": prompt
        }

    def detect_content_type(self, image: Optional[Image.Image] = None) -> str:
        """
        Detect content type in image.

        Args:
            image: Input image (optional for Phase 2)

        Returns:
            Content type (person, object, scene, unknown)
        """
        # TODO: Implement actual object detection with YOLO or similar
        # For now, return unknown
        return "unknown"

    def route(
        self,
        prompt: str,
        base_image: Optional[Image.Image] = None,
        reference_image: Optional[Image.Image] = None,
        force_complex: bool = False
    ) -> Tuple[str, List[str]]:
        """
        Route task to appropriate pipeline.

        Args:
            prompt: Text prompt
            base_image: Base image (optional, for content detection)
            reference_image: Reference image (optional)
            force_complex: Force complex pipeline

        Returns:
            Tuple of (pipeline_type, control_types)
            - pipeline_type: "simple" or "complex"
            - control_types: List of control types for complex pipeline
        """
        logger.info(f"Routing task: '{prompt[:50]}...'")

        # Force complex if requested
        if force_complex:
            logger.info("  → Forced complex pipeline")
            return ("complex", ["pose"])

        # Analyze prompt
        analysis = self.analyze_prompt(prompt)

        # Decision logic
        total_complex_score = (
            analysis["person_score"] +
            analysis["object_3d_score"] +
            analysis["structure_score"]
        )
        simple_score = analysis["simple_score"]

        logger.debug(f"  Scores: complex={total_complex_score}, simple={simple_score}")

        # Determine pipeline
        if total_complex_score > 0:
            # Complex pipeline needed
            control_types = []

            # Determine which controls to use
            if analysis["person_score"] > 0:
                control_types.append("pose")
                logger.info(f"  → Complex pipeline (pose control)")
                logger.info(f"    Reason: Person-related keywords detected")

            elif analysis["object_3d_score"] > 0 or analysis["structure_score"] > 0:
                control_types.append("depth")
                logger.info(f"  → Complex pipeline (depth control)")
                logger.info(f"    Reason: 3D object/structure keywords detected")

            return ("complex", control_types if control_types else ["pose"])

        else:
            # Simple pipeline sufficient
            logger.info(f"  → Simple pipeline")
            if simple_score > 0:
                logger.info(f"    Reason: Simple modification keywords detected")
            else:
                logger.info(f"    Reason: No complexity indicators found (default)")

            return ("simple", [])

    def should_use_ip_adapter(
        self,
        reference_image: Optional[Image.Image],
        prompt: str
    ) -> bool:
        """
        Determine if IP-Adapter should be used.

        Args:
            reference_image: Reference image
            prompt: Text prompt

        Returns:
            True if IP-Adapter should be used
        """
        if reference_image is None:
            return False

        # Check if prompt references the image
        reference_keywords = ["this", "these", "like", "similar", "same"]

        prompt_lower = prompt.lower()
        for keyword in reference_keywords:
            if keyword in prompt_lower:
                return True

        # Default: use IP-Adapter if reference provided
        return True
