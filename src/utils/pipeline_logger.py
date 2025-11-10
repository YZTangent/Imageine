"""Utility for logging pipeline intermediate outputs."""
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class PipelineLogger:
    """Logger for saving pipeline intermediate outputs."""

    def __init__(self, output_dir: str = "outputs", enabled: bool = True):
        """
        Initialize pipeline logger.

        Args:
            output_dir: Base directory for outputs
            enabled: Whether logging is enabled
        """
        self.output_dir = output_dir
        self.enabled = enabled
        self.session_dir = None
        self.step_counter = 0

        if self.enabled:
            # Create base output directory
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
            logger.info(f"PipelineLogger initialized (output_dir={output_dir})")

    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new logging session.

        Args:
            session_id: Optional session ID (defaults to timestamp)

        Returns:
            Session directory path
        """
        if not self.enabled:
            return ""

        if session_id is None:
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        self.session_dir = os.path.join(self.output_dir, session_id)
        Path(self.session_dir).mkdir(parents=True, exist_ok=True)
        self.step_counter = 0

        logger.info(f"Started logging session: {self.session_dir}")
        return self.session_dir

    def save_image(
        self,
        image: Image.Image,
        name: str,
        step: Optional[int] = None
    ) -> Optional[str]:
        """
        Save an image to the current session.

        Args:
            image: PIL Image to save
            name: Name for the image (without extension)
            step: Optional step number (auto-increments if not provided)

        Returns:
            Path to saved image or None if logging disabled
        """
        if not self.enabled or self.session_dir is None:
            return None

        if step is None:
            step = self.step_counter
            self.step_counter += 1

        filename = f"step_{step:02d}_{name}.png"
        filepath = os.path.join(self.session_dir, filename)

        image.save(filepath)
        logger.debug(f"Saved image: {filename}")

        return filepath

    def save_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str = "metadata.json"
    ) -> Optional[str]:
        """
        Save metadata to JSON file.

        Args:
            metadata: Dictionary of metadata
            filename: Name of the JSON file

        Returns:
            Path to saved metadata or None if logging disabled
        """
        if not self.enabled or self.session_dir is None:
            return None

        filepath = os.path.join(self.session_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Saved metadata: {filename}")
        return filepath

    def save_text(
        self,
        text: str,
        filename: str = "info.txt"
    ) -> Optional[str]:
        """
        Save text to file.

        Args:
            text: Text content to save
            filename: Name of the text file

        Returns:
            Path to saved file or None if logging disabled
        """
        if not self.enabled or self.session_dir is None:
            return None

        filepath = os.path.join(self.session_dir, filename)

        with open(filepath, 'w') as f:
            f.write(text)

        logger.debug(f"Saved text: {filename}")
        return filepath

    def end_session(self) -> None:
        """End the current logging session."""
        if self.enabled and self.session_dir:
            logger.info(f"Ended logging session: {self.session_dir}")
            self.session_dir = None
            self.step_counter = 0
