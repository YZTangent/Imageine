"""Image processing utilities."""
from PIL import Image
import numpy as np
from typing import Tuple
import cv2


def preprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int] = (512, 512)
) -> Tuple[Image.Image, Tuple[int, int]]:
    """
    Preprocess image for model input.

    Args:
        image: Input PIL image
        target_size: Target size for model (width, height)

    Returns:
        Tuple of (processed image, original size)
    """
    original_size = image.size

    # Convert RGBA to RGB
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')

    # Resize maintaining aspect ratio
    image.thumbnail(target_size, Image.LANCZOS)

    # Pad to exact size
    padded = Image.new('RGB', target_size, (255, 255, 255))
    paste_pos = (
        (target_size[0] - image.size[0]) // 2,
        (target_size[1] - image.size[1]) // 2
    )
    padded.paste(image, paste_pos)

    return padded, original_size


def postprocess_image(
    image: Image.Image,
    target_size: Tuple[int, int]
) -> Image.Image:
    """
    Upscale image back to target size.

    Args:
        image: Generated image
        target_size: Original size (width, height)

    Returns:
        Resized image
    """
    return image.resize(target_size, Image.LANCZOS)


def create_mask_from_prompt(
    image: Image.Image,
    prompt: str,
    detector=None
) -> Image.Image:
    """
    Create mask for inpainting.

    For MVP: return full white mask (edit entire image)
    Later: use GroundingDINO for targeted masking

    Args:
        image: Input image
        prompt: Text prompt describing what to edit
        detector: Optional detector for targeted masking

    Returns:
        Mask image (L mode, 0-255)
    """
    if detector is None:
        # Full image mask for MVP - edit everything
        return Image.new('L', image.size, 255)

    # TODO: Use detector for targeted masking in Phase 2
    mask_array = detector.detect(image, prompt)
    return Image.fromarray((mask_array * 255).astype(np.uint8))


def apply_mask_blur(
    mask: Image.Image,
    blur_radius: int = 21
) -> Image.Image:
    """
    Apply Gaussian blur to mask for smooth edges.

    Args:
        mask: Input mask image
        blur_radius: Blur kernel radius (must be odd)

    Returns:
        Blurred mask
    """
    # Ensure blur_radius is odd
    if blur_radius % 2 == 0:
        blur_radius += 1

    mask_array = np.array(mask)
    blurred = cv2.GaussianBlur(mask_array, (blur_radius, blur_radius), 0)
    return Image.fromarray(blurred)


def dilate_mask(
    mask: Image.Image,
    dilation_pixels: int = 10
) -> Image.Image:
    """
    Dilate mask to include context around edges.

    Args:
        mask: Input mask
        dilation_pixels: Number of pixels to dilate

    Returns:
        Dilated mask
    """
    mask_array = np.array(mask)
    kernel = np.ones((dilation_pixels, dilation_pixels), np.uint8)
    dilated = cv2.dilate(mask_array, kernel, iterations=1)
    return Image.fromarray(dilated)
