"""Test image processing utilities."""
import pytest
from PIL import Image
from src.utils.image_processing import (
    preprocess_image,
    postprocess_image,
    create_mask_from_prompt,
    apply_mask_blur
)


def test_preprocess_image():
    """Test image preprocessing."""
    # Create test image
    img = Image.new('RGB', (1024, 768), color=(255, 0, 0))

    # Preprocess
    processed, original_size = preprocess_image(img, target_size=(512, 512))

    assert processed.size == (512, 512)
    assert original_size == (1024, 768)
    assert processed.mode == 'RGB'


def test_preprocess_rgba_image():
    """Test RGBA to RGB conversion."""
    img = Image.new('RGBA', (512, 512), color=(255, 0, 0, 255))

    processed, _ = preprocess_image(img)

    assert processed.mode == 'RGB'


def test_postprocess_image():
    """Test image postprocessing."""
    img = Image.new('RGB', (512, 512), color=(0, 255, 0))

    resized = postprocess_image(img, target_size=(1024, 768))

    assert resized.size == (1024, 768)


def test_create_mask_from_prompt():
    """Test mask creation."""
    img = Image.new('RGB', (512, 512))

    mask = create_mask_from_prompt(img, "test prompt")

    assert mask.mode == 'L'
    assert mask.size == img.size


def test_apply_mask_blur():
    """Test mask blurring."""
    mask = Image.new('L', (512, 512), 255)

    blurred = apply_mask_blur(mask, blur_radius=21)

    assert blurred.mode == 'L'
    assert blurred.size == mask.size
