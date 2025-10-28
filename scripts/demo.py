"""Demo script to test Imageine API functionality."""
import requests
import base64
from PIL import Image
import io
import sys
from pathlib import Path

API_URL = "http://localhost:8000"


def encode_image(image_path: str) -> str:
    """
    Encode image file to base64 string.

    Args:
        image_path: Path to image file

    Returns:
        Base64 encoded image string
    """
    with Image.open(image_path) as img:
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()


def decode_image(b64_str: str) -> Image.Image:
    """
    Decode base64 string to PIL Image.

    Args:
        b64_str: Base64 encoded image string

    Returns:
        PIL Image
    """
    image_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_data))


def check_api_health():
    """Check if API is running and healthy."""
    print("\n🔍 Checking API health...")

    try:
        response = requests.get(f"{API_URL}/api/v1/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✓ API Status: {data['status']}")
            print(
                f"✓ Models loaded: {', '.join(data['models_loaded']) if data['models_loaded'] else 'None'}")
            print(f"✓ GPU available: {data['gpu_available']}")
            if data['gpu_memory_free_mb']:
                print(f"✓ GPU memory free: {data['gpu_memory_free_mb']}MB")
            return True
        else:
            print(f"✗ API returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"✗ Cannot connect to API at {API_URL}")
        print("  Make sure the server is running: python -m src.main")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_generation(image_path: str, prompt: str, output_name: str = None):
    """
    Test image generation.

    Args:
        image_path: Path to input image
        prompt: Text prompt for editing
        output_name: Optional output filename
    """
    print(f"\n📸 Testing generation")
    print(f"   Input: {image_path}")
    print(f"   Prompt: '{prompt}'")

    # Check if image exists
    if not Path(image_path).exists():
        print(f"✗ Image not found: {image_path}")
        return False

    # Encode image
    try:
        image_b64 = encode_image(image_path)
    except Exception as e:
        print(f"✗ Failed to load image: {e}")
        return False

    # Make request
    print("   Sending request...")
    try:
        response = requests.post(
            f"{API_URL}/api/v1/generate",
            json={
                "base_image": image_b64,
                "prompt": prompt,
                "config": {
                    "num_inference_steps": 25,
                    "guidance_scale": 7.5,
                    "seed": 42
                }
            },
            timeout=1200
        )
    except requests.exceptions.Timeout:
        print("✗ Request timed out (>1200s)")
        return False
    except Exception as e:
        print(f"✗ Request failed: {e}")
        return False

    # Handle response
    if response.status_code == 200:
        data = response.json()

        if data["success"]:
            result = data["result"]
            print(f"✓ Success!")
            print(f"   Pipeline: {result['pipeline_used']}")
            print(
                f"   Time: {result['processing_time_ms']}ms ({result['processing_time_ms']/1000:.1f}s)")
            print(
                f"   Resolution: {result['metadata']['input_resolution']} → {result['metadata']['output_resolution']}")

            # Save result
            result_image = decode_image(result["image"])

            if output_name is None:
                output_name = f"output_{prompt.replace(' ', '_')[:30]}.png"

            result_image.save(output_name)
            print(f"✓ Saved to: {output_name}")
            return True
        else:
            error = data["error"]
            print(f"✗ Generation failed")
            print(f"   Error code: {error['code']}")
            print(f"   Message: {error['message']}")
            return False
    else:
        print(f"✗ HTTP Error: {response.status_code}")
        try:
            print(f"   Response: {response.json()}")
        except:
            print(f"   Response: {response.text}")
        return False


def create_test_image():
    """Create a simple test image."""
    print("\n📝 Creating test image...")
    test_image = Image.new('RGB', (512, 512), color=(255, 100, 100))
    test_path = "test_input.png"
    test_image.save(test_path)
    print(f"✓ Created: {test_path}")
    return test_path


def main():
    """Run demo tests."""
    print("=" * 60)
    print("🚀 Imageine API Demo")
    print("=" * 60)

    # Check API health
    if not check_api_health():
        print("\n❌ API is not available. Exiting.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Running test generations...")
    print("=" * 60)

    # Create a test image
    test_image_path = create_test_image()

    # Test cases
    test_cases = [
        # (test_image_path, "blue image", "output_blue.png"),
        # (test_image_path, "green gradient background", "output_gradient.png"),
    ]

    # If user provides their own test images, add them
    custom_tests = [
        ("test_images/car.jpg", "make the car blue", "output_blue_car.png"),
        ("test_images/room.jpg", "change the chair to a herman miller chair",
         "output_modern_room.png"),
    ]

    # Try custom tests if images exist
    for image_path, prompt, output in custom_tests:
        if Path(image_path).exists():
            test_cases.append((image_path, prompt, output))

    # Run tests
    successes = 0
    for image_path, prompt, output_name in test_cases:
        if test_generation(image_path, prompt, output_name):
            successes += 1
        print()

    # Summary
    print("=" * 60)
    print(f"✅ Demo Complete: {successes}/{len(test_cases)} tests passed")
    print("=" * 60)

    if successes == len(test_cases):
        print("\n🎉 All tests passed! Your Imageine API is working perfectly.")
    elif successes > 0:
        print(f"\n⚠️  Some tests failed. Check the errors above.")
    else:
        print("\n❌ All tests failed. Please check:")
        print("  1. Is the server running? (python -m src.main)")
        print("  2. Are models downloaded? (python scripts/download_models.py)")
        print("  3. Is CUDA available? (python -c 'import torch; print(torch.cuda.is_available())')")


if __name__ == "__main__":
    main()
