"""Download and cache all required models."""
from diffusers import StableDiffusionInpaintPipeline
from transformers import CLIPProcessor, CLIPModel
import torch
from pathlib import Path
import sys


def download_models(cache_dir: str = "./models_cache"):
    """
    Download and cache all required models.

    Args:
        cache_dir: Directory to cache models
    """
    print("=" * 60)
    print("Imageine - Model Downloader")
    print("=" * 60)
    print(f"\n📦 Downloading models to: {cache_dir}")

    Path(cache_dir).mkdir(exist_ok=True)

    try:
        # 1. Stable Diffusion Inpainting
        print("\n[1/2] Downloading SD 1.5 Inpainting...")
        print("This may take a while (~5GB download)...")
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stable-diffusion-v1-5/stable-diffusion-inpainting",
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
        print("✓ SD 1.5 Inpainting downloaded successfully")

        # 2. CLIP (for routing)
        print("\n[2/2] Downloading CLIP...")
        model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir
        )
        processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            cache_dir=cache_dir
        )
        print("✓ CLIP downloaded successfully")

        print("\n" + "=" * 60)
        print("✅ All models downloaded successfully!")
        print("=" * 60)
        print(f"\nModels cached in: {Path(cache_dir).absolute()}")
        print("You can now run the API server.")

    except Exception as e:
        print(f"\n❌ Error downloading models: {e}", file=sys.stderr)
        print("\nTroubleshooting:")
        print("  1. Check your internet connection")
        print("  2. Ensure you have enough disk space (~10GB)")
        print("  3. Try running with: python scripts/download_models.py")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download Imageine models")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./models_cache",
        help="Directory to cache models (default: ./models_cache)"
    )
    args = parser.parse_args()

    download_models(args.cache_dir)
