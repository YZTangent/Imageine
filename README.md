# Imageine

General-purpose image composition API powered by Stable Diffusion. Transform images with text prompts - change colors, swap objects, try on clothes, and more.

## Features

- **Fast & Efficient**: Optimized for mid-tier consumer GPUs (RTX 3060+)
- **General Purpose**: Works for any image editing task via text prompts
- **Simple API**: REST API with easy-to-use endpoints
- **Docker Ready**: One-command deployment
- **Well Documented**: Interactive API docs included

## Quick Start

### Prerequisites

- Python 3.10+
- **GPU** (recommended, 8GB+ VRAM):
  - Apple Silicon (M1/M2/M3) - MPS acceleration ✅
  - NVIDIA GPU with CUDA support ✅
  - CPU-only mode also supported (slower)
- 20GB disk space for models

### Option 1: Local Development

1. **Clone and setup**:
```bash
cd imageine
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Download models** (one-time, ~5GB download):
```bash
python scripts/download_models.py
```

3. **Run the server**:
```bash
python -m src.main
# Or: uvicorn src.main:app --reload
```

4. **Test it**:
```bash
# Health check
curl http://localhost:8000/api/v1/health

# Interactive docs
open http://localhost:8000/docs
```

### Option 2: Docker (Coming Soon)

```bash
docker-compose up -d
```

## Usage Examples

### Python Client

```python
import requests
import base64
from PIL import Image
import io

# Load and encode image
with Image.open("car.jpg") as img:
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_b64 = base64.b64encode(buffered.getvalue()).decode()

# Make request
response = requests.post("http://localhost:8000/api/v1/generate", json={
    "base_image": img_b64,
    "prompt": "blue sports car",
    "config": {
        "num_inference_steps": 25,
        "guidance_scale": 7.5,
        "seed": 42
    }
})

# Save result
if response.json()["success"]:
    result_b64 = response.json()["result"]["image"]
    result_img = Image.open(io.BytesIO(base64.b64decode(result_b64)))
    result_img.save("result.png")
    print(" Saved to result.png")
```

### cURL

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "base_image": "<base64_encoded_image>",
    "prompt": "blue car",
    "config": {
      "num_inference_steps": 25,
      "guidance_scale": 7.5
    }
  }'
```

## Use Cases

- **Color Changes**: "red car" -> blue car
- **Style Transfer**: "modern minimalist interior"
- **Object Swaps**: "car with sport rims"
- **Virtual Try-On**: "person wearing leather jacket" (Phase 2)
- **Texture Modifications**: "wooden table with marble texture"

## API Endpoints

### `POST /api/v1/generate`
Generate edited image.

**Request**:
```json
{
  "base_image": "string (base64)",
  "prompt": "string",
  "negative_prompt": "string (optional)",
  "config": {
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    "seed": 42
  }
}
```

**Response**:
```json
{
  "success": true,
  "result": {
    "image": "string (base64)",
    "pipeline_used": "simple",
    "processing_time_ms": 8500,
    "metadata": { ... }
  }
}
```

### `GET /api/v1/health`
Check service health and GPU status.

### Full API Documentation
Visit `/docs` for interactive Swagger UI documentation.

## Configuration

Edit `config/default.yaml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8000

models:
  base_model: "stable-diffusion-v1-5/stable-diffusion-inpainting"
  device: "auto"  # auto-detect: mps (Apple), cuda (NVIDIA), or cpu
  dtype: "float16"

generation:
  default_steps: 25
  default_guidance_scale: 7.5
```

**Device Options**:
- `"auto"` - Auto-detect best available device (recommended) ⭐
- `"mps"` - Force Apple Silicon GPU (M1/M2/M3)
- `"cuda"` - Force NVIDIA GPU
- `"cpu"` - Force CPU (slowest)

## Performance

### Expected Latency (RTX 3090)
- Simple edits: 5-10 seconds
- Resolution: 512x512 � same as input

### Memory Usage
- Minimum: 6GB VRAM
- Recommended: 8GB+ VRAM

## Development Roadmap

- **Phase 1 (MVP)**: Simple pipeline with inpainting
- **Phase 2**: ControlNet support for complex tasks (clothing try-on, 3D objects)
- **Phase 3**: Production features (async, scaling, monitoring)

See [plan.md](plan.md) for detailed implementation plan.

## Architecture

See [SPEC.md](SPEC.md) for complete technical specification.

## Troubleshooting

### "CUDA out of memory"
- Reduce image size
- Reduce `num_inference_steps`
- Close other GPU applications

### "Models not found"
Run `python scripts/download_models.py` to download models.

### Slow inference
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Install xformers: `pip install xformers` (if available for your system)

## License

See LICENSE file for details.

## Contributing

Contributions welcome! Please see CONTRIBUTING.md.

## Support

- Documentation: [SPEC.md](SPEC.md)
- Roadmap: [PLAN.md](PLAN.md)

---

**Built using Stable Diffusion, FastAPI, and PyTorch**
