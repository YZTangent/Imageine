# Pipeline Logging Documentation

This document explains the comprehensive logging system for the Imageine image composition pipeline. The logging system captures all intermediate steps, outputs, and metadata to help with debugging, analysis, and quality assessment.

## Overview

The pipeline logging system automatically saves:
- All intermediate images (preprocessed, masks, control images, etc.)
- Final output images
- Complete metadata (parameters, timing, pipeline configuration)
- Timestamped sessions for easy organization

## Configuration

### Enable/Disable Logging

Logging is controlled via `config/default.yaml`:

```yaml
logging:
  enabled: true                      # Master switch for all logging
  output_dir: "./pipeline_outputs"   # Where to save outputs
  save_intermediates: true           # Whether to save intermediate steps
```

**Configuration Options:**
- `enabled`: Master switch - if `false`, no logging occurs
- `output_dir`: Base directory for all pipeline outputs
- `save_intermediates`: If `false`, only final outputs are logged (not yet implemented)

## Output Structure

Each generation creates a timestamped directory:

```
pipeline_outputs/
└── 20250109_143052_123456/          # Timestamp: YYYYMMDD_HHMMSS_microseconds
    ├── step_00_00_input_original.png
    ├── step_01_01_preprocessed.png
    ├── step_02_02_mask_raw.png
    ├── step_03_03_mask_blurred.png
    ├── step_04_04_control_pose.png   # Only in complex pipeline
    ├── step_05_05_generated_raw.png
    ├── step_06_06_final_output.png
    └── metadata.json
```

**Filename Format:** `step_{number:02d}_{name}.png`
- Files are named with step numbers for easy sorting
- Descriptive names indicate what each file contains

## Simple Pipeline Logging

**File:** `src/core/simple_pipeline.py`

The simple pipeline (fast path without ControlNet) logs **6 steps**:

### Step 0: Original Input
**File:** `step_00_00_input_original.png`
- The raw input image as provided by the user
- Before any preprocessing or transformations

### Step 1: Preprocessed Image
**File:** `step_01_01_preprocessed.png`
- Image after preprocessing (resize, padding, color conversion)
- Resized to 512x512 (SD 1.5 input size)
- RGBA → RGB conversion if needed
- Aspect ratio maintained with white padding

### Step 2: Raw Mask
**File:** `step_02_02_mask_raw.png`
- Binary mask created from prompt
- In current implementation: full white mask (edit entire image)
- Future: GroundingDINO-based targeted masking
- Format: Grayscale (L mode), 0-255

### Step 3: Blurred Mask
**File:** `step_03_03_mask_blurred.png`
- Mask after Gaussian blur (radius=21)
- Softens edges for seamless blending
- Prevents hard boundaries in generation

### Step 4: Generated Image (Raw)
**File:** `step_04_04_generated_raw.png`
- Direct output from Stable Diffusion inpainting
- Before postprocessing
- Still at 512x512 resolution

### Step 5: Final Output
**File:** `step_05_05_final_output.png`
- Final image after postprocessing
- Resized back to original input dimensions
- Ready for delivery to user

### Metadata (Simple Pipeline)

**File:** `metadata.json`

```json
{
  "pipeline": "simple",
  "prompt": "blue car",
  "negative_prompt": "deformed, distorted, low quality, blurry, bad anatomy",
  "num_inference_steps": 25,
  "guidance_scale": 7.5,
  "seed": 42,
  "original_size": [1024, 768],
  "processed_size": [512, 512],
  "processing_time_seconds": 8.45
}
```

## Complex Pipeline Logging

**File:** `src/core/complex_pipeline.py`

The complex pipeline (with ControlNet support) logs **8+ steps**:

### Step 0: Original Input(s)
**Files:**
- `step_00_00_input_original.png` - Base image
- `step_00_00_reference_image.png` - Reference image (if provided)

### Step 1: Preprocessed Image
**File:** `step_01_01_preprocessed.png`
- Same as simple pipeline
- Base image preprocessed to 512x512

### Step 2: Raw Mask
**File:** `step_02_02_mask_raw.png`
- Mask from GroundingDINO detector
- Targeted object detection based on prompt
- Binary mask of detected regions

### Step 3: Blurred Mask
**File:** `step_03_03_mask_blurred.png`
- Smoothed mask for seamless inpainting
- Gaussian blur applied

### Steps 4+: Control Images
**Files:** (variable number based on control types)
- `step_04_04_control_pose.png` - Pose keypoints/skeleton (if pose control requested)
- `step_05_04_control_depth.png` - Depth map (if depth control requested)

**Control Types:**
- **Pose:** Human pose estimation using OpenPose/DWPose
  - Skeleton visualization with keypoints
  - Used for clothing try-on, pose preservation
- **Depth:** Monocular depth estimation using Depth-Anything
  - Grayscale depth map (near=dark, far=bright)
  - Used for 3D structure preservation (cars, furniture, etc.)

### Step 10: Generated Image (Raw)
**File:** `step_05_05_generated_raw.png`
- Direct output from ControlNet inpainting
- Combined conditioning from:
  - Text prompt
  - Mask
  - Control images (pose/depth)
  - Reference image (via IP-Adapter, if provided)

### Step 11: Final Output
**File:** `step_06_06_final_output.png`
- Final postprocessed result
- Resized to original dimensions

### Metadata (Complex Pipeline)

**File:** `metadata.json`

```json
{
  "pipeline": "complex",
  "prompt": "person wearing this jacket",
  "negative_prompt": "deformed, distorted, low quality, blurry, bad anatomy",
  "num_inference_steps": 30,
  "guidance_scale": 7.5,
  "controlnet_conditioning_scale": 1.0,
  "ip_adapter_scale": 0.6,
  "seed": 42,
  "control_types": ["pose"],
  "has_reference_image": true,
  "original_size": [1024, 768],
  "processed_size": [512, 512],
  "processing_time_seconds": 15.23
}
```

## Implementation Details

### PipelineLogger Class

**File:** `src/utils/pipeline_logger.py`

The `PipelineLogger` class manages all logging operations:

```python
from src.utils.pipeline_logger import PipelineLogger

# Initialize
logger = PipelineLogger(
    output_dir="./pipeline_outputs",
    enabled=True
)

# Start session (creates timestamped directory)
session_dir = logger.start_session()

# Save images
logger.save_image(image, name="input_original", step=0)

# Save metadata
logger.save_metadata({"prompt": "...", "seed": 42})

# End session
logger.end_session()
```

**Key Methods:**

- `start_session(session_id=None)` - Creates new timestamped directory
- `save_image(image, name, step=None)` - Saves PIL Image to PNG
- `save_metadata(metadata, filename="metadata.json")` - Saves dict to JSON
- `save_text(text, filename)` - Saves text content to file
- `end_session()` - Cleanup and logging

### Integration with Pipelines

Both pipelines initialize the logger in their `__init__` method:

```python
def __init__(self, model_manager: ModelManager, config):
    self.model_manager = model_manager
    self.config = config

    # Initialize pipeline logger
    log_config = config.logging
    self.pipeline_logger = PipelineLogger(
        output_dir=log_config.output_dir,
        enabled=log_config.enabled and log_config.save_intermediates
    )
```

Logging is integrated at each pipeline step:

```python
# Start session
self.pipeline_logger.start_session()

# Save at each step
self.pipeline_logger.save_image(original_image, "00_input_original", step=0)
self.pipeline_logger.save_image(processed_image, "01_preprocessed", step=1)
# ... more steps ...

# Save metadata
self.pipeline_logger.save_metadata(metadata)

# End session
self.pipeline_logger.end_session()
```

## Use Cases

### 1. Debugging Pipeline Issues

When generation fails or produces unexpected results:

1. Check the `metadata.json` for parameters used
2. Examine intermediate images to identify where issues occur:
   - Is the mask correct? (steps 2-3)
   - Are control images extracted properly? (steps 4+)
   - Does the raw generation look correct? (step 10/4)

### 2. Quality Assessment

Compare intermediate outputs across generations:
- Mask quality and coverage
- Control image accuracy (pose detection, depth estimation)
- Effect of different parameters on output

### 3. Performance Analysis

Use `processing_time_seconds` in metadata to:
- Compare simple vs complex pipeline performance
- Identify bottlenecks
- Track performance over time

### 4. Dataset Creation

Logged outputs can be used to:
- Create training datasets
- Build evaluation benchmarks
- Document successful configurations

### 5. User Transparency

Share intermediate outputs with users to:
- Explain how the pipeline works
- Show what was detected/processed
- Build trust in the system

## Performance Considerations

### Storage Usage

**Per Generation (Approximate):**
- Simple Pipeline: ~10-15 MB (6 images + metadata)
- Complex Pipeline: ~15-25 MB (8+ images + metadata)

**Recommendations:**
- Implement automatic cleanup (delete old sessions after N days)
- Add option to save only final output (not yet implemented)
- Compress images for long-term storage

### Performance Impact

Logging overhead is minimal:
- Image saving: ~10-50ms per image (depends on size)
- Total overhead: ~100-300ms per generation
- Relative impact: <5% of total generation time

### Disabling Logging

For production deployments where storage is limited:

```yaml
logging:
  enabled: false  # Disables all logging
```

Or programmatically:

```python
config.logging.enabled = False
```

## Future Enhancements

### Planned Features

1. **Selective Logging**
   - Save only failures/errors
   - Save only specific steps
   - Configurable step filtering

2. **Comparison View**
   - Side-by-side comparisons
   - Before/after overlays
   - Difference maps

3. **Automatic Cleanup**
   - Age-based deletion
   - Size-based limits
   - Archive old sessions

4. **Enhanced Metadata**
   - Model versions/hashes
   - GPU memory usage
   - Per-step timing breakdown

5. **Visualization Tools**
   - Web UI for browsing sessions
   - Automatic report generation
   - Statistical analysis

## Troubleshooting

### Logging Not Working

**Check:**
1. `config/default.yaml` has `logging.enabled: true`
2. Output directory is writable
3. Sufficient disk space available
4. No exceptions in logs

### Missing Intermediate Files

**Possible Causes:**
- Pipeline failed before saving that step
- Logging disabled mid-execution
- File permissions issue

**Solution:**
- Check application logs for errors
- Verify config settings
- Check `metadata.json` for partial information

### Large Disk Usage

**Solutions:**
1. Disable logging: `logging.enabled: false`
2. Implement cleanup script:
   ```bash
   # Delete sessions older than 7 days
   find ./pipeline_outputs -type d -mtime +7 -exec rm -rf {} +
   ```
3. Compress old sessions:
   ```bash
   tar -czf archive.tar.gz pipeline_outputs/20250109_*
   ```

## Examples

### Reading Logged Data

```python
import json
from PIL import Image
from pathlib import Path

# Load session
session_dir = Path("pipeline_outputs/20250109_143052_123456")

# Read metadata
with open(session_dir / "metadata.json") as f:
    metadata = json.load(f)
    print(f"Prompt: {metadata['prompt']}")
    print(f"Processing time: {metadata['processing_time_seconds']}s")

# Load images
original = Image.open(session_dir / "step_00_00_input_original.png")
mask = Image.open(session_dir / "step_03_03_mask_blurred.png")
output = Image.open(session_dir / "step_05_05_final_output.png")
```

### Creating Comparison Grids

```python
from PIL import Image
import matplotlib.pyplot as plt

# Load all steps
session_dir = Path("pipeline_outputs/20250109_143052_123456")
images = sorted(session_dir.glob("step_*.png"))

# Create grid
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, img_path in enumerate(images):
    if i >= 6:
        break
    img = Image.open(img_path)
    axes[i//3, i%3].imshow(img)
    axes[i//3, i%3].set_title(img_path.stem)
    axes[i//3, i%3].axis('off')

plt.tight_layout()
plt.savefig("pipeline_comparison.png")
```

## Summary

The pipeline logging system provides comprehensive visibility into every step of the image generation process:

- **Simple Pipeline:** 6 steps logged (input → preprocessing → masking → generation → output)
- **Complex Pipeline:** 8+ steps logged (includes control images and reference images)
- **Metadata:** Complete parameter tracking and timing information
- **Organization:** Timestamped sessions for easy navigation
- **Configurable:** Enable/disable via config file
- **Low overhead:** <5% performance impact

This system is essential for debugging, quality assurance, and understanding how the pipeline processes images.
