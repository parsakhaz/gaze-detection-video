# Gaze Detection Video Processor

> **⚠️ IMPORTANT:** This project currently uses Moondream 2 (2025-01-09 release) via the Hugging Face Transformers library. We will migrate to the official Moondream client libraries once they become available for this version.

This project uses the Moondream 2 model to detect faces and their gaze directions in videos. It processes videos frame by frame, visualizing face detections and gaze directions with dynamic visual effects.

## Features

- Face detection in video frames
- Gaze direction tracking
- Real-time visualization with:
  - Colored bounding boxes for faces
  - Gradient lines showing gaze direction
  - Gaze target points
- Supports multiple faces per frame
- Processes all common video formats (.mp4, .avi, .mov, .mkv)
- Uses Moondream 2 (2025-01-09 release) via Hugging Face Transformers
  - Note: Will be migrated to official client libraries in future updates
  - No authentication required

## Prerequisites

1. Python 3.8 or later
2. CUDA-capable GPU recommended (but CPU mode works too)
3. FFmpeg installed on your system
4. libvips installed on your system:
   ```bash
   # Ubuntu/Debian
   sudo apt-get update && sudo apt-get install -y libvips42 libvips-dev

   # CentOS/RHEL
   sudo yum install vips vips-devel

   # macOS
   brew install vips

   # Windows
   # Download from https://github.com/libvips/build-win64/releases
   ```

## Installation

1. Clone the repository:
   ```bash
   git clone [repository-url]
   cd gaze-detection-video
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place your input videos in the `input` directory
   - Supported formats: .mp4, .avi, .mov, .mkv
   - The directory will be created automatically if it doesn't exist

2. Run the script:
   ```bash
   python gaze-detection-video.py
   ```

3. The script will:
   - Process all videos in the input directory
   - Show progress bars for each video
   - Save processed videos to the `output` directory with prefix 'processed_'

## Output

- Processed videos are saved as `output/processed_[original_name].[ext]`
- Each frame in the output video shows:
  - Colored boxes around detected faces
  - Lines indicating gaze direction
  - Points showing where each person is looking

## Troubleshooting

1. CUDA/GPU Issues:
   - Ensure you have CUDA installed for GPU support
   - The script will automatically fall back to CPU if no GPU is available

2. Memory Issues:
   - If processing large videos, ensure you have enough RAM
   - Consider reducing video resolution if needed

3. libvips Errors:
   - Make sure libvips is properly installed for your OS
   - Check system PATH includes libvips

4. Video Format Issues:
   - Ensure FFmpeg is installed and in your system PATH
   - Try converting problematic videos to MP4 format

## Performance Notes

- GPU processing is significantly faster than CPU
- Processing time depends on:
  - Video resolution
  - Number of faces per frame
  - Frame rate
  - Available computing power

## Dependencies

- transformers (for Moondream 2 model access)
- torch
- opencv-python
- pillow
- matplotlib
- numpy
- tqdm
- pyvips
- accelerate
- einops

## Model Details

> **⚠️ IMPORTANT:** This project currently uses Moondream 2 (2025-01-09 release) via the Hugging Face Transformers library. We will migrate to the official Moondream client libraries once they become available for this version.

The model is loaded using:

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True
)
```

## License

MIT License