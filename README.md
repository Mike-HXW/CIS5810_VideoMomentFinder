# Human detection with YOLO

This module detects people in an image or video and draws a green bounding box
with a confidence score around each detection.

## Setup

Create and activate your virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Command-line usage

Place your images or videos in the `input/` folder beside `human_detector.py`,
then pass the filename:

```bash
python human_detector.py photo.jpg
python human_detector.py video.mp4
python human_detector.py photo.jpg --confidence 0.4 --device cpu
```

These commands read `input/photo.jpg` or `input/video.mp4` and save results as
`output/photo_detected.jpg` or `output/video_detected.mp4`. The `output/` folder
is created automatically when needed. The default folders are relative to the
module location, so they also work when launching from another directory.
Each invocation processes one file.

You can still pass an explicit input path and override the output path:

```bash
python human_detector.py input/photo.jpg --output output/custom.jpg
python human_detector.py /path/to/video.mp4 --output results/detected.mp4
```

Explicit relative paths are resolved from your terminal's current directory.
When `--output` is omitted, results always go in the module's `output/` folder.
The pretrained weights are downloaded automatically on the first run. Video
output contains the annotated frames and does not retain the input audio track.

## Python usage

```python
from human_detector import HumanDetector

detector = HumanDetector(confidence=0.25)
output_path = detector.process("photo.jpg")  # input/photo.jpg -> output/photo_detected.jpg
print(output_path)
```

`detect_frame()` also returns structured `Detection` objects. The drawing logic
is isolated in `draw_detections()`, so the box representation can be replaced
later without changing inference or file handling.
