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

## Video Moment Finder: tracking and aesthetic ranking

Install the updated dependencies with `python -m pip install -r requirements.txt`.
Then run:

```bash
python moment_finder.py test.mp4
```

For a quick test of the beginning of a video:

```bash
python moment_finder.py test.mp4 --max-seconds 10 --top-k 5
```

The script uses YOLO + ByteTrack on every frame and locks the largest person
at the first tracked detection. It does not automatically switch to another ID
when the person disappears. This baseline is intended for one main person with
little occlusion. Track IDs may change or swap in difficult footage; inspect
`tracks.csv` before interpreting the results as the same person throughout.
Use `--target-id ID` to choose a known ID from a previous run of the same video.

Every 0.5 seconds while the target is detected, it saves a full-resolution,
unannotated PNG candidate. Aesthetic Predictor V2.5 scores the entire original
frame, not a person crop or the boxes. The best 5 frames are chosen greedily by
score, with at least 1 second between picks. This reduces adjacent duplicates
but does not guarantee different-looking images. There may be fewer than 5
results in a short clip. No sharpness, visibility, or composition metric is
included yet. Scoring all candidates on CPU may take several minutes.

Each run creates a new `output/<video>_moments_<timestamp>/` containing:

- `candidates/`: sampled original frames containing the target.
- `best/`: selected frames, ordered by rank.
- `tracks.csv`: all tracked people per frame, with a target flag and box coordinates.
- `scores.csv`: candidate frames and aesthetic scores, sorted from high to low.
- `best_frames.csv`: selected frames, scores, target ID, frame number, and timestamp.
- `summary.json`: run settings and counts; status is complete only after scoring finishes.
- `index.html`: browser gallery. Open it in a browser to review the selected images.

Frame indices start at 0. Timestamps use `frame_index / fps`, assuming constant
frame rate. Sampling can be adjusted with `--sample-seconds`; selection spacing
with `--min-gap`; and the number of photos with `--top-k`. Models are downloaded
on the first run and cached. Default inference uses CPU; separate options
`--device` and `--aesthetic-device` can select other supported devices.

References: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/)
and [Aesthetic Predictor V2.5](https://github.com/discus0434/aesthetic-predictor-v2-5).
