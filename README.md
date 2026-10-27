# Human detection with YOLO

This module detects people in an image or video and draws a green bounding box
with a confidence score around each detection.

## Setup (Windows)

Clone this repository, create a project-local virtual environment, and install
the Python dependencies:

```powershell
git clone https://github.com/Mike-HXW/CIS5810_VideoMomentFinder.git
cd CIS5810_VideoMomentFinder

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Aesthetic Predictor dependency

Video Moment Finder uses the third-party
[Aesthetic Predictor V2.5](https://github.com/discus0434/aesthetic-predictor-v2-5)
source as a local editable dependency. It is intentionally **not** included in
this repository's Git history. Each teammate must place it at the path below
and install it once:

```powershell
git clone https://github.com/discus0434/aesthetic-predictor-v2-5.git `
  external/aesthetic-predictor-v2-5

python -m pip install -e .\external\aesthetic-predictor-v2-5
```

If Git is unavailable, download that repository as a ZIP, extract it, and
rename/move the extracted folder to `external/aesthetic-predictor-v2-5` before
running the editable-install command.

Do not commit `external/`, `.venv/`, `weights/`, `runs/`, `input/`, or `output/`.
They are machine-local dependencies, models, and generated data; the project's
`.gitignore` excludes them.

### Start the web app

After completing setup, start the local Gradio interface with:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

Open the local URL printed in the terminal, normally
`http://127.0.0.1:7860`. The page can upload a video or choose one from
`input/`, select a fixed ranking preference, and display the selected frames
with their target-aware score components.

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

Every 0.5 seconds while the target is detected, the pipeline saves a
full-resolution, unannotated PNG candidate. Aesthetic Predictor V2.5 scores the
whole frame. Model-free target-aware metrics then measure subject sharpness with
Sobel/Tenengrad gradient energy, visibility with target size, edge clearance,
and short-window tracking stability, and composition with a low-weight
rule-of-thirds heuristic. The best 5 frames are chosen greedily by the combined
score, with at least 1 second between picks. This reduces adjacent duplicates
but does not guarantee different-looking images. There may be fewer than 5
results in a short clip. Aesthetic scoring on CPU may take several minutes.

The default `balanced` weights are 0.50 aesthetic, 0.30 sharpness, 0.15
visibility, and 0.05 composition. Fixed presets can be compared without a UI:

```bash
python moment_finder.py test.mp4 --ranking-preset clear-subject
python moment_finder.py test.mp4 --ranking-preset subject-fully-shown
python moment_finder.py test.mp4 --ranking-preset best-looking-scene
python moment_finder.py test.mp4 --ranking-preset aesthetic-only
```

`aesthetic-only` is a diagnostic baseline; the planned user-facing controls
will expose the four target-aware presets rather than numeric sliders.

Each run creates a new `output/<video>_moments_<timestamp>/` containing:

- `candidates/`: sampled original frames containing the target.
- `best/`: selected frames, ordered by rank.
- `tracks.csv`: all tracked people per frame, with a target flag and box coordinates.
- `scores.csv`: candidates with raw and normalized metric components, sorted by
  final target-aware score.
- `best_frames.csv`: selected frames, component scores, target ID, frame number,
  and timestamp.
- `summary.json`: run settings and counts; status is complete only after scoring finishes.
- `index.html`: browser gallery. Open it in a browser to review the selected images.

Frame indices start at 0. Timestamps use `frame_index / fps`, assuming constant
frame rate. Sampling can be adjusted with `--sample-seconds`; selection spacing
with `--min-gap`; and the number of photos with `--top-k`. Models are downloaded
on the first run and cached. Default inference uses CPU; separate options
`--device` and `--aesthetic-device` can select other supported devices.

References: [Ultralytics tracking](https://docs.ultralytics.com/modes/track/)
and [Aesthetic Predictor V2.5](https://github.com/discus0434/aesthetic-predictor-v2-5).

### Target-box copies

Each run also saves `candidates_target/` and `best_target/`, matching the
original candidate and selected PNGs but with only the target person boxed
and labeled with its tracking ID. CSV files include `target_image` paths.
In `index.html`, expand **Show target box** under a selected photo to inspect
its target. Aesthetic scoring continues to use only unannotated originals;
sharpness is measured only inside the unannotated target crop.
